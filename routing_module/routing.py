import osmnx as ox
import heapq
import math
import ast
import re
from collections import deque

from routing_module.database import PostgresConnector


class RoutingEngine:
    def __init__(self, network_data):
        """
        Initialize routing engine with precomputed network data.

        Args:
            network_data: Dictionary from create_network() containing:
                - graph: OSM graph with access_map
                - trip_graph: Comprehensive pathway graph
                - lookups: All lookup dictionaries
                - cost_model: Fare prediction model
                - traffic_data: Timing data
                - distance_data: Distance data
        """
        self.graph = network_data["graph"]
        self.trip_graph = network_data["trip_graph"]
        self.lookups = network_data["lookups"]
        self.cost_model = network_data["cost_model"]
        self.traffic_data = network_data["traffic_data"]
        self.distance_data = network_data["distance_data"]
        self.db = PostgresConnector()

        print(f"RoutingEngine initialized with:")
        print(f"  - OSM graph: {len(self.graph.nodes)} nodes")
        print(f"  - Trip graph: {len(self.trip_graph)} starting trips")
        print(f"  - Lookup tables: {len(self.lookups)} dictionaries")
        print(f"  - Cost model: {len(self.distance_data)} trip distances")

    def explore_trips(self, source, cutoff=1000.0):
        """
        Find all accessible trips from a source node within walking distance.

        Args:
            source: OSM node ID to start from
            cutoff: Maximum walking distance in meters

        Returns:
            Dictionary of trip_id -> {
                'stop_id': GTFS stop ID,
                'agency': agency ID for fare calculation,
                'stop_sequence': stop sequence in trip,
                'osm_node_id': OSM node ID,
                'walk': walking distance in meters,
                'path': walking path coordinates
            }
        """
        dist = {source: 0.0}
        prev = {}
        pq = [(0.0, source)]
        visited = set()
        trips = {}

        def reconstruct_path_coords(node):
            """Convert node path to coordinate path"""
            node_path = []
            current = node
            while current in prev:
                node_path.append(current)
                current = prev[current]
            node_path.append(source)
            node_path = list(reversed(node_path))

            coord_path = []
            for node_id in node_path:
                if node_id in self.graph.nodes:
                    lat = self.graph.nodes[node_id]["y"]
                    lon = self.graph.nodes[node_id]["x"]
                    coord_path.append([lat, lon])
            return coord_path

        while pq:
            d, node = heapq.heappop(pq)

            if d > cutoff:
                break
            if node in visited:
                continue
            visited.add(node)

            # Check for accessible trips at this node
            access_map = self.graph.nodes[node].get("access_map")

            if access_map:
                for trip_id, access_info in access_map.items():
                    # Check if this is the best walk to this trip so far
                    best = trips.get(trip_id)
                    if best is None or d < best["walk"]:
                        trips[trip_id] = {
                            "stop_id": access_info["stop_id"],
                            "agency": access_info["agency_id"],
                            "stop_sequence": access_info["stop_sequence"],
                            "osm_node_id": node,
                            "walk": d,
                            "path": reconstruct_path_coords(node),
                        }

            # Relax neighbors (Standard Dijkstra)
            for nbr, edge_data in self.graph[node].items():
                for _, attr in edge_data.items():
                    length = float(attr.get("length", 1.0))
                    new_dist = d + length

                    if new_dist <= cutoff and new_dist < dist.get(nbr, float("inf")):
                        dist[nbr] = new_dist
                        prev[nbr] = node
                        heapq.heappush(pq, (new_dist, nbr))

        return trips

    def get_distance(self, trip_id, start_stop, end_stop):
        """Get distance for a trip (uses precomputed data)"""

        return (
            self.db.get_distance_between_two_stops_within_route(
                trip_id, start_stop, end_stop
            )
            / 1000
        )

    def get_fare(self, trip_id, start_stop, end_stop, agency="P_O_14"):
        """Calculate fare using precomputed model"""
        passengers = 14 if agency == "P_O_14" else 8
        distance = self.get_distance(trip_id, start_stop, end_stop)
        intercept = self.cost_model.intercept_
        beta_distance, beta_passengers = self.cost_model.coef_
        return math.ceil(
            intercept + beta_distance * distance + beta_passengers * passengers
        )

    def get_transport_time(self, trip_id, start_stop, end_stop):
        """Get transport time between stops using precomputed traffic data"""
        start = str(start_stop)
        end = str(end_stop)

        if trip_id not in self.traffic_data:
            raise KeyError(f"trip_id not found in traffic: '{trip_id}'")

        trip_times = self.traffic_data[trip_id]
        if start not in trip_times:
            raise KeyError(f"start_stop not found for trip '{trip_id}': '{start}'")
        if end not in trip_times:
            raise KeyError(f"end_stop not found for trip '{trip_id}': '{end}'")

        start_val = trip_times[start]
        end_val = trip_times[end]

        try:
            start_num = float(start_val)
            end_num = float(end_val)
        except Exception as e:
            raise ValueError(
                f"Invalid numeric values for trip '{trip_id}': start='{start_val}', end='{end_val}'"
            ) from e

        return end_num - start_num

    def find_journeys_pareto(
        self, start_trips, goal_trips, max_transfers, restricted_modes=None
    ):
        """
        Find optimal journeys using Pareto optimization with mode restrictions.

        Args:
            start_trips: Dictionary of accessible starting trips
            goal_trips: Dictionary of accessible destination trips
            max_transfers: Maximum number of transfers allowed
            restricted_modes: List of agency IDs to ignore (default: empty list)

        Returns:
            List of (trip_path, total_costs, cost_details) tuples
        """
        if restricted_modes is None:
            restricted_modes = []

        results = []
        queue = deque()

        # Pareto frontiers for each trip
        best = {}

        def dominates(v1, v2):
            """Check if v1 dominates v2"""
            better_in_all = all(v1[i] <= v2[i] for i in range(4))
            strictly_better_in_one = any(v1[i] < v2[i] for i in range(4))
            return better_in_all and strictly_better_in_one

        def update_pareto_frontier(frontier, new_cost):
            """Add new_cost to frontier and remove dominated vectors"""
            frontier_updated = [v for v in frontier if not dominates(new_cost, v)]
            if not any(dominates(v, new_cost) for v in frontier_updated):
                frontier_updated.append(new_cost)
            return frontier_updated

        # Initialize start trips
        for start_trip_id, data in start_trips.items():
            if data["agency"] in restricted_modes:
                continue

            initial_walk_time = data["walk"] / 83.33 * 60  # Convert to seconds (5 km/h)
            c0 = (0, 0, initial_walk_time, data["walk"])
            path = [start_trip_id]
            cost_details = []
            start_stop = data["stop_id"]
            start_sequence = data["stop_sequence"]

            queue.append(
                (start_trip_id, start_stop, start_sequence, path, c0, cost_details)
            )
            best[start_trip_id] = [c0]

            # Check 0-transfer goal
            if start_trip_id in goal_trips:
                goal_stop = goal_trips[start_trip_id]["stop_id"]
                goal_seq = goal_trips[start_trip_id]["stop_sequence"]

                if goal_trips[start_trip_id]["agency"] in restricted_modes:
                    continue

                if start_sequence < goal_seq:
                    delta_fare = self.get_fare(
                        start_trip_id, start_stop, goal_stop, data["agency"]
                    )
                    delta_time = self.get_transport_time(
                        start_trip_id, start_stop, goal_stop
                    )

                    trip_cost_detail = {
                        "type": "trip",
                        "trip_id": start_trip_id,
                        "from_stop_id": start_stop,
                        "to_stop_id": goal_stop,
                        "fare": delta_fare,
                        "time": delta_time,
                        "agency_id": data["agency"],
                    }

                    final_walk_time = goal_trips[start_trip_id]["walk"] / 83.33 * 60

                    c_final = (
                        c0[0],  # transfers (0)
                        c0[1] + delta_fare,  # fare
                        c0[2] + delta_time + final_walk_time,  # total_time
                        c0[3] + goal_trips[start_trip_id]["walk"],  # walk distance
                    )
                    results.append((path, c_final, [trip_cost_detail]))

        # BFS with Pareto Pruning
        while queue:
            (
                current_trip,
                current_board_stop,
                current_board_sequence,
                path,
                c,
                cost_details,
            ) = queue.popleft()

            if len(path) - 1 >= max_transfers:
                continue

            for next_trip, pathway in self.trip_graph.get(current_trip, {}).items():

                if next_trip in path:
                    continue

                # Skip restricted modes
                if (
                    pathway["start_agency_id"] in restricted_modes
                    or pathway.get("end_agency_id", pathway["start_agency_id"])
                    in restricted_modes
                ):
                    continue

                # Prune invalid path segment
                if current_board_sequence >= pathway["start_stop_sequence"]:
                    continue

                # Calculate costs
                prev_trip_time = self.get_transport_time(
                    current_trip, current_board_stop, pathway["start_stop_id"]
                )
                prev_trip_money = self.get_fare(
                    current_trip,
                    current_board_stop,
                    pathway["start_stop_id"],
                    pathway["start_agency_id"],
                )

                trip_cost_detail = {
                    "type": "trip",
                    "trip_id": current_trip,
                    "from_stop_id": current_board_stop,
                    "to_stop_id": pathway["start_stop_id"],
                    "fare": prev_trip_money,
                    "time": prev_trip_time,
                    "agency_id": pathway["start_agency_id"],
                }

                transfer_cost_detail = {
                    "type": "transfer",
                    "from_trip_id": current_trip,
                    "to_trip_id": next_trip,
                    "walking_distance_m": pathway["walking_distance_m"],
                    "pathway": pathway,
                }

                transfer_walk_time = pathway["walking_distance_m"] / 83.33 * 60
                c_new = (
                    c[0] + 1,  # transfers
                    c[1] + prev_trip_money,  # fare
                    c[2] + prev_trip_time + transfer_walk_time,  # total_time
                    c[3] + pathway["walking_distance_m"],  # walk distance
                )

                new_cost_details = cost_details + [
                    trip_cost_detail,
                    transfer_cost_detail,
                ]

                # Pareto Pruning
                if next_trip in best:
                    if any(dominates(v, c_new) for v in best[next_trip]):
                        continue

                if next_trip not in best:
                    best[next_trip] = []
                best[next_trip] = update_pareto_frontier(best[next_trip], c_new)

                new_path = path + [next_trip]
                queue.append(
                    (
                        next_trip,
                        pathway["end_stop_id"],
                        pathway["end_stop_sequence"],
                        new_path,
                        c_new,
                        new_cost_details,
                    )
                )

                # Check goal
                if next_trip in goal_trips:
                    goal_stop = goal_trips[next_trip]["stop_id"]

                    if goal_trips[next_trip]["agency"] in restricted_modes:
                        continue

                    transfer_in_seq = pathway["end_stop_sequence"]
                    goal_seq = goal_trips[next_trip]["stop_sequence"]
                    if transfer_in_seq < goal_seq:
                        last_leg_time = self.get_transport_time(
                            next_trip, pathway["end_stop_id"], goal_stop
                        )
                        last_leg_money = self.get_fare(
                            next_trip,
                            pathway["end_stop_id"],
                            goal_stop,
                            goal_trips[next_trip]["agency"],
                        )

                        final_trip_cost_detail = {
                            "type": "trip",
                            "trip_id": next_trip,
                            "from_stop_id": pathway["end_stop_id"],
                            "to_stop_id": goal_stop,
                            "fare": last_leg_money,
                            "time": last_leg_time,
                            "agency_id": goal_trips[next_trip]["agency"],
                        }

                        final_walk_time = goal_trips[next_trip]["walk"] / 83.33 * 60

                        c_final = (
                            c_new[0],  # transfers
                            c_new[1] + last_leg_money,  # fare
                            c_new[2] + last_leg_time + final_walk_time,  # total_time
                            c_new[3] + goal_trips[next_trip]["walk"],  # walk distance
                        )
                        final_cost_details = new_cost_details + [final_trip_cost_detail]
                        results.append((new_path, c_final, final_cost_details))

        return results

    def rank_routing_results(self, routing_results, weights=None, top_n=5):
        """
        Rank routing results based on weighted normalized costs.

        Args:
            routing_results: List of (trip_path, total_costs, cost_details) tuples
            weights: Dictionary with keys 'time', 'cost', 'walk', 'transfer' (default: balanced weights)
            top_n: Number of top results to return
        """
        if weights is None:
            weights = {"time": 0.3, "cost": 0.3, "walk": 0.1, "transfer": 0.3}

        if not routing_results or len(routing_results) <= 1:
            return routing_results

        # Extract metrics from total_costs: (transfers, fare, total_time, walk_distance)
        transfers = [costs[0] for _, costs, _ in routing_results]
        fares = [costs[1] for _, costs, _ in routing_results]
        times = [costs[2] / 60 for _, costs, _ in routing_results]  # Convert to minutes
        walks = [costs[3] for _, costs, _ in routing_results]

        # Get ranges for normalization
        ranges = {
            "transfer": (min(transfers), max(transfers)),
            "fare": (min(fares), max(fares)),
            "time": (min(times), max(times)),
            "walk": (min(walks), max(walks)),
        }

        # Score each journey
        scored_results = []
        for i, (trip_path, total_costs, cost_details) in enumerate(routing_results):
            # Normalize (0-1 scale, handle zero ranges)
            norm_transfer = (
                (total_costs[0] - ranges["transfer"][0])
                / (ranges["transfer"][1] - ranges["transfer"][0])
                if ranges["transfer"][1] > ranges["transfer"][0]
                else 0
            )
            norm_fare = (
                (total_costs[1] - ranges["fare"][0])
                / (ranges["fare"][1] - ranges["fare"][0])
                if ranges["fare"][1] > ranges["fare"][0]
                else 0
            )
            norm_time = (
                (total_costs[2] / 60 - ranges["time"][0])
                / (ranges["time"][1] - ranges["time"][0])
                if ranges["time"][1] > ranges["time"][0]
                else 0
            )
            norm_walk = (
                (total_costs[3] - ranges["walk"][0])
                / (ranges["walk"][1] - ranges["walk"][0])
                if ranges["walk"][1] > ranges["walk"][0]
                else 0
            )

            # Calculate weighted score
            score = (
                weights["transfer"] * norm_transfer
                + weights["cost"] * norm_fare
                + weights["time"] * norm_time
                + weights["walk"] * norm_walk
            )
            scored_results.append((trip_path, total_costs, cost_details, score))

        # Sort by score and return top N
        ranked_results = sorted(scored_results, key=lambda x: x[3])[:top_n]
        return [
            (trip_path, total_costs, cost_details)
            for trip_path, total_costs, cost_details, _ in ranked_results
        ]

    def enrich_journey_results(self, routing_results, start_trips, target_trips):
        """
        Transform routing results into frontend-ready JSON structure.
        """
        detailed_journeys = []

        for journey_idx, (trip_path, total_costs, cost_details) in enumerate(
            routing_results
        ):

            transfers, total_fare, total_time, total_walk = total_costs

            legs = []
            modes_used = set()

            # Initial walking leg
            start_trip_id = trip_path[0]
            start_trip_data = start_trips[start_trip_id]

            if start_trip_data["walk"] > 0:
                legs.append(
                    {
                        "type": "walk",
                        "distance_meters": round(start_trip_data["walk"]),
                        "duration_minutes": max(
                            1, math.ceil(start_trip_data["walk"] / 83.33)
                        ),
                        "path": start_trip_data["path"],
                    }
                )

            # Process legs using preserved cost details
            for detail in cost_details:
                if detail["type"] == "trip":
                    route_id = self.lookups["trip_to_route"].get(detail["trip_id"])
                    route_short_name = self.lookups["route_to_short_name"].get(route_id)

                    # Extract mode from route name
                    if route_short_name:
                        mode = re.sub(r"\d+", "", route_short_name).strip().lower()
                        mode = " ".join(mode.split())
                    else:
                        mode = "unknown"
                    modes_used.add(mode)

                    # Get stop coordinates
                    from_coords = self.lookups["stop_to_coords"].get(
                        detail["from_stop_id"], {}
                    )
                    to_coords = self.lookups["stop_to_coords"].get(
                        detail["to_stop_id"], {}
                    )

                    trip_leg = {
                        "type": "trip",
                        "trip_id": detail["trip_id"],
                        "mode": mode,
                        "route_short_name": route_short_name,
                        "headsign": self.lookups["trip_to_headsign"].get(
                            detail["trip_id"], "Unknown"
                        ),
                        "fare": round(detail["fare"], 2),
                        "duration_minutes": max(1, math.ceil(detail["time"] / 60)),
                        "from": {
                            "stop_id": detail["from_stop_id"],
                            "name": self.lookups["stop_to_name"].get(
                                detail["from_stop_id"], "Unknown Stop"
                            ),
                            "coord": [
                                from_coords.get("stop_lat", 0),
                                from_coords.get("stop_lon", 0),
                            ],
                        },
                        "to": {
                            "stop_id": detail["to_stop_id"],
                            "name": self.lookups["stop_to_name"].get(
                                detail["to_stop_id"], "Unknown Stop"
                            ),
                            "coord": [
                                to_coords.get("stop_lat", 0),
                                to_coords.get("stop_lon", 0),
                            ],
                        },
                        "path": [],
                    }
                    legs.append(trip_leg)

                elif detail["type"] == "transfer":
                    pathway = detail["pathway"]

                    walking_coords = []
                    if pathway.get("walking_path_coords"):
                        try:
                            if isinstance(pathway["walking_path_coords"], str):
                                walking_coords = ast.literal_eval(
                                    pathway["walking_path_coords"]
                                )
                            else:
                                walking_coords = pathway["walking_path_coords"]
                        except:
                            walking_coords = []

                    transfer_duration = max(
                        1, int(detail["walking_distance_m"] / 83.33)
                    )

                    transfer_leg = {
                        "type": "transfer",
                        "from_trip_id": detail["from_trip_id"],
                        "to_trip_id": detail["to_trip_id"],
                        "from_trip_name": pathway.get("start_route_name", "Unknown"),
                        "to_trip_name": pathway.get("end_route_name", "Unknown"),
                        "walking_distance_meters": round(detail["walking_distance_m"]),
                        "duration_minutes": transfer_duration,
                        "path": walking_coords,
                    }
                    legs.append(transfer_leg)

            # Final walking leg
            final_trip_id = trip_path[-1]
            final_walk = target_trips[final_trip_id]["walk"]

            if final_walk > 0:
                legs.append(
                    {
                        "type": "walk",
                        "distance_meters": round(final_walk),
                        "duration_minutes": max(1, math.ceil(final_walk / 83.33)),
                        "path": target_trips[final_trip_id]["path"],
                    }
                )

            # Build journey summary
            total_distance = total_walk
            total_duration_minutes = math.ceil(total_time / 60)

            # Create text summary
            summary_parts = [
                f"Total Duration: {total_duration_minutes} minutes, Total Cost: E£{round(total_fare, 2)}, Transfers: {transfers}, Total Walking: {int(total_walk)}m"
            ]

            for leg in legs:
                if leg["type"] == "walk":
                    summary_parts.append(
                        f"walk {leg['distance_meters']}m ({leg['duration_minutes']} min)"
                    )
                elif leg["type"] == "trip":
                    summary_parts.append(
                        f"take {leg['route_short_name']} to {leg['headsign']} (E£{leg['fare']}, {leg['duration_minutes']} min) - Board at \"{leg['from']['name']}\", Exit at \"{leg['to']['name']}\""
                    )
                elif leg["type"] == "transfer":
                    summary_parts.append(
                        f"walk {leg['walking_distance_meters']}m ({leg['duration_minutes']} min) - Transfer from {leg['from_trip_name']} to {leg['to_trip_name']}"
                    )

            text_summary = ", ".join(summary_parts)

            journey = {
                "id": journey_idx + 1,
                "text_summary": text_summary,
                "summary": {
                    "total_time_minutes": total_duration_minutes,
                    "total_distance_meters": int(total_distance),
                    "walking_distance_meters": int(total_walk),
                    "transfers": transfers,
                    "cost": round(total_fare, 2),
                    "modes": sorted(list(modes_used)),
                },
                "legs": legs,
            }

            detailed_journeys.append(journey)

        return {"num_journeys": len(detailed_journeys), "journeys": detailed_journeys}

    def find_journeys(
        self,
        start_lat,
        start_lon,
        end_lat,
        end_lon,
        max_transfers=2,
        walking_cutoff=1000,
        weights=None,
        restricted_modes=None,
        top_k=5,
    ):
        """
        Complete journey finding pipeline from coordinates to enriched results.

        Args:
            start_lat, start_lon: Starting coordinates
            end_lat, end_lon: Destination coordinates
            max_transfers: Maximum number of transfers allowed (default: 2)
            walking_cutoff: Maximum walking distance in meters (default: 1000)
            weights: Dictionary with keys 'time', 'cost', 'walk', 'transfer' (default: balanced weights)
            restricted_modes: List of agency IDs to avoid (default: empty list)
            top_k: Number of top journeys to return (default: 5)

        Returns:
            Dictionary with journeys and metadata ready for frontend
        """
        if weights is None:
            weights = {"time": 0.3, "cost": 0.3, "walk": 0.1, "transfer": 0.3}
        if restricted_modes is None:
            restricted_modes = []

        # 1. Find nearest nodes and explore accessible trips
        start_node = ox.distance.nearest_nodes(self.graph, X=start_lon, Y=start_lat)
        end_node = ox.distance.nearest_nodes(self.graph, X=end_lon, Y=end_lat)

        start_trips = self.explore_trips(start_node, cutoff=walking_cutoff)
        target_trips = self.explore_trips(end_node, cutoff=walking_cutoff)

        # Check if we found trips
        if not start_trips:
            return {
                "num_journeys": 0,
                "journeys": [],
                "error": f"No transit trips found within {walking_cutoff}m of start location",
                "start_trips_found": 0,
                "end_trips_found": len(target_trips),
            }

        if not target_trips:
            return {
                "num_journeys": 0,
                "journeys": [],
                "error": f"No transit trips found within {walking_cutoff}m of end location",
                "start_trips_found": len(start_trips),
                "end_trips_found": 0,
            }

        # 2. Find optimal journeys
        routing_results = self.find_journeys_pareto(
            start_trips, target_trips, max_transfers, restricted_modes
        )

        if not routing_results:
            return {
                "num_journeys": 0,
                "journeys": [],
                "error": "No valid journeys found between the locations",
                "start_trips_found": len(start_trips),
                "end_trips_found": len(target_trips),
            }

        # 3. Rank results
        ranked_results = self.rank_routing_results(routing_results, weights, top_k)

        # 4. Enrich for frontend
        final_results = self.enrich_journey_results(
            ranked_results, start_trips, target_trips
        )

        # Add metadata
        final_results.update(
            {
                "start_trips_found": len(start_trips),
                "end_trips_found": len(target_trips),
                "total_routes_found": len(routing_results),
                "error": None,
            }
        )

        return final_results


# Global instance management
_routing_engine = None


def get_routing_engine(network_data=None):
    """Get or create the global routing engine instance."""
    global _routing_engine
    if _routing_engine is None:
        if network_data is None:
            from routing_module.network import create_network

            network_data = create_network()
        _routing_engine = RoutingEngine(network_data)
    return _routing_engine
