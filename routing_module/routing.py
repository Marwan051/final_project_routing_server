import json
import heapq
import os
from collections import deque, defaultdict

from routing_module.database import PostgresConnector
from routing_module.price_predictor import TripPricePredictor, load_model


class RoutingEngine:
    """Routing engine that loads the price prediction model once at startup."""

    def __init__(self, model_path=None):
        """Initialize the routing engine and load the price prediction model.

        Args:
            model_path: Path to the price prediction model file
        """
        self.db = PostgresConnector()
        self.price_predictor = load_model(model_path)
        if not isinstance(self.price_predictor, TripPricePredictor):
            raise TypeError(
                f"Expected TripPricePredictor, got {type(self.price_predictor)}"
            )
        self._traffic = None

    def get_distance(self, trip_id, start_stop, end_stop):
        """Get distance for a trip between two stops"""
        return self.db.get_distance_between_two_stops_within_route(
            trip_id, start_stop, end_stop
        )

    def get_distance_coord(self, trip_id, start_lat, start_lon, end_lat, end_lon):
        """Get distance between two coordinates along a route"""
        return self.db.get_distance_between_two_coordinates_within_route(
            trip_id, start_lat, start_lon, end_lat, end_lon
        )

    def get_cost(self, trip_id, start_stop, end_stop):
        """Calculate the cost of a trip between two stops"""
        distance = self.get_distance(trip_id, start_stop, end_stop)
        return self.price_predictor.predict([distance])[0]

    def load_traffic(self, traffic_path=None):
        """Load and cache traffic/timing data from JSON"""
        if self._traffic is None:
            if traffic_path is None:
                traffic_path = os.path.join(
                    os.path.dirname(__file__), "data", "prefixtimes.json"
                )
            with open(traffic_path, "r") as file:
                self._traffic = json.load(file)
        return self._traffic

    def get_transport_time(self, trip_id, start_stop, end_stop, traffic=None):
        """Return travel time (seconds/minutes as stored) between two stops for a trip."""
        if traffic is None:
            traffic = self.load_traffic()

        start = str(start_stop)
        end = str(end_stop)
        if trip_id not in traffic:
            raise KeyError(f"trip_id not found in traffic: '{trip_id}'")
        trip_times = traffic[trip_id]
        if start not in trip_times:
            raise KeyError(f"start_stop not found for trip '{trip_id}': '{start}'")
        if end not in trip_times:
            raise KeyError(f"end_stop not found for trip '{trip_id}': '{end}'")

        start_val = trip_times[start]
        end_val = trip_times[end]
        try:
            start_num = float(start_val)
        except Exception as e:
            raise ValueError(
                f"start_stop value is not numeric for trip '{trip_id}': start='{start}', value={start_val!r}"
            ) from e
        try:
            end_num = float(end_val)
        except Exception as e:
            raise ValueError(
                f"end_stop value is not numeric for trip '{trip_id}': end='{end}', value={end_val!r}"
            ) from e

        return end_num - start_num


# Global instance (can be initialized once at app startup)
_routing_engine = None


def get_routing_engine(model_path=None):
    """Get or create the global routing engine instance."""
    global _routing_engine
    if _routing_engine is None:
        _routing_engine = RoutingEngine(model_path)
    return _routing_engine


def get_distance(trip_id, start_stop, end_stop):
    """Get distance for a trip between two stops"""
    engine = get_routing_engine()
    return engine.get_distance(trip_id, start_stop, end_stop)


def get_distance_coord(trip_id, start_lat, start_lon, end_lat, end_lon):
    engine = get_routing_engine()
    return engine.get_distance_coord(trip_id, start_lat, start_lon, end_lat, end_lon)


def get_cost(trip_id, start_stop, end_stop, distances_path=None, model_path=None):
    """Calculate the cost of a trip between two stops"""
    engine = get_routing_engine(model_path)
    return engine.get_cost(trip_id, start_stop, end_stop)


def load_traffic(traffic_path=None):
    """Load traffic/timing data from JSON"""
    engine = get_routing_engine()
    return engine.load_traffic(traffic_path)


def get_transport_time(trip_id, start_stop, end_stop, traffic=None):
    """Return travel time (seconds/minutes as stored) between two stops for a trip."""
    engine = get_routing_engine()
    return engine.get_transport_time(trip_id, start_stop, end_stop, traffic)


def load_traffic(traffic_path=None):
    """Load traffic/timing data from JSON"""
    if traffic_path is None:
        traffic_path = os.path.join(
            os.path.dirname(__file__), "data", "prefixtimes.json"
        )
    with open(traffic_path, "r") as file:
        prefix_times = json.load(file)
    return prefix_times


def get_transport_time(trip_id, start_stop, end_stop, traffic=None):
    """
    Return travel time (seconds/minutes as stored) between two stops for a trip.
    """
    if traffic is None:
        traffic = load_traffic()

    start = str(start_stop)
    end = str(end_stop)
    if trip_id not in traffic:
        raise KeyError(f"trip_id not found in traffic: '{trip_id}'")
    trip_times = traffic[trip_id]
    if start not in trip_times:
        raise KeyError(f"start_stop not found for trip '{trip_id}': '{start}'")
    if end not in trip_times:
        raise KeyError(f"end_stop not found for trip '{trip_id}': '{end}'")

    start_val = trip_times[start]
    end_val = trip_times[end]
    try:
        start_num = float(start_val)
    except Exception as e:
        raise ValueError(
            f"start_stop value is not numeric for trip '{trip_id}': start='{start}', value={start_val!r}"
        ) from e
    try:
        end_num = float(end_val)
    except Exception as e:
        raise ValueError(
            f"end_stop value is not numeric for trip '{trip_id}': end='{end}', value={end_val!r}"
        ) from e

    return end_num - start_num


def explore_trips(G, source, cutoff=float("inf")):
    """
    Find all reachable trips from a source node within walking distance cutoff.

    Returns:
        dict of trip_id -> {
            'stop_id': gtfs_stop_id,
            'osm_node_id': osm_node,
            'walk': distance_m,
            'path': [...]
        }
    """
    dist = {source: 0.0}
    prev = {}
    pq = [(0.0, source)]
    visited = set()

    trips = {}

    def reconstruct_path(node):
        path = []
        while node in prev:
            path.append(node)
            node = prev[node]
        path.append(source)
        return list(reversed(path))

    while pq:
        d, node = heapq.heappop(pq)

        if d > cutoff:
            break
        if node in visited:
            continue
        visited.add(node)

        # Retrieve the boarding_map
        boarding_map = G.nodes[node].get("boarding_map")

        if boarding_map:
            # Iterate over the trips available at this node
            for trip_id, real_stop_id in boarding_map.items():
                # Check if this is the best walk to this trip so far
                best = trips.get(trip_id)
                if best is None or d < best["walk"]:
                    trips[trip_id] = {
                        "stop_id": real_stop_id,
                        "osm_node_id": node,
                        "walk": d,
                        "path": reconstruct_path(node),
                    }

        # Relax neighbors (Standard Dijkstra)
        for nbr, edge_data in G[node].items():
            for _, attr in edge_data.items():
                length = float(attr.get("length", 1.0))
                new_dist = d + length

                if new_dist <= cutoff and new_dist < dist.get(nbr, float("inf")):
                    dist[nbr] = new_dist
                    prev[nbr] = node
                    heapq.heappush(pq, (new_dist, nbr))

    return trips


def find_journeys(
    graph, pathways_dict, start_trips, goal_trips, max_transfers, traffic=None
):
    """
    Find all journeys from start trips to goal trips with up to max_transfers transfers.

    Args:
        graph: Trip graph (dict of dict) where graph[trip_id][next_trip_id] = pathway_id
        pathways_dict: Dictionary of pathways indexed by pathway_id
        start_trips: Dictionary of starting trips with stop_id and walk distance
        goal_trips: Dictionary of goal trips with stop_id and walk distance
        max_transfers: Maximum number of transfers allowed
        traffic: Optional pre-loaded traffic data

    Returns:
        List of tuples (path, costs) where path is list of trip_ids and costs is dict
    """
    engine = get_routing_engine()

    if traffic is None:
        traffic = engine.load_traffic()

    results = []
    # (current_trip_id, current_board_stop_id, path_list, cumulative_costs)
    queue = deque()

    # Pruning dictionary
    best_costs_to_node = defaultdict(
        lambda: {
            "money": float("inf"),
            "transport_time": float("inf"),
            "walk": float("inf"),
        }
    )

    # --- 1. Initialize Start Trips ---
    for start_trip_id, data in start_trips.items():
        costs = {"money": 0, "transport_time": 0, "walk": data["walk"]}
        path = [start_trip_id]
        start_stop = data["stop_id"]

        queue.append((start_trip_id, start_stop, path, costs))
        best_costs_to_node[start_trip_id] = costs.copy()

        # Check 0-transfer goal
        if start_trip_id in goal_trips:
            goal_stop = goal_trips[start_trip_id]["stop_id"]

            leg_money = engine.get_cost(start_trip_id, start_stop, goal_stop)
            leg_time = engine.get_transport_time(
                start_trip_id, start_stop, goal_stop, traffic
            )

            final_costs = costs.copy()
            final_costs["money"] += leg_money
            final_costs["transport_time"] += leg_time
            final_costs["walk"] += goal_trips[start_trip_id]["walk"]

            results.append((path, final_costs))

    # --- 2. BFS ---
    while queue:
        (current_trip, current_board_stop, path, current_costs) = queue.popleft()

        if len(path) - 1 >= max_transfers:
            continue

        for next_trip, pathway_id in graph.get(current_trip, {}).items():
            pathway = pathways_dict[pathway_id]

            if next_trip in path:
                continue

            # --- Cost Logic ---
            # 1. Transfer Walk
            transfer_walk_cost = pathway["walking_distance_m"]

            # 2. Cost of the PREVIOUS trip segment
            prev_trip_money = engine.get_cost(
                current_trip, current_board_stop, pathway["start_stop_id"]
            )
            prev_trip_time = engine.get_transport_time(
                current_trip, current_board_stop, pathway["start_stop_id"], traffic
            )

            new_costs = {
                "money": current_costs["money"] + prev_trip_money,
                "transport_time": current_costs["transport_time"] + prev_trip_time,
                "walk": current_costs["walk"] + transfer_walk_cost,
            }

            # --- Pruning---
            best_known = best_costs_to_node[next_trip]
            is_potentially_useful = (
                new_costs["money"] < best_known["money"]
                or new_costs["transport_time"] < best_known["transport_time"]
                or new_costs["walk"] < best_known["walk"]
            )

            if is_potentially_useful:
                # Update best knowns
                best_costs_to_node[next_trip]["money"] = min(
                    best_known["money"], new_costs["money"]
                )
                best_costs_to_node[next_trip]["transport_time"] = min(
                    best_known["transport_time"], new_costs["transport_time"]
                )
                best_costs_to_node[next_trip]["walk"] = min(
                    best_known["walk"], new_costs["walk"]
                )

                new_path = path + [next_trip]

                # We board the NEXT trip at pathway['end_stop_id']
                queue.append((next_trip, pathway["end_stop_id"], new_path, new_costs))

                # --- 3. Check Goal ---
                if next_trip in goal_trips:
                    goal_stop = goal_trips[next_trip]["stop_id"]

                    # FINAL leg cost
                    last_leg_money = engine.get_cost(
                        next_trip, pathway["end_stop_id"], goal_stop
                    )
                    last_leg_time = engine.get_transport_time(
                        next_trip, pathway["end_stop_id"], goal_stop, traffic
                    )

                    final_journey_costs = new_costs.copy()
                    final_journey_costs["money"] += last_leg_money
                    final_journey_costs["transport_time"] += last_leg_time
                    final_journey_costs["walk"] += goal_trips[next_trip]["walk"]

                    results.append((new_path, final_journey_costs))

    return results
