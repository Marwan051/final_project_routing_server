import json
import heapq
import os
from collections import deque, defaultdict

from routing_module.database import PostgresConnector
from routing_module.price_predictor import TripPricePredictor


def enrich_journey_results(lean_journeys, start_trips, target_trips, trip_graph, pathway_metadata, enrichment_lookups):
    """
    Transform lean routing results into detailed frontend-ready JSON structure.
    Now uses enrichment_lookups dictionary instead of individual lookup parameters.
    
    Args:
        lean_journeys: List of (trip_path, costs) tuples from routing
        start_trips: Start trips data
        target_trips: Target trips data
        trip_graph: Lean trip graph for routing data
        pathway_metadata: Pathway metadata for enrichment
        enrichment_lookups: Dictionary containing all lookup tables from network.py
    """
    import ast
    import json
    
    # Extract lookups from enrichment_lookups dictionary
    trip_to_route = enrichment_lookups['trip_to_route']
    route_to_name = enrichment_lookups['route_to_name']
    route_to_short_name = enrichment_lookups['route_to_short_name']
    trip_to_headsign = enrichment_lookups['trip_to_headsign']
    stop_to_coords = enrichment_lookups['stop_to_coords']
    stop_to_name = enrichment_lookups['stop_to_name']
    
    detailed_journeys = []
    
    for journey_idx, (trip_path, costs) in enumerate(lean_journeys):
        journey_id = f"journey_{journey_idx + 1}"
        
        # Extract cost components (transfers, fare, time, walk)
        transfers, total_fare, total_time, total_walk = costs
        
        legs = []
        total_distance = 0
        # modes_used = set(["walk"])  # Always includes walking
        modes_used = set()
        # === 1. INITIAL WALKING LEG ===
        start_trip_id = trip_path[0]
        start_trip_data = start_trips[start_trip_id]
        
        if start_trip_data['walk'] > 0:
            legs.append({
                "type": "walk",
                "distance_meters": round(start_trip_data['walk']),
                "duration_minutes": max(1, int(start_trip_data['walk'] / 83.33)),  # 5 km/h = 83.33 m/min
                "path": start_trip_data['path']
            })
            total_distance += start_trip_data['walk']
        
        # === 2. PROCESS TRIP LEGS AND TRANSFERS ===
        for i, current_trip_id in enumerate(trip_path):
            route_id = trip_to_route.get(current_trip_id)
            route_short_name = route_to_short_name.get(route_id)
            # Extract mode (transport type without numbers) from route_short_name
            if route_short_name:
                # Remove numbers and extra whitespace to get base transport type
                import re
                mode = re.sub(r'\d+', '', route_short_name).strip().lower()
                # Handle cases where there might be extra spaces after number removal
                mode = ' '.join(mode.split())  # Normalize whitespace
            else:
                mode = "unknown"
            
            modes_used.add(mode)
            
            # Determine trip segment details
            if i == 0:
                # First trip: from start access point to transfer or destination
                from_stop_id = start_trips[current_trip_id]['stop_id']
                from_coords = stop_to_coords.get(from_stop_id, {})
                
                if i == len(trip_path) - 1:
                    # Single trip journey: go to target
                    to_stop_id = target_trips[current_trip_id]['stop_id']
                    to_coords = stop_to_coords.get(to_stop_id, {})
                else:
                    # Multi-trip: go to transfer point
                    next_trip_id = trip_path[i + 1]
                    lean_pathway = trip_graph.get(current_trip_id, {}).get(next_trip_id, {})
                    to_stop_id = lean_pathway.get('start_stop_id')
                    to_coords = stop_to_coords.get(to_stop_id, {}) if to_stop_id else {}
            else:
                # Subsequent trips: from transfer point
                prev_trip_id = trip_path[i - 1]
                lean_pathway = trip_graph.get(prev_trip_id, {}).get(current_trip_id, {})
                from_stop_id = lean_pathway.get('end_stop_id')
                from_coords = stop_to_coords.get(from_stop_id, {}) if from_stop_id else {}
                
                if i == len(trip_path) - 1:
                    # Last trip: go to target
                    to_stop_id = target_trips[current_trip_id]['stop_id']
                    to_coords = stop_to_coords.get(to_stop_id, {})
                else:
                    # Go to next transfer
                    next_trip_id = trip_path[i + 1]
                    lean_pathway = trip_graph.get(current_trip_id, {}).get(next_trip_id, {})
                    to_stop_id = lean_pathway.get('start_stop_id')
                    to_coords = stop_to_coords.get(to_stop_id, {}) if to_stop_id else {}
            
            # Calculate trip duration and fare (simplified)
            trip_duration = max(5, int(total_time / len(trip_path)))  # Rough estimate
            trip_fare = round(total_fare / len(trip_path), 2)  # Split fare across trips
            
            # Add trip leg
            trip_leg = {
                "type": "trip",
                "trip_id": current_trip_id,
                "mode": mode,
                "route_short_name": route_short_name,
                "headsign": trip_to_headsign.get(current_trip_id, "Unknown"),
                "fare": trip_fare,
                "duration_minutes": trip_duration,
                "from": {
                    "stop_id": from_stop_id,
                    "name": stop_to_name.get(from_stop_id, "Unknown Stop"),
                    "coord": [from_coords.get('stop_lat', 0), from_coords.get('stop_lon', 0)]
                },
                "to": {
                    "stop_id": to_stop_id,
                    "name": stop_to_name.get(to_stop_id, "Unknown Stop"),
                    "coord": [to_coords.get('stop_lat', 0), to_coords.get('stop_lon', 0)]
                },
                "path": []  # Could add route shape here
            }
            
            legs.append(trip_leg)
            
            # === 3. ADD TRANSFER LEG (if not last trip) ===
            if i < len(trip_path) - 1:
                next_trip_id = trip_path[i + 1]
                pathway_key = (current_trip_id, next_trip_id)
                
                # Get transfer distance from lean pathway, walking coords from metadata
                lean_pathway = trip_graph.get(current_trip_id, {}).get(next_trip_id, {})
                transfer_distance = lean_pathway.get('walking_distance_m', 0)
                
                # Get walking coordinates from metadata (enrichment data)
                pathway_key = (current_trip_id, next_trip_id)
                walking_coords = []
                if pathway_key in pathway_metadata:
                    pathway_info = pathway_metadata[pathway_key]
                    from_trip_name = pathway_info['start_route_name']
                    to_trip_name=  pathway_info['end_route_name']
                    try:
                        if isinstance(pathway_info['walking_path_coords'], str):
                            walking_coords = ast.literal_eval(pathway_info['walking_path_coords'])
                        else:
                            walking_coords = pathway_info['walking_path_coords']
                    except:
                        walking_coords = []
                
                transfer_duration = max(1, int(transfer_distance / 83.33))  # 5 km/h
                
                transfer_leg = {
                    "type": "transfer",
                    "from_trip_id": current_trip_id,
                    "to_trip_id": next_trip_id,
                    "from_trip_name": from_trip_name,
                    "to_trip_name": to_trip_name,
                    "walking_distance_meters": round(transfer_distance),
                    "duration_minutes": transfer_duration,
                    "path": walking_coords
                }
                
                legs.append(transfer_leg)
                total_distance += transfer_distance
        
        # === 4. FINAL WALKING LEG TO DESTINATION ===
        final_trip_id = trip_path[-1]
        final_walk = target_trips[final_trip_id]['walk']
        
        if final_walk > 0:
            legs.append({
                "type": "walk",
                "distance_meters": round(final_walk),
                "duration_minutes": max(1, int(final_walk / 83.33)),
                "path": target_trips[final_trip_id]['path']
            })
            total_distance += final_walk
        
        # === 5. BUILD JOURNEY SUMMARY ===
        # Create text summary for LLM consumption
        summary_parts = [f"Total Duration: {int(total_time / 60) if total_time > 60 else int(total_time)} minutes, Total Cost: ${round(total_fare, 2)}, Transfers: {transfers}, Total Walking: {int(total_walk)}m"]
        
        for leg in legs:
            if leg['type'] == 'walk':
                summary_parts.append(f"walk {leg['distance_meters']}m ({leg['duration_minutes']} min)")
            elif leg['type'] == 'trip':
                summary_parts.append(f"take {leg['route_short_name']} to {leg['headsign']} (${leg['fare']}, {leg['duration_minutes']} min) - Board at \"{leg['from']['name']}\", Exit at \"{leg['to']['name']}\"")
            elif leg['type'] == 'transfer':
                summary_parts.append(f"walk {leg['walking_distance_meters']}m ({leg['duration_minutes']} min) - Transfer from {leg['from_trip_name']} to {leg['to_trip_name']}")
        
        text_summary = ", ".join(summary_parts)
        
        journey = {
            "id": journey_id,
            "text_summary": text_summary,
            "summary": {
                "total_time_minutes": int(total_time / 60) if total_time > 60 else int(total_time),  # Convert if in seconds
                "total_distance_meters": int(total_distance),
                "walking_distance_meters": int(total_walk),
                "transfers": transfers,
                "cost": round(total_fare, 2),
                "modes": sorted(list(modes_used))
            },
            "legs": legs
        }
        
        detailed_journeys.append(journey)
    
    return {
        "num_journeys": len(detailed_journeys),
        "journeys": detailed_journeys
    }


class RoutingEngine:
    """Routing engine that loads the price prediction model once at startup."""

    def __init__(self, params_path=None):
        """Initialize the routing engine and load the price prediction model parameters.

        Args:
            params_path: Path to the model params JSON file
        """
        self.db = PostgresConnector()
        self.price_predictor = TripPricePredictor(params_path=params_path)
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

    def get_fare(self, trip_id, start_stop, end_stop, agency='P_O_14'):
        """Calculate the fare of a trip between two stops"""
        passengers = 14
        if agency == 'P_B_8': 
            passengers = 8
        distance = self.get_distance(trip_id, start_stop, end_stop)
        return self.price_predictor.predict(distance, passengers)

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


def get_routing_engine(params_path=None):
    """Get or create the global routing engine instance."""
    global _routing_engine
    if _routing_engine is None:
        _routing_engine = RoutingEngine(params_path)
    return _routing_engine


def get_distance(trip_id, start_stop, end_stop):
    """Get distance for a trip between two stops"""
    engine = get_routing_engine()
    return engine.get_distance(trip_id, start_stop, end_stop)


def get_distance_coord(trip_id, start_lat, start_lon, end_lat, end_lon):
    engine = get_routing_engine()
    return engine.get_distance_coord(trip_id, start_lat, start_lon, end_lat, end_lon)


def get_cost(trip_id, start_stop, end_stop, distances_path=None, params_path=None):
    """Calculate the cost of a trip between two stops"""
    engine = get_routing_engine(params_path)
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
            'stop_id': gtfs_stop_id, # The actual GTFS ID needed for costs
            'osm_node_id': osm_node,       # The physical location
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
    
    def reconstruct_path_coords(node):
        """Convert node path to coordinate path"""
        node_path = reconstruct_path(node)
        coord_path = []
        for node_id in node_path:
            if node_id in G.nodes:
                lat = G.nodes[node_id]['y']
                lon = G.nodes[node_id]['x']
                coord_path.append([lat, lon])
        return coord_path

    while pq:
        d, node = heapq.heappop(pq)

        if d > cutoff:
            break
        if node in visited:
            continue
        visited.add(node)

        # [CHANGE]: Retrieve the access_map with enhanced information
        access_map = G.nodes[node].get("access_map")
        
        # Fallback to boarding_map for backward compatibility
        if not access_map:
            access_map = G.nodes[node].get("boarding_map")
            if access_map:
                # Convert boarding_map format to access_map format
                converted_access_map = {}
                for trip_id, real_stop_id in access_map.items():
                    converted_access_map[trip_id] = {
                        'stop_id': real_stop_id,
                        'agency_id': 'P_O_14',  # default agency
                        'stop_sequence': None
                    }
                access_map = converted_access_map
        
        if access_map:
            # Iterate over the trips available at this node
            for trip_id, access_info in access_map.items():
                
                # Check if this is the best walk to this trip so far
                best = trips.get(trip_id)
                if best is None or d < best["walk"]:
                    trips[trip_id] = {
                        "stop_id": access_info['stop_id'], # GTFS stop ID
                        "agency": access_info.get('agency_id', 'P_O_14'), # needed for fare calculation
                        "stop_sequence": access_info.get('stop_sequence'), # Stop sequence in trip
                        "osm_node_id": node, # OSM node ID
                        "walk": d, # Walking distance
                        "path": reconstruct_path_coords(node) # Walking path coordinates
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


def get_fare(model, trip_id, start_stop, end_stop, agency='P_O_14'):
    """Get fare using the routing engine"""
    engine = get_routing_engine()
    return engine.get_fare(trip_id, start_stop, end_stop, agency)


def get_transport_time(trip_id, start_stop, end_stop, traffic=None):
    """Get transport time using the routing engine"""
    engine = get_routing_engine()

    if traffic is None:
        traffic = engine.load_traffic()

def find_journeys(graph, start_trips, goal_trips, max_transfers):
    results = []
    queue = deque()
    
    # Pareto frontiers for each trip - Maps trip_id -> list of non-dominated cost vectors
    best = {}
    
    # Get the model from routing engine (placeholder for now)
    model = None  # This will be used in get_fare function
    
    def dominates(v1, v2):
        """Check if v1 dominates v2 (v1 is better or equal in all dimensions and strictly better in at least one)"""
        better_in_all = all(v1[i] <= v2[i] for i in range(4))
        strictly_better_in_one = any(v1[i] < v2[i] for i in range(4))
        return better_in_all and strictly_better_in_one
    
    def update_pareto_frontier(frontier, new_cost):
        """Add new_cost to frontier and remove any vectors dominated by new_cost"""
        # Remove vectors dominated by new_cost
        frontier_updated = [v for v in frontier if not dominates(new_cost, v)]
        # Add new_cost if it's not dominated by any existing vector
        if not any(dominates(v, new_cost) for v in frontier_updated):
            frontier_updated.append(new_cost)
        return frontier_updated
    
    # --- 1. Initialize Start Trips ---
    for start_trip_id, data in start_trips.items():
        # Cost vector: (transfers, fare, time, walk)
        c0 = (0, 0, 0, data['walk'])
        path = [start_trip_id]
        start_stop = data['stop_id']
        start_sequence = data['stop_sequence']
        
        queue.append((start_trip_id, start_stop, start_sequence, path, c0))
        best[start_trip_id] = [c0]
        
        # Check 0-transfer goal
        if start_trip_id in goal_trips:
            goal_stop = goal_trips[start_trip_id]['stop_id']
            goal_seq = goal_trips[start_trip_id]['stop_sequence']
            
            # [CHECK 1] If direction is invalid, ignore this result
            if start_sequence < goal_seq:
                delta_fare = get_fare(model, start_trip_id, start_stop, goal_stop, data['agency'])
                delta_time = get_transport_time(start_trip_id, start_stop, goal_stop)
                
                c_final = (
                    c0[0],  # transfers (0)
                    c0[1] + delta_fare,  # fare
                    c0[2] + delta_time,  # time
                    c0[3] + goal_trips[start_trip_id]['walk']  # walk
                )
                results.append((path, c_final))
    
    # --- 2. BFS with Pareto Pruning ---
    while queue:
        (current_trip, current_board_stop, current_board_sequence, path, c) = queue.popleft()
        
        if len(path) - 1 >= max_transfers:
            continue
            
        for next_trip, lean_pathway in graph.get(current_trip, {}).items():
            # lean_pathway now contains only essential routing data
            
            if next_trip in path: 
                continue

            # [CHECK 2 - CRITICAL] Prune invalid path segment immediately
            if current_board_sequence >= lean_pathway['start_stop_sequence']:
                continue
            
            # --- Cost Logic ---
            # Calculate time for the segment we just rode
            prev_trip_time = get_transport_time(
                current_trip, 
                current_board_stop, 
                lean_pathway['start_stop_id']
            )

            # Cost of the PREVIOUS trip segment
            prev_trip_money = get_fare(
                model,
                current_trip, 
                current_board_stop, 
                lean_pathway['start_stop_id'],
                lean_pathway['start_agency_id']
            )

            # New cost vector: (transfers, fare, time, walk)
            c_new = (
                c[0] + 1,  # transfers
                c[1] + prev_trip_money,  # fare
                c[2] + prev_trip_time,  # time
                c[3] + lean_pathway['walking_distance_m']  # walk
            )
            
            # --- Pareto Pruning ---
            # Check if dominated by existing solutions for this trip
            if next_trip in best:
                if any(dominates(v, c_new) for v in best[next_trip]):
                    continue
            
            # Update Pareto frontier
            if next_trip not in best:
                best[next_trip] = []
            best[next_trip] = update_pareto_frontier(best[next_trip], c_new)
            
            new_path = path + [next_trip]
            
            # We board the NEXT trip at lean_pathway['end_stop_id'] with sequence lean_pathway['end_stop_sequence']
            queue.append((next_trip, lean_pathway['end_stop_id'], lean_pathway['end_stop_sequence'], new_path, c_new))
            
            # --- 3. Check Goal ---
            if next_trip in goal_trips:
                goal_stop = goal_trips[next_trip]['stop_id']
                # [CHECK 3] if transfer-in sequence < goal sequence (valid forward direction)
                transfer_in_seq = lean_pathway['end_stop_sequence']
                goal_seq = goal_trips[next_trip]['stop_sequence']
                if transfer_in_seq < goal_seq:
                    # FINAL leg cost (from transfer-in to goal-stop)
                    last_leg_time = get_transport_time(next_trip, lean_pathway['end_stop_id'], goal_stop)
                    last_leg_money = get_fare(model, next_trip, lean_pathway['end_stop_id'], goal_stop, goal_trips[next_trip]['agency'])
                    
                    c_final = (
                        c_new[0],  # transfers (unchanged)
                        c_new[1] + last_leg_money,  # fare
                        c_new[2] + last_leg_time,  # time
                        c_new[3] + goal_trips[next_trip]['walk']  # walk
                    )
                    results.append((new_path, c_final))
    
    return results


def find_route(
    start_lat,
    start_lon,
    end_lat,
    end_lon,
    walking_cutoff,
    max_transfers,
    graph,
    trip_graph,
    pathway_metadata,
    enrichment_lookups,
    routing_engine,
):
    """
    High-level function to find routes from start to end coordinates.

    Args:
        start_lat: Starting latitude
        start_lon: Starting longitude
        end_lat: Ending latitude
        end_lon: Ending longitude
        walking_cutoff: Maximum walking distance in meters
        max_transfers: Maximum number of transfers allowed
        graph: OSM street network graph
        trip_graph: Transit trip graph
        pathway_metadata: Dictionary of pathway metadata for enrichment
        enrichment_lookups: Dictionary of lookup tables for enriching results
        routing_engine: RoutingEngine instance

    Returns:
        dict with keys:
            'journeys': List of (path, costs) tuples
            'start_trips_found': Number of start trips found
            'end_trips_found': Number of end trips found
            'error': Error message if any (None otherwise)
            'error_type': Type of error ('start_trips', 'end_trips', or None)
    """
    import osmnx as ox

    # Find nearest nodes to start and end coordinates
    start_node = ox.distance.nearest_nodes(graph, X=start_lon, Y=start_lat)
    end_node = ox.distance.nearest_nodes(graph, X=end_lon, Y=end_lat)

    # Explore reachable trips from start location
    start_trips = explore_trips(graph, start_node, cutoff=walking_cutoff)

    # Explore reachable trips from end location
    target_trips = explore_trips(graph, end_node, cutoff=walking_cutoff)

    # Check if we found any trips
    if not start_trips:
        return {
            "journeys": [],
            "start_trips_found": 0,
            "end_trips_found": len(target_trips),
            "error": f"No transit trips found within {walking_cutoff}m of start location",
            "error_type": "start_trips",
        }

    if not target_trips:
        return {
            "journeys": [],
            "start_trips_found": len(start_trips),
            "end_trips_found": 0,
            "error": f"No transit trips found within {walking_cutoff}m of end location",
            "error_type": "end_trips",
        }

    # Find journeys using the new Pareto optimization logic
    lean_journeys = find_journeys(
        graph=trip_graph,
        start_trips=start_trips,
        goal_trips=target_trips,
        max_transfers=max_transfers,
    )

    # Enrich the lean journey results into detailed frontend-ready format
    enriched_results = enrich_journey_results(
        lean_journeys=lean_journeys,
        start_trips=start_trips,
        target_trips=target_trips,
        trip_graph=trip_graph,
        pathway_metadata=pathway_metadata,
        enrichment_lookups=enrichment_lookups
    )

    return {
        "num_journeys": enriched_results["num_journeys"],
        "journeys": enriched_results["journeys"],
        "start_trips_found": len(start_trips),
        "end_trips_found": len(target_trips),
        "error": None,
        "error_type": None,
    }