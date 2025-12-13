import numpy as np
import pandas as pd
import json
import heapq
import os
from collections import deque, defaultdict
from joblib import load
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.pipeline import Pipeline

from routing_module.database import PostgresConnector


class TripPricePredictor(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def _round_bus_style(self, vals):
        """Custom rounding logic for bus fare"""
        scalar = np.isscalar(vals)
        arr = np.array([vals]) if scalar else np.asarray(vals)
        out = []
        for v in arr:
            pounds = int(np.floor(v))
            dec = v - pounds
            if dec < 0.125:
                r = pounds + 0.0
            elif dec < 0.375:
                r = pounds + 0.25
            elif dec < 0.75:
                r = pounds + 0.5
            else:
                r = pounds + 1.0
            out.append(round(r, 2))
        return out[0] if scalar else np.array(out)

    def predict(self, distance_km):
        # 1. Preprocessing: Convert KM to Log Distance
        X = np.array(distance_km)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_log = np.log1p(X)

        # 2. Prediction
        raw_pred = self.model.predict(X_log)

        # 3. Post-processing: Custom Rounding
        return self._round_bus_style(raw_pred)


def load_model(model_path=None):
    """Load the trip price prediction model"""
    if model_path is None:
        # Default to data/utils/trip_price_model.joblib relative to this file
        model_path = os.path.join(
            os.path.dirname(__file__), "data", "utils", "trip_price_model.joblib"
        )
    # Load the model - it might already be a TripPricePredictor or just the sklearn model
    loaded_model = load(model_path)

    # If it's already a TripPricePredictor, return it directly
    if isinstance(loaded_model, TripPricePredictor):
        return loaded_model

    # Otherwise, wrap it in TripPricePredictor
    return TripPricePredictor(loaded_model)


def get_distance(trip_id, start_stop, end_stop, distances_path=None):
    """Get distance for a trip between two stops"""
    return PostgresConnector().get_distance_between_two_stops_within_route(
        trip_id, start_stop, end_stop
    )


# TODO: talk to database (start coord,end coord)
def get_distance_coord(trip_id, start_lat, start_lon, end_lat, end_lon):
    db = PostgresConnector()
    dist = db.get_distance_between_two_coordinates_within_route(
        trip_id, start_lat, start_lon, end_lat, end_lon
    )


def get_cost(trip_id, start_stop, end_stop, distances_path=None, model_path=None):
    """Calculate the cost of a trip between two stops"""
    distance = get_distance(trip_id, start_stop, end_stop, distances_path)
    model = load_model(model_path)

    return model.predict([distance])[0]


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
    if traffic is None:
        traffic = load_traffic()

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

            leg_money = get_cost(start_trip_id, start_stop, goal_stop)
            leg_time = get_transport_time(start_trip_id, start_stop, goal_stop, traffic)

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
            prev_trip_money = get_cost(
                current_trip, current_board_stop, pathway["start_stop_id"]
            )
            prev_trip_time = get_transport_time(
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
                    last_leg_money = get_cost(
                        next_trip, pathway["end_stop_id"], goal_stop
                    )
                    last_leg_time = get_transport_time(
                        next_trip, pathway["end_stop_id"], goal_stop, traffic
                    )

                    final_journey_costs = new_costs.copy()
                    final_journey_costs["money"] += last_leg_money
                    final_journey_costs["transport_time"] += last_leg_time
                    final_journey_costs["walk"] += goal_trips[next_trip]["walk"]

                    results.append((new_path, final_journey_costs))

    return results


def _estimator_to_dict(estimator):
    """Convert an sklearn estimator (or pipeline) to a serializable dict.

    - Uses `get_params()` for general params.
    - Adds `coef_`, `intercept_`, and `feature_importances_` if present.
    - If the estimator is a Pipeline, extracts step info recursively.
    """
    if estimator is None:
        return None

    # If wrapped in TripPricePredictor, unwrap
    if isinstance(estimator, TripPricePredictor):
        estimator = estimator.model

    result = {}

    # Basic params
    try:
        params = estimator.get_params(deep=False)
        # Convert any non-serializable values to strings
        safe_params = {}
        for k, v in params.items():
            try:
                json.dumps({k: v})
                safe_params[k] = v
            except Exception:
                safe_params[k] = str(v)
        result["params"] = safe_params
    except Exception:
        result["params"] = str(
            getattr(estimator, "__class__", type(estimator)).__name__
        )

    # Coefficients / intercept
    if hasattr(estimator, "coef_"):
        try:
            coef = estimator.coef_
            result["coef"] = coef.tolist() if hasattr(coef, "tolist") else coef
        except Exception:
            result["coef"] = str(getattr(estimator, "coef_", None))

    if hasattr(estimator, "intercept_"):
        try:
            intercept = estimator.intercept_
            result["intercept"] = (
                intercept.tolist() if hasattr(intercept, "tolist") else intercept
            )
        except Exception:
            result["intercept"] = str(getattr(estimator, "intercept_", None))

    # Feature importances (tree-based)
    if hasattr(estimator, "feature_importances_"):
        try:
            fi = estimator.feature_importances_
            result["feature_importances"] = fi.tolist() if hasattr(fi, "tolist") else fi
        except Exception:
            result["feature_importances"] = str(
                getattr(estimator, "feature_importances_", None)
            )

    # Pipeline handling
    if isinstance(estimator, Pipeline):
        steps_info = []
        for name, step in estimator.steps:
            steps_info.append(
                {
                    "name": name,
                    "class": step.__class__.__name__,
                    "estimator": _estimator_to_dict(step),
                }
            )
        result["pipeline"] = steps_info

    return result


def extract_model_params(model_path=None, out_path=None):
    """Load the saved model and return a JSON-serializable dictionary of its parameters.

    If `out_path` is provided, the resulting dict will be written to that file.
    """
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "data", "utils", "trip_price_model.joblib"
        )

    loaded = load(model_path)

    # If the saved object is the wrapper, use it; otherwise wrap
    if isinstance(loaded, TripPricePredictor):
        estimator = loaded
    else:
        estimator = TripPricePredictor(loaded)

    info = {
        "class": estimator.__class__.__name__,
        "wrapped_model": _estimator_to_dict(estimator),
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(info, f, indent=2)

    return info
