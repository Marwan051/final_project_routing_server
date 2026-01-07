import osmnx as ox
import pandas as pd
from collections import defaultdict
import os
import json
from joblib import load
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_network(osm_file_path=None):
    """
    Load the OSM network from XML file.
    
    Args:
        osm_file_path: Path to the OSM XML file
        
    Returns:
        NetworkX graph object
    """
    if osm_file_path is None:
        osm_file_path = os.path.join(os.path.dirname(__file__), 'data', 'labeled.osm')
    g = ox.graph_from_xml(osm_file_path, bidirectional=True, simplify=True)
    print(f"Loaded network with {len(g.nodes)} nodes and {len(g.edges)} edges")
    return g


def load_gtfs_data(gtfs_path=None):
    """
    Load all GTFS data files.
    
    Args:
        gtfs_path: Path to the GTFS directory
        
    Returns:
        Dictionary containing all GTFS dataframes
    """
    if gtfs_path is None:
        gtfs_path = os.path.join(os.path.dirname(__file__), 'data', 'gtfsAlex')
    gtfs_data = {
        'stops': pd.read_csv(os.path.join(gtfs_path, "stops.txt")),
        'routes': pd.read_csv(os.path.join(gtfs_path, "routes.txt")),
        'trips': pd.read_csv(os.path.join(gtfs_path, "trips.txt")),
        'stop_times': pd.read_csv(os.path.join(gtfs_path, "stop_times.txt")),
        'shapes': pd.read_csv(os.path.join(gtfs_path, "shapes.txt"))
    }
    
    print(f"Loaded GTFS data:")
    print(f"  - {len(gtfs_data['stops'])} stops")
    print(f"  - {len(gtfs_data['routes'])} routes")
    print(f"  - {len(gtfs_data['trips'])} trips")
    print(f"  - {len(gtfs_data['stop_times'])} stop times")
    print(f"  - {len(gtfs_data['shapes'])} shapes")
    
    return gtfs_data


def load_pathways(pathways_file=None):
    """
    Load the trip pathways CSV.
    
    Args:
        pathways_file: Path to the pathways CSV file
        
    Returns:
        Pandas DataFrame with pathways
    """
    if pathways_file is None:
        pathways_file = os.path.join(os.path.dirname(__file__), 'data', 'trip_pathways.csv')
    pathways = pd.read_csv(pathways_file)
    print(f"Loaded {len(pathways)} pathways")
    return pathways


def load_cost_model(model_file=None):
    """
    Load the fare prediction model.
    
    Args:
        model_file: Path to the model file
        
    Returns:
        Loaded joblib model
    """
    if model_file is None:
        model_file = os.path.join(os.path.dirname(__file__), 'data', 'utils', 'model.pkl')
    model = load(model_file)
    print(f"Loaded cost model with intercept: {model.intercept_:.3f}")
    return model


def load_traffic_data(traffic_file=None):
    """
    Load traffic/timing data for trips.
    
    Args:
        traffic_file: Path to the traffic JSON file
        
    Returns:
        Dictionary with traffic timing data
    """
    if traffic_file is None:
        traffic_file = os.path.join(os.path.dirname(__file__), 'data', 'prefixtimes.json')
    with open(traffic_file, 'r') as f:
        traffic_data = json.load(f)
    print(f"Loaded traffic data for {len(traffic_data)} trips")
    return traffic_data


def load_distance_data(distance_file=None):
    """
    Load trip distance data.
    
    Args:
        distance_file: Path to the distance CSV file
        
    Returns:
        Dictionary mapping trip_id to distance_km
    """
    if distance_file is None:
        distance_file = os.path.join(os.path.dirname(__file__), 'data', 'utils', 'trip_distances.csv')
    dist_df = pd.read_csv(distance_file)
    trip_distance = dict(zip(dist_df["trip_id"], dist_df["distance_km"]))
    print(f"Loaded distance data for {len(trip_distance)} trips")
    return trip_distance


def create_lookup_dictionaries(gtfs_data):
    """
    Create all lookup dictionaries needed for routing and enrichment.
    
    Args:
        gtfs_data: Dictionary containing GTFS dataframes
        
    Returns:
        Dictionary containing all lookup tables
    """
    stops = gtfs_data['stops']
    routes = gtfs_data['routes']
    trips = gtfs_data['trips']
    
    lookups = {
        'trip_to_route': trips.set_index('trip_id')['route_id'].to_dict(),
        'route_to_name': routes.set_index('route_id')['route_long_name'].to_dict(),
        'route_to_short_name': routes.set_index('route_id')['route_short_name'].to_dict(),
        'route_to_agency': routes.set_index('route_id')['agency_id'].to_dict(),
        'trip_to_headsign': trips.set_index('trip_id')['trip_headsign'].to_dict(),
        'stop_to_coords': stops.set_index('stop_id')[['stop_lat', 'stop_lon']].to_dict('index'),
        'stop_to_name': stops.set_index('stop_id')['stop_name'].to_dict()
    }
    
    print(f"Created lookup dictionaries:")
    for name, lookup in lookups.items():
        print(f"  - {name}: {len(lookup)} entries")
    
    return lookups


def merge_trips_to_network(graph, gtfs_data):
    """
    Merge GTFS trips to the network graph by attaching enhanced access_map to nodes.
    
    Args:
        graph: NetworkX graph from OSM
        gtfs_data: Dictionary containing GTFS dataframes
        
    Returns:
        Modified graph with access_map attached to nodes
    """
    stops = gtfs_data['stops']
    stop_times = gtfs_data['stop_times']
    trips = gtfs_data['trips']
    routes = gtfs_data['routes']
    
    # 1. Build enhanced mappings with route names and stop sequences
    stop_to_trips = (
        stop_times.groupby('stop_id')['trip_id']
        .apply(list)
        .to_dict()
    )
    
    # Create lookup dictionaries for enriched data
    trip_to_route = trips.set_index('trip_id')['route_id'].to_dict()
    route_to_agency = routes.set_index('route_id')['agency_id'].to_dict()
    stop_to_coords = stops.set_index('stop_id')[['stop_lat', 'stop_lon']].to_dict('index')
    
    # Create (trip_id, stop_id) -> stop_sequence mapping
    trip_stop_to_sequence = stop_times.set_index(['trip_id', 'stop_id'])['stop_sequence'].to_dict()

    # 2. Map stops to nearest nodes (Vectorized)
    stop_nodes = ox.distance.nearest_nodes(
        graph,
        X=stops['stop_lon'].values,
        Y=stops['stop_lat'].values
    )
    stop_to_node_map = pd.Series(stop_nodes, index=stops['stop_id']).to_dict()

    # 3. Attach enhanced access information to OSM nodes
    nodes_updated = 0
    for stop_id, node_id in stop_to_node_map.items():
        trips_at_stop = stop_to_trips.get(stop_id)
        
        if trips_at_stop:
            # Initialize the access map if it doesn't exist
            if 'access_map' not in graph.nodes[node_id]:
                graph.nodes[node_id]['access_map'] = {}
                nodes_updated += 1
            
            # Map each trip with LEAN information (only what's needed for routing)
            for trip_id in trips_at_stop:
                route_id = trip_to_route.get(trip_id)
                agency_id = route_to_agency.get(route_id, "Unknown Agency") if route_id else "Unknown Agency"
                stop_sequence = trip_stop_to_sequence.get((trip_id, stop_id))
                
                graph.nodes[node_id]['access_map'][trip_id] = {
                    'stop_id': stop_id,
                    'stop_sequence': stop_sequence,
                    'agency_id': agency_id  # needed for fare calculation
                }

    print(f"Finished mapping trips to graph nodes with enhanced access information.")
    print(f"  - {nodes_updated} nodes have enhanced access_map")
    
    return graph


def build_trip_graph(pathways):
    """
    Build  trip graph with all needed data for routing and enrichment.
    
    Args:
        pathways: DataFrame with pathways
        
    Returns:
        Dictionary where trip_graph[start_trip][end_trip] = pathway_data
    """
    trip_graph = defaultdict(dict)

    for idx, row in pathways.iterrows():
        # Store ALL data needed for both routing AND enrichment
        pathway = {
            # Essential routing data
            'start_stop_id': row['start_stop_id'],
            'end_stop_id': row['end_stop_id'],
            'start_stop_sequence': row['start_stop_sequence'],
            'end_stop_sequence': row['end_stop_sequence'],
            'start_agency_id': row['start_agency_id'],
            'end_agency_id': row.get('end_agency_id', row['start_agency_id']),
            'walking_distance_m': row['walking_distance_m'],
            # Enrichment data (previously in pathway_metadata)
            'pathway_id': idx,
            'start_route_id': row['start_route_id'],
            'end_route_id': row['end_route_id'],
            'start_route_name': row['start_route_name'],
            'end_route_name': row['end_route_name'],
            'start_route_short_name': row['start_route_short_name'],
            'end_route_short_name': row['end_route_short_name'],
            'start_trip_headsign': row['start_trip_headsign'],
            'end_trip_headsign': row['end_trip_headsign'],
            'start_stop_lat': row['start_stop_lat'],
            'start_stop_lon': row['start_stop_lon'],
            'end_stop_lat': row['end_stop_lat'],
            'end_stop_lon': row['end_stop_lon'],
            'walking_path_coords': row['walking_path_coords']
        }
        
        trip_graph[row['start_trip_id']][row['end_trip_id']] = pathway

    print(f" pathway graph built with {len(trip_graph)} unique starting trips and {sum(len(v) for v in trip_graph.values())} edges.")
    
    return trip_graph


def create_network(osm_file=None, gtfs_path=None, pathways_file=None, model_file=None, traffic_file=None, distance_file=None):
    """
    Complete network creation pipeline with all preprocessing needed for routing.
    
    Args:
        osm_file: Path to OSM XML file (defaults to data/labeled.osm)
        gtfs_path: Path to GTFS directory (defaults to data/gtfsAlex)
        pathways_file: Path to pathways CSV (defaults to data/trip_pathways.csv)
        model_file: Path to cost model file (defaults to data/utils/trip_price_model.joblib)
        traffic_file: Path to traffic JSON file (defaults to data/prefixtimes.json)
        distance_file: Path to distance CSV file (defaults to data/utils/trip_distances.csv)
        
    Returns:
        Dictionary containing all network data needed for routing:
        {
            'graph': NetworkX graph with access_map,
            'gtfs_data': GTFS dataframes,
            'trip_graph': pathway graph,
            'lookups': All lookup dictionaries,
            'cost_model': Fare prediction model,
            'traffic_data': Timing data,
            'distance_data': Distance data
        }
    """
    print("=" * 70)
    print("Starting Network Creation Pipeline")
    print("=" * 70)
    
    # Load network
    print("\n[1/9] Loading OSM network...")
    graph = load_network(osm_file)
    
    # Load GTFS
    print("\n[2/9] Loading GTFS data...")
    gtfs_data = load_gtfs_data(gtfs_path)
    
    # Load pathways
    print("\n[3/9] Loading pathways...")
    pathways = load_pathways(pathways_file)
    
    # Create lookup dictionaries
    print("\n[4/9] Creating lookup dictionaries...")
    lookups = create_lookup_dictionaries(gtfs_data)
    
    # Merge trips to network
    print("\n[5/9] Merging trips to network with enhanced access...")
    graph = merge_trips_to_network(graph, gtfs_data)
    
    # Build  trip graph
    print("\n[6/9] Building  trip graph...")
    trip_graph = build_trip_graph(pathways)
    
    # Load cost model
    print("\n[7/9] Loading cost model...")
    cost_model = load_cost_model(model_file)
    
    # Load traffic data
    print("\n[8/9] Loading traffic data...")
    traffic_data = load_traffic_data(traffic_file)
    
    # Load distance data
    print("\n[9/9] Loading distance data...")
    distance_data = load_distance_data(distance_file)
    
    # Compile all network data
    network_data = {
        'graph': graph,
        'gtfs_data': gtfs_data,
        'trip_graph': trip_graph,
        'lookups': lookups,
        'cost_model': cost_model,
        'traffic_data': traffic_data,
        'distance_data': distance_data
    }
    
    print("\n" + "=" * 70)
    print(" Network Creation Complete!")
    print("=" * 70)
    print(f"Network includes:")
    print(f"  - OSM Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"  - Enhanced Access: {sum(1 for n, data in graph.nodes(data=True) if 'access_map' in data)} nodes with trips")
    print(f"  - Trip Graph: {len(trip_graph)} trips with {sum(len(v) for v in trip_graph.values())} pathways")
    print(f"  - Cost Model: Loaded with {len(lookups['trip_to_route'])} trip mappings")
    print(f"  - Traffic Data: {len(traffic_data)} trips")
    print(f"  - Distance Data: {len(distance_data)} trips")
    
    return network_data


if __name__ == "__main__":
    # Example usage
    network_data = create_network()
    
    # Print some statistics
    graph = network_data['graph']
    nodes_with_trips = sum(1 for n, data in graph.nodes(data=True) if 'access_map' in data)
    print(f"\nFinal Statistics:")
    print(f"  - Network nodes: {len(graph.nodes)}")
    print(f"  - Network edges: {len(graph.edges)}")
    print(f"  - Nodes with enhanced access: {nodes_with_trips}")
    print(f"  - Lookup dictionaries: {len(network_data['lookups'])}")
    print(f"  - Ready for routing!")
