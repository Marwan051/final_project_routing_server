import osmnx as ox
import pandas as pd
from collections import defaultdict
import os
import ast
import json


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
    routes = gtfs_data['routes']
    trips = gtfs_data['trips']
    
    # 1. Build enhanced mappings with route names and stop sequences
    stop_to_trips = (
        stop_times.groupby('stop_id')['trip_id']
        .apply(list)
        .to_dict()
    )
    # Create lookup dictionaries for enriched data
    trip_to_route = trips.set_index('trip_id')['route_id'].to_dict()
    route_to_agency = routes.set_index('route_id')['agency_id'].to_dict()
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
    print(f"  - {nodes_updated} nodes have trips attached")
    
    return graph


def build_trip_graph(pathways):
    """
    Build lean trip graph and pathway metadata from pathways dataframe.
    
    Args:
        pathways: DataFrame with pathways
        
    Returns:
        Tuple of (trip_graph, pathway_metadata)
        - trip_graph: dict of dict with lean pathway data for routing
        - pathway_metadata: dict with enrichment data for results
    """
    # 1. Build LEAN trip graph - only essential routing data
    trip_graph = defaultdict(dict)

    for idx, row in pathways.iterrows():
        # Store ONLY what's needed for routing algorithm
        lean_pathway = {
            'start_stop_id': row['start_stop_id'],
            'end_stop_id': row['end_stop_id'],
            'start_stop_sequence': row['start_stop_sequence'],
            'end_stop_sequence': row['end_stop_sequence'],
            'start_agency_id': row['start_agency_id'],  # needed for fare calculation
            'walking_distance_m': row['walking_distance_m']
        }
        
        trip_graph[row['start_trip_id']][row['end_trip_id']] = lean_pathway

    # 2. Build lookup tables for enriching results later
    pathway_metadata = {}
    for idx, row in pathways.iterrows():
        key = (row['start_trip_id'], row['end_trip_id'])
        pathway_metadata[key] = {
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

    print(f"Built lean trip graph with {len(trip_graph)} unique starting trips")
    print(f"  - Total edges: {sum(len(v) for v in trip_graph.values())}")
    print(f"  - Pathway metadata entries: {len(pathway_metadata)}")
    
    return trip_graph, pathway_metadata


def build_enrichment_lookups(gtfs_data):
    """
    Build lookup dictionaries needed for enriching journey results.
    
    Args:
        gtfs_data: Dictionary containing GTFS dataframes
        
    Returns:
        Dictionary of lookup tables
    """
    stops = gtfs_data['stops']
    routes = gtfs_data['routes']
    trips = gtfs_data['trips']
    
    lookups = {
        'trip_to_route': trips.set_index('trip_id')['route_id'].to_dict(),
        'route_to_name': routes.set_index('route_id')['route_long_name'].to_dict(),
        'route_to_short_name': routes.set_index('route_id')['route_short_name'].to_dict(),
        'trip_to_headsign': trips.set_index('trip_id')['trip_headsign'].to_dict(),
        'stop_to_coords': stops.set_index('stop_id')[['stop_lat', 'stop_lon']].to_dict('index'),
        'stop_to_name': stops.set_index('stop_id')['stop_name'].to_dict()  # Additional lookup for stop names
    }
    
    print(f"Built enrichment lookup tables with {len(lookups)} dictionaries")
    return lookups


def create_network(osm_file=None, gtfs_path=None, pathways_file=None):
    """
    Complete network creation pipeline with enhanced pathway logic.
    
    Args:
        osm_file: Path to OSM XML file (defaults to data/labeled.osm)
        gtfs_path: Path to GTFS directory (defaults to data/gtfsAlex)
        pathways_file: Path to pathways CSV (defaults to data/trip_pathways.csv)
        
    Returns:
        Tuple of (graph, trip_graph, pathway_metadata, enrichment_lookups)
        Note: gtfs_data is used internally but not returned as it's not needed by RoutingEngine
    """
    print("=" * 60)
    print("Starting Enhanced Network Creation Pipeline")
    print("=" * 60)
    
    # Load network
    print("\n[1/6] Loading OSM network...")
    graph = load_network(osm_file)
    
    # Load GTFS
    print("\n[2/6] Loading GTFS data...")
    gtfs_data = load_gtfs_data(gtfs_path)
    
    # Load pathways
    print("\n[3/6] Loading pathways...")
    pathways = load_pathways(pathways_file)
    
    # Merge trips to network with enhanced access mapping
    print("\n[4/6] Merging trips to network with enhanced access info...")
    graph = merge_trips_to_network(graph, gtfs_data)
    
    # Build lean trip graph and pathway metadata
    print("\n[5/6] Building lean trip graph and pathway metadata...")
    trip_graph, pathway_metadata = build_trip_graph(pathways)
    
    # Build enrichment lookups
    print("\n[6/6] Building enrichment lookup tables...")
    enrichment_lookups = build_enrichment_lookups(gtfs_data)
    
    print("\n" + "=" * 60)
    print("Enhanced Network Creation Complete!")
    print("=" * 60)
    
    # Return in the order expected by RoutingEngine.set_graph_data()
    return graph, trip_graph, pathway_metadata, enrichment_lookups

if __name__ == "__main__":
    graph, trip_graph, pathway_metadata, enrichment_lookups = create_network()
    # Print some statistics
    nodes_with_trips = sum(1 for n, data in graph.nodes(data=True) if 'access_map' in data)
    print(f"\nFinal Statistics:")
    print(f"  - Network nodes: {len(graph.nodes)}")
    print(f"  - Network edges: {len(graph.edges)}")
    print(f"  - Nodes with enhanced access maps: {nodes_with_trips}")
    print(f"  - Pathway metadata entries: {len(pathway_metadata)}")
    print(f"  - Enrichment lookup tables: {len(enrichment_lookups)}")
