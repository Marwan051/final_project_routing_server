import osmnx as ox
import pandas as pd
from collections import defaultdict
import os


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
    Merge GTFS trips to the network graph by attaching boarding_map to nodes.
    
    Args:
        graph: NetworkX graph from OSM
        gtfs_data: Dictionary containing GTFS dataframes
        
    Returns:
        Modified graph with boarding_map attached to nodes
    """
    stops = gtfs_data['stops']
    stop_times = gtfs_data['stop_times']
    
    # 1. Build a mapping: stop_id -> list of trip_ids
    stop_to_trips = (
        stop_times.groupby('stop_id')['trip_id']
        .apply(list)
        .to_dict()
    )

    # 2. Map stops to nearest nodes (Vectorized)
    stop_nodes = ox.distance.nearest_nodes(
        graph,
        X=stops['stop_lon'].values,
        Y=stops['stop_lat'].values
    )
    stop_to_node_map = pd.Series(stop_nodes, index=stops['stop_id']).to_dict()

    # 3. Attach {trip_id: stop_id} to the OSM nodes
    nodes_updated = 0
    for stop_id, node_id in stop_to_node_map.items():
        trips_at_stop = stop_to_trips.get(stop_id)
        
        if trips_at_stop:
            # Initialize the map if it doesn't exist
            if 'boarding_map' not in graph.nodes[node_id]:
                graph.nodes[node_id]['boarding_map'] = {}
                nodes_updated += 1
            
            # Map every trip at this stop to THIS specific stop_id
            for trip_id in trips_at_stop:
                graph.nodes[node_id]['boarding_map'][trip_id] = stop_id

    print(f"Finished mapping trips to graph nodes.")
    print(f"  - {nodes_updated} nodes have trips attached")
    
    return graph


def build_trip_graph(pathways):
    """
    Build a trip graph from pathways dataframe.
    
    Args:
        pathways: DataFrame with pathways
        
    Returns:
        Tuple of (trip_graph, pathways_dict)
        - trip_graph: dict of dict where trip_graph[start_trip][end_trip] = pathway_id
        - pathways_dict: dict indexed by pathway_id
    """
    trip_graph = defaultdict(dict)
    pathways_dict = pathways.to_dict('index')

    for idx, row in pathways.iterrows():
        trip_graph[row['start_trip_id']][row['end_trip_id']] = idx

    print(f"Built trip graph with {len(trip_graph)} unique starting trips")
    print(f"  - Total edges: {sum(len(v) for v in trip_graph.values())}")
    
    return trip_graph, pathways_dict


def create_network(osm_file=None, gtfs_path=None, pathways_file=None):
    """
    Complete network creation pipeline.
    
    Args:
        osm_file: Path to OSM XML file (defaults to data/labeled.osm)
        gtfs_path: Path to GTFS directory (defaults to data/gtfsAlex)
        pathways_file: Path to pathways CSV (defaults to data/trip_pathways.csv)
        
    Returns:
        Tuple of (graph, gtfs_data, trip_graph, pathways_dict)
    """
    print("=" * 60)
    print("Starting Network Creation Pipeline")
    print("=" * 60)
    
    # Load network
    print("\n[1/5] Loading OSM network...")
    graph = load_network(osm_file)
    
    # Load GTFS
    print("\n[2/5] Loading GTFS data...")
    gtfs_data = load_gtfs_data(gtfs_path)
    
    # Load pathways
    print("\n[3/5] Loading pathways...")
    pathways = load_pathways(pathways_file)
    
    # Merge trips to network
    print("\n[4/5] Merging trips to network...")
    graph = merge_trips_to_network(graph, gtfs_data)
    
    # Build trip graph
    print("\n[5/5] Building trip graph...")
    trip_graph, pathways_dict = build_trip_graph(pathways)
    
    print("\n" + "=" * 60)
    print("Network Creation Complete!")
    print("=" * 60)
    
    return graph, gtfs_data, trip_graph, pathways_dict


if __name__ == "__main__":
    # Example usage
    graph, gtfs_data, trip_graph, pathways_dict = create_network()
    
    # Print some statistics
    nodes_with_trips = sum(1 for n, data in graph.nodes(data=True) if 'boarding_map' in data)
    print(f"\nFinal Statistics:")
    print(f"  - Network nodes: {len(graph.nodes)}")
    print(f"  - Network edges: {len(graph.edges)}")
    print(f"  - Nodes with trips: {nodes_with_trips}")
