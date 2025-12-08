"""
Example script showing how to use the network and routing modules.
"""

import sys
import os

# Add routing_module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'routing_module'))

from routing_module.network import create_network
from routing_module.routing import explore_trips, find_journeys
import osmnx as ox

# Create the network (uses default paths in routing_module/data/)
print("Creating network...")
graph, gtfs_data, trip_graph, pathways_dict = create_network()

# Define start and end coordinates
start_lon, start_lat = 29.96139328537071, 31.22968895248673
end_lon, end_lat = 29.94194179397711, 31.20775934404925

# Find nearest nodes
print("\nFinding nearest nodes to start and end points...")
start_node = ox.distance.nearest_nodes(graph, X=start_lon, Y=start_lat)
end_node = ox.distance.nearest_nodes(graph, X=end_lon, Y=end_lat)

print(f"Start node: {start_node}")
print(f"End node: {end_node}")

# Explore reachable trips from start
print("\nExploring trips from start location (within 1000m)...")
start_trips = explore_trips(graph, start_node, cutoff=1000)
print(f"Found {len(start_trips)} reachable trips from start")

# Explore reachable trips from end
print("\nExploring trips from end location (within 1000m)...")
target_trips = explore_trips(graph, end_node, cutoff=1000)
print(f"Found {len(target_trips)} reachable trips to destination")

# Find journeys
print("\nFinding journeys with up to 2 transfers...")
journeys = find_journeys(
    trip_graph,
    pathways_dict,
    start_trips,
    target_trips,
    max_transfers=2
)

print(f"\n{'=' * 60}")
print(f"Found {len(journeys)} possible journeys")
print(f"{'=' * 60}")

# Display journeys
for i, (path, costs) in enumerate(journeys[:10], 1):  # Show first 10
    print(f"\nJourney {i}:")
    print(f"  Path: {' -> '.join(path)}")
    print(f"  Costs:")
    print(f"    - Money: {costs['money']:.2f} EGP")
    print(f"    - Transport Time: {costs['transport_time']:.1f} minutes")
    print(f"    - Walking Distance: {costs['walk']:.1f} meters")

if len(journeys) > 10:
    print(f"\n... and {len(journeys) - 10} more journeys")
