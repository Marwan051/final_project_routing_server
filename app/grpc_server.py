import grpc
from concurrent import futures
import sys
import os
import time

# Add the parent directory to the path to import routing module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from routing_module.database import PostgresConnector
from routing_module.routing import (
    find_journeys,
    explore_trips,
    get_routing_engine,
)
from routing_module.network import create_network
from routing_module.price_predictor import TripPricePredictor
import osmnx as ox

# Import generated gRPC code
import routing_pb2
import routing_pb2_grpc

# Make TripPricePredictor available in __main__ for pickle to find it
import __main__

__main__.TripPricePredictor = TripPricePredictor

# Global variables to store loaded data
graph = None
gtfs_data = None
trip_graph = None
pathways_dict = None
routing_engine = None


def initialize_network():
    """Load all network data at startup"""
    global graph, gtfs_data, trip_graph, pathways_dict, routing_engine

    print("=" * 60)
    print("Loading network data at startup...")
    print("=" * 60)

    # Initialize routing engine (loads model once)
    routing_engine = get_routing_engine()

    # Create the network
    graph, gtfs_data, trip_graph, pathways_dict = create_network()

    print("\n" + "=" * 60)
    print("Server ready! All data loaded.")
    print("=" * 60)


class RoutingServiceServicer(routing_pb2_grpc.RoutingServiceServicer):
    """Implementation of RoutingService"""

    def HealthCheck(self, _, context: grpc.ServicerContext):
        """Health check endpoint"""
        context.set_code(grpc.StatusCode.OK)
        return routing_pb2.HealthResponse(
            status="healthy", message="Transit Routing gRPC Service is running"
        )

    def FindRoute(self, request, context: grpc.ServicerContext):
        """Find transit routes between start and end coordinates"""
        try:
            # Validate that network data is loaded
            if (
                graph is None
                or trip_graph is None
                or pathways_dict is None
                or routing_engine is None
            ):
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details(
                    "Server is still loading network data. Please try again in a moment."
                )
                return routing_pb2.RouteResponse()

            # Find nearest nodes to start and end coordinates
            start_node = ox.distance.nearest_nodes(
                graph, X=request.start_lon, Y=request.start_lat
            )
            end_node = ox.distance.nearest_nodes(
                graph, X=request.end_lon, Y=request.end_lat
            )

            # Explore reachable trips from start location
            start_trips = explore_trips(
                graph, start_node, cutoff=request.walking_cutoff
            )

            # Explore reachable trips from end location
            target_trips = explore_trips(graph, end_node, cutoff=request.walking_cutoff)

            # Check if we found any trips
            if not start_trips:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(
                    f"No transit trips found within {request.walking_cutoff}m of start location"
                )
                return routing_pb2.RouteResponse()

            if not target_trips:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(
                    f"No transit trips found within {request.walking_cutoff}m of end location"
                )
                return routing_pb2.RouteResponse()

            # Find journeys
            journeys = find_journeys(
                graph=trip_graph,
                pathways_dict=pathways_dict,
                start_trips=start_trips,
                goal_trips=target_trips,
                max_transfers=request.max_transfers,
                traffic=routing_engine.load_traffic(),
            )

            # Format results for gRPC response
            grpc_journeys = []
            for path, costs in journeys:
                journey = routing_pb2.Journey(
                    path=path,
                    costs=routing_pb2.JourneyCosts(
                        money=costs["money"],
                        transport_time=costs["transport_time"],
                        walk=costs["walk"],
                    ),
                )
                grpc_journeys.append(journey)

            return routing_pb2.RouteResponse(
                num_journeys=len(journeys),
                journeys=grpc_journeys,
                start_trips_found=len(start_trips),
                end_trips_found=len(target_trips),
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return routing_pb2.RouteResponse()


def serve():
    # Initialize network data
    initialize_network()

    db_manager = PostgresConnector()

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    routing_pb2_grpc.add_RoutingServiceServicer_to_server(
        RoutingServiceServicer(), server
    )

    # Listen on port 50051
    port = "50051"
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    print(f"\n{'=' * 60}")
    print(f"gRPC Server started on port {port}")
    print(f"{'=' * 60}\n")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
