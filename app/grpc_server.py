import grpc
from concurrent import futures
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from routing_module.routing import (
    find_route,
    get_routing_engine,
)
from routing_module.network import create_network

import routing_pb2
import routing_pb2_grpc

graph = None
gtfs_data = None
trip_graph = None
pathways_dict = None
routing_engine = None


def initialize_network():
    global graph, gtfs_data, trip_graph, pathways_dict, routing_engine

    print("=" * 60)
    print("Loading network data at startup...")
    print("=" * 60)

    routing_engine = get_routing_engine()

    graph, gtfs_data, trip_graph, pathways_dict = create_network()

    print("\n" + "=" * 60)
    print("Server ready! All data loaded.")
    print("=" * 60)


class RoutingServiceServicer(routing_pb2_grpc.RoutingServiceServicer):

    def HealthCheck(self, _, context: grpc.ServicerContext):
        context.set_code(grpc.StatusCode.OK)
        return routing_pb2.HealthResponse(
            status="healthy", message="Transit Routing gRPC Service is running"
        )

    def FindRoute(
        self,
        request: routing_pb2.RouteRequest,
        context: grpc.ServicerContext,
    ) -> routing_pb2.RouteResponse:
        try:
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

            # Call the routing logic
            result = find_route(
                start_lat=request.start_lat,
                start_lon=request.start_lon,
                end_lat=request.end_lat,
                end_lon=request.end_lon,
                walking_cutoff=request.walking_cutoff,
                max_transfers=request.max_transfers,
                graph=graph,
                trip_graph=trip_graph,
                pathways_dict=pathways_dict,
                routing_engine=routing_engine,
            )

            # Handle errors from routing
            if result["error"]:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(result["error"])
                return routing_pb2.RouteResponse()

            # Format results for gRPC response
            grpc_journeys = []
            for path, costs in result["journeys"]:
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
                num_journeys=len(result["journeys"]),
                journeys=grpc_journeys,
                start_trips_found=result["start_trips_found"],
                end_trips_found=result["end_trips_found"],
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return routing_pb2.RouteResponse()


def serve():

    initialize_network()

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
