import grpc
from concurrent import futures
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from routing_module.routing import create_routing_engine

import routing_pb2
import routing_pb2_grpc

# Global routing engine instance
_routing_engine = None


def initialize_network():
    global _routing_engine

    print("=" * 60)
    print("Loading network data at startup...")
    print("=" * 60)

    # Use the convenience function to create and initialize everything
    _routing_engine = create_routing_engine()

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
            if _routing_engine is None:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details(
                    "Server is still loading network data. Please try again in a moment."
                )
                return routing_pb2.RouteResponse()

            # Call the routing logic using the new RoutingEngine interface
            result = _routing_engine.find_route(
                start_lat=request.start_lat,
                start_lon=request.start_lon,
                end_lat=request.end_lat,
                end_lon=request.end_lon,
                walking_cutoff=request.walking_cutoff,
                max_transfers=request.max_transfers,
            )

            # Handle errors from routing
            if result["error"]:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(result["error"])
                return routing_pb2.RouteResponse()

            # Format results for gRPC response
            # result["journeys"] now contains enriched journey dictionaries
            grpc_journeys = []
            for journey_dict in result["journeys"]:
                # Extract basic info from enriched journey for gRPC
                summary = journey_dict["summary"]
                
                # Extract trip path from legs (only trip legs, not walk/transfer)
                trip_path = []
                for leg in journey_dict["legs"]:
                    if leg["type"] == "trip":
                        trip_path.append(leg["trip_id"])
                
                journey = routing_pb2.Journey(
                    path=trip_path,
                    costs=routing_pb2.JourneyCosts(
                        money=summary["cost"],
                        transport_time=summary["total_time_minutes"],
                        walk=summary["walking_distance_meters"],
                    ),
                )
                grpc_journeys.append(journey)

            return routing_pb2.RouteResponse(
                num_journeys=result["num_journeys"],
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
