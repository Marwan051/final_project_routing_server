import grpc
from concurrent import futures
import sys
import os
import json


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from routing_module.routing import get_routing_engine
from routing_module.network import create_network
import routing_pb2
import routing_pb2_grpc

routing_engine = None


def initialize_network():
    global routing_engine

    print("=" * 60)
    print("Loading network data at startup...")
    print("=" * 60)

    # Create network data
    network_data = create_network()

    # Initialize routing engine with network data
    routing_engine = get_routing_engine(network_data)

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
            if routing_engine is None:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details(
                    "Server is still loading network data. Please try again in a moment."
                )
                return routing_pb2.RouteResponse()

            # Call the new routing logic
            # Debug: print available fields
            print(
                f"Available request fields: {[field.name for field in request.DESCRIPTOR.fields]}"
            )

            # Extract weights if provided
            weights = None
            if request.HasField("weights"):
                weights = {
                    "time": request.weights.time,
                    "cost": request.weights.cost,
                    "walk": request.weights.walk,
                    "transfer": request.weights.transfer,
                }

            result = routing_engine.find_journeys(
                start_lat=request.start_lat,
                start_lon=request.start_lon,
                end_lat=request.end_lat,
                end_lon=request.end_lon,
                walking_cutoff=request.walking_cutoff,
                max_transfers=request.max_transfers,
                weights=weights,
                restricted_modes=(
                    list(request.restricted_modes) if request.restricted_modes else []
                ),
                top_k=(
                    request.top_k if request.top_k > 0 else 5
                ),  # Default to 5 if not specified or invalid
            )

            # Handle errors from routing
            if result.get("error"):
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(result["error"])
                return routing_pb2.RouteResponse(
                    num_journeys=0,
                    start_trips_found=result.get("start_trips_found", 0),
                    end_trips_found=result.get("end_trips_found", 0),
                    total_routes_found=result.get("total_routes_found", 0),
                    error=result["error"],
                )

            # Build structured protobuf response
            journeys = []
            for journey_data in result.get("journeys", []):
                # Build legs
                legs = []
                for leg_data in journey_data.get("legs", []):
                    if leg_data["type"] == "walk":
                        walk_leg = routing_pb2.WalkLeg(
                            distance_meters=leg_data["distance_meters"],
                            duration_minutes=leg_data["duration_minutes"],
                            path=[
                                routing_pb2.Coordinate(lon=coord[0], lat=coord[1])
                                for coord in leg_data.get("path", [])
                            ],
                        )
                        legs.append(routing_pb2.Leg(walk=walk_leg))

                    elif leg_data["type"] == "trip":
                        trip_leg = routing_pb2.TripLeg(
                            trip_id=leg_data["trip_id"],
                            mode=leg_data["mode"],
                            route_short_name=leg_data["route_short_name"],
                            headsign=leg_data["headsign"],
                            fare=leg_data["fare"],
                            duration_minutes=leg_data["duration_minutes"],
                            path=[
                                routing_pb2.Coordinate(lon=coord[0], lat=coord[1])
                                for coord in leg_data.get("path", [])
                            ],
                        )
                        # Set 'from' field using getattr to access the reserved keyword field
                        getattr(trip_leg, "from").CopyFrom(
                            routing_pb2.Stop(
                                stop_id=leg_data["from"]["stop_id"],
                                name=leg_data["from"]["name"],
                                coord=routing_pb2.Coordinate(
                                    lon=leg_data["from"]["coord"][0],
                                    lat=leg_data["from"]["coord"][1],
                                ),
                            )
                        )
                        trip_leg.to.CopyFrom(
                            routing_pb2.Stop(
                                stop_id=leg_data["to"]["stop_id"],
                                name=leg_data["to"]["name"],
                                coord=routing_pb2.Coordinate(
                                    lon=leg_data["to"]["coord"][0],
                                    lat=leg_data["to"]["coord"][1],
                                ),
                            )
                        )
                        legs.append(routing_pb2.Leg(trip=trip_leg))

                    elif leg_data["type"] == "transfer":
                        transfer_leg = routing_pb2.TransferLeg(
                            from_trip_id=leg_data["from_trip_id"],
                            to_trip_id=leg_data["to_trip_id"],
                            from_trip_name=leg_data["from_trip_name"],
                            to_trip_name=leg_data["to_trip_name"],
                            walking_distance_meters=leg_data["walking_distance_meters"],
                            duration_minutes=leg_data["duration_minutes"],
                            path=[
                                routing_pb2.Coordinate(lon=coord[0], lat=coord[1])
                                for coord in leg_data.get("path", [])
                            ],
                        )
                        legs.append(routing_pb2.Leg(transfer=transfer_leg))

                # Build journey
                journey = routing_pb2.Journey(
                    id=journey_data["id"],
                    text_summary=journey_data["text_summary"],
                    summary=routing_pb2.JourneySummary(
                        total_time_minutes=journey_data["summary"][
                            "total_time_minutes"
                        ],
                        total_distance_meters=journey_data["summary"][
                            "total_distance_meters"
                        ],
                        walking_distance_meters=journey_data["summary"][
                            "walking_distance_meters"
                        ],
                        transfers=journey_data["summary"]["transfers"],
                        cost=journey_data["summary"]["cost"],
                        modes=journey_data["summary"]["modes"],
                    ),
                    legs=legs,
                )
                journeys.append(journey)

            # Return structured response
            return routing_pb2.RouteResponse(
                num_journeys=result.get("num_journeys", 0),
                journeys=journeys,
                start_trips_found=result.get("start_trips_found", 0),
                end_trips_found=result.get("end_trips_found", 0),
                total_routes_found=result.get("total_routes_found", 0),
                error=result.get("error", ""),
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
