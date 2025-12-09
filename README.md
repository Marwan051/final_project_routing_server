# Transit Routing Service

A gRPC-based service for finding optimal public transit routes.

## Project Structure

```
.
├── routing_module/
│   ├── data/              # All data files
│   │   ├── gtfsAlex/      # GTFS data files
│   │   ├── utils/         # Utility files (models, distances)
│   │   ├── labeled.osm    # OSM network file
│   │   ├── prefixtimes.json    # Traffic/timing data
│   │   └── trip_pathways.csv   # Trip transfer pathways
│   ├── routing.py         # Core routing logic and helper functions
│   └── network.py         # Network creation and GTFS loading
├── app/
│   ├── routing.proto      # gRPC service definition
│   ├── grpc_server.py     # gRPC server implementation
│   ├── routing_pb2.py     # Generated protobuf code
│   └── routing_pb2_grpc.py # Generated gRPC code
├── example_usage.py       # Example script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate gRPC Code (First Time Only)

```bash
cd app
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. routing.proto
```

### 2. Run the gRPC Server

```bash
cd app
python grpc_server.py
```

The server will start on port `50051` and load all network data at startup.

### 3. Test with Postman or any gRPC client

Use Postman or any gRPC client to connect to `localhost:50051` with the proto file.

### 4. Create Network and Find Routes (Script)

```bash
python example_usage.py
```

## Using the Modules Directly

### Network Creation

```python
from routing_module.network import create_network

# Create complete network (uses default paths in routing_module/data/)
graph, gtfs_data, trip_graph, pathways_dict = create_network()

# Or specify custom paths
graph, gtfs_data, trip_graph, pathways_dict = create_network(
    osm_file="path/to/your.osm",
    gtfs_path="path/to/gtfs",
    pathways_file="path/to/pathways.csv"
)
```

### Finding Routes

```python
from routing_module.routing import explore_trips, find_journeys
import osmnx as ox

# Find nearest node to coordinates
start_node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)

# Explore reachable trips within walking distance
start_trips = explore_trips(graph, start_node, cutoff=1000)

# Find journeys
journeys = find_journeys(
    trip_graph,
    pathways_dict,
    start_trips,
    target_trips,
    max_transfers=2
)
```

## gRPC Service Methods

### HealthCheck

Request: Empty
Response:

```protobuf
{
  status: "healthy"
  message: "Transit Routing gRPC Service is running"
}
```

### FindRoute

Request:

```protobuf
{
  start_lon: 29.96139328537071
  start_lat: 31.22968895248673
  end_lon: 29.94194179397711
  end_lat: 31.20775934404925
  max_transfers: 2
  walking_cutoff: 1000.0
}
```

Response:

```protobuf
{
  num_journeys: int
  journeys: [
    {
      path: ["trip_id1", "trip_id2", ...]
      costs: {
        money: float
        transport_time: float
        walk: float
      }
    }
  ]
  start_trips_found: int
  end_trips_found: int
}
```

## Using Postman

1. **Import Proto File**: In Postman, create a new gRPC request and import `app/routing.proto`
2. **Server URL**: `localhost:50051`
3. **Select Method**:

   - `routing.RoutingService/HealthCheck` for health check
   - `routing.RoutingService/FindRoute` for finding routes

4. **Example Request for FindRoute**:

```json
{
  "start_lon": 29.96139328537071,
  "start_lat": 31.22968895248673,
  "end_lon": 29.94194179397711,
  "end_lat": 31.20775934404925,
  "max_transfers": 2,
  "walking_cutoff": 1000.0
}
```

## Data Files

All data files are located in `routing_module/data/`:

- `labeled.osm` - OSM network file
- `prefixtimes.json` - Traffic timing data
- `trip_pathways.csv` - Trip pathways
- `gtfsAlex/` - Directory with GTFS files (stops.txt, routes.txt, trips.txt, stop_times.txt, shapes.txt)
- `utils/trip_distances.csv` - Trip distances
- `utils/trip_price_model.joblib` - Price prediction model

The modules automatically use these default paths, but you can override them if needed.
