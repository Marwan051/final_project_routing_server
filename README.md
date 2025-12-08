# Transit Routing API

A FastAPI application for finding optimal public transit routes.

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
│   └── main.py            # FastAPI application
├── example_usage.py       # Example script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Create Network and Find Routes (Script)

```bash
python example_usage.py
```

### 2. Run the API Server

```bash
cd app
python main.py
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

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

## API Endpoints

### Health Check

- **GET** `/health`
- Returns the status of the API

### Find Routes

- **POST** `/route`
- Request body: JSON with graph, pathways, start/goal trips, and max transfers
- Returns: List of possible journeys with costs (money, time, walking distance)

## API Documentation

Once running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Data Files

All data files are located in `routing_module/data/`:

- `labeled.osm` - OSM network file
- `prefixtimes.json` - Traffic timing data
- `trip_pathways.csv` - Trip pathways
- `gtfsAlex/` - Directory with GTFS files (stops.txt, routes.txt, trips.txt, stop_times.txt, shapes.txt)
- `utils/trip_distances.csv` - Trip distances
- `utils/trip_price_model.joblib` - Price prediction model

The modules automatically use these default paths, but you can override them if needed.
