from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import sys
import os

# Add the parent directory to the path to import routing module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from routing_module.routing import find_journeys, load_traffic, explore_trips
from routing_module.network import create_network

app = FastAPI(title="Transit Routing API", version="1.0.0")


class HealthResponse(BaseModel):
    status: str
    message: str


class RouteRequest(BaseModel):
    graph: Dict[str, Dict[str, int]]
    pathways_dict: Dict[int, Dict[str, Any]]
    start_trips: Dict[str, Dict[str, Any]]
    goal_trips: Dict[str, Dict[str, Any]]
    max_transfers: int = 2


class RouteResponse(BaseModel):
    num_journeys: int
    journeys: List[Dict[str, Any]]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Transit Routing API is running"
    }


@app.post("/route", response_model=RouteResponse)
async def find_routes(request: RouteRequest):
    """
    Find transit routes between start and goal trips.
    
    Args:
        request: RouteRequest containing graph, pathways, start/goal trips, and max transfers
        
    Returns:
        RouteResponse with list of journeys and their costs
    """
    try:
        # Load traffic data once - uses default path in routing_module/data/
        traffic = load_traffic()
        
        # Find journeys
        journeys = find_journeys(
            graph=request.graph,
            pathways_dict=request.pathways_dict,
            start_trips=request.start_trips,
            goal_trips=request.goal_trips,
            max_transfers=request.max_transfers,
            traffic=traffic
        )
        
        # Format results
        formatted_journeys = [
            {
                "path": path,
                "costs": costs
            }
            for path, costs in journeys
        ]
        
        return {
            "num_journeys": len(journeys),
            "journeys": formatted_journeys
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
