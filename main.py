"""
Smart Traffic Signal Control System (STCS)
Person 3 Deliverable: FastAPI Backend Orchestrator & REST/WebSocket Server (main.py)
Covers Full-Stack Infrastructure & API Services:
- FastAPI Application with CORS & Lifespan Event Loop Management
- Real-Time WebSocket Endpoint: /ws/traffic-stream
- REST Control API: Mode Toggle, Manual Phase Switch, Emergency Preemption, Demand Sliders
- Database Telemetry Queries & Audit History
- Live Benchmark Runner & Soft Computing Model Introspection
- Static Dashboard Mounting
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from orchestrator import get_orchestrator
from database import get_db


# ==============================================================================
# 1. REQUEST & RESPONSE PYDANTIC SCHEMAS
# ==============================================================================

class ModeRequest(BaseModel):
    mode: str = Field(..., description="Control mode: ADAPTIVE, STATIC_FIXED, or MANUAL")

class DemandRequest(BaseModel):
    north: float = Field(..., ge=0.0, le=15.0, description="North arrival rate (veh/min)")
    south: float = Field(..., ge=0.0, le=15.0, description="South arrival rate (veh/min)")
    east: float = Field(..., ge=0.0, le=15.0, description="East arrival rate (veh/min)")
    west: float = Field(..., ge=0.0, le=15.0, description="West arrival rate (veh/min)")

class EmergencyRequest(BaseModel):
    approach: str = Field("North", description="Approach corridor for emergency vehicle: North, South, East, West")
    duration: float = Field(20.0, ge=5.0, le=60.0, description="Preemption hold duration in seconds")

class SwitchRequest(BaseModel):
    target_phase: Optional[int] = Field(None, description="Optional target phase ID (0 or 1)")

class SpeedRequest(BaseModel):
    speed: float = Field(..., ge=0.5, le=5.0, description="Simulation speed multiplier (0.5 to 5.0)")

class BenchmarkRequest(BaseModel):
    duration_seconds: float = Field(150.0, ge=30.0, le=600.0, description="Comparative benchmark test duration")


# ==============================================================================
# 2. LIFESPAN CONTEXT (Starting Background Simulation Loop)
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager starting the continuous async simulation tick loop on startup."""
    orch = get_orchestrator()
    # Launch simulation task
    task = asyncio.create_task(orch.run_loop())
    yield
    # Shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ==============================================================================
# 3. FASTAPI APPLICATION SETUP
# ==============================================================================

app = FastAPI(
    title="Smart Traffic Signal Control System (STCS)",
    description="Soft Computing & Hybrid AI Traffic Signal Control REST & WebSocket API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 4. WEBSOCKET REAL-TIME TELEMETRY ENDPOINT
# ==============================================================================

@app.websocket("/ws/traffic-stream")
async def websocket_traffic_stream(websocket: WebSocket):
    """High-frequency real-time WebSocket stream broadcasting intersection state."""
    orch = get_orchestrator()
    await orch.connect_client(websocket)
    try:
        while True:
            # Keep socket open and receive any incoming client messages/pings
            data = await websocket.receive_text()
            # Client can send simple ping or mode switch commands over WS
            try:
                cmd = json.loads(data)
                if cmd.get("action") == "switch_mode":
                    orch.set_control_mode(cmd.get("mode", "ADAPTIVE"))
                elif cmd.get("action") == "manual_switch":
                    orch.manual_phase_switch()
            except Exception:
                pass
    except WebSocketDisconnect:
        orch.disconnect_client(websocket)
    except Exception:
        orch.disconnect_client(websocket)


# Alias endpoint matching project spec (/ws/traffic-live)
@app.websocket("/ws/traffic-live")
async def websocket_traffic_live(websocket: WebSocket):
    """Alias for /ws/traffic-stream."""
    await websocket_traffic_stream(websocket)


# ==============================================================================
# 5. REST API ENDPOINTS
# ==============================================================================

@app.get("/api/status")
def get_system_status():
    """Returns the current state snapshot including signal lights, queues, and soft computing inferences."""
    orch = get_orchestrator()
    return orch.get_telemetry_snapshot()


@app.post("/api/control/mode")
def set_control_mode(payload: ModeRequest):
    """Toggles control mode between ADAPTIVE (FLC + ANN) and STATIC_FIXED."""
    mode = payload.mode.upper()
    if mode not in ("ADAPTIVE", "STATIC_FIXED", "MANUAL"):
        raise HTTPException(status_code=400, detail="Invalid control mode. Choose ADAPTIVE, STATIC_FIXED, or MANUAL.")
    
    orch = get_orchestrator()
    orch.set_control_mode(mode)
    return {"status": "SUCCESS", "control_mode": orch.control_mode}


@app.api_route("/api/control/switch", methods=["GET", "POST"])
def force_manual_switch(payload: Optional[SwitchRequest] = None):
    """Forces an immediate signal phase transition with yellow clearance safety."""
    orch = get_orchestrator()
    target = payload.target_phase if payload else None
    orch.manual_phase_switch(target)
    return {"status": "SUCCESS", "message": "Manual phase switch initiated with yellow clearance."}


@app.post("/api/control/emergency")
def trigger_emergency(payload: EmergencyRequest):
    """Triggers Emergency Vehicle Preemption on specified corridor."""
    orch = get_orchestrator()
    orch.trigger_emergency_preemption(payload.approach, payload.duration)
    return {
        "status": "SUCCESS",
        "emergency_active": True,
        "approach": payload.approach,
        "duration": payload.duration
    }


@app.api_route("/api/control/emergency/clear", methods=["GET", "POST"])
def clear_emergency():
    """Clears emergency vehicle preemption and restores autonomous adaptive control."""
    orch = get_orchestrator()
    orch.clear_emergency()
    return {"status": "SUCCESS", "emergency_active": False}


@app.post("/api/control/demand")
def set_arrival_demands(payload: DemandRequest):
    """Dynamically updates inflow arrival rates per approach (vehicles/min)."""
    orch = get_orchestrator()
    orch.set_arrival_demands(payload.north, payload.south, payload.east, payload.west)
    return {
        "status": "SUCCESS",
        "arrival_rates": orch.sim.arrival_rates
    }


@app.post("/api/control/speed")
def set_sim_speed(payload: SpeedRequest):
    """Sets simulation speed acceleration factor (0.5x to 5.0x)."""
    orch = get_orchestrator()
    orch.set_speed(payload.speed)
    return {"status": "SUCCESS", "sim_speed": orch.sim_speed}


@app.api_route("/api/control/pause", methods=["GET", "POST"])
def pause_sim():
    """Pauses the simulation."""
    orch = get_orchestrator()
    orch.pause()
    return {"status": "SUCCESS", "is_paused": True}


@app.api_route("/api/control/resume", methods=["GET", "POST"])
def resume_sim():
    """Resumes the simulation."""
    orch = get_orchestrator()
    orch.resume()
    return {"status": "SUCCESS", "is_paused": False}


@app.api_route("/api/control/reset", methods=["GET", "POST"])
def reset_sim():
    """Resets the simulation to initial clean state."""
    orch = get_orchestrator()
    orch.reset_simulation()
    return {"status": "SUCCESS", "message": "Simulation reset successfully."}


@app.get("/api/metrics/history")
def get_metrics_history(limit: int = 50):
    """Returns historical performance logs from SQLite for charting."""
    db = get_db()
    return db.get_recent_metrics(limit=limit)


@app.get("/api/switches/history")
def get_switches_history(limit: int = 20):
    """Returns recent signal switching audit trail with decision rationales."""
    db = get_db()
    return db.get_recent_switches(limit=limit)


@app.get("/api/benchmarks/history")
def get_benchmarks_history(limit: int = 10):
    """Returns previous comparative benchmark experiment results."""
    db = get_db()
    return db.get_benchmarks(limit=limit)


@app.post("/api/benchmark/run")
def run_benchmark(payload: Optional[BenchmarkRequest] = None):
    """Executes a comparative benchmark (Static vs. Adaptive) and returns performance report."""
    duration = payload.duration_seconds if payload else 150.0
    orch = get_orchestrator()
    report = orch.run_comparative_benchmark(duration_seconds=duration)
    return report


@app.get("/api/soft-computing/models")
def get_soft_computing_models():
    """Returns structural details of the Soft Computing models for the inspection dashboard."""
    orch = get_orchestrator()
    return orch.get_soft_computing_models_info()


# ==============================================================================
# 6. STATIC DASHBOARD MOUNTING
# ==============================================================================

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_dashboard():
    """Serves the main interactive React / HTML5 Canvas dashboard."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "STCS Dashboard is starting up. Place index.html in static/"}


if __name__ == "__main__":
    import uvicorn
    print("Starting Smart Traffic Signal Control System (STCS) Server...")
    print("Dashboard available at: http://localhost:8000")
    print("WebSocket endpoint:     ws://localhost:8000/ws/traffic-stream")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
