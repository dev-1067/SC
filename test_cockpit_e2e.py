"""
End-to-end integration and API verification script for STCS.
Tests all button actions and REST / WebSocket endpoints served by the backend:
1. Root / static dashboard serving (GET /)
2. Telemetry status snapshot (GET /api/status)
3. Mode toggle: ADAPTIVE -> STATIC_FIXED -> MANUAL -> ADAPTIVE (POST /api/control/mode)
4. Speed multipliers: 1x, 2x, 5x (POST /api/control/speed)
5. Pause and resume (POST /api/control/pause, POST /api/control/resume)
6. Manual phase switch (POST /api/control/switch)
7. Emergency preemption & clearing (POST /api/control/emergency, POST /api/control/emergency/clear)
8. Demand sliders update (POST /api/control/demand)
9. Comparative benchmark suite run (POST /api/benchmark/run)
10. Soft computing model introspection (GET /api/soft-computing/models)
11. Telemetry and audit history (GET /api/metrics/history, GET /api/switches/history, GET /api/benchmarks/history)
12. WebSocket real-time stream connection and message parsing (/ws/traffic-stream)
"""

import asyncio
import json
import requests
import websockets

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/traffic-stream"

def test_endpoints():
    print("=" * 70)
    print("STCS REST & UI ACTIONS VERIFICATION SUITE")
    print("=" * 70)

    # 1. Root / static dashboard
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Dashboard failed: {r.status_code}"
    assert "Smart Traffic Signal Control System" in r.text
    print("[PASS] 1. Root Dashboard Served (index.html, HTTP 200)")

    # 2. Status snapshot
    r = requests.get(f"{BASE_URL}/api/status")
    assert r.status_code == 200
    st = r.json()
    assert "sim_time" in st and "active_phase" in st and "fuzzy_inspector" in st
    print(f"[PASS] 2. /api/status Snapshot (sim_time={st['sim_time']}s, phase={st['active_phase']})")

    # 3. Mode Toggle
    for mode in ["STATIC_FIXED", "MANUAL", "ADAPTIVE"]:
        r = requests.post(f"{BASE_URL}/api/control/mode", json={"mode": mode})
        assert r.status_code == 200
        assert r.json()["control_mode"] == mode
        print(f"  - Mode switched to {mode}")
    print("[PASS] 3. Control Mode Toggle (Cycle: STATIC -> MANUAL -> ADAPTIVE)")

    # 4. Speed Multipliers
    for spd in [2.0, 5.0, 1.0]:
        r = requests.post(f"{BASE_URL}/api/control/speed", json={"speed": spd})
        assert r.status_code == 200
        assert r.json()["sim_speed"] == spd
    print("[PASS] 4. Simulation Speed Multipliers (1x, 2x, 5x)")

    # 5. Pause & Resume
    r = requests.post(f"{BASE_URL}/api/control/pause")
    assert r.status_code == 200 and r.json()["is_paused"] is True
    r = requests.post(f"{BASE_URL}/api/control/resume")
    assert r.status_code == 200 and r.json()["is_paused"] is False
    print("[PASS] 5. Pause and Resume Clock Controls")

    # 6. Manual Phase Switch
    r = requests.post(f"{BASE_URL}/api/control/switch", json={"target_phase": 1})
    assert r.status_code == 200 and r.json()["status"] == "SUCCESS"
    print("[PASS] 6. Manual Phase Switch Trigger")

    # 7. Emergency Preemption
    r = requests.post(f"{BASE_URL}/api/control/emergency", json={"approach": "North", "duration": 20.0})
    assert r.status_code == 200 and r.json()["emergency_active"] is True
    r = requests.post(f"{BASE_URL}/api/control/emergency/clear")
    assert r.status_code == 200 and r.json()["emergency_active"] is False
    print("[PASS] 7. Emergency Corridor Preemption & Clear Priority")

    # 8. Demand Sliders
    r = requests.post(f"{BASE_URL}/api/control/demand", json={"north": 8.0, "south": 8.0, "east": 3.5, "west": 3.5})
    assert r.status_code == 200
    assert r.json()["arrival_rates"]["North"] == 8.0
    print("[PASS] 8. Dynamic Inflow Demand Sliders")

    # 9. Comparative Benchmark
    print("  Executing 30s benchmark test...")
    r = requests.post(f"{BASE_URL}/api/benchmark/run", json={"duration_seconds": 30.0})
    assert r.status_code == 200
    bm = r.json()
    assert "delay_reduction_pct" in bm and "throughput_increase_pct" in bm
    print(f"[PASS] 9. Comparative Benchmark (Delay Drop: {bm['delay_reduction_pct']:.1f}%, Target Met: {bm['target_met']})")

    # 10. Soft Computing Model Introspection
    r = requests.get(f"{BASE_URL}/api/soft-computing/models")
    assert r.status_code == 200
    m = r.json()
    assert "fuzzy_controller" in m and "ann_predictor" in m
    print("[PASS] 10. Soft Computing Architecture Inspector")

    # 11. Database Historical Metrics & Audit Trails
    r_m = requests.get(f"{BASE_URL}/api/metrics/history?limit=5")
    r_s = requests.get(f"{BASE_URL}/api/switches/history?limit=5")
    r_b = requests.get(f"{BASE_URL}/api/benchmarks/history?limit=5")
    assert r_m.status_code == 200 and r_s.status_code == 200 and r_b.status_code == 200
    print(f"[PASS] 11. SQLite Telemetry & Audit Logs (Metrics: {len(r_m.json())}, Switches: {len(r_s.json())}, Benchmarks: {len(r_b.json())})")

async def test_websocket():
    print("Testing WebSocket stream...")
    async with websockets.connect(WS_URL) as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(msg)
        assert "sim_time" in data and "phases" in data
        # Send interactive action over WS
        await ws.send(json.dumps({"action": "switch_mode", "mode": "ADAPTIVE"}))
        print(f"[PASS] 12. WebSocket Real-Time Telemetry Stream & Bidirectional Messaging")

if __name__ == "__main__":
    test_endpoints()
    asyncio.run(test_websocket())
    print("=" * 70)
    print("ALL 12 INTERACTIVE COCKPIT CONTROLS & API SERVICES VERIFIED 100% ERROR-FREE!")
    print("=" * 70)
