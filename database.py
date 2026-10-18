"""
Smart Traffic Signal Control System (STCS)
Person 3 Deliverable: Database & Telemetry Storage (database.py)
Covers Data Persistence, Metric Logging, Audit Trails & Benchmarking:
- SQLite Database via SQLAlchemy ORM
- MetricLog Table: Periodic high-frequency intersection performance logs
- SignalSwitch Table: Signal phase transition audit trails with FLC rationales
- BenchmarkRecord Table: Comparative benchmark experiment results
- Thread-safe repository operations
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Text,
    DateTime,
    Boolean,
    desc
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

DB_FILE = os.path.join(os.path.dirname(__file__), "traffic_telemetry.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)
session_factory = sessionmaker(bind=engine, autoflush=False)
SessionLocal = scoped_session(session_factory)
Base = declarative_base()


# ==============================================================================
# 1. ORM DATA MODELS
# ==============================================================================

class MetricLog(Base):
    """Stores high-frequency intersection state snapshots for charting & analysis."""
    __tablename__ = "metric_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    sim_time = Column(Float, nullable=False)
    control_mode = Column(String(32), nullable=False)
    active_phase = Column(Integer, nullable=False)
    phase_name = Column(String(64), nullable=False)
    signal_ns = Column(String(16), nullable=False)
    signal_ew = Column(String(16), nullable=False)
    
    # Lane metrics
    queue_north = Column(Integer, default=0)
    queue_south = Column(Integer, default=0)
    queue_east = Column(Integer, default=0)
    queue_west = Column(Integer, default=0)
    total_queued = Column(Integer, default=0)
    
    wait_ns = Column(Float, default=0.0)
    wait_ew = Column(Float, default=0.0)
    
    # Network performance
    total_throughput = Column(Integer, default=0)
    average_delay = Column(Float, default=0.0)
    active_vehicles = Column(Integer, default=0)
    
    # Soft Computing predictions
    predicted_queue_ns = Column(Float, default=0.0)
    predicted_queue_ew = Column(Float, default=0.0)
    surge_factor_ns = Column(Float, default=0.0)
    surge_factor_ew = Column(Float, default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.strftime("%H:%M:%S") if self.created_at else "",
            "sim_time": round(self.sim_time, 1),
            "control_mode": self.control_mode,
            "active_phase": self.active_phase,
            "phase_name": self.phase_name,
            "signal_ns": self.signal_ns,
            "signal_ew": self.signal_ew,
            "queue_north": self.queue_north,
            "queue_south": self.queue_south,
            "queue_east": self.queue_east,
            "queue_west": self.queue_west,
            "total_queued": self.total_queued,
            "wait_ns": round(self.wait_ns, 1),
            "wait_ew": round(self.wait_ew, 1),
            "total_throughput": self.total_throughput,
            "average_delay": round(self.average_delay, 1),
            "active_vehicles": self.active_vehicles,
            "predicted_queue_ns": round(self.predicted_queue_ns, 1),
            "predicted_queue_ew": round(self.predicted_queue_ew, 1),
            "surge_factor_ns": round(self.surge_factor_ns, 3),
            "surge_factor_ew": round(self.surge_factor_ew, 3)
        }


class SignalSwitch(Base):
    """Audit log of signal transitions with Soft Computing decision rationales."""
    __tablename__ = "signal_switches"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    sim_time = Column(Float, nullable=False)
    control_mode = Column(String(32), nullable=False)
    from_phase = Column(Integer, nullable=False)
    to_phase = Column(Integer, nullable=False)
    from_name = Column(String(64), nullable=False)
    to_name = Column(String(64), nullable=False)
    allocated_green = Column(Float, nullable=False)
    trigger_reason = Column(String(128), default="NORMAL_TRANSITION")
    decision_rationale = Column(Text, default="")
    active_rules_json = Column(Text, default="[]")

    def to_dict(self) -> Dict[str, Any]:
        rules = []
        if self.active_rules_json:
            try:
                rules = json.loads(self.active_rules_json)
            except Exception:
                rules = []
        return {
            "id": self.id,
            "created_at": self.created_at.strftime("%H:%M:%S") if self.created_at else "",
            "sim_time": round(self.sim_time, 1),
            "control_mode": self.control_mode,
            "from_phase": self.from_phase,
            "to_phase": self.to_phase,
            "from_name": self.from_name,
            "to_name": self.to_name,
            "allocated_green": round(self.allocated_green, 1),
            "trigger_reason": self.trigger_reason,
            "decision_rationale": self.decision_rationale,
            "active_rules": rules
        }


class BenchmarkRecord(Base):
    """Saves comparative experiment results (Static Fixed-Time vs Adaptive Soft Computing)."""
    __tablename__ = "benchmark_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    benchmark_name = Column(String(128), default="Comparative Surge Benchmark")
    duration_seconds = Column(Float, nullable=False)
    
    static_avg_delay = Column(Float, nullable=False)
    adaptive_avg_delay = Column(Float, nullable=False)
    delay_reduction_pct = Column(Float, nullable=False)
    
    static_throughput = Column(Integer, nullable=False)
    adaptive_throughput = Column(Integer, nullable=False)
    throughput_increase_pct = Column(Float, default=0.0)
    
    static_queued = Column(Integer, default=0)
    adaptive_queued = Column(Integer, default=0)
    queue_reduction_pct = Column(Float, default=0.0)
    
    target_met = Column(Boolean, default=True)
    notes = Column(Text, default="")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else "",
            "benchmark_name": self.benchmark_name,
            "duration_seconds": round(self.duration_seconds, 1),
            "static_avg_delay": round(self.static_avg_delay, 2),
            "adaptive_avg_delay": round(self.adaptive_avg_delay, 2),
            "delay_reduction_pct": round(self.delay_reduction_pct, 2),
            "static_throughput": self.static_throughput,
            "adaptive_throughput": self.adaptive_throughput,
            "throughput_increase_pct": round(self.throughput_increase_pct, 2),
            "static_queued": self.static_queued,
            "adaptive_queued": self.adaptive_queued,
            "queue_reduction_pct": round(self.queue_reduction_pct, 2),
            "target_met": self.target_met,
            "notes": self.notes
        }


# ==============================================================================
# 2. DATABASE REPOSITORY / CONTROLLER
# ==============================================================================

class DatabaseManager:
    """Manages database operations with connection scoping and error safety."""

    def __init__(self):
        self.init_db()

    def init_db(self):
        """Creates tables if they don't already exist."""
        Base.metadata.create_all(bind=engine)

    def log_metric(self, state: Dict[str, Any]) -> Optional[int]:
        """Logs a single simulation telemetry snapshot."""
        session = SessionLocal()
        try:
            lanes = state.get("lanes", {})
            phases = state.get("phases", {})
            ann = state.get("ann_forecast", {})
            
            p0 = phases.get(0, phases.get("0", {}))
            p1 = phases.get(1, phases.get("1", {}))
            ann0 = ann.get(0, ann.get("0", {}))
            ann1 = ann.get(1, ann.get("1", {}))

            log = MetricLog(
                sim_time=float(state.get("sim_time", 0.0)),
                control_mode=str(state.get("control_mode", "ADAPTIVE")),
                active_phase=int(state.get("active_phase", 0)),
                phase_name=str(state.get("phase_name", "North-South")),
                signal_ns=str(state.get("signal_states", {}).get("NS", "GREEN")),
                signal_ew=str(state.get("signal_states", {}).get("EW", "RED")),
                
                queue_north=int(lanes.get("North", {}).get("queue_length", 0)),
                queue_south=int(lanes.get("South", {}).get("queue_length", 0)),
                queue_east=int(lanes.get("East", {}).get("queue_length", 0)),
                queue_west=int(lanes.get("West", {}).get("queue_length", 0)),
                total_queued=int(state.get("total_queued", 0)),
                
                wait_ns=float(p0.get("wait_time", 0.0)),
                wait_ew=float(p1.get("wait_time", 0.0)),
                
                total_throughput=int(state.get("total_throughput", 0)),
                average_delay=float(state.get("average_delay_seconds", 0.0)),
                active_vehicles=int(state.get("active_vehicle_count", 0)),
                
                predicted_queue_ns=float(ann0.get("predicted_queue_10min", p0.get("queue_length", 0.0))),
                predicted_queue_ew=float(ann1.get("predicted_queue_10min", p1.get("queue_length", 0.0))),
                surge_factor_ns=float(ann0.get("surge_factor", 0.0)),
                surge_factor_ew=float(ann1.get("surge_factor", 0.0))
            )
            session.add(log)
            session.commit()
            return log.id
        except Exception as e:
            session.rollback()
            return None
        finally:
            SessionLocal.remove()

    def log_signal_switch(
        self,
        sim_time: float,
        control_mode: str,
        from_phase: int,
        to_phase: int,
        from_name: str,
        to_name: str,
        allocated_green: float,
        trigger_reason: str = "NORMAL_TRANSITION",
        decision_rationale: str = "",
        active_rules: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[int]:
        """Logs a signal switching event with audit rationale."""
        session = SessionLocal()
        try:
            record = SignalSwitch(
                sim_time=sim_time,
                control_mode=control_mode,
                from_phase=from_phase,
                to_phase=to_phase,
                from_name=from_name,
                to_name=to_name,
                allocated_green=allocated_green,
                trigger_reason=trigger_reason,
                decision_rationale=decision_rationale,
                active_rules_json=json.dumps(active_rules or [])
            )
            session.add(record)
            session.commit()
            return record.id
        except Exception:
            session.rollback()
            return None
        finally:
            SessionLocal.remove()

    def save_benchmark(self, bench_data: Dict[str, Any]) -> Optional[int]:
        """Persists a comparative benchmark run."""
        session = SessionLocal()
        try:
            stat_d = float(bench_data.get("static_avg_delay", 0.0))
            adap_d = float(bench_data.get("adaptive_avg_delay", 0.0))
            red_pct = float(bench_data.get("delay_reduction_pct", 0.0))
            stat_t = int(bench_data.get("static_throughput", 0))
            adap_t = int(bench_data.get("adaptive_throughput", 0))
            thru_pct = ((adap_t - stat_t) / max(1, stat_t)) * 100.0 if stat_t > 0 else 0.0

            stat_q = int(bench_data.get("static_queued", 0))
            adap_q = int(bench_data.get("adaptive_queued", 0))
            q_pct = ((stat_q - adap_q) / max(1, stat_q)) * 100.0 if stat_q > 0 else 0.0

            record = BenchmarkRecord(
                benchmark_name=bench_data.get("benchmark_name", "Comparative Surge Benchmark"),
                duration_seconds=float(bench_data.get("duration_seconds", 150.0)),
                static_avg_delay=stat_d,
                adaptive_avg_delay=adap_d,
                delay_reduction_pct=red_pct,
                static_throughput=stat_t,
                adaptive_throughput=adap_t,
                throughput_increase_pct=thru_pct,
                static_queued=stat_q,
                adaptive_queued=adap_q,
                queue_reduction_pct=q_pct,
                target_met=bool(red_pct >= 25.0),
                notes=bench_data.get("notes", "Auto-generated comparative benchmark.")
            )
            session.add(record)
            session.commit()
            return record.id
        except Exception:
            session.rollback()
            return None
        finally:
            SessionLocal.remove()

    def get_recent_metrics(self, limit: int = 60) -> List[Dict[str, Any]]:
        """Returns the most recent metric logs sorted chronologically."""
        session = SessionLocal()
        try:
            records = (
                session.query(MetricLog)
                .order_by(desc(MetricLog.id))
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in reversed(records)]
        finally:
            SessionLocal.remove()

    def get_recent_switches(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Returns recent signal switches."""
        session = SessionLocal()
        try:
            records = (
                session.query(SignalSwitch)
                .order_by(desc(SignalSwitch.id))
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in records]
        finally:
            SessionLocal.remove()

    def get_benchmarks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns past benchmark evaluations."""
        session = SessionLocal()
        try:
            records = (
                session.query(BenchmarkRecord)
                .order_by(desc(BenchmarkRecord.id))
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in records]
        finally:
            SessionLocal.remove()

    def clear_metrics(self):
        """Clears metric logs for a fresh benchmark session."""
        session = SessionLocal()
        try:
            session.query(MetricLog).delete()
            session.commit()
        except Exception:
            session.rollback()
        finally:
            SessionLocal.remove()


# Global Singleton Database Manager
_GLOBAL_DB: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Returns singleton instance of DatabaseManager."""
    global _GLOBAL_DB
    if _GLOBAL_DB is None:
        _GLOBAL_DB = DatabaseManager()
    return _GLOBAL_DB


if __name__ == "__main__":
    db = get_db()
    print(f"Database initialized successfully at: {DB_FILE}")
    sample_state = {
        "sim_time": 1.0,
        "control_mode": "ADAPTIVE",
        "active_phase": 0,
        "phase_name": "North-South",
        "signal_states": {"NS": "GREEN", "EW": "RED"},
        "lanes": {
            "North": {"queue_length": 2, "wait_time": 10.0},
            "South": {"queue_length": 1, "wait_time": 5.0},
            "East": {"queue_length": 3, "wait_time": 25.0},
            "West": {"queue_length": 4, "wait_time": 30.0}
        },
        "phases": {
            0: {"queue_length": 3, "wait_time": 10.0, "name": "North-South"},
            1: {"queue_length": 7, "wait_time": 30.0, "name": "East-West"}
        },
        "ann_forecast": {
            "0": {"predicted_queue_10min": 14.5, "surge_factor": 0.25},
            "1": {"predicted_queue_10min": 28.0, "surge_factor": 0.65}
        },
        "total_queued": 10,
        "total_throughput": 4,
        "average_delay_seconds": 6.8,
        "active_vehicle_count": 12
    }
    log_id = db.log_metric(sample_state)
    print(f"Logged sample metric with ID: {log_id}")
    metrics = db.get_recent_metrics(limit=5)
    print(f"Retrieved {len(metrics)} recent metric records from SQLite database.")
