import asyncio

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.database import Base
from app.models.alerts import AlertEvent
from app.models.logs import SystemLog
from app.models.records import OwnerGestureRecord  # noqa: F401
from app.models.user import User  # noqa: F401
from app.routers.monitor import router as monitor_router
from app.services.alert_agent import EVENT_TYPES, alert_agent


def _memory_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def test_monitoring_defaults_do_not_send_external_notifications():
    defaults = Settings(_env_file=None)
    assert defaults.alert_sse_enabled is True
    assert defaults.alert_email_enabled is False
    assert defaults.alert_webhook_enabled is False


def test_monitoring_public_api_surface_is_registered():
    paths = {route.path for route in monitor_router.routes if hasattr(route, "path")}
    required = {
        "/api/monitor/logs/stats", "/api/monitor/logs/stream",
        "/api/monitor/alerts/analytics", "/api/monitor/alerts/timeline",
        "/api/monitor/alerts/{alert_id}/replay", "/api/monitor/agent/briefing",
        "/api/monitor/assistant", "/api/monitor/notifications/test",
    }
    assert required <= paths


def test_alert_agent_persists_stats_and_replay_without_llm_network():
    db = _memory_session()
    try:
        alert = asyncio.run(alert_agent.trigger_alert(
            db,
            "lpr_consecutive_failure",
            "critical",
            {"count": 5, "module": "lpr"},
            force_template=True,
        ))
        assert alert.id is not None
        assert db.query(AlertEvent).count() == 1
        assert db.query(SystemLog).count() >= 1

        stats = alert_agent.get_stats(db)
        assert stats["total"] == 1
        assert stats["open"] == 1
        assert stats["by_level"]["critical"] == 1

        replay = alert_agent.get_event_replay(db, alert.id)
        assert replay["alert"]["id"] == alert.id
        assert replay["cause_analysis"]["primary_cause"]
        assert "timeline_events" in replay
    finally:
        db.close()


def test_required_anomaly_types_are_available():
    required = {
        "lpr_consecutive_failure", "gesture_low_confidence", "llm_api_timeout",
        "llm_token_exhausted", "unauthorized_access", "database_connection_error",
        "model_load_failure",
    }
    assert required <= EVENT_TYPES.keys()
