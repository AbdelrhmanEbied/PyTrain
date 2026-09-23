from api.routes.config import router as config_router
from api.routes.data_versions import router as data_versions_router
from api.routes.datasets import router as datasets_router
from api.routes.models import router as models_router
from api.routes.monitoring import router as monitoring_router
from api.routes.runs import router as runs_router
from api.routes.training import router as training_router

__all__ = [
    "config_router",
    "data_versions_router",
    "datasets_router",
    "models_router",
    "monitoring_router",
    "runs_router",
    "training_router",
]
