"""Application entrypoint for YojanaSetu backend."""

from fastapi import FastAPI

from app.api.routes.navigation import router as navigation_router
from app.core.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    app = FastAPI(title=settings.app_name, debug=settings.app_debug)
    app.include_router(navigation_router)

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Return service health status for infrastructure probes."""

        return {"status": "ok"}

    return app


app = create_app()
