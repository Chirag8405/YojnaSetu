"""FastAPI route handlers for PMJAY navigation."""

from fastapi import APIRouter, HTTPException

from app.models.navigation import NavigationRequest, NavigationResponse
from app.services.rag_service import CitationMissingError, generate_navigation_response

router = APIRouter(prefix="/api/v1/pmjay", tags=["pmjay"])


@router.post("/navigate", response_model=NavigationResponse)
def navigate_pmjay(request: NavigationRequest) -> NavigationResponse:
    """Handle PMJAY navigation requests for Maharashtra in Hindi/Marathi."""

    try:
        return generate_navigation_response(request)
    except CitationMissingError as exc:
        raise HTTPException(status_code=500, detail="errors.citation_missing") from exc
