"""Request and response models for PMJAY navigation."""

from enum import Enum
from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, Field


class SupportedLanguage(str, Enum):
    """Languages currently supported by MVP scope."""

    HINDI = "hi"
    MARATHI = "mr"


class SupportedState(str, Enum):
    """States currently supported by MVP scope."""

    MAHARASHTRA = "maharashtra"


class SupportedScheme(str, Enum):
    """Schemes currently supported by MVP scope."""

    PMJAY = "pmjay"


class NavigationRequest(BaseModel):
    """Input payload for healthcare scheme navigation."""

    query: str = Field(min_length=3, max_length=600)
    language: SupportedLanguage
    state: SupportedState = SupportedState.MAHARASHTRA
    scheme: SupportedScheme = SupportedScheme.PMJAY


class SuccessResponse(BaseModel):
    """Successful RAG navigation response with mandatory citation."""

    answer_i18n_key: str
    answer_params: Dict[str, str] = Field(default_factory=dict)
    fallback: Literal[False] = False
    notification_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class FallbackResponse(BaseModel):
    """Graceful fallback response when confidence is below threshold."""

    answer: Optional[str] = None
    fallback: Literal[True] = True
    helpline: Literal["14555"] = "14555"


NavigationResponse = Union[SuccessResponse, FallbackResponse]
