"""Application configuration using environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_name: str = Field(default="YojanaSetu Backend")
    app_env: str = Field(default="development")
    app_debug: bool = Field(default=False)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    pinecone_api_key: str = Field(default="")
    twilio_account_sid: str = Field(default="")
    twilio_auth_token: str = Field(default="")
    bhashini_api_key: str = Field(default="")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
