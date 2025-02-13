from getpass import getpass
from pydantic import BaseModel, validator
from typing import Optional
from dynaconf import Dynaconf

class Vantage6Config(BaseModel):
    server_url: str
    server_port: int
    server_api: str
    username: str
    password: str
    ci_mode: bool = False
    organization_key: Optional[str] = None  # Optional if you want to use encryption

    @validator('server_url')
    def url_must_be_valid(cls, v):
        if not v.startswith("http"):
            raise ValueError("server_url must start with http")
        return v

# Load settings via Dynaconf from settings files or environment variables.
settings = Dynaconf(
    settings_files=['settings/infra.toml', 'settings/.secrets.toml'],
    envvar_prefix="V6"
)

# Prepare configuration data. In CI mode, the password should come from settings;
# otherwise, we prompt for it.
config = Vantage6Config(
    server_url = settings.get("SERVER_URL", "http://localhost"),
    server_port = settings.get("SERVER_PORT", 5070),
    server_api = settings.get("SERVER_API", "/api"),
    username = settings.get("USERNAME", "gamma-user"),
    ci_mode = settings.get("CI_MODE", False),
    password = settings.get("PASSWORD") if settings.get("CI_MODE", False) else getpass("Enter <PASSWORD>: "),
    organization_key = settings.get("ORGANIZATION_KEY")
)
