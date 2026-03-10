import os
from netrc import netrc, NetrcParseError
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


def load_project_env() -> bool:
    """
    Load <project_root>/.env while preserving non-empty exported variables.

    Empty exported variables (e.g. WANDB_API_KEY="") are treated as unset and
    will be populated from .env.
    """
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"

    loaded = False
    if env_path.exists():
        values = dotenv_values(env_path)
        for key, value in values.items():
            if not key or value is None:
                continue
            current = os.getenv(key)
            if current is None or current == "":
                os.environ[key] = value
                loaded = True

    # Keep legacy behavior as fallback (search from current working directory).
    return load_dotenv(override=False) or loaded


def ensure_wandb_api_key() -> bool:
    """Populate WANDB_API_KEY from common aliases if present."""
    aliases = ("WANDB_API_KEY", "WANDB_KEY", "WANDB_TOKEN")
    for name in aliases:
        value = os.getenv(name)
        if value and value.strip():
            os.environ["WANDB_API_KEY"] = value.strip()
            return True
    return False


def has_wandb_netrc_credentials() -> bool:
    """Check whether wandb credentials are already stored via `wandb login`."""
    try:
        creds = netrc().authenticators("api.wandb.ai")
        if creds and creds[2]:
            return True
        creds_alt = netrc().authenticators("wandb.ai")
        return bool(creds_alt and creds_alt[2])
    except (FileNotFoundError, NetrcParseError):
        return False


def login_wandb_from_env(wandb_module) -> None:
    """
    Perform non-interactive W&B auth.

    Priority:
    1) Env key (WANDB_API_KEY, WANDB_KEY, WANDB_TOKEN)
    2) Existing `wandb login` credentials in ~/.netrc
    """
    if ensure_wandb_api_key():
        key = os.environ["WANDB_API_KEY"].strip()
        # W&B personal API keys are expected to be 40 characters.
        if len(key) != 40:
            raise RuntimeError(
                f"W&B API key is invalid (length={len(key)}; expected 40). "
                "Update WANDB_API_KEY in /app/.env with your real key."
            )
        wandb_module.login(key=key, relogin=False)
        return

    if has_wandb_netrc_credentials():
        return

    raise RuntimeError(
        "W&B is enabled but no credentials were found. Set WANDB_API_KEY in /app/.env "
        "(or WANDB_KEY/WANDB_TOKEN), or run `wandb login`, or disable --wandb."
    )
