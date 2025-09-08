from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


def _to_bool(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_secret: str
    use_testnet: bool = True

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            api_key=os.getenv("BINANCE_API_KEY", ""),
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            use_testnet=_to_bool(os.getenv("BINANCE_USE_TESTNET", None), default=True),
        )


