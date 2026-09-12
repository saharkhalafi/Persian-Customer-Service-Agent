import os


def skip_live_database() -> bool:
    if not os.getenv("DATABASE_URL"):
        return True
    if os.getenv("CI", "").lower() in {"1", "true", "yes"} and os.getenv("LIVE_DATABASE") != "1":
        return True
    return False
