import time


def current_millis() -> int:
    return int(time.time() * 1000)


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def print_header(message: str) -> None:
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)