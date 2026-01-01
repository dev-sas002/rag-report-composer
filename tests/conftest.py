"""
Shared test configuration.

The project logs through structlog. Whether `caplog` can see those events
depends on structlog being configured to emit through the standard library,
and that only happened as a side effect of importing `src.api.main`, which
calls `setup_logging()` at import time. So the log assertions in
`test_vector_store.py` and `test_cost_tracker.py` passed or failed according
to whether the API tests had been collected first — they failed when either
file was run on its own.

Configuring it here once makes those assertions mean the same thing however
the suite is invoked, and keeps test logs out of the repository's `logs/`
directory.
"""

import socket

import pytest
import structlog


@pytest.fixture(autouse=True, scope="session")
def route_structlog_through_stdlib_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.KeyValueRenderer(key_order=["event"]),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Module-level loggers are created at import time; without this a
        # logger bound under an earlier configuration keeps using it.
        cache_logger_on_first_use=False,
    )


class OutboundNetworkBlockedError(RuntimeError):
    """Raised when a test tries to open a connection to the outside world."""


@pytest.fixture(autouse=True, scope="session")
def block_outbound_network():
    """
    Make "no test hits a real API" enforceable rather than a convention.

    Every LLM, embedding and HTTP call in the suite is faked. This closes the
    gap between that being true today and staying true: a test that reaches for
    api.openai.com fails loudly instead of quietly spending tokens. Loopback
    stays open so FastAPI's TestClient and anything socket-based in-process
    keep working.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def allowed(address) -> bool:
        if not isinstance(address, tuple) or not address:
            return True  # unix sockets and the like are local by definition
        host = str(address[0])
        return host in {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}

    def guard(original):
        def wrapper(self, address, *args, **kwargs):
            if not allowed(address):
                raise OutboundNetworkBlockedError(
                    f"tests must not make real network calls (attempted {address!r})"
                )
            return original(self, address, *args, **kwargs)

        return wrapper

    socket.socket.connect = guard(real_connect)
    socket.socket.connect_ex = guard(real_connect_ex)
    try:
        yield
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
