"""
Tests for the guard that stops the suite making real network calls.

`tests/conftest.py` claims "no test hits a real API". That claim is worth
exactly as much as the evidence for it, and until these tests existed there was
none: the guard was installed, never exercised, and would have gone on passing
the suite if a refactor had silently stopped it working.

So each one *attempts* an outbound connection and asserts it is refused. They
are the only tests in the repository that try to reach the network, and they
pass precisely because they cannot.
"""

import socket

import httpx
import pytest

from tests.conftest import OutboundNetworkBlockedError

# Documentation-reserved address (RFC 5737). It routes nowhere, so even with
# the guard removed these tests would fail on a timeout rather than emit real
# traffic to somebody's server.
UNROUTABLE = ("203.0.113.1", 443)


class TestTheGuardRefusesOutboundConnections:
    def test_a_raw_socket_connect_is_refused(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OutboundNetworkBlockedError):
                sock.connect(UNROUTABLE)
        finally:
            sock.close()

    def test_connect_ex_is_refused_too(self):
        """
        `connect_ex` returns an error code instead of raising, so a library
        using it would have slipped past a guard that only covered `connect`.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OutboundNetworkBlockedError):
                sock.connect_ex(UNROUTABLE)
        finally:
            sock.close()

    def test_create_connection_is_refused(self):
        """The helper every stdlib-based HTTP client reaches for."""
        with pytest.raises(OutboundNetworkBlockedError):
            socket.create_connection(UNROUTABLE, timeout=1)

    def test_an_https_client_is_refused(self):
        """
        httpx is what the OpenAI SDK uses. This is the call that would actually
        spend money, so it is the one that has to be proven blocked.
        """
        with pytest.raises(Exception) as caught:
            httpx.get("https://203.0.113.1/v1/models", timeout=1)

        # httpx wraps the failure in a ConnectError; the guard's message is
        # what has to survive, otherwise a real timeout would look the same.
        assert "tests must not make real network calls" in str(caught.value)

    def test_a_hostname_resolving_off_box_is_refused(self):
        """
        Guarding on the address passed to `connect` covers hostnames too: the
        resolver runs first and hands `connect` a public IP.
        """
        with pytest.raises(Exception) as caught:
            socket.create_connection(("api.openai.com", 443), timeout=2)

        assert "tests must not make real network calls" in str(caught.value)


class TestTheGuardStillAllowsLoopback:
    """
    A guard that blocked everything would break FastAPI's TestClient and be
    switched off within the week. Local traffic has to keep working.
    """

    def test_a_loopback_connection_is_permitted(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                client.connect(listener.getsockname())
                assert client.getpeername()[0] == "127.0.0.1"
            finally:
                client.close()
        finally:
            listener.close()

    def test_the_api_test_client_still_works(self):
        """The whole API suite depends on this, so assert it directly."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            assert client.get("/").status_code == 200
