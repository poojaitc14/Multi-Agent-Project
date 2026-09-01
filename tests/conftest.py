"""Shared pytest fixtures for tests/ that need a real, running MCP server
-- moved here (out of test_orchestrator_server.py, where they originally
lived) so any test module in this directory gets the same real server
automatically, without needing to duplicate the fixture or run only
alongside that one specific file. session-scoped + autouse: started once
per test run, reused by every test module that needs it.
"""

import sys
import threading
import time
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp-servers"))
import orchestrator_server as srv  # noqa: E402

TEST_HTTP_PORT = 8091  # distinct from MCP_SERVER_PORT so a real dev server can run alongside tests
TEST_HTTP_URL = f"http://127.0.0.1:{TEST_HTTP_PORT}/mcp"


@pytest.fixture(scope="session", autouse=True)
def _local_infra():
    srv.ensure_local_infra()


@pytest.fixture(scope="session", autouse=True)
def _http_server():
    """Real HTTP transport (Q55's bearer-token gate is an HTTP-headers
    concern -- in-memory transport carries no headers at all, so it can't
    exercise this). Runs the same `srv.mcp` instance the in-memory client
    fixture below also uses; FastMCP allows both simultaneously."""
    thread = threading.Thread(
        target=srv.mcp.run,
        kwargs={"transport": "http", "host": "127.0.0.1", "port": TEST_HTTP_PORT},
        daemon=True,
    )
    thread.start()
    time.sleep(1.5)  # let it finish binding before the first test tries to connect
    yield


def http_client(token: str | None) -> Client:
    # fastmcp.client.auth.BearerAuth does NOT attach a plain Authorization
    # header the way it sounds like it should (verified empirically -- the
    # server-side middleware saw no authorization header at all with it).
    # StreamableHttpTransport's headers= param does work.
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return Client(StreamableHttpTransport(url=TEST_HTTP_URL, headers=headers))
