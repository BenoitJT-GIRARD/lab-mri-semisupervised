"""The unit tier, and what it refuses.

A unit test touches no input or output outside ``tmp_path``, no network, no container, no
training run, and reads no artefact of this repository from the place where it is published.
One function, one class, one invariant.

The rule is enforced here rather than trusted: an outbound connection is replaced for the
duration of each test, so a test that reaches a service fails with the reason instead of
passing slowly on the machine that happens to have it running. A test caught by this guard is
not broken — it is in the wrong directory.

**Loopback is allowed, and that is not a loophole.** On Windows the asyncio event loop builds
itself a socket pair, so `asyncio.run` in a pure computation would otherwise be refused as
network use. What the tier forbids is a dependency outside the process; a socket a test opens
to a service, even on this machine, is an integration test and belongs one directory up.
"""

from __future__ import annotations

import socket

import pytest

LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}

REASON = (
    "a unit test opened a connection to {peer}. The unit tier forbids the network: move the "
    "test to tests/integration/ and mark it, or replace the call with a double."
)


def _host_of(address) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create = socket.create_connection

    def guard(call):
        def wrapper(self_or_address, *args, **kwargs):
            address = args[0] if args else self_or_address
            host = _host_of(address)
            if host not in LOOPBACK:
                raise RuntimeError(REASON.format(peer=host))
            return call(self_or_address, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(socket.socket, "connect", guard(real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(real_connect_ex))
    monkeypatch.setattr(socket, "create_connection", guard(real_create))
