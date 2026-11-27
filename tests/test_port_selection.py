"""Port handling: the harness must not die because something owns 8000."""

from __future__ import annotations

import socket
import unittest
from pathlib import Path

from bench.common import describe_port_holder, port_is_free
from bench.server import ServerHandle, ServerStartError


def _occupy() -> tuple[socket.socket, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock, sock.getsockname()[1]


class PortSelectionTest(unittest.TestCase):
    def test_auto_advances_to_free_port(self) -> None:
        sock, busy = _occupy()
        try:
            handle = ServerHandle("m", "m", [], Path("/tmp/port-test-a.log"), port=busy)
            self.assertNotEqual(handle.port, busy)
            self.assertTrue(port_is_free(handle.port))
            self.assertEqual(handle.requested_port, busy)
        finally:
            sock.close()

    def test_strict_port_fails_with_holder(self) -> None:
        sock, busy = _occupy()
        try:
            handle = ServerHandle(
                "m", "m", [], Path("/tmp/port-test-b.log"), port=busy, auto_port=False
            )
            self.assertEqual(handle.port, busy)
            with self.assertRaises(ServerStartError) as ctx:
                handle.start()
            message = str(ctx.exception)
            self.assertIn("already in use", message)
            self.assertIn("--strict-port", message)
            # start() must fail before spawning anything
            self.assertIsNone(handle.proc)
        finally:
            sock.close()

    def test_holder_is_described(self) -> None:
        sock, busy = _occupy()
        try:
            holder = describe_port_holder(busy)
            self.assertTrue(holder)
            self.assertNotEqual(holder, "")
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
