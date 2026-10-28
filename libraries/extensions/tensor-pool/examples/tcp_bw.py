#!/usr/bin/env python3
"""Measure raw TCP throughput. Run the server on the Ascend host, the client on the laptop.

This sends incompressible bytes over one TCP connection. It does not go through
dora or tensor-pool. Compare the printed MB/s with a tensor-pool frame only
after both numbers exist.
"""

import argparse
import os
import socket
import time

# sender.py: SIZE = 15000 * 512 int64
FRAME_BYTES = 15000 * 512 * 8
CHUNK = 1024 * 1024


def _report(label: str, nbytes: int, seconds: float) -> None:
    mb = nbytes / 1e6
    mib = nbytes / (1024 * 1024)
    print(
        f"{label}: {nbytes} bytes in {seconds:.3f}s "
        f"= {mb / seconds:.2f} MB/s ({mib / seconds:.2f} MiB/s)",
        flush=True,
    )


def serve(bind: str, port: int) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((bind, port))
    listener.listen(1)
    print(f"listening on {bind}:{port}", flush=True)
    while True:
        conn, addr = listener.accept()
        print(f"accepted {addr[0]}:{addr[1]}", flush=True)
        try:
            received = 0
            started = None
            while True:
                data = conn.recv(CHUNK)
                if not data:
                    break
                if started is None:
                    started = time.perf_counter()
                received += len(data)
            elapsed = 0.0 if started is None else time.perf_counter() - started
            if received == 0:
                print("received 0 bytes", flush=True)
            else:
                _report("recv", received, elapsed)
        finally:
            conn.close()


def send(host: str, port: int, nbytes: int, rounds: int) -> None:
    payload = os.urandom(CHUNK)
    for i in range(rounds):
        sock = socket.create_connection((host, port))
        try:
            left = nbytes
            started = time.perf_counter()
            while left:
                chunk = payload if left >= len(payload) else payload[:left]
                sock.sendall(chunk)
                left -= len(chunk)
            sock.shutdown(socket.SHUT_WR)
            elapsed = time.perf_counter() - started
        finally:
            sock.close()
        _report(f"send {i + 1}/{rounds}", nbytes, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="TCP throughput between two hosts")
    sub = parser.add_subparsers(dest="mode", required=True)

    server = sub.add_parser("server")
    server.add_argument("--bind", default="0.0.0.0")
    server.add_argument("--port", type=int, default=17410)

    client = sub.add_parser("client")
    client.add_argument("--host", required=True)
    client.add_argument("--port", type=int, default=17410)
    client.add_argument("--bytes", type=int, default=FRAME_BYTES)
    client.add_argument("--rounds", type=int, default=3)

    args = parser.parse_args()
    if args.mode == "server":
        serve(args.bind, args.port)
    else:
        send(args.host, args.port, args.bytes, args.rounds)


if __name__ == "__main__":
    main()
