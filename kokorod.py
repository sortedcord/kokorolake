#!/usr/bin/env python3
"""Streaming Kokoro TTS daemon (drop-in for ttsd: same socket protocol).

Socket $XDG_RUNTIME_DIR/ttsd.sock; first line is a command, the rest (until EOF) is text:
  SAY   <text...>   queue text; audio streams chunk by chunk as it is synthesized
  STOP              cut off current speech and clear the queue
The Quickshell island is told what is being said (quickshell ipc call voice say/sayEnd).
"""
import os

# before numpy: single-threaded BLAS (see engine.py; spinning BLAS threads cost latency and power)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import queue
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import SR, Engine  # noqa: E402

RUNTIME = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
_island_pool = ThreadPoolExecutor(max_workers=1)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def island(fn, *args):
    def call():
        try:
            subprocess.run(["quickshell", "ipc", "call", "voice", fn, *args], capture_output=True, timeout=3)
        except Exception:
            pass
    _island_pool.submit(call)


class Speaker:
    def __init__(self, engine: Engine):
        self.e = engine
        self.q: queue.Queue[str] = queue.Queue()
        self.gen = 0  # bumped by STOP
        self.player = None
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text):
        self.q.put(text)

    def stop(self):
        with self.lock:
            self.gen += 1
            while not self.q.empty():
                self.q.get_nowait()
            if self.player and self.player.poll() is None:
                self.player.kill()
        island("sayEnd")

    def _worker(self):
        while True:
            text = self.q.get()
            gen = self.gen
            try:
                self._speak(text, gen)
            except Exception as ex:
                log("speak failed:", ex)
            if gen == self.gen and self.q.empty():
                island("sayEnd")

    def _speak(self, text, gen):
        # The island is updated when the first audio is ready: its redraw (and the
        # quickshell process spawn) would otherwise compete with synthesis for GPU/CPU.
        t0 = time.perf_counter()
        player = None
        try:
            for i, (audio, T) in enumerate(self.e.stream(text)):
                if gen != self.gen:
                    return
                if player is None:
                    with self.lock:
                        if gen != self.gen:
                            return
                        player = subprocess.Popen(
                            ["pw-play", "--raw", "--format", "s16", "--rate", str(SR), "--channels", "1", "-"],
                            stdin=subprocess.PIPE)
                        self.player = player
                    log(f"first audio after {(time.perf_counter() - t0) * 1e3:.0f} ms")
                    island("say", text)
                pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
                player.stdin.write(pcm)
                player.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass  # killed by STOP
        finally:
            if player:
                try:
                    player.stdin.close()
                except OSError:
                    pass
                player.wait()


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=str(RUNTIME / "ttsd.sock"))
    args = ap.parse_args()

    t = time.perf_counter()
    eng = Engine()
    log(f"engine ready in {time.perf_counter() - t:.1f}s")
    sp = Speaker(eng)

    class H(socketserver.StreamRequestHandler):
        def handle(self):
            data = self.rfile.read().decode("utf-8", errors="replace")
            cmd, _, text = data.partition("\n")
            cmd = cmd.strip().upper()
            if cmd == "STOP":
                sp.stop()
            elif cmd == "SAY" and text.strip():
                sp.say(text.strip())

    sock = Path(args.socket)
    sock.unlink(missing_ok=True)
    srv = Server(str(sock), H)
    sock.chmod(0o600)

    def bye(*_):
        sp.stop()
        sock.unlink(missing_ok=True)
        os._exit(0)

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    log(f"kokorod listening on {sock}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
