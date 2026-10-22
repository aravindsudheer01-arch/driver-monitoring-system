"""Nonblocking audible warning for a desk-only driver-monitoring prototype.

Mac: uses the built-in `afplay` command. Windows: uses `winsound`.
No audio packages or prerecorded sound files required.
"""
import argparse
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import wave


class SoundAlerts:
    def __init__(self, *, muted=False, cooldown=2.2):
        self.muted = muted
        self.cooldown = cooldown
        self._last_played = float('-inf')
        self._process = None
        self._temporary_dir = tempfile.TemporaryDirectory(prefix='dms_alarm_')
        self._wav_path = Path(self._temporary_dir.name) / 'alert.wav'
        self._write_alarm(self._wav_path)

    @staticmethod
    def _write_alarm(path):
        """Generate a modest-volume two-tone repeated pattern (about 1.5s)."""
        rate = 16000
        # (frequency, seconds); 0 Hz means silence.
        pattern = [(950, .22), (0, .09), (690, .27), (0, .12),
                   (950, .22), (0, .09), (690, .27), (0, .14)]
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            for frequency, seconds in pattern:
                n = round(rate * seconds)
                samples = bytearray()
                for i in range(n):
                    if frequency:
                        # 6ms fade to avoid sharp clicks at tone boundaries.
                        fade = min(1.0, i / (rate * .006), (n - i) / (rate * .006))
                        value = int(32767 * .28 * fade * math.sin(2 * math.pi * frequency * i / rate))
                    else:
                        value = 0
                    samples.extend(struct.pack('<h', value))
                wav.writeframes(samples)

    def play(self):
        """Start an alarm if not muted and the cooldown has elapsed."""
        if self.muted:
            return
        now = time.monotonic()
        if now - self._last_played < self.cooldown:
            return
        if self._process is not None and self._process.poll() is None:
            return  # Never overlap active alerts.
        if sys.platform == 'darwin':
            try:
                self._process = subprocess.Popen(
                    ['afplay', str(self._wav_path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                print('Could not start afplay; check macOS audio setup.')
                return
        elif sys.platform == 'win32':
            import winsound
            winsound.PlaySound(str(self._wav_path),
                               winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            print('\a', end='', flush=True)  # best-effort terminal bell on Linux
        self._last_played = now

    def stop(self):
        """Stop the current alarm and allow immediate retrigger on new warning."""
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=.3)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=.3)
            self._process = None
        if sys.platform == 'win32':
            import winsound
            winsound.PlaySound(None, 0)
        self._last_played = float('-inf')

    def update(self, warning_active):
        """Call once per camera frame: repeat while warning is active; stop otherwise."""
        if warning_active and not self.muted:
            self.play()
        else:
            # Only stop when there is something to stop, to avoid repeated work.
            if self._process is not None or self._last_played != float('-inf'):
                self.stop()

    def close(self):
        self.stop()
        self._temporary_dir.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the drowsiness alarm without webcam')
    parser.add_argument('--test', action='store_true', help='Play one alarm for audio setup verification')
    args = parser.parse_args()
    if args.test:
        sound = SoundAlerts()
        try:
            sound.play()
            time.sleep(1.8)
        finally:
            sound.close()
    else:
        parser.print_help()
