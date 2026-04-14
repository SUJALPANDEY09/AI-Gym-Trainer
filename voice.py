import threading
import time

try:
    import pyttsx3
    _engine = pyttsx3.init()
    _engine.setProperty("rate", 160)
    _engine.setProperty("volume", 0.9)
    _TTS_AVAILABLE = True
except Exception:
    _TTS_AVAILABLE = False

_last_spoken = ""
_last_time = 0
_COOLDOWN = 3.0  # seconds between same cue


def speak(text, force=False):
    """Non-blocking TTS with cooldown to avoid spamming."""
    global _last_spoken, _last_time
    if not _TTS_AVAILABLE:
        return

    now = time.time()
    if not force and text == _last_spoken and (now - _last_time) < _COOLDOWN:
        return

    _last_spoken = text
    _last_time = now

    def _say():
        try:
            _engine.say(text)
            _engine.runAndWait()
        except Exception:
            pass

    t = threading.Thread(target=_say, daemon=True)
    t.start()


def speak_rep_count(count):
    speak(f"{count} rep{'s' if count != 1 else ''}", force=True)


def speak_feedback(feedback):
    speak(feedback)
