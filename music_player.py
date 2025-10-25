#!/usr/bin/env python3
"""Terminal music player with optional AI-assisted track selection."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

import re

import gi

for namespace, version in (("Gst", "1.0"), ("GObject", "2.0"), ("Gio", "2.0")):
    gi.require_version(namespace, version)

from gi.repository import GLib, Gio, Gst  # noqa: E402

SUPPORTED_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


@dataclass(frozen=True)
class Track:
    uri: str
    label: str
    metadata: Dict[str, str] = field(default_factory=dict)
    category: str = "song"


def _has_supported_extension(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_EXTENSIONS


def _extract_metadata(path: Path) -> Dict[str, str]:
    try:
        import mutagen
    except ImportError:  # pragma: no cover - optional dependency
        return {}
    try:
        audio = mutagen.File(path, easy=True)
    except Exception:  # pragma: no cover
        return {}
    if audio is None:
        return {}
    mapping = {
        "title": ["title"],
        "artist": ["artist", "albumartist"],
        "album": ["album"],
        "genre": ["genre"],
        "date": ["date", "year"],
        "tracknumber": ["tracknumber"],
    }
    result: Dict[str, str] = {}
    for key, names in mapping.items():
        values: List[str] = []
        for name in names:
            tag = audio.get(name)
            if not tag:
                continue
            if isinstance(tag, (list, tuple)):
                values.extend(str(item).strip() for item in tag if item)
            else:
                values.append(str(tag).strip())
        if values:
            unique = [v for v in dict.fromkeys(values) if v]
            if unique:
                result[key] = ", ".join(unique)
    return result


def _augment_metadata_from_label(label: str, metadata: Dict[str, str]) -> Dict[str, str]:
    decoded = unquote(label)
    parts = decoded.split("/")
    filename = Path(decoded).name
    updated: Dict[str, str] = {}

    # Normalise existing metadata (strip whitespace, preserve original keys).
    for key, value in metadata.items():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            updated[key] = text

    def has_field(name: str) -> bool:
        """Return True if metadata already supplies a non-empty value for the field."""
        for key, value in list(updated.items()):
            if key.lower() == name and str(value).strip():
                # Ensure canonical lowercase alias is present.
                if key != name:
                    updated[name] = str(value).strip()
                return True
        return False

    def set_field(name: str, value: str) -> None:
        if value:
            updated[name] = value

    if not has_field("title"):
        title = Path(filename).stem.replace("_", " ").strip()
        if title:
            set_field("title", title)
    if not has_field("album"):
        set_field("album", "Unknown")
    if not has_field("artist"):
        candidate = ""
        if len(parts) >= 3:
            candidate = parts[-3].replace("_", " ").strip()
        elif " - " in filename:
            candidate = filename.split(" - ", 1)[0].replace("_", " ").strip()
        if candidate:
            set_field("artist", candidate)
    return updated


_SPOKEN_KEYWORDS = {
    "episode",
    "ep.",
    " ep ",
    "podcast",
    "audiobook",
    "audio drama",
    "radio drama",
    "radioplay",
    "story",
    "stories",
    "narration",
    "narrative",
    "minutes du peuple",
    "banal fantasy",
    "sketch",
    "skit",
    "sketches",
    "teaser",
    "trailer",
    "bande annonce",
    "announcement",
    "monologue",
    "dialogue",
    "scene",
    "chapter",
    "act ",
    "scene ",
    "bonus episode",
    "commentary",
    "interview",
    "audioplay",
    "audio-play",
    "audio play",
}


def _infer_track_category(label: str, metadata: Dict[str, str]) -> str:
    components: List[str] = [label.lower()]
    for key in ("title", "album", "genre", "comment", "description"):
        value = metadata.get(key)
        if value:
            components.append(str(value).lower())
    combined = " ".join(components)
    if re.search(r"\bepisode\s*\d+\b", combined):
        return "spoken-word"
    if re.search(r"\bs\d+e\d+\b", combined):
        return "spoken-word"
    if "minutes du peuple" in combined:
        return "spoken-word"
    if any(keyword in combined for keyword in _SPOKEN_KEYWORDS):
        return "spoken-word"
    return "song"


def _collect_local_tracks(root: Path) -> List[Track]:
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Directory '{root}' does not exist or is not a directory.")
    tracks: List[Track] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and _has_supported_extension(path.name):
            rel = path.relative_to(root).as_posix()
            metadata = _augment_metadata_from_label(rel, _extract_metadata(path))
            category = _infer_track_category(rel, metadata)
            metadata.setdefault("category", category)
            tracks.append(Track(path.resolve().as_uri(), rel, metadata, category))
    return tracks


def _collect_gio_tracks(uri: str) -> List[Track]:
    root = Gio.File.new_for_uri(uri)
    tracks: List[Track] = []

    def walk(directory: Gio.File, prefix: str) -> None:
        try:
            enumerator = directory.enumerate_children(
                "standard::name,standard::type",
                Gio.FileQueryInfoFlags.NONE,
                None,
            )
        except GLib.Error as exc:  # pragma: no cover
            raise ValueError(f"Unable to access '{directory.get_uri()}': {exc.message}") from exc
        try:
            while True:
                info = enumerator.next_file(None)
                if info is None:
                    break
                name = info.get_name()
                file_type = info.get_file_type()
                child = directory.get_child(name)
                rel_name = f"{prefix}{name}"
                if file_type == Gio.FileType.DIRECTORY:
                    walk(child, f"{rel_name}/")
                elif file_type == Gio.FileType.REGULAR and _has_supported_extension(name):
                    metadata = _augment_metadata_from_label(rel_name, {})
                    category = _infer_track_category(rel_name, metadata)
                    metadata.setdefault("category", category)
                    tracks.append(Track(child.get_uri(), rel_name, metadata, category))
        finally:
            enumerator.close(None)

    walk(root, "")
    return tracks


def collect_tracks(target: Union[str, Path]) -> List[Track]:
    if isinstance(target, Path):
        return _collect_local_tracks(target)
    parsed = urlparse(str(target))
    if parsed.scheme:
        if parsed.scheme == "file":
            return _collect_local_tracks(Path(unquote(parsed.path)))
        return _collect_gio_tracks(str(target))
    return _collect_local_tracks(Path(str(target)).expanduser())


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read config '{config_path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config file '{config_path}' must contain a JSON object.")
    return data


class PlaybackHistory:
    def __init__(self, path: Optional[Union[str, Path]], max_entries: int = 50) -> None:
        self._path = Path(path).expanduser().resolve() if path else None
        self._max_entries = max_entries
        self._entries: List[Dict[str, str]] = []
        self._lock = threading.RLock()
        if self._path:
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"Warning: Failed to read history '{self._path}': {exc}\n")
            return
        tracks = data.get("tracks")
        if not isinstance(tracks, list):
            return
        cleaned: List[Dict[str, str]] = []
        for item in tracks:
            if isinstance(item, dict) and item.get("label"):
                cleaned.append(
                    {
                        "label": str(item.get("label")),
                        "uri": str(item.get("uri", "")),
                        "time": str(item.get("time", "")),
                    }
                )
            elif isinstance(item, str):
                cleaned.append({"label": item, "uri": "", "time": ""})
        self._entries = cleaned[-self._max_entries :]

    def add(self, track: Track) -> None:
        entry = {
            "label": track.label,
            "uri": track.uri,
            "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        with self._lock:
            self._entries.append(entry)
            self._entries = self._entries[-self._max_entries :]
            if self._path:
                try:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    self._path.write_text(json.dumps({"tracks": self._entries}, indent=2), encoding="utf-8")
                except OSError as exc:
                    sys.stderr.write(f"Warning: Failed to update history '{self._path}': {exc}\n")

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            if self._path:
                try:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    self._path.write_text(json.dumps({"tracks": []}, indent=2), encoding="utf-8")
                except OSError as exc:
                    sys.stderr.write(f"Warning: Failed to clear history '{self._path}': {exc}\n")

    def recent_labels(self, limit: int = 10) -> List[str]:
        with self._lock:
            return [entry["label"] for entry in reversed(self._entries[-limit:])]


class MusicPlayer:
    def __init__(self) -> None:
        Gst.init(None)
        self._playbin = Gst.ElementFactory.make("playbin", "player")
        if not self._playbin:
            raise RuntimeError("Failed to create GStreamer playbin element.")
        self._playlist: List[Track] = []
        self._current_index: Optional[int] = None
        self._lock = threading.RLock()
        self._track_callback: Optional[Callable[[Track], None]] = None

        bus = self._playbin.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(target=self._loop.run, daemon=True)
        self._loop_thread.start()

    def set_track_started_callback(self, callback: Callable[[Track], None]) -> None:
        self._track_callback = callback

    def load_directory(self, directory: Union[str, Path], replace: bool = True) -> int:
        tracks = collect_tracks(directory)
        with self._lock:
            if replace:
                self.stop()
                self._playlist = tracks
                self._current_index = 0 if tracks else None
            else:
                if not self._playlist and tracks:
                    self._current_index = 0
                self._playlist.extend(tracks)
        return len(self._playlist)

    def play(self, index: Optional[int] = None) -> Optional[Track]:
        with self._lock:
            if not self._playlist:
                return None
            if index is None:
                index = self._current_index or 0
            if index < 0 or index >= len(self._playlist):
                raise IndexError(f"Track index {index} is out of range.")
            return self._set_track_and_play(index)

    def pause(self) -> None:
        with self._lock:
            self._playbin.set_state(Gst.State.PAUSED)

    def resume(self) -> None:
        with self._lock:
            self._playbin.set_state(Gst.State.PLAYING)

    def stop(self) -> None:
        with self._lock:
            self._playbin.set_state(Gst.State.NULL)

    def next_track(self, auto: bool = False) -> Optional[Track]:
        with self._lock:
            if not self._playlist:
                return None
            if self._current_index is None:
                next_index = 0
            elif auto:
                next_index = self._current_index + 1
                if next_index >= len(self._playlist):
                    self.stop()
                    return None
            else:
                next_index = (self._current_index + 1) % len(self._playlist)
            return self._set_track_and_play(next_index)

    def previous_track(self) -> Optional[Track]:
        with self._lock:
            if not self._playlist:
                return None
            if self._current_index is None:
                prev_index = 0
            else:
                prev_index = (self._current_index - 1) % len(self._playlist)
            return self._set_track_and_play(prev_index)

    def current_track(self) -> Optional[Track]:
        with self._lock:
            if self._current_index is None or not self._playlist:
                return None
            return self._playlist[self._current_index]

    def playlist(self) -> List[Track]:
        with self._lock:
            return list(self._playlist)

    def shutdown(self) -> None:
        with self._lock:
            self._playbin.set_state(Gst.State.NULL)
        if self._loop.is_running():
            self._loop.quit()
        self._loop_thread.join(timeout=1)

    def _set_track_and_play(self, index: int) -> Track:
        track = self._playlist[index]
        self._current_index = index
        self._playbin.set_state(Gst.State.NULL)
        self._playbin.set_property("uri", track.uri)
        result = self._playbin.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"Failed to start playback for {track.label}")
        if self._track_callback:
            try:
                self._track_callback(track)
            except Exception:  # pragma: no cover
                pass
        return track

    def _on_bus_message(self, _bus: Gst.Bus, message: Gst.Message) -> None:
        if message.type == Gst.MessageType.EOS:
            self.next_track(auto=True)
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write(f"GStreamer error: {err}\n")
            if debug:
                sys.stderr.write(f"Debug info: {debug}\n")
            self.stop()


class OllamaClient:
    def __init__(self, base_url: str) -> None:
        import requests

        self._base_url = base_url.rstrip('/')
        self._requests = requests

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream_callback: Optional[Callable[[str], None]] = None,
        *,
        force_json: bool = False,
    ) -> str:
        url = f"{self._base_url}/api/chat"
        stream = stream_callback is not None
        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if force_json:
            payload["format"] = "json"
        try:
            response = self._requests.post(url, json=payload, timeout=120, stream=stream)
        except self._requests.RequestException as exc:  # pragma: no cover
            raise ConnectionError(f"Failed to reach Ollama at {url}: {exc}") from exc
        if response.status_code != 200:
            body = response.text if not stream else response.content.decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed ({response.status_code}): {body}")
        if not stream:
            data = response.json()
            message = data.get("message") or {}
            content = message.get("content")
            if not content:
                raise ValueError("Ollama response missing assistant content.")
            return content

        chunks: List[str] = []
        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise RuntimeError(f"Ollama error: {event['error']}")

            message = event.get("message")
            if isinstance(message, dict):
                delta = message.get("content") or ""
            else:
                delta = ""
            if delta:
                chunks.append(delta)
                if stream_callback:
                    stream_callback(delta)
            if event.get("done"):
                break
        content = "".join(chunks)
        if not content:
            raise ValueError("Ollama stream produced no content.")
        return content


class MusicPlayerCLI:
    def __init__(
        self,
        player: MusicPlayer,
        ollama_client: Optional[object] = None,
        default_model: Optional[str] = None,
        catalog_path: Optional[Union[str, Path]] = "playlist_catalog.txt",
        system_context: Optional[str] = None,
        history: Optional[PlaybackHistory] = None,
        stream_ai: bool = True,
    ) -> None:
        self._player = player
        self._ollama = ollama_client
        self._default_model = default_model
        self._catalog_path = Path(catalog_path).resolve() if catalog_path else None
        self._ai_track_summary = ""
        self._system_context = system_context
        self._history = history
        self._ai_sessions: Dict[str, List[Dict[str, str]]] = {}
        self._session_messages: List[Dict[str, str]] = []
        self._stream_ai = stream_ai

        self._player.set_track_started_callback(self._on_track_started)

    def run(self) -> None:
        print("Simple Music Player")
        print(
            "Commands: /load <dir>, /play [index], /pause, /resume, /stop, /next, /prev, /list, /current, /resethistory, /resetchat, /help, /quit"
        )
        print("Tip: just type a sentence to let the AI pick a track. Prefix commands with '/'.")
        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue
            if raw.startswith("/"):
                if not self._handle_command(shlex.split(raw[1:])):
                    break
            else:
                self._cmd_ai(shlex.split(raw))

    def _handle_command(self, parts: List[str]) -> bool:
        if not parts:
            return True
        cmd, *args = parts
        cmd = cmd.lower()
        if cmd == "load":
            self._cmd_load(args)
        elif cmd == "play":
            self._cmd_play(args)
        elif cmd == "pause":
            self._player.pause()
            print("Paused")
        elif cmd == "resume":
            self._player.resume()
            track = self._player.current_track()
            if track:
                print(f"Resumed: {track.label}")
        elif cmd == "stop":
            self._player.stop()
            print("Stopped")
        elif cmd == "next":
            track = self._player.next_track()
            if track:
                print(f"Playing: {track.label}")
            else:
                print("Reached end of playlist.")
        elif cmd in {"prev", "previous"}:
            track = self._player.previous_track()
            if track:
                print(f"Playing: {track.label}")
        elif cmd == "ai":
            self._cmd_ai(args)
        elif cmd == "list":
            self._cmd_list()
        elif cmd == "current":
            track = self._player.current_track()
            if track:
                print(f"Current: {track.label}")
            else:
                print("No track selected.")
        elif cmd == "resethistory":
            self._reset_history()
        elif cmd == "resetchat":
            self._reset_chat_session()
        elif cmd == "help":
            self._print_help()
        elif cmd in {"quit", "exit"}:
            return False
        else:
            print("Unknown command. Type /help for options.")
        return True

    @classmethod
    def _generate_catalog_summary(cls, playlist: List[Track]) -> str:
        if not playlist:
            return "Playlist is empty."
        return "\\n".join(cls._render_track_line(idx, track) for idx, track in enumerate(playlist))

    def _cmd_load(self, args: Iterable[str]) -> None:
        if not args:
            raise ValueError("Usage: load <directory>")
        target = args[0]
        count = self._player.load_directory(target)
        if count:
            print(f"Playlist now has {count} tracks.")
        else:
            print("No audio files found. Playlist is empty.")
        self.refresh_ai_playlist()

    def _cmd_play(self, args: Iterable[str]) -> None:
        track = self._player.play(int(args[0])) if args else self._player.play()
        if track:
            playlist = self._player.playlist()
            print(f"Playing [{playlist.index(track)}]: {track.label}")
        else:
            print("Playlist is empty.")

    def _cmd_list(self) -> None:
        playlist = self._player.playlist()
        if not playlist:
            print("Playlist is empty.")
            return
        current = self._player.current_track()
        for idx, track in enumerate(playlist):
            marker = "->" if current and track == current else "  "
            print(f"{marker} {self._render_track_line(idx, track)}")

    def _print_help(self) -> None:
        print(
            "Commands (prefix with '/'):\n"
            "  /load <dir>          Load supported audio files from a local directory or SMB URI\n"
            "  /play [idx]          Start playback of current track or the specified index\n"
            "  /pause               Pause playback\n"
            "  /resume              Resume playback\n"
            "  /stop                Stop playback\n"
            "  /next                Skip to the next track (wraps around)\n"
            "  /prev                Return to the previous track\n"
            "  /ai [--model NAME] <prompt>  Ask an Ollama model to choose a track\n"
            "  /list                Show playlist with indexes\n"
            "  /current             Show the currently selected track\n"
            "  /resethistory        Clear the persisted playback history\n"
            "  /resetchat           Clear AI conversation memory\n"
            "  /help                Show this message\n"
            "  /quit                Exit the player\n"
            "\nWithout '/', your input is sent to the AI using the default model."
        )

    def _reset_history(self) -> None:
        if not self._history:
            print("History is disabled.")
            return
        self._history.reset()
        self._update_system_prompts()
        print("Playback history cleared.")

    def _reset_chat_session(self) -> None:
        self._ai_sessions.clear()
        self._session_messages.clear()
        print("Chat session memory cleared.")

    def _cmd_ai(self, args: Iterable[str]) -> None:
        if not self._ollama:
            raise ValueError("AI integration is disabled (no Ollama URL configured).")

        parts = list(args)
        model_override: Optional[str] = None
        prompt_tokens: List[str] = []
        i = 0
        while i < len(parts):
            token = parts[i]
            if token == "--model":
                i += 1
                if i >= len(parts):
                    raise ValueError("Usage: ai --model <name> <prompt>")
                model_override = parts[i]
            elif token.startswith("--model="):
                model_override = token.split("=", 1)[1]
            else:
                prompt_tokens.append(token)
            i += 1
        prompt = " ".join(prompt_tokens).strip()
        if not prompt:
            prompt = input("AI prompt: ").strip()
            if not prompt:
                raise ValueError("Prompt cannot be empty.")

        model = model_override or self._default_model
        if not model:
            raise ValueError("No AI model specified. Use --model or set a default.")

        playlist = self._player.playlist()
        if not playlist:
            print("Playlist is empty.")
            return

        messages = self._get_or_create_ai_session(model)
        user_message = self._build_ai_prompt(playlist, prompt)
        messages.append({"role": "user", "content": user_message})

        allow_spoken = self._prompt_allows_spoken(prompt)
        max_attempts = 3
        payload: Optional[Dict[str, Any]] = None
        reply = ""

        for attempt in range(1, max_attempts + 1):
            chat_kwargs: Dict[str, Any] = {"force_json": True}
            printed_stream = False
            last_chunk_had_newline = True

            if self._stream_ai:

                def handle_chunk(chunk: str) -> None:
                    nonlocal printed_stream, last_chunk_had_newline
                    if not printed_stream:
                        print("AI response (streaming):", flush=True)
                        printed_stream = True
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    last_chunk_had_newline = chunk.endswith("\n")

                chat_kwargs["stream_callback"] = handle_chunk

            try:
                reply = self._ollama.chat(model, messages, **chat_kwargs)
            except TypeError:
                # Fallback for custom clients without streaming/format support.
                if chat_kwargs.get("stream_callback"):
                    print("AI client does not support streaming; falling back to buffered response.")
                reply = self._ollama.chat(model, messages)
                printed_stream = False
                last_chunk_had_newline = True
            except Exception as exc:  # pragma: no cover
                if printed_stream and not last_chunk_had_newline:
                    print()
                messages.pop()  # remove user message on failure
                raise RuntimeError(f"Ollama request failed: {exc}") from exc

            if printed_stream and not last_chunk_had_newline:
                print()

            candidate_track: Optional[Track] = None
            try:
                payload = self._parse_ai_payload(reply)
                self._validate_ai_payload(payload)
                candidate_index = payload["index"]
                if candidate_index < 0 or candidate_index >= len(playlist):
                    raise ValueError("Selected index is out of range.")
                candidate_track = playlist[candidate_index]
                category = self._track_category(candidate_track)
                if category != "song" and not allow_spoken:
                    raise ValueError(
                        f"Selected track '{candidate_track.label}' is categorized as {category}."
                    )
            except (ValueError, IndexError) as exc:
                print(f"AI response invalid (attempt {attempt}/{max_attempts}): {exc}")
                messages.append({"role": "assistant", "content": reply})
                if candidate_track and self._track_category(candidate_track) != "song" and not allow_spoken:
                    correction = (
                        f"'{candidate_track.label}' is not a music track. Please choose a song that sounds like actual music."
                    )
                else:
                    correction = (
                        "That reply was invalid. Respond again with exactly one JSON object "
                        '{"index": <int>, "reason": "<short explanation>"} using valid JSON syntax. '
                        "Always choose a track—never reply with null or say nothing fits."
                    )
                messages.append({"role": "user", "content": correction})
                if attempt >= max_attempts:
                    fallback_index = self._fallback_track_index(playlist, allow_spoken)
                    if fallback_index is None:
                        raise RuntimeError("AI failed to provide a valid JSON response.") from exc
                    payload = {
                        "index": fallback_index,
                        "reason": "Fallback selection due to repeated invalid AI responses.",
                    }
                    print("AI response invalid after multiple attempts. Falling back to automatic selection.")
                    break
                continue

            messages.append({"role": "assistant", "content": reply})
            break

        if payload is None:
            fallback_index = self._fallback_track_index(playlist, allow_spoken)
            if fallback_index is None:
                raise RuntimeError("AI failed to provide a response.")
            payload = {
                "index": fallback_index,
                "reason": "Fallback selection due to missing AI response.",
            }

        self._trim_ai_history(model)

        index, reason = self._interpret_ai_choice(payload, playlist)
        if index is None:
            fallback_index = self._fallback_track_index(playlist, allow_spoken)
            if fallback_index is None:
                print(f"AI response: {reason or json.dumps(payload, indent=2)}")
                return
            print(f"AI response unusable; falling back to [{fallback_index}].")
            track = self._player.play(fallback_index)
            if track:
                print("Reason: Using fallback selection because AI response was unusable.")
            return

        track = self._player.play(index)
        if not track:
            print(f"AI suggested index {index}, but it is not available.")
            return

        print(f"AI selected [{index}]: {track.label}")
        if reason:
            print(f"Reason: {reason}")

    def _get_or_create_ai_session(self, model: str) -> List[Dict[str, str]]:
        session = self._ai_sessions.get(model)
        if session is None:
            session = [{"role": "system", "content": self._build_system_prompt()}]
            self._ai_sessions[model] = session
        else:
            session[0]["content"] = self._build_system_prompt()
        return session

    def _build_ai_prompt(self, playlist: List[Track], prompt: str, limit: int = 50) -> str:
        summary_lines = [
            self._render_track_line(idx, track)
            for idx, track in enumerate(playlist[:limit])
        ]
        if len(playlist) > limit:
            summary_lines.append(f"... ({len(playlist) - limit} more tracks omitted)")
        sections = ["Playlist:\n" + "\n".join(summary_lines)]
        song_count = sum(1 for track in playlist if self._track_category(track) == "song")
        spoken_count = len(playlist) - song_count
        sections.append(
            "Song inventory statistics:"
            f"\n- songs: {song_count}"
            f"\n- spoken-word / episodic tracks: {spoken_count}"
            "\nA metadata field named 'Category' is provided for every track in the playlist summary."
            " When 'Category=song' you must treat it as music. If the category is anything else"
            " (e.g., 'spoken-word'), treat it as narration/podcast and avoid it unless the user explicitly asks for it."
        )
        if self._history:
            recent = self._dedupe_preserve_order(self._history.recent_labels(limit=10))
            if recent:
                sections.append(
                    "Recently played tracks (most recent first; prefer new selections when possible, but reuse if needed):\n"
                    + "\n".join(f"- {label}" for label in recent)
                )
        sections.append(f"User request: {prompt}")
        sections.append(
            'Respond with exactly one JSON object: {"index": <int>, "reason": "<short explanation>"}.\n'
            "No extra keys, code fences, or commentary. Always pick a track (prefer one not in the recent list). "
            "Only choose entries with Category=song unless the user explicitly requests spoken-word content. "
            "Invalid JSON or null indexes will be rejected."
        )
        return "\n\n".join(sections)

    @staticmethod
    def _parse_ai_payload(reply: str) -> Dict[str, Any]:
        text = reply.strip()
        if text.startswith("<think>"):
            text = text.split("</think>", 1)[-1].strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise ValueError("Response does not contain JSON.")
            data = json.loads(match.group())
        if not isinstance(data, dict):
            raise ValueError("JSON response must be an object.")
        return data

    @staticmethod
    def _validate_ai_payload(payload: Dict[str, Any]) -> None:
        if "index" not in payload:
            raise ValueError("Missing required field 'index'.")
        index_value = payload["index"]
        if not isinstance(index_value, int) or index_value < 0:
            raise ValueError("Field 'index' must be a non-negative integer.")
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Field 'reason' must be a non-empty string.")

    def _fallback_track_index(self, playlist: List[Track], allow_spoken: bool = False) -> Optional[int]:
        if not playlist:
            return None
        history_labels: List[str] = []
        if self._history:
            history_labels = self._dedupe_preserve_order(
                self._history.recent_labels(limit=len(playlist) * 2)
            )
        history_set = set(history_labels)

        def category(track: Track) -> str:
            return self._track_category(track)

        def candidates(prefer_songs: bool) -> Iterable[int]:
            for idx, track in enumerate(playlist):
                is_song = category(track) == "song"
                if prefer_songs and not is_song:
                    continue
                if track.label in history_set:
                    continue
                yield idx

        # First try unheard songs
        for idx in candidates(prefer_songs=not allow_spoken):
            return idx

        # Then allow repeats but still prioritise songs unless user explicitly asked
        if not allow_spoken:
            for idx, track in enumerate(playlist):
                if category(track) == "song":
                    return idx

        # Finally return first unheard item of any category or default to first track
        for idx, track in enumerate(playlist):
            if track.label not in history_set:
                return idx
        return 0

    @staticmethod
    def _prompt_allows_spoken(prompt: str) -> bool:
        text = prompt.lower()
        keywords = (
            "podcast",
            "episode",
            "audio drama",
            "audiobook",
            "story",
            "narrative",
            "radio",
            "skit",
            "sketch",
            "comedy",
            "banal fantasy",
            "minutes du peuple",
            "interview",
            "spoken",
            "talk",
        )
        return any(word in text for word in keywords)

    @staticmethod
    def _track_category(track: Track) -> str:
        return (track.metadata.get("category") or track.category or "unknown").lower()

    def _interpret_ai_choice(
        self, payload: Dict[str, Any], playlist: List[Track]
    ) -> Tuple[Optional[int], str]:
        index = self._extract_index(payload)
        reason = self._extract_reason(payload)
        if isinstance(index, int) and 0 <= index < len(playlist):
            return index, reason or "Using index from AI response."

        title_candidates: List[str] = []

        def add_candidate(value: Any) -> None:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    title_candidates.append(stripped)

        for key in (
            "track",
            "trackToPlay",
            "current_track",
            "currentTrack",
            "recommendation",
            "recommendations",
            "alternatives",
            "selected_track",
            "selectedTrack",
        ):
            value = payload.get(key)
            idx_candidate = self._coerce_index(value)
            if isinstance(idx_candidate, int) and 0 <= idx_candidate < len(playlist):
                return idx_candidate, reason or "Using index from AI response."
            if isinstance(value, (list, tuple)):
                for entry in value:
                    idx_candidate = self._coerce_index(entry)
                    if isinstance(idx_candidate, int) and 0 <= idx_candidate < len(playlist):
                        return idx_candidate, reason or "Using index from AI response."
                    if isinstance(entry, dict):
                        add_candidate(entry.get("title"))
                        add_candidate(entry.get("name"))
                    else:
                        add_candidate(entry)
            elif isinstance(value, dict):
                add_candidate(value.get("title"))
                add_candidate(value.get("name"))
            else:
                add_candidate(value)

        for key in ("title", "name", "song", "response", "recommendation"):
            add_candidate(payload.get(key))

        playlist_info = payload.get("playlist")
        if isinstance(playlist_info, list):
            for entry in playlist_info:
                if isinstance(entry, dict):
                    add_candidate(entry.get("title"))
                    add_candidate(entry.get("name"))
                else:
                    add_candidate(entry)
        elif isinstance(playlist_info, dict):
            add_candidate(playlist_info.get("title"))
            add_candidate(playlist_info.get("name"))
        else:
            add_candidate(playlist_info)

        title_candidates = self._dedupe_preserve_order(title_candidates)
        for title in title_candidates:
            matched = self._match_track_by_title(title, playlist)
            if matched is not None:
                return matched, reason or f"Matched requested title '{title}'."

        index_from_text = self._extract_index_from_text(
            payload.get("response"), payload.get("message")
        )
        if isinstance(index_from_text, int) and 0 <= index_from_text < len(playlist):
            return index_from_text, reason or "Using index from AI response."

        message = payload.get("message")
        if isinstance(message, str):
            title_guess = self._extract_title_from_message(message)
            if title_guess:
                matched = self._match_track_by_title(title_guess, playlist)
                if matched is not None:
                    return matched, reason or f"Matched inferred title '{title_guess}'."

        return None, reason

    @staticmethod
    def _extract_index(payload: Dict[str, Any]) -> Optional[int]:
        keys = (
            "index",
            "playlist_position",
            "position",
            "track_index",
            "trackNumber",
            "track_number",
            "playlistIndex",
            "playlist_index",
            "current_index",
            "currentIndex",
        )
        for key in keys:
            idx = MusicPlayerCLI._coerce_index(payload.get(key))
            if isinstance(idx, int):
                return idx
        containers: List[Any] = []
        for name in (
            "current_track",
            "currentTrack",
            "track",
            "trackToPlay",
            "recommendation",
            "recommendations",
            "alternatives",
            "selected_track",
            "selectedTrack",
        ):
            value = payload.get(name)
            if isinstance(value, dict):
                containers.append(value)
            elif isinstance(value, (list, tuple)):
                containers.extend(entry for entry in value if isinstance(entry, dict))
        playlist_info = payload.get("playlist")
        if isinstance(playlist_info, list):
            containers.extend(entry for entry in playlist_info if isinstance(entry, dict))
        elif isinstance(playlist_info, dict):
            containers.append(playlist_info)
        for container in containers:
            for key in keys:
                idx = MusicPlayerCLI._coerce_index(container.get(key))
                if isinstance(idx, int):
                    return idx
        return None

    @staticmethod
    def _extract_index_from_text(*texts: Optional[str]) -> Optional[int]:
        for text in texts:
            if not isinstance(text, str):
                continue
            match = re.search(r"\[\s*(-?\d+)\s*\]", text)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None

    @staticmethod
    def _coerce_index(value: Any) -> Optional[int]:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lstrip("-").isdigit():
                return int(stripped)
        return None

    @staticmethod
    def _extract_reason(payload: Dict[str, Any]) -> str:
        fields: List[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    fields.append(stripped)

        add(payload.get("reason"))
        for name in (
            "track",
            "trackToPlay",
            "current_track",
            "currentTrack",
            "recommendation",
            "recommendations",
            "alternatives",
            "selected_track",
            "selectedTrack",
        ):
            value = payload.get(name)
            if isinstance(value, dict):
                add(value.get("reason"))
                add(value.get("comment"))
            elif isinstance(value, (list, tuple)):
                for entry in value:
                    if isinstance(entry, dict):
                        add(entry.get("reason"))
        add(payload.get("status"))
        seen: set[str] = set()
        ordered: List[str] = []
        for field in fields:
            if field not in seen:
                ordered.append(field)
                seen.add(field)
        return " ".join(ordered)

    def _trim_ai_history(self, model: str, keep: int = 10) -> None:
        history = self._ai_sessions.get(model)
        if not history or len(history) <= keep + 1:
            return
        system_message = history[0]
        history[:] = [system_message] + history[-keep:]

    def refresh_ai_playlist(self) -> None:
        playlist = self._player.playlist()
        summary = (
            "\n".join(self._render_track_line(idx, track) for idx, track in enumerate(playlist))
            if playlist
            else "Playlist is empty."
        )
        self._ai_track_summary = summary
        if self._catalog_path:
            try:
                self._catalog_path.parent.mkdir(parents=True, exist_ok=True)
                self._catalog_path.write_text(summary, encoding="utf-8")
            except OSError as exc:
                print(f"Warning: Failed to write playlist catalog: {exc}")
        self._update_system_prompts()

    def _build_system_prompt(self) -> str:
        context = self._system_context or "You are a helpful DJ assistant."
        instructions = (
            "You must respond with exactly one JSON object of the form {\"index\": <int|null>, \"reason\": \"<short explanation>\"}.\n"
            "Always return an integer index when selecting a track, even if it was played recently.\n"
            "Return JSON only—no narration before or after the object."
        )
        recent = self._recent_history_summary()
        catalog = self._ai_track_summary or "Playlist is currently empty."
        return (
            f"{context}\n"
            f"{instructions}\n\n"
            "Recently played tracks (most recent first; prefer new selections when possible, but reuse if needed):\n"
            f"{recent}\n\n"
            "Current playlist catalog:\n"
            f"{catalog}"
        )

    def _recent_history_summary(self, limit: int = 10) -> str:
        if not self._history:
            return "None yet."
        recent = self._history.recent_labels(limit)
        recent = self._dedupe_preserve_order(recent)
        if not recent:
            return "None yet."
        return "\n".join(f"- {label}" for label in recent)

    def _update_system_prompts(self) -> None:
        prompt = self._build_system_prompt()
        for session in self._ai_sessions.values():
            if session:
                session[0] = {"role": "system", "content": prompt}

    def _on_track_started(self, track: Track) -> None:
        if self._history:
            self._history.add(track)
        self._update_system_prompts()

    @staticmethod
    def _dedupe_preserve_order(labels: Iterable[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for label in labels:
            if label not in seen:
                ordered.append(label)
                seen.add(label)
        return ordered

    @staticmethod
    def _render_track_line(idx: int, track: Track) -> str:
        meta = MusicPlayerCLI._metadata_summary(track.metadata)
        base = f"[{idx}] {track.label}"
        return f"{base} | {meta}" if meta else base

    @staticmethod
    def _metadata_summary(metadata: Dict[str, str]) -> str:
        order = ["title", "artist", "album", "genre", "date", "tracknumber", "category"]
        parts = [f"{key.capitalize()}={metadata[key]}" for key in order if metadata.get(key)]
        if not parts:
            parts = [f"{k}={v}" for k, v in metadata.items() if v]
        return "; ".join(parts)

    @staticmethod
    def _match_track_by_title(title: str, playlist: List[Track]) -> Optional[int]:
        def normalize(value: str) -> str:
            value = re.sub(r"^\[\s*-?\d+\s*\]", "", value)
            value = value.casefold()
            value = re.sub(r"\.[a-z0-9]{1,5}$", "", value)
            value = re.sub(r"[\s_\-/]+", " ", value)
            return value.strip()

        target = normalize(title)
        if not target:
            return None
        for idx, track in enumerate(playlist):
            candidates = [track.label, Path(track.label).name, Path(track.label).stem]
            for candidate in candidates:
                if normalize(candidate) == target:
                    return idx
        for idx, track in enumerate(playlist):
            if target in normalize(track.label):
                return idx
        return None

    @staticmethod
    def _extract_title_from_message(message: str) -> Optional[str]:
        match = re.findall(r"['\"]([^'\"]+)['\"]", message)
        if match:
            return match[-1]
        candidates = re.findall(r"[A-Za-z0-9][A-Za-z0-9\s\-\._/]{2,}", message)
        if not candidates:
            return None
        candidates.sort(key=len, reverse=True)
        return candidates[0].strip()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Simple terminal music player.")
    parser.add_argument("directory", nargs="?", type=str, help="Optional directory or URI to preload")
    parser.add_argument("--append", action="store_true", help="Append to existing playlist")
    parser.add_argument("--ollama-url", default=None, help="Ollama base URL (empty string disables)")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    args = parser.parse_args(argv)

    config: Dict[str, Any] = {}
    if args.config:
        try:
            config = load_config(args.config)
        except ValueError as exc:
            parser.error(str(exc))
    config_dir = Path(args.config).expanduser().resolve().parent if args.config else Path.cwd()

    player = MusicPlayer()

    target = args.directory or config.get("default_music_uri")
    preload_error = None
    if target:
        try:
            count = player.load_directory(target, replace=not args.append)
        except ValueError as exc:
            preload_error = str(exc)
            count = 0
        if count:
            print(f"Preloaded {count} tracks from {target}")
        elif preload_error is None:
            print("No audio files found during preload. Playlist is empty.")
    if preload_error:
        sys.stderr.write(f"Warning: {preload_error}\n")

    ollama_url = args.ollama_url if args.ollama_url is not None else config.get("ollama_url")
    ollama_client = OllamaClient(ollama_url) if ollama_url else None

    default_model = config.get("default_ai_model")
    if default_model is not None and not isinstance(default_model, str):
        parser.error("Config value 'default_ai_model' must be a string.")

    catalog_path = config.get("ai_catalog_path", "playlist_catalog.txt")
    if catalog_path is not None and not isinstance(catalog_path, str):
        parser.error("Config value 'ai_catalog_path' must be a string or null.")
    catalog_path_value = None
    if catalog_path:
        catalog_path_value = Path(catalog_path).expanduser()
        if not catalog_path_value.is_absolute():
            catalog_path_value = (config_dir / catalog_path_value).resolve()

    system_context = None
    context_path = config.get("ai_context_path")
    if context_path:
        if not isinstance(context_path, str):
            parser.error("Config value 'ai_context_path' must be a string or null.")
        context_file = Path(context_path).expanduser()
        if not context_file.is_absolute():
            context_file = (config_dir / context_file).resolve()
        try:
            system_context = context_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            sys.stderr.write(f"Warning: Failed to read context {context_file}: {exc}\n")

    history_path = config.get("ai_history_path", "playback_history.json")
    if history_path is not None and not isinstance(history_path, str):
        parser.error("Config value 'ai_history_path' must be a string or null.")
    history_path_value = None
    if history_path:
        history_path_value = Path(history_path).expanduser()
        if not history_path_value.is_absolute():
            history_path_value = (config_dir / history_path_value).resolve()

    history_limit = config.get("ai_history_limit", 50)
    if not isinstance(history_limit, int) or history_limit <= 0:
        parser.error("Config value 'ai_history_limit' must be a positive integer.")
    history = PlaybackHistory(history_path_value, max_entries=history_limit)

    cli = MusicPlayerCLI(
        player,
        ollama_client,
        default_model=default_model,
        catalog_path=catalog_path_value,
        system_context=system_context,
        history=history,
    )
    cli.refresh_ai_playlist()
    try:
        cli.run()
    finally:
        player.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
