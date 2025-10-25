import contextlib
import io
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_player import (
    MusicPlayerCLI,
    PlaybackHistory,
    Track,
)


class DummyPlayer:
    """Minimal stub that satisfies the CLI contract without touching GStreamer."""

    def __init__(self, playlist):
        self._playlist = list(playlist)
        self._callback = None
        self.last_played = None
        self._queue = deque()

    def playlist(self):
        return list(self._playlist)

    def set_track_started_callback(self, callback):
        self._callback = callback

    def play(self, index=None):
        if not self._playlist:
            return None
        if index is None and self._queue:
            track = self._queue.popleft()
            try:
                index = self._playlist.index(track)
            except ValueError:
                index = 0
        elif index is None:
            index = 0
        if index < 0 or index >= len(self._playlist):
            raise IndexError
        track = self._playlist[index]
        self.last_played = track
        if self._callback:
            self._callback(track)
        return track

    def playlist_stats(self):
        total = len(self._playlist)
        def count(key):
            return sum(1 for track in self._playlist if track.metadata.get(key))
        return {
            'total': total,
            'with_title': count('title'),
            'with_artist': count('artist'),
            'with_album': count('album'),
            'with_genre': count('genre'),
            'with_date': count('date'),
        }

    def queue_track(self, index):
        if index < 0 or index >= len(self._playlist):
            raise IndexError
        track = self._playlist[index]
        self._queue.append(track)
        return track

    def queue(self):
        return list(self._queue)

    def clear_queue(self):
        self._queue.clear()


class PlaybackHistoryTests(unittest.TestCase):
    def test_history_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "history.json"
            history = PlaybackHistory(history_file, max_entries=2)
            history.add(Track("file:///one", "Song One"))
            history.add(Track("file:///two", "Song Two"))
            history.add(Track("file:///three", "Song Three"))

            data = json.loads(history_file.read_text(encoding="utf-8"))
            labels = [entry["label"] for entry in data["tracks"]]
            self.assertEqual(labels, ["Song Two", "Song Three"])

            # Reload to ensure persistence works
            reloaded = PlaybackHistory(history_file, max_entries=2)
            self.assertEqual(reloaded.recent_labels(), ["Song Three", "Song Two"])


class PromptGenerationTests(unittest.TestCase):
    def _make_cli(self, playlist, history=None):
        player = DummyPlayer(playlist)
        cli = MusicPlayerCLI(
            player,
            ollama_client=None,
            default_model="model",
            catalog_path=None,
            history=history,
            stream_ai=False,
        )
        cli._ai_track_summary = cli._generate_catalog_summary(playlist)
        return cli

    def test_ai_prompt_excludes_recent_history(self):
        history = PlaybackHistory(None, max_entries=5)
        history.add(Track("file:///one", "Song One"))
        history.add(Track("file:///two", "Song Two"))

        playlist = [
            Track("file:///one", "Song One"),
            Track("file:///two", "Song Two"),
        ]
        cli = self._make_cli(playlist, history)

        prompt = cli._build_ai_prompt(playlist, "Play something nice")
        self.assertNotIn("Recently played tracks", prompt)
        self.assertNotIn("Song One", prompt.split("Available songs", 1)[0])

    def test_system_prompt_no_longer_mentions_history(self):
        history = PlaybackHistory(None, max_entries=5)
        history.add(Track("file:///one", "Song One"))
        playlist = [Track("file:///one", "Song One")]
        cli = self._make_cli(playlist, history)
        cli.refresh_ai_playlist()
        system_prompt = cli._build_system_prompt()
        self.assertNotIn("Recently played tracks", system_prompt)


class MetadataAugmentationTests(unittest.TestCase):
    def test_augment_from_label(self):
        from music_player import _augment_metadata_from_label
        label = 'Artist Name/Album Title/My Song - Live.mp3'
        metadata = _augment_metadata_from_label(label, {})
        self.assertEqual(metadata.get('title'), 'My Song - Live')
        self.assertEqual(metadata.get('album'), 'Unknown')
        self.assertEqual(metadata.get('artist'), 'Artist Name')

    def test_preserve_existing_album_metadata(self):
        from music_player import _augment_metadata_from_label
        label = 'Some Artist/Some Folder/Track.mp3'
        metadata = _augment_metadata_from_label(label, {'album': 'Existing Album'})
        self.assertEqual(metadata.get('album'), 'Existing Album')

    def test_case_insensitive_album_metadata(self):
        from music_player import _augment_metadata_from_label
        label = 'Artist/Folder/Song.mp3'
        metadata = _augment_metadata_from_label(label, {'Album': 'Proper Album'})
        self.assertEqual(metadata.get('album'), 'Proper Album')

    def test_infer_category_marks_episode(self):
        from music_player import _infer_track_category
        label = 'Banal Fantasy/Banal Fantasy - Episode 24.mp3'
        metadata = {'title': 'Banal Fantasy - Episode 24'}
        self.assertEqual(_infer_track_category(label, metadata), 'spoken-word')

    def test_infer_category_defaults_to_song(self):
        from music_player import _infer_track_category
        label = 'Artist/Album/Regular Song.mp3'
        metadata = {'title': 'Regular Song'}
        self.assertEqual(_infer_track_category(label, metadata), 'song')

class ResetHistoryTests(unittest.TestCase):
    def test_reset_history(self):
        from music_player import MusicPlayerCLI, PlaybackHistory, Track, MusicPlayer
        player = DummyPlayer([Track('file:///1', 'One')])
        history = PlaybackHistory(None, max_entries=5)
        history.add(Track('file:///old', 'Old Song'))
        cli = MusicPlayerCLI(player, history=history, catalog_path=None, stream_ai=False)
        cli._reset_history()
        self.assertEqual(history.recent_labels(), [])

class ResponseParsingTests(unittest.TestCase):
    def setUp(self):
        self.playlist = [
            Track("file:///0", "Zero"),
            Track("file:///1", "One"),
            Track("file:///2", "Two"),
            Track("file:///3", "Three"),
        ]
        self.cli = MusicPlayerCLI(
            DummyPlayer(self.playlist),
            default_model="model",
            catalog_path=None,
            stream_ai=False,
        )

    def test_numeric_track_field(self):
        payload = {"track": 2}
        index, reason = self.cli._interpret_ai_choice(payload, self.playlist)
        self.assertEqual(index, 2)
        self.assertTrue(reason)

    def test_recommendation_and_alternatives(self):
        payload = {
            "recommendation": "Three",
            "alternatives": [
                {"title": "Two"},
                "One",
            ],
        }
        index, _ = self.cli._interpret_ai_choice(payload, self.playlist)
        self.assertEqual(index, 3)

    def test_playlist_string(self):
        payload = {"playlist": "Two"}
        index, _ = self.cli._interpret_ai_choice(payload, self.playlist)
        self.assertEqual(index, 2)

    def test_selected_track_field(self):
        payload = {"selected_track": "Two"}
        index, reason = self.cli._interpret_ai_choice(payload, self.playlist)
        self.assertEqual(index, 2)
        self.assertTrue(reason)

    def test_response_with_bracket_index(self):
        payload = {"response": "Requested '[2] Two' "}
        index, reason = self.cli._interpret_ai_choice(payload, self.playlist)
        self.assertEqual(index, 2)
        self.assertTrue(reason)

    def test_parse_ai_payload_strips_think(self):
        raw = "<think>internal</think>{\"index\": 1}"
        data = MusicPlayerCLI._parse_ai_payload(raw)
        self.assertEqual(data, {"index": 1})


class StreamingOutputTests(unittest.TestCase):
    def test_cli_streams_ai_response(self):
        playlist = [
            Track("file:///0", "Zero"),
            Track("file:///1", "One"),
        ]

        class DummyOllama:
            def __init__(self):
                self.stream_used = False

            def chat(self, model, messages, stream_callback=None, *, force_json=False):
                assert force_json, "Expected chat to enforce JSON format"
                parts = ['{"index": 1, "reason": "Tes', 't stream"}']
                if stream_callback:
                    self.stream_used = True
                    for chunk in parts:
                        stream_callback(chunk)
                return "".join(parts)

        player = DummyPlayer(playlist)
        ollama = DummyOllama()
        cli = MusicPlayerCLI(
            player,
            ollama_client=ollama,
            default_model="model",
            catalog_path=None,
        )

        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            cli._cmd_ai(["Pick", "something"])
        output = buffer.getvalue()

        self.assertTrue(ollama.stream_used)
        self.assertIn("AI response (streaming):", output)
        self.assertIn('{"index": 1, "reason": "Test stream"}', output)
        self.assertIn("AI selected [1]: One", output)


class PayloadValidationTests(unittest.TestCase):
    def test_reject_null_index(self):
        from music_player import MusicPlayerCLI

        with self.assertRaises(ValueError):
            MusicPlayerCLI._validate_ai_payload({"index": None, "reason": "nope"})

    def test_reject_negative_index(self):
        from music_player import MusicPlayerCLI

        with self.assertRaises(ValueError):
            MusicPlayerCLI._validate_ai_payload({"index": -1, "reason": "nope"})


class FallbackSelectionTests(unittest.TestCase):
    def test_fallback_when_ai_returns_null_repeatedly(self):
        playlist = [
            Track("file:///0", "Zero"),
            Track("file:///1", "One"),
        ]

        class NullOllama:
            def chat(self, model, messages, stream_callback=None, *, force_json=False):
                assert force_json
                return '{"index": null, "reason": "nothing"}'

        player = DummyPlayer(playlist)
        cli = MusicPlayerCLI(
            player,
            ollama_client=NullOllama(),
            default_model="model",
            catalog_path=None,
            stream_ai=False,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            cli._cmd_ai(["Pick", "something"])

        self.assertIsNotNone(player.last_played)
        self.assertEqual(player.last_played.label, "Zero")


class QueueCommandTests(unittest.TestCase):
    def test_queue_add_and_list(self):
        playlist = [
            Track("file:///0", "Zero"),
            Track("file:///1", "One"),
            Track("file:///2", "Two"),
        ]
        player = DummyPlayer(playlist)
        cli = MusicPlayerCLI(player, default_model="model", catalog_path=None, stream_ai=False)

        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            cli._cmd_queue(["add", "0", "2"])
        output = buffer.getvalue()
        self.assertIn("Queued [0]", output)
        self.assertIn("Queued [2]", output)

        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            cli._cmd_queue(["list"])
        output = buffer.getvalue()
        self.assertIn("Zero", output)
        self.assertIn("Two", output)

    def test_queue_clear(self):
        playlist = [
            Track("file:///0", "Zero"),
            Track("file:///1", "One"),
        ]
        player = DummyPlayer(playlist)
        cli = MusicPlayerCLI(player, default_model="model", catalog_path=None, stream_ai=False)
        cli._cmd_queue(["add", "0"])
        self.assertTrue(player.queue())
        cli._cmd_queue(["clear"])
        self.assertFalse(player.queue())


if __name__ == "__main__":
    unittest.main()
