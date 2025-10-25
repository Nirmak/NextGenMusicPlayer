import contextlib
import io
import json
import tempfile
import unittest
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

    def playlist(self):
        return list(self._playlist)

    def set_track_started_callback(self, callback):
        self._callback = callback

    def play(self, index=None):
        if not self._playlist:
            return None
        if index is None:
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

    def test_ai_prompt_includes_recent_history(self):
        history = PlaybackHistory(None, max_entries=5)
        history.add(Track("file:///one", "Song One"))
        history.add(Track("file:///two", "Song Two"))

        playlist = [
            Track("file:///one", "Song One"),
            Track("file:///two", "Song Two"),
        ]
        cli = self._make_cli(playlist, history)

        prompt = cli._build_ai_prompt(playlist, "Play something nice")
        self.assertIn("Song One", prompt)
        self.assertIn("Song Two", prompt)
        # Ensure duplicated recent entries are deduplicated
        history.add(Track("file:///one", "Song One"))
        prompt_again = cli._build_ai_prompt(playlist, "Another one")
        self.assertIn("Song One", prompt_again)
        history_section = prompt_again.split("Recently played tracks", 1)[1]
        self.assertEqual(history_section.count("Song One"), 1)

    def test_system_prompt_mentions_history(self):
        history = PlaybackHistory(None, max_entries=5)
        history.add(Track("file:///one", "Song One"))
        playlist = [Track("file:///one", "Song One")]
        cli = self._make_cli(playlist, history)
        cli.refresh_ai_playlist()
        system_prompt = cli._build_system_prompt()
        self.assertIn("Song One", system_prompt)


class MetadataAugmentationTests(unittest.TestCase):
    def test_augment_from_label(self):
        from music_player import _augment_metadata_from_label
        label = 'Artist Name/Album Title/My Song - Live.mp3'
        metadata = _augment_metadata_from_label(label, {})
        self.assertEqual(metadata.get('title'), 'My Song - Live')
        self.assertEqual(metadata.get('album'), 'Album Title')
        self.assertEqual(metadata.get('artist'), 'Artist Name')

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


if __name__ == "__main__":
    unittest.main()
