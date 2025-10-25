# AI-Assisted Terminal Music Player
This project is a terminal-first music player that wraps GStreamer playback with a conversational, AI-assisted DJ. It loads tracks from local paths or remote URIs, keeps a lightweight play history, and lets you ask an Ollama model to pick the next song based on the current playlist and recent listening.

## Key Features
- CLI-driven playback: load directories, start/stop music, navigate the queue, and inspect the playlist without leaving the terminal.
- Flexible media sources: works with normal folders, `file://` URIs, and remote shares (e.g., SMB) through Gio.
- AI track selection: optional integration with an Ollama model that receives playlist metadata, a configurable DJ persona (tuned to avoid episodic content), prioritises unplayed songs, and returns up to five matching tracks (first plays immediately, the rest enqueue).
- Conversational DJ: ask questions or vibe-checks—the assistant can reply in plain text when no queue is needed.
- Ad-hoc song queue: build a temporary play queue from the loaded library without losing the full playlist.
- Live streaming of AI responses so you can watch progress instead of waiting on a blank terminal.
- Automatic cataloguing: exports the loaded playlist to `playlist_catalog.txt` and stores recent selections in `playback_history.json`.
- Metadata awareness: extracts tags with Mutagen when available and infers missing titles/artists from file paths.

## Project Layout
- `music_player.py` – main module containing the GStreamer player, CLI controller, AI integration, and supporting utilities.
- `config.json` – sample configuration file; customize paths, Ollama URL, default model, and history settings.
- `dj_context.txt` – prompt used as the DJ “persona” for Ollama (loaded on startup).
- `playlist_catalog.txt` – auto-generated playlist dump written whenever the library changes.
- `playback_history.json` – persisted play history (capped by `ai_history_limit`).
- `tests/` – unit tests built with `unittest`/`pytest`.

## Requirements
- Python 3.9+.
- GStreamer 1.0 runtime and plugins (`gstreamer1.0-plugins-base`, `gstreamer1.0-libav`, etc.).
- PyGObject bindings (`python3-gi`, `gir1.2-gst-1.0`).
- Optional/extra Python packages:
  - `requests` (required for Ollama integration).
  - `mutagen` (metadata extraction; optional but recommended).
  - `pytest` (for running the test suite).

On Debian/Ubuntu:
```bash
sudo apt install python3-gi gir1.2-gst-1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-libav
python3 -m venv .venv
. .venv/bin/activate
pip install requests mutagen pytest
```
On other platforms install the equivalent GStreamer and PyGObject packages before creating a virtual environment.

## Configuration
`music_player.py` reads a JSON config (defaults to `config.json`). All keys are optional.

| Key | Description |
| --- | --- |
| `default_music_uri` | Directory or URI to preload on startup (e.g., `/media/music`, `smb://server/share/music`). |
| `default_ai_model` | Model name to request from Ollama (e.g., `llama3.1:latest`). Leave unset to require `--model` in AI commands. |
| `ollama_url` | Base URL for the Ollama server. Set to `""` to disable AI features. |
| `ai_catalog_path` | Where to write the generated playlist catalog. Relative paths resolve against the config file location. |
| `ai_context_path` | Path to a text file whose contents seed the system prompt (e.g., `dj_context.txt`). |
| `ai_history_limit` | Maximum number of recent plays retained in history (default 50). |
| `ai_history_path` | JSON file that persists playback history between sessions. |

## Running the Player
Activate your virtual environment and start the CLI:
```bash
python music_player.py            # uses config.json defaults
python music_player.py ~/Music    # preloads a specific directory
python music_player.py --ollama-url ""  # disable AI for the session
```

### CLI Commands
Commands are prefixed with `/`; free-form text triggers the AI DJ (when enabled).

| Command | Action |
| --- | --- |
| `/load <dir-or-uri>` | Replace the playlist with tracks from the path/URI. Use `--append` CLI flag to append instead. |
| `/play [index]` | Play the current track or the given zero-based index. |
| `/pause`, `/resume`, `/stop` | Control playback. |
| `/next`, `/prev` | Move forward/backward in the playlist (wraps when invoked manually). |
| `/list` | Display the playlist with the current track highlighted. |
| `/current` | Show the currently playing track. |
| `/ai <prompt>` | Manually send a prompt to the AI assistant. |
| `/queue add <idx...>` | Append one or more tracks to the temporary play queue. |
| `/queue list` | Display queued tracks in order. |
| `/queue clear` | Empty the play queue. |
| `/resethistory`, `/resetchat` | Clear persisted history or the current AI conversation. |
| `/help` | Show available commands. |
| `/quit` | Exit the player. |

### AI Workflow
When AI support is enabled:
1. The current playlist is summarised to `playlist_catalog.txt`.
2. `dj_context.txt` (or your custom file) is merged with instructions about response format.
3. Playback history is still recorded locally for your reference, but it is no longer shared with the AI.
4. The DJ persona instructs the model to lean on known artists/songs, keep its internal reasoning succinct, avoid episodic/podcast-style tracks, and prefer songs you have not heard recently.
5. The player forces Ollama into JSON mode and rejects any reply that lacks the exact `{"indexes": [<int>, ...], "reason": "..."}` schema (or tries to decline); after repeated failures it falls back to an automatic pick.
6. Free-form input or `/ai ...` calls `OllamaClient`; the assistant either replies in plain text (for pure conversation) or returns JSON indexes. The first index plays immediately and the remaining (up to four) are appended to the temporary queue.
7. While the model composes a reply, its output streams live in the terminal so you can confirm it is still working.

Replace `dj_context.txt` to change the model’s persona or constraints. To disable AI completely, set `ollama_url` to an empty string in the config or use `--ollama-url ""` at runtime.

## Developing & Contributing
- Follow standard Python style (PEP 8, type hints where practical). Add concise docstrings/comments for complex sections.
- Run the existing tests before submitting changes:
  ```bash
  python -m pytest
  ```
- Add test coverage for new behaviour—`tests/test_music_player.py` demonstrates how to stub the player and CLI.
- Keep GStreamer interactions isolated to `MusicPlayer` and prefer pure functions for logic to simplify testing.
- Document new configuration keys or commands in this README.

## Troubleshooting
- **Missing GStreamer plugins**: Playback errors usually point to absent codecs; install the relevant `gstreamer1.0-*` packages.
- **PyGObject import errors**: Ensure `python3-gi` (Linux) or the Homebrew formula `pygobject3` (macOS) is installed before creating the virtual environment.
- **Ollama connection failures**: Verify the server URL and that `requests` is installed; set `--ollama-url ""` to run offline.

Enjoy the music and feel free to extend the DJ! 
