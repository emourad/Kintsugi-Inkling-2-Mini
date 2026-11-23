The layer of this bundle: an OpenAI-compatible proxy in front of llama-server (the start script starts it).

In this build it does one job: saved conversations. After each answer it saves a long conversation to disk (slots;
a full save, then small deltas). When a request continues a saved conversation that the server does not hold (after a
restart, or when two chats take turns), it restores the saved state first, so the server reads only the new text.
It also relays streamed answers of /v1/messages (Claude Code) and /v1/responses (Codex) as they come.

Its other tools (the run_python code interpreter, API cards, the API check, example checks) are off in this build
(the start script passes --no-interpreter and the other --no-* flags).

Needs: Python 3.10 or newer, standard library only. The sandbox imports wasmtime only when a sandbox tool runs, and
they are all off here. Without Python, the start script runs the plain server and prints a note.

Files: bonsai_layer.py (the proxy), slot_cache.py (saved conversations), sandbox_path.py and wasi-python/sandbox.py
(the sandbox, unused here), apicards*.py, apilint.py, example_check.py, stray_tags.py (tools, off here).
Licence: Apache-2.0 (see LICENSE in the bundle's folder).
