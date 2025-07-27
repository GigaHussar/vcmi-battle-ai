# AI Experiments in Heroes 3 Battles (VCMI Fork)

> What happens when you let a language model fight your battles in Heroes of Might and Magic III?

## About This Project

This is a personal experiment where I modified the open-source [VCMI engine](https://github.com/vcmi/vcmi) to:
- Export structured JSON describing the full battle state every turn.
- Export legal actions for the currently active unit.
- Open a socket for external commands to play the battle.

An external agent (a Python script using the Claude or Ollama API) reads the exported JSON, selects an action, and sends it back to the engine via the socket.

## Motivation

I love programming and experimenting with games and AI. One day I wondered: what would happen if I let a language model play a Heroes 3 battle?

## Limitations

- I only tested it on **macOS (Apple Silicon)**.
- It may require adjustments to work on other platforms like Windows or Linux.
- You’ll need to build the project from source.

## Status

- Cleaned up: old neural net code removed.
- README updated for this fork.
- All modifications are shared under GPLv2+ as per VCMI’s original license.

## Installation

If you're adventurous:

```bash
# Clone my fork and build it like VCMI
git clone https://github.com/GigaHussar/vcmi-battle-ai.git
cd vcmi-battle-ai
mkdir build && cd build
cmake ..
make -j8
