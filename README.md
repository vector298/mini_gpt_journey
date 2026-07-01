# Character-Level Bigram Language Model

A minimal character-level bigram language model built in PyTorch — the simplest possible version of a language model, where each character is predicted using only the one character before it.

This is the first step in a larger project to build a small language model (SLM) from scratch, following the progression: **bigram → self-attention → full transformer**.

## Credit

Built while following Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial (part of his *Neural Networks: Zero to Hero* series). All credit for the teaching and original approach goes to him — this repo is my own implementation, written while learning.

## What it does

- Reads a text file (`input.txt`) and builds a character-level vocabulary
- Trains a simple embedding-table-based model to predict the next character given the current one
- Generates new text by sampling one character at a time

Since it only ever looks at a single previous character, the generated text won't look like real words yet — that's expected at this stage. Adding self-attention (next step) is what lets the model actually learn context.

## Requirements
