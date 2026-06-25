---
title: "WhisperTokenizer"
description: "Tokenizer for Whisper speech recognition model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Whisper`

Tokenizer for Whisper speech recognition model.

## For Beginners

A tokenizer converts text to numbers (tokens) and back.
Whisper's tokenizer has special tokens for:

- Language codes (to specify which language to transcribe)
- Task tokens (transcribe vs translate)
- Timestamp tokens (for word-level timing)

## How It Works

Whisper uses a special tokenizer with BPE (Byte Pair Encoding) and special tokens
for controlling transcription behavior (language, task, timestamps).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WhisperTokenizer` | Initializes a tokenizer that operates in byte-level identity mode (no learned merges). |
| `WhisperTokenizer(String,String)` | Initializes a tokenizer that loads the official Whisper / GPT-2 BPE vocabulary. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EndOfText` | Gets the end of text token ID. |
| `NoSpeechToken` | Gets the no speech token ID. |
| `NoTimestampsToken` | Gets the no timestamps token ID. |
| `StartOfTranscript` | Gets the start of transcript token ID. |
| `StrictBpeMode` | When `true` (default `false`), `String)` / `Int64})` throw if the BPE vocab/merges aren't loaded instead of falling back to the byte-level identity mode. |
| `SupportedLanguages` | Gets all supported language codes. |
| `TranscribeToken` | Gets the transcribe task token ID. |
| `TranslateToken` | Gets the translate task token ID. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildByteUnicodeMaps` | Builds the GPT-2 byte-to-unicode mapping (Radford 2019 §2.2). |
| `Decode(IEnumerable<Int64>)` | Decodes a sequence of token IDs back to text. |
| `Encode(String)` | Encodes text to GPT-2 / Whisper BPE token IDs. |
| `GetLanguageToken(String)` | Gets the token ID for a language code. |
| `GetTimeFromToken(Int32)` | Converts a timestamp token ID to time in seconds. |
| `GetTimestampToken(Double)` | Gets the timestamp token ID for a given time in seconds. |
| `IsSpecialToken(Int32)` | Checks if a token ID is a special token. |
| `IsTimestampToken(Int32)` | Checks if a token ID is a timestamp token. |
| `LoadVocab(String,String)` | Loads the BPE vocabulary and merge table at runtime. |

