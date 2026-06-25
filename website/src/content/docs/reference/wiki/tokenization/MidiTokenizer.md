---
title: "MidiTokenizer"
description: "MIDI tokenizer for symbolic music representation using configurable tokenization strategies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.Specialized`

MIDI tokenizer for symbolic music representation using configurable tokenization strategies.

## For Beginners

Think of this tokenizer as a translator between musical notes
and a language that machine learning models can understand. Just like how text tokenizers
convert words into numbers, this tokenizer converts musical notes into tokens.

Imagine a piano: each key has a pitch (which note), and when you press it, you control
the velocity (how hard), duration (how long), and timing (when in the song). This tokenizer
captures all these aspects in different ways:

- **REMI**: Like detailed sheet music notation - captures everything (pitch, velocity,

duration, timing) as separate tokens. Use this when you need to preserve all musical details.
Example output: ["Bar", "Position_0", "Pitch_60", "Velocity_16", "Duration_4"]

- **CPWord**: Like a condensed notation - combines note information into single tokens.

Use this when you want a smaller vocabulary for your model.
Example output: ["Bar", "TimeShift_2", "Note_60_16_4"]

- **SimpleNote**: Like a basic melody line - just pitch and duration, no dynamics.

Use this for simple melody generation or when velocity doesn't matter.
Example output: ["Pitch_60", "Duration_4", "Rest_2", "Pitch_64", "Duration_4"]

## How It Works

This tokenizer converts MIDI music data into sequences of tokens that can be used for
training machine learning models on music generation, classification, or analysis tasks.
It supports three tokenization strategies, each offering different trade-offs between
expressiveness and vocabulary size.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MidiTokenizer(IVocabulary,SpecialTokens,TokenizationStrategy,Int32,Int32)` | Creates a new MIDI tokenizer with the specified configuration. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddCompoundNoteTokens(Vocabulary,Int32)` | Adds compound note tokens (Note_Pitch_VelocityBin_Duration) to the vocabulary. |
| `AddCompoundNoteTokensForPitch(Vocabulary,Int32,Int32)` | Adds compound note tokens for a specific pitch value. |
| `AddDurationAndRestTokens(Vocabulary)` | Adds Duration and Rest tokens (1-128) to the vocabulary. |
| `AddDurationAndTimeShiftTokens(Vocabulary)` | Adds Duration and TimeShift tokens (1-128) to the vocabulary. |
| `AddRangedTokens(Vocabulary,String,Int32,Int32)` | Adds ranged tokens with a prefix to the vocabulary. |
| `AddTimeShiftAndRestTokens(Vocabulary)` | Adds TimeShift and Rest tokens (1-128) to the vocabulary. |
| `CleanupTokens(List<String>)` | Cleans up tokens and converts them back to text. |
| `CreateCPWord(SpecialTokens,Int32,Int32)` | Creates a MIDI tokenizer with CPWord (Compound Word) strategy. |
| `CreateCPWordVocabulary(SpecialTokens,Int32)` | Creates the vocabulary for CPWord (Compound Word) tokenization strategy. |
| `CreateDefaultSpecialTokens` | Creates the default special tokens for MIDI tokenization. |
| `CreateREMI(SpecialTokens,Int32,Int32)` | Creates a MIDI tokenizer with REMI (Revamped MIDI) strategy. |
| `CreateREMIVocabulary(SpecialTokens,Int32)` | Creates the vocabulary for REMI tokenization strategy. |
| `CreateSimpleNote(SpecialTokens,Int32)` | Creates a MIDI tokenizer with SimpleNote strategy. |
| `CreateSimpleNoteVocabulary(SpecialTokens)` | Creates the vocabulary for SimpleNote tokenization strategy. |
| `QuantizeDuration(Int32)` | Quantizes a duration in ticks to the nearest unit of a 16th note. |
| `Tokenize(String)` | Tokenizes text representation of MIDI. |
| `TokenizeBarEvent` | Tokenizes a bar event based on the current strategy. |
| `TokenizeNoteEvent(Int32,Int32,Int32)` | Tokenizes a single note event based on the current strategy. |
| `TokenizeNotes(IEnumerable<MidiNote>)` | Tokenizes a list of MIDI notes using the configured strategy. |
| `TokenizeRestEvent(Int32)` | Tokenizes a rest event based on the current strategy. |

