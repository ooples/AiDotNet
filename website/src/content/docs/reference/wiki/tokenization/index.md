---
title: "Tokenization"
description: "All 30 public types in the AiDotNet.tokenization namespace, organized by kind."
section: "API Reference"
---

**30** public types in this namespace, organized by kind.

## Models & Types (14)

| Type | Summary |
|:-----|:--------|
| [`BpeTokenizer`](/docs/reference/wiki/tokenization/bpetokenizer/) | Byte-Pair Encoding (BPE) tokenizer implementation for subword tokenization. |
| [`CharacterTokenizer`](/docs/reference/wiki/tokenization/charactertokenizer/) | Character-level tokenizer that splits text into individual characters. |
| [`CodeBertTokenizer`](/docs/reference/wiki/tokenization/codeberttokenizer/) | CodeBERT-compatible tokenizer for program synthesis and code understanding tasks. |
| [`CodeTokenizer`](/docs/reference/wiki/tokenization/codetokenizer/) | Code-aware tokenizer that handles programming language constructs. |
| [`MidiNote`](/docs/reference/wiki/tokenization/midinote/) | Represents a MIDI note event with timing, pitch, and velocity information. |
| [`MidiTokenizer`](/docs/reference/wiki/tokenization/miditokenizer/) | MIDI tokenizer for symbolic music representation using configurable tokenization strategies. |
| [`PhonemeTokenizer`](/docs/reference/wiki/tokenization/phonemetokenizer/) | Phoneme-based tokenizer for speech synthesis (TTS) applications. |
| [`SentencePieceTokenizer`](/docs/reference/wiki/tokenization/sentencepiecetokenizer/) | SentencePiece tokenizer implementation using Unigram language model. |
| [`SpecialTokens`](/docs/reference/wiki/tokenization/specialtokens/) | Represents special tokens used by tokenizers. |
| [`TokenizationResult`](/docs/reference/wiki/tokenization/tokenizationresult/) | Represents the result of tokenizing text, including token IDs, tokens, and attention masks. |
| [`TreeSitterTokenizer`](/docs/reference/wiki/tokenization/treesittertokenizer/) | AST-aware tokenizer using Tree-sitter for parsing source code into syntax trees. |
| [`UnigramTokenizer`](/docs/reference/wiki/tokenization/unigramtokenizer/) | Unigram Language Model tokenizer using probabilistic segmentation. |
| [`Vocabulary`](/docs/reference/wiki/tokenization/vocabulary/) | Manages a vocabulary of tokens and their IDs. |
| [`WordPieceTokenizer`](/docs/reference/wiki/tokenization/wordpiecetokenizer/) | WordPiece tokenizer implementation. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`TokenizerBase`](/docs/reference/wiki/tokenization/tokenizerbase/) | Base class for tokenizers providing common functionality. |

## Interfaces (2)

| Type | Summary |
|:-----|:--------|
| [`ITokenizer`](/docs/reference/wiki/tokenization/itokenizer/) | Interface for text tokenizers. |
| [`IVocabulary`](/docs/reference/wiki/tokenization/ivocabulary/) | Interface for vocabulary management. |

## Enums (5)

| Type | Summary |
|:-----|:--------|
| [`PhonemeSet`](/docs/reference/wiki/tokenization/phonemeset/) | Supported phoneme sets. |
| [`PretrainedTokenizerModel`](/docs/reference/wiki/tokenization/pretrainedtokenizermodel/) | Specifies pretrained tokenizer models available from HuggingFace Hub. |
| [`ProgrammingLanguage`](/docs/reference/wiki/tokenization/programminglanguage/) | Programming languages supported by the code tokenizer. |
| [`TokenizationStrategy`](/docs/reference/wiki/tokenization/tokenizationstrategy/) | MIDI tokenization strategies that control how musical notes are converted to tokens. |
| [`TreeSitterLanguage`](/docs/reference/wiki/tokenization/treesitterlanguage/) | Supported programming languages for Tree-sitter parsing. |

## Options & Configuration (3)

| Type | Summary |
|:-----|:--------|
| [`EncodingOptions`](/docs/reference/wiki/tokenization/encodingoptions/) | Options for encoding text into tokens. |
| [`TokenizationConfig`](/docs/reference/wiki/tokenization/tokenizationconfig/) | Configuration options for tokenization in the prediction pipeline. |
| [`TokenizerConfig`](/docs/reference/wiki/tokenization/tokenizerconfig/) | Configuration for HuggingFace tokenizers. |

## Helpers & Utilities (5)

| Type | Summary |
|:-----|:--------|
| [`AutoTokenizer`](/docs/reference/wiki/tokenization/autotokenizer/) | AutoTokenizer provides HuggingFace-style automatic tokenizer loading. |
| [`ClipTokenizerFactory`](/docs/reference/wiki/tokenization/cliptokenizerfactory/) | Factory for creating CLIP-compatible tokenizers. |
| [`HuggingFaceTokenizerLoader`](/docs/reference/wiki/tokenization/huggingfacetokenizerloader/) | Loads HuggingFace pretrained tokenizers. |
| [`LanguageModelTokenizerFactory`](/docs/reference/wiki/tokenization/languagemodeltokenizerfactory/) | Factory for creating tokenizers appropriate for different language model backbones. |
| [`PretrainedTokenizerModelExtensions`](/docs/reference/wiki/tokenization/pretrainedtokenizermodelextensions/) | Extension methods for PretrainedTokenizerModel enum. |

