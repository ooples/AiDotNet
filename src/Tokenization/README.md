# AiDotNet Tokenization Framework

A modern, comprehensive tokenization framework for .NET supporting state-of-the-art subword tokenization algorithms.

## Features

### Core Tokenizers

- **BPE (Byte-Pair Encoding)**: Used by GPT models
- **WordPiece**: Used by BERT and similar models
- **SentencePiece**: Used for multilingual models (Unigram language model)

### Code Tokenization

- **CodeTokenizer**: Language-aware tokenization with identifier splitting
- **CodeBertTokenizer**: CodeBERT-compatible tokenizer for program synthesis

### Essential Capabilities

- Vocabulary training from corpus
- Special tokens management ([CLS], [SEP], [PAD], [UNK], [MASK], [EOS], [BOS])
- Encoding/decoding with truncation and padding
- Attention mask generation
- HuggingFace pretrained model compatibility
- AST-aware code tokenization
- Language-specific handlers (Python, C#, Java, JavaScript)

## Usage Examples

### Training a BPE Tokenizer

```csharp
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;

var corpus = new List<string>
{
    "Hello world",
    "Natural language processing",
    "Machine learning is awesome"
};

var tokenizer = BpeTokenizer.Train(
    corpus,
    vocabSize: 1000,
    specialTokens: SpecialTokens.Gpt()
);

// Encode text
var result = tokenizer.Encode("Hello world", new EncodingOptions
{
    AddSpecialTokens = true,
    Padding = true,
    MaxLength = 128
});

Console.WriteLine($"Tokens: {string.Join(", ", result.Tokens)}");
Console.WriteLine($"Token IDs: {string.Join(", ", result.TokenIds)}");
Console.WriteLine($"Attention Mask: {string.Join(", ", result.AttentionMask)}");

// Decode back to text
var decoded = tokenizer.Decode(result.TokenIds);
Console.WriteLine($"Decoded: {decoded}");
```

### Training a WordPiece Tokenizer

```csharp
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;

var corpus = new List<string>
{
    "Hello world",
    "BERT uses WordPiece tokenization",
    "Subword tokenization is powerful"
};

var tokenizer = WordPieceTokenizer.Train(
    corpus,
    vocabSize: 1000,
    specialTokens: SpecialTokens.Bert()
);

// Encode with BERT-style special tokens
var result = tokenizer.Encode("Hello world", new EncodingOptions
{
    AddSpecialTokens = true  // Adds [CLS] and [SEP]
});
```

### Using Code Tokenization

```csharp
using AiDotNet.Tokenization.CodeTokenization;
using AiDotNet.Tokenization.Algorithms;

// Create base tokenizer
var baseTokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 5000);

// Create code tokenizer
var codeTokenizer = new CodeTokenizer(
    baseTokenizer,
    ProgrammingLanguage.CSharp,
    splitIdentifiers: true
);

// Tokenize code with identifier splitting
var tokens = codeTokenizer.Tokenize("getUserNameById");
// Result: ["get", "User", "Name", "By", "Id"]

// Use CodeBERT for code + natural language
var codeBert = new CodeBertTokenizer(
    vocabulary,
    ProgrammingLanguage.Python
);

var result = codeBert.EncodeCodeAndNL(
    code: "def add(a, b): return a + b",
    naturalLanguage: "return sum of two numbers"
);
```

### Loading HuggingFace Pretrained Tokenizers

```csharp
using AiDotNet.Tokenization.HuggingFace;

// Load a pretrained tokenizer
var tokenizer = HuggingFaceTokenizerLoader.LoadFromDirectory(
    "/path/to/bert-base-uncased"
);

// Use it like any other tokenizer
var result = tokenizer.Encode("Hello world");

// Save a tokenizer
HuggingFaceTokenizerLoader.SaveToDirectory(tokenizer, "/path/to/output");
```

### Advanced Encoding Options

```csharp
var options = new EncodingOptions
{
    AddSpecialTokens = true,
    MaxLength = 512,
    Padding = true,
    PaddingSide = "right",
    Truncation = true,
    TruncationSide = "right",
    ReturnAttentionMask = true,
    ReturnTokenTypeIds = true,
    ReturnOffsets = false
};

var result = tokenizer.Encode("Some text", options);
```

### Batch Processing

```csharp
var texts = new List<string>
{
    "First text",
    "Second text",
    "Third text"
};

var results = tokenizer.EncodeBatch(texts, new EncodingOptions
{
    Padding = true,
    MaxLength = 128
});

foreach (var result in results)
{
    Console.WriteLine($"Tokens: {string.Join(", ", result.Tokens)}");
}
```

## Architecture

```text
Tokenization/
├── Interfaces/
│   ├── ITokenizer.cs          # Main tokenizer interface
│   └── IVocabulary.cs         # Vocabulary management interface
├── Models/
│   ├── TokenizationResult.cs  # Tokenization output
│   ├── EncodingOptions.cs     # Encoding configuration
│   └── SpecialTokens.cs       # Special tokens configuration
├── Core/
│   └── TokenizerBase.cs       # Base tokenizer implementation
├── Vocabulary/
│   └── Vocabulary.cs          # Vocabulary implementation
├── Algorithms/
│   ├── BpeTokenizer.cs        # Byte-Pair Encoding
│   ├── WordPieceTokenizer.cs  # WordPiece algorithm
│   └── SentencePieceTokenizer.cs  # SentencePiece/Unigram
├── HuggingFace/
│   ├── TokenizerConfig.cs     # HF config model
│   └── HuggingFaceTokenizerLoader.cs  # Load/save HF tokenizers
└── CodeTokenization/
    ├── CodeTokenizer.cs       # Language-aware tokenizer
    └── CodeBertTokenizer.cs   # CodeBERT compatibility
```

## Special Tokens

Different model families use different special tokens:

### BERT-style
```csharp
var specialTokens = SpecialTokens.Bert();
// [UNK], [PAD], [CLS], [SEP], [MASK]
```

### GPT-style
```csharp
var specialTokens = SpecialTokens.Gpt();
// <|endoftext|> for all special purposes
```

### T5-style
```csharp
var specialTokens = SpecialTokens.T5();
// <unk>, <pad>, </s>
```

## Performance Considerations

- **Caching**: BPE tokenizer caches word tokenizations for faster repeated tokenization
- **Batch Processing**: Use `EncodeBatch` for processing multiple texts efficiently
- **Vocabulary Size**: Larger vocabularies provide better coverage but slower tokenization
- **Identifier Splitting**: Can be disabled for faster code tokenization when not needed

## Compatibility

This framework is compatible with:
- HuggingFace Transformers tokenizers
- CodeBERT and similar code models
- GPT, BERT, T5, and other transformer models

## Supported Languages (Code Tokenization)

- C#
- Python
- Java
- JavaScript
- TypeScript
- Generic (language-agnostic)

## Contributing

To add a new tokenization algorithm:
1. Implement `ITokenizer` or extend `TokenizerBase`
2. Add appropriate tests in `tests/AiDotNet.Tests/Tokenization/`
3. Update this README with usage examples

## References

- [BPE Paper](https://arxiv.org/abs/1508.07909)
- [WordPiece in BERT](https://arxiv.org/abs/1810.04805)
- [SentencePiece](https://arxiv.org/abs/1808.06226)
- [CodeBERT](https://arxiv.org/abs/2002.08155)
