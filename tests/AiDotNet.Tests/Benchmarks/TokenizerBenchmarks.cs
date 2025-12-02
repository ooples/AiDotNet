using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Vocabulary;
using BenchmarkDotNet.Attributes;

namespace AiDotNet.Tests.Benchmarks;

/// <summary>
/// Performance benchmarks for tokenization algorithms.
/// Measures throughput, encoding speed, and memory efficiency across different tokenizer types.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(launchCount: 1, warmupCount: 3, iterationCount: 10)]
public class TokenizerBenchmarks
{
    private BpeTokenizer _bpeTokenizer = null!;
    private WordPieceTokenizer _wordPieceTokenizer = null!;
    private SentencePieceTokenizer _sentencePieceTokenizer = null!;
    private UnigramTokenizer _unigramTokenizer = null!;
    private CharacterTokenizer _characterTokenizer = null!;

    private string _shortText = null!;
    private string _mediumText = null!;
    private string _longText = null!;
    private List<string> _batchTexts = null!;

    private List<int> _tokenIdsForDecoding = null!;

    [GlobalSetup]
    public void Setup()
    {
        // Create test texts of varying lengths
        _shortText = "Hello world, this is a test.";
        _mediumText = GenerateText(500);
        _longText = GenerateText(5000);
        _batchTexts = Enumerable.Range(0, 100).Select(_ => GenerateText(100)).ToList();

        // Build training corpus
        var corpus = new List<string>
        {
            "Hello world, this is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "Natural language processing enables computers to understand text.",
            "Tokenization is the first step in text processing pipelines."
        };

        // Add more diverse training data
        for (int i = 0; i < 100; i++)
        {
            corpus.Add(GenerateText(200));
        }

        // Initialize tokenizers
        _bpeTokenizer = BpeTokenizer.Train(corpus, 1000);
        _wordPieceTokenizer = WordPieceTokenizer.Train(corpus, 1000);
        _sentencePieceTokenizer = SentencePieceTokenizer.Train(corpus, 1000);
        _unigramTokenizer = UnigramTokenizer.Train(corpus, 1000);
        _characterTokenizer = CharacterTokenizer.CreateAscii();

        // Prepare token IDs for decoding benchmarks
        _tokenIdsForDecoding = _bpeTokenizer.Encode(_mediumText).TokenIds;
    }

    // ===== BPE Tokenizer Benchmarks =====

    [Benchmark(Description = "BPE - Tokenize short text")]
    public List<string> BpeTokenizeShort() => _bpeTokenizer.Tokenize(_shortText);

    [Benchmark(Description = "BPE - Tokenize medium text")]
    public List<string> BpeTokenizeMedium() => _bpeTokenizer.Tokenize(_mediumText);

    [Benchmark(Description = "BPE - Tokenize long text")]
    public List<string> BpeTokenizeLong() => _bpeTokenizer.Tokenize(_longText);

    [Benchmark(Description = "BPE - Encode with options")]
    public TokenizationResult BpeEncode()
    {
        return _bpeTokenizer.Encode(_mediumText, new EncodingOptions
        {
            AddSpecialTokens = true,
            ReturnAttentionMask = true,
            ReturnTokenTypeIds = true,
            ReturnPositionIds = true
        });
    }

    [Benchmark(Description = "BPE - Decode tokens")]
    public string BpeDecode() => _bpeTokenizer.Decode(_tokenIdsForDecoding);

    [Benchmark(Description = "BPE - Batch encode")]
    public List<TokenizationResult> BpeBatchEncode()
    {
        return _bpeTokenizer.EncodeBatch(_batchTexts, new EncodingOptions { AddSpecialTokens = true });
    }

    // ===== WordPiece Tokenizer Benchmarks =====

    [Benchmark(Description = "WordPiece - Tokenize short text")]
    public List<string> WordPieceTokenizeShort() => _wordPieceTokenizer.Tokenize(_shortText);

    [Benchmark(Description = "WordPiece - Tokenize medium text")]
    public List<string> WordPieceTokenizeMedium() => _wordPieceTokenizer.Tokenize(_mediumText);

    [Benchmark(Description = "WordPiece - Tokenize long text")]
    public List<string> WordPieceTokenizeLong() => _wordPieceTokenizer.Tokenize(_longText);

    [Benchmark(Description = "WordPiece - Encode with options")]
    public TokenizationResult WordPieceEncode()
    {
        return _wordPieceTokenizer.Encode(_mediumText, new EncodingOptions
        {
            AddSpecialTokens = true,
            ReturnAttentionMask = true,
            ReturnTokenTypeIds = true,
            ReturnPositionIds = true
        });
    }

    // ===== SentencePiece Tokenizer Benchmarks =====

    [Benchmark(Description = "SentencePiece - Tokenize short text")]
    public List<string> SentencePieceTokenizeShort() => _sentencePieceTokenizer.Tokenize(_shortText);

    [Benchmark(Description = "SentencePiece - Tokenize medium text")]
    public List<string> SentencePieceTokenizeMedium() => _sentencePieceTokenizer.Tokenize(_mediumText);

    [Benchmark(Description = "SentencePiece - Tokenize long text")]
    public List<string> SentencePieceTokenizeLong() => _sentencePieceTokenizer.Tokenize(_longText);

    // ===== Unigram Tokenizer Benchmarks =====

    [Benchmark(Description = "Unigram - Tokenize short text")]
    public List<string> UnigramTokenizeShort() => _unigramTokenizer.Tokenize(_shortText);

    [Benchmark(Description = "Unigram - Tokenize medium text")]
    public List<string> UnigramTokenizeMedium() => _unigramTokenizer.Tokenize(_mediumText);

    [Benchmark(Description = "Unigram - Tokenize long text")]
    public List<string> UnigramTokenizeLong() => _unigramTokenizer.Tokenize(_longText);

    // ===== Character Tokenizer Benchmarks =====

    [Benchmark(Description = "Character - Tokenize short text")]
    public List<string> CharacterTokenizeShort() => _characterTokenizer.Tokenize(_shortText);

    [Benchmark(Description = "Character - Tokenize medium text")]
    public List<string> CharacterTokenizeMedium() => _characterTokenizer.Tokenize(_mediumText);

    [Benchmark(Description = "Character - Tokenize long text")]
    public List<string> CharacterTokenizeLong() => _characterTokenizer.Tokenize(_longText);

    // ===== Training Benchmarks =====

    [Benchmark(Description = "BPE - Train (1000 vocab)")]
    public BpeTokenizer BpeTrain()
    {
        var corpus = Enumerable.Range(0, 50).Select(_ => GenerateText(100)).ToList();
        return BpeTokenizer.Train(corpus, 1000);
    }

    [Benchmark(Description = "WordPiece - Train (1000 vocab)")]
    public WordPieceTokenizer WordPieceTrain()
    {
        var corpus = Enumerable.Range(0, 50).Select(_ => GenerateText(100)).ToList();
        return WordPieceTokenizer.Train(corpus, 1000);
    }

    [Benchmark(Description = "Unigram - Train (1000 vocab)")]
    public UnigramTokenizer UnigramTrain()
    {
        var corpus = Enumerable.Range(0, 50).Select(_ => GenerateText(100)).ToList();
        return UnigramTokenizer.Train(corpus, 1000);
    }

    // ===== Throughput Benchmark =====

    [Benchmark(Description = "BPE - Throughput (tokens/sec)")]
    public int BpeThroughput()
    {
        int totalTokens = 0;
        foreach (var text in _batchTexts)
        {
            var tokens = _bpeTokenizer.Tokenize(text);
            totalTokens += tokens.Count;
        }
        return totalTokens;
    }

    [Benchmark(Description = "WordPiece - Throughput (tokens/sec)")]
    public int WordPieceThroughput()
    {
        int totalTokens = 0;
        foreach (var text in _batchTexts)
        {
            var tokens = _wordPieceTokenizer.Tokenize(text);
            totalTokens += tokens.Count;
        }
        return totalTokens;
    }

    // ===== Helper Methods =====

    private static string GenerateText(int wordCount)
    {
        var words = new[]
        {
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "machine", "learning", "artificial", "intelligence", "natural",
            "language", "processing", "deep", "neural", "network", "transformer",
            "attention", "embedding", "tokenizer", "vocabulary", "sequence",
            "model", "training", "inference", "batch", "parallel", "efficient"
        };

        var random = new System.Random(42);
        var sb = new StringBuilder();

        for (int i = 0; i < wordCount; i++)
        {
            if (i > 0) sb.Append(' ');
            sb.Append(words[random.Next(words.Length)]);
        }

        return sb.ToString();
    }
}

/// <summary>
/// Memory efficiency benchmarks for tokenizers.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(launchCount: 1, warmupCount: 2, iterationCount: 5)]
public class TokenizerMemoryBenchmarks
{
    private string _largeText = null!;
    private BpeTokenizer _bpeTokenizer = null!;
    private WordPieceTokenizer _wordPieceTokenizer = null!;

    [GlobalSetup]
    public void Setup()
    {
        _largeText = GenerateLargeText(50000);

        var corpus = Enumerable.Range(0, 100)
            .Select(_ => GenerateSmallText(200))
            .ToList();

        _bpeTokenizer = BpeTokenizer.Train(corpus, 2000);
        _wordPieceTokenizer = WordPieceTokenizer.Train(corpus, 2000);
    }

    [Benchmark(Description = "BPE - Large text memory usage")]
    public TokenizationResult BpeLargeTextMemory()
    {
        return _bpeTokenizer.Encode(_largeText, new EncodingOptions
        {
            AddSpecialTokens = true,
            ReturnAttentionMask = true,
            ReturnTokenTypeIds = true,
            ReturnPositionIds = true
        });
    }

    [Benchmark(Description = "WordPiece - Large text memory usage")]
    public TokenizationResult WordPieceLargeTextMemory()
    {
        return _wordPieceTokenizer.Encode(_largeText, new EncodingOptions
        {
            AddSpecialTokens = true,
            ReturnAttentionMask = true,
            ReturnTokenTypeIds = true,
            ReturnPositionIds = true
        });
    }

    private static string GenerateLargeText(int wordCount)
    {
        var words = new[]
        {
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "machine", "learning", "artificial", "intelligence", "natural",
            "language", "processing", "deep", "neural", "network"
        };

        var random = new System.Random(42);
        var sb = new StringBuilder(wordCount * 8);

        for (int i = 0; i < wordCount; i++)
        {
            if (i > 0) sb.Append(' ');
            sb.Append(words[random.Next(words.Length)]);
        }

        return sb.ToString();
    }

    private static string GenerateSmallText(int wordCount)
    {
        var words = new[]
        {
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "machine", "learning", "artificial", "intelligence"
        };

        var random = new System.Random(42);
        var sb = new StringBuilder();

        for (int i = 0; i < wordCount; i++)
        {
            if (i > 0) sb.Append(' ');
            sb.Append(words[random.Next(words.Length)]);
        }

        return sb.ToString();
    }
}

/// <summary>
/// Padding and truncation benchmarks.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(launchCount: 1, warmupCount: 2, iterationCount: 5)]
public class TokenizerPaddingBenchmarks
{
    private BpeTokenizer _tokenizer = null!;
    private List<string> _variableLengthTexts = null!;

    [GlobalSetup]
    public void Setup()
    {
        var corpus = Enumerable.Range(0, 50)
            .Select(_ => GenerateText(100))
            .ToList();

        _tokenizer = BpeTokenizer.Train(corpus, 1000);

        // Create texts of varying lengths
        var random = new System.Random(42);
        _variableLengthTexts = Enumerable.Range(0, 100)
            .Select(_ => GenerateText(random.Next(10, 500)))
            .ToList();
    }

    [Benchmark(Description = "Encode with padding (max 512)")]
    public List<TokenizationResult> EncodeWithPadding()
    {
        return _tokenizer.EncodeBatch(_variableLengthTexts, new EncodingOptions
        {
            Padding = true,
            MaxLength = 512,
            PaddingSide = "right",
            AddSpecialTokens = true,
            ReturnAttentionMask = true
        });
    }

    [Benchmark(Description = "Encode with truncation (max 128)")]
    public List<TokenizationResult> EncodeWithTruncation()
    {
        return _tokenizer.EncodeBatch(_variableLengthTexts, new EncodingOptions
        {
            Truncation = true,
            MaxLength = 128,
            TruncationSide = "right",
            AddSpecialTokens = true
        });
    }

    [Benchmark(Description = "Encode with padding and truncation")]
    public List<TokenizationResult> EncodeWithPaddingAndTruncation()
    {
        return _tokenizer.EncodeBatch(_variableLengthTexts, new EncodingOptions
        {
            Padding = true,
            Truncation = true,
            MaxLength = 256,
            PaddingSide = "right",
            TruncationSide = "right",
            AddSpecialTokens = true,
            ReturnAttentionMask = true,
            ReturnTokenTypeIds = true,
            ReturnPositionIds = true
        });
    }

    private static string GenerateText(int wordCount)
    {
        var words = new[]
        {
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "machine", "learning", "artificial", "intelligence"
        };

        var random = new System.Random(42);
        var sb = new StringBuilder();

        for (int i = 0; i < wordCount; i++)
        {
            if (i > 0) sb.Append(' ');
            sb.Append(words[random.Next(words.Length)]);
        }

        return sb.ToString();
    }
}
