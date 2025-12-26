using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Vocabulary;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tokenization;

/// <summary>
/// Factory for creating CLIP-compatible tokenizers.
/// </summary>
/// <remarks>
/// <para>
/// CLIP uses a BPE tokenizer with a vocabulary of 49408 tokens. This factory
/// provides methods to create tokenizers from pretrained vocabulary files
/// or to use a default configuration for testing.
/// </para>
/// <para><b>For Beginners:</b> CLIP needs a special tokenizer to break text into pieces.
///
/// A tokenizer factory is like a tool shop that builds tokenizers:
/// 1. You can load a pretrained tokenizer (recommended for production)
/// 2. You can create a simple tokenizer for testing
/// 3. The factory handles all the configuration details
///
/// Example usage:
/// <code>
/// // Load from pretrained files (recommended)
/// var tokenizer = ClipTokenizerFactory.FromPretrained(
///     "path/to/vocab.json",
///     "path/to/merges.txt"
/// );
///
/// // Or create a simple one for testing
/// var tokenizer = ClipTokenizerFactory.CreateSimple();
/// </code>
/// </para>
/// </remarks>
public static class ClipTokenizerFactory
{
    /// <summary>
    /// The default vocabulary size for CLIP models.
    /// </summary>
    public const int DefaultVocabSize = 49408;

    /// <summary>
    /// The default maximum sequence length for CLIP text encoder.
    /// </summary>
    public const int DefaultMaxLength = 77;

    /// <summary>
    /// The CLIP-specific pre-tokenization pattern.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This pattern is similar to GPT-2 but handles lowercase conversion and
    /// special handling of punctuation that CLIP expects.
    /// </para>
    /// </remarks>
    public const string ClipPattern = @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

    /// <summary>
    /// Creates a CLIP tokenizer from pretrained vocabulary and merge files.
    /// </summary>
    /// <param name="vocabPath">Path to the vocabulary JSON file (vocab.json).</param>
    /// <param name="mergesPath">Path to the merges text file (merges.txt).</param>
    /// <returns>A CLIP-compatible BPE tokenizer.</returns>
    /// <remarks>
    /// <para>
    /// The vocabulary file should be a JSON dictionary mapping tokens to IDs.
    /// The merges file should contain BPE merge rules, one per line.
    /// </para>
    /// <para><b>For Beginners:</b> To use CLIP's pretrained tokenizer:
    ///
    /// 1. Download the vocabulary files from HuggingFace:
    ///    - openai/clip-vit-base-patch32
    ///    - Files: vocab.json, merges.txt
    ///
    /// 2. Load them using this method:
    ///    <code>
    ///    var tokenizer = ClipTokenizerFactory.FromPretrained(
    ///        "vocab.json",
    ///        "merges.txt"
    ///    );
    ///    </code>
    ///
    /// 3. Use the tokenizer:
    ///    <code>
    ///    var result = tokenizer.Encode("a photo of a cat");
    ///    // result.TokenIds: [49406, 320, 1125, 539, 320, 2368, 49407, ...]
    ///    </code>
    /// </para>
    /// </remarks>
    public static BpeTokenizer FromPretrained(string vocabPath, string mergesPath)
    {
        if (string.IsNullOrWhiteSpace(vocabPath))
            throw new ArgumentException("Vocabulary path cannot be null or empty.", nameof(vocabPath));
        if (string.IsNullOrWhiteSpace(mergesPath))
            throw new ArgumentException("Merges path cannot be null or empty.", nameof(mergesPath));
        if (!File.Exists(vocabPath))
            throw new FileNotFoundException($"Vocabulary file not found: {vocabPath}");
        if (!File.Exists(mergesPath))
            throw new FileNotFoundException($"Merges file not found: {mergesPath}");

        // Load vocabulary
        var vocabJson = File.ReadAllText(vocabPath);
        var vocabDict = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabJson);
        if (vocabDict == null)
            throw new InvalidOperationException("Failed to parse vocabulary file.");

        var vocabulary = new Vocabulary.Vocabulary("<|endoftext|>");
        foreach (var kvp in vocabDict)
        {
            vocabulary.AddToken(kvp.Key);
        }

        // Load merges
        var mergesText = File.ReadAllLines(mergesPath);
        var merges = new Dictionary<(string, string), int>();
        int rank = 0;

        // Process merges, skipping header lines and empty lines
        var validLines = mergesText
            .Where(line => !line.StartsWith("#") && !string.IsNullOrWhiteSpace(line))
            .Select(line => line.Split(' '))
            .Where(parts => parts.Length == 2);

        foreach (var parts in validLines)
        {
            merges[(parts[0], parts[1])] = rank++;
        }

        return new BpeTokenizer(vocabulary, merges, SpecialTokens.Clip(), ClipPattern);
    }

    /// <summary>
    /// Creates a simple CLIP tokenizer for testing without pretrained files.
    /// </summary>
    /// <param name="corpus">Optional corpus to train on. If null, uses a minimal configuration.</param>
    /// <param name="vocabSize">The vocabulary size to train. Default is 1000 for quick testing.</param>
    /// <returns>A CLIP-style BPE tokenizer.</returns>
    /// <remarks>
    /// <para>
    /// This creates a minimal tokenizer suitable for testing and development.
    /// For production use, always use <see cref="FromPretrained"/> with the
    /// actual CLIP vocabulary files.
    /// </para>
    /// <para><b>For Beginners:</b> Use this for quick testing only!
    ///
    /// This tokenizer won't produce the same results as the real CLIP tokenizer.
    /// It's only meant for:
    /// - Unit testing
    /// - Development and debugging
    /// - Understanding how the tokenizer works
    ///
    /// For real applications, always use the pretrained vocabulary files.
    /// </para>
    /// </remarks>
    public static BpeTokenizer CreateSimple(IEnumerable<string>? corpus = null, int vocabSize = 1000)
    {
        // Use a basic English corpus if none provided
        corpus ??= new[]
        {
            "a photo of a cat",
            "a photo of a dog",
            "a picture of a bird",
            "an image of a car",
            "a drawing of a house",
            "the quick brown fox jumps over the lazy dog",
            "hello world",
            "artificial intelligence",
            "machine learning",
            "neural network"
        };

        return BpeTokenizer.Train(
            corpus,
            vocabSize,
            SpecialTokens.Clip(),
            ClipPattern
        );
    }

    /// <summary>
    /// Gets the default encoding options for CLIP text encoding.
    /// </summary>
    /// <param name="maxLength">The maximum sequence length. Default is 77.</param>
    /// <returns>Encoding options configured for CLIP.</returns>
    /// <remarks>
    /// <para>
    /// CLIP expects text to be:
    /// - Padded to exactly 77 tokens (or truncated if longer)
    /// - Starting with the BOS token
    /// - Ending with the EOS token
    /// </para>
    /// </remarks>
    public static EncodingOptions GetDefaultEncodingOptions(int maxLength = DefaultMaxLength)
    {
        return new EncodingOptions
        {
            MaxLength = maxLength,
            Padding = true,
            PaddingSide = "right",
            Truncation = true,
            TruncationSide = "right",
            AddSpecialTokens = true,
            ReturnAttentionMask = true
        };
    }

    /// <summary>
    /// Validates that a tokenizer is compatible with CLIP.
    /// </summary>
    /// <param name="tokenizer">The tokenizer to validate.</param>
    /// <returns>True if the tokenizer is compatible, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// A CLIP-compatible tokenizer must:
    /// - Have the correct special tokens
    /// - Support the expected vocabulary size (close to 49408)
    /// </para>
    /// </remarks>
    public static bool IsClipCompatible(ITokenizer tokenizer)
    {
        if (tokenizer == null)
            return false;

        // Check special tokens
        var specialTokens = tokenizer.SpecialTokens;
        if (specialTokens.BosToken != "<|startoftext|>" && specialTokens.ClsToken != "<|startoftext|>")
            return false;
        if (specialTokens.EosToken != "<|endoftext|>")
            return false;

        // Check vocabulary size is reasonable for CLIP
        if (tokenizer.VocabularySize < 1000 || tokenizer.VocabularySize > 100000)
            return false;

        return true;
    }
}
