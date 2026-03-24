using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Shared helper methods for text data loaders.
/// </summary>
internal static class TextLoaderHelper
{
    private static readonly TimeSpan TokenizerTimeout = TimeSpan.FromSeconds(1);
    private static readonly Regex TokenizerRegex = new Regex(
        @"[a-zA-Z]+", RegexOptions.Compiled, TokenizerTimeout);

    internal static List<string> Tokenize(string text)
    {
        var tokens = new List<string>();
        try
        {
            var matches = TokenizerRegex.Matches(text);
            foreach (Match match in matches)
                tokens.Add(match.Value.ToLowerInvariant());
        }
        catch (RegexMatchTimeoutException) { }
        return tokens;
    }

    internal static int[] TokenizeAndEncode(string text, Dictionary<string, int> vocabulary, int maxLength)
    {
        var tokens = Tokenize(text);
        int[] encoded = new int[maxLength];
        int len = Math.Min(tokens.Count, maxLength);
        for (int i = 0; i < len; i++)
            encoded[i] = vocabulary.TryGetValue(tokens[i], out int idx) ? idx : 1; // 1 = UNK
        return encoded;
    }

    internal static Dictionary<string, int> BuildVocabulary(IList<string> texts, int sampleCount, int maxVocabSize)
    {
        // Use Ordinal (case-sensitive) since Tokenize() lowercases all tokens
        var wordCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        int limit = Math.Min(sampleCount, texts.Count);
        for (int i = 0; i < limit; i++)
        {
            foreach (string token in Tokenize(texts[i]))
            {
                if (wordCounts.ContainsKey(token))
                    wordCounts[token]++;
                else
                    wordCounts[token] = 1;
            }
        }

        var vocabulary = new Dictionary<string, int>(StringComparer.Ordinal);
        int idx = 2; // 0 = PAD, 1 = UNK
        foreach (var pair in wordCounts.OrderByDescending(p => p.Value))
        {
            if (idx >= maxVocabSize) break;
            vocabulary[pair.Key] = idx;
            idx++;
        }
        return vocabulary;
    }

    internal static Tensor<T> ExtractTensorBatch<T>(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape._dims.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
