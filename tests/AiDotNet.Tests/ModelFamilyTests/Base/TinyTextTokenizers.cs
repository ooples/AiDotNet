using System.Collections.Generic;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Tokenizers sized for test-scale text vocabularies. A model whose real tokenizer is a pretrained one (Bark's
/// bert-base-multilingual-cased, 119 547 ids) cannot feed a fixture that shrinks the text vocabulary: its ids
/// fall outside the tiny embedding. These keep the model's own tokenization path while emitting ids the fixture
/// can embed.
/// </summary>
public static class TinyTextTokenizers
{
    /// <summary>
    /// Lowercase letters 'a'..'y' as ids 2..26, space as 1 and anything else as [UNK] = 0: 27 ids in all.
    /// </summary>
    public static CharacterTokenizer Letters()
    {
        var ids = new Dictionary<string, int> { ["[UNK]"] = 0, [" "] = 1 };
        for (char c = 'a'; c <= 'y'; c++) ids[c.ToString()] = 2 + (c - 'a');
        return new CharacterTokenizer(new AiDotNet.Tokenization.Vocabulary.Vocabulary(ids), new SpecialTokens(), lowercase: true);
    }
}
