using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tokenization.Algorithms;
using Xunit;

namespace AiDotNet.Tests.Tokenization;

/// <summary>
/// BERT-base-uncased tokenization against Hugging Face's BertTokenizer (tools/reference-data/bert_tokenizer_reference.py):
/// the shipped vocabulary, basic tokenization (punctuation split off, accents stripped, CJK characters separated,
/// control characters dropped) and WordPiece, with [CLS] and [SEP].
/// </summary>
public class BertTokenizerReferenceTests
{
    [Fact]
    public void BertBaseUncased_MatchesTheReferenceTokenizer()
    {
        const string name = "bert_tokenizer_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "Tokenization", "ReferenceData", name);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "Tokenization", "ReferenceData", name)))
                dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", "Tokenization", "ReferenceData", name);
        }
        var tokenizer = WordPieceTokenizer.BertBaseUncased();
        foreach (var c in JsonDocument.Parse(File.ReadAllText(path)).RootElement.GetProperty("cases").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            var expected = c.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            var actual = tokenizer.Encode(text).TokenIds.ToArray();
            Assert.True(expected.SequenceEqual(actual), $"\"{text}\": [{string.Join(", ", expected)}] vs [{string.Join(", ", actual)}]");
        }
    }
}
