using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Tokenization.Specialized;
using Xunit;

namespace AiDotNet.Tests.Tokenization;

/// <summary>
/// The T5 tokenizer gives the ids sentencepiece gives with google-t5/t5-base's model, plus <c>&lt;/s&gt;</c> (what
/// transformers 4.x's slow <c>T5Tokenizer</c> returns), on 2023 texts:
/// music descriptions, whitespace and Unicode edge cases, special-token strings and seeded random multilingual strings
/// (<c>tools/reference-data/t5_tokenizer_reference.py</c>).
/// </summary>
public class T5TextTokenizerReferenceTests
{
    private static JsonElement Cases()
    {
        const string name = "t5_tokenizer_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "Tokenization", "ReferenceData", name);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "Tokenization", "ReferenceData", name)))
                dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", "Tokenization", "ReferenceData", name);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement.GetProperty("cases");
    }

    [Fact(Timeout = 120000)]
    public async Task EveryCase_MatchesTheReferenceIds()
    {
        await Task.Yield();
        var tokenizer = new T5TextTokenizer();
        int total = 0;
        var mismatches = new System.Collections.Generic.List<string>();
        foreach (var c in Cases().EnumerateArray())
        {
            total++;
            string text = c.GetProperty("text").GetString() ?? "";
            var expected = c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var actual = tokenizer.Encode(text);
            if (!expected.SequenceEqual(actual) && mismatches.Count < 10)
                mismatches.Add($"{JsonSerializer.Serialize(text)}: expected [{string.Join(", ", expected)}], got [{string.Join(", ", actual)}]");
        }
        Assert.True(total > 2000, $"Only {total} cases.");
        Assert.True(mismatches.Count == 0, string.Join(Environment.NewLine, mismatches));
    }

    [Fact(Timeout = 30000)]
    public async Task Vocabulary_HasThePiecesAndTheSentinels()
    {
        await Task.Yield();
        var tokenizer = new T5TextTokenizer();
        Assert.Equal(32100, tokenizer.VocabularySize);
        Assert.Equal(new[] { 32099, 1 }, tokenizer.Encode("<extra_id_0>"));
    }
}
