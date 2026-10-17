using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The Mandarin G2P reproduces VALL-E X's reference front end (Plachtaa/VALL-E-X <c>chinese_to_ipa</c>: cn2an, jieba,
/// pypinyin and the bopomofo-to-IPA table) exactly.
/// </summary>
/// <remarks><c>ReferenceData/mandarin_g2p_reference.json</c> holds the reference's output for hand-written sentences and
/// 2000 seeded sentences of random dictionary words, numbers and punctuation, and cn2an's numerals
/// (<c>tools/reference-data/mandarin_g2p_reference.py</c>).</remarks>
public class MandarinG2PTests
{
    private static JObject Reference()
    {
        const string fileName = "mandarin_g2p_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    [Fact(Timeout = 120000)]
    public async Task Ipa_MatchesTheReferenceOnEverySentence()
    {
        await Task.Yield();
        var fixture = Reference();
        var sentences = fixture["sentences"]!.Select(s => (string)s!).ToArray();
        var expected = fixture["ipa"]!.Select(s => (string)s!).ToArray();
        var mismatches = Enumerable.Range(0, sentences.Length)
            .Select(i => (Sentence: sentences[i], Expected: expected[i], Actual: MandarinG2P.Default.ToIpa(sentences[i])))
            .Where(r => r.Expected != r.Actual)
            .ToList();
        Assert.True(mismatches.Count == 0,
            $"{mismatches.Count} of {sentences.Length} sentences differ; first: {mismatches.FirstOrDefault().Sentence} => " +
            $"reference '{mismatches.FirstOrDefault().Expected}', port '{mismatches.FirstOrDefault().Actual}'.");
    }

    [Fact(Timeout = 60000)]
    public async Task Numerals_MatchCn2an()
    {
        await Task.Yield();
        var fixture = Reference();
        var numbers = fixture["numbers"]!.Select(s => (string)s!).ToArray();
        var expected = fixture["an2cn"]!.Select(s => (string)s!).ToArray();
        for (int i = 0; i < numbers.Length; i++)
            Assert.Equal(expected[i], MandarinG2P.AnToCn(numbers[i]));
    }

    [Fact(Timeout = 60000)]
    public async Task Phonemize_SplitsTheIpaIntoCharactersAndWordSeparators()
    {
        await Task.Yield();
        // "你好，世界。" => "ni↓↑xɑʊ↓↑, s`ɹ`↓tʃ⁼iɛ↓." one character each, the space as "_".
        var phonemes = MandarinG2P.Default.Phonemize("你好，世界。");
        Assert.Equal("ni↓↑xɑʊ↓↑,_s`ɹ`↓tʃ⁼iɛ↓.".Select(c => c.ToString()), phonemes);
    }
}
