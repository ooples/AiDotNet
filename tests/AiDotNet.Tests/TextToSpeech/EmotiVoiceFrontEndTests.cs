using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// EmotiVoice's front end reproduces the release's <c>g2p_cn_en</c> (jieba, pypinyin with CC-CEDICT, its cn2an copy,
/// LibriSpeech's lexicon and g2p_en) on a corpus of Chinese, English and mixed text
/// (tools/reference-data/emotivoice_frontend_reference.py).
/// </summary>
public class EmotiVoiceFrontEndTests
{
    private static JObject Reference()
    {
        const string fileName = "emotivoice_frontend_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    [Fact(Timeout = 120000)]
    public async Task Corpus_MatchesTheRelease()
    {
        await Task.Yield();
        var mismatches = Reference()["corpus"]!
            .Select(c => (Text: (string)c["text"]!, Tn: (string)c["tn"]!, Expected: (string)c["phonemes"]!))
            .Select(c => (c.Text, c.Tn, c.Expected, ActualTn: EmotiVoiceFrontEnd.NumbersToChinese(c.Text),
                Actual: string.Join(" ", EmotiVoiceFrontEnd.Default.Phonemes(c.Text))))
            .Where(c => c.Tn != c.ActualTn || string.Join(" ", c.Expected.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)) != c.Actual)
            .Select(c => $"{c.Text}\n  tn  reference {c.Tn}\n      port      {c.ActualTn}\n  g2p reference {c.Expected}\n      port      {c.Actual}")
            .ToList();
        Assert.True(mismatches.Count == 0, string.Join("\n", mismatches));
    }

    [Fact(Timeout = 120000)]
    public async Task Corpus_MapsToTheReleasedTokens()
    {
        await Task.Yield();
        var frontEnd = EmotiVoiceFrontEnd.Default;
        Assert.Equal(502, frontEnd.Tokens.Count);
        Assert.Equal(2014, frontEnd.Speakers.Count);
        int rejected = 0;
        foreach (var c in Reference()["corpus"]!)
        {
            string text = (string)c["text"]!;
            var phonemes = ((string)c["phonemes"]!).Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            // The release's token list lacks a few readings (呣's em2); its inference fails on them (token2id[ph]).
            if (!phonemes.All(frontEnd.Tokens.Contains))
            {
                Assert.Throws<ArgumentException>(() => frontEnd.TokenIds(text));
                rejected++;
                continue;
            }
            Assert.Equal(phonemes, frontEnd.TokenIds(text).Select(i => frontEnd.Tokens[i]));
        }
        Assert.Equal(1, rejected);
    }
}
