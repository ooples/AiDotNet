using System.IO.Compression;
using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceTokenizerTests
{
    private static JsonDocument Reference() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "metavoice_tokenizer_reference.json")));

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void OriginalTextNormalization_MatchesLiteralUnicodeMapWhitespaceAndRejections(bool rejected)
    {
        using var document = Reference();
        foreach (var example in document.RootElement.GetProperty("normalization").EnumerateArray())
        {
            bool invalid = example.TryGetProperty("rejected", out var flag) && flag.GetBoolean();
            if (invalid != rejected) continue;
            string text = example.GetProperty("text").GetString()!;
            if (invalid) Assert.Throws<ArgumentException>(() => MetaVoiceText.Normalize(text));
            else Assert.Equal(example.GetProperty("normalized").GetString(), MetaVoiceText.Normalize(text));
        }
    }

    [Theory]
    [InlineData(2049)]
    [InlineData(1025)]
    public void OriginalTrainedBpe_Matches85UnicodeExamplesAndAppendedEndToken(int offset)
    {
        using var document = Reference(); var root = document.RootElement;
        var ranks = root.GetProperty("ranks").EnumerateArray().ToDictionary(
            x => Convert.FromBase64String(x.GetProperty("bytes").GetString()!), x => x.GetProperty("rank").GetInt32());
        var tokenizer = new MetaVoiceTokenizer(ranks, offset);
        foreach (var example in root.GetProperty("stages").EnumerateArray().Single(x => x.GetProperty("offset").GetInt32() == offset).GetProperty("examples").EnumerateArray())
        {
            var ids = tokenizer.Encode(example.GetProperty("text").GetString()!);
            Assert.Equal(example.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()), ids);
            Assert.Equal(example.GetProperty("decoded").GetString(), tokenizer.Decode(ids));
            Assert.Equal(tokenizer.EndToken, ids[ids.Length - 1]);
        }
        Assert.Throws<ArgumentException>(() => tokenizer.Encode("explicit <|endoftext|> text"));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void OriginalCheckpointMetadata_PreservesAll512ByteKeysAndBigEndianFloats(int protocol)
    {
        using var document = Reference();
        var payload = Convert.FromBase64String(document.RootElement.GetProperty(protocol == 2 ? "metadata_pickle_base64" : "metadata_pickle4_base64").GetString()!);
        using var stream = new MemoryStream(payload); using var reader = new BinaryReader(stream);
        var root = (Dictionary<string, object?>)new PickleParser(reader, preserveByteKeys: true).Parse()!;
        var arguments = (Dictionary<string, object?>)root["model_args"]!;
        Assert.Equal(.1, Assert.IsType<double>(arguments["spkemb_dropout"]));
        Assert.Equal(1e-5, Assert.IsType<double>(arguments["rmsnorm_eps"]));
        string checkpoint = Path.Combine(Path.GetTempPath(), "metavoice-metadata-" + Guid.NewGuid().ToString("N") + ".pt");
        try
        {
            using (var archive = ZipFile.Open(checkpoint, ZipArchiveMode.Create))
            using (var entry = archive.CreateEntry("original/data.pkl").Open()) entry.Write(payload, 0, payload.Length);
            var tokenizer = MetaVoiceTokenizer.FromCheckpoint(checkpoint);
            Assert.Equal(2561, tokenizer.EndToken);
            var expected = document.RootElement.GetProperty("stages")[0].GetProperty("examples")[1];
            Assert.Equal(expected.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()), tokenizer.Encode(expected.GetProperty("text").GetString()!));
        }
        finally { if (File.Exists(checkpoint)) File.Delete(checkpoint); }
    }

    [Fact]
    public void AuthorAdapters_MatchEndMarkersAndUnequalHierarchyTruncation()
    {
        using var document = Reference();
        foreach (var example in document.RootElement.GetProperty("adapters").EnumerateArray())
        {
            int[][] Rows(string key) => example.GetProperty(key).EnumerateArray()
                .Select(row => row.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray();
            var actual = MetaVoiceTokenAdapters.Decode(Rows("streams"), example.GetProperty("flattened").GetBoolean());
            Assert.Equal(example.GetProperty("text").EnumerateArray().Select(x => x.GetInt32()), actual.Text);
            var expected = Rows("audio"); Assert.Equal(expected.Length, actual.Audio.Length);
            for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual.Audio[i]);
        }
    }
}
