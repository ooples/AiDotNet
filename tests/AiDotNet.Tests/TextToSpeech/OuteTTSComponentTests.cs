using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Effects;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// OuteTTS 1.0's parts around the language model against the release's own code (tools/reference-data/
/// outetts_reference.py): loudness normalization, word features, the prompt format and text normalization, chunking,
/// the sampling processors and the DAC interface's windows, chunks and fades.
/// </summary>
public class OuteTTSComponentTests
{
    private static JObject Reference()
    {
        const string fileName = "outetts_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static double[] Values(JToken token) => token.Select(v => (double)v).ToArray();

    private static void Close(double[] expected, double[] actual, string what, double tolerance = 1e-8)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: reference {expected[i]:R}, port {actual[i]:R}.");
    }

    [Fact(Timeout = 120000)]
    public async Task Loudness_MatchesPyloudnorm()
    {
        await Task.Yield();
        foreach (var entry in (JObject)Reference()["loudness"]!)
        {
            var input = Values(entry.Value!["input"]!);
            var padded = input.Length >= 9600 ? input : input.Concat(new double[9600 - input.Length]).ToArray();
            double lufs = LoudnessMeter.IntegratedLoudness(padded, 24000);
            Assert.True(Math.Abs((double)entry.Value!["lufs"]! - lufs) < 1e-9, $"{entry.Key}: LUFS {(double)entry.Value!["lufs"]!} vs {lufs}");
            Close(Values(entry.Value!["output"]!), OuteTTSAudio.ProcessAudio(input, 24000), entry.Key);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Features_MatchTheRelease()
    {
        await Task.Yield();
        foreach (var entry in (JObject)Reference()["features"]!)
        {
            var input = Values(entry.Value!["input"]!);
            var expected = entry.Value!["features"]!;
            var actual = OuteTTSAudio.Features(input, 24000);
            Assert.Equal((int)expected["energy"]!, actual.Energy);
            Assert.Equal((int)expected["spectral_centroid"]!, actual.SpectralCentroid);
            Assert.Equal((int)expected["pitch"]!, actual.Pitch);
            if (entry.Value!["pitch_frames"] is JArray frames)
            {
                Close(Values(frames), OuteTTSAudio.CalculatePitch(input, 24000), entry.Key + " pitch frames");
                Close(new[] { (double)entry.Value!["pitch"]! }, new[] { OuteTTSAudio.SinglePitchValue(input, 24000) }, entry.Key + " pitch");
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Prompts_MatchThePromptProcessor()
    {
        await Task.Yield();
        var p = (JObject)Reference()["prompts"]!;
        var texts = p["texts"]!.Select(t => (string)t!).ToArray();
        var normalized = p["normalized"]!.Select(t => (string)t!).ToArray();
        for (int i = 0; i < texts.Length; i++) Assert.Equal(normalized[i], OuteTTSPrompts.NormalizeText(texts[i]));
        var pairs = new[] { ("Hello there.", "The cat sat"), ("Hi", "Ends with period."), ("こんにちは", "日本語です"), ("こんにちは", "日本語です。"), ("Second", "First?"), ("x", "") };
        var merged = p["merged"]!.ToArray();
        for (int i = 0; i < pairs.Length; i++)
        {
            var (text, separator) = OuteTTSPrompts.MergeSpeakerText(pairs[i].Item1, pairs[i].Item2);
            Assert.Equal((string)merged[i][0]!, text);
            Assert.Equal((string)merged[i][1]!, separator);
        }
        var speaker = OuteTTSSpeaker.FromJson(p["speaker"]!.ToString());
        Assert.Equal((string)p["completion"]!, OuteTTSPrompts.CompletionPrompt("Hello, how are you doing today?", speaker));
        Assert.Equal((string)p["plain"]!, OuteTTSPrompts.CompletionPrompt("No speaker here.", null));
        Assert.Equal((string)p["training"]!, OuteTTSPrompts.TrainingPrompt(speaker));
        // The bundled speaker is the release's.
        Assert.Equal((string)p["training"]!, OuteTTSPrompts.TrainingPrompt(OuteTTSSpeaker.DefaultEnglishFemale()));
    }

    [Fact(Timeout = 120000)]
    public async Task Chunks_MatchChunkText()
    {
        await Task.Yield();
        foreach (var c in Reference()["chunks"]!)
            Assert.Equal(c["chunks"]!.Select(t => (string)t!).ToList(), OuteTTSPrompts.ChunkText((string)c["text"]!));
    }

    [Fact(Timeout = 120000)]
    public async Task Sampling_MatchesTheLogitsProcessors()
    {
        await Task.Yield();
        foreach (var c in Reference()["sampling"]!)
        {
            var ids = c["ids"]!.Select(v => (int)v).ToArray();
            var processed = OuteTTSSampler.Process(Values(c["scores"]!), ids, 1.1, 64, 0.4, 40, 0.9, 0.05);
            var expected = c["processed"]!.Select(v => v.Type == JTokenType.Null ? double.NegativeInfinity : (double)v).ToArray();
            Assert.Equal(expected.Select(double.IsNegativeInfinity), processed.Select(double.IsNegativeInfinity));
            Close(expected.Select(v => double.IsNegativeInfinity(v) ? 0 : v).ToArray(), processed.Select(v => double.IsNegativeInfinity(v) ? 0 : v).ToArray(), "processed");
        }
    }

    [Fact(Timeout = 600000)]
    public async Task DacInterface_MatchesTheRelease()
    {
        await Task.Yield();
        var fixture = Reference();
        var config = fixture["dac_config"]!;
        var codec = new DAC<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            new DACOptions
            {
                SampleRate = 24000, EncoderDim = (int)config["encoder_dim"]!, EncoderRates = config["encoder_rates"]!.Select(v => (int)v).ToArray(),
                LatentDim = (int)config["latent_dim"]!, DecoderDim = (int)config["decoder_dim"]!, DecoderRates = config["decoder_rates"]!.Select(v => (int)v).ToArray(),
                NumQuantizers = (int)config["n_codebooks"]!, CodebookSize = (int)config["codebook_size"]!, CodebookDim = (int)config["codebook_dim"]!,
                TargetBandwidthKbps = 1e6,
            });
        string path = Path.Combine(Path.GetTempPath(), $"outetts_dac_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["dac_safetensors_base64"]!));
        var weights = new WeightLoader().LoadWeights(path);
        File.Delete(path);
        codec.LoadDescriptWeights(name => weights.TryGetValue(name, out var t) ? t.ToVector().Select(v => (double)v).ToArray() : null);
        Assert.Empty(weights.Keys.Where(k => !codec.UsedDescriptNames.Contains(k)));
        var dac = fixture["dac"]!;
        var codes = OuteTTSAudio.Encode(codec, Values(dac["input"]!), 24000);
        var expected = dac["codes"]!.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        Assert.Equal(expected.Length, codes.GetLength(0));
        Assert.Equal(expected[0].Length, codes.GetLength(1));
        int differ = 0;
        for (int q = 0; q < expected.Length; q++)
            for (int t = 0; t < expected[q].Length; t++)
                if (expected[q][t] != codes[q, t]) differ++;
        Assert.True(differ == 0, $"{differ} codes differ.");
        var exact = new int[expected.Length, expected[0].Length];
        for (int q = 0; q < expected.Length; q++)
            for (int t = 0; t < expected[q].Length; t++) exact[q, t] = expected[q][t];
        Close(Values(dac["decoded"]!), OuteTTSAudio.Decode(codec, exact, 24000), "decoded", 1e-7);
    }
}
