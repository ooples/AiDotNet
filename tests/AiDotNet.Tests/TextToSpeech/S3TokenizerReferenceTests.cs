using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CosyVoice 1's S3 speech tokenizer against xingchensong/S3Tokenizer (<c>tools/reference-data/s3tokenizer_reference.py</c>):
/// Whisper's 128-band log-mel, the encoder states and tokens of a padded batch (the port encodes each utterance at its own
/// length), and the tokens of a log-mel longer than 30 s, which the reference windows and merges.
/// </summary>
public class S3TokenizerReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "s3tokenizer_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    }

    private static (S3Tokenizer<double> Tokenizer, double[] Positions, JObject Fixture) Load()
    {
        var f = Reference();
        var c = f["config"]!;
        var configuration = new S3TokenizerConfiguration(Mels: (int)c["n_mels"]!, Context: (int)c["n_audio_ctx"]!, State: (int)c["n_audio_state"]!,
            Heads: (int)c["n_audio_head"]!, Layers: (int)c["n_audio_layer"]!, CodebookSize: (int)c["n_codebook_size"]!);
        var tokenizer = new S3Tokenizer<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), configuration, new Random(0));
        // SafetensorsReader keeps the fixture's float64 weights; WeightLoader would round them to float32.
        var weights = SafetensorsReader.Read(Convert.FromBase64String((string)f["safetensors_base64"]!));
        var used = new HashSet<string>();
        tokenizer.LoadTorchWeights((name, shape) =>
        {
            var values = weights.ReadAsDouble(name);
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            used.Add(name);
            return values;
        });
        Assert.Equal(weights.Names.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
        return (tokenizer, weights.ReadAsDouble("encoder.positional_embedding"), f);
    }

    private static void AssertClose(IReadOnlyList<double> expected, Func<int, double> actual, int count, string what)
    {
        Assert.Equal(expected.Count, count);
        for (int i = 0; i < count; i++)
            Assert.True(Math.Abs(expected[i] - actual(i)) <= 1e-9 * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: reference {expected[i]:R}, port {actual(i):R}.");
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    /// <summary>Frames <c>[0, length)</c> of item <paramref name="item"/> of a flattened <c>[B, mels, frames]</c> batch.</summary>
    private static Tensor<double> Mel(double[] values, int item, int mels, int frames, int length)
    {
        var mel = new Tensor<double>(new[] { mels, length });
        for (int m = 0; m < mels; m++)
            for (int t = 0; t < length; t++) mel[m, t] = values[(item * mels + m) * frames + t];
        return mel;
    }

    [Fact(Timeout = 60000)]
    public async Task LogMel_MatchesWhisper()
    {
        await Task.Yield();
        var f = Reference();
        var log = S3Tokenizer<double>.LogMel(Doubles(f["audio"]!));
        int frames = log.GetLength(1);
        AssertClose(Doubles(f["log_mel"]!), i => log[i / frames, i % frames], log.Length, "log-mel");
    }

    [Fact(Timeout = 60000)]
    public async Task PositionalTable_IsWhisperSinusoids()
    {
        await Task.Yield();
        var (tokenizer, positions, _) = Load();
        var c = tokenizer.Configuration;
        var table = S3Tokenizer<double>.Sinusoids(c.Context, c.State);
        AssertClose(positions, i => table[i / c.State, i % c.State], table.Length, "positional embedding");
    }

    [Fact(Timeout = 120000)]
    public async Task BatchStatesAndTokens_MatchS3Tokenizer()
    {
        await Task.Yield();
        var (tokenizer, _, f) = Load();
        int mels = tokenizer.Configuration.Mels;
        var lengths = f["batch_lengths"]!.Select(v => (int)v).ToArray();
        var batch = Doubles(f["batch"]!);
        for (int i = 0; i < lengths.Length; i++)
        {
            var hidden = tokenizer.Encode(Mel(batch, i, mels, lengths.Max(), lengths[i]));
            AssertClose(Doubles(f["hidden"]![i]!), j => hidden[j], hidden.Length, $"states[{i}]");
            Assert.Equal(f["codes"]![i]!.Select(v => (int)v), tokenizer.Quantize(hidden));
        }
    }

    [Fact(Timeout = 300000)]
    public async Task LongUtterance_WindowsAndMergesLikeS3Tokenizer()
    {
        await Task.Yield();
        var (tokenizer, _, f) = Load();
        int mels = tokenizer.Configuration.Mels;
        var values = Doubles(f["long"]!);
        int frames = values.Length / mels;
        Assert.True(frames > 3000);
        Assert.Equal(f["long_codes"]!.Select(v => (int)v), tokenizer.Tokenize(Mel(values, 0, mels, frames, frames)));
    }
}
