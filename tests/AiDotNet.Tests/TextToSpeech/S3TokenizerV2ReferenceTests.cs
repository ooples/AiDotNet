using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// S3Tokenizer v2, CosyVoice 2's speech tokenizer, against xingchensong/S3Tokenizer's <c>S3TokenizerV2</c>
/// (<c>tools/reference-data/s3tokenizer_v2_reference.py</c>): encoder states and FSQ tokens of a padded batch, and the
/// tokens of an utterance longer than 30 s.
/// </summary>
public class S3TokenizerV2ReferenceTests
{
    private static readonly Lazy<JObject> Data = new(() =>
    {
        const string fileName = "s3tokenizer_v2_reference.json";
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
    });

    /// <summary>The reference script's weights(): exact integer arithmetic on the parameter's name and index.</summary>
    private static double[] Weights(string name, int count)
    {
        long h = 0;
        for (int j = 0; j < name.Length; j++) h += (j + 1) * (long)name[j];
        h %= 100003;
        double b = name.Contains("ln") && name.EndsWith("weight", StringComparison.Ordinal) ? 1.0 : 0.0;
        double scale = name.Contains("project_down") ? 8.0 : 0.2;
        var values = new double[count];
        for (long i = 0; i < count; i++) values[i] = b + scale * (((i * i * 7919 + i * 104729 + h) % 2003) / 2003.0 - 0.5);
        return values;
    }

    private static S3Tokenizer<double> Build()
    {
        var c = Data.Value["config"]!;
        var tokenizer = new S3Tokenizer<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), new S3TokenizerConfiguration(
            Mels: (int)c["n_mels"]!, State: (int)c["n_audio_state"]!, Heads: (int)c["n_audio_head"]!, Layers: (int)c["n_audio_layer"]!,
            CodebookSize: (int)c["n_codebook_size"]!, FirstStride: 2, Version: S3TokenizerVersion.V2), new Random(0));
        var shapes = Data.Value["parameters"]!.ToDictionary(p => (string)p[0]!, p => p[1]!.Aggregate(1, (a, s) => a * (int)s));
        var used = new HashSet<string>();
        tokenizer.LoadTorchWeights((name, shape) =>
        {
            Assert.True(shapes.TryGetValue(name, out int count), $"{name} is not an S3TokenizerV2 parameter.");
            Assert.Equal(count, shape.Aggregate(1, (a, s) => a * s));
            used.Add(name);
            return Weights(name, count);
        });
        Assert.Equal(shapes.Keys.OrderBy(n => n, StringComparer.Ordinal), used.OrderBy(n => n, StringComparer.Ordinal));
        return tokenizer;
    }

    // One utterance's log-mel [Mels, length] out of the padded batch [B, Mels, T].
    private static Tensor<double> Utterance(double[] batch, int index, int mels, int frames, int length)
    {
        var mel = new Tensor<double>(new[] { mels, length });
        for (int m = 0; m < mels; m++)
            for (int t = 0; t < length; t++) mel[m, t] = batch[(index * mels + m) * frames + t];
        return mel;
    }

    [Fact(Timeout = 300000)]
    public async Task StatesAndTokens_MatchS3TokenizerV2()
    {
        await Task.Yield();
        var tokenizer = Build();
        var d = Data.Value;
        int mels = (int)d["config"]!["n_mels"]!;
        var lengths = d["batch_lengths"]!.Select(v => (int)v).ToArray();
        var batch = d["batch"]!.Select(v => (double)v).ToArray();
        for (int i = 0; i < lengths.Length; i++)
        {
            var mel = Utterance(batch, i, mels, lengths.Max(), lengths[i]);
            var hidden = tokenizer.Encode(mel);
            var expected = d["hidden"]![i]!.Select(v => (double)v).ToArray();
            Assert.Equal(expected.Length, hidden.Length);
            for (int j = 0; j < expected.Length; j++)
                Assert.True(Math.Abs(expected[j] - hidden[j]) <= 1e-9 * Math.Max(1, Math.Abs(expected[j])),
                    $"hidden[{i}][{j}]: reference {expected[j]:R}, port {hidden[j]:R}.");
            Assert.Equal(d["codes"]![i]!.Select(v => (int)v).ToArray(), tokenizer.Tokenize(mel));
        }
    }

    [Fact(Timeout = 300000)]
    public async Task LongUtterance_IsWindowedAndMerged()
    {
        await Task.Yield();
        var tokenizer = Build();
        var d = Data.Value;
        int mels = (int)d["config"]!["n_mels"]!;
        var values = d["long"]!.Select(v => (double)v).ToArray();
        var mel = new Tensor<double>(new[] { mels, values.Length / mels }, new Vector<double>(values));
        Assert.Equal(d["long_codes"]!.Select(v => (int)v).ToArray(), tokenizer.Tokenize(mel));
    }
}
