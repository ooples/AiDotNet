using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Engines;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SilentCipher, CSM's watermark, against sony/silentcipher's encode_wav and decode_wav with a reduced transform and message
/// decoder and closed-form weights (tools/reference-data/silentcipher_reference.py): every checkpoint name has a place, the
/// watermarked audio and the message decoder's scores match, and decoding agrees with and without phase-shift search. With
/// the released weights (<c>SILENTCIPHER_DIR</c>, <c>sony/silentcipher</c>'s <c>44_1_khz/73999_iteration</c>) CSM's key
/// embedded at 36 dB reads back, and unmarked audio does not carry it.
/// </summary>
public class SilentCipherReferenceTests
{
    private readonly ITestOutputHelper _output;

    public SilentCipherReferenceTests(ITestOutputHelper output) => _output = output;

    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] ClosedForm(string name, int[] shape)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var values = new double[count];
        for (int i = 0; i < count; i++)
        {
            long k = i;
            double s = Math.Sin(0.37 * i + 0.11 * name.Length + 0.0013 * (k * k % 9973));
            values[i] = name.EndsWith("num_batches_tracked", StringComparison.Ordinal) ? 0
                : name.EndsWith("bn.weight", StringComparison.Ordinal) ? 1 + 0.2 * s
                : name.EndsWith("running_var", StringComparison.Ordinal) ? 1 + 0.5 * s
                : shape.Length >= 2 ? s * Math.Sqrt(2.0 * shape[0] / count)
                : 0.05 * s;
        }
        return values;
    }

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    [Fact(Timeout = 600000)]
    public async Task EncodeAndDecode_MatchSilentCipher()
    {
        await Task.Yield();
        var reference = Load("silentcipher_reference.json");
        var configuration = new SilentCipherConfiguration
        {
            Fft = reference.GetProperty("fft").GetInt32(),
            Hop = reference.GetProperty("hop").GetInt32(),
            MessageBandSize = reference.GetProperty("band").GetInt32(),
            MessageDecoderChannels = reference.GetProperty("message_channels").GetInt32(),
            MessageDecoderLayers = reference.GetProperty("message_layers").GetInt32(),
        };
        var engine = new CpuEngine();
        var network = new SilentCipherNetwork<double>(engine, configuration, new List<LayerBase<double>>());
        Func<string, double[]?> Source(string module)
        {
            var shapes = reference.GetProperty("shapes").GetProperty(module).EnumerateObject()
                .ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(v => v.GetInt32()).ToArray(), StringComparer.Ordinal);
            return name => shapes.TryGetValue(name, out var shape) ? ClosedForm(name, shape) : null;
        }
        var (encoderUsed, carrierUsed, messageUsed) = network.Load(Source("enc_c"), Source("dec_c"), Source("dec_m"));
        foreach (var (module, used) in new[] { ("enc_c", encoderUsed), ("dec_c", carrierUsed), ("dec_m", messageUsed) })
            Assert.Empty(reference.GetProperty("shapes").GetProperty(module).EnumerateObject().Select(p => p.Name).Where(n => !used.Contains(n)));

        var key = reference.GetProperty("key").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var audio = Doubles(reference.GetProperty("audio"));
        var expected = Doubles(reference.GetProperty("encoded"));
        var encoded = network.Encode(audio, key, 36);
        Assert.Equal(expected.Length, encoded.Length);
        double scale = expected.Max(Math.Abs), worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - encoded[i]));
        _output.WriteLine($"encoded audio: worst difference {worst:E2} (scale {scale:E2}); watermark peak {audio.Zip(expected, (a, b) => Math.Abs(a - b)).Max():E2}");
        Assert.True(worst <= 1e-9 * scale, $"encoded audio differs by {worst}");

        var expectedScores = reference.GetProperty("scores").EnumerateArray().Select(Doubles).ToArray();
        var scores = network.Scores(expected);
        Assert.Equal(expectedScores.Length, scores.GetLength(0));
        Assert.Equal(expectedScores[0].Length, scores.GetLength(1));
        double scoreScale = expectedScores.SelectMany(r => r).Max(Math.Abs), scoreWorst = 0;
        for (int d = 0; d < expectedScores.Length; d++)
            for (int f = 0; f < expectedScores[d].Length; f++) scoreWorst = Math.Max(scoreWorst, Math.Abs(expectedScores[d][f] - scores[d, f]));
        Assert.True(scoreWorst <= 1e-9 * scoreScale, $"message scores differ by {scoreWorst}");

        foreach (var label in new[] { "audio", "encoded" })
            foreach (var shift in new[] { false, true })
            {
                var want = reference.GetProperty("decodes").GetProperty($"{label}_{(shift ? "shift" : "plain")}");
                var (found, message, _) = network.Decode(label == "audio" ? audio : expected, shift);
                Assert.Equal(want.GetProperty("status").GetBoolean(), found);
                if (found) Assert.Equal(want.GetProperty("messages")[0].EnumerateArray().Select(v => v.GetInt32()).ToArray(), message);
            }
    }

    [Fact(Timeout = 3600000)]
    public async Task ReleasedWeights_ReadBackCsmsKey_WhenPresent()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("SILENTCIPHER_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "enc_c.ckpt"))) return;
        var engine = new CpuEngine();
        var network = new SilentCipherNetwork<double>(engine, new SilentCipherConfiguration(), new List<LayerBase<double>>());
        network.LoadCheckpointDirectory(dir);
        int n = 44100 * 2;
        var random = new Random(3);
        var audio = Enumerable.Range(0, n).Select(i => 0.3 * Math.Sin(2 * Math.PI * 220 * i / 44100) * (0.6 + 0.4 * Math.Sin(2 * Math.PI * 2 * i / 44100))
            + 0.05 * Math.Sin(2 * Math.PI * 1375 * i / 44100) + 0.01 * (random.NextDouble() - 0.5)).ToArray();
        var marked = network.Encode(audio, SilentCipherWatermark.Csm1BKey, SilentCipherWatermark.CsmMessageSdr);
        double noise = Math.Sqrt(audio.Zip(marked, (a, b) => (a - b) * (a - b)).Average()), signal = Math.Sqrt(audio.Average(v => v * v));
        _output.WriteLine($"watermark at {20 * Math.Log10(signal / noise):F1} dB below the audio");
        var (found, message, confidence) = network.Decode(marked, phaseShiftDecoding: false);
        _output.WriteLine($"marked: found {found}, message [{string.Join(", ", message)}], confidence {confidence:F3}");
        Assert.True(found);
        Assert.Equal(SilentCipherWatermark.Csm1BKey, message);
        var (plainFound, plainMessage, _) = network.Decode(audio, phaseShiftDecoding: false);
        _output.WriteLine($"plain: found {plainFound}, message [{string.Join(", ", plainMessage)}]");
        Assert.False(plainFound && plainMessage.SequenceEqual(SilentCipherWatermark.Csm1BKey));
    }
}
