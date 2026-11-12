using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Vocos decoding EnCodec codes (VALL-E 2's decoder) reproduces the vocos package: summed codebook vectors, the ConvNeXt
/// backbone with bandwidth-adaptive layer norms, and the ISTFT head with "same" padding.
/// </summary>
/// <remarks><c>ReferenceData/vocos_encodec_reference.json</c> holds a tiny seeded model's float32 state dictionary (the
/// released checkpoint's names), codes of 8 codebooks and the float64 audio
/// (<c>tools/reference-data/vocos_encodec_reference.py</c>).</remarks>
public class VocosEncodecParityTests
{
    private static JObject Reference()
    {
        const string fileName = "vocos_encodec_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    [Fact(Timeout = 120000)]
    public async Task Decode_MatchesTheReference()
    {
        await Task.Yield();
        var fixture = Reference();
        var c = fixture["config"]!;
        // 32-code books at 150 frames a second: 12 kbps is 16 codebooks (the table), 6 kbps is 8 (bandwidth class 2).
        var config = new VocosEncodecConfiguration(new[] { 1.5, 3.0, 6.0, 12.0 }, CodebookSize: (int)c["bins"]!,
            LatentDim: (int)c["latent"]!, FrameRate: 150, Dim: (int)c["dim"]!, IntermediateDim: (int)c["intermediate"]!,
            Layers: (int)c["layers"]!, FftSize: (int)c["n_fft"]!, HopSize: (int)c["hop"]!);
        var decoder = new VocosEncodecDecoder<double>(AiDotNetEngine.Current, new Random(0), config);

        string path = Path.Combine(Path.GetTempPath(), $"vocos_encodec_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            decoder.LoadTorchWeights((name, shape) =>
            {
                var tensor = weights[name];
                Assert.Equal(shape.Aggregate(1, (a, b) => a * b), tensor.Length);
                used.Add(name);
                return tensor.ToVector().Select(v => (double)v).ToArray();
            });
            Assert.Equal(weights.Keys.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
        }
        finally
        {
            File.Delete(path);
        }

        var rows = fixture["codes"]!.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        var codes = new int[rows.Length, rows[0].Length];
        for (int q = 0; q < rows.Length; q++)
            for (int f = 0; f < rows[0].Length; f++) codes[q, f] = rows[q][f];
        var audio = decoder.Decode(codes);
        var expected = fixture["audio"]!.Select(v => (double)v).ToArray();
        Assert.Equal(expected.Length, audio.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - audio[i]) <= 1e-8 * Math.Max(1, Math.Abs(expected[i])),
                $"audio[{i}]: reference {expected[i]:R}, port {audio[i]:R}.");
    }
}
