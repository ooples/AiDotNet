using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Codecs;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.Codecs;

/// <summary>
/// NeMo's NanoCodec against NeMo 2.4.0's own modules (tools/reference-data/nanocodec_reference.py): a tiny codec's
/// encoding of 50 samples (zero-padded to whole frames, replicate-padded convolutions), its finite scalar quantization
/// into two codebooks, and the causal decoder's audio (HalfSnake over an odd width, grouped transposed convolutions, the
/// clamped output).
/// </summary>
public class NanoCodecReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "nanocodec_reference.json";
        string relative = Path.Combine("Audio", "Codecs", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    [Fact(Timeout = 300000)]
    public async Task EncodeQuantizeDecode_MatchNeMo()
    {
        await Task.Yield();
        var f = Fixture();
        var enc = f.GetProperty("encoder");
        var dec = f.GetProperty("decoder");
        var q = f.GetProperty("quantizer");
        var configuration = new NanoCodecConfiguration
        {
            EncoderChannels = enc.GetProperty("base_channels").GetInt32(), DownsampleRates = Ints(enc.GetProperty("down_sample_rates")),
            EncodedDim = enc.GetProperty("encoded_dim").GetInt32(), EncoderActivation = enc.GetProperty("activation").GetString()!,
            EncoderPadMode = enc.GetProperty("pad_mode").GetString()!,
            DecoderChannels = dec.GetProperty("base_channels").GetInt32(), UpsampleRates = Ints(dec.GetProperty("up_sample_rates")),
            DecoderActivation = dec.GetProperty("activation").GetString()!, OutputActivation = dec.GetProperty("output_activation").GetString()!,
            QuantizerGroups = q.GetProperty("num_groups").GetInt32(), QuantizerLevels = Ints(q.GetProperty("num_levels_per_group")),
        };
        var codec = new NanoCodec<double>(AiDotNetEngine.Current, configuration, new List<LayerBase<double>>(), new List<LayerBase<double>>());
        var state = f.GetProperty("state");
        var used = new HashSet<string>();
        codec.LoadNemoWeights(name =>
        {
            if (!state.TryGetProperty(name, out var v)) return null;
            used.Add(name);
            return Values(v);
        });
        Assert.Empty(state.EnumerateObject().Select(p => p.Name).Where(n => !used.Contains(n)));

        using var _ = new NoGradScope<double>();
        var audio = Values(f.GetProperty("audio"));
        var input = new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio));
        int length = configuration.SamplesPerFrame * ((audio.Length + configuration.SamplesPerFrame - 1) / configuration.SamplesPerFrame);
        var padded = new double[length];
        Array.Copy(audio, padded, audio.Length);
        var encoded = codec.Encoder.Forward(new Tensor<double>(new[] { 1, 1, length }, new Vector<double>(padded)));
        var expectedEncoding = Values(f.GetProperty("encoded"));
        Assert.Equal(expectedEncoding.Length, encoded.Length);
        for (int i = 0; i < expectedEncoding.Length; i++)
            Assert.True(Math.Abs(expectedEncoding[i] - encoded[i]) <= 1e-10 * Math.Max(1, Math.Abs(expectedEncoding[i])), $"encoded[{i}]: {expectedEncoding[i]} vs {encoded[i]}");

        var codes = codec.Encode(input);
        var expectedCodes = f.GetProperty("codes").EnumerateArray().Select(Ints).ToArray();
        Assert.Equal(expectedCodes.Length, codes.GetLength(0));
        for (int g = 0; g < expectedCodes.Length; g++)
            Assert.Equal(expectedCodes[g], Enumerable.Range(0, codes.GetLength(1)).Select(t => codes[g, t]));

        var decoded = codec.Decode(codes);
        var expectedAudio = Values(f.GetProperty("decoded"));
        Assert.Equal(expectedAudio.Length, decoded.Length);
        for (int i = 0; i < expectedAudio.Length; i++)
            Assert.True(Math.Abs(expectedAudio[i] - decoded[i]) <= 1e-10 * Math.Max(1, Math.Abs(expectedAudio[i])), $"decoded[{i}]: {expectedAudio[i]} vs {decoded[i]}");
    }
}
