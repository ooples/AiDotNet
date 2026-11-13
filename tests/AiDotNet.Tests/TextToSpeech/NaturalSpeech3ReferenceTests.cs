using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// NaturalSpeech 3's FACodec against Amphion's released checkpoints (amphion/naturalspeech3_facodec) in float64
/// (tools/reference-data/facodec_checkpoint_reference.py): v1's encoder, factorized codes, timbre, heads and decoder, v2's
/// mel prosody branch, and the redecoder's voice conversion. Set FACODEC_DIR to the directory holding the five .bin files
/// and the script's JSON.
/// </summary>
public class NaturalSpeech3ReferenceTests
{
    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static Tensor<double> T(JsonElement e, params int[] shape) => new(shape, new Vector<double>(Doubles(e)));

    private static void Close(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= tolerance * (1 + Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual.GetFlat(i)}");
    }

    private static Func<string, double[]?> Reader(string path)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        return name => file.TryGetValue(name, out var t) ? t.ToVector().Select(v => (double)v).ToArray() : null;
    }

    private static int[,] Codes(JsonElement e)
    {
        var rows = e.EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
        var codes = new int[rows.Length, rows[0].Length];
        for (int q = 0; q < rows.Length; q++)
            for (int f = 0; f < rows[0].Length; f++) codes[q, f] = rows[q][f];
        return codes;
    }

    private static FACodecModel<double> Build(bool v2, bool redecoder)
    {
        var layers = new List<LayerBase<double>>();
        return new FACodecModel<double>(AiDotNetEngine.Current, new FACodecConfiguration { MelProsody = v2, Redecoder = redecoder }, new Random(1),
            layers, layers, layers, layers);
    }

    [Fact(Timeout = 3600000)]
    public async Task ReleasedFACodec_MatchesAmphion()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("FACODEC_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "facodec_checkpoint_reference.json"))) return;
        var r = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "facodec_checkpoint_reference.json"))).RootElement;
        using var _ = new NoGradScope<double>();
        var engine = AiDotNetEngine.Current;

        var codec = Build(v2: false, redecoder: true);
        codec.LoadEncoder(Reader(Path.Combine(dir, "ns3_facodec_encoder.bin")));
        codec.LoadDecoder(Reader(Path.Combine(dir, "ns3_facodec_decoder.bin")));
        codec.LoadRedecoder(Reader(Path.Combine(dir, "ns3_facodec_redecoder.bin")));
        var v1 = r.GetProperty("v1");
        var a = Doubles(r.GetProperty("wave_a"));
        var enc = codec.Encoder.Forward(new Tensor<double>(new[] { 1, 1, a.Length }, new Vector<double>(a)));
        Close(Doubles(v1.GetProperty("enc")), enc, 1e-8, "enc");
        var factors = codec.Factorize(enc, null, false, null);
        var expectedCodes = Codes(v1.GetProperty("codes"));
        for (int q = 0; q < expectedCodes.GetLength(0); q++)
            for (int f = 0; f < expectedCodes.GetLength(1); f++) Assert.Equal(expectedCodes[q, f], factors.Codes[q, f]);
        var quantized = v1.GetProperty("quantized").EnumerateArray().ToArray();
        for (int g = 0; g < 3; g++) Close(Doubles(quantized[g]), factors.Groups[g], 1e-8, $"quantized[{g}]");
        Close(Doubles(v1.GetProperty("spk")), factors.Speaker, 1e-8, "spk");
        var emb = factors.Sum(engine);
        Close(Doubles(v1.GetProperty("emb")), emb, 1e-8, "emb");
        Close(Doubles(v1.GetProperty("audio")), codec.Decode(emb, factors.Speaker), 1e-6, "audio");

        var f0 = codec.F0Predictor.Forward(factors.Groups[0]);
        Close(Doubles(v1.GetProperty("f0")), f0[0], 1e-8, "f0");
        Close(Doubles(v1.GetProperty("uv")), f0[1], 1e-8, "uv");
        var phone = codec.PhonePredictor.Forward(factors.Groups[1])[0];
        var head = Doubles(v1.GetProperty("phone_head"));
        for (int f = 0; f < head.Length / 8; f++)
            for (int k = 0; k < 8; k++) Assert.True(Math.Abs(head[f * 8 + k] - phone[f, k]) <= 1e-8 * (1 + Math.Abs(head[f * 8 + k])), $"phone[{f},{k}]");
        var resF0 = codec.DetailF0Predictor!.Forward(factors.Groups[2]);
        Close(Doubles(v1.GetProperty("res_f0")), resF0[0], 1e-8, "res_f0");
        Close(Doubles(v1.GetProperty("res_uv")), resF0[1], 1e-8, "res_uv");
        // The timbre head reads z_p + z_c + m·z_d with the release's random detail mask m; it matches one of the two.
        var timbreHead = Doubles(v1.GetProperty("x_timbre_head"));
        var full = codec.TimbrePredictor!.Forward(emb)[0];
        var dropped = codec.TimbrePredictor.Forward(engine.TensorAdd(factors.Groups[0], factors.Groups[1]))[0];
        bool Matches(Tensor<double> t) => Enumerable.Range(0, 8).All(k => Math.Abs(timbreHead[k] - t[0, k]) <= 1e-8 * (1 + Math.Abs(timbreHead[k])));
        Assert.True(Matches(full) || Matches(dropped), "x_timbre");

        // Voice conversion through the redecoder: utterance a's prosody and content in b's timbre.
        var b = Doubles(r.GetProperty("wave_b"));
        var factorsB = codec.Factorize(codec.Encoder.Forward(new Tensor<double>(new[] { 1, 1, b.Length }, new Vector<double>(b))), null, false, null);
        var re = r.GetProperty("redecoder");
        Close(Doubles(re.GetProperty("spk_b")), factorsB.Speaker, 1e-8, "spk_b");
        var converted = codec.RedecoderEmbed(factors.Codes, factorsB.Speaker, detail: false);
        Close(Doubles(re.GetProperty("emb")), converted, 1e-8, "redecoder emb");
        Close(Doubles(re.GetProperty("audio")), codec.Redecode(converted, factorsB.Speaker), 1e-6, "redecoder audio");

        var codec2 = Build(v2: true, redecoder: false);
        codec2.LoadEncoder(Reader(Path.Combine(dir, "ns3_facodec_encoder_v2.bin")));
        codec2.LoadDecoder(Reader(Path.Combine(dir, "ns3_facodec_decoder_v2.bin")));
        var v2 = r.GetProperty("v2");
        var enc2 = codec2.Encoder.Forward(new Tensor<double>(new[] { 1, 1, a.Length }, new Vector<double>(a)));
        Close(Doubles(v2.GetProperty("enc")), enc2, 1e-8, "v2 enc");
        var prosody = codec2.ProsodyFeature(new Tensor<double>(a, new[] { a.Length }));
        Close(Doubles(v2.GetProperty("prosody")), prosody, 1e-7, "v2 prosody");
        var factors2 = codec2.Factorize(enc2, prosody, false, null);
        var codes2 = Codes(v2.GetProperty("codes"));
        for (int q = 0; q < codes2.GetLength(0); q++)
            for (int f = 0; f < codes2.GetLength(1); f++) Assert.Equal(codes2[q, f], factors2.Codes[q, f]);
        Close(Doubles(v2.GetProperty("spk")), factors2.Speaker, 1e-8, "v2 spk");
        Close(Doubles(v2.GetProperty("audio")), codec2.Decode(factors2.Sum(engine), factors2.Speaker), 1e-6, "v2 audio");
    }

    private static readonly Lazy<JsonElement> Tiny = new(() =>
    {
        const string name = "facodec_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    /// <summary>
    /// FACodec at a tiny configuration against Amphion's own modules with random weights (tools/reference-data/
    /// facodec_reference.py): the encoder, factorized codes, timbre, heads, decoder, code embeddings and the redecoder.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task TinyFACodec_MatchesAmphion()
    {
        await Task.Yield();
        var r = Tiny.Value;
        var state = r.GetProperty("state");
        Func<string, double[]?> Prefixed(string prefix) => n => state.TryGetProperty(prefix + n, out var v) ? Doubles(v) : null;
        var layers = new List<LayerBase<double>>();
        var codec = new FACodecModel<double>(AiDotNetEngine.Current, new FACodecConfiguration
        {
            EncoderDim = 4, EncoderRates = new[] { 2, 2 }, Dim = 16, DecoderInitial = 16, DecoderRates = new[] { 2, 2 }, CodebookDim = 4, CodebookSize = 16,
            TimbreLayers = 1, TimbreHeads = 2, TimbreFilter = 32, Phones = 12, Speakers = 7, Redecoder = true, RedecoderInitial = 16,
        }, new Random(1), layers, layers, layers, layers);
        codec.LoadEncoder(Prefixed("encoder."));
        codec.LoadDecoder(Prefixed("decoder."));
        codec.LoadRedecoder(Prefixed("redecoder."));
        foreach (var q in codec.Quantizers) q.SetTrainingMode(false);
        using var _ = new NoGradScope<double>();
        var engine = AiDotNetEngine.Current;
        var a = Doubles(r.GetProperty("wave_a"));
        var enc = codec.Encoder.Forward(new Tensor<double>(new[] { 1, 1, a.Length }, new Vector<double>(a)));
        Close(Doubles(r.GetProperty("enc")), enc, 1e-10, "enc");
        var f = codec.Factorize(enc, null, false, null);
        var codes = Codes(r.GetProperty("codes"));
        for (int q = 0; q < codes.GetLength(0); q++)
            for (int t = 0; t < codes.GetLength(1); t++) Assert.Equal(codes[q, t], f.Codes[q, t]);
        var quantized = r.GetProperty("quantized").EnumerateArray().ToArray();
        for (int g = 0; g < 3; g++) Close(Doubles(quantized[g]), f.Groups[g], 1e-10, $"quantized[{g}]");
        Close(Doubles(r.GetProperty("spk")), f.Speaker, 1e-10, "spk");
        var emb = f.Sum(engine);
        Close(Doubles(r.GetProperty("emb")), emb, 1e-10, "emb");
        Close(Doubles(r.GetProperty("vq2emb_no_detail")), codec.Embed(f.Codes, detail: false), 1e-10, "vq2emb");
        Close(Doubles(r.GetProperty("audio")), codec.Decode(emb, f.Speaker), 1e-9, "audio");
        var f0 = codec.F0Predictor.Forward(f.Groups[0]);
        Close(Doubles(r.GetProperty("f0")), f0[0], 1e-10, "f0");
        Close(Doubles(r.GetProperty("uv")), f0[1], 1e-10, "uv");
        Close(Doubles(r.GetProperty("phone")), codec.PhonePredictor.Forward(f.Groups[1])[0], 1e-10, "phone");
        var resF0 = codec.DetailF0Predictor!.Forward(f.Groups[2]);
        Close(Doubles(r.GetProperty("res_f0")), resF0[0], 1e-10, "res_f0");
        Close(Doubles(r.GetProperty("res_uv")), resF0[1], 1e-10, "res_uv");
        Close(Doubles(r.GetProperty("res_phone")), codec.DetailPhonePredictor!.Forward(f.Groups[2])[0], 1e-10, "res_phone");
        Close(Doubles(r.GetProperty("x_timbre")), codec.TimbrePredictor!.Forward(emb)[0], 1e-10, "x_timbre");

        var b = Doubles(r.GetProperty("wave_b"));
        var fb = codec.Factorize(codec.Encoder.Forward(new Tensor<double>(new[] { 1, 1, b.Length }, new Vector<double>(b))), null, false, null);
        Close(Doubles(r.GetProperty("spk_b")), fb.Speaker, 1e-10, "spk_b");
        var converted = codec.RedecoderEmbed(f.Codes, fb.Speaker, detail: false);
        Close(Doubles(r.GetProperty("redecoder_emb")), converted, 1e-10, "redecoder emb");
        Close(Doubles(r.GetProperty("redecoder_audio")), codec.Redecode(converted, fb.Speaker), 1e-9, "redecoder audio");
    }
}
