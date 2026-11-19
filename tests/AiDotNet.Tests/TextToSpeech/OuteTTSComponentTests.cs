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
            var scores = Values(c["scores"]!);
            var processed = OuteTTSSampler<double>.Process(new Tensor<double>(scores, new[] { scores.Length }), ids, 1.1, 64, 0.4, 40, 0.9, 0.05).ToArray();
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
        var codes = OuteTTSAudio.Encode<double>(codec.Encode, codec.HopLength, Values(dac["input"]!), 24000);
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
        Close(Values(dac["decoded"]!), OuteTTSAudio.Decode<double>(codec.Decode, exact, 24000), "decoded", 1e-7);
    }

    // pyloudnorm's IIRfilter.generate_coefficients (RBJ biquads, divided by a0) and scipy's lfilter in transposed
    // direct form II: the reference the engine K-weighting (FIR zeros + diagonal-pole scans) must reproduce.
    private static (double[] B, double[] A) ReferenceBiquad(double gainDb, double q, double frequency, int rate, bool highShelf)
    {
        double a = Math.Pow(10, gainDb / 40.0), w0 = 2.0 * Math.PI * (frequency / rate), alpha = Math.Sin(w0) / (2.0 * q), c = Math.Cos(w0);
        double b0, b1, b2, a0, a1, a2;
        if (highShelf)
        {
            b0 = a * (a + 1 + (a - 1) * c + 2 * Math.Sqrt(a) * alpha); b1 = -2 * a * (a - 1 + (a + 1) * c);
            b2 = a * (a + 1 + (a - 1) * c - 2 * Math.Sqrt(a) * alpha); a0 = a + 1 - (a - 1) * c + 2 * Math.Sqrt(a) * alpha;
            a1 = 2 * (a - 1 - (a + 1) * c); a2 = a + 1 - (a - 1) * c - 2 * Math.Sqrt(a) * alpha;
        }
        else
        {
            b0 = (1 + c) / 2; b1 = -(1 + c); b2 = (1 + c) / 2; a0 = 1 + alpha; a1 = -2 * c; a2 = 1 - alpha;
        }
        return (new[] { b0 / a0, b1 / a0, b2 / a0 }, new[] { 1.0, a1 / a0, a2 / a0 });
    }

    private static double[] ReferenceLfilter((double[] B, double[] A) f, double[] x)
    {
        var y = new double[x.Length];
        double z1 = 0, z2 = 0;
        for (int n = 0; n < x.Length; n++)
        {
            double output = f.B[0] * x[n] + z1;
            z1 = f.B[1] * x[n] + z2 - f.A[1] * output;
            z2 = f.B[2] * x[n] - f.A[2] * output;
            y[n] = output;
        }
        return y;
    }

    private static double[] TestSignal(int rate, double seconds)
    {
        var x = new double[(int)(rate * seconds)];
        for (int n = 0; n < x.Length; n++)
            x[n] = 0.5 * Math.Sin(2 * Math.PI * 440 * n / rate) + 0.25 * Math.Sin(2 * Math.PI * 30 * n / rate)
                + 0.1 * Math.Sin(2 * Math.PI * 9000 * n / rate + 1.0) + 0.05 * ((n * 7919) % 13 - 6) / 6.0;
        return x;
    }

    [Theory(Timeout = 120000)]
    [InlineData(48000)]
    [InlineData(24000)]
    [InlineData(16000)]
    public async Task KWeighting_MatchesPyloudnormLfilter(int rate)
    {
        await Task.Yield();
        var x = TestSignal(rate, 1.5);
        var expected = ReferenceLfilter(ReferenceBiquad(0.0, 0.5, 38.0, rate, false), ReferenceLfilter(ReferenceBiquad(4.0, 1 / Math.Sqrt(2), 1500.0, rate, true), x));
        var actual = LoudnessMeter.KWeighted(AiDotNet.Tensors.Engines.AiDotNetEngine.Current,
            new Tensor<double>(new[] { x.Length }, new Vector<double>(x)), rate, clamp: false).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int n = 0; n < expected.Length; n++)
            Assert.True(Math.Abs(expected[n] - actual[n]) <= 1e-9, $"rate {rate}, sample {n}: {expected[n]} vs {actual[n]}");
    }

    // |H(e^{jw})|² of a biquad at a frequency.
    private static double PowerGain((double[] B, double[] A) f, double frequency, int rate)
    {
        var z = System.Numerics.Complex.Exp(new System.Numerics.Complex(0, -2 * Math.PI * frequency / rate));
        var h = (f.B[0] + f.B[1] * z + f.B[2] * z * z) / (f.A[0] + f.A[1] * z + f.A[2] * z * z);
        return h.Magnitude * h.Magnitude;
    }

    // A steady 0 dBFS sine reads −0.691 + 10 log10(½ |H(f)|²), H the K-weighting. ITU-R BS.1770's own coefficients
    // give −3.01 LKFS at 997 Hz; pyloudnorm's RBJ biquads (ported here) have 0.042 dB less gain there, so the meter
    // reads −3.0517 at 48 kHz, as pyloudnorm does (its own test pins −3.0523 for a 1 kHz sine at 44.1 kHz).
    [Fact(Timeout = 120000)]
    public async Task IntegratedLoudness_FullScaleSineReadsTheKWeightingGain()
    {
        await Task.Yield();
        const int rate = 48000;
        var x = new double[rate * 3];
        for (int n = 0; n < x.Length; n++) x[n] = Math.Sin(2 * Math.PI * 997 * n / rate);
        double lufs = LoudnessMeter.IntegratedLoudness(x, rate);
        double expected = -0.691 + 10 * Math.Log10(0.5
            * PowerGain(ReferenceBiquad(4.0, 1 / Math.Sqrt(2), 1500.0, rate, true), 997, rate)
            * PowerGain(ReferenceBiquad(0.0, 0.5, 38.0, rate, false), 997, rate));
        Assert.True(Math.Abs(expected - -3.0517) <= 1e-4, $"analytic {expected} LUFS");
        Assert.True(Math.Abs(lufs - expected) <= 1e-3, $"{lufs} LUFS vs {expected}");
        double generic = LoudnessMeter.IntegratedLoudness(new Tensor<float>(new[] { x.Length }, new Vector<float>(x.Select(v => (float)v).ToArray())), rate);
        Assert.True(Math.Abs(generic - lufs) <= 1e-3, $"float {generic} vs double {lufs}");
    }

    // A draw over 5,000 equal scores is uniform. In Half the running probability stalled at 0.5 (2e-4 is under half
    // an ulp there), so every u above it fell through to the last token: about half of all draws returned 4,999.
    [Fact(Timeout = 120000)]
    public async Task Sample_InHalf_IsNotPulledToTheLastToken()
    {
        await Task.Yield();
        var scores = new AiDotNet.Tensors.LinearAlgebra.Tensor<Half>(new Half[5000], new[] { 5000 });
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(5);
        var draws = Enumerable.Range(0, 400).Select(_ => OuteTTSSampler<Half>.Sample(scores, random)).ToArray();
        Assert.True(draws.Count(d => d == 4999) <= 3, $"{draws.Count(d => d == 4999)} of 400 draws on the last token");
        Assert.True(draws.Distinct().Count() >= 300, $"{draws.Distinct().Count()} distinct draws");
    }

    [Theory(Timeout = 120000)]
    [InlineData(0.4)]
    [InlineData(0.0)]
    public async Task SamplingSession_ReusesScratchAndPreservesTheCallerAndSeededSequence(double temperature)
    {
        await Task.Yield();
        const int vocabulary = 157760, steps = 16;
        var original = Enumerable.Range(0, vocabulary).Select(i => (i - vocabulary / 2) * 0.0001f).ToArray();
        var logits = new Tensor<float>((float[])original.Clone(), new[] { 1, vocabulary });
        var options = new OuteTTSOptions { Temperature = temperature };
        var expected = new List<int> { 2, 5, 9 };
        var referenceRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(37);
        // The unchanged processors define the sequence; no generation-wide arena retains their temporaries here.
        using (AiDotNet.Tensors.Helpers.TensorArena.Suspend())
            for (int i = 0; i < steps; i++)
            {
                var scores = OuteTTSSampler<float>.Process(logits, expected, options.RepetitionPenalty,
                    options.RepetitionRange, temperature > 0 ? temperature : 1.0,
                    temperature > 0 ? options.TopK : 0, temperature > 0 ? options.TopP : 1.0,
                    temperature > 0 ? options.MinP : 0);
                expected.Add(temperature > 0 ? OuteTTSSampler<float>.Sample(scores, referenceRandom)
                    : OuteTTSSampler<float>.Greedy(scores));
            }

        using var outer = AiDotNet.Tensors.Helpers.TensorArena.Create();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        logits = engine.TensorAddScalar(logits, 0f);
        var sentinel = engine.TensorAddScalar(new Tensor<float>(new[] { 3f, -4f }, new[] { 2 }), 1f);
        var sentinelValues = sentinel.ToArray();
        long outerPeak = outer.PeakBackingBytes;
        Assert.True(outerPeak > 0, "The input and sentinel must exercise live caller-arena storage.");
        var actual = new List<int> { 2, 5, 9 };
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(37);
        var session = new OuteTTSSamplingSession<float>();
        long firstPeak = 0;
        using (session)
            for (int i = 0; i < steps; i++)
            {
                actual.Add(session.Select(logits, actual, options, random));
                Assert.Same(outer, AiDotNet.Tensors.Helpers.TensorArena.Current);
                Assert.Equal(outerPeak, outer.PeakBackingBytes);
                Assert.Equal(sentinelValues, sentinel.ToArray());
                Assert.Equal(original, logits.ToArray());
                if (i == 0) firstPeak = session.PeakBackingBytes;
                Assert.True(firstPeak > 0, "The regression must measure real sampling scratch allocations.");
                Assert.Equal(firstPeak, session.PeakBackingBytes);
            }
        Assert.Equal(expected, actual);
        Assert.Equal(referenceRandom.NextDouble(), random.NextDouble());
        Assert.Same(outer, AiDotNet.Tensors.Helpers.TensorArena.Current);
        Assert.Equal(sentinelValues, sentinel.ToArray());
        Assert.Throws<ObjectDisposedException>(() => session.Select(logits, actual, options, random));
    }

    [Fact]
    public void SamplingSession_RestoresCallerAfterFailureAndCanContinue()
    {
        using var outer = AiDotNet.Tensors.Helpers.TensorArena.Create();
        var logits = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 1, 4 });
        var original = logits.ToArray();
        var options = new OuteTTSOptions();
        var ids = new List<int> { 1 };
        using (var session = new OuteTTSSamplingSession<float>())
        {
            Assert.Throws<InvalidOperationException>(() => session.Select(logits, ids, options,
                new FailingSamplingRandom(outer)));
            Assert.Same(outer, AiDotNet.Tensors.Helpers.TensorArena.Current);
            Assert.Equal(original, logits.ToArray());
            int token = session.Select(logits, ids, options, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(4));
            Assert.InRange(token, 0, 3);
            Assert.Same(outer, AiDotNet.Tensors.Helpers.TensorArena.Current);
        }
        Assert.Same(outer, AiDotNet.Tensors.Helpers.TensorArena.Current);
        using (AiDotNet.Tensors.Helpers.TensorArena.Suspend())
        {
            using (var session = new OuteTTSSamplingSession<float>())
            {
                session.Select(logits, ids, options, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(4));
                Assert.Null(AiDotNet.Tensors.Helpers.TensorArena.Current);
            }
            Assert.Null(AiDotNet.Tensors.Helpers.TensorArena.Current);
        }
        Assert.Same(outer, AiDotNet.Tensors.Helpers.TensorArena.Current);
    }

    private sealed class FailingSamplingRandom : Random
    {
        private readonly AiDotNet.Tensors.Helpers.TensorArena _caller;
        internal FailingSamplingRandom(AiDotNet.Tensors.Helpers.TensorArena caller) => _caller = caller;
        public override double NextDouble()
        {
            Assert.NotNull(AiDotNet.Tensors.Helpers.TensorArena.Current);
            Assert.NotSame(_caller, AiDotNet.Tensors.Helpers.TensorArena.Current);
            throw new InvalidOperationException("Injected sampling failure.");
        }
    }
}
