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
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MeloTTS's synthesizer against myshell-ai/MeloTTS's own SynthesizerTrn in float64 (tools/reference-data/melo_reference.py):
/// one training forward with every random draw fixed — the alignment, both priors, the flow, both duration losses, the KL,
/// the duration discriminator and the decoded segment — one SGD step on their combined loss, and generation.
/// </summary>
public class MeloReferenceTests
{
    private static readonly Lazy<JsonElement> Data = new(() =>
    {
        const string name = "melo_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static Func<string, double[]?> Reader(JsonElement state)
    {
        var dict = state.EnumerateObject().ToDictionary(p => p.Name, p => Values(p.Value));
        return name => dict.TryGetValue(name, out var v) ? v : null;
    }

    private static Tensor<double> TensorOf(double[] values, params int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < values.Length; i++) t[i] = values[i];
        return t;
    }

    private static void AssertClose(double[] expected, Tensor<double> actual, string what, double tolerance = 1e-9)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])), $"{what} {i}: {expected[i]} vs {actual[i]}");
    }

    private static MeloConfiguration Configuration(JsonElement c) => new(
        Vocabulary: c.GetProperty("vocab").GetInt32(), Tones: c.GetProperty("tones").GetInt32(), Languages: c.GetProperty("languages").GetInt32(),
        SpectrogramChannels: c.GetProperty("spec").GetInt32(), SegmentFrames: c.GetProperty("segment").GetInt32(),
        InterChannels: c.GetProperty("inter").GetInt32(), HiddenChannels: c.GetProperty("hidden").GetInt32(),
        FilterChannels: c.GetProperty("filter").GetInt32(), Heads: c.GetProperty("heads").GetInt32(), Layers: c.GetProperty("layers").GetInt32(),
        KernelSize: c.GetProperty("kernel").GetInt32(), Dropout: 0, ResblockKernels: Ints(c.GetProperty("kernels")),
        ResblockDilations: c.GetProperty("dilations").EnumerateArray().Select(Ints).ToArray(), UpsampleRates: Ints(c.GetProperty("rates")),
        UpsampleInitialChannels: c.GetProperty("initial").GetInt32(), UpsampleKernels: Ints(c.GetProperty("up_kernels")),
        Speakers: c.GetProperty("speakers").GetInt32(), SpeakerChannels: c.GetProperty("gin").GetInt32(),
        FlowCouplings: c.GetProperty("flows").GetInt32(), FlowLayers: c.GetProperty("flow_layers").GetInt32());

    // The trainable tensors only: an unnormalized convolution's gain is never read, so it is skipped.
    private static void AssertLayersMatch(List<LayerBase<double>> actual, List<LayerBase<double>> expected)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int l = 0; l < actual.Count; l++)
        {
            var a = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new[] { actual[l] });
            var e = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new[] { expected[l] });
            Assert.Equal(e.Count, a.Count);
            for (int t = 0; t < e.Count; t++)
            {
                if (actual[l] is NormedConv1DLayer<double> { Normalization: ConvolutionNormalization.None } && t == 1) continue;
                for (int i = 0; i < e[t].Length; i++)
                    Assert.True(Math.Abs(e[t][i] - a[t][i]) < 1e-9, $"layer {l} ({actual[l].GetType().Name}) tensor {t} value {i}: {e[t][i]} vs {a[t][i]}");
            }
        }
    }

    [Fact(Timeout = 300000)]
    public async Task TrainingForwardSgdStepAndGeneration_MatchMeloTTS()
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var k = Data.Value;
        var c = k.GetProperty("config");
        var config = Configuration(c);
        var layers = new List<LayerBase<double>>();
        var net = new MeloSynthesizer<double>(engine, layers, config);
        net.LoadTorch(Reader(k.GetProperty("initial")));
        var discriminatorLayers = new List<LayerBase<double>>();
        var discriminator = new MeloDurationDiscriminator<double>(engine, discriminatorLayers, config.HiddenChannels, config.FilterChannels, 3, 0.0, config.SpeakerChannels);
        var discriminatorState = Reader(k.GetProperty("duration_discriminator"));
        discriminator.LoadTorch(name => discriminatorState(name) ?? throw new KeyNotFoundException(name), "");

        int tokens = c.GetProperty("tokens").GetInt32(), frames = c.GetProperty("frames").GetInt32();
        var symbols = Ints(k.GetProperty("x"));
        var tones = Ints(k.GetProperty("tone"));
        var languages = Ints(k.GetProperty("language"));
        var bert = TensorOf(Values(k.GetProperty("bert")), 1, 1024, tokens);
        var jaBert = TensorOf(Values(k.GetProperty("ja_bert")), 1, 768, tokens);
        var spec = TensorOf(Values(k.GetProperty("spec")), 1, config.SpectrogramChannels, frames);
        var masValues = Values(k.GetProperty("mas_noise"));
        var mas = new double[frames, tokens];
        for (int f = 0; f < frames; f++)
            for (int l = 0; l < tokens; l++) mas[f, l] = masValues[f * tokens + l];

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var o = net.Forward(symbols, tones, languages, bert, jaBert, spec, c.GetProperty("sid").GetInt32(), c.GetProperty("mas_noise").GetDouble(),
                training: true, new Random(0), TensorOf(Values(k.GetProperty("posterior_noise")), 1, config.InterChannels, frames), mas,
                TensorOf(Values(k.GetProperty("sdp_noise")), 1, 2, tokens), k.GetProperty("slice_start").GetInt32(), durationTraining: false);
            var attn = Values(k.GetProperty("attn"));                                                          // [T, L]
            for (int l = 0, start = 0; l < tokens; start += o.Durations[l], l++)
                for (int f = 0; f < frames; f++)
                    Assert.True(attn[f * tokens + l] == (f >= start && f < start + o.Durations[l] ? 1.0 : 0.0), $"alignment frame {f} token {l}");
            AssertClose(Values(k.GetProperty("m_p")), o.PriorMean, "m_p");
            AssertClose(Values(k.GetProperty("logs_p")), o.PriorLogScale, "logs_p");
            AssertClose(Values(k.GetProperty("m_q")), o.PosteriorMean, "m_q");
            AssertClose(Values(k.GetProperty("logs_q")), o.PosteriorLogScale, "logs_q");
            AssertClose(Values(k.GetProperty("z")), o.Z, "z");
            AssertClose(Values(k.GetProperty("z_p")), o.ZPrior, "z_p");
            AssertClose(Values(k.GetProperty("logw")), o.PredictedLogDurations, "logw");
            AssertClose(Values(k.GetProperty("logw_")), o.AlignedLogDurations, "logw_");
            Assert.True(Math.Abs(o.DurationLoss[0] - k.GetProperty("l_length").GetDouble()) < 1e-9, $"l_length {o.DurationLoss[0]}");
            var kl = net.KlLoss(o);
            Assert.True(Math.Abs(kl[0] - k.GetProperty("loss_kl").GetDouble()) < 1e-9, $"kl {kl[0]}");
            // train.py: the duration discriminator without the speaker, predicted durations as dur_r, the aligned ones as dur_hat.
            var (real, generated) = discriminator.Forward(o.Hidden, null, o.PredictedLogDurations, o.AlignedLogDurations, false, new Random(0));
            AssertClose(Values(k.GetProperty("dur_real")), real, "duration discriminator (dur_r)");
            AssertClose(Values(k.GetProperty("dur_generated")), generated, "duration discriminator (dur_hat)");
            AssertClose(Values(k.GetProperty("o")), o.Waveform, "waveform");

            var oneMinus = engine.TensorSubtract(engine.TensorAddScalar(new Tensor<double>(generated._shape), 1.0), generated);
            var durationGenerator = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(oneMinus, oneMinus), new[] { 0, 1 }, keepDims: false), 1.0 / generated.Length);
            var weights = TensorOf(Values(k.GetProperty("weights")), o.Waveform.Shape.ToArray());
            var total = engine.TensorAdd(engine.TensorAdd(o.DurationLoss, engine.Reshape(kl, new[] { 1 })),
                engine.TensorAdd(engine.Reshape(durationGenerator, new[] { 1 }), engine.Reshape(engine.ReduceSum(engine.TensorMultiply(o.Waveform, weights), new[] { 0, 1, 2 }, keepDims: false), new[] { 1 })));
            Assert.True(Math.Abs(total[0] - k.GetProperty("loss").GetDouble()) < 1e-9, $"loss {total[0]} vs {k.GetProperty("loss").GetDouble()}");
            grads = tape.ComputeGradients(total, sources: parameters);
        }
        double lr = k.GetProperty("lr").GetDouble();
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= lr * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        var expectedLayers = new List<LayerBase<double>>();
        new MeloSynthesizer<double>(engine, expectedLayers, config).LoadTorch(Reader(k.GetProperty("final")));
        AssertLayersMatch(layers, expectedLayers);

        var inference = k.GetProperty("infer");
        var priorNoise = Values(inference.GetProperty("prior_noise"));
        var (waveform, durations) = net.Infer(symbols, tones, languages, bert, jaBert, c.GetProperty("sid").GetInt32(), new Random(0),
            noiseScale: 0.667, lengthScale: 1.0, durationNoiseScale: 0.8, sdpRatio: 0.2,
            durationNoise: TensorOf(Values(inference.GetProperty("sdp_noise")).Select(v => v * 0.8).ToArray(), 1, 2, tokens),
            priorNoise: n => TensorOf(priorNoise, 1, config.InterChannels, n));
        Assert.Equal(inference.GetProperty("frames").GetInt32(), durations.Sum());
        AssertClose(Values(inference.GetProperty("audio")), waveform, "generated audio", 1e-8);
    }
}
