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
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CosyVoice's flow model and HiFT vocoder against the official FunAudioLLM/CosyVoice modules in float64
/// (tools/reference-data/cosyvoice_reference.py), for what the per-module fixtures do not cover: the paper's
/// absolute-error flow loss and the code's squared one, each with an SGD step; generation through the regulator's
/// inference path (a short and a long text); HiFT trained through its source merge; and HiFT's GAN steps.
/// </summary>
public class CosyVoiceReferenceTests
{
    private static readonly Lazy<JsonElement> Data = new(() =>
    {
        const string name = "cosyvoice_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();
    private static int[][] Jagged(JsonElement e) => e.EnumerateArray().Select(Ints).ToArray();

    private static Dictionary<string, double[]> StateDict(JsonElement state) =>
        state.EnumerateObject().ToDictionary(p => p.Name, p => Values(p.Value));

    private static Func<string, int[], double[]> Reader(JsonElement state)
    {
        var dict = StateDict(state);
        return (name, _) => dict.TryGetValue(name, out var v) ? v : throw new KeyNotFoundException(name);
    }

    private static Tensor<double> TensorOf(double[] values, params int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < values.Length; i++) t[i] = values[i];
        return t;
    }

    private static void AssertClose(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < tolerance, $"{what} {i}: {expected[i]} vs {actual[i]}");
    }

    // The trainable tensors only: GetParameters also carries buffers (a normed convolution's spectral-norm vectors),
    // which each instance draws for itself; an unnormalized convolution's gain is never read, so it is skipped.
    private static void AssertLayersMatch(List<LayerBase<double>> actual, List<LayerBase<double>> expected, string what)
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
                Assert.Equal(e[t].Length, a[t].Length);
                for (int i = 0; i < e[t].Length; i++)
                    Assert.True(Math.Abs(e[t][i] - a[t][i]) < 1e-9,
                        $"{what}: layer {l} ({actual[l].GetType().Name}) tensor {t} value {i}: {e[t][i]} vs {a[t][i]}");
            }
        }
    }

    private static void Sgd(IEngine engine, IReadOnlyList<Tensor<double>> parameters, Dictionary<Tensor<double>, Tensor<double>> grads, double lr)
    {
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= lr * g[i];
            engine.InvalidatePersistentTensor(p);
        }
    }

    // ------------------------------------------------------------------ flow

    private static CosyVoiceFlow<double> Flow(IEngine engine, List<LayerBase<double>> layers, JsonElement c, CosyVoiceFlowLoss loss)
    {
        int input = c.GetProperty("input").GetInt32(), mel = c.GetProperty("mel").GetInt32();
        return new CosyVoiceFlow<double>(engine, layers, new CosyVoiceFlowConfiguration(
            VocabSize: c.GetProperty("vocab").GetInt32(), InputSize: input, OutputSize: mel, SpeakerDim: c.GetProperty("spk").GetInt32(),
            InputFrameRate: 50, SampleRate: 22050, HopSize: 256,
            Encoder: new WenetEncoderConfiguration(input, input, c.GetProperty("heads").GetInt32(), c.GetProperty("units").GetInt32(),
                c.GetProperty("blocks").GetInt32(), WenetInputLayer.Linear, Swish: true, Conformer: true, Dropout: 0, PositionalDropout: 0),
            RegulatorStages: 4, DecoderChannels: Ints(c.GetProperty("channels")), DecoderHeads: c.GetProperty("est_heads").GetInt32(),
            DecoderHeadDim: c.GetProperty("head_dim").GetInt32(), DecoderBlocksPerLevel: c.GetProperty("est_blocks").GetInt32(),
            DecoderMidBlocks: c.GetProperty("est_mid").GetInt32(), SigmaMin: 1e-6, TrainingCfgRate: 0.2, InferenceCfgRate: 0.7,
            Timesteps: 10, Loss: loss), new Random(0));
    }

    [Theory(Timeout = 300000)]
    [InlineData("squared")]
    [InlineData("absolute")]
    public async Task Flow_LossAndOneSgdStep_MatchTheOfficialCode(string form)
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var k = Data.Value.GetProperty("flow");
        var c = k.GetProperty("config");
        var loss = form == "squared" ? CosyVoiceFlowLoss.SquaredError : CosyVoiceFlowLoss.AbsoluteError;
        var layers = new List<LayerBase<double>>();
        var flow = Flow(engine, layers, c, loss);
        flow.LoadTorchWeights(Reader(k.GetProperty("initial")), "");
        int frames = k.GetProperty("frames").GetInt32(), mel = c.GetProperty("mel").GetInt32(), prompt = k.GetProperty("prompt").GetInt32();
        var featValues = Values(k.GetProperty("mel"));                                                       // [frames, mel]
        var target = new Tensor<double>(new[] { 1, mel, frames });
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < mel; m++) target[0, m, f] = featValues[f * mel + m];
        var condition = new Tensor<double>(new[] { 1, mel, frames });
        for (int f = 0; f < prompt; f++)
            for (int m = 0; m < mel; m++) condition[0, m, f] = target[0, m, f];
        var noise = TensorOf(Values(k.GetProperty("noise")), 1, mel, frames);
        var xvector = TensorOf(Values(k.GetProperty("embedding")), c.GetProperty("spk").GetInt32());

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        double value;
        using (var tape = new GradientTape<double>())
        {
            var mu = flow.Stretch(flow.Encode(Ints(k.GetProperty("tokens")), training: true, new Random(0)), frames);
            var l = flow.Loss(target, mu, flow.ProjectSpeaker(xvector), condition, k.GetProperty("t").GetDouble(), noise, dropConditions: false);
            value = l[0];
            grads = tape.ComputeGradients(l, sources: parameters);
        }
        var expected = k.GetProperty(form);
        Assert.True(Math.Abs(value - expected.GetProperty("loss").GetDouble()) < 1e-9, $"{form} loss {value} vs {expected.GetProperty("loss").GetDouble()}");
        Sgd(engine, parameters, grads, k.GetProperty("lr").GetDouble());
        var expectedLayers = new List<LayerBase<double>>();
        Flow(engine, expectedLayers, c, loss).LoadTorchWeights(Reader(expected.GetProperty("final")), "");
        AssertLayersMatch(layers, expectedLayers, form);
    }

    [Theory(Timeout = 300000)]
    [InlineData("generate_short")]
    [InlineData("generate_long")]
    public async Task Flow_Generation_MatchesTheOfficialCode(string name)
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var k = Data.Value.GetProperty("flow");
        var c = k.GetProperty("config");
        var g = k.GetProperty(name);
        var flow = Flow(engine, new List<LayerBase<double>>(), c, CosyVoiceFlowLoss.AbsoluteError);
        flow.LoadTorchWeights(Reader(k.GetProperty("initial")), "");
        int mel = c.GetProperty("mel").GetInt32(), promptFrames = g.GetProperty("prompt_frames").GetInt32();
        var tokens = Ints(g.GetProperty("tokens"));
        var promptTokens = Ints(g.GetProperty("prompt_tokens"));
        int textFrames = flow.MelFrames(tokens.Length), total = promptFrames + textFrames;
        Tensor<double> output;
        using (new NoGradScope<double>())
        {
            var promptMel = TensorOf(Values(g.GetProperty("prompt_mel")), promptFrames, mel);
            var xvector = TensorOf(Values(k.GetProperty("embedding")), c.GetProperty("spk").GetInt32());
            output = flow.Inference(tokens, promptTokens, promptMel, xvector, TensorOf(Values(g.GetProperty("noise")), 1, mel, total));
        }
        var expected = Values(g.GetProperty("mel"));                                                          // [frames, mel]
        Assert.Equal(g.GetProperty("frames").GetInt32(), textFrames);
        // The reference returns the generated frames cast to float32 (solve_euler's sol[-1].float()).
        for (int f = 0; f < textFrames; f++)
            for (int m = 0; m < mel; m++)
            {
                double e = expected[f * mel + m], a = output[0, m, f];
                Assert.True(Math.Abs(e - (float)a) <= 1.5e-7 * Math.Max(1.0, Math.Abs(e)), $"{name}: frame {f} bin {m}: {e} vs {a}");
            }
    }

    // ------------------------------------------------------------------ HiFT

    private static HiftGenerator<double> Hift(IEngine engine, List<LayerBase<double>> layers, JsonElement c) =>
        new(engine, layers, new HiftConfiguration(
            InChannels: c.GetProperty("mel").GetInt32(), BaseChannels: c.GetProperty("base").GetInt32(),
            Harmonics: c.GetProperty("harmonics").GetInt32(), SampleRate: 22050, NsfAlpha: 0.1, NsfSigma: 0.003, VoicedThreshold: 10,
            UpsampleRates: Ints(c.GetProperty("rates")), UpsampleKernels: Ints(c.GetProperty("kernels")),
            FftSize: c.GetProperty("n_fft").GetInt32(), HopSize: c.GetProperty("hop").GetInt32(),
            ResblockKernels: Ints(c.GetProperty("res_kernels")), ResblockDilations: Jagged(c.GetProperty("res_dilations")),
            SourceKernels: Ints(c.GetProperty("src_kernels")), SourceDilations: Jagged(c.GetProperty("src_dilations")),
            LeakySlope: 0.1, AudioLimit: 0.99, F0Channels: c.GetProperty("f0_channels").GetInt32()), new Random(0));

    // The fixture's mel [1, mel, frames], phases, noise [harmonics, samples] and pitch.
    private static (Tensor<double> Mel, double[] Phases, double[,] Noise, Tensor<double> Pitch, int Samples) HiftInputs(JsonElement h)
    {
        var c = h.GetProperty("config");
        int mel = c.GetProperty("mel").GetInt32(), frames = c.GetProperty("frames").GetInt32(), harmonics = c.GetProperty("harmonics").GetInt32() + 1;
        var noiseValues = Values(h.GetProperty("noise"));                                                   // [samples, harmonics]
        int samples = noiseValues.Length / harmonics;
        var noise = new double[harmonics, samples];
        for (int s = 0; s < samples; s++)
            for (int i = 0; i < harmonics; i++) noise[i, s] = noiseValues[s * harmonics + i];
        return (TensorOf(Values(h.GetProperty("mel")), 1, mel, frames), Values(h.GetProperty("phases")), noise,
            TensorOf(Values(h.GetProperty("pitch")), frames), samples);
    }

    [Fact(Timeout = 300000)]
    public async Task Hift_TrainsThroughItsSourceMerge_AsTheOfficialCode()
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var k = Data.Value.GetProperty("hift");
        var c = k.GetProperty("config");
        var layers = new List<LayerBase<double>>();
        var hift = Hift(engine, layers, c);
        hift.LoadTorchWeights(Reader(k.GetProperty("initial")));
        var (mel, phases, noise, pitch, samples) = HiftInputs(k);
        var weights = TensorOf(Values(k.GetProperty("weights")), samples);
        int frames = pitch.Length;

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        double loss;
        using (var tape = new GradientTape<double>())
        {
            var f0 = hift.PredictF0(mel);
            AssertClose(Values(k.GetProperty("f0")), f0, 1e-9, "f0");
            var source = hift.Merge(hift.Harmonics(f0, phases, noise));
            AssertClose(Values(k.GetProperty("source")), source, 1e-9, "source");
            var audio = hift.Decode(mel, source);
            AssertClose(Values(k.GetProperty("audio")), audio, 1e-9, "audio");
            var weighted = engine.ReduceSum(engine.TensorMultiply(audio, weights), new[] { 0 }, keepDims: false);
            var l1 = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorAbs(engine.TensorSubtract(f0, pitch)), new[] { 0 }, keepDims: false), 1.0 / frames);
            var total = engine.TensorAdd(weighted, l1);
            loss = total[0];
            grads = tape.ComputeGradients(total, sources: parameters);
        }
        Assert.True(Math.Abs(loss - k.GetProperty("loss").GetDouble()) < 1e-9, $"loss {loss} vs {k.GetProperty("loss").GetDouble()}");
        Assert.True(grads.ContainsKey(AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new[] { hift.SourceMerge })[0]),
            "the source merge got no gradient: the source's STFT is outside the graph.");
        Sgd(engine, parameters, grads, k.GetProperty("lr").GetDouble());
        var expectedLayers = new List<LayerBase<double>>();
        Hift(engine, expectedLayers, c).LoadTorchWeights(Reader(k.GetProperty("final")));
        AssertLayersMatch(layers, expectedLayers, "hift");
    }

    private static CosyVoiceHiFTDiscriminators<double> Discriminators(IEngine engine, JsonElement g)
    {
        var ffts = Ints(g.GetProperty("fft"));
        var hops = Ints(g.GetProperty("hop"));
        var wins = Ints(g.GetProperty("win"));
        return new CosyVoiceHiFTDiscriminators<double>(engine, periodChannels: Ints(g.GetProperty("period_channels")),
            resolutions: ffts.Select((f, i) => (f, hops[i], wins[i])).ToArray());
    }

    [Theory(Timeout = 300000)]
    [InlineData("generator")]
    [InlineData("discriminator")]
    public async Task Hift_GanLossesAndOneSgdStep_MatchTheOfficialCode(string turn)
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var h = Data.Value.GetProperty("hift");
        var k = Data.Value.GetProperty("hift_gan");
        var c = h.GetProperty("config");
        var g = k.GetProperty("config");
        var generatorLayers = new List<LayerBase<double>>();
        var hift = Hift(engine, generatorLayers, c);
        hift.LoadTorchWeights(Reader(h.GetProperty("initial")));
        var discriminators = Discriminators(engine, g);
        discriminators.LoadTorch(StateDict(k.GetProperty("discriminator")));
        var losses = new CosyVoiceHiFTLosses<double>(engine, 22050, melFft: g.GetProperty("mel_fft").GetInt32(),
            melHop: g.GetProperty("mel_hop").GetInt32(), melWindow: g.GetProperty("mel_win").GetInt32(), melBins: g.GetProperty("mel_bins").GetInt32());
        var (mel, phases, noise, pitch, samples) = HiftInputs(h);
        var real = TensorOf(Values(k.GetProperty("real")), samples);

        bool generatorTurn = turn == "generator";
        var trained = generatorTurn ? generatorLayers : discriminators.Layers.ToList();
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(trained);
        var expected = k.GetProperty(generatorTurn ? "generator_losses" : "discriminator_losses");
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            Tensor<double> total;
            if (generatorTurn)
            {
                var f0 = hift.PredictF0(mel);
                var audio = hift.Decode(mel, hift.Merge(hift.Harmonics(f0, phases, noise)));
                var parts = losses.Generator(real, audio, discriminators.Forward(real), discriminators.Forward(audio), f0, pitch);
                Assert.True(Math.Abs(parts.Adversarial[0] - expected.GetProperty("loss_gen").GetDouble()) < 1e-9, $"adversarial {parts.Adversarial[0]}");
                Assert.True(Math.Abs(parts.FeatureMatching[0] - expected.GetProperty("loss_fm").GetDouble()) < 1e-9, $"feature matching {parts.FeatureMatching[0]}");
                Assert.True(Math.Abs(parts.Mel[0] - expected.GetProperty("loss_mel").GetDouble()) < 1e-9, $"mel {parts.Mel[0]}");
                double tpr = expected.GetProperty("loss_tpr").GetDouble();
                Assert.True(Math.Abs(parts.Tpr[0] - tpr) <= 1e-6 * Math.Abs(tpr) + 1e-24, $"tpr {parts.Tpr[0]} vs {tpr}");
                Assert.True(Math.Abs(parts.F0[0] - expected.GetProperty("loss_f0").GetDouble()) < 1e-9, $"f0 {parts.F0[0]}");
                total = parts.Total;
            }
            else
            {
                Tensor<double> audio;
                using (new NoGradScope<double>())
                    audio = hift.Decode(mel, hift.Merge(hift.Harmonics(hift.PredictF0(mel), phases, noise)));
                total = losses.Discriminator(discriminators.Forward(real), discriminators.Forward(audio));
            }
            Assert.True(Math.Abs(total[0] - expected.GetProperty("loss").GetDouble()) < 1e-9, $"{turn} loss {total[0]} vs {expected.GetProperty("loss").GetDouble()}");
            grads = tape.ComputeGradients(total, sources: parameters);
        }
        Sgd(engine, parameters, grads, 0.01);
        var final = k.GetProperty(generatorTurn ? "generator_final" : "discriminator_final");
        if (generatorTurn)
        {
            var expectedLayers = new List<LayerBase<double>>();
            Hift(engine, expectedLayers, c).LoadTorchWeights(Reader(final));
            AssertLayersMatch(generatorLayers, expectedLayers, "generator");
        }
        else
        {
            var expectedDiscriminators = Discriminators(engine, g);
            expectedDiscriminators.LoadTorch(StateDict(final));
            AssertLayersMatch(discriminators.Layers.ToList(), expectedDiscriminators.Layers.ToList(), "discriminator");
        }
    }
}
