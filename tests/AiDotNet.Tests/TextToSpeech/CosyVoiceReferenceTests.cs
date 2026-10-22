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
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CosyVoice's networks against the official FunAudioLLM/CosyVoice modules in float64
/// (tools/reference-data/cosyvoice_reference.py): the same weights, the same inputs, the loss and the weights after one
/// SGD step.
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

    private static Dictionary<string, double[]> StateDict(JsonElement e)
        => e.EnumerateObject().ToDictionary(p => p.Name, p => Values(p.Value));

    private static CosyVoiceLanguageModel<double> LanguageModel(IEngine engine, List<LayerBase<double>> layers, JsonElement c)
    {
        int width = c.GetProperty("width").GetInt32(), heads = c.GetProperty("heads").GetInt32();
        int units = c.GetProperty("units").GetInt32(), blocks = c.GetProperty("blocks").GetInt32();
        int textIn = c.GetProperty("text_in").GetInt32(), textOut = c.GetProperty("text_out").GetInt32();
        return new CosyVoiceLanguageModel<double>(engine, layers, new CosyVoiceLanguageModelConfiguration(
            TextVocabulary: c.GetProperty("text_vocab").GetInt32(), SpeechTokens: c.GetProperty("speech_tokens").GetInt32(),
            TextEncoderInput: textIn,
            TextEncoder: new WenetEncoderConfiguration(textIn, textOut, heads, units, blocks, Dropout: 0, PositionalDropout: 0,
                InputLayer: WenetInputLayer.Linear, SwishActivation: true, StaticChunkSize: 1),
            Language: new WenetEncoderConfiguration(width, width, heads, units, blocks, Dropout: 0, PositionalDropout: 0,
                InputLayer: WenetInputLayer.LinearLegacy, SwishActivation: false, StaticChunkSize: 1),
            SpeakerEmbeddingDim: c.GetProperty("spk").GetInt32()));
    }

    [Fact(Timeout = 120000)]
    public async Task LanguageModel_LossAndOneSgdStep_MatchTheOfficialCode()
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var k = Data.Value.GetProperty("language_model");
        var c = k.GetProperty("config");
        var layers = new List<LayerBase<double>>();
        var model = LanguageModel(engine, layers, c);
        model.LoadTorch(StateDict(k.GetProperty("initial")));
        var text = Ints(k.GetProperty("text"));
        var speech = Ints(k.GetProperty("speech"));
        var embedding = Values(k.GetProperty("embedding"));

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        double loss;
        using (var tape = new GradientTape<double>())
        {
            var (sum, count) = model.Loss(text, speech, embedding, training: true, new Random(0));
            var mean = engine.TensorMultiplyScalar(sum, 1.0 / count);
            loss = mean[0];
            grads = tape.ComputeGradients(mean, sources: parameters);
        }
        double expected = k.GetProperty("loss").GetDouble();
        Assert.True(Math.Abs(loss - expected) < 1e-9, $"loss {loss} vs {expected}");

        double lr = k.GetProperty("lr").GetDouble();
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= lr * g[i];
            engine.InvalidatePersistentTensor(p);
        }

        var expectedLayers = new List<LayerBase<double>>();
        var reference = LanguageModel(engine, expectedLayers, c);
        reference.LoadTorch(StateDict(k.GetProperty("final")));
        Assert.Equal(expectedLayers.Count, layers.Count);
        for (int l = 0; l < layers.Count; l++)
        {
            var a = layers[l].GetParameters();
            var e = expectedLayers[l].GetParameters();
            Assert.Equal(e.Length, a.Length);
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - a[i]) < 1e-9, $"layer {l} ({layers[l].GetType().Name}) parameter {i}: {e[i]} vs {a[i]}");
        }
    }

    private static CosyVoiceFlow<double> Flow(IEngine engine, List<LayerBase<double>> layers, JsonElement c, CosyVoiceFlowLoss loss)
    {
        int input = c.GetProperty("input").GetInt32();
        return new CosyVoiceFlow<double>(engine, layers, new CosyVoiceFlowConfiguration(
            Vocabulary: c.GetProperty("vocab").GetInt32(), InputSize: input, MelChannels: c.GetProperty("mel").GetInt32(),
            SpeakerEmbeddingDim: c.GetProperty("spk").GetInt32(),
            Encoder: new WenetEncoderConfiguration(input, input, c.GetProperty("heads").GetInt32(), c.GetProperty("units").GetInt32(),
                c.GetProperty("blocks").GetInt32(), Dropout: 0, PositionalDropout: 0, InputLayer: WenetInputLayer.Linear,
                SwishActivation: true),
            RegulatorBlocks: 4, EstimatorChannels: Ints(c.GetProperty("channels")), EstimatorHeadDim: c.GetProperty("head_dim").GetInt32(),
            EstimatorHeads: c.GetProperty("est_heads").GetInt32(), EstimatorBlocks: c.GetProperty("est_blocks").GetInt32(),
            EstimatorMidBlocks: c.GetProperty("est_mid").GetInt32(), EstimatorDropout: 0, InputFrameRate: 50, SampleRate: 22050,
            HopSize: 256, Loss: loss));
    }

    private static Tensor<double> Matrix(double[] values, int rows, int cols)
    {
        var t = new Tensor<double>(new[] { rows, cols });
        for (int i = 0; i < values.Length; i++) t[i] = values[i];
        return t;
    }

    private static void AssertLayersMatch(List<LayerBase<double>> actual, List<LayerBase<double>> expected, string what, double tolerance = 1e-9)
    {
        // The trainable tensors only: GetParameters also carries buffers (a normed convolution's spectral-norm vectors),
        // which each instance draws for itself.
        Assert.Equal(expected.Count, actual.Count);
        for (int l = 0; l < actual.Count; l++)
        {
            var a = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new[] { actual[l] });
            var e = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new[] { expected[l] });
            Assert.Equal(e.Count, a.Count);
            for (int t = 0; t < e.Count; t++)
            {
                // An unnormalized convolution's gain is never read (nor trained), so it is not part of the model.
                if (actual[l] is NormedConv1DLayer<double> { Normalization: ConvolutionNormalization.None } && t == 1) continue;
                Assert.Equal(e[t].Length, a[t].Length);
                for (int i = 0; i < e[t].Length; i++)
                    Assert.True(Math.Abs(e[t][i] - a[t][i]) < tolerance,
                        $"{what}: layer {l} ({actual[l].GetType().Name}) tensor {t} value {i}: {e[t][i]} vs {a[t][i]}");
            }
        }
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
        flow.LoadTorch(StateDict(k.GetProperty("initial")));
        int frames = k.GetProperty("frames").GetInt32(), mel = c.GetProperty("mel").GetInt32();
        var target = Matrix(Values(k.GetProperty("mel")), frames, mel);
        var noise = Matrix(Values(k.GetProperty("noise")), mel, frames);

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        double value;
        using (var tape = new GradientTape<double>())
        {
            var l = flow.Loss(Ints(k.GetProperty("tokens")), target, Values(k.GetProperty("embedding")), k.GetProperty("prompt").GetInt32(),
                k.GetProperty("t").GetDouble(), noise, keepConditions: true, training: true, new Random(0));
            value = l[0];
            grads = tape.ComputeGradients(l, sources: parameters);
        }
        var expected = k.GetProperty(form);
        Assert.True(Math.Abs(value - expected.GetProperty("loss").GetDouble()) < 1e-9, $"{form} loss {value} vs {expected.GetProperty("loss").GetDouble()}");
        double lr = k.GetProperty("lr").GetDouble();
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= lr * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        var expectedLayers = new List<LayerBase<double>>();
        Flow(engine, expectedLayers, c, loss).LoadTorch(StateDict(expected.GetProperty("final")));
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
        flow.LoadTorch(StateDict(k.GetProperty("initial")));
        int mel = c.GetProperty("mel").GetInt32(), promptFrames = g.GetProperty("prompt_frames").GetInt32();
        var tokens = Ints(g.GetProperty("tokens"));
        int total = promptFrames + flow.MelFrames(tokens.Length);
        var output = flow.Generate(Ints(g.GetProperty("prompt_tokens")), tokens,
            Matrix(Values(g.GetProperty("prompt_mel")), promptFrames, mel), Values(k.GetProperty("embedding")), new Random(0),
            Matrix(Values(g.GetProperty("noise")), mel, total));
        var expected = Values(g.GetProperty("mel"));
        Assert.Equal(g.GetProperty("frames").GetInt32(), output.Shape[0]);
        Assert.Equal(expected.Length, output.Length);
        // The reference returns the mel cast to float32 (solve_euler's sol[-1].float()); compare at that precision.
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - (float)output[i]) <= 1.5e-7 * Math.Max(1.0, Math.Abs(expected[i])),
                $"{name}: mel {i}: {expected[i]} vs {output[i]}");
    }

    private static int[][] Jagged(JsonElement e) => e.EnumerateArray().Select(Ints).ToArray();

    private static CosyVoiceHiFT<double> HiFT(IEngine engine, List<LayerBase<double>> layers, JsonElement c) =>
        new(engine, layers, new CosyVoiceHiFTConfiguration(
            MelChannels: c.GetProperty("mel").GetInt32(), BaseChannels: c.GetProperty("base").GetInt32(),
            Harmonics: c.GetProperty("harmonics").GetInt32(), UpsampleRates: Ints(c.GetProperty("rates")),
            UpsampleKernels: Ints(c.GetProperty("kernels")), FftSize: c.GetProperty("n_fft").GetInt32(),
            HopSize: c.GetProperty("hop").GetInt32(), ResblockKernels: Ints(c.GetProperty("res_kernels")),
            ResblockDilations: Jagged(c.GetProperty("res_dilations")), SourceResblockKernels: Ints(c.GetProperty("src_kernels")),
            SourceResblockDilations: Jagged(c.GetProperty("src_dilations")), F0Channels: c.GetProperty("f0_channels").GetInt32()));

    private static void AssertClose(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < tolerance, $"{what} {i}: {expected[i]} vs {actual[i]}");
    }

    [Fact(Timeout = 300000)]
    public async Task HiFT_SourceDecodeAndOneSgdStep_MatchTheOfficialCode()
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var k = Data.Value.GetProperty("hift");
        var c = k.GetProperty("config");
        var layers = new List<LayerBase<double>>();
        var hift = HiFT(engine, layers, c);
        hift.LoadTorch(StateDict(k.GetProperty("initial")));
        int mel = c.GetProperty("mel").GetInt32(), frames = c.GetProperty("frames").GetInt32();
        int h = c.GetProperty("harmonics").GetInt32() + 1;
        var melTensor = new Tensor<double>(new[] { 1, mel, frames });
        var melValues = Values(k.GetProperty("mel"));
        for (int i = 0; i < melValues.Length; i++) melTensor[i] = melValues[i];
        var noiseValues = Values(k.GetProperty("noise"));
        int samples = noiseValues.Length / h;
        var noise = new double[samples, h];
        for (int n = 0; n < samples; n++)
            for (int i = 0; i < h; i++) noise[n, i] = noiseValues[n * h + i];
        var weights = Values(k.GetProperty("weights"));
        var pitch = Values(k.GetProperty("pitch"));

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        double loss;
        using (var tape = new GradientTape<double>())
        {
            var f0 = hift.PredictF0(melTensor);
            AssertClose(Values(k.GetProperty("f0")), f0, 1e-9, "f0");
            var source = hift.Source(hift.Sines(hift.UpsampleF0(f0), Values(k.GetProperty("phases")), noise));
            AssertClose(Values(k.GetProperty("source")), source, 1e-9, "source");
            var audio = hift.Decode(melTensor, source);
            AssertClose(Values(k.GetProperty("audio")), audio, 1e-9, "audio");
            var w = new Tensor<double>(new[] { samples });
            var p = new Tensor<double>(new[] { frames });
            for (int i = 0; i < samples; i++) w[i] = weights[i];
            for (int i = 0; i < frames; i++) p[i] = pitch[i];
            var weighted = engine.ReduceSum(engine.TensorMultiply(audio, w), new[] { 0 }, keepDims: false);
            var l1 = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorAbs(engine.TensorSubtract(f0, p)), new[] { 0 }, keepDims: false), 1.0 / frames);
            var total = engine.TensorAdd(weighted, l1);
            loss = total[0];
            grads = tape.ComputeGradients(total, sources: parameters);
        }
        Assert.True(Math.Abs(loss - k.GetProperty("loss").GetDouble()) < 1e-9, $"loss {loss} vs {k.GetProperty("loss").GetDouble()}");
        double lr = k.GetProperty("lr").GetDouble();
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= lr * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        var expectedLayers = new List<LayerBase<double>>();
        HiFT(engine, expectedLayers, c).LoadTorch(StateDict(k.GetProperty("final")));
        AssertLayersMatch(layers, expectedLayers, "hift");
    }
}
