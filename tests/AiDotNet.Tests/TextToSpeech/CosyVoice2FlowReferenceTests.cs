using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CosyVoice 2's flow against FunAudioLLM/CosyVoice's <c>CausalMaskedDiffWithXvec</c>
/// (<c>tools/reference-data/cosyvoice2_flow_reference.py</c>), offline and streaming: the projected x-vector, the
/// upsampling encoder's states, the flow-matching loss with and without condition dropout, the sample of a prompt and a
/// text from the fixed noise, and the loss after one SGD step.
/// </summary>
public class CosyVoice2FlowReferenceTests
{
    private static readonly Lazy<JObject> Data = new(() =>
    {
        const string fileName = "cosyvoice2_flow_reference.json";
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

    private static double[] Weights(string name, int count)
    {
        long h = 0;
        for (int j = 0; j < name.Length; j++) h += (j + 1) * (long)name[j];
        h %= 100003;
        double b = name.Contains("norm") && name.EndsWith("weight", StringComparison.Ordinal) ? 1.0 : 0.0;
        var values = new double[count];
        for (long i = 0; i < count; i++) values[i] = b + 0.1 * (((i * 7919 + h) % 2003) / 2003.0 - 0.5);
        return values;
    }

    private static (CosyVoiceFlow<double> Flow, List<LayerBase<double>> Layers) Build()
    {
        var d = Data.Value;
        var c = d["config"]!;
        var e = d["encoder"]!;
        var k = d["decoder"]!;
        int width = (int)e["output_size"]!;
        var configuration = new CosyVoiceFlowConfiguration(
            VocabSize: (int)c["vocab_size"]!, InputSize: (int)c["input_size"]!, OutputSize: (int)c["output_size"]!,
            SpeakerDim: (int)c["spk_embed_dim"]!, InputFrameRate: (int)c["input_frame_rate"]!, SampleRate: 24000, HopSize: 480,
            Encoder: new WenetEncoderConfiguration(width, width, (int)e["attention_heads"]!, (int)e["linear_units"]!, (int)e["num_blocks"]!,
                WenetInputLayer.Linear, Swish: true, Conformer: true, StaticChunkSize: (int)e["static_chunk_size"]!, Dropout: 0,
                PositionalDropout: 0, AttentionDropout: 0),
            RegulatorStages: 0, DecoderChannels: k["channels"]!.Select(v => (int)v).ToArray(), DecoderHeads: (int)k["num_heads"]!,
            DecoderHeadDim: (int)k["attention_head_dim"]!, DecoderBlocksPerLevel: (int)k["n_blocks"]!, DecoderMidBlocks: (int)k["num_mid_blocks"]!,
            SigmaMin: 1e-6, TrainingCfgRate: 0.2, InferenceCfgRate: 0.7, Timesteps: 10, Loss: CosyVoiceFlowLoss.SquaredError,
            Causal: true, TokenMelRatio: (int)c["token_mel_ratio"]!, Lookahead: (int)c["pre_lookahead_len"]!);
        var layers = new List<LayerBase<double>>();
        var flow = new CosyVoiceFlow<double>(AiDotNetEngine.Current, layers, configuration, new Random(0));
        var shapes = d["parameters"]!.ToDictionary(p => (string)p[0]!, p => p[1]!.Aggregate(1, (a, s) => a * (int)s));
        var used = new HashSet<string>();
        flow.LoadTorchWeights((name, _) =>
        {
            Assert.True(shapes.TryGetValue(name, out int count), $"{name} is not a CausalMaskedDiffWithXvec parameter.");
            used.Add(name);
            return Weights(name, count);
        }, "");
        Assert.Equal(shapes.Keys.OrderBy(n => n, StringComparer.Ordinal), used.OrderBy(n => n, StringComparer.Ordinal));
        return (flow, layers);
    }

    private static Tensor<double> TensorOf(JToken values, params int[] shape) =>
        new(shape, new Vector<double>(values.Select(v => (double)v).ToArray()));

    private static int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();

    private static void AssertClose(JToken expected, Tensor<double> actual, string what)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(values[i])),
                $"{what}[{i}]: reference {values[i]:R}, port {actual[i]:R}.");
    }

    private static void AssertClose(double expected, double actual, string what) =>
        Assert.True(Math.Abs(expected - actual) <= 1e-9 * Math.Max(1, Math.Abs(expected)), $"{what}: reference {expected:R}, port {actual:R}.");

    [Theory(Timeout = 300000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task EncoderLossAndSample_MatchTheOfficialCode(bool streaming)
    {
        await Task.Yield();
        var d = Data.Value;
        var r = d[streaming ? "streaming" : "offline"]!;
        var (flow, _) = Build();
        int mel = (int)d["config"]!["output_size"]!, spk = (int)d["config"]!["spk_embed_dim"]!;
        var tokens = Ints(d["tokens"]!);
        int frames = 2 * tokens.Length;
        var xvector = TensorOf(d["xvector"]!, spk);
        using var _ = new NoGradScope<double>();
        var speaker = flow.ProjectSpeaker(xvector);
        AssertClose(r["speaker"]!, speaker, "speaker");
        var encoded = flow.Encode(tokens, training: false, new Random(0), streaming);
        var mu = AiDotNetEngine.Current.Reshape(AiDotNetEngine.Current.TensorTranspose(encoded), new[] { 1, mel, frames });
        AssertClose(r["mu"]!, mu, "mu");
        var target = TensorOf(d["target"]!, 1, mel, frames);
        var cond = TensorOf(d["cond"]!, 1, mel, frames);
        var noise = TensorOf(d["noise"]!, 1, mel, frames);
        double t = (double)d["t"]!;
        AssertClose(r["loss"]![0]!.Value<double>(), flow.Loss(target, mu, speaker, cond, t, noise, false, streaming)[0], "loss");
        AssertClose(r["loss_dropped"]![0]!.Value<double>(), flow.Loss(target, mu, speaker, cond, t, noise, true, streaming)[0], "loss dropped");

        var prompt = Ints(d["prompt"]!);
        var promptMel = TensorOf(d["prompt_mel"]!, 2 * prompt.Length, mel);
        int total = 2 * (prompt.Length + tokens.Length);
        var sample = flow.Inference(tokens, prompt, promptMel, xvector, TensorOf(r["sample_noise"]!, 1, mel, total), streaming);
        AssertClose(r["sample"]!, sample, "sample");
    }

    [Fact(Timeout = 300000)]
    public async Task OneSgdStep_MatchesTheOfficialCode()
    {
        await Task.Yield();
        var d = Data.Value;
        var (flow, layers) = Build();
        var engine = AiDotNetEngine.Current;
        int mel = (int)d["config"]!["output_size"]!, spk = (int)d["config"]!["spk_embed_dim"]!;
        var tokens = Ints(d["tokens"]!);
        int frames = 2 * tokens.Length;
        var xvector = TensorOf(d["xvector"]!, spk);
        var target = TensorOf(d["target"]!, 1, mel, frames);
        var cond = TensorOf(d["cond"]!, 1, mel, frames);
        var noise = TensorOf(d["noise"]!, 1, mel, frames);
        double t = (double)d["t"]!;
        Tensor<double> Loss()
        {
            var encoded = flow.Encode(tokens, training: false, new Random(0));
            var mu = engine.Reshape(engine.TensorTranspose(encoded), new[] { 1, mel, frames });
            return flow.Loss(target, mu, flow.ProjectSpeaker(xvector), cond, t, noise, false);
        }
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var loss = Loss();
            AssertClose(d["offline"]!["loss"]![0]!.Value<double>(), loss[0], "loss");
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        double rate = (double)d["learning_rate"]!;
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= rate * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        using (new NoGradScope<double>())
            AssertClose((double)d["loss_after_sgd"]!, Loss()[0], "loss after the step");
    }
}
