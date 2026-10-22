using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CosyVoice 1's flow model against FunAudioLLM/CosyVoice's own <c>MaskedDiffWithXvec</c> with its Conformer encoder,
/// <c>InterpolateRegulator</c> and <c>ConditionalCFM</c> (<c>tools/reference-data/cosyvoice_flow_reference.py</c>):
/// the projected x-vector, μ, the flow-matching loss with and without the conditions, a guided Euler sample, and the
/// loss after one SGD step.
/// </summary>
public class CosyVoiceFlowReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "cosyvoice_flow_reference.json";
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
    }

    private static (CosyVoiceFlow<double> Flow, List<LayerBase<double>> Layers, JObject Fixture) Load()
    {
        var f = Reference();
        var c = f["config"]!;
        var e = f["encoder"]!;
        var d = f["decoder"]!;
        int input = (int)c["input_size"]!;
        var configuration = new CosyVoiceFlowConfiguration(
            VocabSize: (int)c["vocab_size"]!, InputSize: input, OutputSize: (int)c["output_size"]!, SpeakerDim: (int)c["spk_embed_dim"]!,
            InputFrameRate: (int)c["input_frame_rate"]!, SampleRate: 22050, HopSize: 256,
            Encoder: new WenetEncoderConfiguration(input, (int)e["output_size"]!, (int)e["attention_heads"]!, (int)e["linear_units"]!,
                (int)e["num_blocks"]!, WenetInputLayer.Linear, Swish: true, Conformer: true),
            RegulatorStages: (int)f["regulator_stages"]!, DecoderChannels: d["channels"]!.Select(v => (int)v).ToArray(),
            DecoderHeads: (int)d["num_heads"]!, DecoderHeadDim: (int)d["attention_head_dim"]!, DecoderBlocksPerLevel: (int)d["n_blocks"]!,
            DecoderMidBlocks: (int)d["num_mid_blocks"]!, SigmaMin: 1e-6, TrainingCfgRate: 0.2, InferenceCfgRate: 0.7,
            Timesteps: (int)f["steps"]!);
        var layers = new List<LayerBase<double>>();
        var flow = new CosyVoiceFlow<double>(AiDotNetEngine.Current, layers, configuration, new Random(0));
        string path = Path.Combine(Path.GetTempPath(), $"cosyvoice_flow_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)f["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            flow.LoadTorchWeights((name, shape) =>
            {
                var tensor = weights[name];
                if (shape.Length > 0) Assert.Equal(shape.Aggregate(1, (a, b) => a * b), tensor.Length);
                used.Add(name);
                return tensor.ToVector().Select(v => (double)v).ToArray();
            }, "");
            Assert.Equal(weights.Keys.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
        }
        finally
        {
            File.Delete(path);
        }
        return (flow, layers, f);
    }

    private static Tensor<double> Values(JObject f, string key, params int[] shape) =>
        new(shape, new Vector<double>(f[key]!.Select(v => (double)v).ToArray()));

    private static void AssertClose(JToken expected, Tensor<double> actual, string what)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(values[i])),
                $"{what}[{i}]: reference {values[i]:R}, port {actual[i]:R}.");
    }

    private static void AssertScalar(double expected, double actual, string what) =>
        Assert.True(Math.Abs(expected - actual) <= 1e-9 * Math.Max(1, Math.Abs(expected)), $"{what}: reference {expected:R}, port {actual:R}.");

    [Fact(Timeout = 120000)]
    public async Task SpeakerMuLossesAndSample_MatchCosyVoice()
    {
        await Task.Yield();
        var (flow, _, f) = Load();
        int mel = flow.Configuration.OutputSize, frames = (int)f["frames"]!;
        var tokens = f["tokens"]!.Select(v => (int)v).ToArray();
        var speaker = flow.ProjectSpeaker(Values(f, "xvector", flow.Configuration.SpeakerDim));
        AssertClose(f["speaker"]!, speaker, "speaker");
        var mu = flow.Stretch(flow.Encode(tokens, training: false, new Random(0)), frames);
        AssertClose(f["mu"]!, mu, "mu");
        var target = Values(f, "target", 1, mel, frames);
        var cond = Values(f, "cond", 1, mel, frames);
        var noise = Values(f, "noise", 1, mel, frames);
        double t = (double)f["t"]!;
        AssertScalar((double)f["loss"]!, flow.Loss(target, mu, speaker, cond, t, noise, dropConditions: false)[0], "loss");
        AssertScalar((double)f["loss_dropped"]!, flow.Loss(target, mu, speaker, cond, t, noise, dropConditions: true)[0], "loss, conditions dropped");
        AssertClose(f["sample"]!, flow.Sample(mu, speaker, cond, Values(f, "start", 1, mel, frames)), "sample");
    }

    [Fact(Timeout = 120000)]
    public async Task OneSgdStep_MatchesCosyVoice()
    {
        await Task.Yield();
        var (flow, layers, f) = Load();
        int mel = flow.Configuration.OutputSize, frames = (int)f["frames"]!;
        var tokens = f["tokens"]!.Select(v => (int)v).ToArray();
        var xvector = Values(f, "xvector", flow.Configuration.SpeakerDim);
        var target = Values(f, "target", 1, mel, frames);
        var cond = Values(f, "cond", 1, mel, frames);
        var noise = Values(f, "noise", 1, mel, frames);
        double t = (double)f["t"]!;
        Tensor<double> Objective() =>
            flow.Loss(target, flow.Stretch(flow.Encode(tokens, false, new Random(0)), frames), flow.ProjectSpeaker(xvector), cond, t, noise, false);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var loss = Objective();
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        double rate = (double)f["learning_rate"]!;
        var engine = AiDotNetEngine.Current;
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= rate * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        AssertScalar((double)f["loss_after_sgd"]!, Objective()[0], "loss after the step");
    }
}
