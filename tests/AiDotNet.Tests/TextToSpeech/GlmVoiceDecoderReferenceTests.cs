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
using AiDotNet.TextToSpeech.MultiModal;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// GLM-4-Voice's speech decoder against THUDM/GLM-4-Voice's cosyvoice fork and <c>flow_inference.AudioDecoder</c>
/// (<c>tools/reference-data/glm_voice_decoder_reference.py</c>): the block-attention flow's μ (nearest stretch), its loss
/// and one SGD step, inference after a prompt (the whole sequence stretched), and two streaming decodes through
/// <c>token2wav</c> (the web demo's blocks and the paper's 10-token blocks).
/// </summary>
public class GlmVoiceDecoderReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "glm_voice_decoder_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static Dictionary<string, double[]> Weights(JObject f, string key)
    {
        string path = Path.Combine(Path.GetTempPath(), $"glm_voice_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)f[key]!));
        try
        {
            return new WeightLoader().LoadWeights(path).ToDictionary(p => p.Key, p => p.Value.ToVector().Select(v => (double)v).ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static (GlmVoiceDecoder<double> Decoder, List<LayerBase<double>> FlowLayers, JObject Fixture) Load()
    {
        var f = Reference();
        var c = f["config"]!;
        var e = f["encoder"]!;
        var d = f["decoder"]!;
        var h = f["hift"]!;
        int input = (int)c["input_size"]!;
        var flow = new CosyVoiceFlowConfiguration(
            VocabSize: (int)c["vocab_size"]!, InputSize: input, OutputSize: (int)c["output_size"]!, SpeakerDim: (int)c["spk_embed_dim"]!,
            InputFrameRate: (double)c["input_frame_rate"]!, SampleRate: 22050, HopSize: 256,
            Encoder: new WenetEncoderConfiguration(input, (int)e["output_size"]!, (int)e["attention_heads"]!, (int)e["linear_units"]!,
                (int)e["num_blocks"]!, WenetInputLayer.Linear, Swish: true, Conformer: true, StaticChunkSize: (int)e["block_size"]!, BlockNormEpsilon: 1e-5),
            RegulatorStages: (int)f["regulator_stages"]!, DecoderChannels: d["channels"]!.Select(v => (int)v).ToArray(),
            DecoderHeads: (int)d["num_heads"]!, DecoderHeadDim: (int)d["attention_head_dim"]!, DecoderBlocksPerLevel: (int)d["n_blocks"]!,
            DecoderMidBlocks: (int)d["num_mid_blocks"]!, SigmaMin: 1e-6, TrainingCfgRate: 0.2, InferenceCfgRate: 0.7, Timesteps: 10,
            Loss: CosyVoiceFlowLoss.SquaredError, Regulator: CosyVoiceRegulator.Nearest, PromptRatio: 0.8, CosineTrainingTime: true);
        int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
        var hift = new HiftConfiguration(InChannels: (int)h["in_channels"]!, BaseChannels: (int)h["base_channels"]!, Harmonics: (int)h["nb_harmonics"]!,
            SampleRate: 22050, NsfAlpha: 0.1, NsfSigma: 0.003, VoicedThreshold: 10, UpsampleRates: Ints(h["upsample_rates"]!),
            UpsampleKernels: Ints(h["upsample_kernel_sizes"]!), FftSize: 16, HopSize: 4, ResblockKernels: Ints(h["resblock_kernel_sizes"]!),
            ResblockDilations: h["resblock_dilation_sizes"]!.Select(Ints).ToArray(), SourceKernels: Ints(h["source_resblock_kernel_sizes"]!),
            SourceDilations: h["source_resblock_dilation_sizes"]!.Select(Ints).ToArray(), LeakySlope: 0.1, AudioLimit: 0.99,
            F0Channels: (int)f["f0_channels"]!);
        var flowLayers = new List<LayerBase<double>>();
        var decoder = new GlmVoiceDecoder<double>(AiDotNetEngine.Current, flowLayers, new List<LayerBase<double>>(),
            new GlmVoiceDecoderConfiguration(flow, hift), new Random(0));
        var flowWeights = Weights(f, "flow_safetensors_base64");
        var hiftWeights = Weights(f, "hift_safetensors_base64");
        var used = new HashSet<string>();
        Func<string, int[], double[]> Reader(Dictionary<string, double[]> weights, string tag) => (name, shape) =>
        {
            var values = weights[name];
            if (shape.Length > 0) Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            used.Add(tag + name);
            return values;
        };
        decoder.Load(Reader(flowWeights, "flow:"), Reader(hiftWeights, "hift:"), hiftWeights.ContainsKey);
        Assert.All(flowWeights.Keys, k => Assert.Contains("flow:" + k, used));
        Assert.All(hiftWeights.Keys, k => Assert.Contains("hift:" + k, used));
        return (decoder, flowLayers, f);
    }

    private static Tensor<double> Values(JObject f, string key, params int[] shape) =>
        new(shape, new Vector<double>(f[key]!.Select(v => (double)v).ToArray()));

    private static void AssertClose(JToken expected, Tensor<double> actual, string what, double tolerance = 1e-9)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(values[i])),
                $"{what}[{i}]: reference {values[i]:R}, port {actual[i]:R}.");
    }

    private static void AssertScalar(double expected, double actual, string what) =>
        Assert.True(Math.Abs(expected - actual) <= 1e-9 * Math.Max(1, Math.Abs(expected)), $"{what}: reference {expected:R}, port {actual:R}.");

    // The fixture's noise: the leading values of a fixed sequence, [1, mel, frames] row-major.
    private static Func<int, Tensor<double>> Noise(JObject f, int mel)
    {
        var sequence = f["sequence"]!.Select(v => (double)v).ToArray();
        return frames => new Tensor<double>(new[] { 1, mel, frames }, new Vector<double>(sequence.Take(mel * frames).ToArray()));
    }

    // SineGen with zero phases and zero noise, as the fixture runs it.
    private static Func<Tensor<double>, Tensor<double>> Source(HiftGenerator<double> hift)
    {
        int harmonics = hift.Configuration.Harmonics + 1;
        return f0 => hift.Merge(hift.Harmonics(f0, new double[harmonics], new double[harmonics, f0.Length * hift.Configuration.UpsampleScale]));
    }

    [Fact(Timeout = 300000)]
    public async Task FlowMuLossAndInference_MatchRelease()
    {
        await Task.Yield();
        var (decoder, _, f) = Load();
        var flow = decoder.Flow;
        int mel = flow.Configuration.OutputSize, frames = (int)f["frames"]!;
        var tokens = f["tokens"]!.Select(v => (int)v).ToArray();
        var xvector = Values(f, "xvector", flow.Configuration.SpeakerDim);
        var speaker = flow.ProjectSpeaker(xvector);
        AssertClose(f["speaker"]!, speaker, "speaker");
        var mu = flow.Stretch(flow.Encode(tokens, training: false, new Random(0)), frames);
        AssertClose(f["mu"]!, mu, "mu");
        var target = Values(f, "target", 1, mel, frames);
        var cond = Values(f, "cond", 1, mel, frames);
        var noise = Values(f, "noise", 1, mel, frames);
        double t = (double)f["t"]!;
        AssertScalar((double)f["loss"]!, flow.Loss(target, mu, speaker, cond, t, noise, dropConditions: false)[0], "loss");
        AssertScalar((double)f["loss_dropped"]!, flow.Loss(target, mu, speaker, cond, t, noise, dropConditions: true)[0], "loss, conditions dropped");

        var promptTokens = f["prompt_tokens"]!.Select(v => (int)v).ToArray();
        var promptMel = Values(f, "prompt_mel", (int)f["prompt_frames"]!, mel);
        AssertClose(f["inferred"]!, decoder.Mel(tokens, promptTokens, promptMel, xvector, Noise(f, mel)), "inferred mel");
    }

    [Fact(Timeout = 300000)]
    public async Task StreamingDecodes_MatchRelease()
    {
        await Task.Yield();
        var (decoder, _, f) = Load();
        var stream = f["stream"]!.Select(v => (int)v).ToArray();
        var zeros = new Tensor<double>(new[] { decoder.Flow.Configuration.SpeakerDim });
        var noise = Noise(f, decoder.Flow.Configuration.OutputSize);
        var source = Source(decoder.Hift);
        using var _ = new NoGradScope<double>();
        Assert.Equal(34, decoder.OverlapFrames);
        AssertClose(f["speech"]!, decoder.Decode(stream, zeros, GlmVoiceDecoding.ReleaseStreaming, new Random(0), noise, source), "web demo speech");
        AssertClose(f["blocks_speech"]!, decoder.Decode(stream, zeros, GlmVoiceDecoding.PaperBlocks, new Random(0), noise, source), "10-token block speech");
    }

    [Fact(Timeout = 3600000)]
    public async Task ReleasedDecoder_MatchesRelease()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("GLM4VOICE_DECODER_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "flow.pt"))) return;
        const string fileName = "glm_voice_decoder_realweights_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && d is not null; d = d.Parent)
            path = Path.Combine(d.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        var decoder = new GlmVoiceDecoder<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), new List<LayerBase<double>>(),
            GlmVoiceDecoderConfiguration.Release(), new Random(0));
        var flowWeights = new WeightLoader().LoadWeights(Path.Combine(dir, "flow.pt"));
        var hiftWeights = new WeightLoader().LoadWeights(Path.Combine(dir, "hift.pt"));
        var used = new HashSet<string>();
        Func<string, int[], double[]> Reader(Dictionary<string, Tensor<float>> weights, string tag) => (name, _) =>
        {
            used.Add(tag + name);
            return weights[name].ToVector().Select(v => (double)v).ToArray();
        };
        decoder.Load(Reader(flowWeights, "flow:"), Reader(hiftWeights, "hift:"), hiftWeights.ContainsKey);
        Assert.All(flowWeights.Keys, k => Assert.Contains("flow:" + k, used));
        Assert.All(hiftWeights.Keys, k => Assert.Contains("hift:" + k, used));
        flowWeights.Clear();
        hiftWeights.Clear();

        var tokens = f["tokens"]!.Select(v => (int)v).ToArray();
        var zeros = new Tensor<double>(new[] { 192 });
        var noise = Noise(f, 80);
        using var _ = new NoGradScope<double>();
        AssertClose(f["mel"]!, decoder.Mel(tokens, Array.Empty<int>(), null, zeros, noise), "offline mel", 1e-8);
        AssertClose(f["speech"]!, decoder.Decode(tokens, zeros, GlmVoiceDecoding.PaperBlocks, new Random(0), noise, Source(decoder.Hift)),
            "10-token block speech", 1e-8);
    }

    [Fact(Timeout = 300000)]
    public async Task OneSgdStep_MatchesRelease()
    {
        await Task.Yield();
        var (decoder, layers, f) = Load();
        var flow = decoder.Flow;
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
