using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// NaturalSpeech 2 against Amphion's implementation in float64 at a tiny configuration
/// (tools/reference-data/naturalspeech2_reference.py): the prompt encoder and query attention, the prior (style-adaptive
/// phoneme encoder, duration and pitch predictors, length regulation, pitch buckets), the WaveNet denoiser, the RVQ
/// cross-entropy, reverse diffusion with both solvers, inference's prior, and the flow-matching variant.
/// </summary>
public class NaturalSpeech2ReferenceTests
{
    private static readonly Lazy<JsonElement> Reference = new(() =>
    {
        const string name = "naturalspeech2_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => (int)Math.Round(v.GetDouble())).ToArray();
    private static Tensor<double> T(JsonElement e, params int[] shape) => new(shape, new Vector<double>(Doubles(e)));

    private static void Close(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= tolerance * (1 + Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual.GetFlat(i)}");
    }

    private static int[,] Codes(JsonElement e)
    {
        var rows = e.EnumerateArray().Select(Ints).ToArray();
        var codes = new int[rows.Length, rows[0].Length];
        for (int q = 0; q < rows.Length; q++)
            for (int f = 0; f < rows[0].Length; f++) codes[q, f] = rows[q][f];
        return codes;
    }

    // The script's configuration; a tiny SoundStream supplies the 4 × 16 × 8 residual quantizer.
    private static (NaturalSpeech2<double> Model, NaturalSpeech2Options Options) Build()
    {
        var options = new NaturalSpeech2Options
        {
            SampleRate = 1000, HopSize = 4, VocabSize = 20, HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 2, FilterChannels = 32, EncoderKernel = 3,
            NumPromptLayers = 2, StyleAdaptivePhonemeEncoder = true, Positions = NaturalSpeech2Positions.AmphionBatchIndexed, DurationLayers = 4,
            DurationKernel = 3, PitchLayers = 4, PitchKernel = 5, PitchBins = 16, QueryTokens = 4, DenoiserLayers = 5, DenoiserDilationCycle = 2,
            DenoiserAttentionOffset = 0, CeResidual = NaturalSpeech2CeResidual.Predicted,
            Codec = new SoundStreamOptions
            {
                SampleRate = 1000, SegmentSize = 1000, Filters = 2, Ratios = [2, 2], Dimension = 8, NumQuantizers = 4, CodebookSize = 16,
                TargetBandwidthKbps = 4.0, FilmPosition = SoundStreamFilmPosition.None, IncludeDiscriminators = false,
            },
        };
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new NaturalSpeech2<double>(architecture, options);
        var state = Reference.Value.GetProperty("state");
        model.LoadAmphionState(n => state.TryGetProperty(n, out var v) ? Doubles(v) : null);
        model.SetTrainingMode(false);
        return (model, options);
    }

    [Fact(Timeout = 120000)]
    public async Task PromptAndPrior_MatchAmphion()
    {
        await Task.Yield();
        var r = Reference.Value;
        var (model, _) = Build();
        using var _ = new NoGradScope<double>();
        var core = model.Core;
        Close(Doubles(r.GetProperty("ref_latent")), model.Latent(Codes(r.GetProperty("ref_code"))), 1e-12, "ref_latent");
        Close(Doubles(r.GetProperty("latent")), model.Latent(Codes(r.GetProperty("code"))), 1e-12, "latent");
        var (prompt, query) = core.Prompt(model.Latent(Codes(r.GetProperty("ref_code"))), false, null);
        Close(Doubles(r.GetProperty("prompt")), prompt, 1e-10, "prompt");
        Close(Doubles(r.GetProperty("query")), query, 1e-10, "query");

        var phonemes = Ints(r.GetProperty("phone_id"));
        var durations = Ints(r.GetProperty("duration"));
        var pitch = Doubles(r.GetProperty("pitch"));
        var prior = r.GetProperty("prior");
        var (hidden, logDurations) = core.Phonemes(phonemes, prompt, false, null);
        var (condition, logPitch) = core.Condition(hidden, durations, pitch, prompt, false, null);
        Close(Doubles(prior.GetProperty("dur_pred_log")), logDurations, 1e-10, "dur_pred_log");
        Close(Doubles(prior.GetProperty("pitch_pred_log")), logPitch, 1e-10, "pitch_pred_log");
        Close(Doubles(prior.GetProperty("prior_out")), condition, 1e-10, "prior_out");
        Assert.Equal(Ints(prior.GetProperty("pitch_token")), pitch.Select(core.PitchToken).ToArray());
        double durationLoss = Enumerable.Range(0, durations.Length).Average(i => Math.Abs(logDurations[i] - Math.Log(1 + durations[i])));
        double pitchLoss = Enumerable.Range(0, pitch.Length).Average(f => Math.Abs(logPitch[f] - Math.Log(pitch[f])));
        Assert.Equal(prior.GetProperty("dur_loss").GetDouble(), durationLoss, 6);
        Assert.Equal(prior.GetProperty("pitch_loss").GetDouble(), pitchLoss, 6);

        // Inference's prior: the predicted durations rounded and the predicted pitch's buckets.
        var inference = r.GetProperty("inference");
        var rounded = NaturalSpeech2Core<double>.RoundDurations(logDurations);
        Assert.Equal(Ints(inference.GetProperty("dur_pred_round")), rounded);
        var (inferred, _) = core.Condition(hidden, rounded, null, prompt, false, null);
        Close(Doubles(inference.GetProperty("prior_out")), inferred, 1e-10, "inference prior_out");
    }

    [Fact(Timeout = 120000)]
    public async Task DenoiserLossesAndReverseDiffusion_MatchAmphion()
    {
        await Task.Yield();
        var r = Reference.Value;
        var (model, options) = Build();
        using var _ = new NoGradScope<double>();
        var core = model.Core;
        int frames = Ints(r.GetProperty("pitch")).Length;
        var condition = T(r.GetProperty("prior").GetProperty("prior_out"), frames, 16);
        var query = T(r.GetProperty("query"), 4, 16);
        var d = r.GetProperty("diffusion");
        double t = d.GetProperty("t").GetDouble();
        var x0 = core.Denoiser.Forward(T(d.GetProperty("xt"), 1, 8, frames), t, condition, query, false, null);
        Close(Doubles(d.GetProperty("x0_pred")), x0, 1e-10, "x0_pred");
        // The RVQ cross-entropy over the residuals of the prediction's own codes (Amphion's), in float32 there; Amphion's
        // targets are the ground-truth latent re-quantized.
        var gt = Ints(d.GetProperty("gt_indices"));
        var codes = new int[4, frames];
        for (int q = 0; q < 4; q++)
            for (int f = 0; f < frames; f++) codes[q, f] = gt[q * frames + f];
        Assert.True(Math.Abs(d.GetProperty("ce_loss").GetDouble() - model.RvqCrossEntropy(x0, codes)[0]) <= 1e-6 * d.GetProperty("ce_loss").GetDouble(), "ce_loss");

        var reverse = r.GetProperty("reverse");
        int steps = reverse.GetProperty("steps").GetInt32();
        var start = T(reverse.GetProperty("start"), 1, 8, frames);
        Close(Doubles(reverse.GetProperty("euler")), model.Reverse(start, condition, query, steps), 1e-9, "euler");
        options.Solver = NaturalSpeech2Solver.Midpoint;
        Close(Doubles(reverse.GetProperty("midpoint")), model.Reverse(start, condition, query, steps), 1e-9, "midpoint");

        var flow = r.GetProperty("flow");
        options.Diffusion = NaturalSpeech2Diffusion.FlowMatching;
        Close(Doubles(flow.GetProperty("flow_pred")), core.Denoiser.Forward(T(flow.GetProperty("xt"), 1, 8, frames), t, condition, query, false, null), 1e-10, "flow_pred");
        Close(Doubles(flow.GetProperty("reverse")), model.Reverse(start, condition, query, steps), 1e-9, "flow reverse");
    }

    /// <summary>
    /// The released checkpoint (amphion/naturalspeech2_libritts) with EnCodec's 24 kHz weights against Amphion in float64
    /// (tools/reference-data/naturalspeech2_checkpoint_reference.py). Set NATURALSPEECH2_DIR to the directory holding
    /// pytorch_model.bin, encodec_24khz.safetensors (facebook/encodec_24khz) and the script's JSON.
    /// </summary>
    [Fact(Timeout = 3600000)]
    public async Task ReleasedCheckpoint_MatchesAmphion()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("NATURALSPEECH2_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "naturalspeech2_checkpoint_reference.json"))) return;
        var r = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "naturalspeech2_checkpoint_reference.json"))).RootElement;
        var options = NaturalSpeech2Options.Amphion();
        var model = new NaturalSpeech2<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), options);
        ((EnCodec<double>)model.Codec).LoadPretrainedWeights(Path.Combine(dir, "encodec_24khz.safetensors"));
        model.LoadAmphionCheckpoint(Path.Combine(dir, "pytorch_model.bin"));
        model.SetTrainingMode(false);
        using var _ = new NoGradScope<double>();
        var core = model.Core;

        var (prompt, query) = core.Prompt(model.Latent(Codes(r.GetProperty("ref_code"))), false, null);
        Close(Doubles(r.GetProperty("prompt")), prompt, 1e-8, "prompt");
        Close(Doubles(r.GetProperty("query")), query, 1e-8, "query");
        var (hidden, logDurations) = core.Phonemes(Ints(r.GetProperty("phone_id")), prompt, false, null);
        Close(Doubles(r.GetProperty("dur_pred_log")), logDurations, 1e-8, "dur_pred_log");
        var durations = NaturalSpeech2Core<double>.RoundDurations(logDurations);
        Assert.Equal(Ints(r.GetProperty("dur_pred_round")), durations);
        var (condition, _) = core.Condition(hidden, durations, null, prompt, false, null);
        Close(Doubles(r.GetProperty("condition")), condition, 1e-8, "condition");
        int frames = condition.Shape[0];
        Close(Doubles(r.GetProperty("x0_pred")), core.Denoiser.Forward(T(r.GetProperty("xt"), 1, 128, frames), 0.5, condition, query, false, null),
            1e-8, "x0_pred");
        var x0 = model.Reverse(T(r.GetProperty("start"), 1, 128, frames), condition, query, 3);
        Close(Doubles(r.GetProperty("x0")), x0, 1e-7, "x0");
        Close(Doubles(r.GetProperty("audio")), model.Codec.DecodeEmbeddings(x0), 1e-6, "audio");
    }
}
