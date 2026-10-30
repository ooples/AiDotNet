using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The unit-based HiFi-GAN against fairseq's <c>CodeGenerator</c> (<c>tools/reference-data/code_hifigan_reference.py</c>):
/// the duration predictor's log durations and the waveform with and without duration prediction, on a tiny generator
/// and on the released mHuBERT-unit LJSpeech vocoder (gated on LLAMAOMNI_VOCODER, the path of <c>g_00500000</c>).
/// </summary>
public class CodeHiFiGanReferenceTests
{
    private static JObject Fixture(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static void Close(JToken expected, Tensor<double> actual, string what, double tolerance)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(values[i])), $"{what}[{i}]: {values[i]:R} vs {actual[i]:R}");
    }

    private static void Check(CodeHiFiGan<double> vocoder, JObject f, double tolerance)
    {
        var units = f["units"]!.Select(v => (int)v).ToArray();
        using var _ = new NoGradScope<double>();
        Close(f["log_durations"]!, vocoder.LogDurations(units), "log durations", tolerance);
        Close(f["with_durations"]!, vocoder.Forward(units, durationPrediction: true), "waveform with durations", tolerance);
        Close(f["without"]!, vocoder.Forward(units, durationPrediction: false), "waveform without durations", tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task TinyGenerator_MatchesFairseq()
    {
        await Task.Yield();
        var f = Fixture("code_hifigan_reference.json");
        var c = f["config"]!;
        int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
        var configuration = new CodeHiFiGanConfiguration
        {
            Units = (int)c["num_embeddings"]!, EmbeddingDim = (int)c["embedding_dim"]!, InitialChannels = (int)c["upsample_initial_channel"]!,
            UpsampleRates = Ints(c["upsample_rates"]!), UpsampleKernels = Ints(c["upsample_kernel_sizes"]!), ResblockKernels = Ints(c["resblock_kernel_sizes"]!),
            ResblockDilations = c["resblock_dilation_sizes"]!.Select(Ints).ToArray(), DurationHidden = (int)c["dur_predictor_params"]!["var_pred_hidden_dim"]!,
            DurationKernel = (int)c["dur_predictor_params"]!["var_pred_kernel_size"]!,
        };
        var vocoder = new CodeHiFiGan<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), configuration, new Random(0));
        string path = Path.Combine(Path.GetTempPath(), $"code_hifigan_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)f["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            vocoder.LoadTorchWeights(name =>
            {
                if (!weights.TryGetValue(name, out var t)) return null;
                used.Add(name);
                return t.ToVector().Select(v => (double)v).ToArray();
            });
            Assert.All(weights.Keys, k => Assert.Contains(k, used));
        }
        finally
        {
            File.Delete(path);
        }
        Check(vocoder, f, 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task MultiSpeakerGenerator_MatchesSpeechResynthesis()
    {
        await Task.Yield();
        var f = Fixture("code_hifigan_multispeaker_reference.json");
        var c = f["config"]!;
        int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
        var configuration = new CodeHiFiGanConfiguration
        {
            Units = (int)c["num_embeddings"]!, EmbeddingDim = (int)c["embedding_dim"]!, InitialChannels = (int)c["upsample_initial_channel"]!,
            UpsampleRates = Ints(c["upsample_rates"]!), UpsampleKernels = Ints(c["upsample_kernel_sizes"]!), ResblockKernels = Ints(c["resblock_kernel_sizes"]!),
            ResblockDilations = c["resblock_dilation_sizes"]!.Select(Ints).ToArray(), DurationHidden = 8, DurationKernel = 3, DurationDropout = 0.0,
            Speakers = (int)c["num_speakers"]!,
        };
        var vocoder = new CodeHiFiGan<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), configuration, new Random(0));
        string path = Path.Combine(Path.GetTempPath(), $"code_hifigan_ms_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)f["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            vocoder.LoadTorchWeights(name =>
            {
                if (!weights.TryGetValue(name, out var t)) return null;
                used.Add(name);
                return t.ToVector().Select(v => (double)v).ToArray();
            });
            Assert.All(weights.Keys, k => Assert.Contains(k, used));
        }
        finally
        {
            File.Delete(path);
        }
        using var _ = new NoGradScope<double>();
        var (wave, loss) = vocoder.TrainingForward(Ints(f["codes"]!), 7, new Random(0));
        Close(f["training_wave"]!, wave, "training waveform", 1e-9);
        Assert.True(Math.Abs((double)f["training_loss"]! - loss![0]) <= 1e-9, $"duration loss {(double)f["training_loss"]!:R} vs {loss[0]:R}");
        var (twoWave, twoLoss) = vocoder.TrainingForward(Ints(f["two_runs"]!), 2, new Random(0));
        Close(f["two_runs_wave"]!, twoWave, "two-run waveform", 1e-9);
        Assert.True(Math.Abs((double)f["two_runs_loss"]! - twoLoss![0]) <= 1e-9, "two-run duration loss");
        foreach (int speaker in new[] { 0, 7 })
            Close(f[$"inferred_{speaker}"]!, vocoder.Forward(Ints(f["units"]!), true, speaker), $"speaker {speaker}", 1e-9);
    }

    [Fact(Timeout = 600000)]
    public async Task ReleasedVocoder_MatchesFairseq()
    {
        await Task.Yield();
        string? checkpoint = Environment.GetEnvironmentVariable("LLAMAOMNI_VOCODER");
        if (checkpoint is null || !File.Exists(checkpoint)) return;
        var weights = new WeightLoader().LoadWeights(checkpoint);
        var vocoder = new CodeHiFiGan<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), new CodeHiFiGanConfiguration(), new Random(0));
        var used = new HashSet<string>();
        vocoder.LoadTorchWeights(name =>
        {
            if (!weights.TryGetValue(name, out var t)) return null;
            used.Add(name);
            return t.ToVector().Select(v => (double)v).ToArray();
        }, "generator.");
        Assert.All(weights.Keys, k => Assert.Contains(k, used));
        Check(vocoder, Fixture("code_hifigan_realweights_reference.json"), 1e-8);
    }
}
