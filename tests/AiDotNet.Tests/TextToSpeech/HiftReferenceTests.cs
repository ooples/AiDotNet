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
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The HiFT generator (CosyVoice's vocoder) against FunAudioLLM/CosyVoice's own <c>HiFTGenerator</c>
/// (<c>tools/reference-data/hift_reference.py</c>): the F0 predictor, the merged excitation of given harmonics, the
/// waveform <c>decode</c> makes, and the waveform after one SGD step.
/// </summary>
public class HiftReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "hift_reference.json";
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

    private static (HiftGenerator<double> Model, List<LayerBase<double>> Layers, JObject Fixture) Load()
    {
        var f = Reference();
        var c = f["config"]!;
        int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
        var configuration = new HiftConfiguration(
            InChannels: (int)c["in_channels"]!, BaseChannels: (int)c["base_channels"]!, Harmonics: (int)c["nb_harmonics"]!,
            SampleRate: (int)c["sampling_rate"]!, NsfAlpha: (double)c["nsf_alpha"]!, NsfSigma: (double)c["nsf_sigma"]!,
            VoicedThreshold: (double)c["nsf_voiced_threshold"]!, UpsampleRates: Ints(c["upsample_rates"]!),
            UpsampleKernels: Ints(c["upsample_kernel_sizes"]!), FftSize: (int)c["istft_params"]!["n_fft"]!,
            HopSize: (int)c["istft_params"]!["hop_len"]!, ResblockKernels: Ints(c["resblock_kernel_sizes"]!),
            ResblockDilations: c["resblock_dilation_sizes"]!.Select(Ints).ToArray(), SourceKernels: Ints(c["source_resblock_kernel_sizes"]!),
            SourceDilations: c["source_resblock_dilation_sizes"]!.Select(Ints).ToArray(), LeakySlope: (double)c["lrelu_slope"]!,
            AudioLimit: (double)c["audio_limit"]!, F0Channels: (int)f["f0_channels"]!);
        var layers = new List<LayerBase<double>>();
        var model = new HiftGenerator<double>(AiDotNetEngine.Current, layers, configuration, new Random(0));
        string path = Path.Combine(Path.GetTempPath(), $"hift_reference_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)f["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            model.LoadTorchWeights((name, shape) =>
            {
                var tensor = weights[name];
                Assert.Equal(shape.Aggregate(1, (a, b) => a * b), tensor.Length);
                used.Add(name);
                return tensor.ToVector().Select(v => (double)v).ToArray();
            });
            Assert.Equal(weights.Keys.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
        }
        finally
        {
            File.Delete(path);
        }
        return (model, layers, f);
    }

    private static void AssertClose(JToken expected, Tensor<double> actual, string what)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(values[i])),
                $"{what}[{i}]: reference {values[i]:R}, port {actual[i]:R}.");
    }

    private static (Tensor<double> Mel, Tensor<double> Harmonics) Inputs(JObject f, HiftConfiguration c)
    {
        var mel = f["mel"]!.Select(v => (double)v).ToArray();
        int frames = mel.Length / c.InChannels, samples = frames * c.UpsampleScale, h = c.Harmonics + 1;
        var raw = f["harmonics"]!.Select(v => (double)v).ToArray();                                    // [1, samples, h]
        var harmonics = new Tensor<double>(new[] { 1, h, samples });
        for (int s = 0; s < samples; s++) for (int k = 0; k < h; k++) harmonics[0, k, s] = raw[s * h + k];
        return (new Tensor<double>(new[] { 1, c.InChannels, frames }, new Vector<double>(mel)), harmonics);
    }

    [Fact(Timeout = 120000)]
    public async Task F0SourceAndWaveform_MatchCosyVoice()
    {
        await Task.Yield();
        var (model, _, f) = Load();
        var (mel, harmonics) = Inputs(f, model.Configuration);
        AssertClose(f["f0"]!, model.PredictF0(mel), "f0");
        var source = model.Merge(harmonics);
        AssertClose(f["source"]!, source, "source");
        AssertClose(f["audio"]!, model.Decode(mel, source), "audio");
    }

    [Fact(Timeout = 120000)]
    public async Task OneSgdStep_MatchesCosyVoice()
    {
        await Task.Yield();
        var (model, layers, f) = Load();
        var (mel, harmonics) = Inputs(f, model.Configuration);
        Tensor<double> source;
        using (new NoGradScope<double>()) source = model.Merge(harmonics);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var audio = model.Decode(mel, source);
            var engine = AiDotNetEngine.Current;
            var loss = engine.ReduceMean(engine.TensorMultiply(audio, audio), Enumerable.Range(0, audio.Rank).ToArray(), false);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        double rate = (double)f["learning_rate"]!;
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= rate * g[i];
            AiDotNetEngine.Current.InvalidatePersistentTensor(p);
        }
        AssertClose(f["audio_after_sgd"]!, model.Decode(mel, source), "audio after the step");
    }
}
