using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Foundations;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.StyleEmotion;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// StyleTTS 2's training objectives reproduce yl4579/StyleTTS2's: WavLM's hidden states (transformers' WavLMModel), the
/// 24 → 16 kHz resampling (torchaudio), the multi-resolution mel loss, the least-squares adversarial, feature-matching and
/// TPR losses, the WavLM feature-matching and SLM adversarial losses, and differentiable duration modeling.
/// </summary>
/// <remarks><c>ReferenceData/styletts2_training_reference.json</c> (<c>tools/reference-data/styletts2_training_reference.py</c>).</remarks>
public class StyleTTS2TrainingPaperTests
{
    private static readonly AiDotNet.Tensors.Engines.IEngine Engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;

    private static JObject Reference()
    {
        const string fileName = "styletts2_training_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static double[] Values(JToken token) => token.Select(v => (double)v).ToArray();

    private static Tensor<double> Make(double[] values, params int[] shape) => new(shape, new Vector<double>(values));

    private static void Close(double[] expected, Tensor<double> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-8 * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: reference {expected[i]:R}, port {actual[i]:R}.");
    }

    private static (WavLmNetwork<double> WavLm, StyleTts2SlmDiscriminator<double> Discriminator, JObject Fixture) Load()
    {
        var fixture = Reference();
        var shared = fixture["shared"]!;
        var w = shared["wavlm"]!;
        var d = shared["wd"]!;
        var layers = new List<LayerBase<double>>();
        var wavlm = new WavLmNetwork<double>(Engine, layers, new WavLmConfiguration
        {
            ConvDims = w["conv_dim"]!.Select(v => (int)v).ToArray(),
            HiddenSize = (int)w["hidden_size"]!, Layers = (int)w["num_hidden_layers"]!, Heads = (int)w["num_attention_heads"]!,
            IntermediateSize = (int)w["intermediate_size"]!, PositionalConvKernel = (int)w["num_conv_pos_embeddings"]!,
            PositionalConvGroups = (int)w["num_conv_pos_embedding_groups"]!, RelativePositionBuckets = (int)w["num_buckets"]!,
            RelativePositionMaxDistance = (int)w["max_bucket_distance"]!,
        });
        var discriminator = new StyleTts2SlmDiscriminator<double>(Engine, layers, (int)d["slm_hidden"]!, (int)d["slm_layers"]!, (int)d["initial_channel"]!);
        // Size the lazily built layers before loading.
        wavlm.HiddenStates(new Tensor<double>(new[] { 800 }));
        discriminator.Forward(new Tensor<double>(new[] { 1, (int)d["slm_hidden"]! * (int)d["slm_layers"]!, 4 }));
        string path = Path.Combine(Path.GetTempPath(), $"styletts2_training_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            double[] Read(string name, int[] shape)
            {
                Assert.True(weights.ContainsKey(name), $"The reference has no tensor '{name}'.");
                Assert.Equal(shape.Aggregate(1, (a, b) => a * b), weights[name].Length);
                used.Add(name);
                return weights[name].ToVector().Select(v => (double)v).ToArray();
            }
            wavlm.LoadTorchWeights((name, shape) => Read("wavlm." + name, shape), name => weights.ContainsKey("wavlm." + name));
            discriminator.Load((name, shape) => Read(name, shape), "wd");
            // The time-mask embedding is used only by pre-training's masking.
            var unused = weights.Keys.Where(k => !used.Contains(k) && k != "wavlm.masked_spec_embed").ToList();
            Assert.True(unused.Count == 0, "Unread reference tensors: " + string.Join(", ", unused));
        }
        finally
        {
            File.Delete(path);
        }
        return (wavlm, discriminator, fixture);
    }

    private static List<(Tensor<double> Score, List<Tensor<double>> Features)> Outputs(JToken scores, JToken maps, JToken shapes)
    {
        var result = new List<(Tensor<double>, List<Tensor<double>>)>();
        var s = scores.ToArray();
        var m = maps.ToArray();
        var sh = shapes.ToArray();
        for (int d = 0; d < s.Length; d++)
        {
            var score = Values(s[d]);
            var features = m[d].Select((map, l) => Make(Values(map), sh[d][l]!.Select(v => (int)v).ToArray())).ToList();
            result.Add((Make(score, 1, score.Length), features));
        }
        return result;
    }

    [Fact(Timeout = 300000)]
    public async Task SpeechLanguageModelAndLosses_MatchTheReference()
    {
        await Task.Yield();
        var (wavlm, discriminator, fixture) = Load();
        var shared = fixture["shared"]!;
        var losses = new StyleTts2Losses<double>(Engine, 24000, 16000);
        var real = Values(shared["real"]!);
        var fake = Values(shared["fake"]!);
        var realWave = Make(real, real.Length);
        var fakeWave = Make(fake, fake.Length);

        var resampled = losses.Resample(realWave);
        Close(Values(fixture["resampled"]!), resampled, "resampled");
        var states = wavlm.HiddenStates(resampled);
        var expectedStates = fixture["wavlm_states"]!.ToArray();
        Assert.Equal(expectedStates.Length, states.Count);
        for (int i = 0; i < states.Count; i++) Close(Values(expectedStates[i]), states[i], $"WavLM state {i}");

        Close(Values(fixture["slm_feature"]!), losses.SlmFeatureMatching(wavlm, realWave, fakeWave), "WavLM feature matching");
        Close(Values(fixture["slm_generator"]!), losses.SlmGenerator(wavlm, discriminator, fakeWave), "SLM generator loss");
        Close(Values(fixture["slm_discriminator"]!), losses.SlmDiscriminator(wavlm, discriminator, realWave, fakeWave), "SLM discriminator loss");
        Close(Values(fixture["slm_scores"]!), losses.SlmScores(wavlm, discriminator, realWave), "SLM scores");

        Close(Values(fixture["mel_loss"]!), losses.MultiResolutionMel(fakeWave, realWave), "multi-resolution mel loss");

        var onReal = Outputs(shared["scores_real"]!, shared["maps_real"]!, shared["map_shapes"]!);
        var onFake = Outputs(shared["scores_fake"]!, shared["maps_fake"]!, shared["map_shapes"]!);
        Close(Values(fixture["generator_loss"]!), losses.Generator(onReal, onFake), "generator loss");
        Close(Values(fixture["discriminator_loss"]!), losses.Discriminator(onReal, onFake), "discriminator loss");
    }

    [Fact(Timeout = 120000)]
    public async Task DifferentiableDurationModeling_MatchesTheReference()
    {
        await Task.Yield();
        var fixture = Reference();
        var shared = fixture["shared"]!;
        var shape = shared["logits_shape"]!.Select(v => (int)v).ToArray();
        var logits = Make(Values(shared["logits"]!), shape);
        var losses = new StyleTts2Losses<double>(Engine, 24000, 16000);
        int frames = (int)fixture["alignment_frames"]!;
        Close(Values(fixture["alignment"]!), losses.DifferentiableAlignment(logits, frames, (double)shared["sigma"]!), "alignment");
    }
}
