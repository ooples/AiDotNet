using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Classification;
using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.Classification;

/// <summary>
/// The classifier <see cref="CLAP{T}"/> is a task over <see cref="CLAPModel{T}"/>'s network, not a second
/// implementation: given the same weights it must produce the same embeddings and the same zero-shot scores.
/// </summary>
public class CLAPAdapterTests
{
    private static CLAPModelOptions Network() => new()
    {
        // LAION-CLAP at test size: two Swin stages over a 32x32 spectrogram image, a one-layer RoBERTa.
        StftWindowSize = 256, HopLength = 64, NumMelBands = 8, ClipSamples = 2048, SpecSize = 32,
        AudioEmbedDim = 8, AudioStageDepths = new[] { 1, 1 }, AudioStageHeads = new[] { 1, 2 }, SwinWindowSize = 4,
        TextHiddenDim = 16, TextEncoderLayers = 1, TextEncoderHeads = 2, TextIntermediateDim = 32,
        VocabSize = 32, MaxTextLength = 6, TextMaxPositions = 16, ProjectionDim = 8, Seed = 3,
        EnableFeatureFusion = false, SpecAugmentTimeStripes = 0, SpecAugmentFreqStripes = 0,
    };

    private static readonly string[] Labels = { "a", "b", "c" };

    private static CLAP<double> Classifier(
        AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? optimizer = null) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.MultiLabelClassification,
            inputSize: 2048, outputSize: Labels.Length) { RandomSeed = 11 },
        new CLAPOptions { Network = Network(), CustomLabels = Labels, WindowSize = 2048.5 / 48000, WindowOverlap = 0 },
        optimizer);

    private static CLAPModel<double> Model() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Embedding,
            inputSize: 2048, outputSize: 8) { RandomSeed = 5 },
        Network());

    private static Tensor<double> Wave(int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>([2048]);
        for (int i = 0; i < t.Length; i++) t[i] = Math.Sin(i * (0.01 + 0.002 * seed)) + 0.1 * (rng.NextDouble() - 0.5);
        return t;
    }

    private static int[] Tokenize(string s) => s switch { "a" => new[] { 1, 2 }, "b" => new[] { 3, 4, 5 }, _ => new[] { 6 } };

    // Materialize both, then copy CLAPModel's towers into the classifier, layer for layer.
    private static (CLAP<double> clap, CLAPModel<double> model) Twins()
    {
        var clap = Classifier();
        var model = Model();
        clap.SetTextPrompts(Labels, Tokenize);
        clap.ExtractAudioEmbedding(Wave(1));
        clap.Detect(Wave(1));
        model.EncodeAudio(Wave(1));
        model.EncodeText(Tokenize("a"));
        var audio = model.LayersReadOnly;
        var head = clap.LayersReadOnly;
        Assert.Equal(audio.Count + 1, head.Count);
        for (int i = 0; i < audio.Count; i++)
        {
            Assert.Equal(audio[i].GetType(), head[i].GetType());
            head[i].SetParameters(audio[i].GetParameters());
        }
        var text = model.TextEncoderLayersReadOnly;
        var clapText = clap.TextEncoderLayersReadOnly;
        Assert.Equal(text.Count, clapText.Count);
        for (int i = 0; i < text.Count; i++)
        {
            Assert.Equal(text[i].GetType(), clapText[i].GetType());
            clapText[i].SetParameters(text[i].GetParameters());
        }
        return (clap, model);
    }

    [Fact(Timeout = 120000)]
    public async Task AudioEmbedding_IsCLAPModels()
    {
        await Task.Yield();
        var (clap, model) = Twins();
        var expected = model.EncodeAudio(Wave(2)).ToVector();
        var actual = clap.ExtractAudioEmbedding(Wave(2)).ToVector();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-12, $"{i}: {expected[i]} vs {actual[i]}");
    }

    [Fact(Timeout = 120000)]
    public async Task ZeroShot_IsTheSoftmaxOfCLAPModelsScaledSimilarities()
    {
        await Task.Yield();
        var (clap, model) = Twins();
        var audio = Wave(4);
        var similarities = model.ZeroShotClassify(audio, Labels, Tokenize);
        double tau = Network().InitialTemperature;
        var exp = Labels.Select(l => Math.Exp(similarities[l] / tau)).ToArray();
        double sum = exp.Sum();
        var probabilities = clap.GetEventProbabilities(audio);
        Assert.Equal(new[] { 1, Labels.Length }, probabilities.Shape.ToArray());
        for (int i = 0; i < Labels.Length; i++)
            Assert.True(Math.Abs(exp[i] / sum - probabilities[0, i]) < 1e-9, $"{Labels[i]}: {exp[i] / sum} vs {probabilities[0, i]}");
    }

    [Fact(Timeout = 120000)]
    public async Task FineTuning_ReducesTheBinaryCrossEntropy()
    {
        await Task.Yield();
        // A constant-rate Adam: the paper's cosine schedule anneals to zero over the optimizer's MaxIterations,
        // which would make the test about the schedule rather than about the training path.
        var clap = Classifier(new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 1e-3, UseAdaptiveLearningRate = false, EnableGradientClipping = false,
            }));
        var input = new Tensor<double>([2, 2048]);
        var a = Wave(1); var b = Wave(7);
        for (int i = 0; i < 2048; i++) { input[0, i] = a[i]; input[1, i] = b[i]; }
        var target = new Tensor<double>([2, 3]);
        target[0, 0] = 1; target[0, 2] = 1; target[1, 1] = 1;
        double Loss()
        {
            var p = clap.Predict(input);
            double l = 0;
            for (int i = 0; i < p.Length; i++)
            {
                double q = Math.Min(Math.Max(p[i], 1e-12), 1 - 1e-12);
                l -= target[i] * Math.Log(q) + (1 - target[i]) * Math.Log(1 - q);
            }
            return l / p.Length;
        }
        double before = Loss();
        for (int step = 0; step < 60; step++) clap.Train(input, target);
        double after = Loss();
        // Measured 2.33 -> 1.87: the tower has no input normalization (LAION's HTSAT batch-normalizes the mel
        // bins first), so its dB-scale input saturates some logits; issue #2313 tracks it.
        Assert.True(after < 0.85 * before, $"BCE {before} -> {after}");
    }

    [Fact(Timeout = 120000)]
    public async Task Prompts_OutsideTheTextTower_AreRejected()
    {
        await Task.Yield();
        var clap = Classifier();
        Assert.Throws<ArgumentException>(() => clap.SetTextPrompts(new[] { "x" }, _ => new[] { 32 }));
        Assert.Throws<ArgumentException>(() => clap.SetTextPrompts(new[] { "x" }, _ => new int[7]));
        Assert.Throws<ArgumentException>(() => clap.SetTextPrompts(new[] { "x" }, _ => Array.Empty<int>()));
        clap.SetTextPrompts(new[] { "x", "y" }, _ => new[] { 1 });
        Assert.Equal(new[] { "x", "y" }, clap.ClassLabels);
        clap.SetTextPrompts(Array.Empty<string>(), _ => new[] { 1 });
        Assert.Equal(Labels, clap.ClassLabels);
    }

    // The text tower has dropout (TextDropoutRate 0.1) and every layer starts in training mode: inference must run it in
    // evaluation mode, or the same caption and the same clip score differently on every call.
    [Fact(Timeout = 120000)]
    public async Task Inference_IsDeterministic()
    {
        await Task.Yield();
        var (clap, model) = Twins();
        Assert.True(Network().TextDropoutRate > 0, "The test needs the text tower's dropout on.");
        var first = model.EncodeText(Tokenize("b")).ToVector();
        var second = model.EncodeText(Tokenize("b")).ToVector();
        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i], second[i]);
        var p1 = clap.GetEventProbabilities(Wave(4));
        var p2 = clap.GetEventProbabilities(Wave(4));
        for (int i = 0; i < p1.Length; i++) Assert.Equal(p1[i], p2[i]);
    }
}
