using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Classification;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// The DCASE 2017 Task 1 baseline (Mesaros et al. 2017, sections 2.2 and 3): log mel-band energies in 40 ms frames with
/// 50% overlap, a 5-frame context, a two-layer 50-unit MLP with 20% dropout, and frame majority voting (#2397).
/// </summary>
public class SceneClassifierPaperTests
{
    private static Tensor<double> Recording(double seconds, double pitch, int seed)
    {
        var noise = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        int samples = (int)(seconds * 44100);
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.3 * Math.Sin(2 * Math.PI * pitch * i / 44100.0) + 0.05 * (noise.NextDouble() - 0.5);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheBaselineSystem()
    {
        await Task.Yield();
        var o = new SceneClassifierOptions();
        Assert.Equal((44100, 1764, 882, 40), (o.SampleRate, o.FftSize, o.HopLength, o.NumMels));
        Assert.Equal((5, 50, 2, 0.2), (o.ContextFrames, o.HiddenUnits, o.NumHiddenLayers, o.DropoutRate));
        Assert.Equal(15, SceneClassifier<double>.StandardScenes.Length);
    }

    [Fact(Timeout = 60000)]
    public async Task Network_IsTwoDenseLayersOf50WithDropout_UnderTheSceneLayer()
    {
        await Task.Yield();
        using var classifier = new SceneClassifier<double>();
        var layers = classifier.Layers;
        Assert.Equal(5, layers.Count);
        for (int i = 0; i < 2; i++)
        {
            var dense = Assert.IsType<DenseLayer<double>>(layers[2 * i]);
            Assert.IsType<ReLUActivation<double>>(dense.ScalarActivation);
            Assert.Equal(0.2, double.Parse(Assert.IsType<DropoutLayer<double>>(layers[2 * i + 1]).GetMetadata()["DropoutRate"], System.Globalization.CultureInfo.InvariantCulture), 12);
        }
        var output = classifier.Predict(classifier.ContextFeatures(Recording(0.2, 300, 1)));
        Assert.Equal(15, output.Shape[^1]);
        Assert.Equal(50, layers[0].GetOutputShape()[^1]);
    }

    [Fact(Timeout = 60000)]
    public async Task ContextFeatures_StackFiveFramesOfFortyBands_CentredOnTheFrame()
    {
        await Task.Yield();
        using var classifier = new SceneClassifier<double>();
        var features = classifier.ContextFeatures(Recording(1.0, 300, 2));
        // 1 s at a 20 ms hop is 51 frames; 5 x 40 = 200 values each.
        Assert.Equal(new[] { 51, 200 }, features.Shape.ToArray());
        // Row 10 holds frames 8..12, so its block k equals block k - 1 of row 11.
        for (int k = 1; k < 5; k++)
            for (int b = 0; b < 40; b++)
                Assert.Equal(features[10, k * 40 + b], features[11, (k - 1) * 40 + b], 12);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_SeparatesTwoScenes_AndTheRecordingTakesTheMajorityVote()
    {
        await Task.Yield();
        using var classifier = new SceneClassifier<double>(new SceneClassifierOptions { CustomScenes = new[] { "low", "high" }, DropoutRate = 0.0, Seed = 3 });
        var low = classifier.ContextFeatures(Recording(0.5, 200, 3));
        var high = classifier.ContextFeatures(Recording(0.5, 2000, 4));
        int frames = low.Shape[0];
        var inputs = new Tensor<double>(new[] { 2 * frames, 200 });
        var targets = new Tensor<double>(new[] { 2 * frames, 2 });
        for (int f = 0; f < frames; f++)
        {
            for (int k = 0; k < 200; k++) { inputs[f, k] = low[f, k]; inputs[frames + f, k] = high[f, k]; }
            targets[f, 0] = 1.0;
            targets[frames + f, 1] = 1.0;
        }
        for (int step = 0; step < 100; step++) classifier.Train(inputs, targets);

        var lowResult = classifier.Classify(Recording(0.5, 200, 5));
        Assert.Equal("low", lowResult.PredictedScene);
        // The confidence is the winning scene's share of the frame votes.
        var shares = classifier.GetSceneProbabilities(Recording(0.5, 200, 5));
        Assert.Equal(1.0, shares.Values.Sum(), 9);
        Assert.Equal(shares["low"], lowResult.Confidence, 12);
        Assert.Equal("high", classifier.Classify(Recording(0.5, 2000, 6)).PredictedScene);
    }
}
