using System;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Paper-fidelity invariant for FT-Transformer (Gorishniy et al. 2021,
/// "Revisiting Deep Learning Models for Tabular Data"). Pins the POC-reported
/// save/load (GetParameters/SetParameters) round-trip defect.
/// </summary>
/// <remarks>
/// The round-trip is exercised via a forward pass (to resolve the lazy layer
/// shapes) rather than training — the classifier's backward/train path is a
/// separate, still-open item, so coupling this invariant to it would mask the
/// serialization fix this test targets.
/// </remarks>
public class FTTransformerClassifierTests
{
    private const int NumNumerical = 4;
    private const int NumClasses = 3;

    // Small, dropout-free config so the forward is deterministic (round-trip
    // prediction-equality is only meaningful when the forward is reproducible).
    private static FTTransformerOptions<double> Opts() => new()
    {
        EmbeddingDimension = 16,
        NumHeads = 2,
        NumLayers = 1,
        FeedForwardMultiplier = 2,
        DropoutRate = 0.0,
        AttentionDropoutRate = 0.0,
        ResidualDropoutRate = 0.0,
    };

    private static FTTransformerClassifier<double> NewModel() =>
        new(NumNumerical, NumClasses, Opts());

    private static Tensor<double> MakeInput(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new Tensor<double>([n, NumNumerical]);
        for (int i = 0; i < n; i++)
            for (int f = 0; f < NumNumerical; f++)
                x[i, f] = rng.NextDouble();
        return x;
    }

    // Linearly-separable dataset: class = i % NumClasses, with the class encoded in
    // the features (feature[cls] = 1, others 0) + tiny deterministic jitter.
    private static (Tensor<double> x, int[] y) MakeSeparable(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new Tensor<double>([n, NumNumerical]);
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            int cls = i % NumClasses;
            y[i] = cls;
            for (int f = 0; f < NumNumerical; f++)
                x[i, f] = (f == cls ? 1.0 : 0.0) + 0.01 * (rng.NextDouble() - 0.5);
        }
        return (x, y);
    }

    [Fact]
    public void SaveLoad_RoundTrip_PreservesParametersAndPredictions()
    {
        var original = NewModel();
        var x = MakeInput(6, seed: 1);

        // Forward once to resolve lazy layer shapes, then snapshot the params.
        var pOrig = original.PredictProbabilities(x).ToArray();
        var saved = original.GetParameters();
        Assert.True(saved.Length > 0);
        Assert.Equal(original.ParameterCount, saved.Length);

        // "Load" into a fresh model: resolve its shapes (forward), then set the
        // saved params. A mis-sliced SetParameters (base/head boundary off by the
        // head size) corrupts both halves and the round-trip below diverges.
        var loaded = NewModel();
        _ = loaded.PredictProbabilities(x);
        loaded.SetParameters(saved);

        var reread = loaded.GetParameters();
        Assert.Equal(saved.Length, reread.Length);
        for (int i = 0; i < saved.Length; i++)
            Assert.True(Math.Abs(saved[i] - reread[i]) < 1e-12,
                $"Round-trip parameter[{i}]={reread[i]} != saved {saved[i]} — SetParameters mis-sliced.");

        // The loaded model must now predict identically to the original — the
        // whole point of save/load.
        var pLoaded = loaded.PredictProbabilities(x).ToArray();
        Assert.Equal(pOrig.Length, pLoaded.Length);
        for (int i = 0; i < pOrig.Length; i++)
            Assert.True(Math.Abs(pOrig[i] - pLoaded[i]) < 1e-9,
                $"Loaded model probability[{i}]={pLoaded[i]} != original {pOrig[i]} — save/load not faithful.");
    }

    [Fact]
    public void Train_ReducesCrossEntropy_OnSeparableData_AndStaysFinite()
    {
        // The classifier train path: TrainStep runs a tape-tracked forward +
        // softmax cross-entropy, backprops to every weight (tokenizer + transformer
        // + head), and applies SGD. On a separable task the loss must drop and stay
        // finite. (Previously TrainStep had no backward and threw on UpdateParameters.)
        var model = NewModel();
        var (x, y) = MakeSeparable(9, seed: 2);

        double firstLoss = Convert.ToDouble(model.TrainStep(x, y, learningRate: 0.05));
        double lastLoss = firstLoss;
        for (int step = 0; step < 60; step++)
        {
            lastLoss = Convert.ToDouble(model.TrainStep(x, y, learningRate: 0.05));
            Assert.False(double.IsNaN(lastLoss), $"Loss NaN at step {step} — training diverged.");
            Assert.False(double.IsInfinity(lastLoss), $"Loss Inf at step {step} — training diverged.");
        }

        Assert.True(lastLoss < firstLoss,
            $"Cross-entropy did not decrease (first={firstLoss:F4}, last={lastLoss:F4}) — classifier train path broken.");

        var p = model.GetParameters();
        for (int i = 0; i < p.Length; i++)
            Assert.False(double.IsNaN(p[i]) || double.IsInfinity(p[i]),
                $"Parameter[{i}] non-finite after training.");
    }
}
