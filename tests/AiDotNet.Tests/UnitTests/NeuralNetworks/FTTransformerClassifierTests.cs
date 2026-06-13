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
}
