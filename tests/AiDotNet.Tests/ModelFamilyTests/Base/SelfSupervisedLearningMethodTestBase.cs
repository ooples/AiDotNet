using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.SelfSupervisedLearning;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Small, real parts for self-supervised fixtures: the methods wrap an encoder and heads the caller supplies (#2139).
/// </summary>
public static class SslFixtureParts
{
    /// <summary>The seeded configuration every fixture uses, so a run is reproducible.</summary>
    public static SelfSupervisedLearningConfig<double> Config() => new() { Seed = 1 };

    /// <summary>A two-layer encoder, already materialized.</summary>
    public static NeuralNetwork<double> Encoder(int inputSize = 5, int outputSize = 8)
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(16, activationFunction: new ReLUActivation<double>()),
            new DenseLayer<double>(outputSize, activationFunction: (IActivationFunction<double>?)null),
        };
        var encoder = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: inputSize, outputSize: outputSize,
            layers: layers));
        _ = encoder.Predict(new Tensor<double>(new[] { 1, inputSize }));
        return encoder;
    }

    /// <summary>A momentum copy of a fresh encoder.</summary>
    public static MomentumEncoder<double> Momentum() => new(Encoder());

    /// <summary>An MLP projection head from the encoder's 8 outputs.</summary>
    public static MLPProjector<double> Mlp(int seed) => new(8, 16, 8, seed: seed);

    /// <summary>A projector, with a predictor head when <paramref name="withPredictor"/> is set.</summary>
    public static SymmetricProjector<double> Symmetric(int seed, bool withPredictor) =>
        new(8, 16, 8, withPredictor ? 16 : 0, seed: seed);
}

/// <summary>
/// Family invariants for self-supervised learning methods, driven through <see cref="ISelfSupervisedLearningMethod{T}.TrainStep"/> (#2139).
/// </summary>
/// <remarks>
/// Writing this family found that no method trained: TrainStep returned a zero loss and changed nothing for all ten.
/// These invariants hold for every method, contrastive, non-contrastive, self-distillation or generative: a training step
/// reports a real (finite, non-zero) loss and moves the encoder, and encoding is a deterministic read that changes
/// nothing.
/// </remarks>
public abstract class SelfSupervisedLearningMethodTestBase
{
    /// <summary>Subclasses return their concrete method, built from <see cref="SslFixtureParts"/>.</summary>
    protected abstract ISelfSupervisedLearningMethod<double> CreateMethod();

    /// <summary>The batch shape: 8 feature vectors, or 8 one-channel 4x4 images for a reconstruction method.</summary>
    protected virtual int[] BatchShape(ISelfSupervisedLearningMethod<double> method) =>
        method is MAE<double> ? new[] { 8, 1, 4, 4 } : new[] { 8, 5 };

    private Tensor<double> Batch(ISelfSupervisedLearningMethod<double> method, int seed)
    {
        var t = new Tensor<double>(BatchShape(method));
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static double MaxDifference(Vector<double> a, Vector<double> b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    [Fact(Timeout = 120000)]
    public async Task TrainStep_ReportsAFiniteNonZeroLoss()
    {
        await Task.Yield();
        var method = CreateMethod();
        var batch = Batch(method, 1);

        for (int step = 0; step < 3; step++)
        {
            double loss = Convert.ToDouble(method.TrainStep(batch).Loss);
            Assert.False(double.IsNaN(loss) || double.IsInfinity(loss), $"Step {step + 1} loss is {loss}.");
            Assert.True(loss != 0.0, $"Step {step + 1} reported a zero loss: the method did not compute its objective.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task TrainStep_MovesTheEncoder()
    {
        await Task.Yield();
        var method = CreateMethod();
        var batch = Batch(method, 2);
        var before = method.GetEncoder().GetParameters();

        for (int step = 0; step < 3; step++) method.TrainStep(batch);

        Assert.True(MaxDifference(before, method.GetEncoder().GetParameters()) > 0,
            "Three training steps left every encoder parameter unchanged.");
    }

    [Fact(Timeout = 120000)]
    public async Task Encode_IsDeterministicAndChangesNothing()
    {
        await Task.Yield();
        var method = CreateMethod();
        var batch = Batch(method, 3);
        var parameters = method.GetParameters();

        var first = method.Encode(batch);
        var second = method.Encode(batch);

        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i], second[i]);
        Assert.Equal(0.0, MaxDifference(parameters, method.GetParameters()));
    }
}
