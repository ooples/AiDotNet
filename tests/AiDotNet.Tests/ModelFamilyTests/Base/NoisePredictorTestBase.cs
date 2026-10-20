using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for diffusion noise predictors (DiT, U-ViT, MMDiT, UNet and their variants), driven through
/// <see cref="INoisePredictor{T}.PredictNoise"/> (#2139).
/// </summary>
/// <remarks>
/// <para>
/// Noise predictors were excluded from generation as internal components. They are public, constructible models, and
/// writing this base found that U-ViT reported cross-attention support while never reading its conditioning.
/// </para>
/// <para>
/// The conditioning invariants run on seeded random parameters, not on the initialization. Several of these
/// architectures initialize adaLN-Zero (DiT, Sec. 3.2): every modulation starts at zero, so an untrained predictor
/// legitimately ignores the timestep and the condition. Random parameters test the wiring, which is what can be
/// wrong.
/// </para>
/// </remarks>
public abstract class NoisePredictorTestBase
{
    /// <summary>Subclasses return their concrete predictor, at a fixture size.</summary>
    protected abstract INoisePredictor<double> CreatePredictor();

    /// <summary>The noisy-sample shape fed to the predictor.</summary>
    protected virtual int[] SampleShape(INoisePredictor<double> predictor) => new[] { 1, predictor.InputChannels, 8, 8 };

    private static Tensor<double> Random(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static Tensor<double>? Condition(INoisePredictor<double> predictor, int seed) =>
        predictor.SupportsCrossAttention ? Random(new[] { 1, 4, predictor.ContextDimension }, seed) : null;

    private static void RandomizeParameters(INoisePredictor<double> predictor)
    {
        var current = predictor.GetParameters();
        var rng = RandomHelper.CreateSeededRandom(2139);
        var values = new Vector<double>(current.Length);
        for (int i = 0; i < values.Length; i++) values[i] = (rng.NextDouble() * 2 - 1) * 0.05;
        predictor.SetParameters(values);
    }

    private static double MaxDifference(Tensor<double> a, Tensor<double> b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    [Fact(Timeout = 120000)]
    public async Task PredictNoise_KeepsTheSampleLayoutAndIsFinite()
    {
        await Task.Yield();
        var predictor = CreatePredictor();
        var shape = SampleShape(predictor);

        var noise = predictor.PredictNoise(Random(shape, 1), 10, Condition(predictor, 2));

        var expected = (int[])shape.Clone();
        expected[1] = predictor.OutputChannels;
        Assert.Equal(expected, noise.Shape.ToArray());
        for (int i = 0; i < noise.Length; i++)
            Assert.False(double.IsNaN(noise[i]) || double.IsInfinity(noise[i]), $"noise[{i}] is {noise[i]}.");
    }

    [Fact(Timeout = 120000)]
    public async Task PredictNoise_IsDeterministic()
    {
        await Task.Yield();
        var predictor = CreatePredictor();
        var sample = Random(SampleShape(predictor), 3);
        var condition = Condition(predictor, 4);

        var first = predictor.PredictNoise(sample, 10, condition);
        var second = predictor.PredictNoise(sample, 10, condition);

        Assert.Equal(0.0, MaxDifference(first, second));
    }

    [Fact(Timeout = 120000)]
    public async Task PredictNoise_DependsOnTheTimestep()
    {
        await Task.Yield();
        var predictor = CreatePredictor();
        RandomizeParameters(predictor);
        var sample = Random(SampleShape(predictor), 5);
        var condition = Condition(predictor, 6);

        double difference = MaxDifference(predictor.PredictNoise(sample, 10, condition), predictor.PredictNoise(sample, 500, condition));

        Assert.True(difference > 0, "The predicted noise is the same at timesteps 10 and 500: the timestep is not wired in.");
    }

    [SkippableFact(Timeout = 120000)]
    public async Task PredictNoise_DependsOnTheConditioning_WhenCrossAttentionIsSupported()
    {
        await Task.Yield();
        var predictor = CreatePredictor();
        Skip.IfNot(predictor.SupportsCrossAttention, "The predictor is unconditional.");
        RandomizeParameters(predictor);
        var sample = Random(SampleShape(predictor), 7);

        double difference = MaxDifference(predictor.PredictNoise(sample, 10, Condition(predictor, 8)),
            predictor.PredictNoise(sample, 10, Condition(predictor, 9)));

        Assert.True(difference > 0,
            "SupportsCrossAttention is true, but two different conditionings give the same noise: the condition is never read.");
    }
}
