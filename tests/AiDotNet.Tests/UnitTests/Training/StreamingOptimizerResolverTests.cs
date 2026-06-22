using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

public sealed class StreamingOptimizerResolverTests
{
    public static IEnumerable<object[]> OptimizerMappings()
    {
        yield return Case("Adam", typeof(StreamingAdam8Bit<float>));
        yield return Case("AdamAmsGrad", typeof(StreamingAMSGrad8Bit<float>));
        yield return Case("AdamW", typeof(StreamingAdamW8Bit<float>));
        yield return Case("AMSGrad", typeof(StreamingAMSGrad8Bit<float>));
        yield return Case("Nadam", typeof(StreamingNadam8Bit<float>));
        yield return Case("AdaMax", typeof(StreamingAdaMax8Bit<float>));
        yield return Case("Lion", typeof(StreamingLion8Bit<float>));
        yield return Case("LAMB", typeof(StreamingLamb8Bit<float>));
        yield return Case("LARS", typeof(StreamingLars8Bit<float>));
        yield return Case("FTRL", typeof(StreamingFtrl8Bit<float>));
        yield return Case("AdaDelta", typeof(StreamingAdaDelta8Bit<float>));
        yield return Case("Adagrad", typeof(StreamingAdagrad8Bit<float>));
        yield return Case("RMSProp", typeof(StreamingRmsProp8Bit<float>));
        yield return Case("Momentum", typeof(StreamingMomentum8Bit<float>));
        yield return Case("Nesterov", typeof(StreamingNesterov8Bit<float>));
        yield return Case("GradientDescent", typeof(StreamingSgd8Bit<float>));
        yield return Case("SGD", typeof(StreamingSgd8Bit<float>));
        yield return Case("MiniBatchGD", typeof(StreamingSgd8Bit<float>));
        yield return Case("ProximalGD", typeof(StreamingSgd8Bit<float>));
        // Newly-covered gradient-based optimizers whose streaming form updates in place.
        yield return Case("Adam8Bit", typeof(StreamingAdam8Bit<float>));
        yield return Case("ADMM", typeof(StreamingSgd8Bit<float>));
        yield return Case("CoordinateDescent", typeof(StreamingSgd8Bit<float>));
    }

    // Full-gradient optimizers (Hessian-free CG, and the dense-Hessian family mapped to the
    // memory-bounded limited-memory quasi-Newton) buffer per-parameter gradients and update only in
    // EndStep, so they are verified for TYPE mapping here rather than via the in-place-Apply theory.
    public static IEnumerable<object[]> FullGradientMappings()
    {
        yield return Case("ConjugateGradient", typeof(StreamingConjugateGradient<float>));
        yield return Case("BFGS", typeof(StreamingLBFGS<float>));
        yield return Case("DFP", typeof(StreamingLBFGS<float>));
        yield return Case("Newton", typeof(StreamingLBFGS<float>));
        yield return Case("TrustRegion", typeof(StreamingLBFGS<float>));
        yield return Case("LevenbergMarquardt", typeof(StreamingLBFGS<float>));
    }

    [Theory]
    [MemberData(nameof(OptimizerMappings))]
    public void Resolver_MapsFirstOrderOptimizers_ToStreamingVariants(
        string optimizerName,
        Type expectedStreamingType)
    {
        var optimizer = CreateOptimizer(optimizerName);
        var streaming = StreamingOptimizerResolver<float>.Create(
            optimizer,
            useStreamingDefaults: false,
            fallbackLearningRate: 0.01,
            fallbackWeightDecay: 0.0);

        Assert.Equal(expectedStreamingType, streaming.GetType());
    }

    [Theory]
    [MemberData(nameof(OptimizerMappings))]
    public void ResolvedStreamingOptimizer_AppliesFiniteInPlaceUpdate(
        string optimizerName,
        Type _)
    {
        var optimizer = CreateOptimizer(optimizerName);
        var streaming = StreamingOptimizerResolver<float>.Create(
            optimizer,
            useStreamingDefaults: false,
            fallbackLearningRate: 0.01,
            fallbackWeightDecay: 0.0);

        var param = new Tensor<float>(new[] { 6 });
        var grad = new Tensor<float>(new[] { 6 });
        for (int i = 0; i < param.Length; i++)
        {
            param[i] = 0.25f + i * 0.1f;
            grad[i] = (i % 2 == 0 ? 0.2f : -0.15f) + i * 0.01f;
        }

        var before = new float[param.Length];
        for (int i = 0; i < param.Length; i++)
            before[i] = param[i];

        streaming.BeginStep();
        streaming.Apply(param, grad);

        bool changed = false;
        for (int i = 0; i < param.Length; i++)
        {
            Assert.False(float.IsNaN(param[i]));
            Assert.False(float.IsInfinity(param[i]));
            changed |= Math.Abs(param[i] - before[i]) > 1e-8f;
        }

        Assert.True(changed, $"{streaming.GetType().Name} did not update any parameter.");
    }

    [Theory]
    [MemberData(nameof(FullGradientMappings))]
    public void Resolver_MapsFullGradientOptimizers_ToStreamingTypes(
        string optimizerName,
        Type expectedStreamingType)
    {
        var optimizer = CreateOptimizer(optimizerName);
        var streaming = StreamingOptimizerResolver<float>.Create(
            optimizer,
            useStreamingDefaults: false,
            fallbackLearningRate: 0.01,
            fallbackWeightDecay: 0.0);

        Assert.Equal(expectedStreamingType, streaming.GetType());
    }

    // The optimizer ctors require a non-null model, but the resolver only inspects the optimizer's
    // runtime type and its options — the model is never touched. Pass null (a supported sentinel)
    // rather than the banned null-forgiving operator, and silence the nullable-argument warning here.
#pragma warning disable CS8625
    private static IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> CreateOptimizer(string optimizerName)
        => optimizerName switch
        {
            "Adam" => new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                UseAMSGrad = false
            }),
            "AdamAmsGrad" => new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                UseAMSGrad = true
            }),
            "AdamW" => new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null, new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                WeightDecay = 0.01
            }),
            "AMSGrad" => new AMSGradOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "Nadam" => new NadamOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "AdaMax" => new AdaMaxOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "Lion" => new LionOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "LAMB" => new LAMBOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "LARS" => new LARSOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "FTRL" => new FTRLOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { Alpha = 0.01, Lambda1 = 0.0 }),
            "AdaDelta" => new AdaDeltaOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "Adagrad" => new AdagradOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "RMSProp" => new RootMeanSquarePropagationOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "Momentum" => new MomentumOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "Nesterov" => new NesterovAcceleratedGradientOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "GradientDescent" => new GradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "SGD" => new StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "MiniBatchGD" => new MiniBatchGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "ProximalGD" => new ProximalGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null, new() { InitialLearningRate = 0.01 }),
            "Adam8Bit" => new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(null, new()),
            "ConjugateGradient" => new ConjugateGradientOptimizer<float, Tensor<float>, Tensor<float>>(null, new()),
            "BFGS" => new BFGSOptimizer<float, Tensor<float>, Tensor<float>>(null, new()),
            "DFP" => new DFPOptimizer<float, Tensor<float>, Tensor<float>>(null, new()),
            "Newton" => new NewtonMethodOptimizer<float, Tensor<float>, Tensor<float>>(null, new()),
            "TrustRegion" => new TrustRegionOptimizer<float, Tensor<float>, Tensor<float>>(null, new()),
            "LevenbergMarquardt" => new LevenbergMarquardtOptimizer<float, Tensor<float>, Tensor<float>>(null, new()),
            "ADMM" => new ADMMOptimizer<float, Tensor<float>, Tensor<float>>(null, new()),
            "CoordinateDescent" => new CoordinateDescentOptimizer<float, Tensor<float>, Tensor<float>>(null, new()),
            _ => throw new ArgumentOutOfRangeException(nameof(optimizerName), optimizerName, "Unknown optimizer mapping.")
        };
#pragma warning restore CS8625

    private static object[] Case(string optimizerName, Type expectedStreamingType)
        => new object[] { optimizerName, expectedStreamingType };
}
