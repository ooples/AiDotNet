using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// <b>#1413 consolidation contract</b>: verifies that the unified
/// UpdateSolution → Step delegation produces sensible, finite parameter
/// updates on a real NeuralNetwork — the regression bar that locks the
/// consolidation in place. If any optimizer subclass shortcircuits the
/// base.UpdateSolution path, its parameter delta would diverge from the
/// known-good Step contract.
///
/// <para>Mirrors the empirical bar from cheatcountry/HarmonicEngine
/// Phase_PAPER_A_BuildAsyncDiagnostic_1380Bisect (Test 5: split data
/// 800/171/173 — shipped 0.206.0 NaN, after this PR finite ≈ 36-50).</para>
/// </summary>
public class Issue1413_UpdateSolutionConsolidationTests
{
    private readonly ITestOutputHelper _output;

    public Issue1413_UpdateSolutionConsolidationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Theory]
    [InlineData("Adam")]
    [InlineData("AdamW")]
    [InlineData("AdaMax")]
    [InlineData("AMSGrad")]
    [InlineData("SGD")]
    [InlineData("Momentum")]
    [InlineData("Nadam")]
    public void Optimizer_UpdateSolution_On_Transformer_Produces_Finite_Params(string optimizerName)
    {
        // Build a tiny Transformer (the original #1380 reproducer
        // architecture, scaled down for fast CI).
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 1, numDecoderLayers: 0, numHeads: 2,
            modelDimension: 32, feedForwardDimension: 64,
            complexity: NetworkComplexity.Simple,
            inputSize: 8, outputSize: 16,
            dropoutRate: 0.0, maxSequenceLength: 8, vocabularySize: 16,
            usePositionalEncoding: true, temperature: 1.0, sequencePooling: null,
            randomSeed: 42);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var initParams = model.GetParameters();
        // Snapshot the pre-step parameter values so we can assert that
        // UpdateSolution actually changed at least one parameter. Without
        // this, a UpdateSolution that's a silent no-op would still pass
        // every assertion below (initL2 == finalL2 satisfies the
        // "finite + ΔL2 < 10×initL2" bounds trivially), defeating the
        // entire point of this consolidation-regression guard.
        var initParamsSnapshot = new float[initParams.Length];
        for (int i = 0; i < initParams.Length; i++) initParamsSnapshot[i] = initParams[i];
        double initL2 = ComputeL2(initParams);
        Assert.False(double.IsNaN(initL2), "Init L2 NaN");
        Assert.True(initL2 > 0, "Init L2 zero — model not initialized");

        // Construct a deterministic gradient vector matching the model's
        // total parameter count, with small magnitudes (so even a buggy
        // optimizer shouldn't blow up on a single step).
        int totalParams = initParams.Length;
        var flatGrad = new Vector<float>(totalParams);
        for (int i = 0; i < totalParams; i++) flatGrad[i] = 0.0001f * ((i % 11) - 5);

        // Construct the named optimizer with deterministic options.
        var optimizer = BuildOptimizer(optimizerName);
        var updateSolutionMethod = optimizer.GetType().GetMethod(
            "UpdateSolution",
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.NonPublic |
            System.Reflection.BindingFlags.Public);
        Assert.NotNull(updateSolutionMethod);

        // Set model on optimizer so its internal state initializes against the right param count.
        var setModelMethod = optimizer.GetType().GetMethod("SetModel",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public);
        setModelMethod?.Invoke(optimizer, new object[] { model });

        // Call UpdateSolution(model, flatGrad). With #1413's consolidation
        // this routes through base.UpdateSolution → SynthesizeTapeStepContext
        // → Step, the same path the per-sample bypass uses.
        updateSolutionMethod!.Invoke(optimizer, new object[] { model, flatGrad });

        var finalParams = model.GetParameters();
        double finalL2 = ComputeL2(finalParams);
        double deltaL2 = Math.Abs(finalL2 - initL2);
        bool anyParamChanged = false;
        for (int i = 0; i < finalParams.Length; i++)
        {
            if (finalParams[i] != initParamsSnapshot[i])
            {
                anyParamChanged = true;
                break;
            }
        }

        _output.WriteLine($"{optimizerName}: init L2={initL2:F4}, final L2={finalL2:F4}, ΔL2={deltaL2:E3}");
        // Real consolidation regression guard — UpdateSolution must
        // actually mutate parameters when handed a non-zero gradient.
        // A silent no-op (e.g. the NN-routing branch never engaging and
        // a fall-through-to-empty stub) would pass every finite-value
        // assertion below.
        Assert.True(anyParamChanged,
            $"{optimizerName}: UpdateSolution made no parameter change. The consolidation's NN " +
            "routing isn't engaging — Step never ran, so initial params === final params.");
        Assert.False(double.IsNaN(finalL2),
            $"{optimizerName}: UpdateSolution produced NaN params. The consolidation should make UpdateSolution " +
            "delegate to Step for NN solutions, which has anomaly guard + gradient clipping safeguards.");
        Assert.False(double.IsInfinity(finalL2),
            $"{optimizerName}: UpdateSolution produced infinite L2. Either anomaly guard didn't fire or the math diverged.");
        // Single step shouldn't move params by more than ~10× their initial magnitude
        // even with a pessimistic optimizer. If we see this it means the consolidation
        // isn't routing through Step's safeguards.
        Assert.True(deltaL2 < 10.0 * initL2,
            $"{optimizerName}: ΔL2 = {deltaL2:E3} exceeds 10× initL2 = {10 * initL2:E3}. Optimizer overshot — safeguards missing.");
    }

    private static IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> BuildOptimizer(string name)
    {
        return name switch
        {
            "Adam" => new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 }),
            "AdamW" => new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 }),
            "AdaMax" => new AdaMaxOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AdaMaxOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 }),
            "AMSGrad" => new AMSGradOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AMSGradOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 }),
            "SGD" => new StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 }),
            "Momentum" => new MomentumOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new MomentumOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 }),
            "Nadam" => new NadamOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new NadamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 }),
            _ => throw new ArgumentException($"Unknown optimizer: {name}"),
        };
    }

    private static double ComputeL2(Vector<float> v)
    {
        double sumSq = 0;
        for (int i = 0; i < v.Length; i++) sumSq += (double)v[i] * (double)v[i];
        return Math.Sqrt(sumSq);
    }
}
