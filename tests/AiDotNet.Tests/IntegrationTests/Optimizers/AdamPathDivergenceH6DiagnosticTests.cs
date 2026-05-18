using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Empirical diagnostic for AiDotNet#1364 H6 — does
/// <see cref="AdamOptimizer{T, TInput, TOutput}.UpdateSolution"/>
/// (the flat-Adam path used by <c>Optimize</c>) produce the same Adam
/// update as
/// <see cref="AdamOptimizer{T, TInput, TOutput}.Step(TapeStepContext{T})"/>
/// (the per-tensor path used by <c>TrainWithTape</c>) on identical
/// (params, gradient) inputs?
///
/// <para>This is the diagnostic that runs FIRST, before any
/// unification refactor. The static reading of both code paths
/// identified four candidate divergence points:</para>
/// <list type="number">
///   <item><b>State store</b>: <c>_m/_v</c> (Vector) vs
///         <c>_tapeM/_tapeV</c> (Dictionary&lt;Tensor, Tensor&gt;).
///         Separate stores never share state, but each path uses its
///         own consistently within a single training session.</item>
///   <item><b>Step counter</b>: <c>_t</c> (incremented in Optimize
///         before UpdateSolution) vs <c>_tapeStep</c> (incremented
///         inside Step). Same value on equivalent calls.</item>
///   <item><b>Beta source</b>: <c>_currentBeta1/_currentBeta2</c>
///         (UpdateSolution) vs <c>_options.Beta1/Beta2</c> (Step).
///         Equal in default config (<c>UseAdaptiveBetas == false</c>).</item>
///   <item><b>Numerics order</b>: UpdateSolution uses Engine.* op
///         chain (allocates intermediates). Step has a T=double/float
///         fast-path with a tight raw-array loop. Same math, but the
///         summation order may produce 1-ULP-class drift on edge
///         elements (last digit only).</item>
/// </list>
///
/// <para>Other divergences exist (anomaly guard + grad clipping live
/// only in Step), but those are gates — they short-circuit the update,
/// they don't change the update when both paths fire.</para>
///
/// <para>This test feeds identical inputs through both paths and
/// asserts the resulting parameter update is identical within
/// FP-summation-order tolerance. If the assertion fails, the failure
/// message identifies which element diverges first and by how much —
/// that's the empirical evidence for the unification PR's design.</para>
/// </summary>
public class AdamPathDivergenceH6DiagnosticTests
{
    private readonly ITestOutputHelper _output;

    public AdamPathDivergenceH6DiagnosticTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static AdamOptimizerOptions<float, Vector<float>, Vector<float>> MakeOptions()
    {
        return new AdamOptimizerOptions<float, Vector<float>, Vector<float>>
        {
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            // Deliberately disable extras that only live on one path
            // so we isolate the core Adam math:
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = false,
            AnomalyGuardMode = AdamAnomalyGuardMode.Never,
            InitialLearningRate = 0.01,
            MinLearningRate = 0.01,
            MaxLearningRate = 0.01,
        };
    }

    private static AdamOptimizer<float, Vector<float>, Vector<float>> MakeOptimizer()
    {
        // Adam ctor signature is (model, options); model can be null when we drive
        // the optimizer directly through UpdateParameters / Step without ever
        // calling Optimize (which is what the diagnostic does).
        return new AdamOptimizer<float, Vector<float>, Vector<float>>(model: null, options: MakeOptions());
    }

    [Fact]
    public void Adam_FlatVsTape_SameInputs_SameOutputs_SingleStep()
    {
        const int n = 8;
        var rng = new Random(0xADA);
        var initialParams = new float[n];
        var fixedGrad = new float[n];
        for (int i = 0; i < n; i++)
        {
            initialParams[i] = (float)(rng.NextDouble() * 2 - 1);
            fixedGrad[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        // -------- Path A: flat-Adam via UpdateParameters (Vector<T>, Vector<T>) --------
        var optA = MakeOptimizer();
        // UpdateParameters is the public Vector-based entry; AdamOptimizer
        // wires it to the same math UpdateSolution does (same _m, _v, _t,
        // same Engine op chain). It does NOT require an IFullModel scaffold.
        var paramsA = new Vector<float>((float[])initialParams.Clone());
        var gradA = new Vector<float>((float[])fixedGrad.Clone());
        var updatedA = optA.UpdateParameters(paramsA, gradA);

        // -------- Path B: per-tensor Adam via Step(TapeStepContext) --------
        var optB = MakeOptimizer();
        var paramTensorB = new Tensor<float>(new[] { n }, new Vector<float>((float[])initialParams.Clone()));
        var gradTensorB = new Tensor<float>(new[] { n }, new Vector<float>((float[])fixedGrad.Clone()));
        var ctxB = new TapeStepContext<float>(
            parameters: new[] { paramTensorB },
            gradients: new Dictionary<Tensor<float>, Tensor<float>>(
                AiDotNet.Helpers.TensorReferenceComparer<Tensor<float>>.Instance)
            {
                [paramTensorB] = gradTensorB,
            },
            loss: 0f);
        optB.Step(ctxB);

        var updatedBArr = new float[n];
        for (int i = 0; i < n; i++) updatedBArr[i] = paramTensorB[i];

        // -------- Compare --------
        double maxAbsDiff = 0;
        int maxIdx = -1;
        for (int i = 0; i < n; i++)
        {
            double diff = Math.Abs(updatedA[i] - updatedBArr[i]);
            if (diff > maxAbsDiff)
            {
                maxAbsDiff = diff;
                maxIdx = i;
            }
        }

        _output.WriteLine($"H6 single-step diagnostic:");
        for (int i = 0; i < n; i++)
        {
            _output.WriteLine($"  [{i}] flat={updatedA[i],14:G7}  tape={updatedBArr[i],14:G7}  diff={Math.Abs(updatedA[i] - updatedBArr[i]):G3}");
        }
        _output.WriteLine($"Max |diff| = {maxAbsDiff:G6} at index {maxIdx}");

        // 1-ULP tolerance scaled by max parameter magnitude. Same-math/different-
        // summation-order should sit at ≤ 1 ULP of typical FP32 (~6e-7 × max|x|).
        // If we see anything larger, the paths are doing different math.
        double maxAbsParam = 0;
        for (int i = 0; i < n; i++)
        {
            double a = Math.Abs(updatedA[i]);
            if (a > maxAbsParam) maxAbsParam = a;
        }
        double tolerance = Math.Max(1e-6, 4e-6 * maxAbsParam);
        Assert.True(maxAbsDiff < tolerance,
            $"Flat-Adam and tape-Adam diverged by {maxAbsDiff:G6} on a single step at index {maxIdx} " +
            $"(tolerance {tolerance:G6}, max|param|={maxAbsParam:G6}). " +
            $"The two implementations are NOT bit-for-bit identical given identical inputs — H6 wiring under question.");
    }

    [Fact]
    public void Adam_FlatVsTape_SameInputs_SameOutputs_MultiStep()
    {
        const int n = 8;
        const int numSteps = 5;
        var rng = new Random(0xBEAD);
        var initialParams = new float[n];
        var grads = new float[numSteps][];
        for (int i = 0; i < n; i++) initialParams[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int t = 0; t < numSteps; t++)
        {
            grads[t] = new float[n];
            for (int i = 0; i < n; i++) grads[t][i] = (float)(rng.NextDouble() * 2 - 1);
        }

        // Path A: drive UpdateParameters across numSteps with the same grad sequence.
        var optA = MakeOptimizer();
        var paramsA = new Vector<float>((float[])initialParams.Clone());
        for (int t = 0; t < numSteps; t++)
        {
            var gradA = new Vector<float>((float[])grads[t].Clone());
            paramsA = optA.UpdateParameters(paramsA, gradA);
        }

        // Path B: drive Step across numSteps with the same grad sequence.
        var optB = MakeOptimizer();
        var paramTensorB = new Tensor<float>(new[] { n }, new Vector<float>((float[])initialParams.Clone()));
        for (int t = 0; t < numSteps; t++)
        {
            var gradTensorB = new Tensor<float>(new[] { n }, new Vector<float>((float[])grads[t].Clone()));
            var ctxB = new TapeStepContext<float>(
                parameters: new[] { paramTensorB },
                gradients: new Dictionary<Tensor<float>, Tensor<float>>(
                    AiDotNet.Helpers.TensorReferenceComparer<Tensor<float>>.Instance)
                {
                    [paramTensorB] = gradTensorB,
                },
                loss: 0f);
            optB.Step(ctxB);
        }

        var updatedBArr = new float[n];
        for (int i = 0; i < n; i++) updatedBArr[i] = paramTensorB[i];

        double maxAbsDiff = 0;
        int maxIdx = -1;
        double maxAbsParam = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = Math.Abs(paramsA[i] - updatedBArr[i]);
            if (diff > maxAbsDiff) { maxAbsDiff = diff; maxIdx = i; }
            double a = Math.Abs(paramsA[i]);
            if (a > maxAbsParam) maxAbsParam = a;
        }

        _output.WriteLine($"H6 {numSteps}-step diagnostic: max |diff|={maxAbsDiff:G6} at {maxIdx}, max|param|={maxAbsParam:G6}");
        for (int i = 0; i < n; i++)
        {
            _output.WriteLine($"  [{i}] flat={paramsA[i],14:G7}  tape={updatedBArr[i],14:G7}  diff={Math.Abs(paramsA[i] - updatedBArr[i]):G3}");
        }

        // Multi-step tolerance: error accumulates roughly like sqrt(numSteps) × 1 ULP.
        double tolerance = Math.Max(1e-6, 4e-6 * maxAbsParam * Math.Sqrt(numSteps));
        Assert.True(maxAbsDiff < tolerance,
            $"Over {numSteps} steps the two Adam paths drifted by {maxAbsDiff:G6} at index {maxIdx} " +
            $"(tolerance {tolerance:G6}). The accumulated drift indicates the implementations are not " +
            $"numerically equivalent — empirical evidence for H6 unification.");
    }

    /// <summary>
    /// Sanity check that both paths actually moved the parameters away
    /// from the initial values — guards against the diagnostic reporting
    /// "0 == 0" if either path silently no-ops.
    /// </summary>
    [Fact]
    public void Adam_BothPaths_ActuallyUpdate_Parameters()
    {
        const int n = 4;
        var initialParams = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var fixedGrad = new float[] { 0.5f, -0.5f, 0.5f, -0.5f };

        var optA = MakeOptimizer();
        var paramsA = new Vector<float>((float[])initialParams.Clone());
        var updatedA = optA.UpdateParameters(paramsA, new Vector<float>((float[])fixedGrad.Clone()));

        var optB = MakeOptimizer();
        var paramTensorB = new Tensor<float>(new[] { n }, new Vector<float>((float[])initialParams.Clone()));
        var gradTensorB = new Tensor<float>(new[] { n }, new Vector<float>((float[])fixedGrad.Clone()));
        var ctxB = new TapeStepContext<float>(
            parameters: new[] { paramTensorB },
            gradients: new Dictionary<Tensor<float>, Tensor<float>>(
                AiDotNet.Helpers.TensorReferenceComparer<Tensor<float>>.Instance)
            {
                [paramTensorB] = gradTensorB,
            },
            loss: 0f);
        optB.Step(ctxB);

        bool anyChangedA = false, anyChangedB = false;
        for (int i = 0; i < n; i++)
        {
            if (Math.Abs(updatedA[i] - initialParams[i]) > 0) anyChangedA = true;
            if (Math.Abs(paramTensorB[i] - initialParams[i]) > 0) anyChangedB = true;
        }
        Assert.True(anyChangedA, "Flat-Adam path produced no parameter update — diagnostic invalid.");
        Assert.True(anyChangedB, "Tape-Adam path produced no parameter update — diagnostic invalid.");
    }
}
