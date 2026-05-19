using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Engines;

/// <summary>
/// Reproducer for training non-determinism in <see cref="Transformer{T}"/>
/// when running consecutive trainings in the same process at the same seed.
///
/// <para><b>Empirical observation</b>: at fixed predictor seed AND
/// <c>architecture.RandomSeed</c> pinned (layer init fully deterministic),
/// two back-to-back trainings of the same model on the same data in the
/// same process produce DIFFERENT trained weights. Initial finding from
/// the HarmonicEngine byte-LM consumer (Phase_PAPER_A_PathB_Int8_Sanity)
/// where FP32 baseline top-1 oscillated 0%/3%/5%/6.5% across separate
/// process invocations on GPU; on CPU within the same process, the
/// parameter L2 norm differs by ~0.08 between Run A and Run B with
/// identical seeds.</para>
///
/// <para><b>Why this is a real bug, not benign FP noise</b>:
/// <list type="bullet">
///   <item>FP-associativity reorder noise in parallel reductions
///   typically produces L2 deltas in the 1e-6 .. 1e-4 range on a
///   small model with ~20-30k parameters. A 0.08 L2 delta on a
///   ~12k-param model is well above that floor.</item>
///   <item>On the byte-LM consumer reproducer, the divergence
///   compounds into wildly different convergence trajectories
///   (top-1 oscillating 0%-19% across runs on GPU). That's not
///   "small jitter around a mean" — it's a model that sometimes
///   doesn't train at all.</item>
///   <item>Suggests a downstream RNG consumer in the training path
///   (AdamOptimizer batch shuffle? Dropout? Augmentation? Lazy
///   parameter init?) that reads from
///   <see cref="RandomHelper.ThreadSafeRandom"/> — whose per-thread
///   Random advances state cumulatively across consecutive
///   trainings, so Run B starts with different RNG state than
///   Run A even though the seed parameter is identical.</item>
/// </list></para>
///
/// <para><b>What this test does</b>: trains two Transformer instances
/// with identical seeds + data on whatever engine is configured.
/// Captures the post-training parameter-vector L2 norm of each.
/// Asserts bit-identical. A deterministic library MUST hit this; a
/// library that depends on cumulative RandomHelper state (or
/// engine-side non-determinism) will fail.</para>
/// </summary>
[Collection("NonParallelIntegration")]
public class GpuTransformerDeterminismTests
{
    private readonly ITestOutputHelper _output;

    public GpuTransformerDeterminismTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // Small reproducer config — V=16 to keep training fast; bigger than V=256
    // adds nothing to the determinism question. Two trainings must take <30s
    // each so the test stays in CI budget.
    private const int VocabSize = 16;
    private const int CtxLen = 8;
    private const int DModel = 32;
    private const int Heads = 2;
    private const int FfDim = 64;
    private const int NumLayers = 1;
    private const int Epochs = 10;
    private const int SampleCount = 64;
    private const int Seed = 42;

    /// <summary>
    /// Trains two Transformer instances back-to-back with identical seeds + data
    /// and asserts post-training first-encoder-block weight L2 norms are
    /// bit-identical. Currently expected to PASS on CpuEngine and FAIL on
    /// DirectGpuTensorEngine; pin the assertion + document the gap as a
    /// determinism bug.
    /// </summary>
    [Fact]
    public void Transformer_Train_TwoRunsAtSameSeed_ProduceIdenticalWeights()
    {
        _output.WriteLine($"Engine: {AiDotNetEngine.Current.GetType().Name}");

        var (arch, xTrain, yTrain) = BuildFixture();

        // Diagnostic: capture L2 at THREE stages to isolate where divergence enters.
        //   ctorOnly: right after Transformer ctor — only non-lazy layer init has fired
        //   postPredict: after a forward pass — lazy layers (MHA / FFN) have materialized
        //   postTrain: after full training loop — accumulates all training-time RNG
        double L2AtStage(int stage, out long paramCount)
        {
            var model = new Transformer<float>(
                arch,
                lossFunction: new CategoricalCrossEntropyLoss<float>());
            if (stage >= 1)
            {
                var dummyX = new Tensor<float>([1, CtxLen]);
                for (int s = 0; s < CtxLen; s++) dummyX[0, s] = xTrain[0, s];
                _ = model.Predict(dummyX);
            }
            if (stage >= 2)
            {
                model.SetTrainingMode(true);
                for (int epoch = 0; epoch < Epochs; epoch++)
                {
                    for (int i = 0; i < SampleCount; i++)
                    {
                        var sampleX = new Tensor<float>([1, CtxLen]);
                        var sampleY = new Tensor<float>([1, VocabSize]);
                        for (int s = 0; s < CtxLen; s++) sampleX[0, s] = xTrain[i, s];
                        for (int c = 0; c < VocabSize; c++) sampleY[0, c] = yTrain[i, c];
                        model.Train(sampleX, sampleY);
                    }
                }
            }
            double sumSq = 0;
            paramCount = 0;
            foreach (var p in model.GetParameters())
            {
                sumSq += (double)p * (double)p;
                paramCount++;
            }
            return Math.Sqrt(sumSq);
        }

        var ctorA = L2AtStage(0, out long cntA0);
        var ctorB = L2AtStage(0, out long cntB0);
        var postPredictA = L2AtStage(1, out long cntA1);
        var postPredictB = L2AtStage(1, out long cntB1);
        _output.WriteLine($"Stage 0 (ctor only): A={ctorA:G17} (n={cntA0}) B={ctorB:G17} (n={cntB0}) diff={Math.Abs(ctorA - ctorB):G6}");
        _output.WriteLine($"Stage 1 (post-Predict): A={postPredictA:G17} (n={cntA1}) B={postPredictB:G17} (n={cntB1}) diff={Math.Abs(postPredictA - postPredictB):G6}");

        double L2RunOnce()
        {
            var model = new Transformer<float>(
                arch,
                lossFunction: new CategoricalCrossEntropyLoss<float>());
            model.SetTrainingMode(true);
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                for (int i = 0; i < SampleCount; i++)
                {
                    var sampleX = new Tensor<float>([1, CtxLen]);
                    var sampleY = new Tensor<float>([1, VocabSize]);
                    for (int s = 0; s < CtxLen; s++) sampleX[0, s] = xTrain[i, s];
                    for (int c = 0; c < VocabSize; c++) sampleY[0, c] = yTrain[i, c];
                    model.Train(sampleX, sampleY);
                }
            }
            double sumSq = 0;
            foreach (var p in model.GetParameters())
            {
                sumSq += (double)p * (double)p;
            }
            return Math.Sqrt(sumSq);
        }

        double l2_a = L2RunOnce();
        double l2_b = L2RunOnce();

        _output.WriteLine($"Run A trained-parameter L2 norm: {l2_a:G17}");
        _output.WriteLine($"Run B trained-parameter L2 norm: {l2_b:G17}");
        _output.WriteLine($"|A - B| = {Math.Abs(l2_a - l2_b):G6}");

        // Tight bit-equality assertion. A deterministic engine MUST hit this;
        // a non-deterministic engine will fail with a measurable gap.
        Assert.Equal(l2_a, l2_b);
    }

    private static (TransformerArchitecture<float>, Tensor<float>, Tensor<float>) BuildFixture()
    {
        // Per project standing rule: NEVER use System.Random directly; route
        // through RandomHelper.CreateSeededRandom for deterministic (test)
        // RNG or CreateSecureRandom for crypto-grade (production) RNG.
        var rng = RandomHelper.CreateSeededRandom(Seed);
        var xTrain = new Tensor<float>([SampleCount, CtxLen]);
        var yTrain = new Tensor<float>([SampleCount, VocabSize]);
        for (int i = 0; i < SampleCount; i++)
        {
            int target = rng.Next() % VocabSize;
            for (int s = 0; s < CtxLen; s++)
            {
                xTrain[i, s] = (byte)((target + s) % VocabSize);
            }
            yTrain[i, target] = 1.0f;
        }

        // Pin layer-init seed too, so the only remaining source of
        // non-determinism between Run A and Run B is the engine's own
        // operation ordering.
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: NumLayers,
            numDecoderLayers: 0,
            numHeads: Heads,
            modelDimension: DModel,
            feedForwardDimension: FfDim,
            inputSize: CtxLen,
            outputSize: VocabSize,
            maxSequenceLength: CtxLen,
            vocabularySize: VocabSize,
            randomSeed: Seed);

        return (arch, xTrain, yTrain);
    }
}
