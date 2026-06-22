using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Collection definition that groups integration tests mutating the
/// thread-static <see cref="TensorCodecOptions.Current"/>/fused-step
/// counter into a single non-parallel sequence. xUnit's default
/// per-class parallelization would race SetCurrent / Invalidate across
/// tests, producing flaky results (one test's EnableCompilation=true
/// would leak into another's EnableCompilation=false assertions).
/// </summary>
[CollectionDefinition("FusedOptimizerGlobalState", DisableParallelization = true)]
public sealed class FusedOptimizerCollection { }

/// <summary>
/// Integration tests for the fused forward+backward+optimizer compiled training path
/// wired through <see cref="NeuralNetworkBase{T}.TrainWithTape(Tensor{T}, Tensor{T}, IGradientBasedOptimizer{T, Tensor{T}, Tensor{T}}?)"/>.
/// Exercises the <c>TryTrainWithFusedOptimizer</c> engage/fallback decision tree end-to-end.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class FusedOptimizerIntegrationTests
{
    /// <summary>
    /// With <see cref="TensorCodecOptions.EnableCompilation"/> true, <c>T=float</c>, and a plain
    /// Adam optimizer (no LR scheduler, no adaptive rates), <c>TryTrainWithFusedOptimizer</c>
    /// engages the compiled fused fwd+bwd+update kernel. Training must complete without
    /// crashing and produce finite losses across multiple steps — any NaN/Inf indicates a
    /// broken kernel that silently corrupts training state.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task FusedAdam_TrainingCompletes_WithFiniteLossAcrossSteps()
    {
        await Task.CompletedTask;

        var network = BuildMlp();
        var input = CreateRandomTensor(new[] { 16, 4 }, seed: 42);
        var target = CreateRandomTensor(new[] { 16, 2 }, seed: 43);

        // Warmup forward to ensure layer tensors exist before we copy weights.
        network.Predict(CreateRandomTensor(new[] { 1, 4 }, seed: 99));

        var optimizer = BuildAdam(network, learningRate: 0.01);
        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();

            var losses = new List<float>();
            for (int step = 0; step < 10; step++)
            {
                network.TrainPublic(input, target, optimizer);
                losses.Add(network.LastLossPublic);
            }

            foreach (var loss in losses)
            {
                Assert.False(float.IsNaN(loss), "Fused path produced NaN loss");
                Assert.False(float.IsInfinity(loss), "Fused path produced Inf loss");
            }

            // Observable signal that the fused compiled kernel actually ran.
            // Without this assertion, a silent fallback to the eager path
            // would produce finite loss too and the "fused" test would pass
            // against the wrong code path. Require that at least one fused
            // step ran; in a healthy config on a supported MLP this should
            // be all 10.
            Assert.True(CompiledTapeTrainingStep<float>.GetFusedStepCount() > 0,
                "Fused path never engaged — the test was silently falling back to eager.");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(originalOptions);
            CompiledTapeTrainingStep<float>.Invalidate();
        }
    }

    /// <summary>
    /// #1662 lever #1 (§5a) bit-identical gate: single-pass full-precision fused
    /// optimizer-in-backward (<see cref="StreamingTrainingMode.ForceOn"/>, unclipped, Adam) must
    /// track the classic collect-then-step eager path to floating-point precision. Both networks
    /// start from identical parameters and train on identical data; the fused path consumes each
    /// gradient and applies the Adam update the moment it's produced (O(largest-grad) peak memory,
    /// hot-in-cache) while classic materializes the full gradient set and steps once. Adam is
    /// element-wise independent given a shared step t, so the trajectories must coincide.
    ///
    /// <para>The fused optimizer accumulates moments in <c>double</c> while classic Adam runs in
    /// <c>float</c>, so agreement is to float precision (the fused path is marginally MORE accurate),
    /// not literally bit-identical. The 1e-3 tolerance cleanly separates "tracks classic" (~1e-6
    /// drift) from a broken update (orders-of-magnitude divergence).</para>
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task FusedInBackward_Unclipped_MatchesClassicAdam_ToFloatPrecision()
    {
        await Task.CompletedTask;

        var input = CreateRandomTensor(new[] { 16, 4 }, seed: 42);
        var target = CreateRandomTensor(new[] { 16, 2 }, seed: 43);

        var classic = BuildMlp();
        var fused = BuildMlp();
        // Materialize layer weights, then force IDENTICAL initial parameters on both.
        classic.Predict(CreateRandomTensor(new[] { 1, 4 }, seed: 99));
        fused.Predict(CreateRandomTensor(new[] { 1, 4 }, seed: 99));
        fused.UpdateParameters(classic.GetParameters());

        // Classic eager collect-then-step (compilation off by default) vs full-precision fused
        // optimizer-in-backward. Explicitly UNCLIPPED (MaxGradNorm defaults to 1.0) so this
        // exercises the single-pass fused path, not the two-pass clipped one.
        classic.SetMaxGradNormForTest(0.0);
        fused.SetMaxGradNormForTest(0.0);
        classic.StreamingTraining = StreamingTrainingMode.ForceOff;
        fused.StreamingTraining = StreamingTrainingMode.ForceOn;

        var classicOpt = BuildAdam(classic, learningRate: 0.01);
        var fusedOpt = BuildAdam(fused, learningRate: 0.01);

        for (int step = 0; step < 20; step++)
        {
            classic.TrainPublic(input, target, classicOpt);
            fused.TrainPublic(input, target, fusedOpt);
            Assert.Equal((double)classic.LastLossPublic, (double)fused.LastLossPublic, 3);
        }

        var pc = SnapshotParameters(classic);
        var pf = SnapshotParameters(fused);
        Assert.False(AnyParameterDiffers(pc, pf, 1e-3f),
            "full-precision fused optimizer-in-backward diverged from classic Adam beyond float precision");
    }

    /// <summary>
    /// End-to-end validation of the FP16-activation training path for a NON-Adam fused
    /// optimizer (Tensors #574 + AiDotNet #1543). With <c>AIDOTNET_FP16_ACTIVATIONS=1</c>,
    /// <c>T=float</c>, <c>EnableCompilation=true</c>, and RMSprop (a fused-mappable optimizer
    /// that is neither Adam nor SGD), <c>TryStepWithFusedOptimizer</c> routes through the
    /// optimizer-agnostic <c>MixedPrecisionCompiledPlan.ComputeGradients</c> branch (FP16
    /// activation storage, grads bridged FP16↔FP32) and applies RMSprop's own master update
    /// via <c>Step</c>. Requires a Tensors build exposing <c>ComputeGradients</c> (0.92.0+);
    /// the reflection bridge gates on <c>IsComputeGradientsAvailable</c>, so a missing API
    /// silently falls back to eager and the fused-engagement assertion below would fail.
    /// Asserts the fused path engages and loss descends with finite values — a broken FP16
    /// bridge or a silent eager fallback fails one of these.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Fp16Activations_NonAdamFusedOptimizer_DescendsLoss_ViaComputeGradients()
    {
        await Task.CompletedTask;

        var network = BuildMlp();
        var input = CreateRandomTensor(new[] { 16, 4 }, seed: 42);
        var target = CreateRandomTensor(new[] { 16, 2 }, seed: 43);

        // Warmup forward so layer weight tensors are materialized.
        network.Predict(CreateRandomTensor(new[] { 1, 4 }, seed: 99));

        var rmsOptions = new RootMeanSquarePropagationOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.01,
            Decay = 0.9,
            Epsilon = 1e-8,
            UseAdaptiveLearningRate = false
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<float, Tensor<float>, Tensor<float>>(network, rmsOptions);

        var originalOptions = TensorCodecOptions.Current;
        string? originalEnv = Environment.GetEnvironmentVariable("AIDOTNET_FP16_ACTIVATIONS");
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_FP16_ACTIVATIONS", "1");
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();

            var losses = new List<float>();
            for (int step = 0; step < 15; step++)
            {
                network.TrainPublic(input, target, optimizer);
                float loss = network.LastLossPublic;
                losses.Add(loss);
                Assert.False(float.IsNaN(loss) || float.IsInfinity(loss),
                    $"FP16 path produced non-finite loss at step {step}");
            }

            // The fused compiled path must have actually engaged — otherwise the FP16
            // ComputeGradients branch never ran and we silently tested eager training.
            Assert.True(CompiledTapeTrainingStep<float>.GetFusedStepCount() > 0,
                "FP16 fused path never engaged — silent eager fallback (ComputeGradients missing from the linked Tensors build?).");

            // FP16-computed grads applied by RMSprop's own master update must still train.
            Assert.True(losses[losses.Count - 1] < losses[0],
                $"FP16 RMSprop training did not descend: first {losses[0]}, last {losses[losses.Count - 1]}");
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_FP16_ACTIVATIONS", originalEnv);
            TensorCodecOptions.SetCurrent(originalOptions);
            CompiledTapeTrainingStep<float>.Invalidate();
        }
    }

    /// <summary>
    /// End-to-end behavioral contract for FP16-activation training through the high-level
    /// <see cref="NeuralNetworkBase{T}"/> path on a model that uses the between-matmul ops the
    /// resident-memory win depends on — LayerNorm + GELU (Tensors #558). A transformer block separates
    /// matmuls with LayerNorm/GELU, so those ops (not just the GEMM the matmul-only tests cover) must
    /// keep their activations Half. Dense → LayerNorm → GELU → Dense, Adam, <c>AIDOTNET_FP16_ACTIVATIONS=1</c>.
    /// <para>On Tensors 0.96.0 (PR #606) the FP16-native LayerNorm/GELU ops keep the activation chain Half,
    /// so this Adam-trained Dense → LayerNorm → GELU → Dense model routes through the fused FP16 plan. The
    /// test asserts both that the fused path actually engaged (<c>GetFusedStepCount() &gt; 0</c>) and that
    /// FP16 activations through LayerNorm + GELU train the FP32 master weights (finite loss every step,
    /// loss descends) — the end-to-end resident-memory win Tensors #558 / #606 targets.</para>
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Fp16Activations_LayerNormGeluBlock_EngagesFusedFp16Path_AndTrains()
    {
        await Task.CompletedTask;

        var network = BuildLayerNormGeluNet();
        var input = CreateRandomTensor(new[] { 16, 4 }, seed: 42);
        var target = CreateRandomTensor(new[] { 16, 2 }, seed: 43);

        // Warmup forward so layer weight tensors are materialized before training.
        network.Predict(CreateRandomTensor(new[] { 1, 4 }, seed: 99));

        var optimizer = BuildAdam(network, learningRate: 0.01);
        var originalOptions = TensorCodecOptions.Current;
        string? originalEnv = Environment.GetEnvironmentVariable("AIDOTNET_FP16_ACTIVATIONS");
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_FP16_ACTIVATIONS", "1");
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();

            var losses = new List<float>();
            for (int step = 0; step < 15; step++)
            {
                network.TrainPublic(input, target, optimizer);
                float loss = network.LastLossPublic;
                losses.Add(loss);
                Assert.False(float.IsNaN(loss) || float.IsInfinity(loss),
                    $"FP16 LayerNorm/GELU path produced non-finite loss at step {step}");
            }

            // The fused FP16 plan must have actually engaged. On 0.96.0 (Tensors PR #606) both the matmul
            // and the LayerNorm/GELU activations stay Half, so this Adam-trained model routes through the
            // fused FP16 ComputeGradients/StepAdam path rather than the eager FP32 fallback. (Engagement
            // required the MixedPrecisionReflection.StepAdam float-arg fix in this PR: 0.96.0 made StepAdam
            // public with float hyperparams, and the reflection bridge was still passing doubles — which
            // threw on invoke and silently dropped every Adam FP16 step to eager.)
            Assert.True(CompiledTapeTrainingStep<float>.GetFusedStepCount() > 0,
                "FP16 fused path never engaged for the LayerNorm/GELU model — silent eager fallback.");

            // FP16 activations through LayerNorm + GELU must still train the FP32 master weights.
            Assert.True(losses[losses.Count - 1] < losses[0],
                $"FP16 LayerNorm/GELU training did not descend: first {losses[0]}, last {losses[losses.Count - 1]}");
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_FP16_ACTIVATIONS", originalEnv);
            TensorCodecOptions.SetCurrent(originalOptions);
            CompiledTapeTrainingStep<float>.Invalidate();
        }
    }

    /// <summary>
    /// Regression guard for the "compiled does nothing" bug (PR #1469): training
    /// with NO explicitly-supplied optimizer must still engage the fused compiled
    /// path. When the caller passes <c>optimizer: null</c>, <c>TrainWithTape</c>
    /// resolves the DEFAULT via <c>GetOrCreateBaseOptimizer()</c>. That default
    /// previously constructed Adam with <c>UseAMSGrad = true</c>, which
    /// <c>TryMapToFusedOptimizerConfig</c> rejected — silently demoting EVERY
    /// default-configured model to the eager tape so "compiled training" never
    /// actually ran. The default must be a fused-mappable optimizer; this test
    /// fails loudly if a non-mappable default is ever reintroduced.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DefaultOptimizer_EngagesFusedPath_NotSilentEagerFallback()
    {
        await Task.CompletedTask;

        var network = BuildMlp();
        var input = CreateRandomTensor(new[] { 16, 4 }, seed: 42);
        var target = CreateRandomTensor(new[] { 16, 2 }, seed: 43);

        // Warmup forward so layer weight tensors are materialized.
        network.Predict(CreateRandomTensor(new[] { 1, 4 }, seed: 99));

        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();

            // network.Train(...) calls TrainWithTape(input, target) with the
            // optimizer defaulted to null → the GetOrCreateBaseOptimizer() path.
            for (int step = 0; step < 10; step++)
            {
                network.Train(input, target);
                Assert.False(float.IsNaN(network.LastLossPublic),
                    $"default-optimizer fused step {step} produced NaN loss");
            }

            Assert.True(CompiledTapeTrainingStep<float>.GetFusedStepCount() > 0,
                "Default optimizer never engaged the fused path — the 'compiled does " +
                "nothing' regression is back: GetOrCreateBaseOptimizer() returned an " +
                "optimizer that TryMapToFusedOptimizerConfig rejects.");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(originalOptions);
            CompiledTapeTrainingStep<float>.Invalidate();
        }
    }

    /// <summary>
    /// The eager tape fallback path (<see cref="TensorCodecOptions.EnableCompilation"/>=false)
    /// must actually update parameters — this is the reference path that must always work,
    /// independent of whether the fused path engages.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task EagerPath_UpdatesParameters_WhenCompilationDisabled()
    {
        await Task.CompletedTask;

        var network = BuildMlp();
        var input = CreateRandomTensor(new[] { 16, 4 }, seed: 42);
        var target = CreateRandomTensor(new[] { 16, 2 }, seed: 43);

        network.Predict(CreateRandomTensor(new[] { 1, 4 }, seed: 99));
        var optimizer = BuildAdam(network, learningRate: 0.01);

        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
            CompiledTapeTrainingStep<float>.Invalidate();

            var before = SnapshotParameters(network);
            for (int step = 0; step < 5; step++)
            {
                network.TrainPublic(input, target, optimizer);
            }
            var after = SnapshotParameters(network);

            Assert.True(AnyParameterDiffers(before, after, tolerance: 1e-6f),
                "Eager fallback path did not update any parameters after 5 steps");
            Assert.False(float.IsNaN(network.LastLossPublic), "Eager path produced NaN");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(originalOptions);
            CompiledTapeTrainingStep<float>.Invalidate();
        }
    }

    /// <summary>
    /// When an LR scheduler is attached, <c>TryMapToFusedOptimizerConfig</c> must refuse the
    /// fused path (returning false → fallback to eager). The fused plan bakes the learning
    /// rate at <c>ConfigureOptimizer</c> time, so per-step LR changes would silently
    /// disappear. Verifies training still completes correctly and parameters update via
    /// the eager fallback.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task LearningRateScheduler_ForcesSafeFallbackToEager_AndParametersUpdate()
    {
        await Task.CompletedTask;

        var network = BuildMlp();
        var input = CreateRandomTensor(new[] { 16, 4 }, seed: 42);
        var target = CreateRandomTensor(new[] { 16, 2 }, seed: 43);

        network.Predict(CreateRandomTensor(new[] { 1, 4 }, seed: 99));

        var adamOptions = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            UseAdaptiveLearningRate = false,
            LearningRateScheduler = new ConstantLRScheduler(0.01)
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(network, adamOptions);

        var originalOptions = TensorCodecOptions.Current;
        try
        {
            // EnableCompilation=true — fused path would be tempted to engage but must
            // refuse due to the attached LR scheduler.
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();

            var before = SnapshotParameters(network);
            for (int step = 0; step < 5; step++)
            {
                network.TrainPublic(input, target, optimizer);
                Assert.False(float.IsNaN(network.LastLossPublic),
                    $"Eager fallback produced NaN loss at step {step}");
            }
            var after = SnapshotParameters(network);

            Assert.True(AnyParameterDiffers(before, after, tolerance: 1e-6f),
                "LR-scheduler fallback to eager did not update any parameters after 5 steps");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(originalOptions);
            CompiledTapeTrainingStep<float>.Invalidate();
        }
    }

    /// <summary>
    /// When <c>UseAdaptiveLearningRate=true</c>, the optimizer mutates its own LR between
    /// steps. The fused kernel bakes the LR at configure time, so we must refuse to map
    /// and fall back to eager. Exercises the adaptive-rate branch of
    /// <c>TryMapToFusedOptimizerConfig</c>.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task AdaptiveLearningRate_ForcesFallbackToEager_AndParametersUpdate()
    {
        await Task.CompletedTask;

        var network = BuildMlp();
        var input = CreateRandomTensor(new[] { 16, 4 }, seed: 42);
        var target = CreateRandomTensor(new[] { 16, 2 }, seed: 43);

        network.Predict(CreateRandomTensor(new[] { 1, 4 }, seed: 99));

        var adamOptions = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            UseAdaptiveLearningRate = true
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(network, adamOptions);

        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();

            var before = SnapshotParameters(network);
            for (int step = 0; step < 5; step++)
            {
                network.TrainPublic(input, target, optimizer);
                Assert.False(float.IsNaN(network.LastLossPublic),
                    $"Eager fallback produced NaN at step {step}");
            }
            var after = SnapshotParameters(network);

            Assert.True(AnyParameterDiffers(before, after, tolerance: 1e-6f),
                "Adaptive-LR fallback to eager did not update any parameters after 5 steps");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(originalOptions);
            CompiledTapeTrainingStep<float>.Invalidate();
        }
    }

    /// <summary>
    /// Double-typed networks (T=double) must fall back to eager — the Tensors-side fused
    /// optimizer operates on <c>float*</c> buffers directly. Exercises the
    /// <c>typeof(T) != typeof(float)</c> early-return branch.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DoubleModel_ForcesFallbackToEager_AndParametersUpdate()
    {
        await Task.CompletedTask;

        var network = BuildMlpDouble();
        var input = CreateRandomTensorDouble(new[] { 16, 4 }, seed: 42);
        var target = CreateRandomTensorDouble(new[] { 16, 2 }, seed: 43);

        network.Predict(CreateRandomTensorDouble(new[] { 1, 4 }, seed: 99));

        var adamOptions = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            UseAdaptiveLearningRate = false
        };
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(network, adamOptions);

        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });

            var before = SnapshotParametersDouble(network);
            for (int step = 0; step < 5; step++)
            {
                network.TrainPublic(input, target, optimizer);
                Assert.False(double.IsNaN(network.LastLossPublic),
                    $"Eager fallback produced NaN at step {step}");
            }
            var after = SnapshotParametersDouble(network);

            Assert.True(AnyParameterDiffersDouble(before, after, tolerance: 1e-9),
                "Double-type fallback to eager did not update any parameters after 5 steps");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }

    // ---------- helpers ----------

    private static FusedTrainingTestNetwork BuildMlp()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var network = new FusedTrainingTestNetwork(architecture);
        network.AddLayer(new DenseLayer<float>(8));
        network.AddLayer(new DenseLayer<float>(2));
        return network;
    }

    /// <summary>
    /// A transformer-style block exercising the FP16-NATIVE between-matmul ops (#558): a Dense projection,
    /// a LayerNorm, a GELU activation, then a Dense output. This is the op mix the resident-memory win
    /// depends on (matmuls separated by LayerNorm/GELU), as opposed to <see cref="BuildMlp"/>'s matmul-only
    /// stack.
    /// </summary>
    private static FusedTrainingTestNetwork BuildLayerNormGeluNet()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var network = new FusedTrainingTestNetwork(architecture);
        network.AddLayer(new DenseLayer<float>(8));
        network.AddLayer(new LayerNormalizationLayer<float>(8));
        network.AddLayer(new ActivationLayer<float>((IActivationFunction<float>)new GELUActivation<float>()));
        network.AddLayer(new DenseLayer<float>(2));
        return network;
    }

    private static FusedTrainingTestNetworkDouble BuildMlpDouble()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var network = new FusedTrainingTestNetworkDouble(architecture);
        network.AddLayer(new DenseLayer<double>(8));
        network.AddLayer(new DenseLayer<double>(2));
        return network;
    }

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> BuildAdam(
        FusedTrainingTestNetwork network, double learningRate)
    {
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = learningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            UseAdaptiveLearningRate = false
        };
        return new AdamOptimizer<float, Tensor<float>, Tensor<float>>(network, options);
    }

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        int length = 1;
        foreach (var d in shape) length *= d;
        var data = new float[length];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return new Tensor<float>(data, shape);
    }

    private static Tensor<double> CreateRandomTensorDouble(int[] shape, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        int length = 1;
        foreach (var d in shape) length *= d;
        var data = new double[length];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble() * 2 - 1;
        }
        return new Tensor<double>(data, shape);
    }

    private static float[] SnapshotParameters(FusedTrainingTestNetwork network)
    {
        var vec = network.GetParameters();
        var result = new float[vec.Length];
        for (int i = 0; i < vec.Length; i++) result[i] = vec[i];
        return result;
    }

    private static double[] SnapshotParametersDouble(FusedTrainingTestNetworkDouble network)
    {
        var vec = network.GetParameters();
        var result = new double[vec.Length];
        for (int i = 0; i < vec.Length; i++) result[i] = vec[i];
        return result;
    }

    private static bool AnyParameterDiffers(float[] a, float[] b, float tolerance)
    {
        if (a.Length != b.Length) return true;
        for (int i = 0; i < a.Length; i++)
        {
            if (Math.Abs(a[i] - b[i]) > tolerance) return true;
        }
        return false;
    }

    private static bool AnyParameterDiffersDouble(double[] a, double[] b, double tolerance)
    {
        if (a.Length != b.Length) return true;
        for (int i = 0; i < a.Length; i++)
        {
            if (Math.Abs(a[i] - b[i]) > tolerance) return true;
        }
        return false;
    }

    /// <summary>
    /// Minimal <see cref="NeuralNetworkBase{T}"/> subclass that exposes <c>TrainWithTape</c>
    /// with a custom optimizer and exposes <see cref="NeuralNetworkBase{T}.LastLoss"/>
    /// for test assertions.
    /// </summary>
    internal sealed class FusedTrainingTestNetwork : NeuralNetworkBase<float>
    {
        public FusedTrainingTestNetwork(NeuralNetworkArchitecture<float> architecture)
            : base(architecture, new MeanSquaredErrorLoss<float>())
        {
        }

        public override bool SupportsTraining => true;

        public void AddLayer(ILayer<float> layer) => AddLayerToCollection(layer);

        /// <summary>Test-only: set the global grad-norm clip threshold (0 disables clipping).</summary>
        public void SetMaxGradNormForTest(double value) => MaxGradNorm = NumOps.FromDouble(value);

        public void TrainPublic(
            Tensor<float> input, Tensor<float> target,
            IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> optimizer)
        {
            TrainWithTape(input, target, optimizer);
        }

        public float LastLossPublic => Convert.ToSingle(LastLoss);

        protected override void InitializeLayers() { }

        public override Tensor<float> Predict(Tensor<float> input)
        {
            bool originalTrainingMode = IsTrainingMode;
            SetTrainingMode(false);
            Tensor<float> current = input;
            foreach (var layer in Layers) current = layer.Forward(current);
            SetTrainingMode(originalTrainingMode);
            return current;
        }

        public override void UpdateParameters(Vector<float> parameters) => SetParameters(parameters);

        public override void Train(Tensor<float> input, Tensor<float> expectedOutput)
            => TrainWithTape(input, expectedOutput);

        public override ModelMetadata<float> GetModelMetadata() =>
            new ModelMetadata<float>
            {
                Name = "FusedTrainingTestNetwork",
                Version = "1.0",
                FeatureCount = Architecture.InputSize,
                Complexity = (int)ParameterCount
            };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }

        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
            => new FusedTrainingTestNetwork(Architecture);
    }

    internal sealed class FusedTrainingTestNetworkDouble : NeuralNetworkBase<double>
    {
        public FusedTrainingTestNetworkDouble(NeuralNetworkArchitecture<double> architecture)
            : base(architecture, new MeanSquaredErrorLoss<double>())
        {
        }

        public override bool SupportsTraining => true;

        public void AddLayer(ILayer<double> layer) => AddLayerToCollection(layer);

        public void TrainPublic(
            Tensor<double> input, Tensor<double> target,
            IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
        {
            TrainWithTape(input, target, optimizer);
        }

        public double LastLossPublic => Convert.ToDouble(LastLoss);

        protected override void InitializeLayers() { }

        public override Tensor<double> Predict(Tensor<double> input)
        {
            bool originalTrainingMode = IsTrainingMode;
            SetTrainingMode(false);
            Tensor<double> current = input;
            foreach (var layer in Layers) current = layer.Forward(current);
            SetTrainingMode(originalTrainingMode);
            return current;
        }

        public override void UpdateParameters(Vector<double> parameters) => SetParameters(parameters);

        public override void Train(Tensor<double> input, Tensor<double> expectedOutput)
            => TrainWithTape(input, expectedOutput);

        public override ModelMetadata<double> GetModelMetadata() =>
            new ModelMetadata<double>
            {
                Name = "FusedTrainingTestNetworkDouble",
                Version = "1.0",
                FeatureCount = Architecture.InputSize,
                Complexity = (int)ParameterCount
            };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }

        protected override IFullModel<double, Tensor<double>, Tensor<double>> CreateNewInstance()
            => new FusedTrainingTestNetworkDouble(Architecture);
    }
}
