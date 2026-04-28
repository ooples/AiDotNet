using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
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
                Complexity = ParameterCount
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
                Complexity = ParameterCount
            };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }

        protected override IFullModel<double, Tensor<double>, Tensor<double>> CreateNewInstance()
            => new FusedTrainingTestNetworkDouble(Architecture);
    }
}
