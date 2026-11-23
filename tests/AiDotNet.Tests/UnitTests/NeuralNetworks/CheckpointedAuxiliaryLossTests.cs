using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training.Memory;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Gradient checkpointing for a network whose layers produce an auxiliary loss inside a checkpointed segment (as an MoE
/// router's balance loss does): the checkpointed step must give the loss and the gradients of the plain step.
/// </summary>
/// <remarks>
/// <para>
/// Each hidden layer below is a dense layer that, in a training forward, leaves <c>weight * sum(y^2)</c> of its output
/// for the network, which returns the sum from <c>ConsumeAuxiliaryTapeLoss</c>. The term depends on the layer's weights
/// only through the segment's activations, so its gradient exists only if the segment's backward differentiates it.
/// Before checkpointed segments returned their auxiliary terms, the term was built from the segment's discarded
/// activations: its gradient was zero on the CPU and its storage released on the GPU.
/// </para>
/// <para>
/// Tolerance: the checkpointed run replays the same CPU float32 operations; only the order in which a parameter's
/// gradient contributions are summed can differ (the segment's recompute adds the output and auxiliary parts in one
/// nested backward). Values are compared to |a - b| &lt;= 1e-6 * max(1, |a|).
/// </para>
/// </remarks>
[Collection("NonParallelIntegration")]
public class CheckpointedAuxiliaryLossTests
{
    private const int Width = 6;
    private const int Batch = 3;
    private const int HiddenLayers = 6;
    private const float Tolerance = 1e-6f;

    /// <summary>A dense layer that leaves <c>weight * sum(y^2)</c> of its training output as an auxiliary loss.</summary>
    private sealed class AuxiliaryDenseLayer : DenseLayer<float>
    {
        private readonly float _weight;

        public AuxiliaryDenseLayer(int outputSize, float weight)
            : base(outputSize, activationFunction: new TanhActivation<float>())
        {
            _weight = weight;
        }

        public Tensor<float>? PendingAuxiliary { get; set; }

        protected override Tensor<float> ForwardTraced(Tensor<float> input)
        {
            var output = base.ForwardTraced(input);
            if (IsTrainingMode && !IsInferringShapes)
            {
                var squares = Engine.TensorMultiply(output, output);
                var total = Engine.ReduceSum(squares, Enumerable.Range(0, squares.Rank).ToArray(), keepDims: false);
                PendingAuxiliary = Engine.TensorMultiplyScalar(Engine.Reshape(total, [1]), _weight);
            }
            return output;
        }
    }

    /// <summary>
    /// A dense layer whose output is scaled by a state its training forward then changes in place, the shape of a
    /// load-balancing bias a router nudges after routing. It keeps the checkpoint contract: the first forward preserves
    /// the scale it reads and makes the change; the recompute recalls the scale and changes nothing.
    /// </summary>
    private sealed class DriftingScaleLayer : DenseLayer<float>
    {
        private readonly Tensor<float> _scale = new Tensor<float>([1], new Vector<float>(new[] { 1f }));

        public DriftingScaleLayer(int outputSize)
            : base(outputSize, activationFunction: new TanhActivation<float>())
        {
        }

        public bool KeepsCheckpointContract { get; set; } = true;

        public float Scale => _scale.GetFlat(0);

        protected override Tensor<float> ForwardTraced(Tensor<float> input)
        {
            var output = base.ForwardTraced(input);
            if (IsInferringShapes || !IsTrainingMode) return output;

            var scale = _scale;
            if (KeepsCheckpointContract)
            {
                if (GradientCheckpointingContext.TryRecall(this, out Tensor<float> recalled)) scale = recalled;
                else GradientCheckpointingContext.Preserve(this, _scale);
            }
            var scaled = Engine.TensorMultiplyScalar(output, scale.GetFlat(0));
            if (!KeepsCheckpointContract || !GradientCheckpointingContext.IsRecomputing)
                _scale.SetFlat(0, _scale.GetFlat(0) * 1.5f);
            return scaled;
        }
    }

    private sealed class AuxiliaryLossNetwork : NeuralNetwork<float>
    {
        public AuxiliaryLossNetwork(NeuralNetworkArchitecture<float> architecture, IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>? optimizer)
            : base(architecture, optimizer, new MeanSquaredErrorLoss<float>())
        {
        }

        public int ConsumeCalls { get; private set; }

        protected override Tensor<float>? ConsumeAuxiliaryTapeLoss()
        {
            ConsumeCalls++;
            Tensor<float>? total = null;
            foreach (var layer in Layers)
            {
                if (layer is not AuxiliaryDenseLayer aux || aux.PendingAuxiliary is not { } term) continue;
                aux.PendingAuxiliary = null;
                total = total is null ? term : Engine.TensorAdd(total, term);
            }
            return total;
        }

        /// <summary>The training objective the tape step differentiates: main loss times <paramref name="mainScale"/> plus the auxiliary terms.</summary>
        public Tensor<float> Objective(Tensor<float> input, Tensor<float> target, float mainScale)
        {
            var prediction = ForwardForTraining(input);
            var main = new MeanSquaredErrorLoss<float>().ComputeTapeLoss(prediction, target);
            if (mainScale != 1f) main = Engine.TensorMultiplyScalar(main, mainScale);
            return ApplyCompositeObjective(main, input);
        }
    }

    private static (AuxiliaryLossNetwork Net, List<ILayer<float>> Layers) Build(
        int seed, bool drifting = false, IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>? optimizer = null)
    {
        var layers = new List<ILayer<float>>();
        for (int i = 0; i < HiddenLayers; i++)
        {
            layers.Add(drifting && i % 3 == 1
                ? new DriftingScaleLayer(Width)
                : new AuxiliaryDenseLayer(Width, weight: 0.05f * (i + 1)));
        }
        layers.Add(new DenseLayer<float>(2, activationFunction: new IdentityActivation<float>()));

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: Width,
            outputSize: 2,
            layers: layers)
        {
            RandomSeed = seed,
        };
        var net = new AuxiliaryLossNetwork(architecture, optimizer);
        return (net, layers);
    }

    private static (Tensor<float> Input, Tensor<float> Target) Data(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var input = new Tensor<float>([Batch, Width]);
        var target = new Tensor<float>([Batch, 2]);
        for (int i = 0; i < input.Length; i++) input.SetFlat(i, (float)(rng.NextDouble() * 2 - 1));
        for (int i = 0; i < target.Length; i++) target.SetFlat(i, (float)(rng.NextDouble() * 2 - 1));
        return (input, target);
    }

    /// <summary>Two networks with the same parameters, the second checkpointed every two layers.</summary>
    private static (AuxiliaryLossNetwork Plain, AuxiliaryLossNetwork Checkpointed, Tensor<float> Input, Tensor<float> Target)
        Pair(bool drifting = false, Func<IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>>? optimizer = null)
    {
        var (input, target) = Data(7);
        var (plain, _) = Build(11, drifting, optimizer?.Invoke());
        var (checkpointed, _) = Build(11, drifting, optimizer?.Invoke());
        plain.SetTrainingMode(false);
        checkpointed.SetTrainingMode(false);
        _ = plain.Predict(input);
        _ = checkpointed.Predict(input);
        checkpointed.SetParameters(plain.GetParameters());
        checkpointed.EnableGradientCheckpointing(segmentSize: 2);
        Assert.True(checkpointed.IsGradientCheckpointingEnabled);
        Assert.False(plain.IsGradientCheckpointingEnabled);
        return (plain, checkpointed, input, target);
    }

    private static List<Tensor<float>> Parameters(NeuralNetworkBase<float> network)
        => network.GetParameterStateChunks()
            .Where(c => c.Tensor is not null && c.Tensor.Length > 0)
            .Select(c => c.Tensor!)
            .ToList();

    private static (float Loss, float[][] Gradients) LossAndGradients(AuxiliaryLossNetwork network, Tensor<float> input, Tensor<float> target, float mainScale)
    {
        network.SetTrainingMode(true);
        var parameters = Parameters(network);
        using var tape = new GradientTape<float>();
        var loss = network.Objective(input, target, mainScale);
        var gradients = tape.ComputeGradients(loss, parameters);
        var result = new float[parameters.Count][];
        for (int i = 0; i < parameters.Count; i++)
        {
            result[i] = gradients.TryGetValue(parameters[i], out var g) && g is not null
                ? g.ToArray()
                : new float[parameters[i].Length];
        }
        return (loss.ToArray()[0], result);
    }

    private static void AssertClose(float expected, float actual, string what)
        => Assert.True(Math.Abs(expected - actual) <= Tolerance * Math.Max(1f, Math.Abs(expected)),
            $"{what}: plain {expected:R}, checkpointed {actual:R}");

    private static void AssertSameGradients(float[][] expected, float[][] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int p = 0; p < expected.Length; p++)
        {
            Assert.Equal(expected[p].Length, actual[p].Length);
            for (int i = 0; i < expected[p].Length; i++)
                AssertClose(expected[p][i], actual[p][i], $"{what}, parameter {p} element {i}");
        }
    }

    [Fact]
    public void CheckpointedStep_GivesThePlainLossAndGradients_WithAuxiliaryTermsInsideSegments()
    {
        var (plain, checkpointed, input, target) = Pair();

        var (plainLoss, plainGrads) = LossAndGradients(plain, input, target, mainScale: 1f);
        var (ckptLoss, ckptGrads) = LossAndGradients(checkpointed, input, target, mainScale: 1f);

        AssertClose(plainLoss, ckptLoss, "loss");
        AssertSameGradients(plainGrads, ckptGrads, "full objective gradient");
        // One call per segment forward, one per segment recompute, one before the first segment and one in the objective.
        Assert.True(checkpointed.ConsumeCalls > plain.ConsumeCalls);
    }

    [Fact]
    public void CheckpointedStep_AuxiliaryGradientIsNonZero_AndEqualsThePlainOne()
    {
        var (plain, checkpointed, input, target) = Pair();

        // The main loss scaled to zero leaves only the auxiliary terms' gradient.
        var (plainLoss, plainGrads) = LossAndGradients(plain, input, target, mainScale: 0f);
        var (ckptLoss, ckptGrads) = LossAndGradients(checkpointed, input, target, mainScale: 0f);

        AssertClose(plainLoss, ckptLoss, "auxiliary loss");
        Assert.True(plainLoss > 0f, $"auxiliary loss should be positive, was {plainLoss}");
        double plainNorm = plainGrads.SelectMany(g => g).Sum(v => (double)v * v);
        double ckptNorm = ckptGrads.SelectMany(g => g).Sum(v => (double)v * v);
        Assert.True(plainNorm > 1e-8, $"plain auxiliary gradient should be non-zero, squared norm {plainNorm}");
        Assert.True(ckptNorm > 1e-8, $"checkpointed auxiliary gradient should be non-zero, squared norm {ckptNorm}");
        AssertSameGradients(plainGrads, ckptGrads, "auxiliary-only gradient");
    }

    [Fact]
    public void CheckpointedTraining_MatchesPlainTraining_OverSeveralSteps_WithoutAReleasedTensor()
    {
        var (plain, checkpointed, input, target) = Pair(optimizer: () =>
            new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AiDotNet.Models.Options.AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }));
        plain.SetTrainingMode(true);
        checkpointed.SetTrainingMode(true);

        for (int step = 0; step < 4; step++)
        {
            plain.Train(input, target);
            var ex = Record.Exception(() => checkpointed.Train(input, target));
            Assert.Null(ex);
            AssertClose(plain.GetLastLoss(), checkpointed.GetLastLoss(), $"step {step} loss");
        }

        var p = plain.GetParameters();
        var c = checkpointed.GetParameters();
        Assert.Equal(p.Length, c.Length);
        for (int i = 0; i < p.Length; i++)
            Assert.True(Math.Abs(p[i] - c[i]) <= 1e-5f * Math.Max(1f, Math.Abs(p[i])),
                $"parameter {i} after 4 steps: plain {p[i]:R}, checkpointed {c[i]:R}");
    }

    [Fact]
    public void CheckpointRecompute_ReplaysPreservedState_AndSkipsStateUpdates()
    {
        var (plain, checkpointed, input, target) = Pair(drifting: true);

        var (plainLoss, plainGrads) = LossAndGradients(plain, input, target, mainScale: 1f);
        var (ckptLoss, ckptGrads) = LossAndGradients(checkpointed, input, target, mainScale: 1f);

        AssertClose(plainLoss, ckptLoss, "loss");
        AssertSameGradients(plainGrads, ckptGrads, "gradient with drifting layer state");
        var plainScales = plain.Layers.OfType<DriftingScaleLayer>().Select(l => l.Scale).ToArray();
        var ckptScales = checkpointed.Layers.OfType<DriftingScaleLayer>().Select(l => l.Scale).ToArray();
        Assert.Equal(plainScales, ckptScales);
        Assert.All(ckptScales, s => Assert.Equal(1.5f, s));
    }

    [Fact]
    public void CheckpointRecompute_WithoutTheContract_SeesTheChangedStateAndDiverges()
    {
        // Control for the test above: the same layer, ignoring the context, recomputes with the scale its first
        // forward already changed, so the checkpointed gradients differ. This is what the context exists to prevent.
        var (plain, checkpointed, input, target) = Pair(drifting: true);
        foreach (var layer in plain.Layers.OfType<DriftingScaleLayer>()) layer.KeepsCheckpointContract = false;
        foreach (var layer in checkpointed.Layers.OfType<DriftingScaleLayer>()) layer.KeepsCheckpointContract = false;

        var (_, plainGrads) = LossAndGradients(plain, input, target, mainScale: 1f);
        var (_, ckptGrads) = LossAndGradients(checkpointed, input, target, mainScale: 1f);

        double maxDiff = 0;
        for (int p = 0; p < plainGrads.Length; p++)
            for (int i = 0; i < plainGrads[p].Length; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(plainGrads[p][i] - ckptGrads[p][i]));
        Assert.True(maxDiff > 1e-3, $"expected the uncontracted recompute to diverge, max gradient difference {maxDiff}");
        Assert.All(checkpointed.Layers.OfType<DriftingScaleLayer>(), l => Assert.Equal(1.5f * 1.5f, l.Scale));
    }

    /// <summary>
    /// On a GPU engine with compilation on, the checkpointed network trains through the compiled fused step (the plan
    /// recomputes each segment inside its backward) instead of falling back to the eager tape, and reaches the losses
    /// and parameters of the plain compiled network, auxiliary terms included. Before checkpointing ran inside the
    /// plan, every checkpointed step was a fused-path miss.
    /// </summary>
    [SkippableFact]
    public void CheckpointedTraining_OnTheGpu_RunsTheCompiledStep_AndMatchesPlainCompiledTraining()
    {
        bool requireGpu = Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1";
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (!requireGpu) { throw new SkipException($"GPU engine construction failed ({ex.GetType().Name})."); }
        var prior = AiDotNetEngine.Current;
        var priorOptions = TensorCodecOptions.Current;
        try
        {
            bool available = gpu.SupportsGpu && gpu.IsGpuAvailable;
            if (requireGpu) Assert.True(available, "AIDOTNET_REQUIRE_GPU_TESTS=1 but no GPU backend is available.");
            Skip.IfNot(available, "No GPU backend available.");
            AiDotNetEngine.Current = gpu;
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            var (plain, checkpointed, input, target) = Pair(optimizer: () =>
                new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
                    new AiDotNet.Models.Options.AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }));
            plain.SetTrainingMode(true);
            checkpointed.SetTrainingMode(true);

            for (int step = 0; step < 6; step++)
            {
                plain.Train(input, target);
                int hits = AiDotNet.Configuration.TrainingDiagnosticsConfig.FusedOptimizerHits;
                int misses = AiDotNet.Configuration.TrainingDiagnosticsConfig.FusedOptimizerMisses;
                checkpointed.Train(input, target);
                Assert.True(AiDotNet.Configuration.TrainingDiagnosticsConfig.FusedOptimizerHits == hits + 1
                    && AiDotNet.Configuration.TrainingDiagnosticsConfig.FusedOptimizerMisses == misses,
                    $"step {step}: the checkpointed step did not run compiled " +
                    $"({AiDotNet.Configuration.TrainingDiagnosticsConfig.LastFusedOptimizerMissReason})");
                AssertClose(plain.GetLastLoss(), checkpointed.GetLastLoss(), $"step {step} loss");
            }

            var p = plain.GetParameters();
            var c = checkpointed.GetParameters();
            Assert.Equal(p.Length, c.Length);
            for (int i = 0; i < p.Length; i++)
                Assert.True(Math.Abs(p[i] - c[i]) <= 1e-5f * Math.Max(1f, Math.Abs(p[i])),
                    $"parameter {i} after 6 steps: plain {p[i]:R}, checkpointed {c[i]:R}");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(priorOptions);
            AiDotNetEngine.Current = prior;
            gpu.Dispose();
        }
    }
}
