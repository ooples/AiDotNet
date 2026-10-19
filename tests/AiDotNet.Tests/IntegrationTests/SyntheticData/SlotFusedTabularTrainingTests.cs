using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// The diffusion-based tabular generators step through MultiSlotFusedStep on the shared fused session (#1804).
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class SlotFusedTabularTrainingTests
{
    public static TheoryData<string> Generators => new() { "TabDDPM", "TabSyn", "FinDiff", "AutoDiffTab" };

    /// <summary>
    /// The adapter packs the slots into one row and unpacks them inside the forward. Several steps, each with its own
    /// slot data, must land where the same steps on the eager tape do: a slot read from a closure instead of the packed
    /// row would be baked at trace time and replayed for every later step.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Adapter_TakesTheEagerSteps_AcrossChangingSlots()
    {
        await Task.Yield();
        double[] Run(bool compile)
        {
            var original = TensorCodecOptions.Current;
            try
            {
                TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
                CompiledTapeTrainingStep<double>.Invalidate();
                var engine = AiDotNetEngine.Current;
                var weights = new Tensor<double>([3, 2]);
                for (int i = 0; i < weights.Length; i++) weights[i] = 0.1 * (i + 1) - 0.25;
                var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                    new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.05 });
                var owner = new object();
                using var step = new MultiSlotFusedStep<double>(owner);
                int fused = 0;
                for (int s = 0; s < 5; s++)
                {
                    var x = new Tensor<double>([4, 3]);
                    var noise = new Tensor<double>([4, 2]);
                    var scale = new Tensor<double>([1]);
                    for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.7 * i + s);
                    for (int i = 0; i < noise.Length; i++) noise[i] = Math.Cos(0.3 * i - s);
                    scale[0] = 0.5 + 0.25 * s;
                    Tensor<double> Forward(IReadOnlyList<Tensor<double>> slots)
                        => engine.TensorMultiply(engine.TensorMatMul(slots[0], weights), slots[2]);
                    Tensor<double> Loss(Tensor<double> predicted, IReadOnlyList<Tensor<double>> slots)
                    {
                        var d = engine.TensorSubtract(predicted, slots[1]);
                        return engine.ReduceMean(engine.TensorMultiply(d, d), new[] { 0, 1 }, keepDims: false);
                    }
                    bool ran = step.TryStep(() => new[] { weights }, null, new[] { x, noise, scale }, Forward, Loss, optimizer, out _);
                    if (!ran)
                    {
                        // The eager arm, as the generators run it when the plan does not apply.
                        TapeTrainingStepper<double>.EagerStep(new FusedTrainingStepRequest<double>
                        {
                            Layers = Array.Empty<ITrainableLayer<double>>(),
                            Input = x, Target = noise,
                            Forward = _ => Forward(new[] { x, noise, scale }),
                            ComputeLoss = (p, _) => Loss(p, new[] { x, noise, scale }),
                            Optimizer = optimizer, ExtraParameters = new[] { weights },
                        });
                    }
                    else fused++;
                }
                Assert.Equal(compile ? 5 : 0, fused);
                return weights.ToArray();
            }
            finally
            {
                TensorCodecOptions.SetCurrent(original);
                CompiledTapeTrainingStep<double>.Invalidate();
            }
        }

        var eager = Run(compile: false);
        var fused = Run(compile: true);
        for (int i = 0; i < eager.Length; i++)
            Assert.True(Math.Abs(eager[i] - fused[i]) <= 1e-6, $"weight {i}: eager {eager[i]}, fused {fused[i]}");
    }

    private static (Matrix<double> Data, List<ColumnMetadata> Columns) Data()
    {
        var random = new Random(42);
        var data = new Matrix<double>(24, 3);
        for (int i = 0; i < 24; i++)
        {
            data[i, 0] = random.NextDouble() * 4 - 2;
            data[i, 1] = random.NextDouble();
            data[i, 2] = random.Next(3);
        }
        var columns = new List<ColumnMetadata>
        {
            new("A", ColumnDataType.Continuous, columnIndex: 0),
            new("B", ColumnDataType.Continuous, columnIndex: 1),
            new("C", ColumnDataType.Categorical, new[] { "x", "y", "z" }, columnIndex: 2),
        };
        return (data, columns);
    }

    private static NeuralNetworkBase<double> Create(string name)
    {
        var arch = new NeuralNetworkArchitecture<double>(3, 3, NetworkComplexity.Simple);
        return name switch
        {
            "TabDDPM" => new TabDDPMGenerator<double>(arch, new TabDDPMOptions<double> { Seed = 7, MLPDimensions = [16, 16], NumTimesteps = 10, BatchSize = 12 }),
            "TabSyn" => new TabSynGenerator<double>(arch, new TabSynOptions<double> { Seed = 7 }),
            "FinDiff" => new FinDiffGenerator<double>(arch, new FinDiffOptions<double> { Seed = 7 }),
            "AutoDiffTab" => new AutoDiffTabGenerator<double>(arch, new AutoDiffTabOptions<double> { Seed = 7 }),
            _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };
    }

    /// <remarks>
    /// The generators' layers draw their initial weights from unseeded generators, so two fits never start equal and
    /// a fused-versus-eager weight comparison is not possible at this level; the adapter test above covers it. This
    /// asserts what the generator itself controls: with compilation on, every diffusion step runs fused from the
    /// first row (none declined, none run eagerly before the plan takes over).
    /// </remarks>
    [Theory(Timeout = 600000)]
    [MemberData(nameof(Generators))]
    public async Task CompiledFit_RunsEveryDiffusionStepFused(string name)
    {
        await Task.Yield();
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<double>.Invalidate();
            var generator = Create(name);
            var (data, columns) = Data();
            var fit = generator.GetType().GetMethod("Fit", new[] { typeof(Matrix<double>), typeof(IReadOnlyList<ColumnMetadata>), typeof(int?) })
                ?? throw new InvalidOperationException($"{name} has no Fit(Matrix, columns, epochs).");
            fit.Invoke(generator, new object?[] { data, columns, 1 });
            fit.Invoke(generator, new object?[] { data, columns, 1 });
            var step = (MultiSlotFusedStep<double>?)generator.GetType()
                .GetField("_fusedMultiSlotStep", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(generator);

            Assert.NotNull(step);
            if (step is null) return;
            Assert.True(step.FusedStepCount > 0, $"{name} ran no fused step.");
            Assert.Equal(0, step.DeclinedStepCount);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }
}