using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LoRA;

/// <summary>
/// Reference tests: merging an adapter into its base layer must give the same outputs as the adapter's own
/// forward pass, on non-square layers (where an untransposed or misaligned delta cannot hide).
/// </summary>
public class LoRAMergeReferenceTests
{
    private const int InputSize = 12;
    private const int OutputSize = 8;
    private const int Rank = 3;

    private static DenseLayer<double> CreateBaseLayer(int inputSize, int outputSize)
    {
        // Identity activation: the merged layer applies the activation to (W + ΔW)x + b,
        // which only equals the adapter's base(x) + ΔWx for a linear base layer.
        var layer = new DenseLayer<double>(outputSize, (IActivationFunction<double>)new IdentityActivation<double>());
        layer.ResolveFromShape(new[] { inputSize });
        return layer;
    }

    private static Tensor<double> CreateInput(int batchSize, int featureSize)
    {
        var data = new Vector<double>(batchSize * featureSize);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = Math.Sin(0.37 * i + 0.5);
        }
        return new Tensor<double>(new[] { batchSize, featureSize }, data);
    }

    /// <summary>
    /// Gives every trainable parameter a distinct non-zero value so the low-rank delta is not the zero matrix.
    /// </summary>
    private static void SetNonZeroParameters(LoRAAdapterBase<double> adapter)
    {
        var payload = new Vector<double>(adapter.GetParameters().Length);
        for (int i = 0; i < payload.Length; i++)
        {
            payload[i] = 0.05 * Math.Cos(0.61 * i + 0.2);
        }
        adapter.SetParameters(payload);
    }

    public static IEnumerable<object[]> Adapters()
    {
        yield return new object[] { "Standard" };
        yield return new object[] { "PiSSA" };
        yield return new object[] { "ReLoRA" };
        yield return new object[] { "LoKr" };
        yield return new object[] { "MoRA" };
    }

    private static LoRAAdapterBase<double> Create(string name)
    {
        return name switch
        {
            "Standard" => new StandardLoRAAdapter<double>(CreateBaseLayer(InputSize, OutputSize), Rank, 2.0 * Rank),
            "PiSSA" => CreatePiSSA(),
            "ReLoRA" => new ReLoRAAdapter<double>(CreateBaseLayer(InputSize, OutputSize), Rank, 2.0 * Rank),
            "LoKr" => new LoKrAdapter<double>(CreateBaseLayer(InputSize, OutputSize), Rank, 2.0 * Rank),
            // MoRA needs a square layer; a rank below the width exercises the compress/decompress layout
            "MoRA" => new MoRAAdapter<double>(CreateBaseLayer(OutputSize, OutputSize), Rank, 1.0),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown adapter")
        };
    }

    private static PiSSAAdapter<double> CreatePiSSA()
    {
        var baseLayer = CreateBaseLayer(InputSize, OutputSize);
        var adapter = new PiSSAAdapter<double>(baseLayer, Rank, 2.0 * Rank);
        var weights = new Matrix<double>(OutputSize, InputSize);
        var parameters = baseLayer.GetParameters();
        for (int o = 0; o < OutputSize; o++)
            for (int i = 0; i < InputSize; i++)
                weights[o, i] = parameters[i * OutputSize + o];
        adapter.InitializeFromSVD(weights);
        return adapter;
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Adapters))]
    public async Task MergeToOriginalLayer_MatchesAdapterForward(string name)
    {
        await Task.Yield();
        var adapter = Create(name);
        SetNonZeroParameters(adapter);
        int inputSize = name == "MoRA" ? OutputSize : InputSize;
        var input = CreateInput(3, inputSize);

        var adapterOutput = adapter.Forward(input);
        var mergedOutput = adapter.MergeToOriginalLayer().Forward(input);

        Assert.Equal(adapterOutput.Length, mergedOutput.Length);
        for (int i = 0; i < adapterOutput.Length; i++)
        {
            Assert.True(
                Math.Abs(adapterOutput[i] - mergedOutput[i]) <= 1e-9 + 1e-7 * Math.Abs(adapterOutput[i]),
                $"{name} output {i}: adapter {adapterOutput[i]:G12} vs merged {mergedOutput[i]:G12}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DyLoRA_TrainingStep_UpdatesOnlySampledRankAlongFiniteDifferenceGradient()
    {
        await Task.Yield();
        const int maxRank = 4;
        const int rank = 2;
        const double learningRate = 0.1;
        const double step = 1e-6;

        // A single active rank makes the sampled rank b deterministic
        var adapter = new DyLoRAAdapter<double>(CreateBaseLayer(InputSize, OutputSize), maxRank, new[] { rank }, alpha: 8.0);
        var lora = adapter.LoRALayer;
        var start = new double[lora.GetParameters().Length];
        for (int i = 0; i < start.Length; i++)
        {
            start[i] = 0.05 * Math.Cos(0.61 * i + 0.2);
        }
        lora.SetParameters(new Vector<double>(start));

        var input = CreateInput(1, InputSize);
        var targetData = new Vector<double>(OutputSize);
        for (int i = 0; i < OutputSize; i++)
        {
            targetData[i] = Math.Sin(0.9 * i + 1.1);
        }
        var target = new Tensor<double>(new[] { 1, OutputSize }, targetData);

        // 0.5 * squared error, whose output gradient (output - target) is what the training step backpropagates
        double Loss(Tensor<double> output, Tensor<double> expected)
        {
            double sum = 0;
            for (int i = 0; i < output.Length; i++)
            {
                double diff = output[i] - expected[i];
                sum += 0.5 * diff * diff;
            }
            return sum;
        }

        var gradient = new double[start.Length];
        for (int i = 0; i < start.Length; i++)
        {
            var plus = (double[])start.Clone();
            var minus = (double[])start.Clone();
            plus[i] += step;
            minus[i] -= step;
            lora.SetParameters(new Vector<double>(plus));
            double lossPlus = Loss(adapter.Forward(input), target);
            lora.SetParameters(new Vector<double>(minus));
            double lossMinus = Loss(adapter.Forward(input), target);
            gradient[i] = (lossPlus - lossMinus) / (2 * step);
        }
        lora.SetParameters(new Vector<double>(start));

        adapter.TrainWithNestedDropout(new[] { input }, new[] { target }, 1, learningRate, Loss);
        var after = lora.GetParameters();

        // Parameters: A [InputSize, maxRank] row-major, then B [maxRank, OutputSize] row-major
        int countA = InputSize * maxRank;
        for (int i = 0; i < start.Length; i++)
        {
            bool inSampledRank = i < countA ? (i % maxRank) < rank : ((i - countA) / OutputSize) < rank;
            if (inSampledRank)
            {
                double expected = start[i] - learningRate * gradient[i];
                Assert.True(
                    Math.Abs(expected - after[i]) <= 1e-8,
                    $"parameter {i}: expected {expected:G12}, got {after[i]:G12}");
            }
            else
            {
                Assert.Equal(start[i], after[i]);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DyLoRA_TrainingStep_WithUnfrozenBase_TrainsBaseAlongFiniteDifferenceGradient()
    {
        await Task.Yield();
        const double learningRate = 0.1;
        const double step = 1e-6;

        var baseLayer = CreateBaseLayer(InputSize, OutputSize);
        var adapter = new DyLoRAAdapter<double>(baseLayer, 4, new[] { 2 }, alpha: 8.0, freezeBaseLayer: false);
        var loraStart = new double[adapter.LoRALayer.GetParameters().Length];
        for (int i = 0; i < loraStart.Length; i++)
        {
            loraStart[i] = 0.05 * Math.Cos(0.61 * i + 0.2);
        }
        adapter.LoRALayer.SetParameters(new Vector<double>(loraStart));
        var start = baseLayer.GetParameters().ToArray();

        var input = CreateInput(1, InputSize);
        var targetData = new Vector<double>(OutputSize);
        for (int i = 0; i < OutputSize; i++)
        {
            targetData[i] = Math.Sin(0.9 * i + 1.1);
        }
        var target = new Tensor<double>(new[] { 1, OutputSize }, targetData);

        double Loss(Tensor<double> output, Tensor<double> expected)
        {
            double sum = 0;
            for (int i = 0; i < output.Length; i++)
            {
                double diff = output[i] - expected[i];
                sum += 0.5 * diff * diff;
            }
            return sum;
        }

        var gradient = new double[start.Length];
        for (int i = 0; i < start.Length; i++)
        {
            var plus = (double[])start.Clone();
            var minus = (double[])start.Clone();
            plus[i] += step;
            minus[i] -= step;
            baseLayer.SetParameters(new Vector<double>(plus));
            double lossPlus = Loss(adapter.Forward(input), target);
            baseLayer.SetParameters(new Vector<double>(minus));
            double lossMinus = Loss(adapter.Forward(input), target);
            gradient[i] = (lossPlus - lossMinus) / (2 * step);
        }
        baseLayer.SetParameters(new Vector<double>(start));

        adapter.TrainWithNestedDropout(new[] { input }, new[] { target }, 1, learningRate, Loss);
        var after = baseLayer.GetParameters();

        for (int i = 0; i < start.Length; i++)
        {
            double expected = start[i] - learningRate * gradient[i];
            Assert.True(
                Math.Abs(expected - after[i]) <= 1e-8,
                $"base parameter {i}: expected {expected:G12}, got {after[i]:G12}");
        }
    }
    private static void AssertSameOutputs(string label, Tensor<double> expected, Tensor<double> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(
                Math.Abs(expected[i] - actual[i]) <= 1e-9 + 1e-7 * Math.Abs(expected[i]),
                $"{label} output {i}: adapter {expected[i]:G12} vs merged {actual[i]:G12}");
        }
    }

    public static IEnumerable<object[]> OutputMajorAdapters()
    {
        yield return new object[] { "Standard" };
        yield return new object[] { "ReLoRA" };
        yield return new object[] { "LoKr" };
    }

    /// <summary>
    /// FullyConnectedLayer stores its weights output-major ([outputSize, inputSize]); on a non-square layer
    /// the merge must transpose the input-major LoRA delta before adding it.
    /// </summary>
    [Theory(Timeout = 120000)]
    [MemberData(nameof(OutputMajorAdapters))]
    public async Task MergeToOriginalLayer_OnOutputMajorBase_MatchesAdapterForward(string name)
    {
        await Task.Yield();
        var baseLayer = new FullyConnectedLayer<double>(InputSize, OutputSize);
        LoRAAdapterBase<double> adapter = name switch
        {
            "Standard" => new StandardLoRAAdapter<double>(baseLayer, Rank, 2.0 * Rank),
            "ReLoRA" => new ReLoRAAdapter<double>(baseLayer, Rank, 2.0 * Rank),
            "LoKr" => new LoKrAdapter<double>(baseLayer, Rank, 2.0 * Rank),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown adapter")
        };
        SetNonZeroParameters(adapter);
        var input = CreateInput(3, InputSize);

        AssertSameOutputs(name, adapter.Forward(input), adapter.MergeToOriginalLayer().Forward(input));
    }

    [Theory(Timeout = 120000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task MultiLoRA_MergeTaskToLayer_MatchesTaskForward(bool outputMajor)
    {
        await Task.Yield();
        ILayer<double> baseLayer = outputMajor
            ? new FullyConnectedLayer<double>(InputSize, OutputSize)
            : CreateBaseLayer(InputSize, OutputSize);
        var adapter = new MultiLoRAAdapter<double>(baseLayer, "first", Rank, 2.0 * Rank);
        adapter.AddTask("second", Rank + 1, 2.0 * Rank);
        SetNonZeroParameters(adapter);
        var input = CreateInput(3, InputSize);

        foreach (var task in new[] { "first", "second" })
        {
            adapter.SetCurrentTask(task);
            AssertSameOutputs(task, adapter.Forward(input), adapter.MergeTaskToLayer(task).Forward(input));
        }
    }

    /// <summary>
    /// The tensor-valued objective overload must train on the objective it is given. With an absolute-error
    /// loss, the step must follow the finite-difference gradient of sum |output - target|, not of the
    /// half squared error that the scalar overload differentiates.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DyLoRA_TensorObjective_TrainsAlongFiniteDifferenceGradientOfThatObjective()
    {
        await Task.Yield();
        const int maxRank = 4;
        const int rank = 2;
        const double learningRate = 0.1;
        const double step = 1e-6;

        var adapter = new DyLoRAAdapter<double>(CreateBaseLayer(InputSize, OutputSize), maxRank, new[] { rank }, alpha: 8.0);
        var lora = adapter.LoRALayer;
        var start = new double[lora.GetParameters().Length];
        for (int i = 0; i < start.Length; i++)
        {
            start[i] = 0.05 * Math.Cos(0.61 * i + 0.2);
        }
        lora.SetParameters(new Vector<double>(start));

        var input = CreateInput(1, InputSize);
        var targetData = new Vector<double>(OutputSize);
        for (int i = 0; i < OutputSize; i++)
        {
            targetData[i] = Math.Sin(0.9 * i + 1.1);
        }
        var target = new Tensor<double>(new[] { 1, OutputSize }, targetData);

        double AbsoluteError(Tensor<double> output)
        {
            double sum = 0;
            for (int i = 0; i < output.Length; i++) sum += Math.Abs(output[i] - target[i]);
            return sum;
        }

        Tensor<double> AbsoluteErrorObjective(Tensor<double> output, Tensor<double> expected)
        {
            var engine = AiDotNetEngine.Current;
            var difference = engine.TensorAbs(engine.TensorSubtract(output, engine.Reshape(expected, output.Shape.ToArray())));
            return engine.ReduceSum(engine.Reshape(difference, new[] { difference.Length }), new[] { 0 }, true);
        }

        var gradient = new double[start.Length];
        for (int i = 0; i < start.Length; i++)
        {
            var plus = (double[])start.Clone();
            var minus = (double[])start.Clone();
            plus[i] += step;
            minus[i] -= step;
            lora.SetParameters(new Vector<double>(plus));
            double lossPlus = AbsoluteError(adapter.Forward(input));
            lora.SetParameters(new Vector<double>(minus));
            double lossMinus = AbsoluteError(adapter.Forward(input));
            gradient[i] = (lossPlus - lossMinus) / (2 * step);
        }
        lora.SetParameters(new Vector<double>(start));

        adapter.TrainWithNestedDropout(new[] { input }, new[] { target }, 1, learningRate, AbsoluteErrorObjective);
        var after = lora.GetParameters();

        for (int i = 0; i < start.Length; i++)
        {
            double expected = start[i] - learningRate * gradient[i];
            Assert.True(
                Math.Abs(expected - after[i]) <= 1e-7,
                $"parameter {i}: expected {expected:G12}, got {after[i]:G12}");
        }
    }
}
