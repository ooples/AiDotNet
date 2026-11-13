using System;
using System.Linq;
using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LoRA;

public class LoRAReviewRegressionTests
{
    [Fact]
    public void Nola_NoGradDoesNotRecordCustomBasisNodes()
    {
        var adapter = new NOLAAdapter<double>(Dense(5, 3), rank: 2, numBasis: 17);
        _ = adapter.GetTrainableParameters();
        using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>();
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<double>())
            _ = adapter.Forward(new Tensor<double>(new[] { 0.2, 0.3, 0.4, 0.5, 0.6 }, new[] { 1, 5 }));
        Assert.Equal(0, tape.EntryCount);
    }


    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static Tensor<double> BuildNolaObjective(NOLAAdapter<double> adapter, Tensor<double> input)
        => Objective(adapter, input);

    [Fact]
    public void Nola_RegeneratesBasesWithoutRetainingForwardBlocksOnTheTape()
    {
        var previous = AiDotNetEngine.Current;
        var engine = new BasisEngine();
        try
        {
            AiDotNetEngine.Current = engine;
            int? entryCount = null;
            foreach (int bases in new[] { 17, 257 })
            {
                engine.BasisBuffers.Clear();
                var adapter = new NOLAAdapter<double>(Dense(64, 5), rank: 5, numBasis: bases);
                adapter.SetParameters(new Vector<double>(Enumerable.Repeat(0.01, 2 * bases).ToArray()));
                var input = new Tensor<double>(Enumerable.Repeat(0.1, 64).ToArray(), new[] { 1, 64 });
                using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>();
                var loss = BuildNolaObjective(adapter, input);
                Assert.NotEmpty(engine.BasisBuffers);
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                Assert.All(engine.BasisBuffers, buffer => Assert.False(buffer.IsAlive,
                    $"entries={tape.EntryCount}, arenaActive={AiDotNet.Tensors.Helpers.TensorArena.Current is not null}"));
                if (entryCount.HasValue) Assert.Equal(entryCount.Value, tape.EntryCount);
                entryCount = tape.EntryCount;
                int forwardBlocks = engine.Blocks;
                var parameters = adapter.GetTrainableParameters();
                var gradients = tape.ComputeGradients(loss, parameters);
                Assert.All(parameters, parameter =>
                {
                    Assert.True(gradients.TryGetValue(parameter, out var gradient));
                    Assert.NotNull(gradient);
                    Assert.Contains(gradient.ToArray(), x => Math.Abs(x) > 1e-12);
                });
                Assert.True(engine.Blocks > forwardBlocks);
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private static DenseLayer<double> Dense(int input, int output)
    {
        var layer = new DenseLayer<double>(output, (IActivationFunction<double>)new IdentityActivation<double>());
        layer.ResolveFromShape(new[] { input });
        return layer;
    }

    private static Matrix<double> Shared(Type type, string name) =>
        (Matrix<double>)(type.GetField(name, BindingFlags.Static | BindingFlags.NonPublic)?.GetValue(null)
            ?? throw new InvalidOperationException(name));

    private static void Close(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-10 + 1e-9 * Math.Abs(expected[i]), $"{i}: {expected[i]} vs {actual[i]}");
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void DirectionAdapters_PreserveWeightLayoutAndBias(bool dVora, bool outputMajor)
    {
        const int inputs = 5, outputs = 3, rank = 2;
        ILayer<double> layer = outputMajor ? new FullyConnectedLayer<double>(inputs, outputs) : Dense(inputs, outputs);
        var parameters = new double[inputs * outputs + outputs];
        var weights = new double[outputs, inputs];
        for (int o = 0; o < outputs; o++)
        {
            for (int i = 0; i < inputs; i++)
            {
                weights[o, i] = 0.03 * (1 + i + 7 * o);
                parameters[outputMajor ? o * inputs + i : i * outputs + o] = weights[o, i];
            }
            parameters[inputs * outputs + o] = 0.4 + 0.1 * o;
        }
        layer.SetParameters(new Vector<double>(parameters));
        var input = new Tensor<double>(new[] { 0.2, -0.3, 0.5, 0.7, -0.1 }, new[] { 1, inputs });
        if (dVora) DVoRAAdapter<double>.InitializeSharedMatrices(inputs, outputs, rank, seed: 13);
        try
        {
            LoRAAdapterBase<double> adapter = dVora
                ? new DVoRAAdapter<double>(layer, rank, alpha: 4)
                : new DoRAAdapter<double>(layer, rank, alpha: 4);
            var expected = new double[outputs];
            for (int o = 0; o < outputs; o++)
            {
                double magnitude = 0;
                for (int i = 0; i < inputs; i++) magnitude += weights[o, i] * weights[o, i];
                magnitude = Math.Sqrt(magnitude);
                var direction = new double[inputs];
                for (int i = 0; i < inputs; i++)
                {
                    double delta = 0;
                    if (dVora)
                    {
                        var a = Shared(typeof(DVoRAAdapter<double>), "_sharedMatrixA");
                        var b = Shared(typeof(DVoRAAdapter<double>), "_sharedMatrixB");
                        for (int r = 0; r < rank; r++) delta += 2 * a[i, r] * b[r, o];
                    }
                    direction[i] = weights[o, i] / magnitude + delta;
                }
                double norm = Math.Sqrt(direction.Sum(v => v * v));
                expected[o] = parameters[inputs * outputs + o];
                for (int i = 0; i < inputs; i++) expected[o] += input[i] * magnitude * direction[i] / norm;
            }
            Close(expected, adapter.Forward(input).ToArray());
            Close(expected, adapter.MergeToOriginalLayer().Forward(input).ToArray());

            var changed = new Vector<double>(Enumerable.Range(0, adapter.GetParameters().Length)
                .Select(i => 0.15 + 0.03 * Math.Sin(i * 0.71)).ToArray());
            adapter.SetParameters(changed);
            Close(adapter.Forward(input).ToArray(), adapter.MergeToOriginalLayer().Forward(input).ToArray());
            Close(parameters, layer.GetParameters().ToArray());
        }
        finally { if (dVora) DVoRAAdapter<double>.ResetSharedMatrices(); }
    }

    private sealed class BasisEngine : CpuEngine
    {
        internal int Blocks;
        internal readonly System.Collections.Generic.List<WeakReference> BasisBuffers = new();
        public override Tensor<T> TensorMultiply<T>(Tensor<T> a, Tensor<T> b)
        {
            if (a.Shape.Length == 3 && b.Shape.Length == 3)
            {
                Blocks++;
                BasisBuffers.Add(new WeakReference(a));
                Assert.InRange(a.Shape[0], 1, 16);
                Assert.True(a.Length <= Math.Max(65_536L, (long)a.Shape[1] * a.Shape[2]));
            }
            return base.TensorMultiply(a, b);
        }
    }

    private static double[,] BasisSum(int rows, int columns, int offset, double[] coefficients)
    {
        var result = new double[rows, columns];
        for (int b = 0; b < coefficients.Length; b++)
        {
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42 + offset + b);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < columns; j++) result[i, j] += coefficients[b] * (2 * random.NextDouble() - 1);
        }
        return result;
    }

    private static double Loss(Tensor<double> output) => output.ToArray().Sum(x => 0.5 * x * x);

    private static Tensor<double> Objective(LoRAAdapterBase<double> adapter, Tensor<double> input)
    {
        var engine = AiDotNetEngine.Current;
        var output = adapter.Forward(input);
        var squared = engine.TensorMultiply(output, output);
        return engine.TensorMultiplyScalar(engine.ReduceSum(engine.Reshape(squared, new[] { squared.Length }), new[] { 0 }, true), 0.5);
    }

    [Theory]
    [InlineData(64, 5, 33)]
    [InlineData(64, 5, 65)]
    [InlineData(257, 257, 3)]
    public void Nola_BoundsBasisBuffersAndPreservesCoefficientGradients(int inputs, int rank, int bases)
    {
        int outputs = Math.Max(3, rank);
        var layer = Dense(inputs, outputs);
        var adapter = new NOLAAdapter<double>(layer, rank, bases, alpha: rank, seed: 42);
        var parameters = Enumerable.Range(0, 2 * bases).Select(i => 0.05 * Math.Sin(0.27 * i + 0.2)).ToArray();
        adapter.SetParameters(new Vector<double>(parameters));
        var input = new Tensor<double>(Enumerable.Range(0, inputs).Select(i => Math.Cos(0.19 * i)).ToArray(), new[] { 1, inputs });
        var a = BasisSum(inputs, rank, 0, parameters.Take(bases).ToArray());
        var b = BasisSum(rank, outputs, bases, parameters.Skip(bases).ToArray());
        var expected = layer.Forward(input).ToArray();
        var intermediate = new double[rank];
        for (int r = 0; r < rank; r++)
            for (int i = 0; i < inputs; i++) intermediate[r] += input[i] * a[i, r];
        for (int o = 0; o < outputs; o++)
            for (int r = 0; r < rank; r++)
                expected[o] += intermediate[r] * b[r, o];
        var previous = AiDotNetEngine.Current;
        var engine = new BasisEngine();
        try
        {
            AiDotNetEngine.Current = engine;
            Close(expected, adapter.Forward(input).ToArray());
            var indices = new[] { 0, 2 * bases - 1 };
            var gradients = new double[indices.Length];
            const double step = 1e-5, rate = 0.001;
            for (int i = 0; i < indices.Length; i++)
            {
                var plus = (double[])parameters.Clone();
                var minus = (double[])parameters.Clone();
                plus[indices[i]] += step;
                minus[indices[i]] -= step;
                adapter.SetParameters(new Vector<double>(plus));
                double upper = Loss(adapter.Forward(input));
                adapter.SetParameters(new Vector<double>(minus));
                double lower = Loss(adapter.Forward(input));
                gradients[i] = (upper - lower) / (2 * step);
            }
            adapter.SetParameters(new Vector<double>(parameters));
            TapeTrainingStepper<double>.EagerSgdObjectiveStep(() => adapter.GetTrainableParameters(), () => Objective(adapter, input), rate);
            var updated = adapter.GetParameters();
            for (int i = 0; i < indices.Length; i++)
            {
                Assert.True(Math.Abs(gradients[i]) > 1e-8);
                double expectedUpdate = parameters[indices[i]] - rate * gradients[i];
                Assert.True(Math.Abs(updated[indices[i]] - expectedUpdate) < 1e-7,
                    $"coefficient {indices[i]}: expected {expectedUpdate:G12}, actual {updated[indices[i]]:G12}, gradient {gradients[i]:G12}");
            }
            Assert.True(engine.Blocks > 1);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void Nola_FreshAdapterPreservesBaseOutputAndCanLearn()
    {
        var layer = Dense(5, 3);
        layer.SetParameters(new Vector<double>(Enumerable.Repeat(0.1, layer.GetParameters().Length).ToArray()));
        var adapter = new NOLAAdapter<double>(layer, rank: 2, numBasis: 5, alpha: 2);
        var input = new Tensor<double>(new[] { 0.2, 0.3, 0.4, 0.5, 0.6 }, new[] { 1, 5 });
        Close(layer.Forward(input).ToArray(), adapter.Forward(input).ToArray());
        double before = Loss(adapter.Forward(input));
        TapeTrainingStepper<double>.EagerSgdObjectiveStep(() => adapter.GetTrainableParameters(), () => Objective(adapter, input), 1e-5);
        Assert.Contains(adapter.GetCoefficientsB().ToArray(), x => Math.Abs(x) > 1e-10);
        Assert.True(Loss(adapter.Forward(input)) < before);
    }

    [Fact]
    public void TiedLora_LayerScalingReceivesTheFiniteDifferenceGradient()
    {
        const int inputs = 5, outputs = 3, rank = 2;
        TiedLoRAAdapter<double>.InitializeSharedMatrices(inputs, outputs, rank, seed: 7);
        try
        {
            var b = Shared(typeof(TiedLoRAAdapter<double>), "_sharedMatrixB");
            for (int r = 0; r < rank; r++)
                for (int o = 0; o < outputs; o++) b[r, o] = 0.1 * (1 + r + o);
            var adapter = new TiedLoRAAdapter<double>(Dense(inputs, outputs), rank, alpha: 4);
            Assert.Equal(1, adapter.GetParameters().Length);
            var input = new Tensor<double>(new[] { 0.2, -0.3, 0.5, 0.7, -0.1 }, new[] { 1, inputs });
            const double initial = 0.7, step = 1e-5, rate = 0.01;
            adapter.SetParameters(new Vector<double>(new[] { initial + step }));
            double upper = Loss(adapter.Forward(input));
            adapter.SetParameters(new Vector<double>(new[] { initial - step }));
            double lower = Loss(adapter.Forward(input));
            double gradient = (upper - lower) / (2 * step);
            Assert.True(Math.Abs(gradient) > 1e-8);
            adapter.SetParameters(new Vector<double>(new[] { initial }));
            TapeTrainingStepper<double>.EagerSgdObjectiveStep(() => adapter.GetTrainableParameters(), () => Objective(adapter, input), rate);
            Assert.True(Math.Abs(adapter.GetParameters()[0] - (initial - rate * gradient)) < 1e-8);
        }
        finally { TiedLoRAAdapter<double>.ResetSharedMatrices(); }
    }
}
