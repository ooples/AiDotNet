using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// GraphAttentionLayer's sparse (CSR) attention kernel must reproduce the dense masked formulation it replaces: the
/// same outputs and the same parameter gradients, for concatenated and averaged heads, eagerly and under compiled
/// training.
/// </summary>
public class GraphAttentionSparseKernelTests
{
    private const int Nodes = 9;
    private const int Features = 5;

    [Theory(Timeout = 120000)]
    [InlineData(3, true)]
    [InlineData(2, false)]
    [InlineData(1, false)]
    public async Task SparseKernel_MatchesDenseMaskedAttention_OutputsAndGradients(int heads, bool concatenate)
    {
        await Task.Yield();
        var input = CreateInput([2, Nodes, Features]);
        var layer = CreateLayer(heads, concatenate);

        var (denseOut, denseGrads) = ForwardAndGradients(layer, input, sparse: false);
        var (sparseOut, sparseGrads) = ForwardAndGradients(layer, input, sparse: true);

        AssertClose(denseOut, sparseOut, 1e-5f, "output");
        Assert.Equal(denseGrads.Length, sparseGrads.Length);
        for (int p = 0; p < denseGrads.Length; p++)
            AssertClose(denseGrads[p], sparseGrads[p], 2e-5f, $"gradient of parameter {p}");
    }

    [Fact(Timeout = 120000)]
    public async Task SparseKernel_TrainingDropout_KeepsOrZeroesEachCoefficientAndStaysFinite()
    {
        await Task.Yield();
        var input = CreateInput([1, Nodes, Features]);
        var layer = new GraphAttentionLayer<float>(Features, 4, numHeads: 2, dropoutRate: 0.5,
            activationFunction: new ELUActivation<float>(), concatenateHeads: true);
        layer.SetAdjacencyMatrix(CreateAdjacency());
        layer.SetTrainingMode(true);
        var parameters = layer.GetTrainableParameters();
        using var tape = new GradientTape<float>();
        var output = layer.Forward(input);
        var loss = AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(output, output), null);
        var grads = tape.ComputeGradients(loss, parameters);
        Assert.All(output.AsSpan().ToArray(), v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));
        foreach (var p in parameters)
        {
            Assert.True(grads.TryGetValue(p, out var g), "every parameter must receive a gradient");
            Assert.All(g.AsSpan().ToArray(), v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SparseKernel_CompiledTraining_MatchesEager()
    {
        await Task.Yield();
        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();
            var input = CreateInput([1, Nodes, Features]);
            var target = new Tensor<float>([1, Nodes, 6]);
            var eager = CreateLayer(3, true, outputs: 2);
            var compiled = CreateLayer(3, true, outputs: 2);
            eager.SetTrainingMode(true);
            compiled.SetTrainingMode(true);
            IReadOnlyList<ITrainableLayer<float>> eagerLayers = [eager];
            IReadOnlyList<ITrainableLayer<float>> compiledLayers = [compiled];
            Func<Tensor<float>, Tensor<float>, Tensor<float>> loss = (prediction, expected) =>
            {
                var d = AiDotNetEngine.Current.TensorSubtract(prediction, expected);
                return AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(d, d), null);
            };
            for (int step = 0; step < 3; step++)
            {
                float e = TapeTrainingStep<float>.Step(eagerLayers, input, target, 0.01f, eager.Forward, loss);
                float c = CompiledTapeTrainingStep<float>.Step(compiledLayers, input, target, 0.01f, compiled.Forward, loss);
                Assert.InRange(Math.Abs(e - c), 0f, 1e-4f);
            }
            var ep = eager.GetTrainableParameters();
            var cp = compiled.GetTrainableParameters();
            for (int p = 0; p < ep.Count; p++) AssertClose(ep[p], cp[p], 1e-4f, $"parameter {p} after training");
        }
        finally
        {
            CompiledTapeTrainingStep<float>.Invalidate();
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }

    private static (Tensor<float> Output, Tensor<float>[] Grads) ForwardAndGradients(
        GraphAttentionLayer<float> layer, Tensor<float> input, bool sparse)
    {
        layer.UseSparseAttention = sparse;
        layer.SetTrainingMode(false);
        var parameters = layer.GetTrainableParameters();
        using var tape = new GradientTape<float>();
        var output = layer.Forward(input);
        var weighted = AiDotNetEngine.Current.TensorMultiply(output, CreateInput(output._shape));
        var loss = AiDotNetEngine.Current.ReduceSum(weighted, null);
        var grads = tape.ComputeGradients(loss, parameters);
        var result = new Tensor<float>[parameters.Count];
        for (int i = 0; i < parameters.Count; i++)
        {
            Assert.True(grads.TryGetValue(parameters[i], out var g), $"parameter {i} must receive a gradient");
            result[i] = new Tensor<float>(g.AsSpan().ToArray(), g._shape);
        }
        return (new Tensor<float>(output.AsSpan().ToArray(), output._shape), result);
    }

    private static GraphAttentionLayer<float> CreateLayer(int heads, bool concatenate, int outputs = 4)
    {
        var layer = new GraphAttentionLayer<float>(Features, outputs, numHeads: heads,
            activationFunction: new ELUActivation<float>(), concatenateHeads: concatenate);
        layer.SetAdjacencyMatrix(CreateAdjacency());
        var parameters = layer.GetTrainableParameters();
        for (int p = 0; p < parameters.Count; p++)
            for (int i = 0; i < parameters[p].Length; i++)
                parameters[p][i] = (float)(0.4 * Math.Sin(1.3 * i + 0.7 * p) + 0.05 * p);
        return layer;
    }

    // A ring with chords and self-loops: degrees differ, every row has an edge, most entries are masked.
    private static Tensor<float> CreateAdjacency()
    {
        var a = new Tensor<float>([Nodes, Nodes]);
        for (int i = 0; i < Nodes; i++)
        {
            a[i, i] = 1f;
            a[i, (i + 1) % Nodes] = 1f;
            a[(i + 1) % Nodes, i] = 1f;
            if (i % 3 == 0) a[i, (i + 4) % Nodes] = 1f;
        }
        return a;
    }

    private static Tensor<float> CreateInput(int[] shape)
    {
        var input = new Tensor<float>(shape);
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(0.8 * Math.Sin(i * 0.37) - 0.3 * Math.Cos(i * 0.11));
        return input;
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float tolerance, string what)
    {
        var e = expected.AsSpan().ToArray();
        var a = actual.AsSpan().ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
        {
            float scale = Math.Max(1f, Math.Abs(e[i]));
            Assert.True(Math.Abs(e[i] - a[i]) <= tolerance * scale,
                $"{what} differs at {i}: dense={e[i]:G9}, sparse={a[i]:G9}");
        }
    }
}
