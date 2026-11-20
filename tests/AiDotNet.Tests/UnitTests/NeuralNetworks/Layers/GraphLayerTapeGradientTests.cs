using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Tape-gradient reachability and finite-difference checks for layers whose forward pass used to copy
/// tensors element-wise (detaching the autodiff graph): every trainable parameter must receive a nonzero
/// gradient after one backward pass, and the tape gradient must match central differences in double.
/// </summary>
public class GraphLayerTapeGradientTests
{
    private const double FiniteDifferenceStep = 1e-6;
    private const double RelativeTolerance = 1e-4;
    private const double AbsoluteTolerance = 1e-7;

    private static Tensor<double> RandomTensor(Random rng, params int[] shape)
    {
        var t = new Tensor<double>(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (rng.NextDouble() * 2) - 1;
        return t;
    }

    /// <summary>sum(output * projection): a scalar loss whose gradient reaches every output element.</summary>
    private static Tensor<double> Loss(Tensor<double> output, Tensor<double> projection)
    {
        var engine = AiDotNetEngine.Current;
        return engine.ReduceSum(
            engine.TensorMultiply(output, projection), Enumerable.Range(0, output.Shape.Length).ToArray(), keepDims: false);
    }

    private static double LossValue(ILayer<double> layer, Tensor<double> input, Tensor<double> projection)
    {
        using var noGrad = new NoGradScope<double>();
        return Loss(layer.Forward(input), projection).GetFlat(0);
    }

    /// <summary>Runs one tape forward/backward and returns the parameters with their gradients.</summary>
    private static (IReadOnlyList<Tensor<double>> Parameters, Dictionary<Tensor<double>, Tensor<double>> Gradients)
        TapeGradients(ILayer<double> layer, Tensor<double> input, Tensor<double> projection)
    {
        using var tape = new GradientTape<double>();
        var output = layer.Forward(input);
        var parameters = TapeTrainingStep<double>.CollectParameters(new[] { layer }, structureVersion: -1);
        var gradients = tape.ComputeGradients(Loss(output, projection), parameters);
        return (parameters, gradients);
    }

    private static void AssertEveryParameterReached(ILayer<double> layer, Tensor<double> input, Tensor<double> projection)
    {
        var (parameters, gradients) = TapeGradients(layer, input, projection);
        Assert.NotEmpty(parameters);
        for (int i = 0; i < parameters.Count; i++)
        {
            Assert.True(gradients.TryGetValue(parameters[i], out var gradient) && gradient is not null,
                $"parameter {i} ({string.Join("x", parameters[i].Shape.ToArray())}) received no gradient");
            double largest = gradient.ToArray().Max(Math.Abs);
            Assert.True(largest > 1e-12,
                $"parameter {i} ({string.Join("x", parameters[i].Shape.ToArray())}) received only zero gradients");
        }
    }

    /// <summary>Central differences on a few elements of every parameter against the tape gradient.</summary>
    private static void AssertMatchesFiniteDifferences(ILayer<double> layer, Tensor<double> input, Tensor<double> projection)
    {
        var (parameters, gradients) = TapeGradients(layer, input, projection);
        Assert.NotEmpty(parameters);
        for (int p = 0; p < parameters.Count; p++)
        {
            var parameter = parameters[p];
            Assert.True(gradients.TryGetValue(parameter, out var gradient) && gradient is not null,
                $"parameter {p} received no gradient");
            var analytic = gradient.ToArray();
            int length = parameter.Length;
            foreach (int k in new[] { 0, length / 2, length - 1 }.Distinct())
            {
                var span = parameter.AsWritableSpan();
                double original = span[k];
                span[k] = original + FiniteDifferenceStep;
                double plus = LossValue(layer, input, projection);
                span = parameter.AsWritableSpan();
                span[k] = original - FiniteDifferenceStep;
                double minus = LossValue(layer, input, projection);
                span = parameter.AsWritableSpan();
                span[k] = original;

                double numeric = (plus - minus) / (2 * FiniteDifferenceStep);
                double tolerance = AbsoluteTolerance + (RelativeTolerance * Math.Max(Math.Abs(numeric), Math.Abs(analytic[k])));
                Assert.True(Math.Abs(numeric - analytic[k]) <= tolerance,
                    $"parameter {p} element {k}: tape gradient {analytic[k]} vs finite difference {numeric}");
            }
        }
    }

    private static GraphTransformerLayer<double> TinyGraphTransformer()
    {
        var layer = new GraphTransformerLayer<double>(
            inputFeatures: 4, outputFeatures: 3, numHeads: 2, headDim: 3, useStructuralEncoding: false, dropoutRate: 0.0,
            activationFunction: null);
        // A connected 4-node graph with self loops.
        var adjacency = new Tensor<double>(new[] { 4, 4 });
        foreach (var (i, j) in new[] { (0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3) })
            adjacency[i, j] = 1.0;
        layer.SetAdjacencyMatrix(adjacency);
        return layer;
    }

    private static EdgeConditionalConvolutionalLayer<double> TinyEdgeConditional(Random rng)
    {
        var layer = new EdgeConditionalConvolutionalLayer<double>(
            inputFeatures: 3, outputFeatures: 2, edgeFeatures: 2, edgeNetworkHiddenDim: 4);
        // Five directed edges in row-major order, one edge-feature row each.
        var adjacency = new Tensor<double>(new[] { 3, 3 });
        foreach (var (i, j) in new[] { (0, 1), (1, 0), (1, 2), (2, 0), (2, 2) })
            adjacency[i, j] = 1.0;
        layer.SetAdjacencyMatrix(adjacency);
        // Positive edge features keep the edge network's ReLU units active, away from the kink.
        var edgeFeatures = RandomTensor(rng, 5, 2);
        var span = edgeFeatures.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = 0.5 + Math.Abs(span[i]);
        layer.SetEdgeFeatures(edgeFeatures);
        return layer;
    }

    /// <summary>
    /// Two node types of equal width on a 4-node graph (a, b, a, b). With <paramref name="useBasis"/> three
    /// edge types share two bases, so the basis matrices and per-edge coefficients are the edge weights.
    /// </summary>
    private static HeterogeneousGraphLayer<double> TinyHeterogeneous(Random rng, bool useBasis)
    {
        var metadata = new HeterogeneousGraphMetadata
        {
            NodeTypes = new[] { "a", "b" },
            EdgeTypes = useBasis ? new[] { "ab", "ba", "aa" } : new[] { "ab", "ba" },
            NodeTypeFeatures = new Dictionary<string, int> { ["a"] = 3, ["b"] = 3 },
            EdgeTypeSchema = new Dictionary<string, (string SourceType, string TargetType)>
            {
                ["ab"] = ("a", "b"),
                ["ba"] = ("b", "a"),
                ["aa"] = ("a", "a"),
            },
        };
        var layer = new HeterogeneousGraphLayer<double>(metadata, outputFeatures: 2, useBasis: useBasis, numBases: 2);
        var adjacencies = new Dictionary<string, Tensor<double>>();
        foreach (var edgeType in metadata.EdgeTypes)
        {
            var adjacency = RandomTensor(rng, 4, 4);
            var span = adjacency.AsWritableSpan();
            for (int i = 0; i < span.Length; i++) span[i] = span[i] > -0.2 ? 1.0 : 0.0;
            adjacencies[edgeType] = adjacency;
        }
        layer.SetAdjacencyMatrices(adjacencies);
        layer.SetNodeTypeMap(new Dictionary<int, string> { [0] = "a", [1] = "b", [2] = "a", [3] = "b" });
        return layer;
    }

    private static DirectionalGraphLayer<double> TinyDirectional()
    {
        var layer = new DirectionalGraphLayer<double>(inputFeatures: 3, outputFeatures: 2, useGating: true);
        // A directed 4-node graph: adjacency[i, j] = 1 for the edge j -> i.
        var adjacency = new Tensor<double>(new[] { 4, 4 });
        foreach (var (i, j) in new[] { (1, 0), (2, 1), (3, 2), (0, 3), (2, 0) })
            adjacency[i, j] = 1.0;
        layer.SetAdjacencyMatrix(adjacency);
        return layer;
    }

    [Fact]
    public void PositiveControl_DenseLayer_EveryParameterReceivesGradient()
    {
        // The harness itself: a layer known to train through the tape passes it.
        var rng = RandomHelper.CreateSeededRandom(17);
        var layer = new DenseLayer<double>(3, (IActivationFunction<double>)new IdentityActivation<double>());
        var input = RandomTensor(rng, 2, 4);
        _ = layer.Forward(input);
        AssertEveryParameterReached(layer, input, RandomTensor(rng, 2, 3));
    }

    [Fact]
    public void PositiveControl_DetachedParameter_IsReportedWithoutGradient()
    {
        // The harness can fail: a parameter used only inside a no-grad scope gets no gradient.
        var engine = AiDotNetEngine.Current;
        var rng = RandomHelper.CreateSeededRandom(18);
        var weight = RandomTensor(rng, 3);
        using var tape = new GradientTape<double>();
        Tensor<double> detached;
        using (new NoGradScope<double>())
        {
            detached = engine.TensorMultiplyScalar(weight, 2.0);
        }
        var loss = engine.ReduceSum(detached, new[] { 0 }, keepDims: false);
        var gradients = tape.ComputeGradients(loss, new[] { weight });
        bool reached = gradients.TryGetValue(weight, out var gradient)
            && gradient is not null && gradient.ToArray().Any(g => Math.Abs(g) > 1e-12);
        Assert.False(reached);
    }

    [Fact]
    public void GraphTransformerLayer_EveryTrainableParameterReceivesGradient()
    {
        var rng = RandomHelper.CreateSeededRandom(21);
        var layer = TinyGraphTransformer();
        AssertEveryParameterReached(layer, RandomTensor(rng, 4, 4), RandomTensor(rng, 4, 3));
    }

    [Fact]
    public void GraphTransformerLayer_TapeGradientMatchesFiniteDifferences()
    {
        var rng = RandomHelper.CreateSeededRandom(22);
        var layer = TinyGraphTransformer();
        AssertMatchesFiniteDifferences(layer, RandomTensor(rng, 4, 4), RandomTensor(rng, 4, 3));
    }

    [Fact]
    public void EdgeConditionalConvolutionalLayer_EveryTrainableParameterReceivesGradient()
    {
        var rng = RandomHelper.CreateSeededRandom(31);
        var layer = TinyEdgeConditional(rng);
        AssertEveryParameterReached(layer, RandomTensor(rng, 3, 3), RandomTensor(rng, 3, 2));
    }

    [Fact]
    public void EdgeConditionalConvolutionalLayer_TapeGradientMatchesFiniteDifferences()
    {
        var rng = RandomHelper.CreateSeededRandom(32);
        var layer = TinyEdgeConditional(rng);
        AssertMatchesFiniteDifferences(layer, RandomTensor(rng, 3, 3), RandomTensor(rng, 3, 2));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HeterogeneousGraphLayer_EveryTrainableParameterReceivesGradient(bool useBasis)
    {
        // The self-loop weights and biases were written into the output in place (off the tape), and the
        // basis reconstruction ran in a no-grad scope, so those parameters never trained.
        var rng = RandomHelper.CreateSeededRandom(useBasis ? 52 : 51);
        var layer = TinyHeterogeneous(rng, useBasis);
        AssertEveryParameterReached(layer, RandomTensor(rng, 4, 3), RandomTensor(rng, 4, 2));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HeterogeneousGraphLayer_TapeGradientMatchesFiniteDifferences(bool useBasis)
    {
        var rng = RandomHelper.CreateSeededRandom(useBasis ? 54 : 53);
        var layer = TinyHeterogeneous(rng, useBasis);
        AssertMatchesFiniteDifferences(layer, RandomTensor(rng, 4, 3), RandomTensor(rng, 4, 2));
    }

    [Fact]
    public void DirectionalGraphLayer_Gated_EveryTrainableParameterReceivesGradient()
    {
        // The concatenation of the three branches and the gating ran in no-grad scopes, so the incoming,
        // outgoing and self weights and biases, and the gate parameters, never trained.
        var rng = RandomHelper.CreateSeededRandom(61);
        var layer = TinyDirectional();
        AssertEveryParameterReached(layer, RandomTensor(rng, 4, 3), RandomTensor(rng, 4, 2));
    }

    [Fact]
    public void DirectionalGraphLayer_Gated_TapeGradientMatchesFiniteDifferences()
    {
        var rng = RandomHelper.CreateSeededRandom(62);
        var layer = TinyDirectional();
        AssertMatchesFiniteDifferences(layer, RandomTensor(rng, 4, 3), RandomTensor(rng, 4, 2));
    }

    [Fact]
    public void DeformableConvolutionalLayer_EveryTrainableParameterReceivesGradient()
    {
        // The main, offset and mask biases were added in place outside the tape, so they never trained.
        var rng = RandomHelper.CreateSeededRandom(71);
        var layer = new DeformableConvolutionalLayer<double>(outputChannels: 2, kernelSize: 3, stride: 1, padding: 1);
        var input = RandomTensor(rng, 1, 2, 5, 5);
        _ = layer.Forward(input);
        var projection = RandomTensor(rng, 1, 2, 5, 5);
        AssertEveryParameterReached(layer, input, projection);
    }

    [Fact]
    public void DeformableConvolutionalLayer_OutputBiasGradientIsProjectionSumPerChannel()
    {
        // d(sum(p * (conv + b[c])))/db[c] = sum over batch and space of p[:, c].
        var rng = RandomHelper.CreateSeededRandom(72);
        var layer = new DeformableConvolutionalLayer<double>(outputChannels: 2, kernelSize: 3, stride: 1, padding: 1);
        var input = RandomTensor(rng, 1, 2, 5, 5);
        _ = layer.Forward(input);
        var projection = RandomTensor(rng, 1, 2, 5, 5);
        var (parameters, gradients) = TapeGradients(layer, input, projection);
        var bias = parameters.Single(p => p.Shape.Length == 1 && p.Shape[0] == 2);
        Assert.True(gradients.TryGetValue(bias, out var gradient) && gradient is not null,
            "the output bias received no gradient");
        for (int c = 0; c < 2; c++)
        {
            double expected = 0;
            for (int h = 0; h < 5; h++)
                for (int w = 0; w < 5; w++)
                    expected += projection[0, c, h, w];
            Assert.Equal(expected, gradient[c], 10);
        }
    }

    /// <summary>A 4-vertex path-graph Laplacian with direct (non-spectral) diffusion at two time scales.</summary>
    private static DiffusionConvLayer<double> TinyDiffusion()
    {
        var layer = new DiffusionConvLayer<double>(
            outputChannels: 2, numTimeScales: 2, numEigenvectors: 4,
            activation: (IActivationFunction<double>)new IdentityActivation<double>(), preferSpectralDiffusion: false);
        var laplacian = new Tensor<double>(new[] { 4, 4 });
        foreach (var (i, j) in new[] { (0, 1), (1, 2), (2, 3) })
        {
            laplacian[i, j] = -1.0;
            laplacian[j, i] = -1.0;
            laplacian[i, i] += 1.0;
            laplacian[j, j] += 1.0;
        }
        layer.SetLaplacian(laplacian);
        return layer;
    }

    [Fact]
    public void DiffusionConvLayer_Batched_EveryTrainableParameterReceivesGradient()
    {
        // The batched path copied each sample's output into a raw array, so the weights and biases
        // never trained on batched input.
        var rng = RandomHelper.CreateSeededRandom(81);
        var layer = TinyDiffusion();
        var input = RandomTensor(rng, 2, 4, 3);
        _ = layer.Forward(input);
        AssertEveryParameterReached(layer, input, RandomTensor(rng, 2, 4, 2));
    }

    [Fact]
    public void DiffusionConvLayer_Batched_TapeGradientMatchesFiniteDifferences()
    {
        var rng = RandomHelper.CreateSeededRandom(82);
        var layer = TinyDiffusion();
        var input = RandomTensor(rng, 2, 4, 3);
        _ = layer.Forward(input);
        AssertMatchesFiniteDifferences(layer, input, RandomTensor(rng, 2, 4, 2));
    }

    [Fact]
    public void DiffusionConvLayer_BatchedOutput_MatchesEachSampleAlone()
    {
        var engine = AiDotNetEngine.Current;
        var rng = RandomHelper.CreateSeededRandom(83);
        var layer = TinyDiffusion();
        var input = RandomTensor(rng, 2, 4, 3);
        var batched = layer.Forward(input);
        for (int b = 0; b < 2; b++)
        {
            var sample = engine.Reshape(engine.TensorSlice(input, new[] { b, 0, 0 }, new[] { 1, 4, 3 }), new[] { 4, 3 });
            var single = layer.Forward(sample);
            for (int v = 0; v < 4; v++)
                for (int o = 0; o < 2; o++)
                    Assert.Equal(single[v, o], batched[b, v, o], 12);
        }
    }

    [Fact]
    public void ALiBiPositionalBiasLayer_PassesScoreGradientThrough()
    {
        // ALiBi adds a constant bias, so d(sum(p * (x + bias)))/dx = p exactly. The layer has no trainable
        // parameters; the scores (and through them the Q/K projections upstream) must keep their gradient.
        var engine = AiDotNetEngine.Current;
        var rng = RandomHelper.CreateSeededRandom(41);
        var layer = new ALiBiPositionalBiasLayer<double>(numHeads: 2, maxSequenceLength: 8);
        var scores = RandomTensor(rng, 2, 3, 3);
        var projection = RandomTensor(rng, 2, 3, 3);
        using var tape = new GradientTape<double>();
        var watched = engine.TensorMultiplyScalar(scores, 1.0);
        var loss = Loss(layer.Forward(watched), projection);
        var gradients = tape.ComputeGradients(loss, new[] { watched });
        Assert.True(gradients.TryGetValue(watched, out var gradient) && gradient is not null,
            "the attention scores received no gradient through ALiBi");
        var expected = projection.ToArray();
        var actual = gradient.ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 12);
    }
}
