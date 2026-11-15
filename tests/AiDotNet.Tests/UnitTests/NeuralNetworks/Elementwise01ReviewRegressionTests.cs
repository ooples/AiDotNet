#if NET5_0_OR_GREATER
using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public partial class Elementwise01NumericalRegressionTests
{
    private const BindingFlags InstanceFields = BindingFlags.NonPublic | BindingFlags.Instance;

    private static void SetField(Type type, object instance, string name, object? value)
        => (type.GetField(name, InstanceFields) ?? throw new InvalidOperationException($"Missing {name}"))
            .SetValue(instance, value);

    private static Tensor<double> Seeded(int seed, params int[] shape)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, (rng.NextDouble() * 2) - 1);
        return t;
    }

    private static Tensor<double> ProjectedLoss(Tensor<double> output, Tensor<double> projection)
    {
        var engine = AiDotNetEngine.Current;
        return engine.ReduceSum(engine.TensorMultiply(output, projection),
            Enumerable.Range(0, output.Shape.Length).ToArray(), keepDims: false);
    }

    private static void AssertReached(Dictionary<Tensor<double>, Tensor<double>> gradients, Tensor<double> source, string what)
    {
        Assert.True(gradients.TryGetValue(source, out var gradient) && gradient is not null, $"{what} received no gradient");
        Assert.True(gradient.ToArray().Max(Math.Abs) > 1e-12, $"{what} received only zero gradients");
    }

    [Fact]
    public void Blip2CrossAttention_KeepsQueryAndVisionGradientAndValue()
    {
        // ForwardForTraining runs the Q-Former through ApplyCrossAttention and ProjectVisionToQformer; both
        // were plain copies off the tape, so the query tokens, every Q-Former layer before the last
        // cross-attention and the whole vision encoder received no gradient.
        var model = NumericModel<double, Blip2NeuralNetwork<double>>();
        var type = typeof(Blip2NeuralNetwork<double>);
        SetField(type, model, "_qformerHiddenDim", 4);
        SetField(type, model, "_visionHiddenDim", 6);
        var engine = AiDotNetEngine.Current;
        var rawQueries = Seeded(101, 3, 4);
        var rawVision = Seeded(102, 5, 6);
        var projection = Seeded(103, 3, 4);
        using var unusedLayer = new DenseLayer<double>(4, (IActivationFunction<double>)new IdentityActivation<double>());

        using var tape = new GradientTape<double>();
        var queries = engine.TensorMultiplyScalar(rawQueries, 1.0);
        var vision = engine.TensorMultiplyScalar(rawVision, 1.0);
        var keyValues = Assert.IsType<Tensor<double>>(Invoke(type, model, "ProjectVisionToQformer", vision));
        var output = Assert.IsType<Tensor<double>>(Invoke(type, model, "ApplyCrossAttention", unusedLayer, queries, keyValues));
        var gradients = tape.ComputeGradients(ProjectedLoss(output, projection), new[] { queries, vision });

        AssertReached(gradients, queries, "the Q-Former queries");
        AssertReached(gradients, vision, "the vision features");

        // Column j of the projection is vision column j * 6 / 4: 0, 1, 3, 4.
        int[] columns = { 0, 1, 3, 4 };
        for (int k = 0; k < 5; k++)
            for (int j = 0; j < 4; j++)
                Assert.Equal(rawVision[k, columns[j]], keyValues[k, j]);

        // softmax(Q K^T / sqrt(4)) V + Q, in double.
        for (int q = 0; q < 3; q++)
        {
            var scores = new double[5];
            for (int k = 0; k < 5; k++)
            {
                double dot = 0;
                for (int d = 0; d < 4; d++) dot += rawQueries[q, d] * keyValues[k, d];
                scores[k] = dot / 2.0;
            }
            double max = scores.Max();
            var weights = scores.Select(s => Math.Exp(s - max)).ToArray();
            double total = weights.Sum();
            for (int d = 0; d < 4; d++)
            {
                double expected = rawQueries[q, d];
                for (int k = 0; k < 5; k++) expected += weights[k] / total * keyValues[k, d];
                Assert.Equal(expected, output[q, d], 12);
            }
        }
    }

    [Fact]
    public void FlamingoVisionAndPerceiverQueries_StayOnTheTrainingTape()
    {
        // ForwardForTraining is ExtractPerceiverFeatures: the patch + position sum and the perceiver query
        // block were copied into host buffers, so the patch embedding and the latent queries never trained.
        var model = NumericModel<double, FlamingoNeuralNetwork<double>>();
        var type = typeof(FlamingoNeuralNetwork<double>);
        using var patchEmbedding = new DenseLayer<double>(4, (IActivationFunction<double>)new IdentityActivation<double>());
        var image = Seeded(201, 2, 5);
        var positional = Seeded(202, 3, 4);
        var expectedPatches = patchEmbedding.Forward(image);
        SetField(type, model, "_patchEmbedding", patchEmbedding);
        SetField(type, model, "_visionPositionalEmbeddings", positional);
        SetField(type, model, "_visionEncoderLayers", new List<ILayer<double>>());
        SetField(type, model, "_perceiverLayers", new List<ILayer<double>>());
        SetField(type, model, "_visionHiddenDim", 4);
        SetField(type, model, "_numPerceiverTokens", 2);
        SetField(type, model, "_lmHiddenDim", 3);
        var engine = AiDotNetEngine.Current;
        var rawQueries = Seeded(203, 3, 3);

        using var tape = new GradientTape<double>();
        var features = Assert.IsType<Tensor<double>>(Invoke(type, model, "ExtractVisionFeaturesNative", image));
        var queries = engine.TensorMultiplyScalar(rawQueries, 1.0);
        SetField(type, model, "_perceiverQueries", queries);
        var perceiver = Assert.IsType<Tensor<double>>(Invoke(type, model, "ExtractPerceiverFeaturesNative", features));
        var loss = engine.TensorAdd(
            ProjectedLoss(features, Seeded(204, 2, 4)), ProjectedLoss(perceiver, Seeded(205, 2, 3)));
        var parameters = TapeTrainingStep<double>.CollectParameters(new ILayer<double>[] { patchEmbedding }, structureVersion: -1);
        var gradients = tape.ComputeGradients(loss, parameters.Concat(new[] { queries }).ToArray());

        Assert.NotEmpty(parameters);
        for (int i = 0; i < parameters.Count; i++) AssertReached(gradients, parameters[i], $"patch embedding parameter {i}");
        AssertReached(gradients, queries, "the perceiver queries");

        Assert.Equal(new[] { 2, 4 }, features.Shape.ToArray());
        for (int p = 0; p < 2; p++)
            for (int c = 0; c < 4; c++)
                Assert.Equal(expectedPatches[p, c] + positional[p, c], features[p, c], 12);
        Assert.Equal(new[] { 2, 3 }, perceiver.Shape.ToArray());
        for (int t = 0; t < 2; t++)
            for (int c = 0; c < 3; c++)
                Assert.Equal(rawQueries[t, c], perceiver[t, c]);
    }

    [Fact]
    public void HalfGraphAdjacencyNormalization_CountsDegreesPast2048()
    {
        // The degree was summed in T; a Half sum stops at 2048, so a 2100-neighbour row scaled by 1/2048.
        const int nodes = 2100;
        var ops = MathHelper.GetNumericOperations<Half>();
        Half expected = ops.Divide(ops.One, ops.FromDouble(nodes));
        var adjacency = new Tensor<Half>(new[] { nodes, nodes });
        adjacency.Fill((Half)1);

        using var edgeConditional = new EdgeConditionalConvolutionalLayer<Half>(
            inputFeatures: 3, outputFeatures: 2, edgeFeatures: 2, edgeNetworkHiddenDim: 4);
        var metadata = new HeterogeneousGraphMetadata
        {
            NodeTypes = new[] { "a" },
            EdgeTypes = new[] { "aa" },
            NodeTypeFeatures = new Dictionary<string, int> { ["a"] = 3 },
            EdgeTypeSchema = new Dictionary<string, (string SourceType, string TargetType)> { ["aa"] = ("a", "a") },
        };
        using var heterogeneous = new HeterogeneousGraphLayer<Half>(metadata, outputFeatures: 2);

        foreach (var (type, layer) in new (Type, object)[]
        {
            (typeof(EdgeConditionalConvolutionalLayer<Half>), edgeConditional),
            (typeof(HeterogeneousGraphLayer<Half>), heterogeneous),
        })
        {
            var normalized = Assert.IsType<Tensor<Half>>(Invoke(type, layer, "NormalizeAdjacency", adjacency, 1, nodes));
            Assert.Equal(new[] { 1, nodes, nodes }, normalized.Shape.ToArray());
            foreach (int row in new[] { 0, nodes / 2, nodes - 1 })
                for (int column = 0; column < nodes; column++)
                    Assert.Equal(expected, normalized[0, row, column]);
        }
    }

    [Fact]
    public void HalfGraphTransformerLayerNorm_DoesNotOverflowTheVariance()
    {
        // With every projection zeroed the output is LN2(LN1(r)), r the first three input features.
        // Engine.LayerNorm sums (x - mean)^2 in Half: a centred 400 squares past 65504 and the row came
        // out as zeros.
        using var layer = new GraphTransformerLayer<Half>(
            inputFeatures: 4, outputFeatures: 3, numHeads: 2, headDim: 3, useStructuralEncoding: false, dropoutRate: 0.0,
            activationFunction: null);
        var adjacency = new Tensor<Half>(new[] { 2, 2 });
        adjacency.Fill((Half)1);
        layer.SetAdjacencyMatrix(adjacency);
        foreach (var field in typeof(GraphTransformerLayer<Half>).GetFields(InstanceFields))
        {
            if (field.FieldType == typeof(Tensor<Half>) && field.GetValue(layer) is Tensor<Half> tensor
                && field.Name != "_adjacencyMatrix" && !field.Name.StartsWith("_layerNorm", StringComparison.Ordinal))
                tensor.Fill((Half)0);
        }

        double[][] rows = { new[] { 300.0, -300.0, 300.0, 7.0 }, new[] { -1000.0, 2000.0, 500.0, -3.0 } };
        var input = new Tensor<Half>(new[] { 2, 4 });
        for (int n = 0; n < 2; n++)
            for (int f = 0; f < 4; f++)
                input[n, f] = (Half)rows[n][f];

        var output = layer.Forward(input);

        static double[] Normalize(double[] x)
        {
            double mean = x.Average();
            double variance = x.Select(v => (v - mean) * (v - mean)).Average();
            return x.Select(v => (v - mean) / Math.Sqrt(variance + 1e-5)).ToArray();
        }
        for (int n = 0; n < 2; n++)
        {
            var expected = Normalize(Normalize(rows[n].Take(3).ToArray()));
            for (int f = 0; f < 3; f++)
                Assert.InRange((double)output[n, f], expected[f] - 0.01, expected[f] + 0.01);
        }
    }
}
#endif
