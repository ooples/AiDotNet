using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Reference tests for the TabR retrieval module: k-nearest selection by squared Euclidean distance and
/// softmax(-distance / temperature) attention over the selected neighbours.
/// </summary>
public class RetrievalModuleTests
{
    private const double Tolerance = 1e-12;

    private static Tensor<double> Column(params double[] values) => new Tensor<double>(values, new[] { values.Length, 1 });

    [Fact]
    public void Retrieve_WidelySpreadDistances_WeightsStayFiniteAndNormalized()
    {
        // Keys 0 and 30 seen from query 0: squared distances 0 and 900. softmax(-[0, 900]) is
        // [1 / (1 + e^-900), e^-900 / (1 + e^-900)] = [1, 0] in double. Shifting the exponent by the
        // LARGEST distance instead gives exp(900) = +inf and NaN weights.
        var module = new RetrievalModule<double>(embeddingDim: 1, numNeighbors: 2, temperature: 1.0);
        module.StoreTrainingData(Column(0.0, 30.0), Column(5.0, 7.0), Column(1.0, 2.0));

        var context = module.Retrieve(Column(0.0));

        Assert.Equal(2, context.NumNeighbors);
        Assert.Equal(1.0, context.AttentionWeights[0, 0], Tolerance);
        Assert.Equal(0.0, context.AttentionWeights[0, 1], Tolerance);
        Assert.Equal(5.0, context.Values[0, 0, 0], Tolerance);
        Assert.Equal(7.0, context.Values[0, 1, 0], Tolerance);
    }

    [Fact]
    public void Retrieve_OrdersByDistanceAndKeepsTrainingOrderOnTies()
    {
        // Query 0 against keys [1, -1, 2, 1]: squared distances [1, 1, 4, 1]. The two nearest are the
        // first two of the three tied keys, in training order (indices 0 and 1), each with weight 1/2.
        var module = new RetrievalModule<double>(embeddingDim: 1, numNeighbors: 2, temperature: 1.0);
        module.StoreTrainingData(Column(1.0, -1.0, 2.0, 1.0), Column(10.0, 20.0, 30.0, 40.0), Column(0.0, 1.0, 2.0, 3.0));

        var context = module.Retrieve(Column(0.0));

        Assert.Equal(0.0, context.Labels[0, 0, 0], Tolerance);
        Assert.Equal(1.0, context.Labels[0, 1, 0], Tolerance);
        Assert.Equal(0.5, context.AttentionWeights[0, 0], Tolerance);
        Assert.Equal(0.5, context.AttentionWeights[0, 1], Tolerance);
    }

    [Fact]
    public void Retrieve_WeightsAreSoftmaxOfNegativeDistanceOverTemperature()
    {
        // Query 0, keys [1, 2], temperature 2: distances [1, 4], logits [-0.5, -2].
        // w0 = 1 / (1 + e^-1.5), w1 = e^-1.5 / (1 + e^-1.5).
        var module = new RetrievalModule<double>(embeddingDim: 1, numNeighbors: 2, temperature: 2.0);
        module.StoreTrainingData(Column(1.0, 2.0), Column(0.0, 0.0), Column(0.0, 0.0));

        var context = module.Retrieve(Column(0.0));

        double e = System.Math.Exp(-1.5);
        Assert.Equal(1.0 / (1.0 + e), context.AttentionWeights[0, 0], Tolerance);
        Assert.Equal(e / (1.0 + e), context.AttentionWeights[0, 1], Tolerance);
    }
}
