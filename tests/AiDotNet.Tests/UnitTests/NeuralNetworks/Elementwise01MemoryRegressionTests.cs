#if NET5_0_OR_GREATER
using System.Reflection;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public partial class Elementwise01NumericalRegressionTests
{
    [Fact]
    public void GraphSmoothness_DenseWideGraphMatchesKnownAverage()
    {
        const int nodes = 257, features = 1024;
        var model = NumericModel<double, GraphNeuralNetwork<double>>();
        model.UseAuxiliaryLoss = true;
        var adjacency = new Tensor<double>(new[] { nodes, nodes });
        adjacency.Fill(1.0);
        var representations = new Tensor<double>(new[] { nodes, features });
        for (int node = 0; node < nodes; node++)
            representations.AsWritableSpan().Slice(node * features, features).Fill(node % 2);
        var flags = BindingFlags.NonPublic | BindingFlags.Instance;
        (typeof(GraphNeuralNetwork<double>).GetField("_lastNodeRepresentations", flags)
            ?? throw new InvalidOperationException("Missing graph representations")).SetValue(model, representations);
        (typeof(GraphNeuralNetwork<double>).GetField("_lastAdjacencyMatrix", flags)
            ?? throw new InvalidOperationException("Missing adjacency")).SetValue(model, adjacency);

        // Opposite-parity pairs differ by 1 in every feature; within-parity pairs have zero distance.
        double expected = 2.0 * (nodes / 2) * (nodes - nodes / 2) * features / (nodes * nodes);
        Assert.Equal(expected, model.ComputeAuxiliaryLoss(), 10);
        adjacency.Fill(0.0);
        Assert.Equal(0.0, model.ComputeAuxiliaryLoss());
    }

    [Theory]
    [InlineData(7, 3)]
    [InlineData(13, 5)]
    [InlineData(65536, 32768)]
    public void LstmGradientBuckets_MatchScalarPartitionMeans(int hidden, int output)
    {
        var model = NumericModel<double, LSTMNeuralNetwork<double>>();
        var gradient = new Tensor<double>(new[] { 2, hidden });
        for (int i = 0; i < gradient.Length; i++) gradient.SetFlat(i, (i % 17) - 8);
        var result = Assert.IsType<Tensor<double>>(Invoke(typeof(LSTMNeuralNetwork<double>), model,
            "TransformHiddenGradientToOutputGradient", gradient, new[] { 2, output }));

        for (int row = 0; row < 2; row++)
        for (int column = 0; column < output; column++)
        {
            int start = (int)((long)column * hidden / output);
            int end = (int)((column + 1L) * hidden / output);
            double sum = 0;
            for (int feature = start; feature < end; feature++) sum += gradient.GetFlat(row * hidden + feature);
            Assert.Equal(sum / (end - start), result.GetFlat(row * output + column), 12);
        }
    }

    [Fact]
    public void ColbertMaxSim_LargeTokenCountsIncludeMatchesInTheLastBlock()
    {
        var model = NumericModel<double, ColBERT<double>>();
        var queries = new Matrix<double>(1025, 1);
        queries.AsWritableSpan().Fill(1.0);
        var documents = new Matrix<double>(65537, 1);
        documents[65536, 0] = 2.0;

        Assert.Equal(2050.0, model.LateInteractionScore(queries, documents), 10);
    }

    [Fact]
    public void ColbertMaxSim_PreservesNaNAndEmptyDocumentScanSemantics()
    {
        var model = NumericModel<double, ColBERT<double>>();
        var queries = new Matrix<double>(new[,] { { 1.0, 0.0 }, { -1.0, 0.0 } });
        var documents = new Matrix<double>(new[,] { { double.NaN, 0.0 }, { 1.0, 0.0 }, { 1.0, 0.0 }, { -1.0, 0.0 } });

        Assert.Equal(2.0, model.LateInteractionScore(queries, documents));
        Assert.True(double.IsNegativeInfinity(model.LateInteractionScore(queries, new Matrix<double>(0, 2))));
        Assert.Equal(0.0, model.LateInteractionScore(new Matrix<double>(0, 2), documents));
        Assert.Equal(0.0, model.LateInteractionScore(new Matrix<double>(2, 0), new Matrix<double>(3, 0)));
    }
}
#endif
