using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FedBNFullModelAggregationStrategyTests
{
    [Fact]
    public void Aggregate_LeavesBatchNormParametersFromFirstClient()
    {
        var model1 = CreateNetwork();
        var model2 = CreateNetwork();

        var parameters1 = CreateParameters(model1.ParameterCount, offset: 0.0);
        var parameters2 = CreateParameters(model2.ParameterCount, offset: 1000.0);
        model1.SetParameters(parameters1);
        model2.SetParameters(parameters2);

        var clientModels = new Dictionary<int, IFullModel<double, Tensor<double>, Tensor<double>>>
        {
            [1] = model1,
            [2] = model2
        };
        var clientWeights = new Dictionary<int, double>
        {
            [1] = 1.0,
            [2] = 1.0
        };

        var aggregator = new FedBNFullModelAggregationStrategy<double, Tensor<double>, Tensor<double>>();
        var aggregatedModel = aggregator.Aggregate(clientModels, clientWeights);

        var aggregatedParameters = aggregatedModel.GetParameters();
        var bnRanges = GetBatchNormParameterRanges(model1);

        for (int i = 0; i < aggregatedParameters.Length; i++)
        {
            bool isBatchNorm = bnRanges.Any(r => i >= r.Start && i < r.Start + r.Length);
            double expected = isBatchNorm ? parameters1[i] : (parameters1[i] + parameters2[i]) / 2.0;
            Assert.Equal(expected, aggregatedParameters[i], precision: 10);
        }

        Assert.Equal("FedBN", aggregator.GetStrategyName());
    }

    private static NeuralNetwork<double> CreateNetwork()
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(inputSize: 4, outputSize: 4, activationFunction: null),
            new BatchNormalizationLayer<double>(featureSize: 4),
            new DenseLayer<double>(inputSize: 4, outputSize: 2, activationFunction: null)
        };

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2,
            layers: layers);

        return new NeuralNetwork<double>(architecture);
    }

    private static Vector<double> CreateParameters(int count, double offset)
    {
        var values = new double[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = offset + i;
        }

        return new Vector<double>(values);
    }

    private static List<(int Start, int Length)> GetBatchNormParameterRanges(NeuralNetworkBase<double> network)
    {
        var ranges = new List<(int Start, int Length)>();

        int current = 0;
        foreach (var layer in network.Layers)
        {
            int count = layer.ParameterCount;
            if (count <= 0)
            {
                continue;
            }

            if (layer is BatchNormalizationLayer<double>)
            {
                ranges.Add((current, count));
            }

            current += count;
        }

        return ranges;
    }
}
