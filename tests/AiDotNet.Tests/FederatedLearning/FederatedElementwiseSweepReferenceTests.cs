using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.Interfaces;
using AiDotNet.Tests.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Reference tests for the paper fixes made while moving the federated-learning code onto engine ops.
/// Every expected value is hand-computed from the cited paper's algorithm.
/// </summary>
public class FederatedElementwiseSweepReferenceTests
{
    [Fact(Timeout = 60000)]
    public async Task Bulyan_AveragesBetaValuesClosestToMedian()
    {
        await Task.Yield();

        // El Mhamdi et al. 2018 (Bulyan), n = 7, f = 1: theta = n - 2f = 5, beta = theta - 2f = 3.
        // Multi-Krum scores (sum of squared distances to the n - f - 2 = 4 nearest):
        //   0 -> 20, 1 -> 9, 1 -> 9, 3 -> 17, 3 -> 17, 50 and 60 -> > 2000, so {0, 1, 1, 3, 3} is selected.
        // Median of the selection is 1; the beta = 3 values closest to it are {0, 1, 1}: mean 2/3.
        // (A plain trimmed mean would give (1 + 1 + 3) / 3 = 5/3.)
        var aggregator = new BulyanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(byzantineClientCount: 1);
        var models = CreateClientModels(new[] { 0.0, 1.0, 1.0, 3.0, 3.0, 50.0, 60.0 });

        var aggregated = aggregator.Aggregate(models, CreateEqualWeights(models.Keys));

        Assert.Equal(2.0 / 3.0, Parameters(aggregated)[0], 12);
    }

    private static Vector<double> Parameters(IFullModel<double, Matrix<double>, Vector<double>> model)
        => ((IParameterizable<double, Matrix<double>, Vector<double>>)model).GetParameters();

    private static Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>> CreateClientModels(double[] firstParameterValues)
    {
        var models = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>();
        for (int i = 0; i < firstParameterValues.Length; i++)
        {
            int clientId = i + 1;
            var model = new MockFullModel(_ => new Vector<double>(1), parameterCount: 1);
            model.SetParameters(new Vector<double>(new[] { firstParameterValues[i] }));
            models[clientId] = model;
        }

        return models;
    }

    private static Dictionary<int, double> CreateEqualWeights(IEnumerable<int> clientIds)
    {
        var weights = new Dictionary<int, double>();
        foreach (var id in clientIds)
        {
            weights[id] = 1.0;
        }

        return weights;
    }
}
