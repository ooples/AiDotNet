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

    [Fact(Timeout = 60000)]
    public async Task DivideAndConquer_RemovesCTimesMLowestRankedAtOnce()
    {
        await Task.Yield();

        // Shejwalkar & Houmansadr 2021, Algorithm 2 with niters = 1, c = 1, m = 2 on 1-D updates
        // {0, 1, 2, 3, 20, 40}: mu = 11, v = +-1, scores (x - mu)^2 = {121, 100, 81, 64, 81, 841}.
        // The n - c*m = 4 lowest scores keep {1, 2, 3, 20}; their mean is 6.5.
        // (Removing one outlier at a time and re-centring would instead drop 40 then 20 and return 1.5.)
        var aggregator = new DivideAndConquerAggregationStrategy<double>(numByzantine: 2);
        var values = new[] { 0.0, 1.0, 2.0, 3.0, 20.0, 40.0 };
        var models = new Dictionary<int, Dictionary<string, double[]>>();
        var weights = new Dictionary<int, double>();
        for (int i = 0; i < values.Length; i++)
        {
            models[i] = new Dictionary<string, double[]> { ["w"] = new[] { values[i] } };
            weights[i] = 1.0;
        }

        var aggregated = aggregator.Aggregate(models, weights);

        Assert.Equal(6.5, aggregated["w"][0], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task FedDecorr_LossIsBetaOverDSquaredTimesCorrelationFrobenius()
    {
        await Task.Yield();

        // Shi et al. ICLR 2023: L = beta * (1 / d^2) * ||K||_F^2. Two perfectly correlated columns give
        // K = [[1, 1], [1, 1]], ||K||_F^2 = 4, so with beta = 0.1 and d = 2 the loss is 0.1 * 4 / 4 = 0.1.
        var strategy = new FedDecorrAggregationStrategy<double>(decorrelationWeight: 0.1);
        var features = new Matrix<double>(new[,] { { 1.0, 2.0 }, { 2.0, 4.0 }, { 3.0, 6.0 } });

        Assert.Equal(0.1, strategy.ComputeDecorrelationLoss(features), 10);
    }

    [Fact(Timeout = 60000)]
    public async Task FedLc_SubtractsTauTimesCountToMinusQuarter()
    {
        await Task.Yield();

        // Zhang et al. ICML 2022 (FedLC), eq. 9: z_cal[c] = z[c] - tau * n[c]^(-1/4).
        // Counts {16, 1} with tau = 1: margins {0.5, 1}; logits {2, 2} -> {1.5, 1}.
        var strategy = new FedLcAggregationStrategy<double>(calibrationTemperature: 1.0);
        var counts = FedLcAggregationStrategy<double>.ComputeClassCounts(
            new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, 2);
        strategy.RegisterClassCounts(7, counts);

        var calibrated = strategy.CalibrateLogits(new Vector<double>(new[] { 2.0, 2.0 }), 7);

        Assert.Equal(1.5, calibrated[0], 12);
        Assert.Equal(1.0, calibrated[1], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task FedAdam_FirstStepHasNoBiasCorrection()
    {
        await Task.Yield();

        // Reddi et al. ICLR 2021, Algorithm 2 (FedAdam), defaults beta1 = 0.9, beta2 = 0.99, tau = 1e-3:
        // delta = 1 -> m = 0.1, v = 0.01, x = 0 + 1 * 0.1 / (0.1 + 0.001) = 0.990099...
        var optimizer = new AiDotNet.FederatedLearning.ServerOptimizers.FedAdamServerOptimizer<double>();

        var updated = optimizer.Step(new Vector<double>(new[] { 0.0 }), new Vector<double>(new[] { 1.0 }));

        Assert.Equal(0.1 / 0.101, updated[0], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task FedAdagrad_UsesFirstMomentOverAccumulatedSquares()
    {
        await Task.Yield();

        // Reddi et al. ICLR 2021, Algorithm 2 (FedAdagrad), beta1 = 0.9, tau = 1e-3:
        // delta = 1 -> m = 0.1, v = 1, x = 0 + 1 * 0.1 / (1 + 0.001).
        var optimizer = new AiDotNet.FederatedLearning.ServerOptimizers.FedAdagradServerOptimizer<double>();

        var updated = optimizer.Step(new Vector<double>(new[] { 0.0 }), new Vector<double>(new[] { 1.0 }));

        Assert.Equal(0.1 / 1.001, updated[0], 12);
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
