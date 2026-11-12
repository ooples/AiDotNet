#if NET5_0_OR_GREATER
using AiDotNet.Evaluation;
using AiDotNet.Evaluation.Metrics.Classification;
using AiDotNet.Evaluation.Metrics.Probabilistic;
using AiDotNet.Evaluation.Metrics.Regression;
using AiDotNet.Evaluation.Statistics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Evaluation;

public class EvaluationNumericalMemoryTests
{
    [Fact]
    public void DatasetRanks_PreserveIndependentTiesAndDatasetOrder()
    {
        double[][] samples = { new[] { 3.0, 1, 2 }, new[] { 2.0, 3, 2 }, new[] { 1.0, 2, 3 } };
        var result = new NemenyiPostHocTest<double>().Test(samples);
        Assert.Equal(6.5 / 3, result.AverageRanks[0], 12);
        Assert.Equal(5.5 / 3, result.AverageRanks[1], 12);
        Assert.Equal(2.0, result.AverageRanks[2], 12);
        Assert.Equal(2.0 / 35, new FriedmanTest<double>().Test(samples).Statistic, 12);
    }

    [Fact]
    public void EnergyScore_LargeEnsembleMatchesTwoPointDistribution()
    {
        const int members = 513, dimensions = 513;
        var samples = Enumerable.Range(0, members)
            .Select(i => Enumerable.Repeat((double)(i % 2), dimensions).ToArray()).ToArray();
        var observation = Enumerable.Repeat(0.5, dimensions).ToArray();
        double distance = Math.Sqrt(dimensions);
        double expected = 0.5 * distance - (double)(members / 2) * (members - members / 2) / (members * members) * distance;
        Assert.Equal(expected, new EnergyScoreMetric<double>().ComputeFromEnsemble(samples, observation), 10);
    }

    [Fact]
    public void EnergyScore_ArbitraryVectorsMatchScalarPairOracle()
    {
        double[][] samples = { new[] { 1.0, -2, 3 }, new[] { 4.0, 0, -1 }, new[] { -2.0, 5, 2 } };
        double[] observation = { 0.5, 1, -3 };
        static double Distance(double[] a, double[] b)
        {
            double sum = 0;
            for (int d = 0; d < a.Length; d++) sum += (a[d] - b[d]) * (a[d] - b[d]);
            return Math.Sqrt(sum);
        }
        double expected = samples.Sum(s => Distance(s, observation)) / samples.Length;
        foreach (var left in samples)
        foreach (var right in samples) expected -= 0.5 * Distance(left, right) / (samples.Length * samples.Length);
        var metric = new EnergyScoreMetric<double>();
        Assert.Equal(expected, metric.ComputeFromEnsemble(samples, observation), 12);
        Assert.Equal(0.0, metric.ComputeFromEnsemble(new[] { Array.Empty<double>() }, Array.Empty<double>()));
        Assert.Equal(Distance(samples[0], observation), metric.ComputeFromEnsemble(new[] { samples[0] }, observation), 12);
    }

    [Fact]
    public void HalfEnergyScore_KeepsRepresentableResultAfterWideDistances()
    {
        Half[][] samples = { new[] { (Half)(-60000) }, new[] { (Half)60000 } };
        Assert.Equal((Half)30000, new EnergyScoreMetric<Half>().ComputeFromEnsemble(samples, new[] { (Half)0 }));
    }

    [Fact]
    public void Pearson_WideMomentsPreserveFiniteCorrelations()
    {
        var metric = new PearsonCorrelationMetric<float>();
        float[] values = { -1e20f, 1e20f };
        Assert.Equal(1f, metric.Compute(values, values));
        Assert.Equal(-1f, metric.Compute(values, new[] { 1e20f, -1e20f }));
        Half[] half = { (Half)(-1000), (Half)1000 };
        Assert.Equal((Half)1, new PearsonCorrelationMetric<Half>().Compute(half, half));
        Assert.Equal((Half)0, new PearsonCorrelationMetric<Half>().Compute(new[] { (Half)1, (Half)1 }, half));
    }

    [Fact]
    public void HalfAucAndAveragePrecision_KeepLargeCountsWide()
    {
        var scores = Enumerable.Repeat((Half)0.5, 70000).ToArray();
        var labels = Enumerable.Range(0, scores.Length).Select(i => (Half)(i % 2)).ToArray();
        Assert.Equal((Half)0.5, new AUCROCMetric<Half>().Compute(scores, labels));
        Assert.Equal((Half)0.5, new AUCPRMetric<Half>().Compute(scores, labels));
        scores = Enumerable.Range(0, 514).Select(i => (Half)(i / 257)).ToArray();
        Assert.Equal((Half)1, new AUCROCMetric<Half>().Compute(scores, scores));
    }

    [Fact]
    public void HalfMidRanks_DoNotOverflowBeforeHalving()
    {
        const int size = 65504, group = size / 2;
        var values = new Tensor<Half>(Enumerable.Range(0, size).Select(i => (Half)(i / group)).ToArray(), new[] { size });
        var ranks = EvaluationTensorOps<Half>.MidRanks(values);
        Assert.Equal((Half)((group + 1.0) / 2), ranks.GetFlat(0));
        Assert.Equal((Half)((group + 1.0 + size) / 2), ranks.GetFlat(size - 1));
    }

    [Fact]
    public void HalfRunRanks_PreserveTieRunsBeyond2048Groups()
    {
        const int size = 4000;
        var values = Enumerable.Range(0, size).Select(i => (Half)(i / 4.0)).ToArray();
        var tensor = new Tensor<Half>(values, new[] { size });
        var (_, _, ranks) = EvaluationTensorOps<Half>.SortedRunRanks(tensor, 1e-10);
        var rows = new Tensor<Half>(values.Concat(values.Reverse()).ToArray(), new[] { 2, size });
        var descending = EvaluationTensorOps<Half>.RowRanksDescending(rows, 1e-10);
        for (int start = 0; start < size;)
        {
            int end = start + 1;
            while (end < size && values[end] == values[start]) end++;
            Half ascendingRank = (Half)((start + 1.0 + end) / 2);
            Half descendingRank = (Half)((size - end + 1.0 + size - start) / 2);
            for (int index = start; index < end; index++)
            {
                Assert.Equal(ascendingRank, ranks.GetFlat(index));
                Assert.Equal(descendingRank, descending[0, index]);
                Assert.Equal(descendingRank, descending[1, size - 1 - index]);
            }
            start = end;
        }
    }

    [Fact]
    public void HalfOneHot_UsesExactIntegerClassIndices()
    {
        var result = EvaluationTensorOps<Half>.RoundedOneHot(new Tensor<Half>(new[] { (Half)2052 }, new[] { 1 }), 2053);
        Assert.Equal((Half)1, result[0, 2052]);
        Assert.Equal(1, result.ToArray().Count(value => value != (Half)0));
    }

    [Fact]
    public void DecimalBestIndex_DoesNotConstructNaNForFiniteValues()
    {
        var values = new Tensor<decimal>(new[] { 3m, 9m, 9m, -2m }, new[] { 4 });
        Assert.Equal(1, EvaluationTensorOps<decimal>.CompareScanBestIndex(values, true));
        Assert.Equal(3, EvaluationTensorOps<decimal>.CompareScanBestIndex(values, false));
    }
}
#endif
