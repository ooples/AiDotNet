using System;
using System.Linq;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

public class DistillationNumericalMemoryTests
{
    [Theory]
    [InlineData(0.5, 60000)]
    [InlineData(2.0, 60000)]
    [InlineData(400.0, 1)]
    [InlineData(1e-9, 1)]
    public void HalfLossAndGradient_KeepTemperatureAndWorkingProbabilitiesWide(double temperature, double magnitude)
    {
        var student = new Matrix<double>(new[,] { { magnitude, -magnitude } });
        var teacher = new Matrix<double>(new[,] { { -magnitude, magnitude } });
        var labels = new Matrix<double>(new[,] { { 0.0, 1.0 } });
        var wide = new DistillationLoss<double>(temperature);
        var narrow = new DistillationLoss<Half>(temperature);
        var s = Tensor<double>.FromMatrix(student).Cast<Half>().ToMatrix();
        var t = Tensor<double>.FromMatrix(teacher).Cast<Half>().ToMatrix();
        var y = Tensor<double>.FromMatrix(labels).Cast<Half>().ToMatrix();
        double expected = wide.ComputeLoss(student, teacher, labels);
        double actual = (double)narrow.ComputeLoss(s, t, y);
        Assert.True(!double.IsNaN(actual) && !double.IsInfinity(actual));
        Assert.True(Math.Abs(expected - actual) <= 0.002 * Math.Max(1, Math.Abs(expected)));
        var expectedGradient = wide.ComputeGradient(student, teacher, labels);
        var actualGradient = narrow.ComputeGradient(s, t, y);
        for (int c = 0; c < 2; c++)
            Assert.True(Math.Abs(expectedGradient[0, c] - (double)actualGradient[0, c]) <= 0.002 * Math.Max(1, Math.Abs(expectedGradient[0, c])));
    }

    [Fact]
    public void HalfLoss_LargeBatchAveragesBeforeNarrowing()
    {
        const int count = 70000;
        var logits = new Matrix<Half>(count, 2);
        var labels = new Matrix<Half>(count, 2);
        for (int i = 0; i < count; i++) labels[i, 0] = (Half)1;
        var loss = new DistillationLoss<Half>(alpha: 1).ComputeLoss(logits, logits, labels);
        Assert.True(Math.Abs((double)loss - Math.Log(2)) < 0.001);
    }

    [Fact]
    public void HalfProbabilityHelpers_KeepPositiveFloorsForZeroProbabilities()
    {
        var p = new Vector<Half>(new Half[] { (Half)0, (Half)1 });
        var q = new Vector<Half>(new Half[] { (Half)1, (Half)0 });
        double expected = -Math.Log(1e-10);
        Assert.True(Math.Abs((double)DistillationHelper<Half>.CrossEntropy(p, q) - expected) < 0.02);
        Assert.True(Math.Abs((double)DistillationHelper<Half>.KLDivergence(p, q) - expected) < 0.02);
        var pm = new Matrix<Half>(new[,] { { (Half)0, (Half)1 } });
        var qm = new Matrix<Half>(new[,] { { (Half)1, (Half)0 } });
        Assert.True(Math.Abs((double)DistillationHelper<Half>.CrossEntropyRows(pm, qm)[0] - expected) < 0.02);
        Assert.True(Math.Abs((double)DistillationHelper<Half>.KLDivergenceRows(pm, qm)[0] - expected) < 0.02);
    }

    [Theory]
    [InlineData(2049)]
    [InlineData(70001)]
    public void HalfArgMax_UsesExactIntegerPositions(int index)
    {
        var values = new Vector<Half>(index + 3);
        values[index] = (Half)2;
        values[index + 1] = (Half)2;
        Assert.Equal(index, DistillationHelper<Half>.ArgMax(values));
        values[0] = Half.NaN;
        Assert.Equal(0, DistillationHelper<Half>.ArgMax(values));
    }

    [Theory]
    [InlineData(FactorMode.LowRankApproximation)]
    [InlineData(FactorMode.NuclearNormMatching)]
    [InlineData(FactorMode.FactorMatching)]
    public void HalfFactorTransfer_KeepsCovarianceAndEigenvectorsWide(FactorMode mode)
    {
        var s = new[] { new Vector<double>(new[] { 1000.0, 0 }), new Vector<double>(new[] { -1000.0, 0 }),
            new Vector<double>(new[] { 0.0, 500 }), new Vector<double>(new[] { 0.0, -500 }) };
        var t = new[] { new Vector<double>(new[] { 1010.0, 0 }), new Vector<double>(new[] { -1010.0, 0 }),
            new Vector<double>(new[] { 0.0, 505 }), new Vector<double>(new[] { 0.0, -505 }) };
        var sh = s.Select(x => Tensor<double>.FromVector(x).Cast<Half>().ToVector()).ToArray();
        var th = t.Select(x => Tensor<double>.FromVector(x).Cast<Half>().ToVector()).ToArray();
        double expected = new FactorTransferDistillationStrategy<double>(mode: mode, numFactors: 2, seed: 7).ComputeFactorLoss(s, t);
        double actual = (double)new FactorTransferDistillationStrategy<Half>(mode: mode, numFactors: 2, seed: 7).ComputeFactorLoss(sh, th);
        Assert.True(!double.IsNaN(actual) && !double.IsInfinity(actual));
        Assert.True(Math.Abs(actual - expected) <= 0.002 * Math.Max(1, Math.Abs(expected)));
    }

    private static Vector<double>[] Rows(int count, int dimensions, double phase, bool normalize = true, double offset = 0)
    {
        return Enumerable.Range(0, count).Select(i =>
        {
            var values = Enumerable.Range(0, dimensions).Select(c => 2.0 + Math.Sin(0.17 * i + 0.071 * c + phase + 0.003 * i * c)).ToArray();
            double sum = normalize ? values.Sum() : 1;
            return new Vector<double>(values.Select(v => v / sum + offset).ToArray());
        }).ToArray();
    }

    private static double KernelAverage(Vector<double>[] a, Vector<double>[] b, double bandwidth, bool sameSet)
    {
        double sum = 0;
        long count = 0;
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < b.Length; j++)
            {
                if (sameSet && i == j) continue;
                double squared = 0;
                for (int c = 0; c < a[i].Length; c++)
                {
                    double difference = a[i][c] - b[j][c];
                    squared += difference * difference;
                }
                sum += Math.Exp(-squared / (2 * bandwidth * bandwidth));
                count++;
            }
        return count == 0 ? 0 : sum / count;
    }

    private sealed class BoundedPairEngine : CpuEngine
    {
        internal int PairOperations;
        public override Tensor<T> TensorSubtract<T>(Tensor<T> a, Tensor<T> b)
        {
            if (a.Shape.Length == 3 && b.Shape.Length == 3)
            {
                PairOperations++;
                Assert.InRange(Math.Max(a.Shape[0], b.Shape[0]), 1, 32);
                Assert.InRange(Math.Max(a.Shape[1], b.Shape[1]), 1, 32);
                Assert.InRange(Math.Max(a.Shape[2], b.Shape[2]), 1, 256);
            }
            return base.TensorSubtract(a, b);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(33)]
    [InlineData(67)]
    public void Mmd_UsesBoundedBlocksAndMatchesScalarReference(int count)
    {
        const double bandwidth = 0.05;
        var s = Rows(count, 513, 0.2);
        var t = Rows(count, 513, 1.8);
        double expected = KernelAverage(s, s, bandwidth, true) + KernelAverage(t, t, bandwidth, true) - 2 * KernelAverage(s, t, bandwidth, false);
        var previous = AiDotNetEngine.Current;
        var engine = new BoundedPairEngine();
        try
        {
            AiDotNetEngine.Current = engine;
            var strategy = new ProbabilisticDistillationStrategy<double>(distributionWeight: 1, mode: ProbabilisticMode.MaximumMeanDiscrepancy, mmdBandwidth: bandwidth);
            Assert.Equal(expected, strategy.ComputeDistributionalLoss(s, t), 10);
            Assert.True(engine.PairOperations > 0);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void Mmd_GradientMatchesFiniteDifferencesAcrossBlockBoundaries()
    {
        var s = DistillationHelper<double>.StackRows(Rows(67, 513, 0.2, false)).ToMatrix();
        var t = DistillationHelper<double>.StackRows(Rows(67, 513, 1.8, false)).ToMatrix();
        var strategy = new ProbabilisticDistillationStrategy<double>(mode: ProbabilisticMode.MaximumMeanDiscrepancy, mmdBandwidth: 0.05, temperature: 1);
        var previous = AiDotNetEngine.Current;
        var engine = new BoundedPairEngine();
        try
        {
            AiDotNetEngine.Current = engine;
            var gradient = strategy.ComputeGradient(s, t);
            foreach (var (row, column) in new[] { (0, 0), (32, 256), (66, 512) })
            {
                const double step = 1e-4;
                var plus = s.Clone();
                var minus = s.Clone();
                plus[row, column] += step;
                minus[row, column] -= step;
                double expected = (strategy.ComputeLoss(plus, t) - strategy.ComputeLoss(minus, t)) / (2 * step);
                Assert.True(Math.Abs(expected - gradient[row, column]) < 1e-8);
            }
            Assert.True(engine.PairOperations > 0);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Theory]
    [InlineData(RelationalDistanceMetric.Euclidean)]
    [InlineData(RelationalDistanceMetric.Manhattan)]
    public void RelationalDistances_UseBoundedDirectDifferences(RelationalDistanceMetric metric)
    {
        const int count = 67;
        var s = Rows(count, 513, 0.2, false, 1e12);
        var t = Rows(count, 513, 1.8, false, 1e12);
        double loss = 0;
        for (int i = 0; i < count; i++)
            for (int j = i + 1; j < count; j++)
            {
                double ds = 0, dt = 0;
                for (int c = 0; c < 513; c++)
                {
                    double x = s[i][c] - s[j][c], y = t[i][c] - t[j][c];
                    ds += metric == RelationalDistanceMetric.Manhattan ? Math.Abs(x) : x * x;
                    dt += metric == RelationalDistanceMetric.Manhattan ? Math.Abs(y) : y * y;
                }
                if (metric == RelationalDistanceMetric.Euclidean) { ds = Math.Sqrt(ds); dt = Math.Sqrt(dt); }
                double difference = Math.Abs(ds - dt);
                loss += difference < 1 ? difference * difference : 2 * difference - 1;
            }
        loss /= count * (count - 1) / 2.0;
        var previous = AiDotNetEngine.Current;
        var engine = new BoundedPairEngine();
        try
        {
            AiDotNetEngine.Current = engine;
            var strategy = new RelationalDistillationStrategy<double>(angleWeight: 0, maxSamplesPerBatch: count, distanceMetric: metric);
            Assert.Equal(loss, strategy.ComputeRelationalLoss(s, t), 8);
            var sm = DistillationHelper<double>.StackRows(Rows(count, 513, 0.2, false)).ToMatrix();
            var tm = DistillationHelper<double>.StackRows(Rows(count, 513, 1.8, false)).ToMatrix();
            Assert.All(strategy.ComputeGradient(sm, tm).ToRowVector().ToArray(), x => Assert.True(!double.IsNaN(x) && !double.IsInfinity(x)));
            Assert.True(engine.PairOperations > 0);
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
