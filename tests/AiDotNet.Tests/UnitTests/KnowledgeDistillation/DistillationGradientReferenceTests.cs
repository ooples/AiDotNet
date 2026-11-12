using System;
using System.Threading.Tasks;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

/// <summary>
/// Reference tests: every analytical distillation gradient must match a central finite difference of
/// the loss it claims to differentiate.
/// </summary>
public class DistillationGradientReferenceTests
{
    private const double Step = 1e-5;

    private static Matrix<double> Deterministic(int rows, int cols, double seed, double scale = 1.0, double offset = 0.0)
    {
        var m = new Matrix<double>(rows, cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                m[r, c] = offset + scale * Math.Sin(1.37 * r + 0.71 * c + seed + 0.13 * r * c);
            }
        }
        return m;
    }

    private static Matrix<double> FiniteDifference(Matrix<double> point, Func<Matrix<double>, double> loss)
    {
        var gradient = new Matrix<double>(point.Rows, point.Columns);
        for (int r = 0; r < point.Rows; r++)
        {
            for (int c = 0; c < point.Columns; c++)
            {
                var plus = point.Clone();
                var minus = point.Clone();
                plus[r, c] += Step;
                minus[r, c] -= Step;
                gradient[r, c] = (loss(plus) - loss(minus)) / (2 * Step);
            }
        }
        return gradient;
    }

    private static void AssertClose(Matrix<double> expected, Matrix<double> actual)
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int r = 0; r < expected.Rows; r++)
        {
            for (int c = 0; c < expected.Columns; c++)
            {
                double tolerance = 1e-6 + 1e-4 * Math.Abs(expected[r, c]);
                Assert.True(
                    Math.Abs(expected[r, c] - actual[r, c]) <= tolerance,
                    $"[{r},{c}] finite difference {expected[r, c]:G10} vs analytical {actual[r, c]:G10}");
            }
        }
    }

    [Theory(Timeout = 60000)]
    [InlineData(RelationalDistanceMetric.Euclidean, 32)]
    [InlineData(RelationalDistanceMetric.Cosine, 32)]
    [InlineData(RelationalDistanceMetric.Manhattan, 32)]
    [InlineData(RelationalDistanceMetric.Euclidean, 4)]
    public async Task Relational_ComputeGradient_MatchesFiniteDifferenceOfComputeLoss(RelationalDistanceMetric metric, int maxSamples)
    {
        await Task.Yield();
        var strategy = new RelationalDistillationStrategy<double>(
            distanceWeight: 1.0, angleWeight: 2.0, temperature: 2.0, alpha: 0.3,
            maxSamplesPerBatch: maxSamples, distanceMetric: metric);
        var student = Deterministic(5, 4, 0.2, 1.5);
        var teacher = Deterministic(5, 4, 1.9, 1.2);

        var expected = FiniteDifference(student, s => strategy.ComputeLoss(s, teacher));
        AssertClose(expected, strategy.ComputeGradient(student, teacher));
    }

    [Theory(Timeout = 60000)]
    [InlineData(ProbabilisticMode.MomentMatching)]
    [InlineData(ProbabilisticMode.MaximumMeanDiscrepancy)]
    [InlineData(ProbabilisticMode.EntropyTransfer)]
    public async Task Probabilistic_ComputeGradient_MatchesFiniteDifferenceOfComputeLoss(ProbabilisticMode mode)
    {
        await Task.Yield();
        var strategy = new ProbabilisticDistillationStrategy<double>(
            distributionWeight: 0.5, mode: mode, mmdBandwidth: 1.0, temperature: 2.0, alpha: 0.3);
        var student = Deterministic(4, 3, 0.4, 2.0);
        var teacher = Deterministic(4, 3, 2.3, 2.0);

        var expected = FiniteDifference(student, s => strategy.ComputeLoss(s, teacher));
        AssertClose(expected, strategy.ComputeGradient(student, teacher));
    }

    [Fact(Timeout = 60000)]
    public async Task SimilarityPreserving_ComputeGradient_MatchesFiniteDifferenceOfComputeLoss()
    {
        await Task.Yield();
        var strategy = new SimilarityPreservingStrategy<double>(similarityWeight: 0.5, temperature: 2.0, alpha: 0.3);
        var student = Deterministic(4, 3, 0.8, 2.0);
        var teacher = Deterministic(4, 3, 3.1, 2.0);

        var expected = FiniteDifference(student, s => strategy.ComputeLoss(s, teacher));
        AssertClose(expected, strategy.ComputeGradient(student, teacher));
    }

    [Theory(Timeout = 60000)]
    [InlineData(SelectivityMetric.Variance)]
    [InlineData(SelectivityMetric.PeakToAverage)]
    public async Task NeuronSelectivity_IntermediateGradient_MatchesFiniteDifferenceOfIntermediateLoss(SelectivityMetric metric)
    {
        await Task.Yield();
        const string layer = "hidden";
        var strategy = new NeuronSelectivityDistillationStrategy<double>(
            targetLayerName: layer, selectivityWeight: 0.5, metric: metric);
        // Positive activations with a unique maximum per neuron keep peak-to-average differentiable
        var student = Deterministic(5, 3, 0.3, 0.4, 0.6);
        var teacher = Deterministic(5, 3, 2.7, 0.4, 0.6);
        var teacherActivations = new IntermediateActivations<double>();
        teacherActivations.Add(layer, teacher);

        double Loss(Matrix<double> s)
        {
            var studentActivations = new IntermediateActivations<double>();
            studentActivations.Add(layer, s);
            return strategy.ComputeIntermediateLoss(studentActivations, teacherActivations);
        }

        var studentAt = new IntermediateActivations<double>();
        studentAt.Add(layer, student);
        var gradient = strategy.ComputeIntermediateGradient(studentAt, teacherActivations).Get(layer);

        Assert.NotNull(gradient);
        if (gradient is not null)
        {
            AssertClose(FiniteDifference(student, Loss), gradient);
        }
    }

    [Theory(Timeout = 60000)]
    [InlineData(AttentionMatchingMode.MSE)]
    [InlineData(AttentionMatchingMode.Cosine)]
    public async Task Attention_IntermediateGradient_IsScaledLikeIntermediateLoss(AttentionMatchingMode mode)
    {
        await Task.Yield();
        string[] layers = ["attn1", "attn2"];
        var strategy = new AttentionDistillationStrategy<double>(layers, attentionWeight: 0.3, matchingMode: mode);
        var students = new[] { Deterministic(3, 4, 0.5), Deterministic(3, 4, 1.1) };
        var teachers = new[] { Deterministic(3, 4, 2.2), Deterministic(3, 4, 2.9) };
        var teacherActivations = new IntermediateActivations<double>();
        teacherActivations.Add(layers[0], teachers[0]);
        teacherActivations.Add(layers[1], teachers[1]);

        IntermediateActivations<double> StudentActivations(Matrix<double> first)
        {
            var activations = new IntermediateActivations<double>();
            activations.Add(layers[0], first);
            activations.Add(layers[1], students[1]);
            return activations;
        }

        var gradient = strategy.ComputeIntermediateGradient(StudentActivations(students[0]), teacherActivations).Get(layers[0]);
        var expected = FiniteDifference(
            students[0], s => strategy.ComputeIntermediateLoss(StudentActivations(s), teacherActivations));

        Assert.NotNull(gradient);
        if (gradient is not null)
        {
            AssertClose(expected, gradient);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task FactorTransfer_SameSubspace_GivesZeroLowRankLossOnEveryRun()
    {
        await Task.Yield();
        var strategy = new FactorTransferDistillationStrategy<double>(
            factorWeight: 1.0, mode: FactorMode.LowRankApproximation, numFactors: 2);

        // Strongly separated variances along a rotated basis, so power iteration converges fully
        double h = Math.Sqrt(0.5);
        var features = new Vector<double>[6];
        var negated = new Vector<double>[6];
        for (int r = 0; r < features.Length; r++)
        {
            double a = 10.0 * Math.Sin(r + 0.3);
            double b = Math.Cos(1.7 * r);
            double c = 0.01 * Math.Sin(2.3 * r);
            features[r] = new Vector<double>(new[] { h * (a + c), h * (a - c), b });
            negated[r] = new Vector<double>(new[] { -h * (a + c), -h * (a - c), -b });
        }

        // Eigenvectors are sign-ambiguous; with a canonical sign, independent random starts (and the negated
        // features, which share the covariance) must give identical factors
        for (int run = 0; run < 20; run++)
        {
            Assert.True(Math.Abs(strategy.ComputeFactorLoss(features, features)) <= 1e-10, $"run {run}: identical features");
            Assert.True(Math.Abs(strategy.ComputeFactorLoss(negated, features)) <= 1e-10, $"run {run}: negated features");
        }
    }
}
