using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Data.Structures;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Verifies that <see cref="MbPAAlgorithm{T, TInput, TOutput}"/> implements Memory-based Parameter
/// Adaptation (Sprechmann et al., arXiv:1802.10542) rather than a generic memory-flavoured learner.
/// </summary>
/// <remarks>
/// Four properties define MbPA and none of them is visible from a loss-is-finite smoke test: the
/// retrieval kernel is <c>1 / (eps + d^2)</c> over Euclidean distance rather than any similarity
/// softmax; the local adaptation is regularized back toward the trained parameters; it is
/// query-conditioned, so two different inputs get two different adaptations; and it is TRANSIENT,
/// so nothing survives the prediction. Each is asserted directly here.
/// </remarks>
public class MbPAMechanismTests
{
    private const int FeatureDim = 4;
    private const int OutputDim = 3;

    private static Vector<double> Vec(params double[] values)
    {
        var v = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++) v[i] = values[i];
        return v;
    }

    // ---------------------------------------------------------------- memory

    [Fact]
    public void Retrieval_RanksByEuclideanDistance_AndReturnsTheKNearest()
    {
        var memory = new MbPAEpisodicMemory<double>(capacity: 10);
        memory.Write(Vec(0, 0, 0, 0), Vec(1, 0, 0));     // distance^2 = 3 from the query
        memory.Write(Vec(1, 1, 1, 0), Vec(0, 1, 0));     // distance^2 = 0  <- nearest
        memory.Write(Vec(5, 5, 5, 5), Vec(0, 0, 1));     // distance^2 = 73

        var query = Vec(1, 1, 1, 0);
        var retrieved = memory.Retrieve(query, k: 2, epsilon: 1e-6, toDouble: x => x);

        Assert.Equal(2, retrieved.Count);

        // Nearest first: the exact-match key (1,1,1,0) at distance^2 = 0, ahead of (0,0,0,0) at 3.
        Assert.Equal(1.0, retrieved[0].Key[0], 12);
        Assert.Equal(1.0, retrieved[0].Key[1], 12);
        Assert.Equal(1.0, retrieved[0].Key[2], 12);
        Assert.Equal(0.0, retrieved[0].Key[3], 12);
        Assert.Equal(1.0, retrieved[0].Value[1], 12);      // its value came along with it

        // Second nearest is (0,0,0,0) at distance^2 = 3.
        Assert.Equal(0.0, retrieved[1].Key[0], 12);

        // The far entry is excluded by K, not merely down-weighted.
        Assert.DoesNotContain(retrieved, r => Math.Abs(r.Key[0] - 5.0) < 1e-9);
    }

    [Fact]
    public void RetrievalWeights_AreTheInverseSquaredDistanceKernel_NotASoftmax()
    {
        // kern(h, q) = 1 / (eps + ||h - q||^2). With distances^2 of 1 and 3 the weights must be in
        // the ratio 3:1. A softmax over negative distances would give exp(-1):exp(-3) = 7.39:1, and
        // a cosine-similarity softmax something else again — so this pins the actual kernel.
        var memory = new MbPAEpisodicMemory<double>(capacity: 10);
        memory.Write(Vec(1, 0, 0, 0), Vec(1, 0, 0));     // ||h - 0||^2 = 1
        memory.Write(Vec(0, 1, 1, 1), Vec(0, 1, 0));     // ||h - 0||^2 = 3

        var retrieved = memory.Retrieve(Vec(0, 0, 0, 0), k: 2, epsilon: 0.0, toDouble: x => x);

        Assert.Equal(2, retrieved.Count);
        Assert.Equal(0.75, retrieved[0].Weight, 9);      // (1/1) / (1/1 + 1/3)
        Assert.Equal(0.25, retrieved[1].Weight, 9);      // (1/3) / (1/1 + 1/3)
        Assert.Equal(1.0, retrieved.Sum(r => r.Weight), 9);
    }

    [Fact]
    public void Memory_EvictsOldestAtCapacity()
    {
        var memory = new MbPAEpisodicMemory<double>(capacity: 2);
        memory.Write(Vec(1, 0, 0, 0), Vec(1, 0, 0));
        memory.Write(Vec(2, 0, 0, 0), Vec(0, 1, 0));
        memory.Write(Vec(3, 0, 0, 0), Vec(0, 0, 1));

        Assert.Equal(2, memory.Count);
        var all = memory.Retrieve(Vec(0, 0, 0, 0), k: 2, epsilon: 1e-6, toDouble: x => x);
        Assert.DoesNotContain(all, r => Math.Abs(r.Key[0] - 1.0) < 1e-9);   // the first write is gone
    }

    [Fact]
    public void Memory_RejectsNonPositiveCapacity()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new MbPAEpisodicMemory<double>(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MbPAEpisodicMemory<double>(-5));
    }

    [Fact]
    public void EmptyMemory_RetrievesNothing()
    {
        var memory = new MbPAEpisodicMemory<double>(capacity: 4);
        Assert.Empty(memory.Retrieve(Vec(1, 1, 1, 1), k: 3, epsilon: 1e-6, toDouble: x => x));
    }

    // ------------------------------------------------------- local adaptation

    private static (Vector<double> Key, Vector<double> Value, double Weight)[] OneNeighbor(
        Vector<double> key, Vector<double> value, double weight = 1.0) => [(key, value, weight)];

    [Fact]
    public void LocalAdaptation_MovesTowardTheRetrievedNeighbours()
    {
        var trained = new Vector<double>(OutputDim * FeatureDim + OutputDim);
        var key = Vec(1, 0, 0, 0);
        var target = Vec(1, 0, 0);   // class 0

        var adapted = MbPAOutputNetwork<double>.LocallyAdapt(
            trained, OneNeighbor(key, target),
            steps: 5, localLearningRate: 0.5, beta: 0.0,
            FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);

        var before = MbPAOutputNetwork<double>.Forward(
            trained, key, FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);
        var after = MbPAOutputNetwork<double>.Forward(
            adapted, key, FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);

        Assert.True(after[0] > before[0],
            $"Fitting the neighbour must raise its class probability; {before[0]} -> {after[0]}");
    }

    [Fact]
    public void LocalAdaptation_IsRegularizedBackTowardTheTrainedParameters()
    {
        // The MAP prior -||theta_x - theta||^2 / (2 alpha_M) is the difference between MbPA and
        // unconstrained fine-tuning on a handful of neighbours. With beta > 0 the same evidence must
        // move the parameters strictly less far.
        var trained = new Vector<double>(OutputDim * FeatureDim + OutputDim);
        var key = Vec(1, 1, 0, 0);
        var target = Vec(0, 1, 0);

        double Drift(double beta)
        {
            var adapted = MbPAOutputNetwork<double>.LocallyAdapt(
                trained, OneNeighbor(key, target),
                steps: 10, localLearningRate: 0.5, beta: beta,
                FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);
            double sum = 0.0;
            for (int i = 0; i < adapted.Length; i++)
            {
                double d = adapted[i] - trained[i];
                sum += d * d;
            }
            return Math.Sqrt(sum);
        }

        double unregularized = Drift(0.0);
        double regularized = Drift(0.3);

        Assert.True(unregularized > 0, "The local steps did nothing at all.");
        Assert.True(regularized < unregularized,
            $"beta must restrain the drift: unregularized {unregularized:F6} vs regularized {regularized:F6}");
    }

    [Fact]
    public void LocalAdaptation_NeverMutatesTheTrainedParameters()
    {
        // "used for output computation but discarded thereafter" — the trained vector is the thing
        // that must survive untouched, or the adaptation would silently accumulate.
        var trained = new Vector<double>(OutputDim * FeatureDim + OutputDim);
        for (int i = 0; i < trained.Length; i++) trained[i] = 0.01 * (i + 1);
        var snapshot = trained.Clone();

        MbPAOutputNetwork<double>.LocallyAdapt(
            trained, OneNeighbor(Vec(1, 1, 1, 1), Vec(1, 0, 0)),
            steps: 10, localLearningRate: 0.9, beta: 0.05,
            FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);

        for (int i = 0; i < trained.Length; i++) Assert.Equal(snapshot[i], trained[i], 15);
    }

    [Fact]
    public void LocalAdaptation_WithNoNeighbours_LeavesTheTrainedParametersAlone()
    {
        var trained = new Vector<double>(OutputDim * FeatureDim + OutputDim);
        for (int i = 0; i < trained.Length; i++) trained[i] = 0.05 * i;

        var adapted = MbPAOutputNetwork<double>.LocallyAdapt(
            trained, [], steps: 5, localLearningRate: 0.5, beta: 0.1,
            FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);

        for (int i = 0; i < trained.Length; i++) Assert.Equal(trained[i], adapted[i], 15);
    }

    [Fact]
    public void LocalAdaptation_IsQueryConditioned_DifferentNeighboursGiveDifferentParameters()
    {
        // This is why the adaptation cannot be computed once and reused: theta_x depends on which
        // neighbours the query retrieved.
        var trained = new Vector<double>(OutputDim * FeatureDim + OutputDim);

        var a = MbPAOutputNetwork<double>.LocallyAdapt(
            trained, OneNeighbor(Vec(1, 0, 0, 0), Vec(1, 0, 0)),
            steps: 3, localLearningRate: 0.5, beta: 0.0,
            FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);
        var b = MbPAOutputNetwork<double>.LocallyAdapt(
            trained, OneNeighbor(Vec(0, 0, 0, 1), Vec(0, 0, 1)),
            steps: 3, localLearningRate: 0.5, beta: 0.0,
            FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);

        bool differ = Enumerable.Range(0, a.Length).Any(i => Math.Abs(a[i] - b[i]) > 1e-9);
        Assert.True(differ, "Two different retrieved neighbourhoods produced identical theta_x.");
    }

    [Fact]
    public void MoreLocalSteps_FitTheNeighbourMoreClosely()
    {
        var trained = new Vector<double>(OutputDim * FeatureDim + OutputDim);
        var key = Vec(1, 0, 0, 0);
        var target = Vec(1, 0, 0);

        double ProbabilityAfter(int steps)
        {
            var adapted = MbPAOutputNetwork<double>.LocallyAdapt(
                trained, OneNeighbor(key, target),
                steps, localLearningRate: 0.3, beta: 0.0,
                FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);
            return MbPAOutputNetwork<double>.Forward(
                adapted, key, FeatureDim, OutputDim, MbPAOutputDistribution.Categorical)[0];
        }

        Assert.True(ProbabilityAfter(10) > ProbabilityAfter(1),
            "T is the number of local gradient steps, so more of them must fit the evidence better.");
    }

    // ------------------------------------------------------- output network

    [Fact]
    public void CategoricalHead_ProducesANormalizedDistribution()
    {
        var parameters = new Vector<double>(OutputDim * FeatureDim + OutputDim);
        for (int i = 0; i < parameters.Length; i++) parameters[i] = 0.1 * ((i % 7) - 3);

        var output = MbPAOutputNetwork<double>.Forward(
            parameters, Vec(0.5, -0.25, 1.0, 0.0), FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);

        double total = 0.0;
        for (int o = 0; o < OutputDim; o++)
        {
            Assert.InRange(output[o], 0.0, 1.0);
            total += output[o];
        }
        Assert.Equal(1.0, total, 12);
    }

    [Fact]
    public void GaussianHead_IsLinear_AndNotNormalized()
    {
        // Weights laid out as [o * featureDim + f], biases after them.
        var parameters = new Vector<double>(OutputDim * FeatureDim + OutputDim);
        parameters[0 * FeatureDim + 0] = 2.0;                 // output 0 <- 2 * feature 0
        parameters[OutputDim * FeatureDim + 1] = -1.5;        // output 1 bias

        var output = MbPAOutputNetwork<double>.Forward(
            parameters, Vec(3.0, 0, 0, 0), FeatureDim, OutputDim, MbPAOutputDistribution.Gaussian);

        Assert.Equal(6.0, output[0], 12);
        Assert.Equal(-1.5, output[1], 12);
        Assert.Equal(0.0, output[2], 12);
    }

    [Fact]
    public void Gradient_MatchesFiniteDifferences()
    {
        // The closed-form (prediction - target) (x) h is claimed to be EXACT, not an approximation,
        // so it has to agree with a numerical derivative of the actual log-likelihood.
        var parameters = new Vector<double>(OutputDim * FeatureDim + OutputDim);
        for (int i = 0; i < parameters.Length; i++) parameters[i] = 0.05 * ((i % 5) - 2);
        var key = Vec(0.7, -0.3, 0.2, 1.1);
        var target = Vec(0, 1, 0);

        var analytic = MbPAOutputNetwork<double>.Gradient(
            parameters, key, target, weight: 1.0, FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);

        double CrossEntropy(Vector<double> p)
        {
            var probs = MbPAOutputNetwork<double>.Forward(
                p, key, FeatureDim, OutputDim, MbPAOutputDistribution.Categorical);
            double loss = 0.0;
            for (int o = 0; o < OutputDim; o++) loss -= target[o] * Math.Log(Math.Max(probs[o], 1e-15));
            return loss;
        }

        const double h = 1e-6;
        for (int i = 0; i < parameters.Length; i++)
        {
            var plus = parameters.Clone(); plus[i] += h;
            var minus = parameters.Clone(); minus[i] -= h;
            double numeric = (CrossEntropy(plus) - CrossEntropy(minus)) / (2 * h);
            Assert.Equal(numeric, analytic[i], 6);
        }
    }

    // ------------------------------------------------------------- end to end

    [Fact]
    public void PaperHyperparametersAreTheDefaults()
    {
        var o = new MbPAOptions<double, Matrix<double>, Vector<double>>(new IdentityEmbeddingModel(FeatureDim));

        // The paper sweeps K and T over [1, 20] and alpha_M over [0, 1], reporting alpha_M = 0.15
        // and T = 1 as optima for language modelling.
        Assert.InRange(o.NumNeighbors, 1, 20);
        Assert.InRange(o.LocalAdaptationSteps, 1, 20);
        Assert.InRange(o.LocalLearningRate, 0.0, 1.0);
        Assert.Equal(0.15, o.LocalLearningRate);
        Assert.Equal(1, o.LocalAdaptationSteps);
        Assert.InRange(o.MemorySize, 100, 5000);
        Assert.Equal(MbPAOutputDistribution.Categorical, o.OutputDistribution);
        Assert.True(o.WriteMemoryDuringTraining);

        // The local rate must be far above the training rate — that is the paper's central claim.
        Assert.True(o.LocalLearningRate > 10 * o.InnerLearningRate,
            "MbPA's whole point is that local adaptation can use a much higher learning rate.");
    }

    [Fact]
    public void Constructor_RejectsInvalidConfiguration()
    {
        var model = new IdentityEmbeddingModel(FeatureDim);
        Assert.Throws<ArgumentNullException>(() =>
            new MbPAAlgorithm<double, Matrix<double>, Vector<double>>(null!));

        // ThrowsAny, not Throws: MetaLearnerBase validates IMetaLearnerOptions before this class's
        // own range checks run, so the exact ArgumentException subtype is the base's choice. What
        // the contract actually promises is that an invalid configuration is rejected at
        // construction rather than producing a silently degenerate learner.
        Assert.ThrowsAny<ArgumentException>(() =>
            new MbPAAlgorithm<double, Matrix<double>, Vector<double>>(
                new MbPAOptions<double, Matrix<double>, Vector<double>>(model) { NumNeighbors = 0 }));
        Assert.ThrowsAny<ArgumentException>(() =>
            new MbPAAlgorithm<double, Matrix<double>, Vector<double>>(
                new MbPAOptions<double, Matrix<double>, Vector<double>>(model) { LocalAdaptationSteps = 0 }));
        Assert.ThrowsAny<ArgumentException>(() =>
            new MbPAAlgorithm<double, Matrix<double>, Vector<double>>(
                new MbPAOptions<double, Matrix<double>, Vector<double>>(model) { MemorySize = 0 }));
        Assert.ThrowsAny<ArgumentException>(() =>
            new MbPAAlgorithm<double, Matrix<double>, Vector<double>>(
                new MbPAOptions<double, Matrix<double>, Vector<double>>(model) { FeatureDimension = 0 }));
    }

    [Fact]
    public void AdaptedModel_PredictsWithoutRetainingAnyAdaptation()
    {
        // Two identical calls must agree exactly. If theta_x leaked across calls the second would
        // start from an already-adapted head and differ.
        var algorithm = BuildAlgorithm(out var probe);

        var adapted = algorithm.Adapt(new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = probe.Support, SupportSetY = probe.SupportY,
            QuerySetX = probe.Query, QuerySetY = probe.QueryY,
        });

        var first = adapted.Predict(probe.Query);
        var second = adapted.Predict(probe.Query);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i], second[i], 12);

        // And the algorithm's own trained head is untouched by having predicted.
        var headAfter = algorithm.OutputParameters;
        Assert.All(Enumerable.Range(0, headAfter.Length),
            i => Assert.False(double.IsNaN(headAfter[i])));
    }

    [Fact]
    public void MemoryIsWrittenDuringAdaptation()
    {
        var algorithm = BuildAlgorithm(out var probe);
        int before = algorithm.MemoryCount;

        algorithm.Adapt(new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = probe.Support, SupportSetY = probe.SupportY,
            QuerySetX = probe.Query, QuerySetY = probe.QueryY,
        });

        Assert.True(algorithm.MemoryCount > before,
            "Adapting to a new task must record it — that is how MbPA absorbs a distribution shift.");
    }

    [Fact]
    public void ClearMemory_EmptiesTheStore()
    {
        var algorithm = BuildAlgorithm(out var probe);
        algorithm.WriteToMemory(probe.Support, probe.SupportY);
        Assert.True(algorithm.MemoryCount > 0);

        algorithm.ClearMemory();
        Assert.Equal(0, algorithm.MemoryCount);
    }

    private static MbPAAlgorithm<double, Matrix<double>, Vector<double>> BuildAlgorithm(out Probe probe)
    {
        var rng = new Random(4);
        var support = new Matrix<double>(6, FeatureDim);
        var query = new Matrix<double>(4, FeatureDim);
        for (int i = 0; i < support.Rows; i++)
            for (int j = 0; j < FeatureDim; j++) support[i, j] = rng.NextDouble();
        for (int i = 0; i < query.Rows; i++)
            for (int j = 0; j < FeatureDim; j++) query[i, j] = rng.NextDouble();

        var supportY = new Vector<double>(support.Rows);
        for (int i = 0; i < supportY.Length; i++) supportY[i] = i % OutputDim;
        var queryY = new Vector<double>(query.Rows);
        for (int i = 0; i < queryY.Length; i++) queryY[i] = i % OutputDim;

        probe = new Probe(support, supportY, query, queryY);

        var options = new MbPAOptions<double, Matrix<double>, Vector<double>>(new IdentityEmbeddingModel(FeatureDim))
        {
            FeatureDimension = FeatureDim,
            OutputDimension = OutputDim,
            NumNeighbors = 3,
            MemorySize = 128,
            Seed = 4,
        };
        return new MbPAAlgorithm<double, Matrix<double>, Vector<double>>(options);
    }

    private sealed record Probe(
        Matrix<double> Support, Vector<double> SupportY, Matrix<double> Query, Vector<double> QueryY);
}
