using AiDotNet.FederatedLearning.Adapters;
using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.FederatedLearning.Compression;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Regression tests for inputs the per-element loops handled and the engine-op rewrites (plan 29b) did not:
/// a client that sends NaN or Infinity, and differential-privacy noise calibrated to the wrong sensitivity.
/// Every expected value is derived in the test's comment, not copied from the implementation.
/// </summary>
public class FederatedSweep29bNonFiniteAndPrivacyTests
{
    private const int Precision = 10;

    [Theory(Timeout = 60000)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public async Task Flame_ClientWithNonFiniteModelIsNotAdmitted(double poison)
    {
        await Task.Yield();

        // Three honest models [1, 0], [1, 0.01], [1, -0.01] and one client that sends a non-finite parameter.
        // Its cosine distance to every other model is NaN, so no finite distance threshold connects it to the
        // honest cluster and FLAME must not admit it (the Prim MST this replaced never picked a NaN edge).
        // Distances to G_{t-1} = 0 of the finite models: 1, sqrt(1.0001), sqrt(1.0001); their median is
        // sqrt(1.0001), nothing exceeds it, so nothing is clipped and G_t = mean = [1, 0].
        var flame = new FlameAggregationStrategy<double>(noiseMultiplier: 0.0);
        var points = new[] { new[] { 1.0, 0.0 }, new[] { 1.0, 0.01 }, new[] { 1.0, -0.01 }, new[] { poison, 0.0 } };
        var models = new Dictionary<int, Dictionary<string, double[]>>();
        var weights = new Dictionary<int, double>();
        for (int i = 0; i < points.Length; i++)
        {
            models[i] = new Dictionary<string, double[]> { ["w"] = points[i] };
            weights[i] = 1.0;
        }

        var aggregated = flame.Aggregate(models, weights);

        Assert.Equal(1.0, aggregated["w"][0], Precision);
        Assert.Equal(0.0, aggregated["w"][1], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task DPFedLoRA_NoiseCoversTheLargestClientWeight()
    {
        await Task.Yield();

        // Weighted DP-FedAvg: the release is sum_k (w_k / W) * clip(x_k, C). Replacing client k's update moves
        // it by at most (w_k / W) * C, so the L2 sensitivity is C * max_k w_k / W, not C / n.
        // Weights 9 and 1: sensitivity 0.9 C. With sigma = 1 and C = 1 every coordinate gets std 0.9
        // (C / n would give 0.5, which under-noises the client that carries 90% of the average).
        // Client 0 = [3, 4, 0, 0] (norm 5) clips to [0.6, 0.8, 0, 0]; client 1 = [0, 0, 0.3, 0.4] is kept.
        // Average = 0.9 * [0.6, 0.8, 0, 0] + 0.1 * [0, 0, 0.3, 0.4] = [0.54, 0.72, 0.03, 0.04].
        var strategy = new DPFedLoRA<double>(
            modelDim: 100, rank: 1, noiseMultiplier: 1.0, clipNorm: 1.0,
            numAdaptedLayers: 2, layerInputDim: 1, layerOutputDim: 1, seed: 42);
        var clients = new Dictionary<int, Vector<double>>
        {
            [0] = new Vector<double>(new[] { 3.0, 4.0, 0.0, 0.0 }),
            [1] = new Vector<double>(new[] { 0.0, 0.0, 0.3, 0.4 }),
        };

        var aggregated = strategy.AggregateAdapters(clients, new Dictionary<int, double> { [0] = 9.0, [1] = 1.0 });

        var average = new[] { 0.54, 0.72, 0.03, 0.04 };
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < average.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            Assert.Equal(average[i] + 0.9 * z, aggregated[i], Precision);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task FetchSGD_RecoverTopK_KeepsLargestFiniteEstimateWhenACoordinateIsNaN()
    {
        await Task.Yield();

        // g = [3, 1, NaN] in a 3 x 1000 sketch (collision-free for 3 coordinates): the estimates are
        // [3, 1, NaN]. The descending CompareTo sort this replaced ordered NaN last, so top-1 is index 0.
        var fetch = new FetchSGDCompressor<double>(sketchRows: 3, sketchCols: 1000, topK: 1);

        var recovered = fetch.RecoverTopK(fetch.Sketch(new[] { 3.0, 1.0, double.NaN }), 3);

        Assert.Equal(new[] { 3.0, 0.0, 0.0 }, recovered);
    }

    [Fact(Timeout = 60000)]
    public async Task SparseLoRA_SparsifyDelta_KeepsLargestFiniteEntryWhenAnEntryIsNaN()
    {
        await Task.Yield();

        // delta = [NaN, 5, 1, 0.1], k = max(1, 4 * 0.25) = 1. The ascending Array.Sort this replaced ordered NaN
        // first, so the k-th largest magnitude was 5 and only index 1 was kept.
        var lora = new SparseLoRA<double>(modelDim: 100, sparsityRatio: 0.25);

        var sparse = lora.SparsifyDelta(
            new Vector<double>(4), new Vector<double>(new[] { double.NaN, 5.0, 1.0, 0.1 }));

        Assert.Equal(new[] { 0.0, 5.0, 0.0, 0.0 }, SparseLoRA<double>.ToDense(sparse).ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task FedKD_LossesStayFiniteWhenBothModelsAreConfidentAndCorrect()
    {
        await Task.Yield();

        // Teacher and student both output logits [40, 0] for label 0. In float, softmax is exactly [1, 0]
        // (exp(-40) = 4.2e-18 is below half an ulp of 1), so both cross-entropies are -log(1) = 0 and the
        // adaptive scale 1 / (0 + 0) was Infinity. The two outputs and the two hidden states are identical,
        // so every distillation and hidden term is 0, and 0 * Infinity made the totals NaN.
        // Expected: every loss is exactly 0.
        var fedKd = new FedKDCompressor<float>();
        var logits = new Tensor<float>(new[] { 40f, 0f }, new[] { 1, 2 });
        var hidden = new Tensor<float>(new[] { 1f, 2f }, new[] { 1, 2 });

        var losses = fedKd.ComputeMutualDistillationLosses(logits, logits, new[] { 0 }, hidden, hidden);

        Assert.Equal(0f, losses.StudentTaskLoss);
        Assert.Equal(0f, losses.StudentDistillationLoss);
        Assert.Equal(0f, losses.HiddenLoss);
        Assert.Equal(0f, losses.StudentTotalLoss);
        Assert.Equal(0f, losses.TeacherTotalLoss);
    }
}
