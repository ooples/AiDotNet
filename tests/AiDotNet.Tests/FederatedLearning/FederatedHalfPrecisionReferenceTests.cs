using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.FederatedLearning.BackdoorDefense;
using AiDotNet.FederatedLearning.Compression;
using AiDotNet.FederatedLearning.Decentralized;
using AiDotNet.FederatedLearning.Fairness;
using AiDotNet.FederatedLearning.Personalization;
using AiDotNet.FederatedLearning.Privacy;
using AiDotNet.FederatedLearning.ServerOptimizers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Half-precision reference tests for the federated-learning reductions. Half stops accumulating at 2048
/// (2048 + 1 rounds to 2048), overflows past 65504 and flushes 1e-8 / 1e-10 guards to 0, so a sum, norm,
/// dot product or (a + b) / 2 formed in Half is wrong even when every input and the result fit in Half.
/// Each expected value is either hand-computed or the same algorithm evaluated in double; inputs are exactly
/// representable in Half, and the tolerance is a few Half ulps of the result (2^-11 relative per rounding).
/// </summary>
public class FederatedHalfPrecisionReferenceTests
{
    private const double HalfRelTol = 4e-3;

    // ---------------------------------------------------------------- robust full-model aggregators

    [Fact(Timeout = 60000)]
    public async Task Median_EvenCount_DoesNotOverflowHalf()
    {
        await Task.Yield();

        // Median of {40000, 50000} = 45000; (40000 + 50000) = 90000 overflows Half.
        var aggregated = AggregateFullModel(new MedianFullModelAggregationStrategy<Half, Tensor<Half>, Tensor<Half>>(), 40000, 50000);

        AssertClose(45000, aggregated, HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task TrimmedMean_DoesNotOverflowHalf()
    {
        await Task.Yield();

        // trimFraction 0.2 on n = 2 trims nothing: mean 45000.
        var aggregated = AggregateFullModel(new TrimmedMeanFullModelAggregationStrategy<Half, Tensor<Half>, Tensor<Half>>(0.2), 40000, 50000);

        AssertClose(45000, aggregated, HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task WinsorizedMean_DoesNotOverflowHalf()
    {
        await Task.Yield();

        var aggregated = AggregateFullModel(new WinsorizedMeanFullModelAggregationStrategy<Half, Tensor<Half>, Tensor<Half>>(0.2), 40000, 50000);

        AssertClose(45000, aggregated, HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task Bulyan_DoesNotOverflowHalf()
    {
        await Task.Yield();

        // f = 0, n = 3: every update is selected, beta = 3, the mean of {40000, 50000, 60000} is 50000.
        var aggregated = AggregateFullModel(
            new BulyanFullModelAggregationStrategy<Half, Tensor<Half>, Tensor<Half>>(byzantineClientCount: 0), 40000, 50000, 60000);

        AssertClose(50000, aggregated, HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task Rfa_DoesNotOverflowHalf()
    {
        await Task.Yield();

        // Weiszfeld from the mean of two points stays at their midpoint 45000.
        var aggregated = AggregateFullModel(new RfaFullModelAggregationStrategy<Half, Tensor<Half>, Tensor<Half>>(), 40000, 50000);

        AssertClose(45000, aggregated, HalfRelTol);
    }

    // ---------------------------------------------------------------- cosine / norm based aggregators

    [Fact(Timeout = 60000)]
    public async Task FLTrust_HalfMatchesDouble()
    {
        await Task.Yield();

        var serverGradient = new Dictionary<string, double[]> { ["w"] = new[] { 300.0, 400.0 } };
        var halfStrategy = new FLTrustAggregationStrategy<Half>();
        halfStrategy.SetServerGradient(ToHalf(serverGradient));
        var doubleStrategy = new FLTrustAggregationStrategy<double>();
        doubleStrategy.SetServerGradient(serverGradient);

        AssertAggregateMatches(halfStrategy, doubleStrategy, CosineFixture());
    }

    [Fact(Timeout = 60000)]
    public async Task FedAa_HalfMatchesDouble()
    {
        await Task.Yield();

        AssertAggregateMatches(new FedAaAggregationStrategy<Half>(), new FedAaAggregationStrategy<double>(), CosineFixture());
    }

    [Fact(Timeout = 60000)]
    public async Task FedMa_HalfMatchesDouble()
    {
        await Task.Yield();

        AssertAggregateMatches(new FedMaAggregationStrategy<Half>(), new FedMaAggregationStrategy<double>(), CosineFixture());
    }

    [Fact(Timeout = 60000)]
    public async Task OptiGradTrust_HalfMatchesDouble()
    {
        await Task.Yield();

        AssertAggregateMatches(
            new OptiGradTrustAggregationStrategy<Half>(), new OptiGradTrustAggregationStrategy<double>(), CosineFixture());
    }

    [Fact(Timeout = 60000)]
    public async Task Boba_HalfMatchesDouble()
    {
        await Task.Yield();

        var models = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = new() { ["w"] = new[] { 3000.0, 4000.0 } },
            [1] = new() { ["w"] = new[] { 3200.0, 3904.0 } },
            [2] = new() { ["w"] = new[] { 3104.0, 4096.0 } },
            [3] = new() { ["w"] = new[] { -30000.0, 40000.0 } },
        };

        AssertAggregateMatches(new BobaAggregationStrategy<Half>(), new BobaAggregationStrategy<double>(), models);
    }

    [Fact(Timeout = 60000)]
    public async Task Moon_ContrastiveLoss_HalfMatchesDouble()
    {
        await Task.Yield();

        // sim = cos / tau; the Half dot products (180000) and norm products overflow.
        double expected = new MoonAggregationStrategy<double>().ComputeContrastiveLoss(
            new Vector<double>(new[] { 300.0, 300.0 }),
            new Vector<double>(new[] { 300.0, 304.0 }),
            new Vector<double>(new[] { -300.0, 300.0 }));
        Half actual = new MoonAggregationStrategy<Half>().ComputeContrastiveLoss(
            HalfVector(300, 300), HalfVector(300, 304), HalfVector(-300, 300));

        AssertClose(expected, (double)actual, HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task FedSam_Perturbation_IsRhoTimesUnitGradientInHalf()
    {
        await Task.Yield();

        // Foret et al. 2021: eps = rho * g / ||g||. g = [30000, 40000], ||g|| = 50000, rho = 0.05 -> [0.03, 0.04].
        // ||g||^2 = 2.5e9 overflows Half, and rho / ||g|| = 1e-6 is a Half subnormal.
        var strategy = new FedSamAggregationStrategy<Half>(perturbationRadius: 0.05);

        var perturbation = strategy.ComputePerturbation(new Dictionary<string, Half[]> { ["w"] = new[] { (Half)30000.0, (Half)40000.0 } });

        AssertClose(0.03, (double)perturbation["w"][0], HalfRelTol);
        AssertClose(0.04, (double)perturbation["w"][1], HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task FedAlign_L2_DoesNotOverflowHalf()
    {
        await Task.Yield();

        // L2 alignment = ||F_local - F_global||_F^2 / n = 4 * 200^2 / 4 = 40000; the sum 160000 overflows Half.
        var strategy = new FedAlignAggregationStrategy<Half>(1.0, AlignmentDistanceMetric.L2);
        var local = HalfMatrix(new[,] { { 200.0 }, { 200.0 }, { 200.0 }, { 200.0 } });
        var global = HalfMatrix(new[,] { { 0.0 }, { 0.0 }, { 0.0 }, { 0.0 } });

        AssertClose(40000, (double)strategy.ComputeAlignmentLoss(local, global), HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task FedAlign_Cka_IdenticalLargeFeatures_IsZeroInHalf()
    {
        await Task.Yield();

        // Kornblith et al. 2019: CKA(X, X) = 1, so the distance 1 - CKA is 0; X^T X = 5e6 overflows Half.
        var strategy = new FedAlignAggregationStrategy<Half>(1.0, AlignmentDistanceMetric.CKA);
        var features = HalfMatrix(new[,] { { 1000.0 }, { 2000.0 }, { 3000.0 }, { 4000.0 } });

        AssertClose(0.0, (double)strategy.ComputeAlignmentLoss(features, features), 1e-2);
    }

    [Fact(Timeout = 60000)]
    public async Task FedDecorr_LargeFeatures_LossIsBetaOverDSquaredTimesFrobeniusInHalf()
    {
        await Task.Yield();

        // Shi et al. ICLR 2023: perfectly correlated columns give ||K||_F^2 = 4, loss 0.1 * 4 / 4 = 0.1.
        // The centred squares (1e6) overflow Half.
        var strategy = new FedDecorrAggregationStrategy<Half>(decorrelationWeight: 0.1);
        var features = HalfMatrix(new[,] { { 1000.0, 2000.0 }, { 2000.0, 4000.0 }, { 3000.0, 6000.0 } });

        AssertClose(0.1, (double)strategy.ComputeDecorrelationLoss(features), HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task FedLc_ZeroCountClass_UsesClampedMarginInHalf()
    {
        await Task.Yield();

        // FedLC eq. 9 with the 1e-8 count floor: (1e-8)^(-1/4) = 100, so logit 2 -> -98; count 16 -> 2 - 0.5 = 1.5.
        // In Half the 1e-8 floor rounds to 0 and 0^(-1/4) is infinite.
        var strategy = new FedLcAggregationStrategy<Half>(calibrationTemperature: 1.0);
        var counts = FedLcAggregationStrategy<Half>.ComputeClassCounts(Enumerable.Repeat(0, 16).ToArray(), 2);
        strategy.RegisterClassCounts(3, counts);

        var calibrated = strategy.CalibrateLogits(HalfVector(2, 2), 3);

        AssertClose(1.5, (double)calibrated[0], HalfRelTol);
        AssertClose(-98.0, (double)calibrated[1], HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task FedNtd_LargeLogits_HalfMatchesDouble()
    {
        await Task.Yield();

        double expected = new FedNtdAggregationStrategy<double>().ComputeNTDLoss(
            new Vector<double>(new[] { 0.0, 200.0, -200.0 }), new Vector<double>(new[] { 0.0, -200.0, 200.0 }), 0);
        Half actual = new FedNtdAggregationStrategy<Half>().ComputeNTDLoss(HalfVector(0, 200, -200), HalfVector(0, -200, 200), 0);

        AssertClose(expected, (double)actual, HalfRelTol);
    }

    // ---------------------------------------------------------------- backdoor defenses and fairness

    [Fact(Timeout = 60000)]
    public async Task DirectionAlignment_LargeUpdates_MatchesSignAgreementInHalf()
    {
        await Task.Yield();

        // Median of {-50000, -20000, 10000, 30000} is -5000; clients -50000 and -20000 align (suspicion 0),
        // 10000 and 30000 oppose it (suspicion 1). The Half squared norms (up to 2.5e9) overflow.
        var inspector = new DirectionAlignmentInspector<Half>(numSubspaces: 1, alignmentThreshold: 0.3);
        var updates = new Dictionary<int, Vector<Half>>
        {
            [0] = HalfVector(-50000),
            [1] = HalfVector(-20000),
            [2] = HalfVector(10000),
            [3] = HalfVector(30000),
        };

        var scores = inspector.DetectSuspiciousUpdates(updates, new Vector<Half>(1));

        Assert.Equal(0.0, scores[0], 6);
        Assert.Equal(0.0, scores[1], 6);
        Assert.Equal(1.0, scores[2], 6);
        Assert.Equal(1.0, scores[3], 6);
    }

    [Fact(Timeout = 60000)]
    public async Task NeuralCleanse_LargeUpdates_HalfMatchesDouble()
    {
        await Task.Yield();

        var doubleUpdates = new Dictionary<int, Vector<double>>();
        var halfUpdates = new Dictionary<int, Vector<Half>>();
        for (int k = 0; k < 4; k++)
        {
            var values = new double[20];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = k == 3 ? (i < 2 ? 30000.0 : 0.0) : 300.0 * (((i + k) % 5) + 1);
            }

            doubleUpdates[k] = new Vector<double>(values);
            halfUpdates[k] = HalfVector(values);
        }

        var expected = new NeuralCleanseDetector<double>().DetectSuspiciousUpdates(doubleUpdates, new Vector<double>(20));
        var actual = new NeuralCleanseDetector<Half>().DetectSuspiciousUpdates(halfUpdates, new Vector<Half>(20));

        foreach (var (clientId, score) in expected)
        {
            AssertClose(score, actual[clientId], HalfRelTol);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task GroupFairness_LargeModels_HalfMatchesDouble()
    {
        await Task.Yield();

        var values = new[] { new[] { 3000.0, 4000.0 }, new[] { 0.0, 0.0 }, new[] { 300.0, 400.0 }, new[] { 600.0, 800.0 } };
        var groups = new Dictionary<int, int> { [0] = 0, [1] = 1, [2] = 0, [3] = 1 };
        var doubleModels = new Dictionary<int, Tensor<double>>();
        var halfModels = new Dictionary<int, Tensor<Half>>();
        for (int i = 0; i < values.Length; i++)
        {
            doubleModels[i] = new Tensor<double>(values[i], new[] { 2 });
            halfModels[i] = new Tensor<Half>(values[i].Select(v => (Half)v).ToArray(), new[] { 2 });
        }

        double expected = new GroupFairnessConstraint<double>(new FederatedFairnessOptions())
            .EvaluateFairness(doubleModels, new Tensor<double>(new[] { 2 }), groups);
        double actual = new GroupFairnessConstraint<Half>(new FederatedFairnessOptions())
            .EvaluateFairness(halfModels, new Tensor<Half>(new[] { 2 }), groups);

        AssertClose(expected, actual, HalfRelTol);
    }

    // ---------------------------------------------------------------- compression

    [Fact(Timeout = 60000)]
    public async Task ErrorFeedback_OneBitScaleAndErrorNorm_DoNotOverflowHalf()
    {
        await Task.Yield();

        // Karimireddy et al. 2019: scale = mean |x| = (40000 + 20000) / 2 = 30000; compressed = [30000, -30000];
        // error = x - compressed = [10000, 10000], ||error|| = 10000 sqrt(2). Both sums overflow Half.
        var compressor = new ErrorFeedbackCompressor<Half>(new AdvancedCompressionOptions());

        var (_, scale) = compressor.CompressOneBit(new Tensor<Half>(new[] { (Half)40000.0, (Half)(-20000.0) }, new[] { 2 }), 0);

        AssertClose(30000, (double)scale, HalfRelTol);
        AssertClose(10000 * Math.Sqrt(2.0), (double)compressor.GetErrorNorm(0), HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task TopK_AggregateSparse_DoesNotOverflowHalf()
    {
        await Task.Yield();

        // Mean of two clients' 40000 at index 0 is 40000; the sum 80000 overflows Half.
        var compressor = new TopKSparsificationCompressor<Half>();
        var sparse = new Dictionary<int, Dictionary<string, List<(int Index, Half Value)>>>
        {
            [0] = new() { ["w"] = new List<(int, Half)> { (0, (Half)40000.0) } },
            [1] = new() { ["w"] = new List<(int, Half)> { (0, (Half)40000.0) } },
        };

        var aggregated = compressor.AggregateSparse(sparse, new Dictionary<string, int> { ["w"] = 2 });

        AssertClose(40000, (double)aggregated["w"][0], HalfRelTol);
        Assert.Equal(0.0, (double)aggregated["w"][1]);
    }

    [Fact(Timeout = 60000)]
    public async Task SignSgd_MajorityVote_CountsPastHalfSaturation()
    {
        await Task.Yield();

        // Bernstein et al. 2018: sign(sum of signs). 3000 votes of +1 then 2100 of -1 sum to +900;
        // a Half running tally sticks at 2048 and then falls to -52.
        var compressor = new SignSGDCompressor<Half>(learningRate: 1.0, useMajorityVote: true);
        var signs = new Dictionary<int, Dictionary<string, Half[]>>();
        for (int i = 0; i < 5100; i++)
        {
            signs[i] = new Dictionary<string, Half[]> { ["w"] = new[] { (Half)(i < 3000 ? 1.0 : -1.0) } };
        }

        var aggregated = compressor.AggregateVote(signs);

        Assert.Equal(1.0, (double)aggregated["w"][0]);
    }

    [Fact(Timeout = 60000)]
    public async Task PowerSgd_RankIsClampedToMatrixShape()
    {
        await Task.Yield();

        // Vogels et al. 2019 reference hook: rank = min(r, n, m). Size 4 -> 2 x 2 matrix, so r = 4 becomes 2:
        // P and Q are 2 x 2 and the ratio is (2*2 + 2*2) / 4 = 2 (not (2*4 + 2*4) / 4 = 4).
        var compressor = new PowerSGDCompressor<double>(new AdvancedCompressionOptions { PowerSGDRank = 4 });

        var (p, q, _, _) = compressor.Compress(new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 4 }), 0);

        Assert.Equal(2, p.Columns);
        Assert.Equal(2, q.Columns);
        Assert.Equal((double)((p.Rows * p.Columns) + (q.Rows * q.Columns)) / 4, compressor.GetCompressionRatio(4), 12);
        Assert.Equal(2.0, compressor.GetCompressionRatio(4), 12);
    }

    [Fact(Timeout = 60000)]
    public async Task GradientSketch_KeepsValueAtSignificanceBound()
    {
        await Task.Yield();

        // A value with |x| = 1e-15 is significant (>= bound) and must be inserted into the sketch.
        var compressor = new GradientSketchCompressor<double>(new AdvancedCompressionOptions { SketchDepth = 1, SketchWidth = 10 });

        var (sketch, size, width) = compressor.Compress(new Tensor<double>(new[] { 1e-15 }, new[] { 1 }));

        Assert.Equal(1e-15, compressor.Decompress(sketch, size, width)[0]);
    }

    [Fact(Timeout = 60000)]
    public async Task GradientSketch_EvenDepthMedian_DoesNotOverflowHalf()
    {
        await Task.Yield();

        // Depth 2, width 1: both rows estimate 60000, the median (mean of the two) is 60000; 120000 overflows Half.
        var compressor = new GradientSketchCompressor<Half>(new AdvancedCompressionOptions { SketchDepth = 2, SketchWidth = 1 });

        var (sketch, size, width) = compressor.Compress(new Tensor<Half>(new[] { (Half)60000.0 }, new[] { 1 }));

        AssertClose(60000, (double)compressor.Decompress(sketch, size, width)[0], HalfRelTol);
    }

    // ---------------------------------------------------------------- privacy

    [Fact(Timeout = 60000)]
    public async Task GaussianDp_ClipsLargeHalfUpdate()
    {
        await Task.Yield();

        // [30000, 40000] has norm 50000 (its square overflows Half); clipped to C = 1 it is [0.6, 0.8], and with
        // epsilon = 1e9 the noise sigma is ~5e-9.
        var dp = new GaussianDifferentialPrivacy<Half>(clipNorm: 1.0, randomSeed: 7);

        var privatized = dp.ApplyPrivacy(new Dictionary<string, Half[]> { ["w"] = new[] { (Half)30000.0, (Half)40000.0 } }, 1e9, 1e-5);

        AssertClose(0.6, (double)privatized["w"][0], HalfRelTol);
        AssertClose(0.8, (double)privatized["w"][1], HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task ShuffleModelDp_ClipsLargeHalfUpdate()
    {
        await Task.Yield();

        var dp = new ShuffleModelDP<Half>(localEpsilon: 1e9, clipNorm: 1.0);

        var privatized = dp.ApplyPrivacy(new Dictionary<string, Half[]> { ["w"] = new[] { (Half)30000.0, (Half)40000.0 } }, 1e9, 1e-5);

        AssertClose(0.6, (double)privatized["w"][0], HalfRelTol);
        AssertClose(0.8, (double)privatized["w"][1], HalfRelTol);
    }

    // ---------------------------------------------------------------- personalization and decentralized

    [Fact(Timeout = 60000)]
    public async Task FedAghn_GradientSimilarity_IsOneForParallelLargeHalfGradients()
    {
        await Task.Yield();

        var personalization = new FedAGHNPersonalization<Half>();

        Half similarity = personalization.ComputeGradientSimilarity(
            new[] { (Half)300.0, (Half)300.0 }, new[] { (Half)600.0, (Half)600.0 });

        AssertClose(1.0, (double)similarity, HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task FedPac_ClassPrototype_CountsPastHalfSaturation()
    {
        await Task.Yield();

        // Mean of 3000 feature vectors [1] is 1; a Half running sum stops at 2048 (2048 / 3000 = 0.68).
        var personalization = new FedPACPersonalization<Half>();
        var features = Enumerable.Range(0, 3000).Select(_ => new[] { (Half)1.0 }).ToArray();

        var prototypes = personalization.ComputeClassPrototypes(features, new int[3000]);

        AssertClose(1.0, (double)prototypes[0][0], HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task FedPac_GlobalPrototype_SampleWeightedMeanDoesNotOverflowHalf()
    {
        await Task.Yield();

        // Tan et al. 2023: global prototype = sum_k n_k p_k / sum_k n_k = 40000 for equal prototypes;
        // n_k p_k = 5000 * 40000 overflows Half.
        var personalization = new FedPACPersonalization<Half>();
        var prototypes = new Dictionary<int, Dictionary<int, Half[]>>
        {
            [0] = new() { [0] = new[] { (Half)40000.0 } },
            [1] = new() { [0] = new[] { (Half)40000.0 } },
        };
        var counts = new Dictionary<int, Dictionary<int, int>>
        {
            [0] = new() { [0] = 5000 },
            [1] = new() { [0] = 5000 },
        };

        var global = personalization.ComputeGlobalPrototypes(prototypes, counts);

        AssertClose(40000, (double)global[0][0], HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task KnnPer_EquidistantNeighbors_GiveUniformProbabilitiesInHalf()
    {
        await Task.Yield();

        // Both cached rows are sqrt(300^2 + 300^2) from the query, so each class gets weight 1/2.
        // The squared distances (180000) overflow Half and the weights become NaN.
        var knn = new KNNPersonalization<Half>(k: 2, lambda: 0.5, temperature: 1.0, distanceMetric: DistanceMetricType.Euclidean);
        knn.BuildCache(
            new[] { new[] { (Half)300.0, (Half)0.0 }, new[] { (Half)(-300.0), (Half)0.0 } },
            new[] { 0, 1 });

        var probabilities = knn.KNNPredict(new[] { (Half)0.0, (Half)300.0 }, 2);

        AssertClose(0.5, (double)probabilities[0], HalfRelTol);
        AssertClose(0.5, (double)probabilities[1], HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task SegmentedGossip_AveragesLargeHalfValues()
    {
        await Task.Yield();

        // The protocol needs at least 2 segments. With 2 segments over 2 parameters the first round averages
        // coordinate 0 only: (60000 + 60000) / 2 = 60000, whose sum 120000 overflows Half; coordinate 1 is kept.
        var gossip = new SegmentedGossipProtocol<Half>(numSegments: 2);

        var updated = gossip.GossipExchange(
            new Dictionary<string, Half[]> { ["w"] = new[] { (Half)60000.0, (Half)1.0 } },
            new Dictionary<string, Half[]> { ["w"] = new[] { (Half)60000.0, (Half)3.0 } });

        AssertClose(60000, (double)updated["w"][0], HalfRelTol);
        Assert.Equal(1.0, (double)updated["w"][1]);
    }

    [Fact(Timeout = 60000)]
    public async Task DFedBca_AverageBlock_NormalizesWeightsBeforeHalf()
    {
        await Task.Yield();

        // (10 * 10000 + 10 * 10000) / 20 = 10000; 10 * 10000 overflows Half.
        var protocol = new DFedBCAProtocol<Half>();
        var blocks = new Dictionary<int, Dictionary<string, Half[]>>
        {
            [0] = new() { ["w"] = new[] { (Half)10000.0 } },
            [1] = new() { ["w"] = new[] { (Half)10000.0 } },
        };

        var averaged = protocol.AverageBlock(blocks, new Dictionary<int, double> { [0] = 10.0, [1] = 10.0 });

        AssertClose(10000, (double)averaged["w"][0], HalfRelTol);
    }

    // ---------------------------------------------------------------- server optimizers

    [Fact(Timeout = 60000)]
    public async Task FedAdam_LargeDelta_DoesNotOverflowHalfState()
    {
        await Task.Yield();

        // delta = 1000: m = 100, v = 0.01 * 1e6 = 1e4, x = 100 / (100 + 0.001); d^2 = 1e6 overflows Half.
        var optimizer = new FedAdamServerOptimizer<Half>();

        var updated = optimizer.Step(HalfVector(0), HalfVector(1000));

        AssertClose(100.0 / 100.001, (double)updated[0], HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task FedAdagrad_LargeDelta_DoesNotOverflowHalfState()
    {
        await Task.Yield();

        // delta = 1000: m = 100, v = 1e6, x = 100 / (1000 + 0.001).
        var optimizer = new FedAdagradServerOptimizer<Half>();

        var updated = optimizer.Step(HalfVector(0), HalfVector(1000));

        AssertClose(100.0 / 1000.001, (double)updated[0], HalfRelTol);
    }

    [Fact(Timeout = 60000)]
    public async Task FedAvgM_VelocityPastHalfRange_KeepsMomentum()
    {
        await Task.Yield();

        // Hsu et al. 2019, lr 0.1, beta 0.9: v1 = 40000, x1 = 4000; delta 40000 again: v2 = 76000 (> 65504),
        // x2 = 4000 + 7600 = 11600.
        var optimizer = new FedAvgMServerOptimizer<Half>(learningRate: 0.1, momentum: 0.9);

        var x1 = optimizer.Step(HalfVector(0), HalfVector(40000));
        var x2 = optimizer.Step(x1, HalfVector((double)x1[0] + 40000));

        AssertClose(4000, (double)x1[0], HalfRelTol);
        AssertClose(11600, (double)x2[0], HalfRelTol);
    }

    // ---------------------------------------------------------------- helpers

    private static Dictionary<int, Dictionary<string, double[]>> CosineFixture() => new()
    {
        [0] = new() { ["w"] = new[] { 300.0, 400.0 } },
        [1] = new() { ["w"] = new[] { 400.0, 300.0 } },
        [2] = new() { ["w"] = new[] { -300.0, 400.0 } },
    };

    private static void AssertAggregateMatches(
        IAggregationStrategy<Dictionary<string, Half[]>> halfStrategy,
        IAggregationStrategy<Dictionary<string, double[]>> doubleStrategy,
        Dictionary<int, Dictionary<string, double[]>> models)
    {
        var weights = models.Keys.ToDictionary(id => id, _ => 1.0);
        var expected = doubleStrategy.Aggregate(models, weights);
        var actual = halfStrategy.Aggregate(models.ToDictionary(kv => kv.Key, kv => ToHalf(kv.Value)), weights);

        foreach (var (layer, values) in expected)
        {
            double scale = values.Max(v => Math.Abs(v));
            for (int i = 0; i < values.Length; i++)
            {
                AssertClose(values[i], (double)actual[layer][i], HalfRelTol, scale);
            }
        }
    }

    private static double AggregateFullModel(
        IAggregationStrategy<IFullModel<Half, Tensor<Half>, Tensor<Half>>> strategy, params double[] values)
    {
        var models = new Dictionary<int, IFullModel<Half, Tensor<Half>, Tensor<Half>>>();
        var weights = new Dictionary<int, double>();
        for (int i = 0; i < values.Length; i++)
        {
            var model = new MockFullModel<Half>(numClasses: 2, numParameters: 1);
            model.SetParameters(HalfVector(values[i]));
            models[i] = model;
            weights[i] = 1.0;
        }

        var aggregated = strategy.Aggregate(models, weights);
        return (double)((IParameterizable<Half, Tensor<Half>, Tensor<Half>>)aggregated).GetParameters()[0];
    }

    private static Dictionary<string, Half[]> ToHalf(Dictionary<string, double[]> model) =>
        model.ToDictionary(kv => kv.Key, kv => kv.Value.Select(v => (Half)v).ToArray());

    private static Vector<Half> HalfVector(params double[] values) => new(values.Select(v => (Half)v).ToArray());

    private static Matrix<Half> HalfMatrix(double[,] values)
    {
        var matrix = new Matrix<Half>(values.GetLength(0), values.GetLength(1));
        for (int r = 0; r < values.GetLength(0); r++)
        {
            for (int c = 0; c < values.GetLength(1); c++)
            {
                matrix[r, c] = (Half)values[r, c];
            }
        }

        return matrix;
    }

    private static void AssertClose(double expected, double actual, double relTol, double scale = 0)
    {
        double bound = relTol * Math.Max(1.0, Math.Max(Math.Abs(expected), scale));
        Assert.True(
            !double.IsNaN(actual) && Math.Abs(expected - actual) <= bound,
            $"Expected {expected} (+/- {bound}), got {actual}.");
    }
}
