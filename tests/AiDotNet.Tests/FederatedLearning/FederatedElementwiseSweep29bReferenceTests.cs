using AiDotNet.FederatedLearning.Adapters;
using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.FederatedLearning.Compression;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Reference tests for the paper fixes made while moving the remaining federated-learning code (plan 29b)
/// onto engine ops. Every expected value is hand-computed from the cited paper's algorithm.
/// </summary>
public class FederatedElementwiseSweep29bReferenceTests
{
    private const int Precision = 10;

    [Fact(Timeout = 60000)]
    public async Task DPFedLoRA_AddsDataIndependentNoiseSigmaClipOverN()
    {
        await Task.Yield();

        // DP-FedAvg Gaussian mechanism: clip each whole update to C, average, add N(0, (sigma * C / n)^2).
        // Client 0 = [3, 4, 0, 0] (norm 5) clips to [0.6, 0.8, 0, 0]; client 1 = [0, 0, 0.3, 0.4] (norm 0.5)
        // is kept. The average is [0.3, 0.4, 0.15, 0.2] and every coordinate gets std 1 * 1 / 2 = 0.5.
        // (Splitting the clip budget by the clients' own per-layer norms gave stds 1/3 and 1/6.)
        var strategy = new DPFedLoRA<double>(
            modelDim: 100, rank: 1, noiseMultiplier: 1.0, clipNorm: 1.0,
            numAdaptedLayers: 2, layerInputDim: 1, layerOutputDim: 1, seed: 42);
        var clients = new Dictionary<int, Vector<double>>
        {
            [0] = new Vector<double>(new[] { 3.0, 4.0, 0.0, 0.0 }),
            [1] = new Vector<double>(new[] { 0.0, 0.0, 0.3, 0.4 }),
        };

        var aggregated = strategy.AggregateAdapters(clients, null);

        var average = new[] { 0.3, 0.4, 0.15, 0.2 };
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < average.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            Assert.Equal(average[i] + 0.5 * z, aggregated[i], Precision);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task LoRAMerge_WritesStoredFactorsUnscaled()
    {
        await Task.Yield();

        // LoRA (Hu et al. 2021) applies alpha / r to B*A in the forward pass. The stored factors are merged
        // back as they are; scaling both by alpha / r = 2 would scale the update by 4 and compound per round.
        var full = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var adapters = new Vector<double>(new[] { 7.0, 8.0 });
        var expected = new[] { 1.0, 2.0, 3.0, 7.0, 8.0 };

        IFederatedAdapterStrategy<double>[] strategies =
        {
            new DPFedLoRA<double>(modelDim: 5, rank: 8, alpha: 16.0),
            new FLoRA<double>(modelDim: 5, rank: 8, alpha: 16.0),
            new FederatedLoRA<double>(modelDim: 5, rank: 8, alpha: 16.0),
            new SparseLoRA<double>(modelDim: 5, rank: 8, alpha: 16.0),
        };

        foreach (var strategy in strategies)
        {
            var merged = strategy.MergeAdapterParameters(full, adapters);
            Assert.Equal(expected, merged.ToArray());
        }
    }

    [Fact(Timeout = 60000)]
    public async Task FLoRA_ReturnsBestRankRApproximationOfStackedUpdate()
    {
        await Task.Yield();

        // FLoRA (Wang et al. 2024) stacking gives dW = sum_k p_k B_k A_k. Client 0: B = [1, 0], A = [2, 0];
        // client 1: B = [0, 1], A = [0, 1]; equal weights give dW = diag(1, 0.5). Its best rank-1
        // approximation (Eckart-Young) is diag(1, 0), so the returned factors must satisfy B*A = diag(1, 0).
        var strategy = new FLoRA<double>(modelDim: 100, rank: 1, numAdaptedLayers: 1, layerInputDim: 2, layerOutputDim: 2);
        var clients = new Dictionary<int, Vector<double>>
        {
            [0] = new Vector<double>(new[] { 1.0, 0.0, 2.0, 0.0 }),
            [1] = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 }),
        };

        var aggregated = strategy.AggregateAdapters(clients, null);

        Assert.Equal(1.0, aggregated[0] * aggregated[2], Precision);
        Assert.Equal(0.0, aggregated[0] * aggregated[3], Precision);
        Assert.Equal(0.0, aggregated[1] * aggregated[2], Precision);
        Assert.Equal(0.0, aggregated[1] * aggregated[3], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task FedPETuning_LoRAAggregationIsFedAvgOverEveryEntry()
    {
        await Task.Yield();

        // FedPETuning (Zhang et al. 2023) aggregates PEFT parameters with FedAvg. With layerDim = 1, rank = 1
        // and one layer the B/A layout covers 2 entries; a third entry must be averaged, not zeroed.
        var strategy = new FedPETuning<double>(modelDim: 10, method: PEFTMethod.LoRA, bottleneckDim: 1, numAdaptedLayers: 1, layerDim: 1);
        var clients = new Dictionary<int, Vector<double>>
        {
            [0] = new Vector<double>(new[] { 1.0, 2.0, 3.0 }),
            [1] = new Vector<double>(new[] { 3.0, 4.0, 5.0 }),
        };

        var aggregated = strategy.AggregateAdaptersMethodAware(clients, null);

        Assert.Equal(new[] { 2.0, 3.0, 4.0 }, aggregated.ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task FederatedAdapterTuning_HoulsbyAdaptersHaveTwoPerLayerWithBothBiases()
    {
        await Task.Yield();

        // Houlsby et al. 2019 adapter: x + Up(ReLU(Down x + b_down)) + b_up, two adapters per layer.
        // d = 2, m = 1: params per adapter = 2*2*1 + 1 + 2 = 7, so one layer holds 14 parameters.
        var tuning = new FederatedAdapterTuning<double>(modelDim: 100, bottleneckDim: 1, numAdaptedLayers: 1, layerDim: 2);
        Assert.Equal(14, tuning.AdapterParameterCount);

        // Down = [1, 0], Up = [2, 3], b_down = 0.5, b_up = [0.1, 0.2]; x = [1, -1]:
        // hidden = ReLU(1 + 0.5) = 1.5, adapter = [3 + 0.1, 4.5 + 0.2], output = [4.1, 3.7].
        var ffnAdapter = new[] { 1.0, 0.0, 2.0, 3.0, 0.5, 0.1, 0.2 };
        var all = new Vector<double>(new double[7].Concat(ffnAdapter).ToArray());
        var extracted = tuning.GetLayerAdapterParams(all, layerIndex: 0, position: 1);
        Assert.Equal(ffnAdapter, extracted);

        var output = tuning.ApplyAdapterForward(new[] { 1.0, -1.0 }, extracted);
        Assert.Equal(4.1, output[0], Precision);
        Assert.Equal(3.7, output[1], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task FederatedLoRA_ExactAggregationResidualIsMeanProductMinusProductOfMeans()
    {
        await Task.Yield();

        // FedEx-LoRA (Sun et al. 2025): residual = avg(B_k A_k) - avg(B) avg(A).
        // Client 0: B = 1, A = 2; client 1: B = 3, A = 4 (rank 1, 1x1 layer): avg(BA) = (2 + 12) / 2 = 7,
        // avg(B) avg(A) = 2 * 3 = 6, so the residual is 1. The averaged factors are B = 2, A = 3.
        var lora = new FederatedLoRA<double>(modelDim: 10, rank: 1, alpha: 2.0, numAdaptedLayers: 1, layerInputDim: 1, layerOutputDim: 1);
        var clients = new Dictionary<int, Vector<double>>
        {
            [0] = new Vector<double>(new[] { 1.0, 2.0 }),
            [1] = new Vector<double>(new[] { 3.0, 4.0 }),
        };

        var residual = lora.ComputeExactAggregationResidual(clients, null);
        Assert.Equal(new[] { 1, 1, 1 }, residual.Shape.ToArray());
        Assert.Equal(1.0, residual[0, 0, 0], Precision);
        Assert.Equal(new[] { 2.0, 3.0 }, lora.AggregateAdapters(clients, null).ToArray());

        // Merge writes the factors unscaled (alpha / rank = 2 is applied in the forward pass).
        var merged = lora.MergeAdapterParameters(new Vector<double>(new[] { 9.0, 1.0, 1.0 }), new Vector<double>(new[] { 2.0, 3.0 }));
        Assert.Equal(new[] { 9.0, 2.0, 3.0 }, merged.ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task HeterogeneousLoRA_FlexLoRAAveragesFullUpdatesThenTruncatesBySvd()
    {
        await Task.Yield();

        // FlexLoRA (Bai et al. 2024): dW = sum_k p_k B_k A_k, then SVD truncation at the requested rank.
        // layerDim = 2. Client 0 (rank 1): B = [1, 0]^T, A = [2, 0] -> diag(2, 0).
        // Client 1 (rank 2): B = I, A = [[0, 0], [0, 1]] -> diag(0, 1). Equal weights: dW = diag(1, 0.5).
        var lora = new HeterogeneousLoRA<double>(modelDim: 100, layerDim: 2, targetRank: 1, maxRank: 2);
        var clients = new Dictionary<int, Vector<double>>
        {
            [0] = new Vector<double>(new[] { 1.0, 0.0, 2.0, 0.0 }),
            [1] = new Vector<double>(new[] { 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 }),
        };

        // Rank 1: the best approximation is diag(1, 0).
        var rank1 = lora.AggregateAdapters(clients, null);
        Assert.Equal(4, rank1.Length);
        Assert.Equal(1.0, rank1[0] * rank1[2], Precision);
        Assert.Equal(0.0, rank1[0] * rank1[3], Precision);
        Assert.Equal(0.0, rank1[1] * rank1[2], Precision);
        Assert.Equal(0.0, rank1[1] * rank1[3], Precision);

        // Rank 2 reproduces dW = diag(1, 0.5) exactly. B is [2 x 2] then A is [2 x 2], both row-major.
        var rank2 = lora.AggregateAdaptersForRank(clients, null, 2);
        var expected = new[,] { { 1.0, 0.0 }, { 0.0, 0.5 } };
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                double product = rank2[i * 2] * rank2[4 + j] + rank2[i * 2 + 1] * rank2[4 + 2 + j];
                Assert.Equal(expected[i, j], product, Precision);
            }
        }
    }

    [Fact(Timeout = 60000)]
    public async Task HierarchicalFedLoRA_CloudAggregatesFullUpdatesInGlobalRankSpace()
    {
        await Task.Yield();

        // Edge 0: B = [1, 0]^T, A = [2, 0] -> diag(2, 0); edge 1: B = [0, 1]^T, A = [0, 1] -> diag(0, 1).
        // Equal weights: dW = diag(1, 0.5), which has rank 2. At global rank 2 the cloud keeps it exactly;
        // averaging zero-padded factors would give avg(B) avg(A) = [0.5, 0.5]^T [1, 0.5] instead.
        var lora = new HierarchicalFedLoRA<double>(modelDim: 100, localRank: 1, globalRank: 2, numAdaptedLayers: 1, layerDim: 2);
        var edges = new Dictionary<int, Vector<double>>
        {
            [0] = new Vector<double>(new[] { 1.0, 0.0, 2.0, 0.0 }),
            [1] = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 }),
        };

        var global = lora.AggregateCloud(edges, null, demoteResult: false);
        var expected = new[,] { { 1.0, 0.0 }, { 0.0, 0.5 } };
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                double product = global[i * 2] * global[4 + j] + global[i * 2 + 1] * global[4 + 2 + j];
                Assert.Equal(expected[i, j], product, Precision);
            }
        }

        // Demotion keeps the leading singular direction: diag(1, 0).
        var local = lora.AggregateCloud(edges, null);
        Assert.Equal(1.0, local[0] * local[2], Precision);
        Assert.Equal(0.0, local[0] * local[3], Precision);
        Assert.Equal(0.0, local[1] * local[2], Precision);
        Assert.Equal(0.0, local[1] * local[3], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task SparseLoRA_SparsifyDeltaKeepsTrueTopKWhenTiesComeFirst()
    {
        await Task.Yield();

        // Delta magnitudes [5, 5, 9], k = (int)(3 * 0.7) = 2. The top-2 by magnitude are 9 and one 5
        // (the first in index order). Scanning in index order and stopping at k would keep both 5s and drop 9.
        var lora = new SparseLoRA<double>(modelDim: 100, sparsityRatio: 0.7);
        var before = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var after = new Vector<double>(new[] { 5.0, -5.0, 9.0 });

        var sparse = lora.SparsifyDelta(before, after);

        Assert.Equal(new[] { 0, 2 }, sparse.Indices);
        Assert.Equal(new[] { 5.0, 9.0 }, sparse.Values);
        Assert.Equal(new[] { 5.0, 0.0, 9.0 }, SparseLoRA<double>.ToDense(sparse).ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task Flame_AdmitsMajorityClusterAndClipsToMedianOfAllDistances()
    {
        await Task.Yield();

        // FLAME (Nguyen et al. 2022), Algorithm 1, first round (G_{t-1} = 0), lambda = 0:
        // honest models [1, 0], [1, 0.1], [1, -0.1], [0.9, 0] and an attacker [-1, 0] (cosine distance ~2).
        // HDBSCAN with min_cluster_size = 5/2 + 1 = 3 admits the 4 honest models.
        // S_t = median of all 5 distances {1, 1.00499, 1.00499, 0.9, 1} = 1, so [1, +-0.1] are scaled by
        // 1 / sqrt(1.01). G_t = plain mean of the clipped admitted models.
        var flame = new FlameAggregationStrategy<double>(noiseMultiplier: 0.0);
        var points = new[] { new[] { 1.0, 0.0 }, new[] { 1.0, 0.1 }, new[] { 1.0, -0.1 }, new[] { 0.9, 0.0 }, new[] { -1.0, 0.0 } };
        var models = new Dictionary<int, Dictionary<string, double[]>>();
        var weights = new Dictionary<int, double>();
        for (int i = 0; i < points.Length; i++)
        {
            models[i] = new Dictionary<string, double[]> { ["w"] = points[i] };
            weights[i] = 1.0;
        }

        var aggregated = flame.Aggregate(models, weights);

        double scale = 1.0 / Math.Sqrt(1.01);
        Assert.Equal((1.0 + 2.0 * scale + 0.9) / 4.0, aggregated["w"][0], Precision);
        Assert.Equal(0.0, aggregated["w"][1], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task FetchSGD_KeepsMomentumAndErrorInSketchSpaceOnTheServer()
    {
        await Task.Yield();

        // FetchSGD (Rothchild et al. 2020), Algorithm 1, with rho = 0.9, eta = 1, k = 1 and one client sending
        // g = [3, 1] twice. Sketches are linear and (3 rows x 1000 columns, 2 coordinates) collision-free
        // under the median, so in vector terms:
        //   round 1: u = g, e = g, Delta = top-1(e) = [3, 0], e = [0, 1]
        //   round 2: u = 0.9 g + g = [5.7, 1.9], e = [0, 1] + u = [5.7, 2.9], Delta = [5.7, 0]
        // (Without server momentum and error accumulation round 2 would return [3, 0] again.)
        var fetch = new FetchSGDCompressor<double>(sketchRows: 3, sketchCols: 1000, topK: 1, momentum: 0.9, learningRate: 1.0);
        var gradient = new[] { 3.0, 1.0 };

        var round1 = fetch.ServerStep(fetch.MergeSketches(new[] { fetch.Sketch(gradient) }), gradient.Length);
        Assert.Equal(3.0, round1[0], Precision);
        Assert.Equal(0.0, round1[1], Precision);

        var round2 = fetch.ServerStep(fetch.MergeSketches(new[] { fetch.Sketch(gradient) }), gradient.Length);
        Assert.Equal(5.7, round2[0], Precision);
        Assert.Equal(0.0, round2[1], Precision);

        // The aggregate is the mean of the client sketches: two clients sending g and 3g recover 2g.
        var mean = fetch.RecoverTopK(
            fetch.MergeSketches(new[] { fetch.Sketch(gradient), fetch.Sketch(new[] { 9.0, 3.0 }) }), gradient.Length);
        Assert.Equal(6.0, mean[0], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task FedKD_AdaptiveMutualDistillationScalesByInverseTaskLossSum()
    {
        await Task.Yield();

        // Wu et al. 2022 (FedKD). One sample, label 0. Teacher logits [0, 0] -> y^t = [0.5, 0.5];
        // student logits [ln 3, 0] -> y^s = [0.75, 0.25].
        // L_t^t = -ln 0.5, L_t^s = -ln 0.75; scale = 1 / (L_t^t + L_t^s).
        // L_d^s = KL(y^t || y^s) * scale, L_d^t = KL(y^s || y^t) * scale.
        // Hidden: H^t = [1, 2], H^s = [1, 1] (identity projection), MSE = 0.5, L_h = 0.5 * scale.
        var fedKd = new FedKDCompressor<double>();
        var teacher = new Tensor<double>(new[] { 0.0, 0.0 }, new[] { 1, 2 });
        var student = new Tensor<double>(new[] { Math.Log(3.0), 0.0 }, new[] { 1, 2 });
        var teacherHidden = new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 1, 2 });
        var studentHidden = new Tensor<double>(new[] { 1.0, 1.0 }, new[] { 1, 2 });

        var losses = fedKd.ComputeMutualDistillationLosses(teacher, student, new[] { 0 }, teacherHidden, studentHidden);

        double teacherTask = -Math.Log(0.5);
        double studentTask = -Math.Log(0.75);
        double scale = 1.0 / (teacherTask + studentTask);
        double klTeacherStudent = 0.5 * Math.Log(0.5 / 0.75) + 0.5 * Math.Log(0.5 / 0.25);
        double klStudentTeacher = 0.75 * Math.Log(0.75 / 0.5) + 0.25 * Math.Log(0.25 / 0.5);

        Assert.Equal(teacherTask, losses.TeacherTaskLoss, Precision);
        Assert.Equal(studentTask, losses.StudentTaskLoss, Precision);
        Assert.Equal(klTeacherStudent * scale, losses.StudentDistillationLoss, Precision);
        Assert.Equal(klStudentTeacher * scale, losses.TeacherDistillationLoss, Precision);
        Assert.Equal(0.5 * scale, losses.HiddenLoss, Precision);
        Assert.Equal(studentTask + (klTeacherStudent + 0.5) * scale, losses.StudentTotalLoss, Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task FedKD_SvdCompressionKeepsFewestSingularValuesReachingEnergyThreshold()
    {
        await Task.Yield();

        // g = diag(3, 1): energies 9 and 1, cumulative ratios 0.9 and 1.0.
        // T(t) grows linearly from 0.95 to 0.98: T(0) = 0.95, T(5 of 10) = 0.965, T(10) = 0.98.
        var fedKd = new FedKDCompressor<double>();
        Assert.Equal(0.95, fedKd.GetEnergyThreshold(0, 10), Precision);
        Assert.Equal(0.965, fedKd.GetEnergyThreshold(5, 10), Precision);
        Assert.Equal(0.98, fedKd.GetEnergyThreshold(10, 10), Precision);

        var gradient = new Matrix<double>(new[,] { { 3.0, 0.0 }, { 0.0, 1.0 } });
        Assert.Equal(2, fedKd.CompressGradient(gradient, 0.95).Rank);

        var rankOne = fedKd.CompressGradient(gradient, 0.85);
        Assert.Equal(1, rankOne.Rank);
        var reconstructed = fedKd.DecompressGradient(rankOne);
        Assert.Equal(3.0, reconstructed[0, 0], Precision);
        Assert.Equal(0.0, reconstructed[0, 1], Precision);
        Assert.Equal(0.0, reconstructed[1, 0], Precision);
        Assert.Equal(0.0, reconstructed[1, 1], Precision);

        // The server averages the reconstructed client gradients.
        var average = fedKd.AggregateCompressedGradients(new Dictionary<int, FedKDCompressedGradient<double>>
        {
            [0] = rankOne,
            [1] = fedKd.CompressGradient(gradient, 0.95),
        });
        Assert.Equal(3.0, average[0, 0], Precision);
        Assert.Equal(0.5, average[1, 1], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task FedAGC_AdaptiveProjectionScalesConflictingGradientEvenForTinyImportance()
    {
        await Task.Yield();

        // Per parameter, (g * imp / imp^2) * imp = g, so a conflicting entry becomes g * (1 - alpha) whatever
        // the importance magnitude. imp = 1e-6 (imp^2 = 1e-12) used to bypass the correction.
        var fedAgc = new AiDotNet.FederatedLearning.ContinualLearning.FedAGCContinualLearning<double>(correctionStrength: 0.5);
        var gradient = new Vector<double>(new[] { 2.0, -4.0, 3.0 });
        var importance = new Vector<double>(new[] { -1e-6, 1.0, 1.0 });
        var strengths = new Vector<double>(new[] { 0.5, 0.25, 0.5 });

        var corrected = fedAgc.ProjectGradientAdaptive(gradient, importance, strengths);
        Assert.Equal(1.0, corrected[0], Precision);
        Assert.Equal(-3.0, corrected[1], Precision);
        Assert.Equal(3.0, corrected[2], Precision);

        // Adaptive strength: base * |imp| / max|imp| on conflicting entries only.
        var adaptive = fedAgc.ComputeAdaptiveCorrectionStrength(gradient, importance);
        Assert.Equal(0.5e-6, adaptive[0], Precision);
        Assert.Equal(0.5, adaptive[1], Precision);
        Assert.Equal(0.0, adaptive[2], Precision);

        // Importance: surrogate L = (theta . x)^2, forward difference 2 (theta . x) x_i + eps x_i^2, squared,
        // summed over the seeded draws and divided by the sample count. With one parameter every draw hits 0.
        var single = new AiDotNet.FederatedLearning.ContinualLearning.FedAGCContinualLearning<double>();
        var fisher = single.ComputeImportance(new Vector<double>(new[] { 3.0 }), new Matrix<double>(new[,] { { 2.0 }, { 1.0 } }));
        double g0 = 2 * 6.0 * 2.0 + 1e-5 * 4.0;
        double g1 = 2 * 3.0 * 1.0 + 1e-5 * 1.0;
        Assert.Equal((g0 * g0 + g1 * g1) / 2.0, fisher[0], 6);
    }

    // FedCIL ACGAN with 2 classes, 1 feature, 1 noise input and 2 hidden units (24 parameters). The generator maps
    // class 0 to +2 and class 1 to -2 (noise weight 0); the discriminator hidden layer is (relu(x), relu(-x)) with
    // the given 2x2 class weights and a zero real/fake head.
    private static Vector<double> FedCilParameters(double[] classifierWeights)
    {
        var values = new List<double>
        {
            1, 0, 0, 1, 0, 0, // generator W1 [onehot0; onehot1; noise] x 2 hidden
            0, 0,             // generator b1
            2, -2,            // generator W2
            0,                // generator b2
            1, -1,            // discriminator Wh
            0, 0,             // discriminator bh
            0, 0,             // adversarial W
            0,                // adversarial b
        };
        values.AddRange(classifierWeights);
        values.AddRange(new[] { 0.0, 0.0 }); // classifier bias
        return new Vector<double>(values.ToArray());
    }

    private static AiDotNet.FederatedLearning.ContinualLearning.FedCILContinualLearning<double> SmallFedCil(int consolidationSteps = 300)
    {
        return new AiDotNet.FederatedLearning.ContinualLearning.FedCILContinualLearning<double>(
            numClasses: 2, featureDim: 1, noiseDim: 1, hiddenDim: 2, batchSize: 32, learningRate: 0.1,
            consistencyWeight: 1.0, consolidationSteps: consolidationSteps, seed: 11);
    }

    [Fact(Timeout = 60000)]
    public async Task FedCIL_ConsistencyLossIsKlFromGlobalToLocalClassifier()
    {
        await Task.Yield();

        // Qi et al. 2023 consistency enforcement. x = 2: global logits (2, 0), local (swapped head) logits (0, 2).
        // With a = sigmoid(2): KL = a ln(a / (1 - a)) + (1 - a) ln((1 - a) / a) = 2 (2a - 1) = 2 tanh(1).
        var fedCil = SmallFedCil();
        Assert.Equal(24, fedCil.ParameterCount);
        var global = fedCil.CreateModel(FedCilParameters(new[] { 1.0, 0.0, 0.0, 1.0 }));
        var local = fedCil.CreateModel(FedCilParameters(new[] { 0.0, 1.0, 1.0, 0.0 }));
        Assert.Equal(FedCilParameters(new[] { 1.0, 0.0, 0.0, 1.0 }).ToArray(), fedCil.GetParameters(global).ToArray());

        var loss = fedCil.ComputeConsistencyLoss(global, local, new Matrix<double>(new[,] { { 2.0 } }));
        Assert.Equal(2.0 * Math.Tanh(1.0), loss, Precision);
        Assert.Equal(0.0, fedCil.ComputeConsistencyLoss(global, global, new Matrix<double>(new[,] { { 2.0 } })), Precision);

        // The generator draws class 0 at +2 and class 1 at -2.
        Assert.Equal(2.0, fedCil.GenerateReplay(global, 0, 3)[1, 0], Precision);
        Assert.Equal(-2.0, fedCil.GenerateReplay(global, 1, 3)[2, 0], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task FedCIL_GenerativeReplayKeepsOldClassWhileLearningNewTask()
    {
        await Task.Yield();

        // Task 2 brings only class 1 (real samples at -2), and the global classifier head is blank. Replaying
        // class 0 from the global generator (+2) teaches the client classifier that +2 is class 0 even though no
        // real class-0 sample is present; training on the new task alone labels +2 as class 1.
        var task2 = new Matrix<double>(new[,] { { -2.0 }, { -2.1 }, { -1.9 } });
        var labels = new[] { 1, 1, 1 };
        var blank = new[] { 0.0, 0.0, 0.0, 0.0 };

        var replaying = SmallFedCil();
        var global = replaying.CreateModel(FedCilParameters(blank));
        var withReplay = replaying.TrainClient(global, task2, labels, new[] { 0 }, steps: 300);
        Assert.Equal(new[] { 0, 1 }, replaying.Classify(withReplay, new Matrix<double>(new[,] { { 2.0 }, { -2.0 } })));

        // The global model is not modified by client training.
        Assert.Equal(FedCilParameters(blank).ToArray(), replaying.GetParameters(global).ToArray());

        var forgetting = SmallFedCil();
        var withoutReplay = forgetting.TrainClient(
            forgetting.CreateModel(FedCilParameters(blank)), task2, labels, Array.Empty<int>(), steps: 300);
        Assert.Equal(1, forgetting.Classify(withoutReplay, new Matrix<double>(new[,] { { 2.0 } }))[0]);
    }

    [Fact(Timeout = 60000)]
    public async Task FedCIL_ConsolidationAveragesClientsThenLearnsEveryClientsClasses()
    {
        await Task.Yield();

        // Client A has learned class 0 only, client B class 1 only; both classifier heads are blank (zero).
        var blank = new[] { 0.0, 0.0, 0.0, 0.0 };

        // Without consolidation steps the server model is the weighted parameter average (weights 1 and 3).
        var averaging = SmallFedCil(consolidationSteps: 0);
        var a = FedCilParameters(blank);
        var b = FedCilParameters(new[] { 4.0, 0.0, 0.0, 4.0 });
        var averaged = averaging.Consolidate(
            new Dictionary<int, AiDotNet.FederatedLearning.ContinualLearning.FedCILGenerativeModel<double>>
            {
                [0] = averaging.CreateModel(a),
                [1] = averaging.CreateModel(b),
            },
            new Dictionary<int, IReadOnlyCollection<int>> { [0] = new[] { 0 }, [1] = new[] { 1 } },
            new Dictionary<int, double> { [0] = 1.0, [1] = 3.0 });
        var averagedParameters = averaging.GetParameters(averaged).ToArray();
        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(0.25 * a[i] + 0.75 * b[i], averagedParameters[i], Precision);
        }

        // Model consolidation: ACGAN steps on samples from each client generator for its own classes teach the
        // global classifier both classes, which neither client classifier knew.
        var consolidating = SmallFedCil();
        var global = consolidating.Consolidate(
            new Dictionary<int, AiDotNet.FederatedLearning.ContinualLearning.FedCILGenerativeModel<double>>
            {
                [0] = consolidating.CreateModel(FedCilParameters(blank)),
                [1] = consolidating.CreateModel(FedCilParameters(blank)),
            },
            new Dictionary<int, IReadOnlyCollection<int>> { [0] = new[] { 0 }, [1] = new[] { 1 } },
            null);
        Assert.Equal(new[] { 0, 1 }, consolidating.Classify(global, new Matrix<double>(new[,] { { 2.0 }, { -2.0 } })));
        Assert.Equal(new[] { 0, 1 }, consolidating.KnownClasses.OrderBy(c => c).ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task FederatedEWC_FisherIsThetaSquaredTimesColumnMeanSquareAndEmptyDataThrows()
    {
        await Task.Yield();

        // F_j = (1/n) sum_i (theta_j x_{i, j mod C})^2 over the first n = min(fisherSamples, rows) rows.
        // theta = [1, 2, 3], C = 2, rows [1, 2] and [3, 4] (third row ignored with fisherSamples = 2):
        // mean x_0^2 = 5, mean x_1^2 = 10 -> F = [1 * 5, 4 * 10, 9 * 5].
        var ewc = new AiDotNet.FederatedLearning.ContinualLearning.FederatedEWC<double>(fisherSamples: 2);
        var data = new Matrix<double>(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 100.0, 100.0 } });
        var fisher = ewc.ComputeImportance(new Vector<double>(new[] { 1.0, 2.0, 3.0 }), data);
        Assert.Equal(new[] { 5.0, 40.0, 45.0 }, fisher.ToArray());

        // Zero rows used to divide by zero and return NaN importances.
        Assert.Throws<ArgumentException>(() => ewc.ComputeImportance(new Vector<double>(new[] { 1.0 }), new Matrix<double>(0, 2)));

        // Weighted average: (1 * [2, 4] + 3 * [6, 8]) / 4 = [5, 7]; non-positive total weight is rejected.
        var aggregated = ewc.AggregateImportance(
            new Dictionary<int, Vector<double>> { [0] = new Vector<double>(new[] { 2.0, 4.0 }), [1] = new Vector<double>(new[] { 6.0, 8.0 }) },
            new Dictionary<int, double> { [0] = 1.0, [1] = 3.0 });
        Assert.Equal(5.0, aggregated[0], Precision);
        Assert.Equal(7.0, aggregated[1], Precision);
        Assert.Throws<ArgumentException>(() => ewc.AggregateImportance(
            new Dictionary<int, Vector<double>> { [0] = new Vector<double>(new[] { 2.0 }) },
            new Dictionary<int, double> { [0] = 0.0 }));
    }

    [Fact(Timeout = 60000)]
    public async Task ContinualLearning_ImportanceAggregationNormalizesOverParticipatingClientWeights()
    {
        await Task.Yield();

        // Weights for clients 0 and 1 are 1 and 3; client 9 is not participating and must not dilute the
        // average. Client 2 has no weight entry and counts as 1. Expected: (1*[4] + 3*[8] + 1*[0]) / 5 = [5.6].
        var importances = new Dictionary<int, Vector<double>>
        {
            [0] = new Vector<double>(new[] { 4.0 }),
            [1] = new Vector<double>(new[] { 8.0 }),
            [2] = new Vector<double>(new[] { 0.0 }),
        };
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 3.0, [9] = 100.0 };

        var replay = new AiDotNet.FederatedLearning.ContinualLearning.FederatedExperienceReplay<double>();
        var agc = new AiDotNet.FederatedLearning.ContinualLearning.FedAGCContinualLearning<double>();
        var cil = new AiDotNet.FederatedLearning.ContinualLearning.FedCILContinualLearning<double>();
        Assert.Equal(5.6, replay.AggregateImportance(importances, weights)[0], Precision);
        Assert.Equal(5.6, agc.AggregateImportance(importances, weights)[0], Precision);
        Assert.Equal(5.6, cil.AggregateImportance(importances, weights)[0], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task FederatedOrthogonalProjection_RemovesComponentAlongThresholdedImportance()
    {
        await Task.Yield();

        // e = importance above the threshold 0.5 -> [2, 0, 1] from [2, 0.1, 1]; g = [1, 5, 3].
        // g·e = 5, e·e = 5, scale 1 -> g - e = [-1, 5, 2] (entry 1 keeps its gradient).
        var projection = new AiDotNet.FederatedLearning.ContinualLearning.FederatedOrthogonalProjection<double>(projectionThreshold: 0.5);
        var projected = projection.ProjectGradient(new Vector<double>(new[] { 1.0, 5.0, 3.0 }), new Vector<double>(new[] { 2.0, 0.1, 1.0 }));
        Assert.Equal(-1.0, projected[0], Precision);
        Assert.Equal(5.0, projected[1], Precision);
        Assert.Equal(2.0, projected[2], Precision);

        // Importance: theta_j^2 * mean_i x_{i, j mod C}^2 over all rows; rows [1, 2], [3, 4] -> [5, 10].
        var importance = projection.ComputeImportance(new Vector<double>(new[] { 1.0, 2.0, 3.0 }), new Matrix<double>(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }));
        Assert.Equal(new[] { 5.0, 40.0, 45.0 }, importance.ToArray());

        // Zero rows used to divide by zero and return NaN importances.
        Assert.Throws<ArgumentException>(() => projection.ComputeImportance(new Vector<double>(new[] { 1.0 }), new Matrix<double>(0, 2)));

        // Aggregation keeps the elementwise max of weighted importances: max(1*[1, 4], 2*[3, 1]) = [6, 4].
        var aggregated = projection.AggregateImportance(
            new Dictionary<int, Vector<double>> { [0] = new Vector<double>(new[] { 1.0, 4.0 }), [1] = new Vector<double>(new[] { 3.0, 1.0 }) },
            new Dictionary<int, double> { [0] = 1.0, [1] = 2.0 });
        Assert.Equal(new[] { 6.0, 4.0 }, aggregated.ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task FedDF_AveragesLogitsAndDistillsWithKlGradientThroughProjection()
    {
        await Task.Yield();

        // Lin et al. 2020, Algorithm 1. d = 4 -> 2 outputs, C = 2 columns, W[j, k] = theta[j * 2 + k].
        // theta = [1, 0, 0, 1] (W = I), x = [1, 0]: client logits are [1, 0].
        var fedDf = new AiDotNet.FederatedLearning.Distillation.FedDFDistillation<double>(
            temperature: 3.0, serverDistillationEpochs: 1, serverLearningRate: 1.0);
        var publicData = new Matrix<double>(new[,] { { 1.0, 0.0 } });
        var theta = new Vector<double>(new[] { 1.0, 0.0, 0.0, 1.0 });
        var logits = fedDf.ExtractKnowledge(theta, publicData);
        Assert.Equal(1.0, logits[0, 0], Precision);
        Assert.Equal(0.0, logits[0, 1], Precision);

        // The ensemble is the mean of client logits: ([1, 0] + [-1, 2]) / 2 = [0, 1].
        var ensemble = fedDf.AggregateKnowledge(new Dictionary<int, Matrix<double>>
        {
            [0] = logits,
            [1] = new Matrix<double>(new[,] { { -1.0, 2.0 } }),
        }, null);
        Assert.Equal(0.0, ensemble[0, 0], Precision);
        Assert.Equal(1.0, ensemble[0, 1], Precision);

        // tau = 2: p = softmax([0.5, 0]), q = softmax([0, 0.5]), p - q = [a, -a] with a = tanh(0.25).
        // dL/dW = (p - q)^T x / (n tau) = [[a / 2, 0], [-a / 2, 0]] -> theta = [1 - a / 2, 0, a / 2, 1].
        var updated = fedDf.ApplyKnowledge(theta, ensemble, publicData, temperature: 2.0);
        double a = Math.Tanh(0.25);
        Assert.Equal(1.0 - a / 2.0, updated[0], Precision);
        Assert.Equal(0.0, updated[1], Precision);
        Assert.Equal(a / 2.0, updated[2], Precision);
        Assert.Equal(1.0, updated[3], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task FedGEN_GeneratorLearnsLabelWeightedEnsembleAndDistillsIntoClientHead()
    {
        await Task.Yield();

        // Zhu et al. 2021. Two classes, latent dimension 2. Client A's head is the identity (class = argmax z) and
        // it only has class-0 data; client B's head swaps the classes and only has class-1 data. With the
        // label-count weights alpha_k^y the generator must satisfy A for y = 0 and B for y = 1, so both classes
        // end up with z[0] > z[1]. (An unweighted ensemble sums to logits z0 + z1 for both classes and cannot
        // separate them.) The old FedGEN shared parameter statistics and had no generator at all.
        var fedGen = new AiDotNet.FederatedLearning.Distillation.FedGENDistillation<double>(
            numClasses: 2, featureDim: 2, generatorEpochs: 400, generatorLearningRate: 0.1, seed: 3,
            noiseDim: 4, generatorHiddenDim: 16, batchSize: 32, distillationEpochs: 400, distillationLearningRate: 0.2);
        Assert.Equal(6, fedGen.HeadParameterCount);

        var identityHead = new Vector<double>(new[] { 1.0, 0.0, 0.0, 1.0, 0.0, 0.0 });
        var swappedHead = new Vector<double>(new[] { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0 });
        var knowledgeA = fedGen.ExtractHeadKnowledge(identityHead, new[] { 10, 0 });
        Assert.Equal(10.0, knowledgeA[0, 3], Precision);
        var global = fedGen.AggregateKnowledge(new Dictionary<int, Matrix<double>>
        {
            [0] = knowledgeA,
            [1] = fedGen.ExtractHeadKnowledge(swappedHead, new[] { 0, 10 }),
        }, null);
        Assert.Equal(1, global.Rows);
        Assert.Equal(fedGen.GeneratorParameterCount + 2, global.Columns);
        Assert.Equal(0.5, global[0, global.Columns - 2], Precision);

        for (int label = 0; label < 2; label++)
        {
            var features = fedGen.GenerateFeatures(label, 50);
            int agreeing = 0;
            for (int i = 0; i < features.Rows; i++)
            {
                if (features[i, 0] > features[i, 1])
                {
                    agreeing++;
                }
            }

            Assert.True(agreeing >= 45, $"label {label}: only {agreeing}/50 generated samples have z[0] > z[1]");
        }

        // Distillation: train a second generator against two identity heads that both see every class (class
        // y -> large z_y). A client whose head starts swapped (and carries two body parameters that must not
        // change) learns to classify the generated features of each class as its label.
        var distiller = new AiDotNet.FederatedLearning.Distillation.FedGENDistillation<double>(
            numClasses: 2, featureDim: 2, generatorEpochs: 400, generatorLearningRate: 0.1, seed: 5,
            noiseDim: 4, generatorHiddenDim: 16, batchSize: 32, distillationEpochs: 400, distillationLearningRate: 0.2);
        var identityGlobal = distiller.AggregateKnowledge(new Dictionary<int, Matrix<double>>
        {
            [0] = distiller.ExtractHeadKnowledge(identityHead, new[] { 10, 10 }),
            [1] = distiller.ExtractHeadKnowledge(identityHead, new[] { 10, 10 }),
        }, null);
        var client = new Vector<double>(new[] { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 7.0, 8.0 });
        var updated = distiller.ApplyKnowledge(client, identityGlobal, null, 1.0);
        Assert.Equal(8, updated.Length);
        Assert.Equal(7.0, updated[6], Precision);
        Assert.Equal(8.0, updated[7], Precision);

        // Generated class-1 and class-0 features now map to their labels under the distilled head on average.
        for (int label = 0; label < 2; label++)
        {
            var features = distiller.GenerateFeatures(label, 50);
            int correct = 0;
            for (int i = 0; i < features.Rows; i++)
            {
                double s0 = updated[0] * features[i, 0] + updated[1] * features[i, 1] + updated[4];
                double s1 = updated[2] * features[i, 0] + updated[3] * features[i, 1] + updated[5];
                if ((s1 > s0) == (label == 1))
                {
                    correct++;
                }
            }

            Assert.True(correct >= 45, $"label {label}: {correct}/50 generated samples classified correctly");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task FedMD_DigestStepFollowsSquaredErrorGradientThroughProjection()
    {
        await Task.Yield();

        // d = 10 -> one output; C = 2 columns read theta[0..1]. tau = 2, alpha = 1, lr = 1, one epoch.
        var fedMd = new AiDotNet.FederatedLearning.Distillation.FedMDDistillation<double>(
            temperature: 2.0, distillationAlpha: 1.0, distillationEpochs: 1, learningRate: 1.0);
        var theta = new Vector<double>(new[] { 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
        var publicData = new Matrix<double>(new[,] { { 1.0, 1.0 } });

        // z = 1 + 2 = 3, scaled by 1 / tau -> 1.5.
        Assert.Equal(1.5, fedMd.ExtractKnowledge(theta, publicData)[0, 0], Precision);

        // Consensus 0.5: grad W = alpha (1.5 - 0.5) x / (n tau) = [0.5, 0.5] -> theta[0..1] = [0.5, 1.5].
        var updated = fedMd.ApplyKnowledge(theta, new Matrix<double>(new[,] { { 0.5 } }), publicData, 1.0);
        Assert.Equal(0.5, updated[0], Precision);
        Assert.Equal(1.5, updated[1], Precision);
        Assert.Equal(0.0, updated[2], Precision);

        // Consensus is the normalized weighted mean: (1 * 2 + 3 * 6) / 4 = 5.
        var consensus = fedMd.AggregateKnowledge(new Dictionary<int, Matrix<double>>
        {
            [0] = new Matrix<double>(new[,] { { 2.0 } }),
            [1] = new Matrix<double>(new[,] { { 6.0 } }),
        }, new Dictionary<int, double> { [0] = 1.0, [1] = 3.0 });
        Assert.Equal(5.0, consensus[0, 0], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task DataShapley_AveragesOverRoundsActuallySampledWhenConvergingEarly()
    {
        await Task.Yield();

        // One client [3, 4] against global [0, 0]: every permutation gives marginal -||[3, 4]|| = -5. The
        // estimate converges at the round-30 check, so the mean over the 31 sampled rounds is -5; dividing by
        // the configured 100 rounds gave -1.55. A single score has zero range, so it is not min-max scaled.
        var options = new AiDotNet.Models.Options.ContributionEvaluationOptions { SamplingRounds = 100 };
        var evaluator = new AiDotNet.FederatedLearning.Fairness.DataShapleyEvaluator<double>(options);
        var clients = new Dictionary<int, Tensor<double>> { [7] = new Tensor<double>(new[] { 3.0, 4.0 }, new[] { 2 }) };
        var scores = evaluator.EvaluateContributions(clients, new Tensor<double>(new[] { 2 }), new Dictionary<int, List<Tensor<double>>>());
        Assert.Equal(-5.0, scores[7], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task ShapleyValue_SampledPathUsesTheSameEmptyCoalitionValueAsEnumeration()
    {
        await Task.Yield();

        // Identical clients [1] and global [1]: v(S) = 0 for every non-empty coalition and the empty
        // coalition is the zero model, v({}) = -1.
        var global = new Tensor<double>(new[] { 1.0 }, new[] { 1 });

        // Enumeration (n = 3): each client gets (v(N) - v({})) / n = 1/3; equal scores are not rescaled.
        var exact = new AiDotNet.FederatedLearning.Fairness.ShapleyValueEvaluator<double>(new AiDotNet.Models.Options.ContributionEvaluationOptions());
        var small = new Dictionary<int, Tensor<double>>();
        for (int i = 0; i < 3; i++)
        {
            small[i] = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        }

        foreach (var score in exact.EvaluateContributions(small, global, new Dictionary<int, List<Tensor<double>>>()).Values)
        {
            Assert.Equal(1.0 / 3.0, score, Precision);
        }

        // Sampling (n = 16, one permutation): only the first client has a non-zero marginal, 0 - (-1) = 1, so
        // after min-max scaling exactly one client scores 1 and the rest 0. Treating v({}) as 0 made every
        // marginal 0.
        var sampled = new AiDotNet.FederatedLearning.Fairness.ShapleyValueEvaluator<double>(
            new AiDotNet.Models.Options.ContributionEvaluationOptions { SamplingRounds = 1 });
        var large = new Dictionary<int, Tensor<double>>();
        for (int i = 0; i < 16; i++)
        {
            large[i] = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        }

        var scores = sampled.EvaluateContributions(large, global, new Dictionary<int, List<Tensor<double>>>()).Values.ToList();
        Assert.Equal(1, scores.Count(s => Math.Abs(s - 1.0) < 1e-12));
        Assert.Equal(15, scores.Count(s => Math.Abs(s) < 1e-12));
    }

    [Fact(Timeout = 60000)]
    public async Task GraphNodeGenerator_TrainingFitsNodesFromTheirNeighbors()
    {
        await Task.Yield();

        // FedSage+ NeighGen: the generator learns node features from mean neighbor features. Two connected
        // nodes with features [1, 0] and [0, 1]: the generator must map [0, 1] -> [1, 0] and [1, 0] -> [0, 1].
        // Training used to compute the loss gradient without ever applying it, so the output never changed.
        var generator = new AiDotNet.FederatedLearning.Graph.GraphNodeGenerator<double>(featureDim: 2, hiddenDim: 16, learningRate: 0.1);
        var adjacency = new Tensor<double>(new[] { 0.0, 1.0, 1.0, 0.0 }, new[] { 4 });
        var features = new Tensor<double>(new[] { 1.0, 0.0, 0.0, 1.0 }, new[] { 4 });

        generator.Train(adjacency, features, featureDim: 2, epochs: 3000);

        var fromNode1 = generator.Generate(new Tensor<double>(new[] { 0.0, 1.0 }, new[] { 2 }));
        var fromNode0 = generator.Generate(new Tensor<double>(new[] { 1.0, 0.0 }, new[] { 2 }));
        Assert.Equal(1.0, fromNode1[0], 2);
        Assert.Equal(0.0, fromNode1[1], 2);
        Assert.Equal(0.0, fromNode0[0], 2);
        Assert.Equal(1.0, fromNode0[1], 2);
    }

    [Fact(Timeout = 60000)]
    public async Task SubgraphExpander_AverageFallbackFillsOnlyPseudoNodesWithoutCrossClientFeatures()
    {
        await Task.Yield();

        // 10 nodes with one feature 0..9 -> 2 pseudo-nodes. One cross-client row [42] fills pseudo-node 0 and the
        // average 4.5 fills pseudo-node 1. The fallback used to restart at pseudo-node 0, overwriting the
        // cross-client row with the average and leaving pseudo-node 1 at zero.
        var options = new AiDotNet.Models.Options.FederatedGraphOptions { PseudoNodeStrategy = AiDotNet.Models.Options.PseudoNodeStrategy.GeneratorBased };
        var expander = new AiDotNet.FederatedLearning.Graph.SubgraphExpander<double>(options);
        var features = new double[10];
        for (int i = 0; i < 10; i++)
        {
            features[i] = i;
        }

        var expanded = expander.Expand(
            new Tensor<double>(new[] { 100 }),
            new Tensor<double>(features, new[] { 10 }),
            new Dictionary<int, Tensor<double>> { [3] = new Tensor<double>(new[] { 42.0 }, new[] { 1 }) });

        Assert.Equal(2, expanded.PseudoNodeCount);
        Assert.Equal(12, expanded.NodeFeatures.Length);
        Assert.Equal(9.0, expanded.NodeFeatures[9], Precision);
        Assert.Equal(42.0, expanded.NodeFeatures[10], Precision);
        Assert.Equal(4.5, expanded.NodeFeatures[11], Precision);

        // The original adjacency block is preserved and each pseudo-node links symmetrically to original nodes.
        Assert.Equal(144, expanded.Adjacency.Length);
        for (int p = 10; p < 12; p++)
        {
            for (int j = 0; j < 12; j++)
            {
                Assert.Equal(expanded.Adjacency[p * 12 + j], expanded.Adjacency[j * 12 + p]);
            }
        }
    }

    [Fact(Timeout = 60000)]
    public async Task SubgraphFederatedTrainer_RoundWithMultiFeatureNodesScalesModelByLabelError()
    {
        await Task.Yield();

        // Two linked nodes with flattened features [1, 2 | 3, 4]. Each embedding is the mean summed features of
        // the neighbors: node 0 -> 7, node 1 -> 3. Labels [7, 0] give squared errors 0 and 9, mean 4.5, so the
        // local update scales every parameter by 1 - 0.01 * 4.5 * 0.01. The old embedding loop took the node
        // count from the flattened feature length and indexed past the adjacency.
        var options = new AiDotNet.Models.Options.FederatedGraphOptions
        {
            PseudoNodeStrategy = AiDotNet.Models.Options.PseudoNodeStrategy.None,
            HiddenDimension = 2,
        };
        var trainer = new AiDotNet.FederatedLearning.Graph.SubgraphFederatedTrainer<double>(
            options, new AiDotNet.FederatedLearning.Graph.FedGnnAggregationStrategy<double>());
        trainer.RegisterSubgraph(
            0,
            new Tensor<double>(new[] { 0.0, 1.0, 1.0, 0.0 }, new[] { 4 }),
            new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 4 }),
            new Tensor<double>(new[] { 7.0, 0.0 }, new[] { 2 }));

        var before = trainer.GetGlobalModel().ToArray();
        var after = trainer.ExecuteRound(0).ToArray();

        double factor = 1.0 - 0.01 * 4.5 * 0.01;
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
        {
            Assert.Equal(before[i] * factor, after[i], Precision);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task ArithmeticSecretSharing_MultiplyUsesTripleMatchingPartyCountAndShape()
    {
        await Task.Yield();

        // Triples pre-generated for 3 parties and shape [1] cannot mask a 2-party product of shape [3]. The old
        // code appended fresh triples but still used the mismatched one at the current index, failing on the
        // share-count check (or indexing past the triple for larger shapes).
        var ss = new AiDotNet.FederatedLearning.MPC.ArithmeticSecretSharing<double>(numberOfParties: 3, preGenerateTriples: 0, seed: 7);
        ss.PreGenerateBeaverTriples(new[] { 1 }, 1);

        var x = new Tensor<double>(new[] { 2.0, -3.0, 0.5 }, new[] { 3 });
        var y = new Tensor<double>(new[] { 4.0, 5.0, -6.0 }, new[] { 3 });
        var product = ss.Reconstruct(ss.SecureMultiply(ss.Share(x, 2), ss.Share(y, 2)));

        Assert.Equal(8.0, product[0], Precision);
        Assert.Equal(-15.0, product[1], Precision);
        Assert.Equal(-3.0, product[2], Precision);
    }

    [Fact(Timeout = 60000)]
    public async Task FedBABU_MaskHeadGradientsFreezesTheSameLayersExtractHeadReturns()
    {
        await Task.Yield();

        // Head fraction 0.5 of four layers: ordinal order a, b, c, d -> head is {c, d}. The gradients are
        // inserted in another order; masking used insertion order and zeroed {a, b} instead.
        var babu = new AiDotNet.FederatedLearning.Personalization.FedBABUPersonalization<double>(headFraction: 0.5);
        var gradients = new Dictionary<string, double[]>
        {
            ["d"] = new[] { 1.0 },
            ["c"] = new[] { 2.0 },
            ["b"] = new[] { 3.0 },
            ["a"] = new[] { 4.0 },
        };

        var head = babu.ExtractHead(gradients);
        Assert.Equal(new[] { "c", "d" }, head.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());

        var masked = babu.MaskHeadGradients(gradients);
        Assert.Equal(new[] { 0.0 }, masked["c"]);
        Assert.Equal(new[] { 0.0 }, masked["d"]);
        Assert.Equal(new[] { 3.0 }, masked["b"]);
        Assert.Equal(new[] { 4.0 }, masked["a"]);
    }

    [Fact(Timeout = 60000)]
    public async Task PrototypeFederatedGraphLearning_GlobalPrototypeIsSampleCountWeighted()
    {
        await Task.Yield();

        // FedProto (Tan et al. 2022): p_c = sum_i |D_ic| p_ic / sum_i |D_ic|. Client 0 has 3 nodes of class 0 at
        // [1, 1], client 1 has 1 node at [5, 9]: p_0 = (3 [1, 1] + [5, 9]) / 4 = [2, 3]. The unweighted mean
        // gave [3, 5].
        var learner = new AiDotNet.FederatedLearning.Graph.PrototypeFederatedGraphLearning<double>(
            new AiDotNet.Models.Options.FederatedGraphOptions(), prototypeDim: 2, numClasses: 1);

        var protos0 = learner.ComputePrototypes(
            new Tensor<double>(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, new[] { 6 }),
            new Tensor<double>(new[] { 0.0, 0.0, 0.0 }, new[] { 3 }),
            embeddingDim: 2,
            out var counts0);
        Assert.Equal(3, counts0[0]);

        learner.RegisterClientPrototypes(0, protos0, counts0);
        learner.RegisterClientPrototypes(
            1,
            new Dictionary<int, Tensor<double>> { [0] = new Tensor<double>(new[] { 5.0, 9.0 }, new[] { 2 }) },
            new Dictionary<int, int> { [0] = 1 });

        var global = learner.AggregatePrototypes();
        Assert.Equal(2.0, global[0][0], Precision);
        Assert.Equal(3.0, global[0][1], Precision);

        // A prototype without a positive sample count is rejected.
        Assert.Throws<ArgumentException>(() => learner.RegisterClientPrototypes(
            2,
            new Dictionary<int, Tensor<double>> { [0] = new Tensor<double>(new[] { 1.0, 1.0 }, new[] { 2 }) },
            new Dictionary<int, int>()));
    }

    // ---- Narrow-T (Half) regressions: Half overflows at 65504 and a 0/1 count in Half stops at 2048. ----

    private static readonly AiDotNet.Tensors.Interfaces.INumericOperations<Half> HalfOps =
        MathHelper.GetNumericOperations<Half>();

    private static Half[] HalfArray(params double[] values)
    {
        var result = new Half[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = HalfOps.FromDouble(values[i]);
        }

        return result;
    }

    [Fact(Timeout = 60000)]
    public async Task DPFedLoRA_HalfClipUsesTrueNormAboveHalfSquareRange()
    {
        await Task.Yield();

        // DP-FedAvg clips the update [300, 400] (norm 500) to C = 1: [0.6, 0.8]. 300^2 + 400^2 = 250000
        // overflows Half, so a norm formed from raw squares is infinite and clipped the update to 0.
        var strategy = new DPFedLoRA<Half>(
            modelDim: 100, rank: 1, noiseMultiplier: 0.0, clipNorm: 1.0,
            numAdaptedLayers: 1, layerInputDim: 1, layerOutputDim: 1, seed: 42);
        var clients = new Dictionary<int, Vector<Half>> { [0] = new Vector<Half>(HalfArray(300, 400)) };

        var aggregated = strategy.AggregateAdapters(clients, null);

        Assert.Equal(0.6, HalfOps.ToDouble(aggregated[0]), 3);
        Assert.Equal(0.8, HalfOps.ToDouble(aggregated[1]), 3);
    }

    [Fact(Timeout = 60000)]
    public async Task DataFreeFCL_HalfImportanceUsesRmsAboveHalfSquareRange()
    {
        await Task.Yield();

        // Importance = min(|theta| / rms(theta), 10); rms([300, 400]) = 500 / sqrt(2) = 353.553, giving
        // [0.848528, 1.131371]. The raw sum of squares overflowed Half and made every importance 0.
        var fcl = new AiDotNet.FederatedLearning.ContinualLearning.DataFreeFCL<Half>();
        var importance = fcl.ComputeImportance(new Vector<Half>(HalfArray(300, 400)), new Matrix<Half>(1, 1));

        Assert.Equal(300 / (500 / Math.Sqrt(2)), HalfOps.ToDouble(importance[0]), 2);
        Assert.Equal(400 / (500 / Math.Sqrt(2)), HalfOps.ToDouble(importance[1]), 2);
    }

    [Fact(Timeout = 60000)]
    public async Task Flame_HalfModelsAboveHalfSquareRangeAreClusteredAndClipped()
    {
        await Task.Yield();

        // FLAME, first round (G = 0), lambda = 0: [300, 400], [600, 800], [150, 200] all point the same way, so
        // the cluster admits all 3. Distances {500, 1000, 250}, median S = 500: [600, 800] clips to [300, 400].
        // G = ([300, 400] + [300, 400] + [150, 200]) / 3 = [250, 333.33]. The squares (up to 10^6) overflow Half.
        var flame = new FlameAggregationStrategy<Half>(noiseMultiplier: 0.0);
        var models = new Dictionary<int, Dictionary<string, Half[]>>
        {
            [0] = new Dictionary<string, Half[]> { ["w"] = HalfArray(300, 400) },
            [1] = new Dictionary<string, Half[]> { ["w"] = HalfArray(600, 800) },
            [2] = new Dictionary<string, Half[]> { ["w"] = HalfArray(150, 200) },
        };
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0 };

        var aggregated = flame.Aggregate(models, weights);

        Assert.InRange(HalfOps.ToDouble(aggregated["w"][0]), 249.5, 250.5);
        Assert.InRange(HalfOps.ToDouble(aggregated["w"][1]), 333.0, 333.75);
    }

    // Star graph: node 0 is linked to every other node, node 1 has feature 3000 and the rest 0.
    private static (Tensor<Half> Adjacency, Tensor<Half> Features) HalfStarGraph(int nodes)
    {
        var adjacency = new Half[nodes * nodes];
        var one = HalfOps.One;
        for (int j = 1; j < nodes; j++)
        {
            adjacency[j] = one;
            adjacency[j * nodes] = one;
        }

        var features = new Half[nodes];
        features[1] = HalfOps.FromDouble(3000);
        return (new Tensor<Half>(adjacency, new[] { nodes * nodes }), new Tensor<Half>(features, new[] { nodes }));
    }

    [Fact(Timeout = 60000)]
    public async Task GraphNodeGenerator_HalfMeanNeighborFeatureDividesByTrueDegreeAbove2048()
    {
        await Task.Yield();

        // FedSage+ NeighGen input = mean neighbor features. Node 0 has 4096 neighbors summing to 3000:
        // 3000 / 4096 = 0.732. A degree summed in Half stops at 2048 and gave 1.465.
        const int nodes = 4097;
        var (adjacency, features) = HalfStarGraph(nodes);
        var generator = new AiDotNet.FederatedLearning.Graph.GraphNodeGenerator<Half>(featureDim: 1);

        var (inputs, _, _) = generator.BuildTrainingPairs(adjacency, features, nodes, 1);

        Assert.NotNull(inputs);
        if (inputs is not null)
        {
            Assert.Equal(3000.0 / 4096.0, HalfOps.ToDouble(inputs[0, 0]), 3);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task SubgraphFederatedTrainer_HalfEmbeddingDividesByTrueNeighborCountAbove2048()
    {
        await Task.Yield();

        // Mean aggregation over the 4096 neighbors of node 0: 3000 / 4096 = 0.732. Counting neighbors in Half
        // stops at 2048 and gave 1.465.
        const int nodes = 4097;
        var (adjacency, features) = HalfStarGraph(nodes);
        var trainer = new AiDotNet.FederatedLearning.Graph.SubgraphFederatedTrainer<Half>(
            new AiDotNet.Models.Options.FederatedGraphOptions(),
            new AiDotNet.FederatedLearning.Graph.FedGnnAggregationStrategy<Half>());

        var embeddings = trainer.ComputeNodeEmbeddings(
            new AiDotNet.FederatedLearning.Graph.ExpandedSubgraph<Half> { Adjacency = adjacency, NodeFeatures = features });

        Assert.Equal(3000.0 / 4096.0, HalfOps.ToDouble(embeddings[0]), 3);
    }

    [Fact(Timeout = 60000)]
    public async Task PrototypicalContribution_HalfMagnitudeAboveHalfMaxStaysFinite()
    {
        await Task.Yield();

        // Both prototypes point along the global direction, so the larger one scores higher and min-max scaling
        // gives 1 and 0. ||[50000, 50000]|| = 70711 is above Half.MaxValue; forming it in Half gave infinity
        // and NaN scores.
        var evaluator = new AiDotNet.FederatedLearning.Fairness.PrototypicalContributionEvaluator<Half>(
            new AiDotNet.Models.Options.ContributionEvaluationOptions());
        var clients = new Dictionary<int, Tensor<Half>>
        {
            [0] = new Tensor<Half>(HalfArray(50000, 50000), new[] { 2 }),
            [1] = new Tensor<Half>(HalfArray(10000, 10000), new[] { 2 }),
        };

        var scores = evaluator.EvaluateContributions(
            clients, new Tensor<Half>(HalfArray(1, 1), new[] { 2 }), new Dictionary<int, List<Tensor<Half>>>());

        Assert.Equal(1.0, scores[0], 6);
        Assert.Equal(0.0, scores[1], 6);
    }

    [Fact(Timeout = 60000)]
    public async Task ShapleyValue_HalfCoalitionDistanceAboveHalfMaxStaysFinite()
    {
        await Task.Yield();

        // Exact Shapley, global 0: v({}) = 0, v({0}) = -||[50000, 50000]|| = -70711, v({1}) = 0,
        // v({0, 1}) = -35355. phi_0 = (-70711 - 35355) / 2 < phi_1 = 35355 / 2, so min-max gives 0 and 1.
        // The distance 70711 overflowed Half to -infinity and the marginals became NaN.
        var evaluator = new AiDotNet.FederatedLearning.Fairness.ShapleyValueEvaluator<Half>(
            new AiDotNet.Models.Options.ContributionEvaluationOptions());
        var clients = new Dictionary<int, Tensor<Half>>
        {
            [0] = new Tensor<Half>(HalfArray(50000, 50000), new[] { 2 }),
            [1] = new Tensor<Half>(HalfArray(0, 0), new[] { 2 }),
        };

        var scores = evaluator.EvaluateContributions(
            clients, new Tensor<Half>(HalfArray(0, 0), new[] { 2 }), new Dictionary<int, List<Tensor<Half>>>());

        Assert.Equal(0.0, scores[0], 6);
        Assert.Equal(1.0, scores[1], 6);
    }

    [Fact(Timeout = 60000)]
    public async Task ShapleyValue_HalfSampledPrefixMeansDoNotOverflowWhenTheMeanFits()
    {
        await Task.Yield();

        // 16 identical clients [40000] and global [40000]: every coalition mean is 40000 (distance 0) and
        // v({}) = -40000, so one permutation gives the first client marginal 40000 and the rest ~0. After
        // min-max scaling one client scores 1 and the rest ~0. A running sum in Half reached infinity at the
        // second client and turned the marginals into NaN.
        var evaluator = new AiDotNet.FederatedLearning.Fairness.ShapleyValueEvaluator<Half>(
            new AiDotNet.Models.Options.ContributionEvaluationOptions { SamplingRounds = 1 });
        var clients = new Dictionary<int, Tensor<Half>>();
        for (int i = 0; i < 16; i++)
        {
            clients[i] = new Tensor<Half>(HalfArray(40000), new[] { 1 });
        }

        var scores = evaluator.EvaluateContributions(
            clients, new Tensor<Half>(HalfArray(40000), new[] { 1 }), new Dictionary<int, List<Tensor<Half>>>()).Values.ToList();

        Assert.Equal(1, scores.Count(s => s > 0.99));
        Assert.Equal(15, scores.Count(s => s >= 0 && s < 0.01));
    }

    [Fact(Timeout = 60000)]
    public async Task FedGEN_GeneratorObjectiveMatchesHandComputedTeacherStudentAndDiversityTerms()
    {
        await Task.Yield();

        // FedGen (Zhu et al. 2021; official serverpFedGen.train_generator with DiversityLoss('l1')). C = 2, L = 1,
        // B = 2, noise dim 1. Heads h_0 = [1, -1] z and h_1 = [3, 0] z (zero bias); alpha_0 = [0.75, 0.5] and
        // alpha_1 = [0.25, 0.5] per class; labels [0, 1]; z = [1, -1]; eps = [0, 1]; server head = 0.
        // Ensemble logits: row 0 = 0.75 [1, -1] + 0.25 [3, 0] = [1.5, -0.75]; row 1 = 0.5 [-1, 1] + 0.5 [-3, 0]
        // = [-2, 0.5]. Teacher CE = (log(1 + e^-2.25) + log(1 + e^-2.5)) / 2.
        // Weighted client CE = (0.75 log(1 + e^-2) + 0.25 log(1 + e^-3) + 0.5 log(1 + e^-2) + 0.5 log(1 + e^-3)) / 2.
        // KL(p || uniform) averaged over the 4 entries; diversity = exp(-mean([0, 1*2, 1*2, 0])) = e^-1.
        // dJ/dz (alpha = 1, beta = 0, eta = 1): teacher row 0 = -1.125 p_01, row 1 = 1.25 p_10; diversity
        // -/+ 0.5 e^-1.
        var fedGen = new AiDotNet.FederatedLearning.Distillation.FedGENDistillation<double>(
            numClasses: 2, featureDim: 1, generatorEpochs: 0, noiseDim: 1, generatorHiddenDim: 2, batchSize: 2);
        Assert.Equal(1.0, fedGen.TeacherLossWeight);
        Assert.Equal(0.0, fedGen.StudentLossWeight);
        Assert.Equal(1.0, fedGen.DiversityLossWeight);

        var heads = new[]
        {
            (new Tensor<double>(new[] { 1.0, -1.0 }, new[] { 2, 1 }), new Tensor<double>(new[] { 0.0, 0.0 }, new[] { 2 })),
            (new Tensor<double>(new[] { 3.0, 0.0 }, new[] { 2, 1 }), new Tensor<double>(new[] { 0.0, 0.0 }, new[] { 2 })),
        };
        var alpha = new[] { new[] { 0.75, 0.5 }, new[] { 0.25, 0.5 } };
        var server = (new Tensor<double>(new[] { 0.0, 0.0 }, new[] { 2, 1 }), new Tensor<double>(new[] { 0.0, 0.0 }, new[] { 2 }));
        var labels = new[] { 0, 1 };
        var latent = new Tensor<double>(new[] { 1.0, -1.0 }, new[] { 2, 1 });
        var noise = new[] { 0.0, 1.0 };

        var result = fedGen.GeneratorObjective(heads, alpha, server, labels, latent, noise);

        double ce225 = Math.Log(1 + Math.Exp(-2.25));
        double ce25 = Math.Log(1 + Math.Exp(-2.5));
        double p00 = 1 / (1 + Math.Exp(-2.25));
        double p01 = 1 - p00;
        double p11 = 1 / (1 + Math.Exp(-2.5));
        double p10 = 1 - p11;
        double kl = (p00 * Math.Log(2 * p00) + p01 * Math.Log(2 * p01) + p10 * Math.Log(2 * p10) + p11 * Math.Log(2 * p11)) / 4;
        Assert.Equal((ce225 + ce25) / 2, result.Teacher, Precision);
        Assert.Equal(kl, result.Student, Precision);
        Assert.Equal(Math.Exp(-1), result.Diversity, Precision);
        Assert.Equal(-1.125 * p01 - 0.5 * Math.Exp(-1), result.GradLatent[0, 0], Precision);
        Assert.Equal(1.25 * p10 + 0.5 * Math.Exp(-1), result.GradLatent[1, 0], Precision);

        // The official code's teacher form: label-weighted sum of each client's own cross-entropy.
        fedGen.TeacherLoss = AiDotNet.Enums.FedGenTeacherLoss.WeightedClientCrossEntropy;
        double ce2 = Math.Log(1 + Math.Exp(-2));
        double ce3 = Math.Log(1 + Math.Exp(-3));
        var weighted = fedGen.GeneratorObjective(heads, alpha, server, labels, latent, noise);
        Assert.Equal((0.75 * ce2 + 0.25 * ce3 + 0.5 * ce2 + 0.5 * ce3) / 2, weighted.Teacher, Precision);

        // dJ/dz matches a central difference of J = teacher - beta KL + eta diversity with every term switched on.
        fedGen.TeacherLoss = AiDotNet.Enums.FedGenTeacherLoss.EnsembleCrossEntropy;
        fedGen.TeacherLossWeight = 0.7;
        fedGen.StudentLossWeight = 0.4;
        fedGen.DiversityLossWeight = 1.3;
        var server2 = (new Tensor<double>(new[] { 0.5, -0.2 }, new[] { 2, 1 }), new Tensor<double>(new[] { 0.1, 0.0 }, new[] { 2 }));
        var analytic = fedGen.GeneratorObjective(heads, alpha, server2, labels, latent, noise).GradLatent;
        const double h = 1e-6;
        for (int row = 0; row < 2; row++)
        {
            var plus = latent.Clone();
            var minus = latent.Clone();
            plus[row, 0] += h;
            minus[row, 0] -= h;
            var up = fedGen.GeneratorObjective(heads, alpha, server2, labels, plus, noise);
            var down = fedGen.GeneratorObjective(heads, alpha, server2, labels, minus, noise);
            double jUp = 0.7 * up.Teacher - 0.4 * up.Student + 1.3 * up.Diversity;
            double jDown = 0.7 * down.Teacher - 0.4 * down.Student + 1.3 * down.Diversity;
            Assert.Equal((jUp - jDown) / (2 * h), analytic[row, 0], 6);
        }

        Assert.Throws<ArgumentOutOfRangeException>(() => fedGen.StudentLossWeight = -1);
    }

    [Fact(Timeout = 60000)]
    public async Task FedGEN_HalfLabelCountsAbove2048ReachTheServerExactly()
    {
        await Task.Yield();

        // Client counts {2049, 70001}: p(y) = [2049, 70001] / 72050 = [0.028439, 0.971561]. A count stored as one
        // Half rounds 2049 to 2048 and overflows 70001 to infinity (NaN label distribution).
        var fedGen = new AiDotNet.FederatedLearning.Distillation.FedGENDistillation<Half>(
            numClasses: 2, featureDim: 1, generatorEpochs: 0, noiseDim: 1, generatorHiddenDim: 2, batchSize: 2);
        var head = new Vector<Half>(HalfArray(1, -1, 0, 0));
        var knowledge = fedGen.ExtractHeadKnowledge(head, new[] { 2049, 70001 });

        for (int c = 0; c < 2; c++)
        {
            double decoded = 0;
            for (int d = 3; d >= 0; d--)
            {
                decoded = decoded * 256 + HalfOps.ToDouble(knowledge[c, 2 + d]);
            }

            Assert.Equal(c == 0 ? 2049.0 : 70001.0, decoded);
        }

        var global = fedGen.AggregateKnowledge(new Dictionary<int, Matrix<Half>> { [0] = knowledge }, null);
        Assert.Equal(2049.0 / 72050.0, HalfOps.ToDouble(global[0, global.Columns - 2]), 4);
        Assert.Equal(70001.0 / 72050.0, HalfOps.ToDouble(global[0, global.Columns - 1]), 3);
    }

    [Fact(Timeout = 60000)]
    public async Task DataFreeFCL_HalfRmsOfLongVectorAccumulatesSquaresInDouble()
    {
        await Task.Yield();

        // 4096 ones: rms = 1, so every importance is 1. Summing the 4096 squares in Half stops at 2048, giving
        // rms = sqrt(2048 / 4096) = 0.707 and importance 1.414.
        var values = new double[4096];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = 1;
        }

        var fcl = new AiDotNet.FederatedLearning.ContinualLearning.DataFreeFCL<Half>();
        var importance = fcl.ComputeImportance(new Vector<Half>(HalfArray(values)), new Matrix<Half>(1, 1));

        Assert.Equal(1.0, HalfOps.ToDouble(importance[0]), 3);
        Assert.Equal(1.0, HalfOps.ToDouble(importance[4095]), 3);
    }

    [Fact(Timeout = 60000)]
    public async Task PrototypeFederatedGraphLearning_HalfLossOfLongPrototypeAccumulatesSquaresInDouble()
    {
        await Task.Yield();

        // FedProto regularizer lambda * ||p_local - p_global||^2 with a zero global prototype and 4096 ones:
        // 0.1 * 4096 = 409.6. Summing the squares in Half stops at 2048 (loss 204.8).
        const int dim = 4096;
        var learner = new AiDotNet.FederatedLearning.Graph.PrototypeFederatedGraphLearning<Half>(
            new AiDotNet.Models.Options.FederatedGraphOptions(), prototypeDim: dim, numClasses: 1);
        learner.RegisterClientPrototypes(
            0,
            new Dictionary<int, Tensor<Half>> { [0] = new Tensor<Half>(new Half[dim], new[] { dim }) },
            new Dictionary<int, int> { [0] = 1 });
        learner.AggregatePrototypes();

        var ones = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            ones[i] = 1;
        }

        double loss = learner.ComputePrototypeLoss(
            new Dictionary<int, Tensor<Half>> { [0] = new Tensor<Half>(HalfArray(ones), new[] { dim }) }, 0.1);

        Assert.Equal(409.6, loss, 6);
    }

    [Fact(Timeout = 60000)]
    public async Task FedGEN_GeneratorStepIsOfficialAdamWithWeightDecayAndPerRoundLearningRateDecay()
    {
        await Task.Yield();

        // Official FedGen: torch.optim.Adam(lr = ensemble_lr = 3e-4, betas (0.9, 0.999), eps 1e-8,
        // weight_decay 1e-2) with ExponentialLR(gamma = 0.98) stepped once per round. Generator with 2 classes,
        // noise dim 1, hidden 1, feature 1; input [onehot(0); 0.5] = [1, 0, 0.5], hidden [2], dJ/dz = 0.4:
        // dW2 = 0.8, db2 = 0.4, dW1 = 0.4 w2 [1, 0, 0.5], db1 = 0.4 w2. Step 1 uses lr 3e-4; after one round the
        // second step uses 3e-4 * 0.98 with the accumulated moments.
        var fedGen = new AiDotNet.FederatedLearning.Distillation.FedGENDistillation<double>(
            numClasses: 2, featureDim: 1, generatorEpochs: 0, seed: 7, noiseDim: 1, generatorHiddenDim: 1, batchSize: 1);
        Assert.Equal(AiDotNet.Enums.FedGenGeneratorOptimizer.Adam, fedGen.GeneratorOptimizer);
        Assert.Equal(3e-4, fedGen.CurrentGeneratorLearningRate, 15);

        var state = fedGen.GeneratorState;
        var w1 = state.W1.ToArray();
        var b1 = state.B1.ToArray();
        var w2 = state.W2.ToArray();
        var b2 = state.B2.ToArray();
        var m = new double[6];
        var v = new double[6];

        var input = new Tensor<double>(new[] { 1.0, 0.0, 0.5 }, new[] { 1, 3 });
        var hidden = new Tensor<double>(new[] { 2.0 }, new[] { 1, 1 });
        var gradLatent = new Tensor<double>(new[] { 0.4 }, new[] { 1, 1 });

        for (int step = 1; step <= 2; step++)
        {
            double lr = step == 1 ? 3e-4 : 3e-4 * 0.98;
            var theta = new[] { w2[0], b2[0], w1[0], w1[1], w1[2], b1[0] };
            var grad = new[] { 0.8, 0.4, 0.4 * w2[0], 0.0, 0.2 * w2[0], 0.4 * w2[0] };
            for (int i = 0; i < 6; i++)
            {
                double g = grad[i] + 1e-2 * theta[i];
                m[i] = 0.9 * m[i] + 0.1 * g;
                v[i] = 0.999 * v[i] + 0.001 * g * g;
                double mHat = m[i] / (1 - Math.Pow(0.9, step));
                double vHat = v[i] / (1 - Math.Pow(0.999, step));
                theta[i] -= lr * mHat / (Math.Sqrt(vHat) + 1e-8);
            }

            fedGen.BackpropGenerator(input, hidden, gradLatent);
            w2[0] = theta[0];
            b2[0] = theta[1];
            w1[0] = theta[2];
            w1[1] = theta[3];
            w1[2] = theta[4];
            b1[0] = theta[5];

            var after = fedGen.GeneratorState;
            Assert.Equal(w2[0], after.W2.ToArray()[0], 12);
            Assert.Equal(b2[0], after.B2.ToArray()[0], 12);
            Assert.Equal(w1[0], after.W1.ToArray()[0], 12);
            Assert.Equal(w1[1], after.W1.ToArray()[1], 12);
            Assert.Equal(w1[2], after.W1.ToArray()[2], 12);
            Assert.Equal(b1[0], after.B1.ToArray()[0], 12);

            if (step == 1)
            {
                // One aggregation round (no generator steps) advances the ExponentialLR schedule.
                var head = new Vector<double>(new[] { 1.0, -1.0, 0.0, 0.0 });
                fedGen.AggregateKnowledge(
                    new Dictionary<int, Matrix<double>> { [0] = fedGen.ExtractHeadKnowledge(head, new[] { 1, 1 }) }, null);
                Assert.Equal(3e-4 * 0.98, fedGen.CurrentGeneratorLearningRate, 15);
            }
        }

        // SGD with the same weight decay is available: theta -= lr (g + 1e-2 theta).
        fedGen.GeneratorOptimizer = AiDotNet.Enums.FedGenGeneratorOptimizer.Sgd;
        double lrNow = fedGen.CurrentGeneratorLearningRate;
        double expectedB2 = b2[0] - lrNow * (0.4 + 1e-2 * b2[0]);
        fedGen.BackpropGenerator(input, hidden, gradLatent);
        Assert.Equal(expectedB2, fedGen.GeneratorState.B2.ToArray()[0], 12);
    }
}
