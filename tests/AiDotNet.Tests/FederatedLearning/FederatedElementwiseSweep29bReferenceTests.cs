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
}
