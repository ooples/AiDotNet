using AiDotNet.DistributedTraining;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;
using Xunit;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// Deep integration tests for DistributedTraining:
/// ActivationCheckpointConfig (defaults, validation, strategies),
/// InMemoryCommunicationBackend (construction, rank/worldSize, initialization, collective ops),
/// RecomputeStrategy enum.
/// </summary>
// Serialized: the ZeRO/hybrid math invariants below spin up multi-rank (multi-thread)
// simulations over the shared-static InMemoryCommunicationBackend, whose point-to-point
// Send/Receive is timing-sensitive under the full suite's heavy parallelism. Running this class
// in the non-parallel phase gives those simulations dedicated CPU without changing their math.
[Collection("ConvergenceSensitive")]
public class DistributedTrainingDeepMathIntegrationTests
{
    // ============================
    // ActivationCheckpointConfig: Defaults
    // ============================

    [Fact(Timeout = 120000)]
    public async Task ActivationCheckpointConfig_Defaults_Disabled()
    {
        var config = new ActivationCheckpointConfig();
        Assert.False(config.Enabled);
    }

    [Fact(Timeout = 120000)]
    public async Task ActivationCheckpointConfig_Defaults_CheckpointEveryTenLayers()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Equal(10, config.CheckpointEveryNLayers);
    }

    [Fact(Timeout = 120000)]
    public async Task ActivationCheckpointConfig_Defaults_RecomputeStrategyNone()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Equal(RecomputeStrategy.None, config.RecomputeStrategy);
    }

    [Fact(Timeout = 120000)]
    public async Task ActivationCheckpointConfig_Defaults_MaxActivationsZero()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Equal(0, config.MaxActivationsInMemory);
    }

    [Fact(Timeout = 120000)]
    public async Task ActivationCheckpointConfig_Defaults_CheckpointFirstLayerTrue()
    {
        var config = new ActivationCheckpointConfig();
        Assert.True(config.CheckpointFirstLayer);
    }

    // ============================
    // ActivationCheckpointConfig: Property Setting
    // ============================

    [Fact(Timeout = 120000)]
    public async Task ActivationCheckpointConfig_SetAll()
    {
        var config = new ActivationCheckpointConfig
        {
            Enabled = true,
            CheckpointEveryNLayers = 5,
            RecomputeStrategy = RecomputeStrategy.Selective,
            MaxActivationsInMemory = 20,
            CheckpointFirstLayer = false
        };

        Assert.True(config.Enabled);
        Assert.Equal(5, config.CheckpointEveryNLayers);
        Assert.Equal(RecomputeStrategy.Selective, config.RecomputeStrategy);
        Assert.Equal(20, config.MaxActivationsInMemory);
        Assert.False(config.CheckpointFirstLayer);
    }

    // ============================
    // ActivationCheckpointConfig: Validation
    // ============================

    [Fact(Timeout = 120000)]
    public async Task ActivationCheckpointConfig_CheckpointEveryNLayers_ZeroThrows()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Throws<ArgumentOutOfRangeException>(() => config.CheckpointEveryNLayers = 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ActivationCheckpointConfig_CheckpointEveryNLayers_NegativeThrows()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Throws<ArgumentOutOfRangeException>(() => config.CheckpointEveryNLayers = -1);
    }

    [Fact(Timeout = 120000)]
    public async Task ActivationCheckpointConfig_CheckpointEveryNLayers_OneIsValid()
    {
        var config = new ActivationCheckpointConfig();
        config.CheckpointEveryNLayers = 1;
        Assert.Equal(1, config.CheckpointEveryNLayers);
    }

    [Fact(Timeout = 120000)]
    public async Task ActivationCheckpointConfig_MaxActivationsInMemory_NegativeThrows()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Throws<ArgumentOutOfRangeException>(() => config.MaxActivationsInMemory = -1);
    }

    [Fact(Timeout = 120000)]
    public async Task ActivationCheckpointConfig_MaxActivationsInMemory_ZeroIsValid()
    {
        var config = new ActivationCheckpointConfig();
        config.MaxActivationsInMemory = 0;
        Assert.Equal(0, config.MaxActivationsInMemory);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(50)]
    [InlineData(100)]
    public void ActivationCheckpointConfig_CheckpointEveryNLayers_ValidValues(int value)
    {
        var config = new ActivationCheckpointConfig { CheckpointEveryNLayers = value };
        Assert.Equal(value, config.CheckpointEveryNLayers);
    }

    // ============================
    // RecomputeStrategy Enum
    // ============================

    [Fact(Timeout = 120000)]
    public async Task RecomputeStrategy_HasThreeValues()
    {
        var values = (((RecomputeStrategy[])Enum.GetValues(typeof(RecomputeStrategy))));
        Assert.Equal(3, values.Length);
    }

    [Theory]
    [InlineData(RecomputeStrategy.Selective)]
    [InlineData(RecomputeStrategy.Full)]
    [InlineData(RecomputeStrategy.None)]
    public void RecomputeStrategy_AllValuesValid(RecomputeStrategy strategy)
    {
        Assert.True(Enum.IsDefined(typeof(RecomputeStrategy), strategy));
    }

    // ============================
    // InMemoryCommunicationBackend: Construction
    // ============================

    [Fact(Timeout = 120000)]
    public async Task InMemoryBackend_Construction_RankAndWorldSize()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4);
        Assert.Equal(0, backend.Rank);
        Assert.Equal(4, backend.WorldSize);
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(0, 4)]
    [InlineData(3, 4)]
    [InlineData(0, 8)]
    [InlineData(7, 8)]
    public void InMemoryBackend_ValidRankWorldSize(int rank, int worldSize)
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: rank, worldSize: worldSize);
        Assert.Equal(rank, backend.Rank);
        Assert.Equal(worldSize, backend.WorldSize);
    }

    [Fact(Timeout = 120000)]
    public async Task InMemoryBackend_Initialize_Succeeds()
    {
        var envId = Guid.NewGuid().ToString("N");
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        // Initialize should not throw for a single-process backend
    }

    [Fact(Timeout = 120000)]
    public async Task InMemoryBackend_FloatType_Constructs()
    {
        var backend = new InMemoryCommunicationBackend<float>(rank: 0, worldSize: 2);
        Assert.Equal(0, backend.Rank);
        Assert.Equal(2, backend.WorldSize);
    }

    // ============================
    // InMemoryCommunicationBackend: Broadcast (single rank)
    // ============================

    [Fact(Timeout = 120000)]
    public async Task InMemoryBackend_Broadcast_SingleProcess_PreservesData()
    {
        var envId = Guid.NewGuid().ToString("N");
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();

        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        backend.Broadcast(data, 0);

        // With worldSize=1, broadcast should be a no-op
        Assert.Equal(1.0, data[0]);
        Assert.Equal(2.0, data[1]);
        Assert.Equal(3.0, data[2]);
    }

    // ============================
    // ActivationCheckpointConfig: Optimal Checkpoint Frequency
    // ============================

    [Theory]
    [InlineData(100, 10)]  // sqrt(100) = 10
    [InlineData(16, 4)]    // sqrt(16) = 4
    [InlineData(25, 5)]    // sqrt(25) = 5
    [InlineData(64, 8)]    // sqrt(64) = 8
    public void ActivationCheckpoint_OptimalFrequency_IsSqrtLayers(int totalLayers, int expectedCheckpointInterval)
    {
        // Optimal checkpoint frequency is approximately sqrt(L) where L = total layers
        // This gives O(sqrt(L)) memory instead of O(L)
        int optimalInterval = (int)Math.Round(Math.Sqrt(totalLayers));
        Assert.Equal(expectedCheckpointInterval, optimalInterval);

        // Verify we can set this value
        var config = new ActivationCheckpointConfig { CheckpointEveryNLayers = optimalInterval };
        Assert.Equal(expectedCheckpointInterval, config.CheckpointEveryNLayers);
    }

    [Theory]
    [InlineData(100, 10)]  // 100/10 = 10 checkpoints
    [InlineData(100, 1)]   // 100/1 = 100 checkpoints (every layer = no recomputation)
    [InlineData(100, 100)] // 100/100 = 1 checkpoint (maximum recomputation)
    public void ActivationCheckpoint_NumberOfCheckpoints(int totalLayers, int interval)
    {
        // Number of checkpoints = ceil(totalLayers / interval)
        int numCheckpoints = (int)Math.Ceiling((double)totalLayers / interval);

        var config = new ActivationCheckpointConfig { CheckpointEveryNLayers = interval };
        Assert.True(numCheckpoints >= 1);
        Assert.True(numCheckpoints <= totalLayers);
    }

    // ============================
    // Memory savings calculation
    // ============================

    [Theory]
    [InlineData(100)]
    [InlineData(200)]
    [InlineData(400)]
    public void ActivationCheckpoint_MemorySavings_SublinearVsLinear(int totalLayers)
    {
        // Without checkpointing: memory = O(L) = totalLayers activations
        int withoutCheckpointing = totalLayers;

        // With optimal checkpointing: memory = O(sqrt(L))
        int optimalInterval = (int)Math.Round(Math.Sqrt(totalLayers));
        int numCheckpoints = (int)Math.Ceiling((double)totalLayers / optimalInterval);
        // Memory = checkpoints stored + max segment between checkpoints
        int withCheckpointing = numCheckpoints + optimalInterval;

        // Checkpointing should use significantly less memory than no checkpointing
        Assert.True(withCheckpointing < withoutCheckpointing,
            $"Checkpointing ({withCheckpointing}) should use less memory than no checkpointing ({withoutCheckpointing}) for {totalLayers} layers");

        // Savings ratio should be > 2x for large models
        if (totalLayers >= 100)
        {
            double ratio = (double)withoutCheckpointing / withCheckpointing;
            Assert.True(ratio > 2.0,
                $"Memory savings ratio should be > 2x for {totalLayers} layers, got {ratio:F2}x");
        }
    }

    // =====================================================================================
    // ZeRO sharded-optimizer MATH INVARIANTS
    // (Rajbhandari et al. 2020, "ZeRO: Memory Optimizations Toward Training Trillion Parameter
    //  Models"; Ren et al. 2021, "ZeRO-Offload").
    //
    // The defining correctness property of ZeRO is that partitioning the optimizer state /
    // gradients / parameters across ranks is MATHEMATICALLY TRANSPARENT: it changes only where
    // memory lives, never the numerical result. Adam updates each parameter independently from its
    // own (gradient, m, v), so a per-rank shard update is bit-for-bit the same as the corresponding
    // slice of the full-vector update. These invariants prove exactly that.
    // =====================================================================================

    private const double ZeroTol = 1e-9;
    private const int ZeroN = 8;

    private static OptimizationInputData<double, Vector<double>, Vector<double>> ZeroInput(
        IFullModel<double, Vector<double>, Vector<double>> model)
        => new OptimizationInputData<double, Vector<double>, Vector<double>>
        {
            XTrain = new Vector<double>(new double[ZeroN]),
            YTrain = new Vector<double>(new double[ZeroN]),
            InitialSolution = model,
        };

    private static DistributedTrainingIntegrationTests.MockDistributedModel NewZeroModel(double gradientScale = 1.0)
        => new DistributedTrainingIntegrationTests.MockDistributedModel(ZeroN, gradientScale);

    private static IGradientBasedOptimizer<double, Vector<double>, Vector<double>> NewAdam()
        => new AdamOptimizer<double, Vector<double>, Vector<double>>(null);

    /// <summary>
    /// INVARIANT: one ZeRO-1 step on a single rank is bit-for-bit a single Adam update on the full
    /// parameter vector. (World size 1 ⇒ the AllReduce is identity and the one shard is the whole
    /// vector, so the only thing that runs is the wrapped optimizer's UpdateParameters.)
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ZeRO1_WorldSize1_EqualsDirectAdamStep()
    {
        await Task.Yield();
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };

        // Reference: plain Adam applied to the same params + the model's (fixed) gradient.
        var refModel = NewZeroModel();
        var p0 = refModel.GetParameters();
        var g = refModel.ComputeGradients(new Vector<double>(new double[ZeroN]), new Vector<double>(new double[ZeroN]));
        var expected = NewAdam().UpdateParameters(p0.Clone(), g.Clone());

        // ZeRO-1 sharded step.
        var model = NewZeroModel();
        var zero1 = new ZeRO1Optimizer<double, Vector<double>, Vector<double>>(NewAdam(), config);
        zero1.Optimize(ZeroInput(model));
        var got = model.GetParameters();

        for (int i = 0; i < ZeroN; i++)
            Assert.Equal(expected[i], got[i], ZeroTol);

        backend.Shutdown();
    }

    /// <summary>
    /// INVARIANT: ZeRO-1 and ZeRO-2 are numerically identical — they differ only in whether the
    /// gradient is replicated (AllReduce) or sharded (ReduceScatter), which is a memory choice, not
    /// a math one. On a single rank both reduce to the same full Adam step.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ZeRO1_And_ZeRO2_WorldSize1_ProduceIdenticalParameters()
    {
        await Task.Yield();
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };

        var m1 = NewZeroModel();
        new ZeRO1Optimizer<double, Vector<double>, Vector<double>>(NewAdam(), config).Optimize(ZeroInput(m1));
        var z1 = m1.GetParameters();

        var m2 = NewZeroModel();
        new ZeRO2Optimizer<double, Vector<double>, Vector<double>>(NewAdam(), config).Optimize(ZeroInput(m2));
        var z2 = m2.GetParameters();

        for (int i = 0; i < ZeroN; i++)
            Assert.Equal(z1[i], z2[i], ZeroTol);

        backend.Shutdown();
    }

    /// <summary>
    /// INVARIANT (the crux of ZeRO correctness): a TWO-rank ZeRO-1 run — where each rank holds and
    /// updates only its optimizer-state shard, then AllGathers — reconstructs the EXACT SAME full
    /// parameter vector on both ranks, and that vector equals the single-rank result. This proves the
    /// state partitioning is transparent (per-parameter Adam independence) and the shards tile and
    /// reassemble correctly.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ZeRO1_TwoRanks_ReconstructIdenticalParameters_MatchingSingleRank()
    {
        await Task.Yield();
        // The two ranks produce DIFFERENT gradients (scale 1 and 2); a correct AllReduce(Average) makes
        // both update with the mean gradient (scale 1.5), so the single-rank reference uses scale 1.5.
        // If the collective were removed, rank 0 (scale 1) and rank 1 (scale 2) would diverge from the
        // 1.5 reference and this test would fail — so it genuinely exercises the reduce.
        var refBackend = new InMemoryCommunicationBackend<double>(0, 1, Guid.NewGuid().ToString());
        refBackend.Initialize();
        var refConfig = new ShardingConfiguration<double>(refBackend) { AutoSyncGradients = true };
        var refModel = NewZeroModel(1.5);
        new ZeRO1Optimizer<double, Vector<double>, Vector<double>>(NewAdam(), refConfig).Optimize(ZeroInput(refModel));
        var single = refModel.GetParameters();
        refBackend.Shutdown();

        // Two-rank run over a shared InMemory environment.
        var envId = Guid.NewGuid().ToString();
        var results = new Vector<double>[2];
        var tasks = new List<Task>();
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            tasks.Add(Task.Run(() =>
            {
                var backend = new InMemoryCommunicationBackend<double>(rank, 2, envId);
                backend.Initialize();
                var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };
                var model = NewZeroModel(rank + 1);   // rank 0 -> scale 1, rank 1 -> scale 2
                new ZeRO1Optimizer<double, Vector<double>, Vector<double>>(NewAdam(), config).Optimize(ZeroInput(model));
                results[rank] = model.GetParameters();
                backend.Shutdown();
            }));
        }
        await Task.WhenAll(tasks);

        // Both ranks reconstruct the identical full vector ...
        for (int i = 0; i < ZeroN; i++)
            Assert.Equal(results[0][i], results[1][i], ZeroTol);
        // ... and it is bit-for-bit the single-rank (un-sharded) result.
        for (int i = 0; i < ZeroN; i++)
            Assert.Equal(single[i], results[0][i], ZeroTol);
    }

    /// <summary>
    /// INVARIANT: the same two-rank transparency holds for ZeRO-2, where the GRADIENT is also sharded
    /// (ReduceScatter) so no rank ever materializes the full averaged gradient — yet the reassembled
    /// parameters still match the single-rank result exactly.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ZeRO2_TwoRanks_ReconstructIdenticalParameters_MatchingSingleRank()
    {
        await Task.Yield();
        // Rank-dependent gradients (scale 1 and 2): a correct ReduceScatter(Average) gives each rank the
        // mean-gradient (scale 1.5) shard, matching the single-rank reference at scale 1.5. Removing the
        // collective would leave the ranks at scales 1/2 and fail the comparison.
        var refBackend = new InMemoryCommunicationBackend<double>(0, 1, Guid.NewGuid().ToString());
        refBackend.Initialize();
        var refConfig = new ShardingConfiguration<double>(refBackend) { AutoSyncGradients = true };
        var refModel = NewZeroModel(1.5);
        new ZeRO2Optimizer<double, Vector<double>, Vector<double>>(NewAdam(), refConfig).Optimize(ZeroInput(refModel));
        var single = refModel.GetParameters();
        refBackend.Shutdown();

        var envId = Guid.NewGuid().ToString();
        var results = new Vector<double>[2];
        var tasks = new List<Task>();
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            tasks.Add(Task.Run(() =>
            {
                var backend = new InMemoryCommunicationBackend<double>(rank, 2, envId);
                backend.Initialize();
                var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };
                var model = NewZeroModel(rank + 1);   // rank 0 -> scale 1, rank 1 -> scale 2
                new ZeRO2Optimizer<double, Vector<double>, Vector<double>>(NewAdam(), config).Optimize(ZeroInput(model));
                results[rank] = model.GetParameters();
                backend.Shutdown();
            }));
        }
        await Task.WhenAll(tasks);

        for (int i = 0; i < ZeroN; i++)
            Assert.Equal(results[0][i], results[1][i], ZeroTol);
        for (int i = 0; i < ZeroN; i++)
            Assert.Equal(single[i], results[0][i], ZeroTol);
    }

    /// <summary>
    /// INVARIANT: a ZeRO-1 step actually MOVES parameters against the gradient (real training, not a
    /// no-op). With the model's strictly-positive gradient, every parameter must strictly decrease.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ZeRO1_Step_MovesEveryParameterAgainstGradient()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, Guid.NewGuid().ToString());
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };

        var model = NewZeroModel();
        var before = model.GetParameters();
        new ZeRO1Optimizer<double, Vector<double>, Vector<double>>(NewAdam(), config).Optimize(ZeroInput(model));
        var after = model.GetParameters();

        for (int i = 0; i < ZeroN; i++)
            Assert.True(after[i] < before[i],
                $"param[{i}] must decrease under a positive gradient: before={before[i]}, after={after[i]}");

        backend.Shutdown();
    }

    /// <summary>
    /// INVARIANT: HybridShardedModel now reduces gradients within the DATA-PARALLEL subgroup instead of
    /// throwing NotSupportedException. With pipeline=1, tensor=1, dataParallelSize=2 the two ranks are
    /// pure data replicas; each runs Train (which averages gradients across the subgroup via Send/Recv)
    /// and both must (a) complete without throwing or deadlocking and (b) end with identical parameters.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Hybrid_DataParallelSubgroup_ReducesGradientsWithoutThrowing_AndStaysConsistent()
    {
        await Task.Yield();
        var envId = Guid.NewGuid().ToString();
        var results = new Vector<double>[2];
        var errors = new Exception?[2];
        var tasks = new List<Task>();
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    var backend = new InMemoryCommunicationBackend<double>(rank, 2, envId);
                    backend.Initialize();
                    var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };
                    // Rank-dependent gradients (scale 1 and 2): only a correct data-parallel subgroup
                    // average leaves both replicas consistent; a removed/broken reduce would leave rank 0
                    // (scale 1) and rank 1 (scale 2) with different parameters and fail the check below.
                    var model = NewZeroModel(rank + 1);
                    // pipeline=1, tensor=1 => the two ranks form a single data-parallel group of size 2.
                    var hybrid = new HybridShardedModel<double, Vector<double>, Vector<double>>(model, config, 1, 1, 2);
                    hybrid.Train(new Vector<double>(new double[ZeroN]), new Vector<double>(new double[ZeroN]));
                    // Read the wrapped model's LOCAL parameters (no collective) — with pipeline=1/tensor=1
                    // there is no parameter sharding, so the wrapped model holds the full updated vector.
                    // (Calling the collective hybrid.GetParameters() here would desync the two ranks,
                    // which are no longer in lockstep after Train.)
                    results[rank] = model.GetParameters();
                    backend.Shutdown();
                }
                catch (Exception ex) { errors[rank] = ex; }
            }));
        }
        await Task.WhenAll(tasks);

        // The data-parallel subgroup reduction must run to completion on both ranks (previously threw).
        Assert.True(errors[0] is null, "rank 0: " + errors[0]);
        Assert.True(errors[1] is null, "rank 1: " + errors[1]);
        // (assertions continue below)
        AssertHybridConsistent(results);
    }

    private static void AssertHybridConsistent(Vector<double>[] results)
    {
        Assert.Equal(results[0].Length, results[1].Length);
        for (int i = 0; i < results[0].Length; i++)
            Assert.Equal(results[0][i], results[1][i], ZeroTol);
    }

    /// <summary>
    /// INVARIANT (Megatron-LM tensor parallelism, Shoeybi et al. 2019 §3): the canonical two-layer MLP
    /// ColumnParallelLinear -> RowParallelLinear run across TWO ranks (each holding only its weight
    /// shards, communicating via the f/ḡ conjugate operators) produces output bit-identical (to tol) to
    /// the SAME MLP computed non-parallel on the full weights. Proves column-split · row-split + all-reduce
    /// == full matmul, i.e. the partitioning is mathematically transparent. This is also the TENSOR-PARALLEL
    /// NO-DOUBLE-REDUCE guard: the row-parallel all-reduce runs exactly once (in the ḡ operator). A second,
    /// redundant reduce (e.g. an all-reduce added in the optimizer wrapper) would scale the reduced term and
    /// break this bit-identical match — so this test fails loudly if TP gradient/activation sync is ever
    /// double-counted.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task MegatronMLP_TwoRank_ColumnThenRowParallel_EqualsNonParallelMLP()
    {
        const int batch = 2, inputSize = 4, ffn = 8, outputSize = 3;
        var A = DeterministicMatrix(ffn, inputSize, 1);      // [ffn, inputSize]
        var aBias = DeterministicVector(ffn, 2);
        var B = DeterministicMatrix(outputSize, ffn, 3);     // [outputSize, ffn]
        var bBias = DeterministicVector(outputSize, 4);
        var X = DeterministicMatrix(batch, inputSize, 5);    // [batch, inputSize]

        // Non-parallel reference (hand-computed): Y = (X·Aᵀ + aBias)·Bᵀ + bBias.
        var yRef = new double[batch, outputSize];
        for (int b = 0; b < batch; b++)
        {
            var h = new double[ffn];
            for (int f = 0; f < ffn; f++)
            {
                double acc = aBias[f];
                for (int i = 0; i < inputSize; i++) acc += X[b, i] * A[f, i];
                h[f] = acc;
            }
            for (int o = 0; o < outputSize; o++)
            {
                double acc = bBias[o];
                for (int f = 0; f < ffn; f++) acc += h[f] * B[o, f];
                yRef[b, o] = acc;
            }
        }

        var Xt = MatrixToTensor(X, batch, inputSize);
        var At = MatrixToTensor(A, ffn, inputSize);
        var aBt = VectorToTensor(aBias);
        var Bt = MatrixToTensor(B, outputSize, ffn);
        var bBt = VectorToTensor(bBias);

        var envId = System.Guid.NewGuid().ToString();
        var results = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>[2];
        var errors = new Exception?[2];
        var tasks = new List<Task>();
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    var backend = new InMemoryCommunicationBackend<double>(rank, 2, envId);
                    backend.Initialize();
                    var col = new AiDotNet.DistributedTraining.Layers.ColumnParallelLinear<double>(backend, inputSize, ffn);
                    col.SetFromFullWeights(At, aBt);
                    var row = new AiDotNet.DistributedTraining.Layers.RowParallelLinear<double>(backend, ffn, outputSize);
                    row.SetFromFullWeights(Bt, bBt);

                    var h = col.Forward(Xt);   // [batch, ffn/2] (split, no forward comm)
                    var y = row.Forward(h);    // [batch, outputSize] after ḡ all-reduce of the partials
                    results[rank] = y;
                    backend.Shutdown();
                }
                catch (Exception ex) { errors[rank] = ex; }
            }));
        }
        await Task.WhenAll(tasks);

        Assert.True(errors[0] is null, "rank 0: " + errors[0]);
        Assert.True(errors[1] is null, "rank 1: " + errors[1]);
        for (int rr = 0; rr < 2; rr++)
            for (int b = 0; b < batch; b++)
                for (int o = 0; o < outputSize; o++)
                    Assert.Equal(yRef[b, o], results[rr][b, o], 9);
    }

    /// <summary>
    /// INVARIANT (AUTOMATIC tensor parallelism): the TensorParallelLayerPartitioner rewrites a plain MLP
    /// built from two FullyConnectedLayers into ColumnParallel → RowParallel Megatron layers WITHOUT a
    /// manual model rewrite, and the auto-partitioned forward is bit-identical (to tol) to the original
    /// non-parallel forward across two ranks. This proves the auto-substitution is numerically transparent
    /// (real compute partitioning, not the replication fallback) AND that the weight extraction order is
    /// correct. The first layer's ReLU is applied on each rank's output-column slice, exercising the
    /// element-wise-activation-on-split property the Megatron MLP relies on.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task AutoTensorParallel_PartitionedMLP_EqualsNonParallelMLP()
    {
        await Task.Yield();
        const int batch = 2, inputSize = 4, ffn = 8, outputSize = 3;
        var fc1Params = new Vector<double>(DeterministicVector(ffn * inputSize + ffn, 11));
        var fc2Params = new Vector<double>(DeterministicVector(outputSize * ffn + outputSize, 12));
        var Xm = DeterministicMatrix(batch, inputSize, 13);
        var X = MatrixToTensor(Xm, batch, inputSize);

        // Non-parallel reference, HAND-COMPUTED (deliberately not by running FullyConnectedLayer.Forward:
        // the auto-partition path already exercises the real layers, and running an extra layer forward on
        // the test thread would be comparing the partitioner against itself). FullyConnectedLayer stores
        // weights [out, in] then bias [out]; forward is Y = act(X·Wᵀ + b). fc1 uses ReLU, fc2 identity.
        var yRef = new double[batch, outputSize];
        for (int b = 0; b < batch; b++)
        {
            var h = new double[ffn];
            for (int f = 0; f < ffn; f++)
            {
                double acc = fc1Params[ffn * inputSize + f]; // bias1[f]
                for (int i = 0; i < inputSize; i++) acc += Xm[b, i] * fc1Params[f * inputSize + i];
                h[f] = acc > 0 ? acc : 0.0; // ReLU
            }
            for (int o = 0; o < outputSize; o++)
            {
                double acc = fc2Params[outputSize * ffn + o]; // bias2[o]
                for (int f = 0; f < ffn; f++) acc += h[f] * fc2Params[o * ffn + f];
                yRef[b, o] = acc;
            }
        }

        var envId = System.Guid.NewGuid().ToString();
        var results = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>[2];
        var errors = new Exception?[2];
        var tasks = new List<Task>();
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    var backend = new InMemoryCommunicationBackend<double>(rank, 2, envId);
                    backend.Initialize();
                    var (fc1, fc2) = BuildMlp(fc1Params, fc2Params, inputSize, ffn, outputSize);
                    var part = AiDotNet.DistributedTraining.Layers.TensorParallelLayerPartitioner<double>.Partition(
                        new List<AiDotNet.Interfaces.ILayer<double>> { fc1, fc2 }, backend);

                    // The consecutive linear pair must be recognized as a Megatron MLP (2 partitioned, 0 replicated).
                    Assert.Equal(2, part.PartitionedLinearCount);
                    Assert.Equal(0, part.ReplicatedLayerCount);

                    var y = part.Layers[1].Forward(part.Layers[0].Forward(X));
                    results[rank] = y;
                    backend.Shutdown();
                }
                catch (Exception ex) { errors[rank] = ex; }
            }));
        }
        await Task.WhenAll(tasks);

        Assert.True(errors[0] is null, "rank 0: " + errors[0]);
        Assert.True(errors[1] is null, "rank 1: " + errors[1]);
        for (int rr = 0; rr < 2; rr++)
            for (int b = 0; b < batch; b++)
                for (int o = 0; o < outputSize; o++)
                    Assert.Equal(yRef[b, o], results[rr][b, o], 9);
    }

    private static (AiDotNet.NeuralNetworks.Layers.FullyConnectedLayer<double> fc1,
                    AiDotNet.NeuralNetworks.Layers.FullyConnectedLayer<double> fc2) BuildMlp(
        Vector<double> p1, Vector<double> p2, int inputSize, int ffn, int outputSize)
    {
        var fc1 = new AiDotNet.NeuralNetworks.Layers.FullyConnectedLayer<double>(
            inputSize, ffn, new AiDotNet.ActivationFunctions.ReLUActivation<double>());
        fc1.SetParameters(p1);
        var fc2 = new AiDotNet.NeuralNetworks.Layers.FullyConnectedLayer<double>(
            ffn, outputSize, new AiDotNet.ActivationFunctions.IdentityActivation<double>());
        fc2.SetParameters(p2);
        return (fc1, fc2);
    }

    /// <summary>
    /// INVARIANT (ZeRO Stage-3 / FSDP residency, Zhao et al. 2023): a Stage3ShardedLinear where each of
    /// two ranks stores only HALF the flat weight, materializing the full weight just-in-time via
    /// AllGather in Forward, produces output bit-identical (to tol) to the full non-sharded linear
    /// Y = X·Wᵀ + b — AND each rank's resident parameter count is ~half the full weight (true residency,
    /// not the full-vector gather of the cache-eviction path).
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Stage3ShardedLinear_TwoRank_EqualsFullLinear_WithHalfResidentParams()
    {
        const int batch = 2, inputSize = 6, outputSize = 4;
        var W = DeterministicMatrix(outputSize, inputSize, 11);   // [outputSize, inputSize]
        var b = DeterministicVector(outputSize, 12);
        var X = DeterministicMatrix(batch, inputSize, 13);

        // Reference: Y = X·Wᵀ + b.
        var yRef = new double[batch, outputSize];
        for (int r = 0; r < batch; r++)
            for (int o = 0; o < outputSize; o++)
            {
                double acc = b[o];
                for (int i = 0; i < inputSize; i++) acc += X[r, i] * W[o, i];
                yRef[r, o] = acc;
            }

        var Wt = MatrixToTensor(W, outputSize, inputSize);
        var bt = VectorToTensor(b);
        var Xt = MatrixToTensor(X, batch, inputSize);

        var envId = System.Guid.NewGuid().ToString();
        var results = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>[2];
        var residentParams = new long[2];
        var errors = new Exception?[2];
        var tasks = new List<Task>();
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    var backend = new InMemoryCommunicationBackend<double>(rank, 2, envId);
                    backend.Initialize();
                    var layer = new AiDotNet.DistributedTraining.Layers.Stage3ShardedLinear<double>(backend, inputSize, outputSize);
                    layer.SetFromFullWeights(Wt, bt);
                    residentParams[rank] = layer.ParameterCount;
                    results[rank] = layer.Forward(Xt);
                    backend.Shutdown();
                }
                catch (Exception ex) { errors[rank] = ex; }
            }));
        }
        await Task.WhenAll(tasks);

        Assert.True(errors[0] is null, "rank 0: " + errors[0]);
        Assert.True(errors[1] is null, "rank 1: " + errors[1]);
        for (int rr = 0; rr < 2; rr++)
            for (int rb = 0; rb < batch; rb++)
                for (int o = 0; o < outputSize; o++)
                    Assert.Equal(yRef[rb, o], results[rr][rb, o], 9);

        // True residency (EXACT): each rank stores only its flat-weight shard — ceil(outputSize*inputSize /
        // worldSize) — plus the replicated bias (outputSize). With 4x6 weights over 2 ranks that is
        // ceil(24/2)=12 + 4 = 16 on BOTH ranks, strictly less than the full weight+bias (24 + 4 = 28).
        long weightParams = (long)outputSize * inputSize;
        long expectedResident = (weightParams + 1) / 2 + outputSize; // ceil(weight/2) shard + replicated bias
        long full = weightParams + outputSize;
        Assert.Equal(expectedResident, residentParams[0]);
        Assert.Equal(expectedResident, residentParams[1]);
        Assert.True(expectedResident < full, $"resident {expectedResident} must be < full {full}");
    }

    /// <summary>
    /// INVARIANT: FSDP == ZeRO Stage-3. A two-rank FSDP step ReduceScatters the (rank-dependent) gradients,
    /// updates only each rank's shard, and AllGathers — reconstructing the identical full vector on both
    /// ranks, equal to the single-rank result at the averaged gradient. Proves the FSDP path is the real
    /// sharded step (not a full-replication placeholder): removing the collective would leave rank 0 (scale
    /// 1) and rank 1 (scale 2) diverged from the scale-1.5 reference.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task FSDP_TwoRanks_ReconstructIdenticalParameters_MatchingSingleRank()
    {
        await Task.Yield();
        var refBackend = new InMemoryCommunicationBackend<double>(0, 1, Guid.NewGuid().ToString());
        refBackend.Initialize();
        var refConfig = new ShardingConfiguration<double>(refBackend) { AutoSyncGradients = true };
        var refModel = NewZeroModel(1.5);
        new FSDPOptimizer<double, Vector<double>, Vector<double>>(NewAdam(), refConfig).Optimize(ZeroInput(refModel));
        var single = refModel.GetParameters();
        refBackend.Shutdown();

        var envId = Guid.NewGuid().ToString();
        var results = new Vector<double>[2];
        var tasks = new List<Task>();
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            tasks.Add(Task.Run(() =>
            {
                var backend = new InMemoryCommunicationBackend<double>(rank, 2, envId);
                backend.Initialize();
                var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };
                var model = NewZeroModel(rank + 1);   // rank 0 -> scale 1, rank 1 -> scale 2
                new FSDPOptimizer<double, Vector<double>, Vector<double>>(NewAdam(), config).Optimize(ZeroInput(model));
                results[rank] = model.GetParameters();
                backend.Shutdown();
            }));
        }
        await Task.WhenAll(tasks);

        for (int i = 0; i < ZeroN; i++)
            Assert.Equal(results[0][i], results[1][i], ZeroTol);
        for (int i = 0; i < ZeroN; i++)
            Assert.Equal(single[i], results[0][i], ZeroTol);
    }

    /// <summary>
    /// INVARIANT: the async SGD staleness bound (Stale-Synchronous Parallel, Ho et al. 2013) forces a
    /// synchronization at least every (s+1) steps. staleness 0 ⇒ sync every step (synchronous SGD);
    /// staleness 2 ⇒ sync at iterations 0,3,6,9 only, bounding the maximum drift to 2 steps.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task AsyncSGD_ShouldSync_EnforcesStalenessBound()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, Guid.NewGuid().ToString());
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };

        var sync = new AsyncSGDOptimizer<double, Vector<double>, Vector<double>>(NewAdam(), config, allowStaleness: 0);
        for (int i = 0; i < 6; i++)
            Assert.True(sync.ShouldSync(i), $"staleness 0 must synchronize at every step (i={i})");

        var stale = new AsyncSGDOptimizer<double, Vector<double>, Vector<double>>(NewAdam(), config, allowStaleness: 2);
        for (int i = 0; i < 12; i++)
            Assert.Equal(i % 3 == 0, stale.ShouldSync(i));
        Assert.Equal(2, stale.MaxStaleness);

        backend.Shutdown();
    }

    /// <summary>
    /// INVARIANT: elastic re-sharding on a membership change reconciles state. Rank 0 carries the
    /// authoritative parameters across the rendezvous and broadcasts them, so a worker that (re)joins with
    /// DIFFERENT parameters resumes from the identical vector. Give each rank distinct parameters,
    /// re-synchronize, and assert every rank equals rank 0's authoritative values. The previous placeholder
    /// (no broadcast) would leave the ranks with their distinct starting parameters and fail this.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Elastic_ResynchronizeParameters_BroadcastsRank0StateToAllWorkers()
    {
        await Task.Yield();
        var rank0Params = new double[ZeroN];
        for (int i = 0; i < ZeroN; i++) rank0Params[i] = 0.5 + 0.1 * i;

        var envId = Guid.NewGuid().ToString();
        var results = new Vector<double>[2];
        var tasks = new List<Task>();
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            tasks.Add(Task.Run(() =>
            {
                var backend = new InMemoryCommunicationBackend<double>(rank, 2, envId);
                backend.Initialize();
                var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };

                var model = NewZeroModel();
                var p = new double[ZeroN];
                for (int i = 0; i < ZeroN; i++) p[i] = rank == 0 ? rank0Params[i] : -7.0 - i;  // rank 1 joins DIFFERENT
                model.SetParameters(new Vector<double>(p));

                var elastic = new ElasticOptimizer<double, Vector<double>, Vector<double>>(
                    NewAdam(), config, minWorkers: 1, maxWorkers: 8);
                elastic.ResynchronizeParametersAcrossWorkers(model);
                results[rank] = model.GetParameters();
                backend.Shutdown();
            }));
        }
        await Task.WhenAll(tasks);

        for (int rank = 0; rank < 2; rank++)
            for (int i = 0; i < ZeroN; i++)
                Assert.Equal(rank0Params[i], results[rank][i], ZeroTol);
    }

    // ---- Pure data-parallel (single-step gradient path) invariants --------------------------------
    // Every DP optimizer below now performs the paper-faithful per-step gradient hook (backward-only
    // ComputeGradients -> AllReduce(Average) -> single ApplyGradients from the original params), the SAME
    // shape the ZeRO tests exercise. So each gets the identical transparency invariant: two ranks with
    // rank-dependent gradients (scale 1 and 2) reduce to the mean (scale 1.5) and reconstruct the exact
    // single-rank result at scale 1.5. If the per-step all-reduce were dropped (e.g. reverting to local
    // parameter/optimize averaging) rank 0 and rank 1 would sit at scales 1/2 and diverge from the 1.5
    // reference — so these genuinely exercise the gradient synchronization.

    private static async Task<(Vector<double> single, Vector<double>[] ranks)> RunTwoRankVsSingleAtMeanGradient(
        Func<IShardingConfiguration<double>, ShardedOptimizerBase<double, Vector<double>, Vector<double>>> makeOptimizer)
    {
        var refBackend = new InMemoryCommunicationBackend<double>(0, 1, Guid.NewGuid().ToString());
        refBackend.Initialize();
        var refConfig = new ShardingConfiguration<double>(refBackend) { AutoSyncGradients = true };
        var refModel = NewZeroModel(1.5);
        makeOptimizer(refConfig).Optimize(ZeroInput(refModel));
        var single = refModel.GetParameters();
        refBackend.Shutdown();

        var envId = Guid.NewGuid().ToString();
        var results = new Vector<double>[2];
        var tasks = new List<Task>();
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            tasks.Add(Task.Run(() =>
            {
                var backend = new InMemoryCommunicationBackend<double>(rank, 2, envId);
                backend.Initialize();
                var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };
                var model = NewZeroModel(rank + 1);   // rank 0 -> scale 1, rank 1 -> scale 2
                makeOptimizer(config).Optimize(ZeroInput(model));
                results[rank] = model.GetParameters();
                backend.Shutdown();
            }));
        }
        await Task.WhenAll(tasks);
        return (single, results);
    }

    private static void AssertTwoRankReconstructsSingle(Vector<double> single, Vector<double>[] ranks)
    {
        for (int i = 0; i < ZeroN; i++)
            Assert.Equal(ranks[0][i], ranks[1][i], ZeroTol);       // both ranks identical
        for (int i = 0; i < ZeroN; i++)
            Assert.Equal(single[i], ranks[0][i], ZeroTol);          // == single-rank at the mean gradient
    }

    /// <summary>
    /// INVARIANT: true DDP (Li et al. 2020) is the per-step gradient all-reduce — two ranks reconstruct the
    /// exact single-rank result at the mean gradient. Guards against regressing to local-SGD parameter
    /// averaging (the class's own docs promise gradient averaging).
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DDP_TwoRanks_PerStepGradientAllReduce_MatchesSingleRankAtMeanGradient()
    {
        await Task.Yield();
        var (single, ranks) = await RunTwoRankVsSingleAtMeanGradient(
            cfg => new DDPOptimizer<double, Vector<double>, Vector<double>>(NewAdam(), cfg));
        AssertTwoRankReconstructsSingle(single, ranks);
    }

    /// <summary>
    /// INVARIANT: async SGD at staleness 0 (the synchronous InMemory limit) is the gradient-based
    /// Downpour/parameter-server update (Dean 2012), NOT parameter averaging — two ranks reconstruct the
    /// single-rank result at the mean gradient. Now testable because the optimizer takes the same
    /// backward-only single-step path as DDP/ZeRO (no full wrapped-Optimize loop).
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task AsyncSGD_TwoRanks_StalenessZero_MatchesSingleRankAtMeanGradient()
    {
        await Task.Yield();
        var (single, ranks) = await RunTwoRankVsSingleAtMeanGradient(
            cfg => new AsyncSGDOptimizer<double, Vector<double>, Vector<double>>(NewAdam(), cfg, allowStaleness: 0));
        AssertTwoRankReconstructsSingle(single, ranks);
    }

    /// <summary>
    /// INVARIANT: elastic DDP performs the same per-step gradient all-reduce as DDP on every step where the
    /// worker set is stable — two ranks reconstruct the single-rank result at the mean gradient.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Elastic_TwoRanks_PerStepGradientAllReduce_MatchesSingleRankAtMeanGradient()
    {
        await Task.Yield();
        var (single, ranks) = await RunTwoRankVsSingleAtMeanGradient(
            cfg => new ElasticOptimizer<double, Vector<double>, Vector<double>>(NewAdam(), cfg, minWorkers: 1, maxWorkers: 8));
        AssertTwoRankReconstructsSingle(single, ranks);
    }

    /// <summary>
    /// INVARIANT: gradient-compressed DDP with compression disabled (no quantization, no sparsification) is
    /// exactly true DDP — the compress/decompress pipeline is transparent, so two ranks reconstruct the
    /// single-rank result at the mean gradient. Proves the compression plumbing is wired into the correct
    /// per-step gradient path (compression WITH loss is validated separately by the compression methods).
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task GradientCompression_NoCompression_TwoRanks_MatchesSingleRankAtMeanGradient()
    {
        await Task.Yield();
        var (single, ranks) = await RunTwoRankVsSingleAtMeanGradient(
            cfg => new GradientCompressionOptimizer<double, Vector<double>, Vector<double>>(
                NewAdam(), cfg, compressionRatio: 1.0, useQuantization: false, useSparsification: false));
        AssertTwoRankReconstructsSingle(single, ranks);
    }

    /// <summary>
    /// AUDIT INVARIANT (pipeline parallel, honest scope): the pipeline optimizer performs ONLY the
    /// per-stage parameter update; the micro-batch schedule (GPipe/1F1B/ZB) lives in PipelineParallelModel.
    /// It must therefore REFUSE to advertise numMicroBatches > 1 rather than silently pretend to schedule
    /// micro-batches. This guards the stage-local contract that also justifies doing no cross-stage reduce.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task PipelineParallelOptimizer_MultiMicroBatch_RejectedToDeferToModelSchedule()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, Guid.NewGuid().ToString());
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };

        // numMicroBatches == 1 is the honest per-stage-update contract and must construct fine.
        var ok = new PipelineParallelOptimizer<double, Vector<double>, Vector<double>>(NewAdam(), config, numMicroBatches: 1);
        Assert.NotNull(ok);

        // numMicroBatches > 1 would be a false claim at the optimizer level -> rejected.
        Assert.Throws<NotSupportedException>(() =>
            new PipelineParallelOptimizer<double, Vector<double>, Vector<double>>(NewAdam(), config, numMicroBatches: 4));

        backend.Shutdown();
    }




    /// <summary>
    /// INVARIANT (end-to-end automatic tensor parallelism): wrapping a real NeuralNetwork MLP in
    /// TensorParallelModel across two ranks makes each rank build its OWN Megatron-partitioned model
    /// (Column/Row weight shards) and run true compute-partitioned inference. The distributed Predict is
    /// bit-identical (to tol) to the non-parallel model's Predict, and both ranks report they used the real
    /// compute-partitioned path (not the replication fallback).
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task TensorParallelModel_ComputePartitioned_PredictEqualsNonParallel()
    {
        await Task.Yield();
        const int inputSize = 4, ffn = 8, outputSize = 3, batch = 2;
        var combined = new Vector<double>(DeterministicVector(ffn * inputSize + ffn + outputSize * ffn + outputSize, 21));
        var X = MatrixToTensor(DeterministicMatrix(batch, inputSize, 22), batch, inputSize);

        var refModel = BuildMlpNetwork(inputSize, ffn, outputSize);
        refModel.SetParameters(combined);
        var yRef = refModel.Predict(X);

        var envId = System.Guid.NewGuid().ToString();
        var results = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>[2];
        var errors = new Exception?[2];
        var usedPartition = new bool[2];
        var tasks = new List<Task>();
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    var backend = new InMemoryCommunicationBackend<double>(rank, 2, envId);
                    backend.Initialize();
                    var config = new ShardingConfiguration<double>(backend);
                    var nn = BuildMlpNetwork(inputSize, ffn, outputSize);
                    nn.SetParameters(combined);
                    var tp = new TensorParallelModel<double, AiDotNet.Tensors.LinearAlgebra.Tensor<double>, AiDotNet.Tensors.LinearAlgebra.Tensor<double>>(nn, config);
                    results[rank] = tp.Predict(X);
                    usedPartition[rank] = (tp.GetModelMetadata().Properties["ComputePartitioned"] as bool?) ?? false;
                    backend.Shutdown();
                }
                catch (Exception ex) { errors[rank] = ex; }
            }));
        }
        await Task.WhenAll(tasks);

        Assert.True(errors[0] is null, "rank 0: " + errors[0]);
        Assert.True(errors[1] is null, "rank 1: " + errors[1]);
        Assert.True(usedPartition[0] && usedPartition[1], "both ranks must use the real compute-partitioned path, not the replication fallback");
        for (int rr = 0; rr < 2; rr++)
            for (int b = 0; b < batch; b++)
                for (int o = 0; o < outputSize; o++)
                    Assert.Equal(yRef[b, o], results[rr][b, o], 9);
    }

    private static AiDotNet.NeuralNetworks.NeuralNetwork<double> BuildMlpNetwork(int inputSize, int ffn, int outputSize)
    {
        var layers = new List<AiDotNet.Interfaces.ILayer<double>>
        {
            new AiDotNet.NeuralNetworks.Layers.FullyConnectedLayer<double>(inputSize, ffn, new AiDotNet.ActivationFunctions.ReLUActivation<double>()),
            new AiDotNet.NeuralNetworks.Layers.FullyConnectedLayer<double>(ffn, outputSize, new AiDotNet.ActivationFunctions.IdentityActivation<double>()),
        };
        var arch = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: inputSize, outputSize: outputSize, layers: layers);
        return new AiDotNet.NeuralNetworks.NeuralNetwork<double>(arch);
    }

    /// <summary>
    /// INVARIANT (activation recompute, honest scope): PipelineParallelModel now ACCEPTS Selective/Full
    /// activation-recompute when the wrapped model is layer-introspectable (its stage forward runs through
    /// GradientCheckpointing over the layer segments), and still REJECTS it for a black-box model (which
    /// cannot be decomposed). And recompute is transparent to the result: training a layered stage with
    /// Selective recompute yields the SAME parameter update as training with RecomputeStrategy.None.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task PipelineActivationRecompute_LayeredAllowedAndTransparent_BlackBoxRejected()
    {
        await Task.Yield();
        const int inputSize = 4, ffn = 8, outputSize = 3, batch = 2;
        var combined = new Vector<double>(DeterministicVector(ffn * inputSize + ffn + outputSize * ffn + outputSize, 31));
        var X = MatrixToTensor(DeterministicMatrix(batch, inputSize, 32), batch, inputSize);
        var Y = MatrixToTensor(DeterministicMatrix(batch, outputSize, 33), batch, outputSize);

        // Black-box (non-layered) model + recompute -> rejected at construction.
        {
            var backend = new InMemoryCommunicationBackend<double>(0, 1, System.Guid.NewGuid().ToString());
            backend.Initialize();
            var cfg = new ShardingConfiguration<double>(backend);
            var blackBox = new DistributedTrainingIntegrationTests.MockDistributedModel(inputSize);
            Assert.Throws<NotSupportedException>(() =>
                new PipelineParallelModel<double, Vector<double>, Vector<double>>(
                    blackBox, cfg, microBatchCount: 1,
                    checkpointConfig: new ActivationCheckpointConfig { Enabled = true, RecomputeStrategy = RecomputeStrategy.Selective }));
            backend.Shutdown();
        }

        // Layered model + recompute -> allowed, and transparent (same result as None).
        var paramsNone = TrainOnePipelineStep(combined, X, Y, inputSize, ffn, outputSize, RecomputeStrategy.None);
        var paramsSelective = TrainOnePipelineStep(combined, X, Y, inputSize, ffn, outputSize, RecomputeStrategy.Selective);
        Assert.Equal(paramsNone.Length, paramsSelective.Length);
        for (int i = 0; i < paramsNone.Length; i++)
            Assert.Equal(paramsNone[i], paramsSelective[i], 9);
    }

    private static Vector<double> TrainOnePipelineStep(
        Vector<double> combined, AiDotNet.Tensors.LinearAlgebra.Tensor<double> X, AiDotNet.Tensors.LinearAlgebra.Tensor<double> Y,
        int inputSize, int ffn, int outputSize, RecomputeStrategy strategy)
    {
        var backend = new InMemoryCommunicationBackend<double>(0, 1, System.Guid.NewGuid().ToString());
        backend.Initialize();
        var cfg = new ShardingConfiguration<double>(backend);
        var nn = BuildMlpNetwork(inputSize, ffn, outputSize);
        nn.SetParameters(combined);
        var ckpt = strategy == RecomputeStrategy.None
            ? new ActivationCheckpointConfig { Enabled = false }
            : new ActivationCheckpointConfig { Enabled = true, RecomputeStrategy = strategy, CheckpointEveryNLayers = 1 };
        var pipe = new PipelineParallelModel<double, AiDotNet.Tensors.LinearAlgebra.Tensor<double>, AiDotNet.Tensors.LinearAlgebra.Tensor<double>>(
            nn, cfg, microBatchCount: 1, checkpointConfig: ckpt);
        pipe.Train(X, Y);
        var p = pipe.GetParameters();
        backend.Shutdown();
        return p;
    }

    /// <summary>
    /// INVARIANT (atomic distributed checkpoint): a sharded optimizer's SaveModel writes each rank's shard
    /// via a two-phase commit and publishes a .committed marker LAST; the checkpoint round-trips through
    /// LoadModel, and a checkpoint WITHOUT the marker (a partial/interrupted save) is rejected rather than
    /// silently loaded.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ShardedCheckpoint_AtomicCommit_RoundTripsAndRejectsUncommitted()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, System.Guid.NewGuid().ToString());
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };
        var adam = NewAdam();
        adam.SetModel(new DistributedTrainingIntegrationTests.MockDistributedModel(8));
        var opt = new ZeRO1Optimizer<double, Vector<double>, Vector<double>>(adam, config);

        var dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidn_ckpt_" + System.Guid.NewGuid().ToString("N"));
        System.IO.Directory.CreateDirectory(dir);
        try
        {
            var path = System.IO.Path.Combine(dir, "ckpt");
            opt.SaveModel(path);

            Assert.True(System.IO.File.Exists(path + ".committed"), "commit marker must be written");
            Assert.True(System.IO.File.Exists(path + ".rank0"), "rank 0 shard must be published");
            Assert.False(System.IO.File.Exists(path + ".rank0.tmp"), "temp file must be renamed away");

            // Committed checkpoint loads.
            var adam2 = NewAdam();
            adam2.SetModel(new DistributedTrainingIntegrationTests.MockDistributedModel(8));
            var opt2 = new ZeRO1Optimizer<double, Vector<double>, Vector<double>>(adam2, config);
            opt2.LoadModel(path);

            // Uncommitted checkpoint (marker removed) is rejected.
            System.IO.File.Delete(path + ".committed");
            Assert.Throws<InvalidOperationException>(() => opt2.LoadModel(path));
        }
        finally
        {
            System.IO.Directory.Delete(dir, recursive: true);
            backend.Shutdown();
        }
    }

    /// <summary>
    /// INVARIANT (elastic optimizer-state transfer): the elastic rendezvous broadcasts rank 0's serialized
    /// optimizer state (Adam m/v etc.) to every worker, so after ResynchronizeOptimizerStateAcrossWorkers
    /// all ranks hold the identical authoritative optimizer state — not just identical parameters.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Elastic_OptimizerStateTransfer_AllRanksMatchRank0()
    {
        await Task.Yield();
        var envId = System.Guid.NewGuid().ToString();
        var serialized = new byte[2][];
        var errors = new Exception?[2];
        var tasks = new List<Task>();
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    var backend = new InMemoryCommunicationBackend<double>(rank, 2, envId);
                    backend.Initialize();
                    var config = new ShardingConfiguration<double>(backend) { AutoSyncGradients = true };
                    var elastic = new ElasticOptimizer<double, Vector<double>, Vector<double>>(
                        NewAdam(), config, minWorkers: 1, maxWorkers: 8);
                    elastic.ResynchronizeOptimizerStateAcrossWorkers();   // collective broadcast from rank 0
                    serialized[rank] = elastic.SerializeWrappedOptimizerForTest();
                    backend.Shutdown();
                }
                catch (Exception ex) { errors[rank] = ex; }
            }));
        }
        await Task.WhenAll(tasks);

        Assert.True(errors[0] is null, "rank 0: " + errors[0]);
        Assert.True(errors[1] is null, "rank 1: " + errors[1]);
        Assert.NotNull(serialized[0]);
        Assert.Equal(serialized[0].Length, serialized[1].Length);
        for (int i = 0; i < serialized[0].Length; i++)
            Assert.Equal(serialized[0][i], serialized[1][i]);
    }

    private static double[,] DeterministicMatrix(int rows, int cols, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var m = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = rng.NextDouble() * 2 - 1;
        return m;
    }

    private static double[] DeterministicVector(int n, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var v = new double[n];
        for (int i = 0; i < n; i++) v[i] = rng.NextDouble() * 2 - 1;
        return v;
    }

    private static AiDotNet.Tensors.LinearAlgebra.Tensor<double> MatrixToTensor(double[,] m, int rows, int cols)
    {
        var t = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { rows, cols });
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[i, j] = m[i, j];
        return t;
    }

    private static AiDotNet.Tensors.LinearAlgebra.Tensor<double> VectorToTensor(double[] v)
    {
        var t = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { v.Length });
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }
}
