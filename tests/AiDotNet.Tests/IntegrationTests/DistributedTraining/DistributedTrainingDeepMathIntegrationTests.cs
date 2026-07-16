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

    private static DistributedTrainingIntegrationTests.MockDistributedModel NewZeroModel()
        => new DistributedTrainingIntegrationTests.MockDistributedModel(ZeroN);

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
        // Single-rank reference.
        var refBackend = new InMemoryCommunicationBackend<double>(0, 1, Guid.NewGuid().ToString());
        refBackend.Initialize();
        var refConfig = new ShardingConfiguration<double>(refBackend) { AutoSyncGradients = true };
        var refModel = NewZeroModel();
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
                var model = NewZeroModel();
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
        var refBackend = new InMemoryCommunicationBackend<double>(0, 1, Guid.NewGuid().ToString());
        refBackend.Initialize();
        var refConfig = new ShardingConfiguration<double>(refBackend) { AutoSyncGradients = true };
        var refModel = NewZeroModel();
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
                var model = NewZeroModel();
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
                    var model = NewZeroModel();
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
    /// == full matmul, i.e. the partitioning is mathematically transparent.
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
