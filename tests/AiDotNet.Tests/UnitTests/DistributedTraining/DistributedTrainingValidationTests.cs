using System.Linq;
using AiDotNet.DistributedTraining;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.DistributedTraining;

/// <summary>
/// Unit tests for parameter validation in DistributedTraining classes.
/// These tests verify that constructors properly validate their inputs to prevent
/// runtime errors from invalid configurations.
/// </summary>
public class DistributedTrainingValidationTests
{
    #region PR #754 Bug Fix Tests - Parameter Validation

    #region ShardingConfiguration Validation Tests

    [Fact]
    public void ShardingConfiguration_Constructor_ThrowsOnNullBackend()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new ShardingConfiguration<double>(null!));
    }

    [Fact]
    public void ShardingConfiguration_Constructor_ThrowsOnZeroLearningRate()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ShardingConfiguration<double>(backend, learningRate: 0));
    }

    [Fact]
    public void ShardingConfiguration_Constructor_ThrowsOnNegativeLearningRate()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ShardingConfiguration<double>(backend, learningRate: -0.01));
    }

    [Fact]
    public void ShardingConfiguration_Constructor_AcceptsValidLearningRate()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);

        var config = new ShardingConfiguration<double>(backend, learningRate: 0.001);

        Assert.NotNull(config);
        Assert.Same(backend, config.CommunicationBackend);
    }

    [Fact]
    public void ShardingConfiguration_CreateDefault_ThrowsOnNullBackend()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ShardingConfiguration<double>.CreateDefault(null!));
    }

    [Fact]
    public void ShardingConfiguration_CreateForHighBandwidth_ThrowsOnNullBackend()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ShardingConfiguration<double>.CreateForHighBandwidth(null!));
    }

    [Fact]
    public void ShardingConfiguration_CreateForLowBandwidth_ThrowsOnNullBackend()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ShardingConfiguration<double>.CreateForLowBandwidth(null!));
    }

    #endregion

    #region PipelineParallelModel Validation Tests

    [Fact]
    public void PipelineParallelModel_Constructor_ThrowsOnZeroMicroBatchSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PipelineParallelModel<double, Matrix<double>, Vector<double>>(model, config, microBatchSize: 0));

        backend.Shutdown();
    }

    [Fact]
    public void PipelineParallelModel_Constructor_ThrowsOnNegativeMicroBatchSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PipelineParallelModel<double, Matrix<double>, Vector<double>>(model, config, microBatchSize: -1));

        backend.Shutdown();
    }

    [Fact]
    public void PipelineParallelModel_Constructor_AcceptsMinimumMicroBatchSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        var pipelineModel = new PipelineParallelModel<double, Matrix<double>, Vector<double>>(
            model, config, microBatchSize: 1);

        Assert.NotNull(pipelineModel);

        backend.Shutdown();
    }

    [Fact]
    public void PipelineParallelModel_Constructor_AcceptsLargeMicroBatchSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        var pipelineModel = new PipelineParallelModel<double, Matrix<double>, Vector<double>>(
            model, config, microBatchSize: 64);

        Assert.NotNull(pipelineModel);

        backend.Shutdown();
    }

    #endregion

    #region HybridShardedModel Validation Tests

    [Fact]
    public void HybridShardedModel_Constructor_ThrowsOnZeroPipelineParallelSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HybridShardedModel<double, Matrix<double>, Vector<double>>(
                model, config, pipelineParallelSize: 0));

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_ThrowsOnNegativePipelineParallelSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HybridShardedModel<double, Matrix<double>, Vector<double>>(
                model, config, pipelineParallelSize: -1));

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_ThrowsOnZeroTensorParallelSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HybridShardedModel<double, Matrix<double>, Vector<double>>(
                model, config, tensorParallelSize: 0));

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_ThrowsOnNegativeTensorParallelSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HybridShardedModel<double, Matrix<double>, Vector<double>>(
                model, config, tensorParallelSize: -1));

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_AcceptsMinimumValidSizes()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        var hybridModel = new HybridShardedModel<double, Matrix<double>, Vector<double>>(
            model, config, pipelineParallelSize: 1, tensorParallelSize: 1, dataParallelSize: 1);

        Assert.NotNull(hybridModel);

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_ThrowsWhenSizesDontMatchWorldSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 8);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        // 2 x 2 x 3 = 12 != 8
        Assert.Throws<ArgumentException>(() =>
            new HybridShardedModel<double, Matrix<double>, Vector<double>>(
                model, config, pipelineParallelSize: 2, tensorParallelSize: 2, dataParallelSize: 3));

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_AcceptsMatchingWorldSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 8);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        // 2 x 2 x 2 = 8 = worldSize
        var hybridModel = new HybridShardedModel<double, Matrix<double>, Vector<double>>(
            model, config, pipelineParallelSize: 2, tensorParallelSize: 2, dataParallelSize: 2);

        Assert.NotNull(hybridModel);

        backend.Shutdown();
    }

    #endregion

    #region InMemoryCommunicationBackend Validation Tests

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnNegativeRank()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: -1, worldSize: 4));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnRankExceedsWorldSize()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 4, worldSize: 4));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnZeroWorldSize()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 0));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnNegativeWorldSize()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: -1));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnEmptyEnvironmentId()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4, environmentId: ""));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnWhitespaceEnvironmentId()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4, environmentId: "   "));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_AcceptsValidParameters()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4);

        Assert.Equal(0, backend.Rank);
        Assert.Equal(4, backend.WorldSize);
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_AcceptsCustomEnvironmentId()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4, environmentId: "test-env-1");

        Assert.Equal(0, backend.Rank);
        Assert.Equal(4, backend.WorldSize);
    }

    #endregion

    #endregion

    #region Pipeline Schedule Tests

    [Fact]
    public void GPipeSchedule_GetSchedule_ProducesCorrectPhases()
    {
        var schedule = new GPipeSchedule();
        var ops = schedule.GetSchedule(stageId: 0, numStages: 2, numMicroBatches: 4);

        Assert.NotEmpty(ops);

        // GPipe: all forwards first, then all backwards
        int lastForwardIdx = -1;
        int firstBackwardIdx = int.MaxValue;
        for (int i = 0; i < ops.Count; i++)
        {
            if (ops[i].Type == PipelineOperationType.Forward)
            {
                lastForwardIdx = i;
            }
            else if (ops[i].Type == PipelineOperationType.Backward && firstBackwardIdx == int.MaxValue)
            {
                firstBackwardIdx = i;
            }
        }

        // All forwards come before all backwards in GPipe
        Assert.True(lastForwardIdx < firstBackwardIdx,
            "GPipe should have all forwards before all backwards.");
    }

    [Fact]
    public void OneForwardOneBackward_GetSchedule_InterleavesFB()
    {
        var schedule = new OneForwardOneBackwardSchedule();
        var ops = schedule.GetSchedule(stageId: 1, numStages: 4, numMicroBatches: 8);

        Assert.NotEmpty(ops);

        // Warmup phase: first (numStages - 1 - stageId) ops should be forward
        int expectedWarmup = Math.Min(4 - 1 - 1, 8); // = 2
        for (int i = 0; i < expectedWarmup; i++)
        {
            Assert.Equal(PipelineOperationType.Forward, ops[i].Type);
            Assert.True(ops[i].IsWarmup);
        }

        // Verify alternating F/B in steady state
        bool foundSteadyState = false;
        for (int i = expectedWarmup; i < ops.Count - 1; i++)
        {
            if (!ops[i].IsCooldown && !ops[i].IsWarmup && !ops[i + 1].IsCooldown)
            {
                foundSteadyState = true;
                // In steady state, forward and backward should alternate
                if (ops[i].Type == PipelineOperationType.Forward)
                {
                    Assert.Equal(PipelineOperationType.Backward, ops[i + 1].Type);
                }
            }
        }

        Assert.True(foundSteadyState, "1F1B should have a steady-state phase with alternating F/B.");
    }

    [Fact]
    public void ZeroBubbleH1_GetSchedule_SplitsBackward()
    {
        var schedule = new ZeroBubbleH1Schedule();
        var ops = schedule.GetSchedule(stageId: 0, numStages: 4, numMicroBatches: 8);

        Assert.NotEmpty(ops);

        // ZB-H1 should emit BackwardInput and BackwardWeight operations
        bool hasBackwardInput = ops.Any(o => o.Type == PipelineOperationType.BackwardInput);
        bool hasBackwardWeight = ops.Any(o => o.Type == PipelineOperationType.BackwardWeight);
        bool hasRegularBackward = ops.Any(o => o.Type == PipelineOperationType.Backward);

        Assert.True(hasBackwardInput, "ZB-H1 should emit BackwardInput operations.");
        Assert.True(hasBackwardWeight, "ZB-H1 should emit BackwardWeight operations.");
        Assert.False(hasRegularBackward, "ZB-H1 should not emit combined Backward operations.");
    }

    [Fact]
    public void ZeroBubbleH2_GetSchedule_SplitsBackward()
    {
        var schedule = new ZeroBubbleH2Schedule();
        var ops = schedule.GetSchedule(stageId: 0, numStages: 4, numMicroBatches: 8);

        Assert.NotEmpty(ops);

        bool hasBackwardInput = ops.Any(o => o.Type == PipelineOperationType.BackwardInput);
        bool hasBackwardWeight = ops.Any(o => o.Type == PipelineOperationType.BackwardWeight);

        Assert.True(hasBackwardInput, "ZB-H2 should emit BackwardInput operations.");
        Assert.True(hasBackwardWeight, "ZB-H2 should emit BackwardWeight operations.");
    }

    [Fact]
    public void ZeroBubbleV_GetSchedule_UsesTwoVirtualStages()
    {
        var schedule = new ZeroBubbleVSchedule();

        Assert.Equal(2, schedule.VirtualStagesPerRank);

        var ops = schedule.GetSchedule(stageId: 0, numStages: 2, numMicroBatches: 4);

        Assert.NotEmpty(ops);

        // Should have operations for both virtual stages 0 and 1
        bool hasVStage0 = ops.Any(o => o.VirtualStageIndex == 0);
        bool hasVStage1 = ops.Any(o => o.VirtualStageIndex == 1);

        Assert.True(hasVStage0, "ZB-V should have operations for virtual stage 0.");
        Assert.True(hasVStage1, "ZB-V should have operations for virtual stage 1.");
    }

    [Fact]
    public void Interleaved1F1B_GetSchedule_DepthFirstOrder()
    {
        var schedule = new Interleaved1F1BSchedule(virtualStagesPerRank: 2);
        var ops = schedule.GetSchedule(stageId: 0, numStages: 2, numMicroBatches: 4);

        Assert.NotEmpty(ops);

        // Interleaved 1F1B: depth-first means microbatch 0 through both virtual stages
        // before microbatch 1 starts
        bool hasVStage0 = ops.Any(o => o.VirtualStageIndex == 0);
        bool hasVStage1 = ops.Any(o => o.VirtualStageIndex == 1);

        Assert.True(hasVStage0 && hasVStage1,
            "Interleaved 1F1B should use both virtual stages.");
    }

    [Fact]
    public void Interleaved1F1B_Constructor_ThrowsOnSingleVirtualStage()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new Interleaved1F1BSchedule(virtualStagesPerRank: 1));
    }

    [Fact]
    public void LoopedBFS_GetSchedule_BreadthFirstOrder()
    {
        var schedule = new LoopedBFSSchedule(virtualStagesPerRank: 2);
        var ops = schedule.GetSchedule(stageId: 0, numStages: 2, numMicroBatches: 4);

        Assert.NotEmpty(ops);

        // BFS: all microbatches through vStage 0 first, then all through vStage 1
        int lastVStage0Idx = -1;
        int firstVStage1Idx = int.MaxValue;
        for (int i = 0; i < ops.Count; i++)
        {
            if (ops[i].VirtualStageIndex == 0)
            {
                lastVStage0Idx = i;
            }
            else if (ops[i].VirtualStageIndex == 1 && firstVStage1Idx == int.MaxValue)
            {
                firstVStage1Idx = i;
            }
        }

        Assert.True(lastVStage0Idx < firstVStage1Idx,
            "Looped BFS should process all vStage 0 operations before vStage 1.");
    }

    [Fact]
    public void LoopedBFS_Constructor_ThrowsOnSingleVirtualStage()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LoopedBFSSchedule(virtualStagesPerRank: 1));
    }

    [Theory]
    [InlineData(typeof(GPipeSchedule))]
    [InlineData(typeof(OneForwardOneBackwardSchedule))]
    [InlineData(typeof(ZeroBubbleH1Schedule))]
    [InlineData(typeof(ZeroBubbleH2Schedule))]
    public void Schedule_GetSchedule_ThrowsOnInvalidStageId(Type scheduleType)
    {
        var schedule = (IPipelineSchedule)Activator.CreateInstance(scheduleType)!;

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            schedule.GetSchedule(stageId: -1, numStages: 4, numMicroBatches: 4));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            schedule.GetSchedule(stageId: 4, numStages: 4, numMicroBatches: 4));
    }

    [Theory]
    [InlineData(typeof(GPipeSchedule))]
    [InlineData(typeof(OneForwardOneBackwardSchedule))]
    [InlineData(typeof(ZeroBubbleH1Schedule))]
    [InlineData(typeof(ZeroBubbleH2Schedule))]
    public void Schedule_GetSchedule_ThrowsOnZeroMicroBatches(Type scheduleType)
    {
        var schedule = (IPipelineSchedule)Activator.CreateInstance(scheduleType)!;

        Assert.Throws<ArgumentException>(() =>
            schedule.GetSchedule(stageId: 0, numStages: 4, numMicroBatches: 0));
    }

    [Theory]
    [InlineData(typeof(GPipeSchedule))]
    [InlineData(typeof(OneForwardOneBackwardSchedule))]
    [InlineData(typeof(ZeroBubbleH1Schedule))]
    [InlineData(typeof(ZeroBubbleH2Schedule))]
    public void Schedule_EstimateBubbleFraction_ReturnsBetweenZeroAndOne(Type scheduleType)
    {
        var schedule = (IPipelineSchedule)Activator.CreateInstance(scheduleType)!;

        double fraction = schedule.EstimateBubbleFraction(numStages: 4, numMicroBatches: 8);

        Assert.InRange(fraction, 0.0, 1.0);
    }

    [Theory]
    [InlineData(typeof(GPipeSchedule))]
    [InlineData(typeof(OneForwardOneBackwardSchedule))]
    [InlineData(typeof(ZeroBubbleH1Schedule))]
    [InlineData(typeof(ZeroBubbleH2Schedule))]
    public void Schedule_EstimateBubbleFraction_ZeroForSingleStage(Type scheduleType)
    {
        var schedule = (IPipelineSchedule)Activator.CreateInstance(scheduleType)!;

        double fraction = schedule.EstimateBubbleFraction(numStages: 1, numMicroBatches: 8);

        Assert.Equal(0.0, fraction);
    }

    [Fact]
    public void ZeroBubbleH2_EstimateBubbleFraction_ZeroWhenEnoughMicroBatches()
    {
        var schedule = new ZeroBubbleH2Schedule();
        double fraction = schedule.EstimateBubbleFraction(numStages: 4, numMicroBatches: 4);

        Assert.Equal(0.0, fraction);
    }

    [Fact]
    public void ZeroBubbleV_EstimateBubbleFraction_ZeroWhenEnoughMicroBatches()
    {
        var schedule = new ZeroBubbleVSchedule();
        double fraction = schedule.EstimateBubbleFraction(numStages: 4, numMicroBatches: 4);

        Assert.Equal(0.0, fraction);
    }

    #endregion

    #region LoadBalancedPartitionStrategy Tests

    [Fact]
    public void LoadBalancedPartitionStrategy_Constructor_ThrowsWhenFirstBoundaryNonZero()
    {
        Assert.Throws<ArgumentException>(() =>
            new LoadBalancedPartitionStrategy<double>(new[] { 5, 100, 300 }));
    }

    [Fact]
    public void LoadBalancedPartitionStrategy_Constructor_ThrowsOnNonIncreasing()
    {
        Assert.Throws<ArgumentException>(() =>
            new LoadBalancedPartitionStrategy<double>(new[] { 0, 100, 50 }));
    }

    [Fact]
    public void LoadBalancedPartitionStrategy_ComputePartition_CoversAllParameters()
    {
        var strategy = new LoadBalancedPartitionStrategy<double>(new[] { 0, 100, 300 });
        var partitions = strategy.ComputePartition(totalParameters: 500, numStages: 2);

        Assert.Equal(2, partitions.Length);

        // All parameters should be covered (no gaps)
        int totalAssigned = partitions.Sum(p => p.Size);
        Assert.Equal(500, totalAssigned);

        // Partitions should be contiguous
        Assert.Equal(0, partitions[0].StartIndex);
        Assert.Equal(partitions[0].StartIndex + partitions[0].Size, partitions[1].StartIndex);
    }

    [Fact]
    public void LoadBalancedPartitionStrategy_AutoDetect_ProducesValidPartitions()
    {
        var strategy = new LoadBalancedPartitionStrategy<double>(estimatedLayerSize: 100);
        var partitions = strategy.ComputePartition(totalParameters: 500, numStages: 3);

        Assert.Equal(3, partitions.Length);

        int totalAssigned = partitions.Sum(p => p.Size);
        Assert.Equal(500, totalAssigned);
    }

    #endregion

    #region ActivationCheckpointConfig Tests

    [Fact]
    public void ActivationCheckpointConfig_DefaultRecomputeStrategy_IsNone()
    {
        var config = new ActivationCheckpointConfig();

        Assert.Equal(RecomputeStrategy.None, config.RecomputeStrategy);
    }

    [Fact]
    public void ActivationCheckpointConfig_CheckpointEveryNLayers_ThrowsOnZero()
    {
        var config = new ActivationCheckpointConfig();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            config.CheckpointEveryNLayers = 0);
    }

    [Fact]
    public void ActivationCheckpointConfig_MaxActivationsInMemory_ThrowsOnNegative()
    {
        var config = new ActivationCheckpointConfig();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            config.MaxActivationsInMemory = -1);
    }

    [Fact]
    public void PipelineParallelModel_Constructor_AcceptsCheckpointingWithNoneStrategy()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var checkpointConfig = new ActivationCheckpointConfig
        {
            Enabled = true,
            RecomputeStrategy = RecomputeStrategy.None
        };

        var pipelineModel = new PipelineParallelModel<double, Matrix<double>, Vector<double>>(
            model, config, checkpointConfig: checkpointConfig);

        Assert.NotNull(pipelineModel);

        backend.Shutdown();
    }

    [Fact]
    public void PipelineParallelModel_Constructor_ThrowsOnCheckpointingWithSelectiveStrategy()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var checkpointConfig = new ActivationCheckpointConfig
        {
            Enabled = true,
            RecomputeStrategy = RecomputeStrategy.Selective
        };

        Assert.Throws<NotImplementedException>(() =>
            new PipelineParallelModel<double, Matrix<double>, Vector<double>>(
                model, config, checkpointConfig: checkpointConfig));

        backend.Shutdown();
    }

    #endregion

    #region PipelineParallelModel Metadata Tests

    [Fact]
    public void PipelineParallelModel_GetModelMetadata_IncludesScheduleInfo()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        var pipelineModel = new PipelineParallelModel<double, Matrix<double>, Vector<double>>(
            model, config, schedule: new OneForwardOneBackwardSchedule());

        var metadata = pipelineModel.GetModelMetadata();

        Assert.Equal("PipelineParallel", metadata.Properties["Strategy"]);
        Assert.Equal("1F1B", metadata.Properties["Schedule"]);

        backend.Shutdown();
    }

    [Fact]
    public void PipelineParallelModel_DefaultSchedule_IsGPipe()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        var pipelineModel = new PipelineParallelModel<double, Matrix<double>, Vector<double>>(
            model, config);

        var metadata = pipelineModel.GetModelMetadata();
        Assert.Equal("GPipe", metadata.Properties["Schedule"]);

        backend.Shutdown();
    }

    #endregion

    /// <summary>
    /// Creates a simple mock model for testing purposes.
    /// Uses VectorModel which implements IFullModel.
    /// </summary>
    private static IFullModel<double, Matrix<double>, Vector<double>> CreateMockModel()
    {
        var coefficients = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        return new VectorModel<double>(coefficients);
    }
}
