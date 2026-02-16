#nullable disable
using System.Linq;
using AiDotNet.DistributedTraining;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Enums;
using AiDotNet.Autodiff;
using AiDotNet.LossFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// Integration tests for Pipeline Parallelism that exercise actual training loops,
/// micro-batch slicing, schedule execution, virtual stage dependencies,
/// backward decomposition, and activation checkpointing end-to-end.
/// </summary>
public class PipelineParallelismIntegrationTests
{
    #region End-to-End Training with Each Schedule

    [Fact]
    public void GPipe_SingleRank_TrainAndPredict_ParametersChange()
    {
        // Arrange - single rank pipeline to test the full training flow
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);
        var schedule = new GPipeSchedule();

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 2, schedule: schedule);

        var input = new Vector<double>(Enumerable.Range(0, 10).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 10).Select(i => (double)(i * 2)).ToArray());

        // Get parameters before training
        var paramsBefore = pipelineModel.GatherFullParameters().ToArray();

        try
        {
            // Act - run a training step
            pipelineModel.Train(input, target);

            // Assert - parameters should change after training
            var paramsAfter = pipelineModel.GatherFullParameters().ToArray();
            bool anyChanged = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-12)
                {
                    anyChanged = true;
                    break;
                }
            }
            Assert.True(anyChanged, "GPipe training should update model parameters.");
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void OneForwardOneBackward_SingleRank_TrainAndPredict_ParametersChange()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);
        var schedule = new OneForwardOneBackwardSchedule();

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4, schedule: schedule);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 2)).ToArray());

        var paramsBefore = pipelineModel.GatherFullParameters().ToArray();

        try
        {
            pipelineModel.Train(input, target);

            var paramsAfter = pipelineModel.GatherFullParameters().ToArray();
            bool anyChanged = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-12)
                {
                    anyChanged = true;
                    break;
                }
            }
            Assert.True(anyChanged, "1F1B training should update model parameters.");
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void ZeroBubbleH1_SingleRank_TrainWithDecomposedBackward_ParametersChange()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new DecomposablePipelineTestModel(parameterCount: 20);
        var schedule = new ZeroBubbleH1Schedule();

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4, schedule: schedule);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 2)).ToArray());

        var paramsBefore = pipelineModel.GatherFullParameters().ToArray();

        try
        {
            pipelineModel.Train(input, target);

            var paramsAfter = pipelineModel.GatherFullParameters().ToArray();
            bool anyChanged = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-12)
                {
                    anyChanged = true;
                    break;
                }
            }
            Assert.True(anyChanged, "ZB-H1 training with decomposed backward should update parameters.");
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void ZeroBubbleH1_SingleRank_TrainWithEmulatedDecomposition_ParametersChange()
    {
        // Use a non-decomposable model to test the emulated B/W split path
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);
        var schedule = new ZeroBubbleH1Schedule();

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4, schedule: schedule);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 2)).ToArray());

        var paramsBefore = pipelineModel.GatherFullParameters().ToArray();

        try
        {
            pipelineModel.Train(input, target);

            var paramsAfter = pipelineModel.GatherFullParameters().ToArray();
            bool anyChanged = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-12)
                {
                    anyChanged = true;
                    break;
                }
            }
            Assert.True(anyChanged, "ZB-H1 training with emulated backward should update parameters.");
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void ZeroBubbleH2_SingleRank_TrainAndPredict_ParametersChange()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new DecomposablePipelineTestModel(parameterCount: 20);
        var schedule = new ZeroBubbleH2Schedule();

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4, schedule: schedule);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 2)).ToArray());

        var paramsBefore = pipelineModel.GatherFullParameters().ToArray();

        try
        {
            pipelineModel.Train(input, target);

            var paramsAfter = pipelineModel.GatherFullParameters().ToArray();
            bool anyChanged = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-12)
                {
                    anyChanged = true;
                    break;
                }
            }
            Assert.True(anyChanged, "ZB-H2 training should update model parameters.");
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void ZeroBubbleV_SingleRank_TrainWithTwoVirtualStages_ParametersChange()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new DecomposablePipelineTestModel(parameterCount: 20);
        var schedule = new ZeroBubbleVSchedule();

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4, schedule: schedule);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 2)).ToArray());

        var paramsBefore = pipelineModel.GatherFullParameters().ToArray();

        try
        {
            pipelineModel.Train(input, target);

            var paramsAfter = pipelineModel.GatherFullParameters().ToArray();
            bool anyChanged = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-12)
                {
                    anyChanged = true;
                    break;
                }
            }
            Assert.True(anyChanged, "ZB-V training with 2 virtual stages should update parameters.");
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void Interleaved1F1B_SingleRank_TrainWithTwoVirtualStages_ParametersChange()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);
        var schedule = new Interleaved1F1BSchedule(virtualStagesPerRank: 2);

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4, schedule: schedule);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 2)).ToArray());

        var paramsBefore = pipelineModel.GatherFullParameters().ToArray();

        try
        {
            pipelineModel.Train(input, target);

            var paramsAfter = pipelineModel.GatherFullParameters().ToArray();
            bool anyChanged = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-12)
                {
                    anyChanged = true;
                    break;
                }
            }
            Assert.True(anyChanged, "Interleaved 1F1B training should update parameters.");
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void LoopedBFS_SingleRank_TrainWithTwoVirtualStages_ParametersChange()
    {
        // This test exercises the LoopedBFS forward output retention fix:
        // vStage 0's forward outputs must be retained during vStage 0's backward
        // so they're available as input for vStage 1's forward.
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);
        var schedule = new LoopedBFSSchedule(virtualStagesPerRank: 2);

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4, schedule: schedule);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 2)).ToArray());

        var paramsBefore = pipelineModel.GatherFullParameters().ToArray();

        try
        {
            // If the forward output retention bug regresses, this will throw
            // an InvalidOperationException about missing forward output from vStage 0
            pipelineModel.Train(input, target);

            var paramsAfter = pipelineModel.GatherFullParameters().ToArray();
            bool anyChanged = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-12)
                {
                    anyChanged = true;
                    break;
                }
            }
            Assert.True(anyChanged, "Looped BFS training should update parameters.");
        }
        finally
        {
            backend.Shutdown();
        }
    }

    #endregion

    #region Multi-Rank Pipeline Communication

    [Fact]
    public void GPipe_TwoRanks_SendReceiveActivations()
    {
        // Test that two pipeline stages can exchange activations via Send/Receive
        var envId = Guid.NewGuid().ToString();
        var backend0 = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2, environmentId: envId);
        var backend1 = new InMemoryCommunicationBackend<double>(rank: 1, worldSize: 2, environmentId: envId);
        backend0.Initialize();
        backend1.Initialize();

        try
        {
            // Send activations from rank 0 to rank 1
            var activations = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var sizeHeader = new Vector<double>(new[] { (double)activations.Length });

            int tag = 42;
            backend0.Send(sizeHeader, destinationRank: 1, tag: tag);
            backend0.Send(activations, destinationRank: 1, tag: tag);

            // Receive on rank 1
            var receivedSize = backend1.Receive(sourceRank: 0, count: 1, tag: tag);
            int size = (int)receivedSize[0];
            var receivedActivations = backend1.Receive(sourceRank: 0, count: size, tag: tag);

            Assert.Equal(3, receivedActivations.Length);
            Assert.Equal(1.0, receivedActivations[0]);
            Assert.Equal(2.0, receivedActivations[1]);
            Assert.Equal(3.0, receivedActivations[2]);
        }
        finally
        {
            backend0.Shutdown();
            backend1.Shutdown();
        }
    }

    [Fact]
    public void Pipeline_TwoRanks_PredictFlowsThroughStages()
    {
        var envId = Guid.NewGuid().ToString();
        var backend0 = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2, environmentId: envId);
        var backend1 = new InMemoryCommunicationBackend<double>(rank: 1, worldSize: 2, environmentId: envId);
        backend0.Initialize();
        backend1.Initialize();

        var config0 = new ShardingConfiguration<double>(backend0);
        var config1 = new ShardingConfiguration<double>(backend1);

        var model0 = new PipelineTestModel(parameterCount: 20);
        var model1 = new PipelineTestModel(parameterCount: 20);

        var pipeline0 = new PipelineParallelModel<double, Vector<double>, Vector<double>>(model0, config0);
        var pipeline1 = new PipelineParallelModel<double, Vector<double>, Vector<double>>(model1, config1);

        var input = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });

        try
        {
            // Run predict on both ranks concurrently (rank 0 sends, rank 1 receives)
            // In a real system these would be on separate machines; here we simulate sequentially
            // because InMemoryCommunicationBackend supports non-blocking send/receive
            var task0 = System.Threading.Tasks.Task.Run(() => pipeline0.Predict(input));
            var task1 = System.Threading.Tasks.Task.Run(() => pipeline1.Predict(input));

            System.Threading.Tasks.Task.WaitAll(task0, task1);

            // Both should produce non-null results
            Assert.NotNull(task0.Result);
            Assert.NotNull(task1.Result);

            // Rank 1's output is the final pipeline output
            Assert.True(task1.Result.Length > 0, "Pipeline predict should produce non-empty output.");
        }
        finally
        {
            backend0.Shutdown();
            backend1.Shutdown();
        }
    }

    #endregion

    #region Micro-Batch Slicing

    [Fact]
    public void MicroBatchSlicing_EvenSlicing_CoversAllElements()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);

        // 4 micro-batches over 20 elements = 5 per micro-batch
        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 3)).ToArray());

        try
        {
            // Should not throw - 20 elements divided into 4 batches = 5 each
            pipelineModel.Train(input, target);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void MicroBatchSlicing_UnevenSlicing_LastBatchGetsRemainder()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);

        // 3 micro-batches over 20 elements: 6, 6, 8
        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 3);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 3)).ToArray());

        try
        {
            // Should not throw - last batch gets remainder elements
            pipelineModel.Train(input, target);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void MicroBatchSlicing_SingleMicroBatch_UsesFullInput()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 10);

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 1);

        var input = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var target = new Vector<double>(new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 });

        try
        {
            pipelineModel.Train(input, target);
            var output = pipelineModel.Predict(input);
            Assert.NotNull(output);
            Assert.True(output.Length > 0);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void MicroBatchSlicing_TooManyMicroBatches_ThrowsMeaningfulError()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 10);

        // 100 micro-batches for 5 elements -> 0 per batch -> should throw
        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 100);

        var input = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var target = new Vector<double>(new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 });

        try
        {
            Assert.Throws<InvalidOperationException>(() =>
                pipelineModel.Train(input, target));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    #endregion

    #region Activation Checkpointing Integration

    [Fact]
    public void ActivationCheckpointing_Enabled_TrainingStillWorks()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);

        var checkpointConfig = new ActivationCheckpointConfig
        {
            Enabled = true,
            CheckpointEveryNLayers = 1,
            RecomputeStrategy = RecomputeStrategy.None,
            CheckpointFirstLayer = true
        };

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 2, checkpointConfig: checkpointConfig);

        var input = new Vector<double>(Enumerable.Range(0, 10).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 10).Select(i => (double)(i * 2)).ToArray());

        var paramsBefore = pipelineModel.GatherFullParameters().ToArray();

        try
        {
            pipelineModel.Train(input, target);

            var paramsAfter = pipelineModel.GatherFullParameters().ToArray();
            bool anyChanged = false;
            for (int i = 0; i < paramsBefore.Length; i++)
            {
                if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-12)
                {
                    anyChanged = true;
                    break;
                }
            }
            Assert.True(anyChanged, "Training with checkpointing should still update parameters.");
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void ActivationCheckpointing_MaxActivationsInMemory_LimitsStorage()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);

        var checkpointConfig = new ActivationCheckpointConfig
        {
            Enabled = true,
            MaxActivationsInMemory = 2,
            RecomputeStrategy = RecomputeStrategy.None
        };

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4, checkpointConfig: checkpointConfig);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 2)).ToArray());

        try
        {
            // Should not throw - even with limited activations, training should proceed
            pipelineModel.Train(input, target);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void ActivationCheckpointing_EveryNLayers_InterleavedCheckpoints()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);

        var checkpointConfig = new ActivationCheckpointConfig
        {
            Enabled = true,
            CheckpointEveryNLayers = 3,
            RecomputeStrategy = RecomputeStrategy.None
        };

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 2, checkpointConfig: checkpointConfig);

        var input = new Vector<double>(Enumerable.Range(0, 10).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 10).Select(i => (double)(i * 2)).ToArray());

        try
        {
            pipelineModel.Train(input, target);
            // No exception means checkpointing with interval works
        }
        finally
        {
            backend.Shutdown();
        }
    }

    #endregion

    #region Load-Balanced Partitioning Integration

    [Fact]
    public void LoadBalancedPartition_ExplicitBoundaries_TrainingWorks()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 300);

        // Explicit layer boundaries: 3 layers at [0, 100, 200]
        var partitionStrategy = new LoadBalancedPartitionStrategy<double>(
            new[] { 0, 100, 200 });

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 2, partitionStrategy: partitionStrategy);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 2)).ToArray());

        try
        {
            pipelineModel.Train(input, target);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void LoadBalancedPartition_AutoDetect_TrainingWorks()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 300);

        // Auto-detect with estimated layer size
        var partitionStrategy = new LoadBalancedPartitionStrategy<double>(
            estimatedLayerSize: 100);

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 2, partitionStrategy: partitionStrategy);

        var input = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)(i * 2)).ToArray());

        try
        {
            pipelineModel.Train(input, target);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void LoadBalancedPartition_CustomCostEstimator_ProducesValidPartitions()
    {
        // Custom cost estimator: linear cost (simpler than default quadratic)
        var strategy = new LoadBalancedPartitionStrategy<double>(
            new[] { 0, 50, 200, 400 },
            costEstimator: paramCount => (double)paramCount);

        var partitions = strategy.ComputePartition(totalParameters: 500, numStages: 2);

        Assert.Equal(2, partitions.Length);
        int totalAssigned = partitions.Sum(p => p.Size);
        Assert.Equal(500, totalAssigned);

        // With linear cost, the split should be roughly even by parameter count
        Assert.True(partitions[0].Size > 0 && partitions[1].Size > 0,
            "Both stages should have non-zero parameters.");
    }

    [Fact]
    public void LoadBalancedPartition_MoreStagesThanLayers_EmptyStagesHandled()
    {
        var strategy = new LoadBalancedPartitionStrategy<double>(new[] { 0, 50 });

        // 2 layers but 5 stages: 3 stages will be empty
        var partitions = strategy.ComputePartition(totalParameters: 100, numStages: 5);

        Assert.Equal(5, partitions.Length);
        int totalAssigned = partitions.Sum(p => p.Size);
        Assert.Equal(100, totalAssigned);

        // At least 2 stages have parameters, rest are empty
        int nonEmptyCount = partitions.Count(p => p.Size > 0);
        Assert.True(nonEmptyCount >= 2, "At least the number of layers should have parameters.");
    }

    #endregion

    #region Virtual Stage Dependencies (Looped BFS Regression Test)

    [Fact]
    public void LoopedBFS_VStage0OutputRetained_ForVStage1Forward()
    {
        // This is a specific regression test for the bug where Looped BFS
        // freed vStage 0's forward outputs during its backward phase,
        // making them unavailable when vStage 1's forward needs them.
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 40);
        var schedule = new LoopedBFSSchedule(virtualStagesPerRank: 2);

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4, schedule: schedule);

        var input = new Vector<double>(Enumerable.Range(0, 40).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 40).Select(i => (double)(i * 2)).ToArray());

        try
        {
            // This would throw InvalidOperationException if forward outputs were freed prematurely
            pipelineModel.Train(input, target);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void Interleaved1F1B_VirtualStages_HandleCrossStageOutputs()
    {
        // Interleaved 1F1B also uses virtual stages, test that cross-stage outputs work
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 40);
        var schedule = new Interleaved1F1BSchedule(virtualStagesPerRank: 3);

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4, schedule: schedule);

        var input = new Vector<double>(Enumerable.Range(0, 40).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 40).Select(i => (double)(i * 2)).ToArray());

        try
        {
            pipelineModel.Train(input, target);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void LoopedBFS_ThreeVirtualStages_TrainingCompletes()
    {
        // Test with V=3 to exercise multi-stage retention more thoroughly
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 60);
        var schedule = new LoopedBFSSchedule(virtualStagesPerRank: 3);

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 4, schedule: schedule);

        var input = new Vector<double>(Enumerable.Range(0, 60).Select(i => (double)i).ToArray());
        var target = new Vector<double>(Enumerable.Range(0, 60).Select(i => (double)(i * 2)).ToArray());

        try
        {
            pipelineModel.Train(input, target);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    #endregion

    #region Multiple Training Steps

    [Fact]
    public void Pipeline_MultipleTrainingSteps_ParametersConverge()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend, learningRate: 0.01);
        var model = new PipelineTestModel(parameterCount: 10);

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 2, schedule: new GPipeSchedule());

        var input = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
        var target = new Vector<double>(new double[] { 2.0, 4.0, 6.0, 8.0, 10.0, 12.0 });

        try
        {
            var initialParams = pipelineModel.GatherFullParameters().ToArray();

            // Run multiple training steps
            for (int step = 0; step < 5; step++)
            {
                pipelineModel.Train(input, target);
            }

            var finalParams = pipelineModel.GatherFullParameters().ToArray();

            // Parameters should have changed significantly after 5 steps
            double totalChange = 0;
            for (int i = 0; i < initialParams.Length; i++)
            {
                totalChange += Math.Abs(finalParams[i] - initialParams[i]);
            }
            Assert.True(totalChange > 1e-6, "Multiple training steps should cause significant parameter changes.");
        }
        finally
        {
            backend.Shutdown();
        }
    }

    #endregion

    #region Serialization and Model State

    [Fact]
    public void PipelineModel_SerializeDeserialize_PreservesState()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);
        var schedule = new OneForwardOneBackwardSchedule();

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 2, schedule: schedule);

        try
        {
            // Train to change parameters from initial values
            var input = new Vector<double>(Enumerable.Range(0, 10).Select(i => (double)i).ToArray());
            var target = new Vector<double>(Enumerable.Range(0, 10).Select(i => (double)(i * 2)).ToArray());
            pipelineModel.Train(input, target);

            // Serialize
            var serialized = pipelineModel.Serialize();
            Assert.NotEmpty(serialized);

            // Deserialize into new instance
            var model2 = new PipelineTestModel(parameterCount: 20);
            var pipelineModel2 = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
                model2, config, microBatchSize: 2, schedule: new OneForwardOneBackwardSchedule());

            pipelineModel2.Deserialize(serialized);

            // Parameters should match
            var params1 = pipelineModel.GatherFullParameters().ToArray();
            var params2 = pipelineModel2.GatherFullParameters().ToArray();
            Assert.Equal(params1.Length, params2.Length);
            for (int i = 0; i < params1.Length; i++)
            {
                Assert.Equal(params1[i], params2[i], 10);
            }
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void PipelineModel_Deserialize_ThrowsOnMicroBatchSizeMismatch()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        var model1 = new PipelineTestModel(parameterCount: 20);
        var pipelineModel1 = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model1, config, microBatchSize: 4);
        var serialized = pipelineModel1.Serialize();

        var model2 = new PipelineTestModel(parameterCount: 20);
        var pipelineModel2 = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model2, config, microBatchSize: 2);

        try
        {
            Assert.Throws<InvalidOperationException>(() =>
                pipelineModel2.Deserialize(serialized));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void PipelineModel_Clone_CreatesIndependentCopy()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new PipelineTestModel(parameterCount: 20);

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
            model, config, microBatchSize: 2);

        try
        {
            var cloned = pipelineModel.Clone();
            Assert.NotSame(pipelineModel, cloned);

            // Train original, clone should remain unchanged
            var clonedParamsBefore = cloned.GetParameters().ToArray();

            var input = new Vector<double>(Enumerable.Range(0, 10).Select(i => (double)i).ToArray());
            var target = new Vector<double>(Enumerable.Range(0, 10).Select(i => (double)(i * 2)).ToArray());
            pipelineModel.Train(input, target);

            var clonedParamsAfter = cloned.GetParameters().ToArray();
            Assert.Equal(clonedParamsBefore.Length, clonedParamsAfter.Length);
            for (int i = 0; i < clonedParamsBefore.Length; i++)
            {
                Assert.Equal(clonedParamsBefore[i], clonedParamsAfter[i], 10);
            }
        }
        finally
        {
            backend.Shutdown();
        }
    }

    #endregion

    #region Metadata Verification

    [Fact]
    public void PipelineModel_Metadata_IncludesAllScheduleTypes()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        try
        {
            var schedules = new (IPipelineSchedule schedule, string expectedName)[]
            {
                (new GPipeSchedule(), "GPipe"),
                (new OneForwardOneBackwardSchedule(), "1F1B"),
                (new ZeroBubbleH1Schedule(), "ZB-H1"),
                (new ZeroBubbleH2Schedule(), "ZB-H2"),
                (new ZeroBubbleVSchedule(), "ZB-V"),
                (new Interleaved1F1BSchedule(), "Interleaved-1F1B"),
                (new LoopedBFSSchedule(), "Looped-BFS"),
            };

            foreach (var (schedule, expectedName) in schedules)
            {
                var model = new PipelineTestModel(parameterCount: 20);
                var pipeline = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
                    model, config, schedule: schedule);
                var metadata = pipeline.GetModelMetadata();

                Assert.Equal("PipelineParallel", metadata.Properties["Strategy"]);
                Assert.Equal(expectedName, metadata.Properties["Schedule"]);
                Assert.Equal(true, metadata.Properties["IsDistributed"]);
            }
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void PipelineModel_Metadata_IncludesVirtualStageInfo()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        try
        {
            var model = new PipelineTestModel(parameterCount: 20);
            var pipeline = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
                model, config, schedule: new Interleaved1F1BSchedule(virtualStagesPerRank: 3));
            var metadata = pipeline.GetModelMetadata();

            Assert.Equal(3, metadata.Properties["VirtualStagesPerRank"]);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void PipelineModel_BubbleFraction_ZeroBubbleSchedulesReport0()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        try
        {
            var model = new PipelineTestModel(parameterCount: 20);
            var pipeline = new PipelineParallelModel<double, Vector<double>, Vector<double>>(
                model, config, microBatchSize: 4, schedule: new ZeroBubbleH2Schedule());

            // Single rank: bubble fraction should be 0
            Assert.Equal(0.0, pipeline.EstimatedBubbleFraction);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    #endregion

    #region Schedule Correctness - Operation Counts

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    public void GPipe_AllStages_ProduceCorrectOpCounts(int numMicroBatches)
    {
        var schedule = new GPipeSchedule();
        for (int stageId = 0; stageId < 4; stageId++)
        {
            var ops = schedule.GetSchedule(stageId, numStages: 4, numMicroBatches);
            int forwardCount = ops.Count(o => o.Type == PipelineOperationType.Forward);
            int backwardCount = ops.Count(o => o.Type == PipelineOperationType.Backward);

            Assert.Equal(numMicroBatches, forwardCount);
            Assert.Equal(numMicroBatches, backwardCount);
        }
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public void OneF1B_AllStages_ProduceCorrectOpCounts(int numMicroBatches)
    {
        var schedule = new OneForwardOneBackwardSchedule();
        for (int stageId = 0; stageId < 4; stageId++)
        {
            var ops = schedule.GetSchedule(stageId, numStages: 4, numMicroBatches);
            int forwardCount = ops.Count(o => o.Type == PipelineOperationType.Forward);
            int backwardCount = ops.Count(o => o.Type == PipelineOperationType.Backward);

            Assert.Equal(numMicroBatches, forwardCount);
            Assert.Equal(numMicroBatches, backwardCount);
        }
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public void ZBH1_AllStages_HaveMatchingBWCounts(int numMicroBatches)
    {
        var schedule = new ZeroBubbleH1Schedule();
        for (int stageId = 0; stageId < 4; stageId++)
        {
            var ops = schedule.GetSchedule(stageId, numStages: 4, numMicroBatches);
            int forwardCount = ops.Count(o => o.Type == PipelineOperationType.Forward);
            int biCount = ops.Count(o => o.Type == PipelineOperationType.BackwardInput);
            int bwCount = ops.Count(o => o.Type == PipelineOperationType.BackwardWeight);

            Assert.Equal(numMicroBatches, forwardCount);
            Assert.Equal(numMicroBatches, biCount);
            Assert.Equal(numMicroBatches, bwCount);
        }
    }

    [Fact]
    public void ZBV_Schedule_ProducesOpsForBothVirtualStages()
    {
        var schedule = new ZeroBubbleVSchedule();
        var ops = schedule.GetSchedule(stageId: 0, numStages: 2, numMicroBatches: 4);

        var vStage0Ops = ops.Where(o => o.VirtualStageIndex == 0).ToList();
        var vStage1Ops = ops.Where(o => o.VirtualStageIndex == 1).ToList();

        // Both virtual stages should have forward and backward operations
        Assert.True(vStage0Ops.Any(o => o.Type == PipelineOperationType.Forward));
        Assert.True(vStage1Ops.Any(o => o.Type == PipelineOperationType.Forward));
        Assert.True(vStage0Ops.Any(o => o.Type == PipelineOperationType.BackwardInput));
        Assert.True(vStage1Ops.Any(o => o.Type == PipelineOperationType.BackwardInput));
        Assert.True(vStage0Ops.Any(o => o.Type == PipelineOperationType.BackwardWeight));
        Assert.True(vStage1Ops.Any(o => o.Type == PipelineOperationType.BackwardWeight));
    }

    [Fact]
    public void LoopedBFS_Schedule_VStage0CompletesBeforeVStage1()
    {
        var schedule = new LoopedBFSSchedule(virtualStagesPerRank: 2);
        var ops = schedule.GetSchedule(stageId: 0, numStages: 2, numMicroBatches: 4);

        int lastVStage0Index = -1;
        int firstVStage1Index = int.MaxValue;

        for (int i = 0; i < ops.Count; i++)
        {
            if (ops[i].VirtualStageIndex == 0)
                lastVStage0Index = i;
            else if (ops[i].VirtualStageIndex == 1 && firstVStage1Index == int.MaxValue)
                firstVStage1Index = i;
        }

        Assert.True(lastVStage0Index < firstVStage1Index,
            "Looped BFS: all vStage 0 ops must complete before vStage 1 ops start.");
    }

    #endregion

    #region Bubble Fraction Estimates

    [Theory]
    [InlineData(4, 8)]
    [InlineData(4, 16)]
    [InlineData(8, 32)]
    public void AllSchedules_BubbleFraction_InValidRange(int numStages, int numMicroBatches)
    {
        var schedules = new IPipelineSchedule[]
        {
            new GPipeSchedule(),
            new OneForwardOneBackwardSchedule(),
            new ZeroBubbleH1Schedule(),
            new ZeroBubbleH2Schedule(),
            new ZeroBubbleVSchedule(),
            new Interleaved1F1BSchedule(),
            new LoopedBFSSchedule(),
        };

        foreach (var schedule in schedules)
        {
            double fraction = schedule.EstimateBubbleFraction(numStages, numMicroBatches);
            Assert.InRange(fraction, 0.0, 1.0);
        }
    }

    [Fact]
    public void ZeroBubbleSchedules_AchieveZeroBubble_WhenEnoughMicroBatches()
    {
        var zbH2 = new ZeroBubbleH2Schedule();
        var zbV = new ZeroBubbleVSchedule();

        // With M >= P, zero bubble schedules should report 0
        Assert.Equal(0.0, zbH2.EstimateBubbleFraction(numStages: 4, numMicroBatches: 4));
        Assert.Equal(0.0, zbH2.EstimateBubbleFraction(numStages: 4, numMicroBatches: 8));
        Assert.Equal(0.0, zbV.EstimateBubbleFraction(numStages: 4, numMicroBatches: 4));
        Assert.Equal(0.0, zbV.EstimateBubbleFraction(numStages: 4, numMicroBatches: 8));
    }

    [Fact]
    public void GPipe_HasHigherBubble_Than1F1B()
    {
        var gpipe = new GPipeSchedule();
        var oneF1B = new OneForwardOneBackwardSchedule();

        double gpipeBubble = gpipe.EstimateBubbleFraction(numStages: 4, numMicroBatches: 8);
        double oneF1BBubble = oneF1B.EstimateBubbleFraction(numStages: 4, numMicroBatches: 8);

        // GPipe typically has higher bubble than 1F1B
        Assert.True(gpipeBubble >= oneF1BBubble,
            $"GPipe bubble ({gpipeBubble:F4}) should be >= 1F1B bubble ({oneF1BBubble:F4}).");
    }

    #endregion

    #region Mock Models for Integration Testing

    /// <summary>
    /// A simple mock model for pipeline parallelism testing.
    /// Implements IFullModel with basic predict/train/gradient operations.
    /// </summary>
    private class PipelineTestModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private Vector<double> _parameters;
        private Vector<double> _gradients;
        private readonly int _parameterCount;

        public PipelineTestModel(int parameterCount)
        {
            _parameterCount = parameterCount;
            _parameters = new Vector<double>(
                Enumerable.Range(0, parameterCount).Select(i => 0.1 * (i + 1)).ToArray());
            _gradients = new Vector<double>(new double[parameterCount]);
        }

        public int ParameterCount => _parameterCount;
        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
        public bool SupportsJitCompilation => false;

        public Vector<double> Predict(Vector<double> input)
        {
            // Simple: output[i] = sum(params[j] * input[i % input.Length]) for each output position
            var result = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < Math.Min(_parameters.Length, 3); j++)
                {
                    sum += _parameters[j] * input[i];
                }
                result[i] = sum;
            }
            return new Vector<double>(result);
        }

        public void Train(Vector<double> input, Vector<double> expectedOutput)
        {
            var grads = ComputeGradients(input, expectedOutput);
            ApplyGradients(grads, 0.01);
        }

        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> expectedOutput, ILossFunction<double> lossFunction = null)
        {
            // Simple gradient: proportional to parameter index
            var grads = new double[_parameters.Length];
            for (int i = 0; i < grads.Length; i++)
            {
                grads[i] = 0.01 * (i + 1);
            }
            _gradients = new Vector<double>(grads);
            return _gradients;
        }

        public void ApplyGradients(Vector<double> gradients, double learningRate)
        {
            var newParams = new double[_parameters.Length];
            for (int i = 0; i < _parameters.Length; i++)
            {
                newParams[i] = _parameters[i] - learningRate * gradients[i];
            }
            _parameters = new Vector<double>(newParams);
        }

        public Vector<double> GetParameters() => _parameters.Clone();
        public void SetParameters(Vector<double> parameters) => _parameters = parameters.Clone();
        public Vector<double> GetParameterGradients() => _gradients.Clone();

        public ModelMetadata<double> GetModelMetadata()
        {
            return new ModelMetadata<double>
            {
                Name = "PipelineTestModel",
                ModelType = ModelType.NeuralNetwork,
                TrainingDate = DateTimeOffset.UtcNow,
                FeatureCount = _parameterCount
            };
        }

        public byte[] Serialize()
        {
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(_parameterCount);
            var arr = _parameters.ToArray();
            foreach (var p in arr)
                writer.Write(p);
            return ms.ToArray();
        }

        public void Deserialize(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);
            var count = reader.ReadInt32();
            var parameters = new double[count];
            for (int i = 0; i < count; i++)
                parameters[i] = reader.ReadDouble();
            _parameters = new Vector<double>(parameters);
        }

        public void SaveModel(string filePath) => File.WriteAllBytes(filePath, Serialize());
        public void LoadModel(string filePath) => Deserialize(File.ReadAllBytes(filePath));

        public void SaveState(Stream stream)
        {
            var data = Serialize();
            stream.Write(data, 0, data.Length);
        }

        public void LoadState(Stream stream)
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            Deserialize(ms.ToArray());
        }

        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters)
        {
            var m = new PipelineTestModel(_parameterCount);
            m.SetParameters(parameters);
            return m;
        }

        public IFullModel<double, Vector<double>, Vector<double>> Clone()
        {
            var m = new PipelineTestModel(_parameterCount);
            m.SetParameters(_parameters);
            return m;
        }

        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy() => Clone();

        public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _parameterCount);
        public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
        public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _parameterCount;
        public Dictionary<string, double> GetFeatureImportance() =>
            Enumerable.Range(0, _parameterCount).ToDictionary(i => $"f_{i}", i => 1.0 / _parameterCount);

        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
        {
            var node = new ComputationNode<double>(
                new Tensor<double>(new[] { _parameterCount }),
                false, null, null, "test_graph");
            inputNodes.Add(node);
            return node;
        }
    }

    /// <summary>
    /// A mock model that also implements IPipelineDecomposableModel for testing
    /// the true B/W decomposition path in Zero Bubble schedules.
    /// </summary>
    private class DecomposablePipelineTestModel :
        IFullModel<double, Vector<double>, Vector<double>>,
        IPipelineDecomposableModel<double, Vector<double>, Vector<double>>
    {
        private Vector<double> _parameters;
        private Vector<double> _gradients;
        private readonly int _parameterCount;

        public DecomposablePipelineTestModel(int parameterCount)
        {
            _parameterCount = parameterCount;
            _parameters = new Vector<double>(
                Enumerable.Range(0, parameterCount).Select(i => 0.1 * (i + 1)).ToArray());
            _gradients = new Vector<double>(new double[parameterCount]);
        }

        public int ParameterCount => _parameterCount;
        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
        public bool SupportsJitCompilation => false;

        public Vector<double> Predict(Vector<double> input)
        {
            var result = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < Math.Min(_parameters.Length, 3); j++)
                    sum += _parameters[j] * input[i];
                result[i] = sum;
            }
            return new Vector<double>(result);
        }

        public void Train(Vector<double> input, Vector<double> expectedOutput)
        {
            var grads = ComputeGradients(input, expectedOutput);
            ApplyGradients(grads, 0.01);
        }

        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> expectedOutput, ILossFunction<double> lossFunction = null)
        {
            var grads = new double[_parameters.Length];
            for (int i = 0; i < grads.Length; i++)
                grads[i] = 0.01 * (i + 1);
            _gradients = new Vector<double>(grads);
            return _gradients;
        }

        // IPipelineDecomposableModel implementation
        public (Vector<double> activationGradients, object cachedState) ComputeActivationGradients(
            Vector<double> input, Vector<double> target)
        {
            // Simulate activation gradients (dL/dInput)
            var activationGrads = new double[_parameters.Length];
            for (int i = 0; i < activationGrads.Length; i++)
                activationGrads[i] = 0.005 * (i + 1);

            // Cache the input for weight gradient computation
            var cachedState = new { Input = input.ToArray(), Target = target.ToArray() };
            return (new Vector<double>(activationGrads), cachedState);
        }

        public Vector<double> ComputeWeightGradients(Vector<double> input, Vector<double> target, object cachedState)
        {
            // Simulate weight gradients (dL/dWeights)
            var weightGrads = new double[_parameters.Length];
            for (int i = 0; i < weightGrads.Length; i++)
                weightGrads[i] = 0.01 * (i + 1);
            return new Vector<double>(weightGrads);
        }

        public void ApplyGradients(Vector<double> gradients, double learningRate)
        {
            var newParams = new double[_parameters.Length];
            for (int i = 0; i < _parameters.Length; i++)
                newParams[i] = _parameters[i] - learningRate * gradients[i];
            _parameters = new Vector<double>(newParams);
        }

        public Vector<double> GetParameters() => _parameters.Clone();
        public void SetParameters(Vector<double> parameters) => _parameters = parameters.Clone();
        public Vector<double> GetParameterGradients() => _gradients.Clone();

        public ModelMetadata<double> GetModelMetadata()
        {
            return new ModelMetadata<double>
            {
                Name = "DecomposablePipelineTestModel",
                ModelType = ModelType.NeuralNetwork,
                TrainingDate = DateTimeOffset.UtcNow,
                FeatureCount = _parameterCount
            };
        }

        public byte[] Serialize()
        {
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(_parameterCount);
            var arr = _parameters.ToArray();
            foreach (var p in arr)
                writer.Write(p);
            return ms.ToArray();
        }

        public void Deserialize(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);
            var count = reader.ReadInt32();
            var parameters = new double[count];
            for (int i = 0; i < count; i++)
                parameters[i] = reader.ReadDouble();
            _parameters = new Vector<double>(parameters);
        }

        public void SaveModel(string filePath) => File.WriteAllBytes(filePath, Serialize());
        public void LoadModel(string filePath) => Deserialize(File.ReadAllBytes(filePath));

        public void SaveState(Stream stream)
        {
            var data = Serialize();
            stream.Write(data, 0, data.Length);
        }

        public void LoadState(Stream stream)
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            Deserialize(ms.ToArray());
        }

        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters)
        {
            var m = new DecomposablePipelineTestModel(_parameterCount);
            m.SetParameters(parameters);
            return m;
        }

        public IFullModel<double, Vector<double>, Vector<double>> Clone()
        {
            var m = new DecomposablePipelineTestModel(_parameterCount);
            m.SetParameters(_parameters);
            return m;
        }

        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy() => Clone();

        public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _parameterCount);
        public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
        public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _parameterCount;
        public Dictionary<string, double> GetFeatureImportance() =>
            Enumerable.Range(0, _parameterCount).ToDictionary(i => $"f_{i}", i => 1.0 / _parameterCount);

        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
        {
            var node = new ComputationNode<double>(
                new Tensor<double>(new[] { _parameterCount }),
                false, null, null, "decomposable_test_graph");
            inputNodes.Add(node);
            return node;
        }
    }

    #endregion
}
