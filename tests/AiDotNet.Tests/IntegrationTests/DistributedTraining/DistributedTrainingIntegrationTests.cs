using Xunit;
using AiDotNet.DistributedTraining;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Enums;
using AiDotNet.Autodiff;
using AiDotNet.LossFunctions;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// Comprehensive integration tests for the DistributedTraining module.
/// Tests cover all distributed training strategies including DDP, FSDP, ZeRO, Pipeline, and Tensor parallelism.
/// </summary>
public class DistributedTrainingIntegrationTests
{
    private const int TestVectorSize = 8;
    private const int DefaultWorldSize = 4;

    #region InMemoryCommunicationBackend Tests

    [Fact]
    public void InMemoryBackend_Constructor_InitializesCorrectly()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4);

        Assert.Equal(0, backend.Rank);
        Assert.Equal(4, backend.WorldSize);
        Assert.False(backend.IsInitialized);
    }

    [Fact]
    public void InMemoryBackend_Initialize_SetsIsInitialized()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2, envId);

        backend.Initialize();

        Assert.True(backend.IsInitialized);

        backend.Shutdown();
    }

    [Fact]
    public void InMemoryBackend_Constructor_ThrowsOnInvalidRank()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: -1, worldSize: 4));

        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 4, worldSize: 4));
    }

    [Fact]
    public void InMemoryBackend_Constructor_ThrowsOnInvalidWorldSize()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 0));

        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: -1));
    }

    [Fact]
    public void InMemoryBackend_AllReduce_SingleProcess_ReturnsOriginal()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, envId);
        backend.Initialize();

        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var original = data.Clone();

        backend.AllReduce(data, ReductionOperation.Sum);

        // With single process, data should remain unchanged
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(original[i], data[i]);
        }

        backend.Shutdown();
    }

    [Fact]
    public async Task InMemoryBackend_AllReduce_MultiProcess_SumsCorrectly()
    {
        var envId = Guid.NewGuid().ToString();
        const int worldSize = 4;

        var tasks = Enumerable.Range(0, worldSize).Select(rank => Task.Run(() =>
        {
            var backend = new InMemoryCommunicationBackend<double>(rank, worldSize, envId);
            backend.Initialize();

            // Each rank contributes [rank+1, rank+1, rank+1, rank+1]
            var data = new Vector<double>(new double[] { rank + 1.0, rank + 1.0, rank + 1.0, rank + 1.0 });

            backend.AllReduce(data, ReductionOperation.Sum);

            // Expected sum: 1+2+3+4 = 10 for each element
            return data;
        })).ToArray();

        var results = await Task.WhenAll(tasks);

        foreach (var result in results)
        {
            foreach (var value in result.ToArray())
            {
                Assert.Equal(10.0, value, 0.001);
            }
        }

        InMemoryCommunicationBackend<double>.ClearEnvironment(envId);
    }

    [Fact]
    public async Task InMemoryBackend_AllReduce_Average_CalculatesCorrectly()
    {
        var envId = Guid.NewGuid().ToString();
        const int worldSize = 4;

        var tasks = Enumerable.Range(0, worldSize).Select(rank => Task.Run(() =>
        {
            var backend = new InMemoryCommunicationBackend<double>(rank, worldSize, envId);
            backend.Initialize();

            // Each rank contributes [rank+1, rank+1, rank+1, rank+1]
            var data = new Vector<double>(new double[] { rank + 1.0, rank + 1.0, rank + 1.0, rank + 1.0 });

            backend.AllReduce(data, ReductionOperation.Average);

            // Expected average: (1+2+3+4)/4 = 2.5 for each element
            return data;
        })).ToArray();

        var results = await Task.WhenAll(tasks);

        foreach (var result in results)
        {
            foreach (var value in result.ToArray())
            {
                Assert.Equal(2.5, value, 0.001);
            }
        }

        InMemoryCommunicationBackend<double>.ClearEnvironment(envId);
    }

    [Fact]
    public async Task InMemoryBackend_Broadcast_DistributesFromRoot()
    {
        var envId = Guid.NewGuid().ToString();
        const int worldSize = 4;
        const int rootRank = 0;

        var tasks = Enumerable.Range(0, worldSize).Select(rank => Task.Run(() =>
        {
            var backend = new InMemoryCommunicationBackend<double>(rank, worldSize, envId);
            backend.Initialize();

            Vector<double>? data = null;
            if (rank == rootRank)
            {
                data = new Vector<double>(new double[] { 42.0, 84.0, 126.0, 168.0 });
            }
            else
            {
                data = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0 });
            }

            var result = backend.Broadcast(data, rootRank);

            return result;
        })).ToArray();

        var results = await Task.WhenAll(tasks);

        // All ranks should have root's data
        foreach (var result in results)
        {
            Assert.Equal(42.0, result[0], 0.001);
            Assert.Equal(84.0, result[1], 0.001);
            Assert.Equal(126.0, result[2], 0.001);
            Assert.Equal(168.0, result[3], 0.001);
        }

        InMemoryCommunicationBackend<double>.ClearEnvironment(envId);
    }

    [Fact]
    public async Task InMemoryBackend_AllGather_ConcatenatesAllData()
    {
        var envId = Guid.NewGuid().ToString();
        const int worldSize = 4;

        var tasks = Enumerable.Range(0, worldSize).Select(rank => Task.Run(() =>
        {
            var backend = new InMemoryCommunicationBackend<double>(rank, worldSize, envId);
            backend.Initialize();

            // Each rank contributes [rank*2, rank*2+1]
            var localData = new Vector<double>(new double[] { rank * 2.0, rank * 2.0 + 1.0 });

            var gathered = backend.AllGather(localData);

            return gathered;
        })).ToArray();

        var results = await Task.WhenAll(tasks);

        // All ranks should have concatenated data [0,1,2,3,4,5,6,7]
        foreach (var result in results)
        {
            Assert.Equal(8, result.Length);
            for (int i = 0; i < 8; i++)
            {
                Assert.Equal((double)i, result[i], 0.001);
            }
        }

        InMemoryCommunicationBackend<double>.ClearEnvironment(envId);
    }

    [Fact]
    public async Task InMemoryBackend_Scatter_DistributesChunks()
    {
        var envId = Guid.NewGuid().ToString();
        const int worldSize = 4;
        const int rootRank = 0;

        var tasks = Enumerable.Range(0, worldSize).Select(rank => Task.Run(() =>
        {
            var backend = new InMemoryCommunicationBackend<double>(rank, worldSize, envId);
            backend.Initialize();

            Vector<double>? data = null;
            if (rank == rootRank)
            {
                // Root has data [0,1,2,3,4,5,6,7]
                data = new Vector<double>(new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });
            }
            else
            {
                data = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
            }

            var chunk = backend.Scatter(data, rootRank);

            return (rank, chunk);
        })).ToArray();

        var results = await Task.WhenAll(tasks);

        // Each rank should get their chunk
        foreach (var (rank, chunk) in results)
        {
            Assert.Equal(2, chunk.Length);
            Assert.Equal(rank * 2.0, chunk[0], 0.001);
            Assert.Equal(rank * 2.0 + 1.0, chunk[1], 0.001);
        }

        InMemoryCommunicationBackend<double>.ClearEnvironment(envId);
    }

    [Fact]
    public async Task InMemoryBackend_Barrier_SynchronizesAllProcesses()
    {
        var envId = Guid.NewGuid().ToString();
        const int worldSize = 4;
        var arrivedAtBarrier = new int[worldSize];

        var tasks = Enumerable.Range(0, worldSize).Select(rank => Task.Run(() =>
        {
            var backend = new InMemoryCommunicationBackend<double>(rank, worldSize, envId);
            backend.Initialize();

            // Simulate some work with different durations
            Thread.Sleep(rank * 10);

            // Mark arrival
            Interlocked.Exchange(ref arrivedAtBarrier[rank], 1);

            // Wait at barrier
            backend.Barrier();

            // After barrier, all should have arrived
            var allArrived = arrivedAtBarrier.All(a => a == 1);

            backend.Shutdown();

            return allArrived;
        })).ToArray();

        var results = await Task.WhenAll(tasks);

        Assert.All(results, r => Assert.True(r));

        InMemoryCommunicationBackend<double>.ClearEnvironment(envId);
    }

    [Fact]
    public async Task InMemoryBackend_SendReceive_PointToPoint()
    {
        var envId = Guid.NewGuid().ToString();
        const int worldSize = 2;

        var tasks = new Task<Vector<double>>[worldSize];

        // Rank 0 sends, Rank 1 receives
        tasks[0] = Task.Run(() =>
        {
            var backend = new InMemoryCommunicationBackend<double>(0, worldSize, envId);
            backend.Initialize();

            var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
            backend.Send(data, destinationRank: 1, tag: 0);

            return data;
        });

        tasks[1] = Task.Run(() =>
        {
            var backend = new InMemoryCommunicationBackend<double>(1, worldSize, envId);
            backend.Initialize();

            var received = backend.Receive(sourceRank: 0, count: 4, tag: 0);

            return received;
        });

        var results = await Task.WhenAll(tasks);

        // Rank 1 should have received rank 0's data
        Assert.Equal(1.0, results[1][0], 0.001);
        Assert.Equal(2.0, results[1][1], 0.001);
        Assert.Equal(3.0, results[1][2], 0.001);
        Assert.Equal(4.0, results[1][3], 0.001);

        InMemoryCommunicationBackend<double>.ClearEnvironment(envId);
    }

    [Fact]
    public void InMemoryBackend_ReduceScatter_WorksCorrectly()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, envId);
        backend.Initialize();

        var data = new Vector<double>(new double[] { 4.0, 8.0, 12.0, 16.0 });

        var result = backend.ReduceScatter(data, ReductionOperation.Sum);

        // Single process: returns the whole vector
        Assert.Equal(4, result.Length);

        backend.Shutdown();
    }

    #endregion

    #region ShardingConfiguration Tests

    [Fact]
    public void ShardingConfiguration_DefaultSettings_AreCorrect()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();

        var config = new ShardingConfiguration<double>(backend);

        Assert.True(config.AutoSyncGradients);
        Assert.Equal(1024, config.MinimumParameterGroupSize);
        Assert.False(config.EnableGradientCompression);
        Assert.Equal(backend, config.CommunicationBackend);

        backend.Shutdown();
    }

    [Fact]
    public void ShardingConfiguration_CreateDefault_UsesDefaults()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();

        var config = ShardingConfiguration<double>.CreateDefault(backend);

        Assert.True(config.AutoSyncGradients);
        Assert.Equal(1024, config.MinimumParameterGroupSize);
        Assert.False(config.EnableGradientCompression);

        backend.Shutdown();
    }

    [Fact]
    public void ShardingConfiguration_HighBandwidth_UsesOptimizedSettings()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();

        var config = ShardingConfiguration<double>.CreateForHighBandwidth(backend);

        Assert.True(config.AutoSyncGradients);
        Assert.Equal(512, config.MinimumParameterGroupSize);
        Assert.False(config.EnableGradientCompression);

        backend.Shutdown();
    }

    [Fact]
    public void ShardingConfiguration_LowBandwidth_UsesOptimizedSettings()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();

        var config = ShardingConfiguration<double>.CreateForLowBandwidth(backend);

        Assert.True(config.AutoSyncGradients);
        Assert.Equal(4096, config.MinimumParameterGroupSize);
        Assert.True(config.EnableGradientCompression);

        backend.Shutdown();
    }

    [Fact]
    public void ShardingConfiguration_Constructor_ThrowsOnNullBackend()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new ShardingConfiguration<double>(null!));
    }

    #endregion

    #region DDPModel Tests

    [Fact]
    public void DDPModel_Constructor_InitializesCorrectly()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var ddpModel = new DDPModel<double, Vector<double>, Vector<double>>(model, config);

        Assert.Equal(0, ddpModel.Rank);
        Assert.Equal(4, ddpModel.WorldSize);
        Assert.NotNull(ddpModel.WrappedModel);

        backend.Shutdown();
    }

    [Fact]
    public void DDPModel_GetModelMetadata_IncludesDistributedInfo()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var ddpModel = new DDPModel<double, Vector<double>, Vector<double>>(model, config);

        var metadata = ddpModel.GetModelMetadata();

        Assert.True(metadata.Properties["IsDistributed"] as bool? ?? false);
        Assert.Equal("DDP", metadata.Properties["Strategy"] as string ?? "");
        Assert.Equal(4, metadata.Properties["WorldSize"] as int? ?? 0);
        Assert.Equal(0, metadata.Properties["Rank"] as int? ?? -1);

        backend.Shutdown();
    }

    [Fact]
    public void DDPModel_LocalParameterShard_ContainsFullParameters()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var ddpModel = new DDPModel<double, Vector<double>, Vector<double>>(model, config);

        // DDP keeps full parameters on each process
        Assert.Equal(8, ddpModel.LocalParameterShard.Length);

        backend.Shutdown();
    }

    [Fact]
    public void DDPModel_GatherFullParameters_ReturnsSameAsLocal()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var ddpModel = new DDPModel<double, Vector<double>, Vector<double>>(model, config);

        var gathered = ddpModel.GatherFullParameters();

        // In DDP, gathered should equal local
        Assert.Equal(ddpModel.LocalParameterShard.Length, gathered.Length);

        backend.Shutdown();
    }

    [Fact]
    public void DDPModel_Clone_CreatesNewInstance()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var ddpModel = new DDPModel<double, Vector<double>, Vector<double>>(model, config);

        var cloned = ddpModel.Clone();

        Assert.NotSame(ddpModel, cloned);
        Assert.IsType<DDPModel<double, Vector<double>, Vector<double>>>(cloned);

        backend.Shutdown();
    }

    [Fact]
    public void DDPModel_WithParameters_CreatesNewModel()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var ddpModel = new DDPModel<double, Vector<double>, Vector<double>>(model, config);

        var newParams = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var newModel = ddpModel.WithParameters(newParams);

        Assert.NotSame(ddpModel, newModel);

        backend.Shutdown();
    }

    [Fact]
    public void DDPModel_Serialize_Deserialize_PreservesState()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var ddpModel = new DDPModel<double, Vector<double>, Vector<double>>(model, config);

        var serialized = ddpModel.Serialize();
        Assert.NotEmpty(serialized);

        var ddpModel2 = new DDPModel<double, Vector<double>, Vector<double>>(new MockDistributedModel(8), config);
        ddpModel2.Deserialize(serialized);

        Assert.Equal(ddpModel.WorldSize, ddpModel2.WorldSize);
        Assert.Equal(ddpModel.Rank, ddpModel2.Rank);

        backend.Shutdown();
    }

    #endregion

    #region FSDPModel Tests

    [Fact]
    public void FSDPModel_Constructor_InitializesCorrectly()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var fsdpModel = new FSDPModel<double, Vector<double>, Vector<double>>(model, config);

        Assert.Equal(0, fsdpModel.Rank);
        Assert.Equal(4, fsdpModel.WorldSize);
        Assert.NotNull(fsdpModel.WrappedModel);

        backend.Shutdown();
    }

    [Fact]
    public void FSDPModel_LocalParameterShard_ContainsPartialParameters()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var fsdpModel = new FSDPModel<double, Vector<double>, Vector<double>>(model, config);

        // FSDP shards parameters across processes
        // With 8 parameters and 4 processes, each gets 2
        Assert.Equal(2, fsdpModel.LocalParameterShard.Length);

        backend.Shutdown();
    }

    [Fact]
    public void FSDPModel_GetModelMetadata_IncludesDistributedInfo()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var fsdpModel = new FSDPModel<double, Vector<double>, Vector<double>>(model, config);

        var metadata = fsdpModel.GetModelMetadata();

        Assert.True(metadata.Properties["IsDistributed"] as bool? ?? false);
        Assert.Equal("FSDP", metadata.Properties["Strategy"] as string ?? "");
        Assert.Equal(4, metadata.Properties["WorldSize"] as int? ?? 0);

        backend.Shutdown();
    }

    [Fact]
    public void FSDPModel_Clone_CreatesNewInstance()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 1, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var fsdpModel = new FSDPModel<double, Vector<double>, Vector<double>>(model, config);

        var cloned = fsdpModel.Clone();

        Assert.NotSame(fsdpModel, cloned);
        Assert.IsType<FSDPModel<double, Vector<double>, Vector<double>>>(cloned);

        backend.Shutdown();
    }

    #endregion

    #region ZeRO Models Tests

    [Fact]
    public void ZeRO1Model_Constructor_InitializesCorrectly()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var zero1Model = new ZeRO1Model<double, Vector<double>, Vector<double>>(model, config);

        Assert.Equal(0, zero1Model.Rank);
        Assert.Equal(4, zero1Model.WorldSize);

        backend.Shutdown();
    }

    [Fact]
    public void ZeRO1Model_GetModelMetadata_IncludesStrategy()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var zero1Model = new ZeRO1Model<double, Vector<double>, Vector<double>>(model, config);

        var metadata = zero1Model.GetModelMetadata();

        Assert.True(metadata.Properties["IsDistributed"] as bool? ?? false);
        Assert.Equal("ZeRO-1", metadata.Properties["Strategy"] as string ?? "");

        backend.Shutdown();
    }

    [Fact]
    public void ZeRO2Model_Constructor_InitializesCorrectly()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var zero2Model = new ZeRO2Model<double, Vector<double>, Vector<double>>(model, config);

        Assert.Equal(0, zero2Model.Rank);
        Assert.Equal(4, zero2Model.WorldSize);

        backend.Shutdown();
    }

    [Fact]
    public void ZeRO2Model_GetModelMetadata_IncludesStrategy()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var zero2Model = new ZeRO2Model<double, Vector<double>, Vector<double>>(model, config);

        var metadata = zero2Model.GetModelMetadata();

        Assert.True(metadata.Properties["IsDistributed"] as bool? ?? false);
        Assert.Equal("ZeRO-2", metadata.Properties["Strategy"] as string ?? "");

        backend.Shutdown();
    }

    [Fact]
    public void ZeRO3Model_Constructor_InitializesCorrectly()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var zero3Model = new ZeRO3Model<double, Vector<double>, Vector<double>>(model, config);

        Assert.Equal(0, zero3Model.Rank);
        Assert.Equal(4, zero3Model.WorldSize);

        backend.Shutdown();
    }

    [Fact]
    public void ZeRO3Model_GetModelMetadata_IncludesStrategy()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var zero3Model = new ZeRO3Model<double, Vector<double>, Vector<double>>(model, config);

        var metadata = zero3Model.GetModelMetadata();

        Assert.True(metadata.Properties["IsDistributed"] as bool? ?? false);
        Assert.Equal("ZeRO-3", metadata.Properties["Strategy"] as string ?? "");

        backend.Shutdown();
    }

    [Fact]
    public void ZeRO3Model_LocalParameterShard_IsSharded()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var zero3Model = new ZeRO3Model<double, Vector<double>, Vector<double>>(model, config);

        // ZeRO-3 shards parameters across processes
        Assert.Equal(2, zero3Model.LocalParameterShard.Length);

        backend.Shutdown();
    }

    #endregion

    #region PipelineParallelModel Tests

    [Fact]
    public void PipelineParallelModel_Constructor_InitializesCorrectly()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(model, config);

        Assert.Equal(0, pipelineModel.Rank);
        Assert.Equal(4, pipelineModel.WorldSize);

        backend.Shutdown();
    }

    [Fact]
    public void PipelineParallelModel_GetModelMetadata_IncludesStrategy()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var pipelineModel = new PipelineParallelModel<double, Vector<double>, Vector<double>>(model, config);

        var metadata = pipelineModel.GetModelMetadata();

        Assert.True(metadata.Properties["IsDistributed"] as bool? ?? false);
        Assert.Equal("PipelineParallel", metadata.Properties["Strategy"] as string ?? "");

        backend.Shutdown();
    }

    #endregion

    #region TensorParallelModel Tests

    [Fact]
    public void TensorParallelModel_Constructor_InitializesCorrectly()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var tensorModel = new TensorParallelModel<double, Vector<double>, Vector<double>>(model, config);

        Assert.Equal(0, tensorModel.Rank);
        Assert.Equal(4, tensorModel.WorldSize);

        backend.Shutdown();
    }

    [Fact]
    public void TensorParallelModel_GetModelMetadata_IncludesStrategy()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var tensorModel = new TensorParallelModel<double, Vector<double>, Vector<double>>(model, config);

        var metadata = tensorModel.GetModelMetadata();

        Assert.True(metadata.Properties["IsDistributed"] as bool? ?? false);
        Assert.Equal("TensorParallel", metadata.Properties["Strategy"] as string ?? "");

        backend.Shutdown();
    }

    #endregion

    #region HybridShardedModel Tests

    [Fact]
    public void HybridShardedModel_Constructor_InitializesCorrectly()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var hybridModel = new HybridShardedModel<double, Vector<double>, Vector<double>>(model, config, pipelineParallelSize: 2);

        Assert.Equal(0, hybridModel.Rank);
        Assert.Equal(4, hybridModel.WorldSize);

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_GetModelMetadata_IncludesStrategy()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var hybridModel = new HybridShardedModel<double, Vector<double>, Vector<double>>(model, config, pipelineParallelSize: 2);

        var metadata = hybridModel.GetModelMetadata();

        Assert.True(metadata.Properties["IsDistributed"] as bool? ?? false);
        Assert.Equal("3D-Parallelism (Hybrid)", metadata.Properties["Strategy"] as string ?? "");

        backend.Shutdown();
    }

    #endregion

    #region Optimizer Tests

    // Note: DDPOptimizer, FSDPOptimizer, ZeRO1Optimizer require wrapped optimizers
    // and IShardingConfiguration - they don't take sharded models directly.
    // The optimizer wrappers handle gradient synchronization for existing optimizers.

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void InMemoryBackend_MultipleEnvironments_AreIsolated()
    {
        var envId1 = Guid.NewGuid().ToString();
        var envId2 = Guid.NewGuid().ToString();

        var backend1 = new InMemoryCommunicationBackend<double>(0, 1, envId1);
        var backend2 = new InMemoryCommunicationBackend<double>(0, 1, envId2);

        backend1.Initialize();
        backend2.Initialize();

        // Operations in one environment shouldn't affect the other
        var data1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var data2 = new Vector<double>(new double[] { 10.0, 20.0, 30.0, 40.0 });

        backend1.AllReduce(data1, ReductionOperation.Sum);
        backend2.AllReduce(data2, ReductionOperation.Sum);

        Assert.Equal(1.0, data1[0], 0.001);
        Assert.Equal(10.0, data2[0], 0.001);

        backend1.Shutdown();
        backend2.Shutdown();
    }

    [Fact]
    public void ShardedModel_Constructor_ThrowsOnNullModel()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentNullException>(() =>
            new DDPModel<double, Vector<double>, Vector<double>>(null!, config));

        backend.Shutdown();
    }

    [Fact]
    public void ShardedModel_Constructor_ThrowsOnNullConfig()
    {
        var model = new MockDistributedModel(8);

        Assert.Throws<ArgumentNullException>(() =>
            new DDPModel<double, Vector<double>, Vector<double>>(model, null!));
    }

    [Fact]
    public void DDPModel_Deserialize_ThrowsOnWorldSizeMismatch()
    {
        var envId1 = Guid.NewGuid().ToString();
        var envId2 = Guid.NewGuid().ToString();

        // Create and serialize with worldSize=2
        var backend1 = new InMemoryCommunicationBackend<double>(0, 2, envId1);
        backend1.Initialize();
        var config1 = new ShardingConfiguration<double>(backend1);
        var model1 = new MockDistributedModel(8);
        var ddpModel1 = new DDPModel<double, Vector<double>, Vector<double>>(model1, config1);
        var serialized = ddpModel1.Serialize();
        backend1.Shutdown();

        // Try to deserialize with worldSize=4
        var backend2 = new InMemoryCommunicationBackend<double>(0, 4, envId2);
        backend2.Initialize();
        var config2 = new ShardingConfiguration<double>(backend2);
        var model2 = new MockDistributedModel(8);
        var ddpModel2 = new DDPModel<double, Vector<double>, Vector<double>>(model2, config2);

        Assert.Throws<InvalidOperationException>(() => ddpModel2.Deserialize(serialized));

        backend2.Shutdown();
    }

    [Fact]
    public void AllDistributedStrategies_HaveUniqueMetadataStrategies()
    {
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(0, 4, envId);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);
        var model = new MockDistributedModel(8);

        var strategies = new List<string>();

        strategies.Add(new DDPModel<double, Vector<double>, Vector<double>>(model.Clone() as MockDistributedModel, config)
            .GetModelMetadata().Properties["Strategy"] as string ?? "");
        strategies.Add(new FSDPModel<double, Vector<double>, Vector<double>>(model.Clone() as MockDistributedModel, config)
            .GetModelMetadata().Properties["Strategy"] as string ?? "");
        strategies.Add(new ZeRO1Model<double, Vector<double>, Vector<double>>(model.Clone() as MockDistributedModel, config)
            .GetModelMetadata().Properties["Strategy"] as string ?? "");
        strategies.Add(new ZeRO2Model<double, Vector<double>, Vector<double>>(model.Clone() as MockDistributedModel, config)
            .GetModelMetadata().Properties["Strategy"] as string ?? "");
        strategies.Add(new ZeRO3Model<double, Vector<double>, Vector<double>>(model.Clone() as MockDistributedModel, config)
            .GetModelMetadata().Properties["Strategy"] as string ?? "");
        strategies.Add(new PipelineParallelModel<double, Vector<double>, Vector<double>>(model.Clone() as MockDistributedModel, config)
            .GetModelMetadata().Properties["Strategy"] as string ?? "");
        strategies.Add(new TensorParallelModel<double, Vector<double>, Vector<double>>(model.Clone() as MockDistributedModel, config)
            .GetModelMetadata().Properties["Strategy"] as string ?? "");
        strategies.Add(new HybridShardedModel<double, Vector<double>, Vector<double>>(model.Clone() as MockDistributedModel, config, 2)
            .GetModelMetadata().Properties["Strategy"] as string ?? "");

        // All strategies should have unique names
        Assert.Equal(strategies.Count, strategies.Distinct().Count());

        backend.Shutdown();
    }

    #endregion

    #region Mock Model for Testing

    /// <summary>
    /// Mock model that implements IFullModel for distributed training tests.
    /// </summary>
    private class MockDistributedModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private Vector<double> _parameters;
        private Vector<double>? _gradients;
        private readonly int _parameterCount;

        public MockDistributedModel(int parameterCount)
        {
            _parameterCount = parameterCount;
            _parameters = new Vector<double>(Enumerable.Range(0, parameterCount).Select(i => (double)i * 0.1).ToArray());
        }

        public int ParameterCount => _parameterCount;

        public ILossFunction<double> DefaultLossFunction => new AiDotNet.LossFunctions.MeanSquaredErrorLoss<double>();

        public Vector<double> Predict(Vector<double> input)
        {
            // Simple mock prediction
            var result = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                result[i] = input[i] * 2.0;
            }
            return new Vector<double>(result);
        }

        public void Train(Vector<double> input, Vector<double> expectedOutput)
        {
            // Simple mock training - compute gradients and update
            _gradients = ComputeGradients(input, expectedOutput, null);
            ApplyGradients(_gradients, 0.01);
        }

        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> expectedOutput, ILossFunction<double>? lossFunction = null)
        {
            // Mock gradient computation
            var gradients = new double[_parameters.Length];
            for (int i = 0; i < gradients.Length; i++)
            {
                gradients[i] = 0.01 * (i + 1);
            }
            _gradients = new Vector<double>(gradients);
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

        public Vector<double> GetParameters()
        {
            return _parameters.Clone();
        }

        public void SetParameters(Vector<double> parameters)
        {
            _parameters = parameters.Clone();
        }

        public Vector<double> GetParameterGradients()
        {
            return _gradients?.Clone() ?? new Vector<double>(new double[_parameterCount]);
        }

        public ModelMetadata<double> GetModelMetadata()
        {
            return new ModelMetadata<double>
            {
                Name = "MockDistributedModel",
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
            foreach (var param in _parameters.ToArray())
            {
                writer.Write(param);
            }

            return ms.ToArray();
        }

        public void Deserialize(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);

            var count = reader.ReadInt32();
            var parameters = new double[count];
            for (int i = 0; i < count; i++)
            {
                parameters[i] = reader.ReadDouble();
            }
            _parameters = new Vector<double>(parameters);
        }

        public void SaveModel(string filePath)
        {
            File.WriteAllBytes(filePath, Serialize());
        }

        public void LoadModel(string filePath)
        {
            Deserialize(File.ReadAllBytes(filePath));
        }

        public void SaveState(Stream stream)
        {
            var data = Serialize();
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }

        public void LoadState(Stream stream)
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            Deserialize(ms.ToArray());
        }

        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters)
        {
            var newModel = new MockDistributedModel(_parameterCount);
            newModel.SetParameters(parameters);
            return newModel;
        }

        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return Enumerable.Range(0, _parameterCount);
        }

        public void SetActiveFeatureIndices(IEnumerable<int> indices)
        {
            // No-op for mock
        }

        public bool IsFeatureUsed(int featureIndex)
        {
            return featureIndex >= 0 && featureIndex < _parameterCount;
        }

        public Dictionary<string, double> GetFeatureImportance()
        {
            return Enumerable.Range(0, _parameterCount)
                .ToDictionary(i => $"feature_{i}", i => 1.0 / _parameterCount);
        }

        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
        {
            var node = new ComputationNode<double>(
                new Tensor<double>(new[] { _parameterCount }),
                false,
                null,
                null,
                "mock_graph"
            );
            inputNodes.Add(node);
            return node;
        }

        public bool SupportsJitCompilation => false;

        public IFullModel<double, Vector<double>, Vector<double>> Clone()
        {
            var cloned = new MockDistributedModel(_parameterCount);
            cloned.SetParameters(_parameters);
            return cloned;
        }

        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy()
        {
            return Clone();
        }
    }

    #endregion
}
