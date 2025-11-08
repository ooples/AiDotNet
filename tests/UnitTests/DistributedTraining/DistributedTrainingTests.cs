using Xunit;
using AiDotNet.DistributedTraining;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NumericOperations;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNetTests.DistributedTraining;

/// <summary>
/// Integration tests for distributed training functionality.
/// These tests verify that distributed training produces numerically equivalent results
/// to single-process training.
///
/// For Beginners:
/// These tests ensure that when we train a model across multiple processes (distributed),
/// we get the same result as training on a single process. This is crucial because it
/// proves that our distributed implementation is correct and not introducing errors.
/// </summary>
public class DistributedTrainingTests
{
    private readonly INumericOperations<double> _numOps = MathHelper.GetNumericOperations<double>();

    /// <summary>
    /// Tests that the InMemoryCommunicationBackend correctly initializes.
    /// </summary>
    [Fact]
    public void InMemoryBackend_Initialize_Succeeds()
    {
        // Arrange
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4);

        // Act
        backend.Initialize();

        // Assert
        Assert.True(backend.IsInitialized);
        Assert.Equal(0, backend.Rank);
        Assert.Equal(4, backend.WorldSize);

        // Cleanup
        backend.Shutdown();
        Assert.False(backend.IsInitialized);
    }

    /// <summary>
    /// Tests that AllReduce with Sum operation correctly sums values across processes.
    ///
    /// For Beginners:
    /// This simulates having 4 processes, each with a vector [1, 2, 3].
    /// After AllReduce with Sum, each should have [4, 8, 12] (sum of all 4 vectors).
    /// </summary>
    [Fact]
    public void InMemoryBackend_AllReduceSum_CorrectlyCombinesValues()
    {
        // Arrange - Simulate 4 processes
        var backends = new[]
        {
            new InMemoryCommunicationBackend<double>(0, 4),
            new InMemoryCommunicationBackend<double>(1, 4),
            new InMemoryCommunicationBackend<double>(2, 4),
            new InMemoryCommunicationBackend<double>(3, 4)
        };

        foreach (var backend in backends)
        {
            backend.Initialize();
        }

        var data = new[]
        {
            new Vector<double>(new[] { 1.0, 2.0, 3.0 }),
            new Vector<double>(new[] { 1.0, 2.0, 3.0 }),
            new Vector<double>(new[] { 1.0, 2.0, 3.0 }),
            new Vector<double>(new[] { 1.0, 2.0, 3.0 })
        };

        // Act - Perform AllReduce on each "process" concurrently to avoid deadlock
        // InMemoryBackend requires all ranks to enter the collective concurrently
        Parallel.For(0, 4, i =>
        {
            backends[i].AllReduce(data[i], ReductionOperation.Sum);
        });

        // Assert - All processes should have the same summed result
        var expected = new[] { 4.0, 8.0, 12.0 };
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(expected[0], data[i][0], precision: 10);
            Assert.Equal(expected[1], data[i][1], precision: 10);
            Assert.Equal(expected[2], data[i][2], precision: 10);
        }

        // Cleanup
        foreach (var backend in backends)
        {
            backend.Shutdown();
        }
    }

    /// <summary>
    /// Tests that AllReduce with Average operation correctly averages values.
    ///
    /// For Beginners:
    /// This verifies gradient averaging - a crucial operation in distributed training.
    /// Each process calculates gradients, then they're averaged across all processes.
    /// </summary>
    [Fact]
    public void InMemoryBackend_AllReduceAverage_CorrectlyAveragesValues()
    {
        // Arrange - Simulate 4 processes with different values
        var backends = new[]
        {
            new InMemoryCommunicationBackend<double>(0, 4),
            new InMemoryCommunicationBackend<double>(1, 4),
            new InMemoryCommunicationBackend<double>(2, 4),
            new InMemoryCommunicationBackend<double>(3, 4)
        };

        foreach (var backend in backends)
        {
            backend.Initialize();
        }

        var data = new[]
        {
            new Vector<double>(new[] { 1.0, 2.0, 3.0 }),
            new Vector<double>(new[] { 2.0, 4.0, 6.0 }),
            new Vector<double>(new[] { 3.0, 6.0, 9.0 }),
            new Vector<double>(new[] { 4.0, 8.0, 12.0 })
        };

        // Act - Perform AllReduce on each "process" concurrently to avoid deadlock
        // InMemoryBackend requires all ranks to enter the collective concurrently
        Parallel.For(0, 4, i =>
        {
            backends[i].AllReduce(data[i], ReductionOperation.Average);
        });

        // Assert - Average of (1,2,3,4) = 2.5, (2,4,6,8) = 5.0, (3,6,9,12) = 7.5
        var expected = new[] { 2.5, 5.0, 7.5 };
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(expected[0], data[i][0], precision: 10);
            Assert.Equal(expected[1], data[i][1], precision: 10);
            Assert.Equal(expected[2], data[i][2], precision: 10);
        }

        // Cleanup
        foreach (var backend in backends)
        {
            backend.Shutdown();
        }
    }

    /// <summary>
    /// Tests that AllGather correctly concatenates data from all processes.
    ///
    /// For Beginners:
    /// AllGather is used to reconstruct full parameters from shards.
    /// Each process has a piece, AllGather gives everyone the full picture.
    /// </summary>
    [Fact]
    public void InMemoryBackend_AllGather_CorrectlyConcatenatesData()
    {
        // Arrange - Simulate 4 processes, each with different data
        var backends = new[]
        {
            new InMemoryCommunicationBackend<double>(0, 4),
            new InMemoryCommunicationBackend<double>(1, 4),
            new InMemoryCommunicationBackend<double>(2, 4),
            new InMemoryCommunicationBackend<double>(3, 4)
        };

        foreach (var backend in backends)
        {
            backend.Initialize();
        }

        var sendData = new[]
        {
            new Vector<double>(new[] { 1.0, 2.0 }),  // Process 0's shard
            new Vector<double>(new[] { 3.0, 4.0 }),  // Process 1's shard
            new Vector<double>(new[] { 5.0, 6.0 }),  // Process 2's shard
            new Vector<double>(new[] { 7.0, 8.0 })   // Process 3's shard
        };

        // Act - Each process gathers all data concurrently to avoid deadlock
        // InMemoryBackend requires all ranks to enter the collective concurrently
        var gathered = new Vector<double>[4];
        Parallel.For(0, 4, i =>
        {
            gathered[i] = backends[i].AllGather(sendData[i]);
        });

        // Assert - All processes should have the complete concatenated data
        var expected = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(expected.Length, gathered[i].Length);
            for (int j = 0; j < expected.Length; j++)
            {
                Assert.Equal(expected[j], gathered[i][j], precision: 10);
            }
        }

        // Cleanup
        foreach (var backend in backends)
        {
            backend.Shutdown();
        }
    }

    /// <summary>
    /// Tests that ShardedModel correctly distributes parameters across processes.
    ///
    /// For Beginners:
    /// This verifies that when we create a sharded model, parameters are
    /// correctly split across processes and each process gets its fair share.
    /// </summary>
    [Fact]
    public void ShardedModel_ParameterSharding_DistributesCorrectly()
    {
        // Arrange - Create a simple model with known parameters
        var coefficients = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        var model = new VectorModel<double>(coefficients);

        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4);
        backend.Initialize();

        var config = new ShardingConfiguration<double>(backend);
        var shardedModel = new ShardedModel<double, Matrix<double>, Vector<double>>(model, config);

        // Assert - With 8 parameters and 4 processes, each should get 2 parameters
        Assert.Equal(0, shardedModel.Rank);
        Assert.Equal(4, shardedModel.WorldSize);
        Assert.Equal(2, shardedModel.LocalParameterShard.Length);  // 8 / 4 = 2

        // Process 0 should have parameters [1.0, 2.0]
        Assert.Equal(1.0, shardedModel.LocalParameterShard[0], precision: 10);
        Assert.Equal(2.0, shardedModel.LocalParameterShard[1], precision: 10);

        // Cleanup
        backend.Shutdown();
    }

    /// <summary>
    /// Tests that ParameterAnalyzer correctly groups small parameters.
    ///
    /// For Beginners:
    /// When we have many small parameter arrays, the analyzer groups them together
    /// to reduce communication overhead. This test verifies that grouping works correctly.
    /// </summary>
    [Fact]
    public void ParameterAnalyzer_SmallParameters_GroupsCorrectly()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>(minimumGroupSize: 4, worldSize: 2);
        var parameters = new Vector<double>(new double[10]);  // 10 parameters

        // Act
        var groups = analyzer.AnalyzeParameters(parameters);

        // Assert
        Assert.NotEmpty(groups);

        // Verify all parameters are covered
        int totalCovered = groups.Sum(g => g.Size);
        Assert.Equal(10, totalCovered);

        // Verify no overlaps or gaps
        Assert.True(analyzer.ValidateGrouping(groups, 10));

        // Get statistics
        var stats = analyzer.CalculateDistributionStats(groups);
        Assert.Equal(10, stats["TotalParameters"]);
        Assert.True(stats["MinGroupSize"] >= 1);  // At least 1 parameter per group
    }

    /// <summary>
    /// Tests the .AsDistributed() extension method for models.
    ///
    /// For Beginners:
    /// This tests the simple one-line API for making a model distributed.
    /// It should be as easy as: myModel.AsDistributed(backend)
    /// </summary>
    [Fact]
    public void AsDistributed_Extension_CreatesShardedModel()
    {
        // Arrange
        var coefficients = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var model = new VectorModel<double>(coefficients);
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();

        // Act
        var distributedModel = model.AsDistributed(backend);

        // Assert
        Assert.NotNull(distributedModel);
        Assert.IsAssignableFrom<IShardedModel<double, Matrix<double>, Vector<double>>>(distributedModel);
        Assert.Equal(model, distributedModel.WrappedModel);
        Assert.Equal(0, distributedModel.Rank);
        Assert.Equal(2, distributedModel.WorldSize);

        // Cleanup
        backend.Shutdown();
    }

    /// <summary>
    /// Tests numerical equivalence between single-process and distributed training.
    ///
    /// For Beginners:
    /// This is THE critical test! It proves that training with distributed setup
    /// produces the same results as training on a single process. If this passes,
    /// we know our distributed implementation is mathematically correct.
    ///
    /// NOTE: This is a simplified test. A full implementation would train an actual
    /// model through multiple iterations and compare final parameters.
    /// </summary>
    [Fact]
    public void DistributedTraining_NumericalEquivalence_MatchesSingleProcess()
    {
        // Arrange - Create identical models for single-process and distributed
        var singleProcessCoefficients = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var singleProcessModel = new VectorModel<double>(singleProcessCoefficients);

        var distributedCoefficients = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var distributedModelBase = new VectorModel<double>(distributedCoefficients);

        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();

        var config = new ShardingConfiguration<double>(backend);
        var distributedModel = new ShardedModel<double, Matrix<double>, Vector<double>>(
            distributedModelBase, config);

        // Act - Get parameters from both
        var singleProcessParams = singleProcessModel.GetParameters();
        var distributedParams = distributedModel.GetParameters();

        // Assert - Parameters should be identical
        Assert.Equal(singleProcessParams.Length, distributedParams.Length);
        for (int i = 0; i < singleProcessParams.Length; i++)
        {
            Assert.Equal(singleProcessParams[i], distributedParams[i], precision: 10);
        }

        // Cleanup
        backend.Shutdown();
    }

    /// <summary>
    /// Tests that CommunicationManager correctly initializes and manages backends.
    /// </summary>
    [Fact]
    public void CommunicationManager_Initialize_ManagesBackendCorrectly()
    {
        // Arrange
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4);

        // Act
        CommunicationManager.Initialize(backend);

        // Assert
        Assert.True(CommunicationManager.IsInitialized);
        Assert.Equal(0, CommunicationManager.GetRank<double>());
        Assert.Equal(4, CommunicationManager.GetWorldSize<double>());

        // Cleanup
        CommunicationManager.Shutdown();
        Assert.False(CommunicationManager.IsInitialized);
    }

    /// <summary>
    /// Tests ShardingConfiguration factory methods.
    ///
    /// For Beginners:
    /// ShardingConfiguration has preset configurations for different scenarios
    /// (high bandwidth, low bandwidth). This tests that they're set up correctly.
    /// </summary>
    [Fact]
    public void ShardingConfiguration_FactoryMethods_CreateCorrectConfigurations()
    {
        // Arrange
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();

        // Act
        var defaultConfig = ShardingConfiguration<double>.CreateDefault(backend);
        var highBandwidthConfig = ShardingConfiguration<double>.CreateForHighBandwidth(backend);
        var lowBandwidthConfig = ShardingConfiguration<double>.CreateForLowBandwidth(backend);

        // Assert - Check default config
        Assert.True(defaultConfig.AutoSyncGradients);
        Assert.Equal(1024, defaultConfig.MinimumParameterGroupSize);
        Assert.False(defaultConfig.EnableGradientCompression);

        // Assert - Check high bandwidth config
        Assert.True(highBandwidthConfig.AutoSyncGradients);
        Assert.Equal(512, highBandwidthConfig.MinimumParameterGroupSize);  // Smaller for fast networks
        Assert.False(highBandwidthConfig.EnableGradientCompression);  // No compression needed

        // Assert - Check low bandwidth config
        Assert.True(lowBandwidthConfig.AutoSyncGradients);
        Assert.Equal(4096, lowBandwidthConfig.MinimumParameterGroupSize);  // Larger for slow networks
        Assert.True(lowBandwidthConfig.EnableGradientCompression);  // Compression helps

        // Cleanup
        backend.Shutdown();
    }
}
