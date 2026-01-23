using Xunit;
using AiDotNet.DistributedTraining;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// Comprehensive integration tests for the DistributedTraining module.
/// Tests communication backends, sharding configuration, and parameter analysis.
/// </summary>
public class DistributedTrainingIntegrationTests
{
    #region ReductionOperation Tests

    [Fact]
    public void ReductionOperation_HasExpectedValues()
    {
        // Assert
        var values = (ReductionOperation[])Enum.GetValues(typeof(ReductionOperation));
        Assert.Contains(ReductionOperation.Sum, values);
        Assert.Contains(ReductionOperation.Product, values);
        Assert.Contains(ReductionOperation.Min, values);
        Assert.Contains(ReductionOperation.Max, values);
        Assert.Contains(ReductionOperation.Average, values);
    }

    [Fact]
    public void ReductionOperation_Sum_HasCorrectValue()
    {
        // Assert
        Assert.Equal(0, (int)ReductionOperation.Sum);
    }

    #endregion

    #region InMemoryCommunicationBackend Constructor Tests

    [Fact]
    public void InMemoryCommunicationBackend_ValidConstruction()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();

        // Act
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4, environmentId: envId);

        // Assert
        Assert.Equal(0, backend.Rank);
        Assert.Equal(4, backend.WorldSize);
        Assert.False(backend.IsInitialized);
    }

    [Fact]
    public void InMemoryCommunicationBackend_InvalidRank_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: -1, worldSize: 4, environmentId: envId));
    }

    [Fact]
    public void InMemoryCommunicationBackend_RankGreaterThanWorldSize_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 4, worldSize: 4, environmentId: envId));
    }

    [Fact]
    public void InMemoryCommunicationBackend_InvalidWorldSize_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 0, environmentId: envId));
    }

    [Fact]
    public void InMemoryCommunicationBackend_EmptyEnvironmentId_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4, environmentId: ""));
    }

    [Fact]
    public void InMemoryCommunicationBackend_SingleProcess_ValidConstruction()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();

        // Act
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);

        // Assert
        Assert.Equal(0, backend.Rank);
        Assert.Equal(1, backend.WorldSize);
    }

    #endregion

    #region InMemoryCommunicationBackend Initialize/Shutdown Tests

    [Fact]
    public void InMemoryCommunicationBackend_Initialize_SetsIsInitialized()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);

        // Act
        backend.Initialize();

        // Assert
        Assert.True(backend.IsInitialized);

        // Cleanup
        backend.Shutdown();
    }

    [Fact]
    public void InMemoryCommunicationBackend_Shutdown_ClearsIsInitialized()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();

        // Act
        backend.Shutdown();

        // Assert
        Assert.False(backend.IsInitialized);
    }

    [Fact]
    public void InMemoryCommunicationBackend_ClearEnvironment_StaticCleanup()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        backend.Shutdown();

        // Act - static cleanup method
        InMemoryCommunicationBackend<double>.ClearEnvironment(envId);

        // Assert - no exception thrown
    }

    #endregion

    #region InMemoryCommunicationBackend SingleProcess Operations Tests

    [Fact]
    public void InMemoryCommunicationBackend_AllReduce_SingleProcess_NoChange()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        try
        {
            // Act
            backend.AllReduce(data, ReductionOperation.Sum);

            // Assert - single process, no change
            Assert.Equal(1.0, data[0]);
            Assert.Equal(2.0, data[1]);
            Assert.Equal(3.0, data[2]);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_AllGather_SingleProcess_ReturnsCopy()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        try
        {
            // Act
            var result = backend.AllGather(data);

            // Assert - single process, returns copy
            Assert.Equal(3, result.Length);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(2.0, result[1]);
            Assert.Equal(3.0, result[2]);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_Broadcast_SingleProcess_ReturnsCopy()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        try
        {
            // Act
            var result = backend.Broadcast(data, root: 0);

            // Assert - single process, returns copy
            Assert.Equal(3, result.Length);
            Assert.Equal(1.0, result[0]);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_Scatter_SingleProcess_ReturnsCopy()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        try
        {
            // Act
            var result = backend.Scatter(data, root: 0);

            // Assert - single process, returns copy
            Assert.Equal(3, result.Length);
            Assert.Equal(1.0, result[0]);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_ReduceScatter_SingleProcess_ReturnsCopy()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        try
        {
            // Act
            var result = backend.ReduceScatter(data, ReductionOperation.Sum);

            // Assert - single process, returns copy
            Assert.Equal(3, result.Length);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_Barrier_SingleProcess_NoBlock()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();

        try
        {
            // Act - should not block for single process
            backend.Barrier();

            // Assert - if we reach here, it worked
            Assert.True(true);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    #endregion

    #region InMemoryCommunicationBackend Validation Tests

    [Fact]
    public void InMemoryCommunicationBackend_AllReduce_NullData_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                backend.AllReduce(null!, ReductionOperation.Sum));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_AllGather_NullData_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                backend.AllGather(null!));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_Broadcast_InvalidRoot_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4, environmentId: envId);
        backend.Initialize();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                backend.Broadcast(new Vector<double>(new double[] { 1.0 }), root: 5));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_Scatter_InvalidRoot_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4, environmentId: envId);
        backend.Initialize();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                backend.Scatter(new Vector<double>(new double[] { 1.0 }), root: -1));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    #endregion

    #region ShardingConfiguration Tests

    [Fact]
    public void ShardingConfiguration_ValidConstruction()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);

        // Act
        var config = new ShardingConfiguration<double>(backend);

        // Assert
        Assert.True(config.AutoSyncGradients);
        Assert.Equal(1024, config.MinimumParameterGroupSize);
        Assert.False(config.EnableGradientCompression);
        Assert.Same(backend, config.CommunicationBackend);
    }

    [Fact]
    public void ShardingConfiguration_CustomLearningRate()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);

        // Act
        var config = new ShardingConfiguration<double>(backend, learningRate: 0.001);

        // Assert
        Assert.Equal(0.001, config.LearningRate);
    }

    [Fact]
    public void ShardingConfiguration_NullBackend_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new ShardingConfiguration<double>(null!));
    }

    [Fact]
    public void ShardingConfiguration_CreateDefault_ReturnsDefaultConfig()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);

        // Act
        var config = ShardingConfiguration<double>.CreateDefault(backend);

        // Assert
        Assert.True(config.AutoSyncGradients);
        Assert.Equal(1024, config.MinimumParameterGroupSize);
        Assert.False(config.EnableGradientCompression);
    }

    [Fact]
    public void ShardingConfiguration_CreateForHighBandwidth_ReturnsOptimizedConfig()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);

        // Act
        var config = ShardingConfiguration<double>.CreateForHighBandwidth(backend);

        // Assert
        Assert.True(config.AutoSyncGradients);
        Assert.Equal(512, config.MinimumParameterGroupSize);
        Assert.False(config.EnableGradientCompression);
    }

    [Fact]
    public void ShardingConfiguration_CreateForLowBandwidth_ReturnsOptimizedConfig()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);

        // Act
        var config = ShardingConfiguration<double>.CreateForLowBandwidth(backend);

        // Assert
        Assert.True(config.AutoSyncGradients);
        Assert.Equal(4096, config.MinimumParameterGroupSize);
        Assert.True(config.EnableGradientCompression);
    }

    [Fact]
    public void ShardingConfiguration_MutableProperties_CanBeSet()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        var config = new ShardingConfiguration<double>(backend);

        // Act
        config.AutoSyncGradients = false;
        config.MinimumParameterGroupSize = 2048;
        config.EnableGradientCompression = true;

        // Assert
        Assert.False(config.AutoSyncGradients);
        Assert.Equal(2048, config.MinimumParameterGroupSize);
        Assert.True(config.EnableGradientCompression);
    }

    #endregion

    #region ParameterAnalyzer Constructor Tests

    [Fact]
    public void ParameterAnalyzer_ValidConstruction()
    {
        // Act
        var analyzer = new ParameterAnalyzer<double>(minimumGroupSize: 1024, worldSize: 4);

        // Assert - no exception means success
        Assert.NotNull(analyzer);
    }

    [Fact]
    public void ParameterAnalyzer_DefaultConstruction()
    {
        // Act
        var analyzer = new ParameterAnalyzer<double>();

        // Assert - no exception means success
        Assert.NotNull(analyzer);
    }

    [Fact]
    public void ParameterAnalyzer_InvalidMinimumGroupSize_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new ParameterAnalyzer<double>(minimumGroupSize: 0));
    }

    [Fact]
    public void ParameterAnalyzer_InvalidWorldSize_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new ParameterAnalyzer<double>(minimumGroupSize: 1024, worldSize: 0));
    }

    #endregion

    #region ParameterAnalyzer.ParameterGroup Tests

    [Fact]
    public void ParameterGroup_DefaultValues()
    {
        // Arrange & Act
        var group = new ParameterAnalyzer<double>.ParameterGroup();

        // Assert
        Assert.Equal(0, group.StartIndex);
        Assert.Equal(0, group.Size);
        Assert.Equal(string.Empty, group.Name);
        Assert.False(group.IsMerged);
    }

    [Fact]
    public void ParameterGroup_SetProperties()
    {
        // Arrange & Act
        var group = new ParameterAnalyzer<double>.ParameterGroup
        {
            StartIndex = 100,
            Size = 500,
            Name = "Layer1.Weights",
            IsMerged = true
        };

        // Assert
        Assert.Equal(100, group.StartIndex);
        Assert.Equal(500, group.Size);
        Assert.Equal("Layer1.Weights", group.Name);
        Assert.True(group.IsMerged);
    }

    #endregion

    #region ParameterAnalyzer AnalyzeParameters Tests

    [Fact]
    public void ParameterAnalyzer_AnalyzeParameters_EmptyVector_ReturnsEmptyList()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();
        var parameters = new Vector<double>(Array.Empty<double>());

        // Act
        var groups = analyzer.AnalyzeParameters(parameters);

        // Assert
        Assert.Empty(groups);
    }

    [Fact]
    public void ParameterAnalyzer_AnalyzeParameters_NullVector_ThrowsException()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            analyzer.AnalyzeParameters(null!));
    }

    [Fact]
    public void ParameterAnalyzer_AnalyzeParameters_SmallVector_ReturnsSingleGroup()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>(minimumGroupSize: 1024);
        var parameters = new Vector<double>(new double[100]);

        // Act
        var groups = analyzer.AnalyzeParameters(parameters);

        // Assert
        Assert.Single(groups);
        Assert.Equal(0, groups[0].StartIndex);
        Assert.Equal(100, groups[0].Size);
    }

    [Fact]
    public void ParameterAnalyzer_AnalyzeParameters_ExactMultiple_ReturnsCorrectGroups()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>(minimumGroupSize: 100);
        var parameters = new Vector<double>(new double[300]);

        // Act
        var groups = analyzer.AnalyzeParameters(parameters);

        // Assert
        Assert.Equal(3, groups.Count);
        Assert.Equal(0, groups[0].StartIndex);
        Assert.Equal(100, groups[0].Size);
        Assert.Equal(100, groups[1].StartIndex);
        Assert.Equal(100, groups[1].Size);
        Assert.Equal(200, groups[2].StartIndex);
        Assert.Equal(100, groups[2].Size);
    }

    [Fact]
    public void ParameterAnalyzer_AnalyzeParameters_GroupsHaveNames()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>(minimumGroupSize: 100);
        var parameters = new Vector<double>(new double[200]);

        // Act
        var groups = analyzer.AnalyzeParameters(parameters);

        // Assert
        Assert.Equal("ParameterGroup_0", groups[0].Name);
        Assert.Equal("ParameterGroup_1", groups[1].Name);
    }

    #endregion

    #region ParameterAnalyzer AnalyzeForDistribution Tests

    [Fact]
    public void ParameterAnalyzer_AnalyzeForDistribution_EmptyVector_ReturnsEmptyList()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>(minimumGroupSize: 100, worldSize: 4);
        var parameters = new Vector<double>(Array.Empty<double>());

        // Act
        var groups = analyzer.AnalyzeForDistribution(parameters);

        // Assert
        Assert.Empty(groups);
    }

    [Fact]
    public void ParameterAnalyzer_AnalyzeForDistribution_NullVector_ThrowsException()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>(minimumGroupSize: 100, worldSize: 4);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            analyzer.AnalyzeForDistribution(null!));
    }

    [Fact]
    public void ParameterAnalyzer_AnalyzeForDistribution_CreatesDistributedGroups()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>(minimumGroupSize: 100, worldSize: 4);
        var parameters = new Vector<double>(new double[10000]);

        // Act
        var groups = analyzer.AnalyzeForDistribution(parameters);

        // Assert
        Assert.NotEmpty(groups);
        // Verify all parameters are covered
        var totalSize = groups.Sum(g => g.Size);
        Assert.Equal(10000, totalSize);
    }

    [Fact]
    public void ParameterAnalyzer_AnalyzeForDistribution_GroupsHaveDistributedNames()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>(minimumGroupSize: 100, worldSize: 2);
        var parameters = new Vector<double>(new double[1000]);

        // Act
        var groups = analyzer.AnalyzeForDistribution(parameters);

        // Assert
        Assert.All(groups, g => Assert.StartsWith("DistributedGroup_", g.Name));
    }

    #endregion

    #region ParameterAnalyzer CalculateDistributionStats Tests

    [Fact]
    public void ParameterAnalyzer_CalculateDistributionStats_EmptyList_ReturnsEmptyDict()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();
        var groups = new List<ParameterAnalyzer<double>.ParameterGroup>();

        // Act
        var stats = analyzer.CalculateDistributionStats(groups);

        // Assert
        Assert.Empty(stats);
    }

    [Fact]
    public void ParameterAnalyzer_CalculateDistributionStats_NullList_ReturnsEmptyDict()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();

        // Act
        var stats = analyzer.CalculateDistributionStats(null!);

        // Assert
        Assert.Empty(stats);
    }

    [Fact]
    public void ParameterAnalyzer_CalculateDistributionStats_ValidGroups_ReturnsStats()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>(minimumGroupSize: 100, worldSize: 4);
        var groups = new List<ParameterAnalyzer<double>.ParameterGroup>
        {
            new() { StartIndex = 0, Size = 100, Name = "Group1" },
            new() { StartIndex = 100, Size = 100, Name = "Group2" },
            new() { StartIndex = 200, Size = 100, Name = "Group3" }
        };

        // Act
        var stats = analyzer.CalculateDistributionStats(groups);

        // Assert
        Assert.Equal(3.0, stats["TotalGroups"]);
        Assert.Equal(300.0, stats["TotalParameters"]);
        Assert.Equal(100.0, stats["AverageGroupSize"]);
        Assert.Equal(100.0, stats["MinGroupSize"]);
        Assert.Equal(100.0, stats["MaxGroupSize"]);
        Assert.Equal(0.0, stats["MergedGroups"]);
        Assert.True(stats.ContainsKey("GroupsPerProcess"));
        Assert.True(stats.ContainsKey("GroupSizeVariance"));
        Assert.True(stats.ContainsKey("GroupSizeStdDev"));
    }

    [Fact]
    public void ParameterAnalyzer_CalculateDistributionStats_UnevenGroups_ReturnsCorrectStats()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();
        var groups = new List<ParameterAnalyzer<double>.ParameterGroup>
        {
            new() { StartIndex = 0, Size = 50, Name = "Group1" },
            new() { StartIndex = 50, Size = 100, Name = "Group2" },
            new() { StartIndex = 150, Size = 150, Name = "Group3", IsMerged = true }
        };

        // Act
        var stats = analyzer.CalculateDistributionStats(groups);

        // Assert
        Assert.Equal(3.0, stats["TotalGroups"]);
        Assert.Equal(300.0, stats["TotalParameters"]);
        Assert.Equal(100.0, stats["AverageGroupSize"]);
        Assert.Equal(50.0, stats["MinGroupSize"]);
        Assert.Equal(150.0, stats["MaxGroupSize"]);
        Assert.Equal(1.0, stats["MergedGroups"]);
        Assert.True(stats["GroupSizeStdDev"] > 0);
    }

    #endregion

    #region ParameterAnalyzer ValidateGrouping Tests

    [Fact]
    public void ParameterAnalyzer_ValidateGrouping_ValidGroups_ReturnsTrue()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();
        var groups = new List<ParameterAnalyzer<double>.ParameterGroup>
        {
            new() { StartIndex = 0, Size = 100 },
            new() { StartIndex = 100, Size = 100 },
            new() { StartIndex = 200, Size = 100 }
        };

        // Act
        var result = analyzer.ValidateGrouping(groups, totalParameterCount: 300);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void ParameterAnalyzer_ValidateGrouping_EmptyGroups_ThrowsException()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();
        var groups = new List<ParameterAnalyzer<double>.ParameterGroup>();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            analyzer.ValidateGrouping(groups, totalParameterCount: 100));
    }

    [Fact]
    public void ParameterAnalyzer_ValidateGrouping_NullGroups_ThrowsException()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            analyzer.ValidateGrouping(null!, totalParameterCount: 100));
    }

    [Fact]
    public void ParameterAnalyzer_ValidateGrouping_GroupNotStartingAtZero_ThrowsException()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();
        var groups = new List<ParameterAnalyzer<double>.ParameterGroup>
        {
            new() { StartIndex = 10, Size = 100 }
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            analyzer.ValidateGrouping(groups, totalParameterCount: 110));
    }

    [Fact]
    public void ParameterAnalyzer_ValidateGrouping_GroupWithGap_ThrowsException()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();
        var groups = new List<ParameterAnalyzer<double>.ParameterGroup>
        {
            new() { StartIndex = 0, Size = 100 },
            new() { StartIndex = 150, Size = 100 } // Gap from 100 to 150
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            analyzer.ValidateGrouping(groups, totalParameterCount: 250));
    }

    [Fact]
    public void ParameterAnalyzer_ValidateGrouping_GroupWithOverlap_ThrowsException()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();
        var groups = new List<ParameterAnalyzer<double>.ParameterGroup>
        {
            new() { StartIndex = 0, Size = 100 },
            new() { StartIndex = 50, Size = 100 } // Overlap from 50 to 100
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            analyzer.ValidateGrouping(groups, totalParameterCount: 150));
    }

    [Fact]
    public void ParameterAnalyzer_ValidateGrouping_IncompleteParameterCoverage_ThrowsException()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();
        var groups = new List<ParameterAnalyzer<double>.ParameterGroup>
        {
            new() { StartIndex = 0, Size = 100 }
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            analyzer.ValidateGrouping(groups, totalParameterCount: 200));
    }

    [Fact]
    public void ParameterAnalyzer_ValidateGrouping_UnsortedGroups_StillValidates()
    {
        // Arrange
        var analyzer = new ParameterAnalyzer<double>();
        var groups = new List<ParameterAnalyzer<double>.ParameterGroup>
        {
            new() { StartIndex = 200, Size = 100 },
            new() { StartIndex = 0, Size = 100 },
            new() { StartIndex = 100, Size = 100 }
        };

        // Act
        var result = analyzer.ValidateGrouping(groups, totalParameterCount: 300);

        // Assert - should sort and validate correctly
        Assert.True(result);
    }

    #endregion

    #region CommunicationManager Tests

    [Fact]
    public void CommunicationManager_IsInitialized_InitiallyFalse()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Assert
        Assert.False(CommunicationManager.IsInitialized);
    }

    [Fact]
    public void CommunicationManager_Initialize_SetsIsInitialized()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);

        try
        {
            // Act
            CommunicationManager.Initialize(backend);

            // Assert
            Assert.True(CommunicationManager.IsInitialized);
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    [Fact]
    public void CommunicationManager_Shutdown_ClearsIsInitialized()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        CommunicationManager.Initialize(backend);

        // Act
        CommunicationManager.Shutdown();

        // Assert
        Assert.False(CommunicationManager.IsInitialized);
    }

    [Fact]
    public void CommunicationManager_Initialize_NullBackend_ThrowsException()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            CommunicationManager.Initialize<double>(null!));
    }

    [Fact]
    public void CommunicationManager_Initialize_AlreadyInitialized_ThrowsException()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId1 = Guid.NewGuid().ToString();
        var envId2 = Guid.NewGuid().ToString();
        var backend1 = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId1);
        var backend2 = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId2);

        try
        {
            CommunicationManager.Initialize(backend1);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                CommunicationManager.Initialize(backend2));
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    [Fact]
    public void CommunicationManager_GetRank_WhenInitialized_ReturnsRank()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 2, worldSize: 4, environmentId: envId);

        try
        {
            CommunicationManager.Initialize(backend);

            // Act
            var rank = CommunicationManager.GetRank<double>();

            // Assert
            Assert.Equal(2, rank);
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    [Fact]
    public void CommunicationManager_GetWorldSize_WhenInitialized_ReturnsWorldSize()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 8, environmentId: envId);

        try
        {
            CommunicationManager.Initialize(backend);

            // Act
            var worldSize = CommunicationManager.GetWorldSize<double>();

            // Assert
            Assert.Equal(8, worldSize);
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    [Fact]
    public void CommunicationManager_GetRank_WhenNotInitialized_ThrowsException()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            CommunicationManager.GetRank<double>());
    }

    [Fact]
    public void CommunicationManager_AllReduce_WhenInitialized_Works()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        try
        {
            CommunicationManager.Initialize(backend);

            // Act
            CommunicationManager.AllReduce(data, ReductionOperation.Sum);

            // Assert - single process, no change
            Assert.Equal(1.0, data[0]);
            Assert.Equal(2.0, data[1]);
            Assert.Equal(3.0, data[2]);
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    [Fact]
    public void CommunicationManager_AllReduce_NullData_ThrowsException()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);

        try
        {
            CommunicationManager.Initialize(backend);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                CommunicationManager.AllReduce<double>(null!, ReductionOperation.Sum));
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    [Fact]
    public void CommunicationManager_AllGather_WhenInitialized_Works()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        try
        {
            CommunicationManager.Initialize(backend);

            // Act
            var result = CommunicationManager.AllGather(data);

            // Assert
            Assert.Equal(3, result.Length);
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    [Fact]
    public void CommunicationManager_Broadcast_WhenInitialized_Works()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        try
        {
            CommunicationManager.Initialize(backend);

            // Act
            var result = CommunicationManager.Broadcast(data, root: 0);

            // Assert
            Assert.Equal(3, result.Length);
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    [Fact]
    public void CommunicationManager_Scatter_WhenInitialized_Works()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        try
        {
            CommunicationManager.Initialize(backend);

            // Act
            var result = CommunicationManager.Scatter(data, root: 0);

            // Assert
            Assert.Equal(3, result.Length);
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    [Fact]
    public void CommunicationManager_Barrier_WhenInitialized_Works()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);

        try
        {
            CommunicationManager.Initialize(backend);

            // Act - should not throw for single process
            CommunicationManager.Barrier<double>();

            // Assert - if we reach here, it worked
            Assert.True(true);
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    [Fact]
    public void CommunicationManager_ReduceScatter_WhenInitialized_Works()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        try
        {
            CommunicationManager.Initialize(backend);

            // Act
            var result = CommunicationManager.ReduceScatter(data, ReductionOperation.Sum);

            // Assert
            Assert.Equal(3, result.Length);
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    #endregion

    #region Float Backend Tests

    [Fact]
    public void CommunicationManager_Initialize_FloatBackend_Works()
    {
        // Ensure clean state
        if (CommunicationManager.IsInitialized)
        {
            CommunicationManager.Shutdown();
        }

        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<float>(rank: 0, worldSize: 1, environmentId: envId);

        try
        {
            // Act
            CommunicationManager.Initialize(backend);

            // Assert
            Assert.True(CommunicationManager.IsInitialized);
            Assert.Equal(0, CommunicationManager.GetRank<float>());
            Assert.Equal(1, CommunicationManager.GetWorldSize<float>());
        }
        finally
        {
            CommunicationManager.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_Float_SingleProcess_Operations()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<float>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();

        try
        {
            // Test AllReduce
            var data = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });
            backend.AllReduce(data, ReductionOperation.Sum);
            Assert.Equal(1.0f, data[0]);

            // Test AllGather
            var gathered = backend.AllGather(new Vector<float>(new float[] { 4.0f, 5.0f }));
            Assert.Equal(2, gathered.Length);

            // Test Broadcast
            var broadcast = backend.Broadcast(new Vector<float>(new float[] { 6.0f }), root: 0);
            Assert.Equal(1, broadcast.Length);
        }
        finally
        {
            backend.Shutdown();
        }
    }

    #endregion

    #region Send/Receive Tests

    [Fact]
    public void InMemoryCommunicationBackend_Send_NullData_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2, environmentId: envId);
        backend.Initialize();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                backend.Send(null!, destinationRank: 1));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_Send_InvalidDestinationRank_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2, environmentId: envId);
        backend.Initialize();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                backend.Send(new Vector<double>(new double[] { 1.0 }), destinationRank: 5));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_Send_NegativeTag_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2, environmentId: envId);
        backend.Initialize();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                backend.Send(new Vector<double>(new double[] { 1.0 }), destinationRank: 1, tag: -1));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_Receive_InvalidSourceRank_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2, environmentId: envId);
        backend.Initialize();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                backend.Receive(sourceRank: -1, count: 10));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_Receive_InvalidCount_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2, environmentId: envId);
        backend.Initialize();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                backend.Receive(sourceRank: 1, count: 0));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    [Fact]
    public void InMemoryCommunicationBackend_Receive_NegativeTag_ThrowsException()
    {
        // Arrange
        var envId = Guid.NewGuid().ToString();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2, environmentId: envId);
        backend.Initialize();

        try
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                backend.Receive(sourceRank: 1, count: 10, tag: -1));
        }
        finally
        {
            backend.Shutdown();
        }
    }

    #endregion
}
