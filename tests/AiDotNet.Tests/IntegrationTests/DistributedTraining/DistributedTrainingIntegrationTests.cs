#nullable disable
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

        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

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
