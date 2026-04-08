using AiDotNet.CheckpointManagement;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TrainingInfrastructure;

/// <summary>
/// Unit tests for CheckpointManager checkpoint saving and loading.
/// </summary>
public class CheckpointManagerTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly CheckpointManager<double, double[], double> _manager;

    public CheckpointManagerTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"checkpoint_manager_tests_{Guid.NewGuid():N}");
        _manager = new CheckpointManager<double, double[], double>(_testDirectory);
    }

    public void Dispose()
    {
        // Clean up test directory
        if (Directory.Exists(_testDirectory))
        {
            try
            {
                Directory.Delete(_testDirectory, true);
            }
            catch
            {
                // Ignore cleanup errors in tests
            }
        }
    }

    #region Mock Classes

    private class MockModel : IModel<double[], double, ModelMetadata<double>>
    {
        public string Name { get; set; } = "MockModel";
        public double[] Weights { get; set; } = new double[] { 1.0, 2.0, 3.0 };

        public void Train(double[] input, double expectedOutput) { }
        public double Predict(double[] input) => Weights.Sum();
        public ModelMetadata<double> GetModelMetadata() => new() { Name = Name };
    }

    private class MockOptimizer : IOptimizer<double, double[], double>
    {
        public double LearningRate { get; set; } = 0.001;
        public double Momentum { get; set; } = 0.9;

        public OptimizationResult<double, double[], double> Optimize(OptimizationInputData<double, double[], double> inputData)
        {
            return new OptimizationResult<double, double[], double>();
        }

        public bool ShouldEarlyStop() => false;

        public OptimizationAlgorithmOptions<double, double[], double> GetOptions()
        {
            return new OptimizationAlgorithmOptions<double, double[], double>();
        }

        public void Reset() { }

        // IModelSerializer implementation
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }

        public void SetModel(IFullModel<double, double[], double> model)
        {
            // No-op for test optimizer
        }
    }

    #endregion

    #region SaveCheckpoint Tests

    [Fact]
    public void SaveCheckpoint_WithValidInput_ReturnsCheckpointId()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();
        var metrics = new Dictionary<string, double> { ["loss"] = 0.5, ["accuracy"] = 0.85 };

        // Act
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 5, step: 100, metrics);

        // Assert
        Assert.NotNull(checkpointId);
        Assert.NotEmpty(checkpointId);
    }

    [Fact]
    public void SaveCheckpoint_WithMetadata_StoresMetadata()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();
        var metrics = new Dictionary<string, double> { ["loss"] = 0.5 };
        var metadata = new Dictionary<string, object>
        {
            ["batch_size"] = 32,
            ["learning_rate"] = 0.001
        };

        // Act
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 10, metrics, metadata);
        var checkpoint = _manager.LoadCheckpoint(checkpointId);

        // Assert
        Assert.NotNull(checkpoint.Metadata);
        Assert.Equal(32, Convert.ToInt32(checkpoint.Metadata["batch_size"]));
    }

    [Fact]
    public void SaveCheckpoint_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var optimizer = new MockOptimizer();
        var metrics = new Dictionary<string, double> { ["loss"] = 0.5 };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            _manager.SaveCheckpoint<ModelMetadata<double>>(null!, optimizer, epoch: 1, step: 1, metrics));
    }

    [Fact]
    public void SaveCheckpoint_WithNullOptimizer_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new MockModel();
        var metrics = new Dictionary<string, double> { ["loss"] = 0.5 };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            _manager.SaveCheckpoint(model, null!, epoch: 1, step: 1, metrics));
    }

    #endregion

    #region LoadCheckpoint Tests

    [Fact]
    public void LoadCheckpoint_WithValidId_ReturnsCheckpoint()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();
        var metrics = new Dictionary<string, double> { ["loss"] = 0.5 };
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 5, step: 100, metrics);

        // Act
        var checkpoint = _manager.LoadCheckpoint(checkpointId);

        // Assert
        Assert.NotNull(checkpoint);
        Assert.Equal(5, checkpoint.Epoch);
        Assert.Equal(100, checkpoint.Step);
        Assert.Equal(0.5, checkpoint.Metrics["loss"]);
    }

    [Fact]
    public void LoadCheckpoint_WithInvalidId_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _manager.LoadCheckpoint("nonexistent-id"));
    }

    [Fact]
    public void LoadLatestCheckpoint_ReturnsNewestCheckpoint()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["loss"] = 0.9 });
        Thread.Sleep(100); // Ensure timestamp difference
        _manager.SaveCheckpoint(model, optimizer, epoch: 2, step: 200, new Dictionary<string, double> { ["loss"] = 0.7 });
        Thread.Sleep(100);
        var latestId = _manager.SaveCheckpoint(model, optimizer, epoch: 3, step: 300, new Dictionary<string, double> { ["loss"] = 0.5 });

        // Act
        var latest = _manager.LoadLatestCheckpoint();

        // Assert
        Assert.NotNull(latest);
        Assert.Equal(3, latest.Epoch);
        Assert.Equal(300, latest.Step);
    }

    [Fact]
    public void LoadLatestCheckpoint_WhenNoCheckpoints_ReturnsNull()
    {
        // Arrange - Fresh manager with no checkpoints

        // Act
        var latest = _manager.LoadLatestCheckpoint();

        // Assert
        Assert.Null(latest);
    }

    [Fact]
    public void LoadBestCheckpoint_WithMinimize_ReturnsLowestMetric()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["loss"] = 0.9 });
        _manager.SaveCheckpoint(model, optimizer, epoch: 2, step: 200, new Dictionary<string, double> { ["loss"] = 0.5 }); // Best for minimize
        _manager.SaveCheckpoint(model, optimizer, epoch: 3, step: 300, new Dictionary<string, double> { ["loss"] = 0.7 });

        // Act
        var best = _manager.LoadBestCheckpoint("loss", MetricOptimizationDirection.Minimize);

        // Assert
        Assert.NotNull(best);
        Assert.Equal(2, best.Epoch);
        Assert.Equal(0.5, best.Metrics["loss"]);
    }

    [Fact]
    public void LoadBestCheckpoint_WithMaximize_ReturnsHighestMetric()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["accuracy"] = 0.7 });
        _manager.SaveCheckpoint(model, optimizer, epoch: 2, step: 200, new Dictionary<string, double> { ["accuracy"] = 0.95 }); // Best for maximize
        _manager.SaveCheckpoint(model, optimizer, epoch: 3, step: 300, new Dictionary<string, double> { ["accuracy"] = 0.85 });

        // Act
        var best = _manager.LoadBestCheckpoint("accuracy", MetricOptimizationDirection.Maximize);

        // Assert
        Assert.NotNull(best);
        Assert.Equal(2, best.Epoch);
        Assert.Equal(0.95, best.Metrics["accuracy"]);
    }

    [Fact]
    public void LoadBestCheckpoint_WhenMetricNotFound_ReturnsNull()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();
        _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["loss"] = 0.5 });

        // Act
        var best = _manager.LoadBestCheckpoint("nonexistent_metric", MetricOptimizationDirection.Minimize);

        // Assert
        Assert.Null(best);
    }

    #endregion

    #region ListCheckpoints Tests

    [Fact]
    public void ListCheckpoints_ReturnsAllCheckpoints()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["loss"] = 0.9 });
        _manager.SaveCheckpoint(model, optimizer, epoch: 2, step: 200, new Dictionary<string, double> { ["loss"] = 0.7 });
        _manager.SaveCheckpoint(model, optimizer, epoch: 3, step: 300, new Dictionary<string, double> { ["loss"] = 0.5 });

        // Act
        var checkpoints = _manager.ListCheckpoints();

        // Assert
        Assert.Equal(3, checkpoints.Count);
    }

    [Fact]
    public void ListCheckpoints_SortedByCreated_ReturnsCorrectOrder()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["loss"] = 0.9 });
        Thread.Sleep(50);
        _manager.SaveCheckpoint(model, optimizer, epoch: 2, step: 200, new Dictionary<string, double> { ["loss"] = 0.7 });
        Thread.Sleep(50);
        _manager.SaveCheckpoint(model, optimizer, epoch: 3, step: 300, new Dictionary<string, double> { ["loss"] = 0.5 });

        // Act
        var checkpoints = _manager.ListCheckpoints(sortBy: "created", descending: true);

        // Assert
        Assert.Equal(3, checkpoints.Count);
        Assert.Equal(3, checkpoints[0].Epoch); // Latest first when descending
    }

    [Fact]
    public void ListCheckpoints_SortedByStep_ReturnsCorrectOrder()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 300, new Dictionary<string, double> { ["loss"] = 0.9 });
        _manager.SaveCheckpoint(model, optimizer, epoch: 2, step: 100, new Dictionary<string, double> { ["loss"] = 0.7 });
        _manager.SaveCheckpoint(model, optimizer, epoch: 3, step: 200, new Dictionary<string, double> { ["loss"] = 0.5 });

        // Act
        var checkpoints = _manager.ListCheckpoints(sortBy: "step", descending: false);

        // Assert
        Assert.Equal(100, checkpoints[0].Step); // Lowest first when ascending
    }

    #endregion

    #region DeleteCheckpoint Tests

    [Fact]
    public void DeleteCheckpoint_RemovesCheckpoint()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["loss"] = 0.5 });

        // Act
        _manager.DeleteCheckpoint(checkpointId);

        // Assert
        Assert.Throws<ArgumentException>(() => _manager.LoadCheckpoint(checkpointId));
    }

    [Fact]
    public void DeleteCheckpoint_WithInvalidId_DoesNotThrow()
    {
        // Act & Assert - Should not throw for nonexistent ID
        _manager.DeleteCheckpoint("nonexistent-id");
    }

    #endregion

    #region Cleanup Tests

    [Fact]
    public void CleanupOldCheckpoints_KeepsOnlySpecifiedNumber()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        for (int i = 0; i < 10; i++)
        {
            _manager.SaveCheckpoint(model, optimizer, epoch: i, step: i * 100, new Dictionary<string, double> { ["loss"] = 1.0 - i * 0.1 });
            Thread.Sleep(50); // Ensure timestamp difference
        }

        // Act
        var deletedCount = _manager.CleanupOldCheckpoints(keepLast: 3);

        // Assert
        Assert.Equal(7, deletedCount);
        var remaining = _manager.ListCheckpoints();
        Assert.Equal(3, remaining.Count);
    }

    [Fact]
    public void CleanupOldCheckpoints_KeepsNewestCheckpoints()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["loss"] = 0.9 });
        Thread.Sleep(50);
        _manager.SaveCheckpoint(model, optimizer, epoch: 2, step: 200, new Dictionary<string, double> { ["loss"] = 0.7 });
        Thread.Sleep(50);
        _manager.SaveCheckpoint(model, optimizer, epoch: 3, step: 300, new Dictionary<string, double> { ["loss"] = 0.5 });
        Thread.Sleep(50);
        _manager.SaveCheckpoint(model, optimizer, epoch: 4, step: 400, new Dictionary<string, double> { ["loss"] = 0.3 });

        // Act
        _manager.CleanupOldCheckpoints(keepLast: 2);

        // Assert
        var remaining = _manager.ListCheckpoints();
        Assert.Equal(2, remaining.Count);
        Assert.Contains(remaining, c => c.Epoch == 3);
        Assert.Contains(remaining, c => c.Epoch == 4);
    }

    [Fact]
    public void CleanupKeepBest_KeepsBestByMetric()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["loss"] = 0.9 }); // Worst
        _manager.SaveCheckpoint(model, optimizer, epoch: 2, step: 200, new Dictionary<string, double> { ["loss"] = 0.3 }); // Best
        _manager.SaveCheckpoint(model, optimizer, epoch: 3, step: 300, new Dictionary<string, double> { ["loss"] = 0.7 });
        _manager.SaveCheckpoint(model, optimizer, epoch: 4, step: 400, new Dictionary<string, double> { ["loss"] = 0.5 }); // Second best

        // Act
        var deletedCount = _manager.CleanupKeepBest("loss", keepBest: 2, MetricOptimizationDirection.Minimize);

        // Assert
        Assert.Equal(2, deletedCount);
        var remaining = _manager.ListCheckpoints();
        Assert.Equal(2, remaining.Count);
        Assert.Contains(remaining, c => c.Metrics["loss"] == 0.3); // Best
        Assert.Contains(remaining, c => c.Metrics["loss"] == 0.5); // Second best
    }

    #endregion

    #region Persistence Tests

    [Fact]
    public void Manager_PersistsCheckpointsToDisk()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();
        _manager.SaveCheckpoint(model, optimizer, epoch: 5, step: 500, new Dictionary<string, double> { ["loss"] = 0.25 });

        // Act - Create new manager pointing to same directory and check metadata persists
        var manager2 = new CheckpointManager<double, double[], double>(_testDirectory);
        var checkpoints = manager2.ListCheckpoints();

        // Assert - Metadata should be found (checkpoint ID may differ due to deserialization)
        Assert.Single(checkpoints);
        var metadata = checkpoints.First();
        Assert.Equal(5, metadata.Epoch);
        Assert.Equal(500, metadata.Step);
        Assert.Equal(0.25, metadata.Metrics["loss"]);
    }

    [Fact]
    public void Manager_LoadsExistingCheckpointsOnCreation()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["loss"] = 0.9 });
        _manager.SaveCheckpoint(model, optimizer, epoch: 2, step: 200, new Dictionary<string, double> { ["loss"] = 0.7 });

        // Act - Create new manager pointing to same directory
        var manager2 = new CheckpointManager<double, double[], double>(_testDirectory);
        var checkpoints = manager2.ListCheckpoints();

        // Assert
        Assert.Equal(2, checkpoints.Count);
    }

    #endregion

    #region Checkpoint Properties Tests

    [Fact]
    public void Checkpoint_ContainsCorrectEpochAndStep()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        // Act
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 42, step: 12345, new Dictionary<string, double> { ["loss"] = 0.1 });
        var checkpoint = _manager.LoadCheckpoint(checkpointId);

        // Assert
        Assert.Equal(42, checkpoint.Epoch);
        Assert.Equal(12345, checkpoint.Step);
    }

    [Fact]
    public void Checkpoint_ContainsAllMetrics()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();
        var metrics = new Dictionary<string, double>
        {
            ["loss"] = 0.25,
            ["accuracy"] = 0.92,
            ["f1_score"] = 0.88,
            ["precision"] = 0.90,
            ["recall"] = 0.86
        };

        // Act
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, metrics);
        var checkpoint = _manager.LoadCheckpoint(checkpointId);

        // Assert
        Assert.Equal(5, checkpoint.Metrics.Count);
        Assert.Equal(0.25, checkpoint.Metrics["loss"]);
        Assert.Equal(0.92, checkpoint.Metrics["accuracy"]);
        Assert.Equal(0.88, checkpoint.Metrics["f1_score"]);
    }

    [Fact]
    public void Checkpoint_HasCreatedAtTimestamp()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();
        var beforeSave = DateTime.UtcNow.AddSeconds(-1); // Add buffer for timing

        // Act
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["loss"] = 0.5 });
        var afterSave = DateTime.UtcNow.AddSeconds(1); // Add buffer for timing
        var metadata = _manager.ListCheckpoints().First(c => c.CheckpointId == checkpointId);

        // Assert
        Assert.True(metadata.CreatedAt >= beforeSave && metadata.CreatedAt <= afterSave);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void SaveCheckpoint_WithEmptyMetrics_Succeeds()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        // Act
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double>());
        var checkpoint = _manager.LoadCheckpoint(checkpointId);

        // Assert
        Assert.NotNull(checkpoint);
        Assert.Empty(checkpoint.Metrics);
    }

    [Fact]
    public void CleanupOldCheckpoints_WhenFewerThanKeepLast_DeletesNone()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();

        _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 100, new Dictionary<string, double> { ["loss"] = 0.5 });
        _manager.SaveCheckpoint(model, optimizer, epoch: 2, step: 200, new Dictionary<string, double> { ["loss"] = 0.4 });

        // Act
        var deletedCount = _manager.CleanupOldCheckpoints(keepLast: 10);

        // Assert
        Assert.Equal(0, deletedCount);
        Assert.Equal(2, _manager.ListCheckpoints().Count);
    }

    #endregion
}
