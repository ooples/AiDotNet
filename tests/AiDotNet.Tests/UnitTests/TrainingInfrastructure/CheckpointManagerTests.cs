using AiDotNet.CheckpointManagement;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using Xunit;
using System.Threading.Tasks;

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

    /// <summary>
    /// A model that opts in to <see cref="ICheckpointableModel"/> by round-tripping its
    /// weights through the state stream, so we can verify typed restore reproduces state.
    /// </summary>
    private class CheckpointableMockModel : IModel<double[], double, ModelMetadata<double>>, ICheckpointableModel
    {
        public string Name { get; set; } = "CheckpointableMockModel";
        public double[] Weights { get; set; } = new double[] { 1.0, 2.0, 3.0 };

        public void Train(double[] input, double expectedOutput) { }
        public double Predict(double[] input) => Weights.Sum();
        public ModelMetadata<double> GetModelMetadata() => new() { Name = Name };

        public void SaveState(Stream stream)
        {
            var writer = new BinaryWriter(stream);
            writer.Write(Weights.Length);
            foreach (var w in Weights)
            {
                writer.Write(w);
            }
            writer.Flush();
        }

        public void LoadState(Stream stream)
        {
            var reader = new BinaryReader(stream);
            var count = reader.ReadInt32();
            var restored = new double[count];
            for (int i = 0; i < count; i++)
            {
                restored[i] = reader.ReadDouble();
            }
            Weights = restored;
        }
    }

    private class MockOptimizer : IOptimizer<double, double[], double>
    {
        public double LearningRate { get; set; } = 0.001;
        public double Momentum { get; set; } = 0.9;
        public byte[] SerializedState { get; set; } = new byte[] { 1, 2, 3, 5, 8, 13 };
        public byte[] LastDeserializedState { get; private set; } = Array.Empty<byte>();

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
        public byte[] Serialize() => SerializedState.ToArray();
        public void Deserialize(byte[] data) => LastDeserializedState = data.ToArray();
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }

        public void SetModel(IFullModel<double, double[], double> model)
        {
            // No-op for test optimizer
        }
    }

    #endregion

    #region SaveCheckpoint Tests

    [Fact(Timeout = 60000)]
    public async Task SaveCheckpoint_WithValidInput_ReturnsCheckpointId()
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

    [Fact(Timeout = 60000)]
    public async Task SaveCheckpoint_WithMetadata_StoresMetadata()
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

    [Fact(Timeout = 60000)]
    public async Task SaveCheckpoint_WithOptimizer_PersistsSerializedOptimizerPayload()
    {
        await Task.Yield();

        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer
        {
            SerializedState = new byte[] { 42, 99, 123, 7 }
        };
        var metrics = new Dictionary<string, double> { ["loss"] = 0.5 };

        // Act
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 10, metrics);
        var checkpoint = _manager.LoadCheckpoint(checkpointId);

        // Assert
        Assert.Equal(optimizer.SerializedState, checkpoint.OptimizerData);
        Assert.Equal(optimizer.SerializedState.Length, Convert.ToInt32(checkpoint.OptimizerState["SerializedStateLengthBytes"]));
    }

    [Fact(Timeout = 60000)]
    public async Task RestoreOptimizer_WithSerializedPayload_RestoresOptimizerState()
    {
        await Task.Yield();

        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer
        {
            SerializedState = new byte[] { 3, 1, 4, 1, 5, 9 }
        };
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 10, new Dictionary<string, double> { ["loss"] = 0.5 });
        var checkpoint = _manager.LoadCheckpoint(checkpointId);
        var restoredOptimizer = new MockOptimizer();

        // Act
        checkpoint.RestoreOptimizer(restoredOptimizer);

        // Assert
        Assert.Equal(optimizer.SerializedState, restoredOptimizer.LastDeserializedState);
    }

    [Fact(Timeout = 60000)]
    public async Task SaveCheckpoint_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var optimizer = new MockOptimizer();
        var metrics = new Dictionary<string, double> { ["loss"] = 0.5 };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            _manager.SaveCheckpoint<ModelMetadata<double>>(null!, optimizer, epoch: 1, step: 1, metrics));
    }

    [Fact(Timeout = 60000)]
    public async Task SaveCheckpoint_WithNullOptimizer_ThrowsArgumentNullException()
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

    [Fact(Timeout = 60000)]
    public async Task LoadCheckpoint_WithValidId_ReturnsCheckpoint()
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

    [Fact(Timeout = 60000)]
    public async Task LoadCheckpoint_WithInvalidId_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _manager.LoadCheckpoint("nonexistent-id"));
    }

    [Fact(Timeout = 60000)]
    public async Task LoadLatestCheckpoint_ReturnsNewestCheckpoint()
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

    [Fact(Timeout = 60000)]
    public async Task LoadLatestCheckpoint_WhenNoCheckpoints_ReturnsNull()
    {
        // Arrange - Fresh manager with no checkpoints

        // Act
        var latest = _manager.LoadLatestCheckpoint();

        // Assert
        Assert.Null(latest);
    }

    [Fact(Timeout = 60000)]
    public async Task LoadBestCheckpoint_WithMinimize_ReturnsLowestMetric()
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

    [Fact(Timeout = 60000)]
    public async Task LoadBestCheckpoint_WithMaximize_ReturnsHighestMetric()
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

    [Fact(Timeout = 60000)]
    public async Task LoadBestCheckpoint_WhenMetricNotFound_ReturnsNull()
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

    [Fact(Timeout = 60000)]
    public async Task ListCheckpoints_ReturnsAllCheckpoints()
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

    [Fact(Timeout = 60000)]
    public async Task ListCheckpoints_SortedByCreated_ReturnsCorrectOrder()
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

    [Fact(Timeout = 60000)]
    public async Task ListCheckpoints_SortedByStep_ReturnsCorrectOrder()
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

    [Fact(Timeout = 60000)]
    public async Task DeleteCheckpoint_RemovesCheckpoint()
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

    [Fact(Timeout = 60000)]
    public async Task DeleteCheckpoint_WithInvalidId_DoesNotThrow()
    {
        // Act & Assert - Should not throw for nonexistent ID
        _manager.DeleteCheckpoint("nonexistent-id");
    }

    #endregion

    #region Cleanup Tests

    [Fact(Timeout = 60000)]
    public async Task CleanupOldCheckpoints_KeepsOnlySpecifiedNumber()
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

    [Fact(Timeout = 60000)]
    public async Task CleanupOldCheckpoints_KeepsNewestCheckpoints()
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

    [Fact(Timeout = 60000)]
    public async Task CleanupKeepBest_KeepsBestByMetric()
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

    [Fact(Timeout = 60000)]
    public async Task Manager_PersistsCheckpointsToDisk()
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

    [Fact(Timeout = 60000)]
    public async Task Manager_LoadsExistingCheckpointsOnCreation()
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

    [Fact(Timeout = 60000)]
    public async Task Checkpoint_ContainsCorrectEpochAndStep()
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

    [Fact(Timeout = 60000)]
    public async Task Checkpoint_ContainsAllMetrics()
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

    [Fact(Timeout = 60000)]
    public async Task Checkpoint_HasCreatedAtTimestamp()
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

    [Fact(Timeout = 60000)]
    public async Task SaveCheckpoint_WithEmptyMetrics_Succeeds()
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

    [Fact(Timeout = 60000)]
    public async Task CleanupOldCheckpoints_WhenFewerThanKeepLast_DeletesNone()
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

    #region RestoreModelState Tests

    [Fact(Timeout = 60000)]
    public async Task RestoreModelState_WithCheckpointableModel_RestoresSavedState()
    {
        await Task.Yield();

        // Arrange - a checkpointable model with distinctive weights
        var model = new CheckpointableMockModel { Weights = new[] { 3.14, 2.71, 1.41, 1.61 } };
        var optimizer = new MockOptimizer();
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 7, step: 700,
            new Dictionary<string, double> { ["loss"] = 0.05 });

        // Act - restore into a FRESH model with different weights
        var target = new CheckpointableMockModel { Weights = new[] { 0.0 } };
        var restored = _manager.RestoreModelState(checkpointId, target);

        // Assert - the fresh model now reproduces the saved model's state exactly
        Assert.True(restored);
        Assert.Equal(model.Weights, target.Weights);
        Assert.Equal(model.Predict(Array.Empty<double>()), target.Predict(Array.Empty<double>()));
    }

    [Fact(Timeout = 60000)]
    public async Task RestoreLatestModelState_RestoresNewestCheckpointState()
    {
        await Task.Yield();

        // Arrange - three checkpoints; the newest has distinctive weights
        var optimizer = new MockOptimizer();
        _manager.SaveCheckpoint(new CheckpointableMockModel { Weights = new[] { 1.0 } },
            optimizer, epoch: 1, step: 10, new Dictionary<string, double> { ["loss"] = 0.9 });
        Thread.Sleep(50);
        _manager.SaveCheckpoint(new CheckpointableMockModel { Weights = new[] { 2.0 } },
            optimizer, epoch: 2, step: 20, new Dictionary<string, double> { ["loss"] = 0.7 });
        Thread.Sleep(50);
        _manager.SaveCheckpoint(new CheckpointableMockModel { Weights = new[] { 10.0, 20.0, 30.0 } },
            optimizer, epoch: 3, step: 30, new Dictionary<string, double> { ["loss"] = 0.5 });

        // Act
        var target = new CheckpointableMockModel();
        var restored = _manager.RestoreLatestModelState(target);

        // Assert - restores the NEWEST checkpoint's state
        Assert.True(restored);
        Assert.Equal(new[] { 10.0, 20.0, 30.0 }, target.Weights);
    }

    [Fact(Timeout = 60000)]
    public async Task RestoreLatestModelState_WhenNoCheckpoints_ReturnsFalse()
    {
        await Task.Yield();

        // Act
        var target = new CheckpointableMockModel();
        var restored = _manager.RestoreLatestModelState(target);

        // Assert
        Assert.False(restored);
    }

    [Fact(Timeout = 60000)]
    public async Task RestoreModelState_WhenNoStateSidecar_ReturnsFalse()
    {
        await Task.Yield();

        // Arrange - a non-checkpointable model writes no state sidecar
        var model = new MockModel();
        var optimizer = new MockOptimizer();
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 10,
            new Dictionary<string, double> { ["loss"] = 0.5 });

        // Act
        var target = new CheckpointableMockModel();
        var restored = _manager.RestoreModelState(checkpointId, target);

        // Assert - no sidecar to restore from
        Assert.False(restored);
    }

    [Fact(Timeout = 60000)]
    public async Task RestoreModelState_WithInvalidId_ThrowsArgumentException()
    {
        await Task.Yield();

        // Act & Assert
        var target = new CheckpointableMockModel();
        Assert.Throws<ArgumentException>(() => _manager.RestoreModelState("nonexistent-id", target));
    }

    [Fact(Timeout = 60000)]
    public async Task RestoreModelState_WithNullTarget_ThrowsArgumentNullException()
    {
        await Task.Yield();

        // Arrange
        var model = new CheckpointableMockModel();
        var optimizer = new MockOptimizer();
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 10,
            new Dictionary<string, double> { ["loss"] = 0.5 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _manager.RestoreModelState(checkpointId, null!));
    }

    [Fact(Timeout = 60000)]
    public async Task DeleteCheckpoint_RemovesStateSidecar()
    {
        await Task.Yield();

        // Arrange
        var model = new CheckpointableMockModel { Weights = new[] { 5.0, 6.0 } };
        var optimizer = new MockOptimizer();
        var checkpointId = _manager.SaveCheckpoint(model, optimizer, epoch: 1, step: 10,
            new Dictionary<string, double> { ["loss"] = 0.5 });

        // Sanity: a sidecar exists to restore from
        var probe = new CheckpointableMockModel();
        Assert.True(_manager.RestoreModelState(checkpointId, probe));

        // Act
        _manager.DeleteCheckpoint(checkpointId);

        // Assert - the sidecar is gone along with the checkpoint
        var sidecars = Directory.GetFiles(_testDirectory, "*" + ".state");
        Assert.Empty(sidecars);
    }

    #endregion
}
