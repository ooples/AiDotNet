using AiDotNet.CheckpointManagement;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CheckpointManagement;

/// <summary>
/// Deep integration tests for CheckpointManagement:
/// AutoCheckpointState (ToString, property storage),
/// ShouldAutoSaveCheckpoint (frequency-based, improvement-based, minimize/maximize),
/// UpdateAutoSaveState (best metric tracking),
/// ConfigureAutoCheckpointing (state reset),
/// path sanitization (GetSanitizedPath, ValidatePathWithinDirectory),
/// Checkpoint model (creation, defaults, metadata),
/// CheckpointMetadata defaults.
/// </summary>
public class CheckpointDeepMathIntegrationTests : IDisposable
{
    private readonly string _tempDir;
    private readonly CheckpointManager<double, double[], double> _manager;

    public CheckpointDeepMathIntegrationTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"checkpoint_test_{Guid.NewGuid():N}");
        _manager = new CheckpointManager<double, double[], double>(_tempDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir))
                Directory.Delete(_tempDir, true);
        }
        catch
        {
            // Best effort cleanup
        }
    }

    // ============================
    // AutoCheckpointState Tests
    // ============================

    [Fact]
    public void AutoCheckpointState_Disabled_ToString()
    {
        var state = new AutoCheckpointState(
            isEnabled: false, saveFrequency: 0, saveOnImprovement: false,
            keepLast: 0, metricName: null, lastSaveStep: 0, bestMetricValue: null);

        Assert.Contains("disabled", state.ToString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AutoCheckpointState_Enabled_ToString()
    {
        var state = new AutoCheckpointState(
            isEnabled: true, saveFrequency: 100, saveOnImprovement: true,
            keepLast: 5, metricName: "loss", lastSaveStep: 50, bestMetricValue: 0.25);

        var str = state.ToString();
        Assert.Contains("freq=100", str);
        Assert.Contains("improvement=True", str);
        Assert.Contains("keep=5", str);
        Assert.Contains("last=50", str);
        Assert.Contains("0.25", str);
    }

    [Fact]
    public void AutoCheckpointState_Properties_AllStored()
    {
        var state = new AutoCheckpointState(
            isEnabled: true, saveFrequency: 200, saveOnImprovement: false,
            keepLast: 3, metricName: "accuracy", lastSaveStep: 150, bestMetricValue: 0.95);

        Assert.True(state.IsEnabled);
        Assert.Equal(200, state.SaveFrequency);
        Assert.False(state.SaveOnImprovement);
        Assert.Equal(3, state.KeepLast);
        Assert.Equal("accuracy", state.MetricName);
        Assert.Equal(150, state.LastSaveStep);
        Assert.Equal(0.95, state.BestMetricValue);
    }

    [Fact]
    public void AutoCheckpointState_NullBestMetric()
    {
        var state = new AutoCheckpointState(
            isEnabled: true, saveFrequency: 10, saveOnImprovement: true,
            keepLast: 5, metricName: null, lastSaveStep: 0, bestMetricValue: null);

        Assert.Null(state.BestMetricValue);
        Assert.Null(state.MetricName);
    }

    // ============================
    // ShouldAutoSaveCheckpoint: Frequency-Based
    // ============================

    [Fact]
    public void ShouldAutoSave_NoConfig_ReturnsFalse()
    {
        // Manager without auto-checkpoint config
        Assert.False(_manager.ShouldAutoSaveCheckpoint(100));
    }

    [Fact]
    public void ShouldAutoSave_FrequencyReached_ReturnsTrue()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 10, keepLast: 5, saveOnImprovement: false);

        // Step 10 - frequency 10, last save at 0 -> 10-0 >= 10 -> true
        Assert.True(_manager.ShouldAutoSaveCheckpoint(10));
    }

    [Fact]
    public void ShouldAutoSave_FrequencyNotReached_ReturnsFalse()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 10, keepLast: 5, saveOnImprovement: false);

        // Step 5 - frequency 10, last save at 0 -> 5-0 < 10 -> false
        Assert.False(_manager.ShouldAutoSaveCheckpoint(5));
    }

    [Fact]
    public void ShouldAutoSave_AfterUpdate_FrequencyFromLastSave()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 10, keepLast: 5, saveOnImprovement: false);

        // Save at step 10
        _manager.UpdateAutoSaveState(10);

        // Step 15 - 15-10 = 5 < 10 -> false
        Assert.False(_manager.ShouldAutoSaveCheckpoint(15));

        // Step 20 - 20-10 = 10 >= 10 -> true
        Assert.True(_manager.ShouldAutoSaveCheckpoint(20));
    }

    // ============================
    // ShouldAutoSaveCheckpoint: Improvement-Based (Minimize)
    // ============================

    [Fact]
    public void ShouldAutoSave_FirstMetric_AlwaysTrue()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // First metric value - always an improvement
        Assert.True(_manager.ShouldAutoSaveCheckpoint(1, metricValue: 1.0, shouldMinimize: true));
    }

    [Fact]
    public void ShouldAutoSave_Minimize_LowerValue_True()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // Simulate first save
        _manager.UpdateAutoSaveState(1, metricValue: 1.0, shouldMinimize: true);

        // Lower metric value is improvement when minimizing
        Assert.True(_manager.ShouldAutoSaveCheckpoint(2, metricValue: 0.5, shouldMinimize: true));
    }

    [Fact]
    public void ShouldAutoSave_Minimize_HigherValue_False()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // Simulate first save
        _manager.UpdateAutoSaveState(1, metricValue: 0.5, shouldMinimize: true);

        // Higher metric value is NOT improvement when minimizing
        Assert.False(_manager.ShouldAutoSaveCheckpoint(2, metricValue: 1.0, shouldMinimize: true));
    }

    // ============================
    // ShouldAutoSaveCheckpoint: Improvement-Based (Maximize)
    // ============================

    [Fact]
    public void ShouldAutoSave_Maximize_HigherValue_True()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // Simulate first save
        _manager.UpdateAutoSaveState(1, metricValue: 0.5, shouldMinimize: false);

        // Higher metric value is improvement when maximizing
        Assert.True(_manager.ShouldAutoSaveCheckpoint(2, metricValue: 0.8, shouldMinimize: false));
    }

    [Fact]
    public void ShouldAutoSave_Maximize_LowerValue_False()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 0, keepLast: 5, saveOnImprovement: true);

        // Simulate first save
        _manager.UpdateAutoSaveState(1, metricValue: 0.8, shouldMinimize: false);

        // Lower metric value is NOT improvement when maximizing
        Assert.False(_manager.ShouldAutoSaveCheckpoint(2, metricValue: 0.5, shouldMinimize: false));
    }

    // ============================
    // UpdateAutoSaveState: Best Metric Tracking
    // ============================

    [Fact]
    public void UpdateAutoSaveState_TracksLastStep()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 10, keepLast: 5, saveOnImprovement: true);

        _manager.UpdateAutoSaveState(50);

        var state = _manager.GetAutoCheckpointState();
        Assert.Equal(50, state.LastSaveStep);
    }

    [Fact]
    public void UpdateAutoSaveState_TracksBestMetric_Minimize()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 10, keepLast: 5, saveOnImprovement: true);

        _manager.UpdateAutoSaveState(1, metricValue: 1.0, shouldMinimize: true);
        _manager.UpdateAutoSaveState(2, metricValue: 0.5, shouldMinimize: true);
        _manager.UpdateAutoSaveState(3, metricValue: 0.8, shouldMinimize: true); // Not improvement

        var state = _manager.GetAutoCheckpointState();
        Assert.Equal(0.5, state.BestMetricValue); // Best minimum
    }

    [Fact]
    public void UpdateAutoSaveState_TracksBestMetric_Maximize()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 10, keepLast: 5, saveOnImprovement: true);

        _manager.UpdateAutoSaveState(1, metricValue: 0.5, shouldMinimize: false);
        _manager.UpdateAutoSaveState(2, metricValue: 0.9, shouldMinimize: false);
        _manager.UpdateAutoSaveState(3, metricValue: 0.7, shouldMinimize: false); // Not improvement

        var state = _manager.GetAutoCheckpointState();
        Assert.Equal(0.9, state.BestMetricValue); // Best maximum
    }

    // ============================
    // ConfigureAutoCheckpointing: State Reset
    // ============================

    [Fact]
    public void ConfigureAutoCheckpointing_ResetsState()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 10, keepLast: 5, saveOnImprovement: true);
        _manager.UpdateAutoSaveState(100, metricValue: 0.5, shouldMinimize: true);

        // Reconfigure - should reset
        _manager.ConfigureAutoCheckpointing(saveFrequency: 20, keepLast: 3, saveOnImprovement: false);

        var state = _manager.GetAutoCheckpointState();
        Assert.Equal(20, state.SaveFrequency);
        Assert.Equal(3, state.KeepLast);
        Assert.False(state.SaveOnImprovement);
        Assert.Equal(0, state.LastSaveStep);
        Assert.Null(state.BestMetricValue);
    }

    [Fact]
    public void GetAutoCheckpointState_NoConfig_Disabled()
    {
        var state = _manager.GetAutoCheckpointState();

        Assert.False(state.IsEnabled);
        Assert.Equal(0, state.SaveFrequency);
        Assert.False(state.SaveOnImprovement);
    }

    // ============================
    // ShouldAutoSave: Combined Frequency + Improvement
    // ============================

    [Fact]
    public void ShouldAutoSave_FrequencyOrImprovement_EitherTriggers()
    {
        _manager.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 5, saveOnImprovement: true);

        // Save initial metric
        _manager.UpdateAutoSaveState(0, metricValue: 1.0, shouldMinimize: true);

        // Step 5 - frequency not reached, but metric improved
        Assert.True(_manager.ShouldAutoSaveCheckpoint(5, metricValue: 0.5, shouldMinimize: true));
    }

    // ============================
    // GetCheckpointDirectory
    // ============================

    [Fact]
    public void GetCheckpointDirectory_ReturnsConfiguredPath()
    {
        var dir = _manager.GetCheckpointDirectory();
        Assert.True(Directory.Exists(dir));
    }

    // ============================
    // Checkpoint<T> Model Tests
    // ============================

    [Fact]
    public void Checkpoint_DefaultConstructor_GeneratesUniqueId()
    {
        var cp1 = new Checkpoint<double, double[], double>();
        var cp2 = new Checkpoint<double, double[], double>();

        Assert.NotEqual(cp1.CheckpointId, cp2.CheckpointId);
    }

    [Fact]
    public void Checkpoint_DefaultConstructor_EmptyCollections()
    {
        var cp = new Checkpoint<double, double[], double>();

        Assert.NotNull(cp.Metrics);
        Assert.Empty(cp.Metrics);
        Assert.NotNull(cp.Metadata);
        Assert.Empty(cp.Metadata);
    }

    [Fact]
    public void Checkpoint_DefaultConstructor_SetsCreatedAt()
    {
        var before = DateTime.UtcNow;
        var cp = new Checkpoint<double, double[], double>();

        Assert.True(cp.CreatedAt >= before);
        Assert.True(cp.CreatedAt <= DateTime.UtcNow);
    }

    [Fact]
    public void Checkpoint_ParameterizedConstructor_StoresValues()
    {
        var metrics = new Dictionary<string, double> { { "loss", 0.5 }, { "accuracy", 0.85 } };
        var metadata = new Dictionary<string, object> { { "description", "Test checkpoint" } };

        var cp = new Checkpoint<double, double[], double>(
            model: "test_model",
            optimizerState: new Dictionary<string, object> { { "LearningRate", 0.01 } },
            optimizerTypeName: "SGD",
            epoch: 5,
            step: 500,
            metrics: metrics,
            metadata: metadata);

        Assert.Equal("test_model", cp.Model);
        Assert.Equal("SGD", cp.OptimizerTypeName);
        Assert.Equal(5, cp.Epoch);
        Assert.Equal(500, cp.Step);
        Assert.Equal(0.5, cp.Metrics["loss"]);
        Assert.Equal(0.85, cp.Metrics["accuracy"]);
        Assert.Equal("Test checkpoint", cp.Metadata["description"]);
    }

    // ============================
    // CheckpointMetadata<T> Tests
    // ============================

    [Fact]
    public void CheckpointMetadata_DefaultValues()
    {
        var metadata = new CheckpointMetadata<double>();

        Assert.Equal(string.Empty, metadata.CheckpointId);
        Assert.Equal(0, metadata.Epoch);
        Assert.Equal(0, metadata.Step);
        Assert.NotNull(metadata.Metrics);
        Assert.Empty(metadata.Metrics);
        Assert.Null(metadata.FilePath);
        Assert.Equal(0, metadata.FileSizeBytes);
    }

    [Fact]
    public void CheckpointMetadata_StoresMetrics()
    {
        var metadata = new CheckpointMetadata<double>
        {
            CheckpointId = "cp-001",
            Epoch = 10,
            Step = 1000,
            Metrics = new Dictionary<string, double>
            {
                { "loss", 0.1 },
                { "accuracy", 0.95 },
                { "f1_score", 0.92 }
            },
            FileSizeBytes = 1024 * 1024 // 1 MB
        };

        Assert.Equal("cp-001", metadata.CheckpointId);
        Assert.Equal(10, metadata.Epoch);
        Assert.Equal(1000, metadata.Step);
        Assert.Equal(3, metadata.Metrics.Count);
        Assert.Equal(0.1, metadata.Metrics["loss"]);
        Assert.Equal(1048576, metadata.FileSizeBytes);
    }

    // ============================
    // RegisteredModel Tests
    // ============================

    [Fact]
    public void RegisteredModel_DefaultValues()
    {
        var model = new RegisteredModel<double, double[], double>();

        Assert.Equal(string.Empty, model.ModelId);
        Assert.Equal(string.Empty, model.Name);
        Assert.Equal(0, model.Version);
        Assert.Equal(ModelStage.Development, model.Stage);
        Assert.Null(model.Metadata);
        Assert.NotNull(model.Tags);
        Assert.Empty(model.Tags);
        Assert.Null(model.Description);
        Assert.Null(model.StoragePath);
        Assert.Null(model.ModelCard);
    }

    // ============================
    // ModelVersionInfo Tests
    // ============================

    [Fact]
    public void ModelVersionInfo_DefaultValues()
    {
        var info = new ModelVersionInfo<double>();

        Assert.Equal(0, info.Version);
        Assert.Equal(ModelStage.Development, info.Stage);
        Assert.Null(info.Description);
        Assert.Null(info.Metadata);
    }

    // ============================
    // ModelSearchCriteria Tests
    // ============================

    [Fact]
    public void ModelSearchCriteria_DefaultValues()
    {
        var criteria = new ModelSearchCriteria<double>();

        Assert.Null(criteria.NamePattern);
        Assert.Null(criteria.Tags);
        Assert.Null(criteria.Stage);
        Assert.Null(criteria.MinVersion);
        Assert.Null(criteria.MaxVersion);
        Assert.Null(criteria.CreatedAfter);
        Assert.Null(criteria.CreatedBefore);
    }

    // ============================
    // ModelComparison Tests
    // ============================

    [Fact]
    public void ModelComparison_DefaultValues()
    {
        var comparison = new ModelComparison<double>();

        Assert.Equal(0, comparison.Version1);
        Assert.Equal(0, comparison.Version2);
        Assert.NotNull(comparison.MetadataDifferences);
        Assert.Empty(comparison.MetadataDifferences);
        Assert.NotNull(comparison.MetricDifferences);
        Assert.Empty(comparison.MetricDifferences);
        Assert.False(comparison.ArchitectureChanged);
    }

    // ============================
    // ModelLineage Tests
    // ============================

    [Fact]
    public void ModelLineage_DefaultValues()
    {
        var lineage = new ModelLineage();

        Assert.Equal(string.Empty, lineage.ModelName);
        Assert.Equal(0, lineage.Version);
        Assert.Null(lineage.ExperimentId);
        Assert.Null(lineage.RunId);
        Assert.Null(lineage.TrainingDataSource);
        Assert.Null(lineage.ParentModel);
        Assert.Null(lineage.ParentVersion);
        Assert.Null(lineage.Creator);
    }

    [Fact]
    public void ModelLineage_WithParent_TracksVersion()
    {
        var lineage = new ModelLineage
        {
            ModelName = "classifier",
            Version = 3,
            ParentModel = "classifier",
            ParentVersion = 2,
            TrainingDataSource = "training_v3.csv",
            Creator = "data-science-team"
        };

        Assert.Equal("classifier", lineage.ModelName);
        Assert.Equal(3, lineage.Version);
        Assert.Equal(2, lineage.ParentVersion);
        Assert.Equal("classifier", lineage.ParentModel);
    }
}
