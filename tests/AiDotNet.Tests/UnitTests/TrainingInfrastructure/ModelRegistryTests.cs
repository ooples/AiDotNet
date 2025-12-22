using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.ModelRegistry;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TrainingInfrastructure;

/// <summary>
/// Unit tests for ModelRegistry model versioning and lifecycle management.
/// </summary>
public class ModelRegistryTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly ModelRegistry<double, double[], double> _registry;

    public ModelRegistryTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"model_registry_tests_{Guid.NewGuid():N}");
        _registry = new ModelRegistry<double, double[], double>(_testDirectory);
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

    #region Mock Model

    private class MockModel : IModel<double[], double, ModelMetadata<double>>
    {
        public void Train(double[] input, double expectedOutput) { }
        public double Predict(double[] input) => 0.0;
        public ModelMetadata<double> GetModelMetadata() => new()
        {
            Name = "MockModel",
            ModelType = ModelType.None,
            FeatureCount = 5,
            Complexity = 10
        };
    }

    private static ModelMetadata<double> CreateTestMetadata(string name = "TestModel")
    {
        return new ModelMetadata<double>
        {
            Name = name,
            ModelType = ModelType.None,
            FeatureCount = 10,
            Complexity = 5,
            Description = "Test model for unit tests",
            FeatureImportance = new Dictionary<string, double>
            {
                ["feature1"] = 0.5,
                ["feature2"] = 0.3,
                ["feature3"] = 0.2
            }
        };
    }

    #endregion

    #region RegisterModel Tests

    [Fact]
    public void RegisterModel_WithValidInput_ReturnsModelId()
    {
        // Arrange
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        // Act
        var modelId = _registry.RegisterModel("test-model", model, metadata);

        // Assert
        Assert.NotNull(modelId);
        Assert.NotEmpty(modelId);
    }

    [Fact]
    public void RegisterModel_WithTags_StoresTags()
    {
        // Arrange
        var model = new MockModel();
        var metadata = CreateTestMetadata();
        var tags = new Dictionary<string, string>
        {
            ["team"] = "ml-research",
            ["framework"] = "aidotnet"
        };

        // Act
        var modelId = _registry.RegisterModel("tagged-model", model, metadata, tags);
        var registeredModel = _registry.GetModel("tagged-model");

        // Assert
        Assert.NotNull(registeredModel);
        Assert.Equal("ml-research", registeredModel.Tags["team"]);
        Assert.Equal("aidotnet", registeredModel.Tags["framework"]);
    }

    [Fact]
    public void RegisterModel_WithNullName_ThrowsArgumentException()
    {
        // Arrange
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _registry.RegisterModel(null!, model, metadata));
    }

    [Fact]
    public void RegisterModel_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var metadata = CreateTestMetadata();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _registry.RegisterModel<ModelMetadata<double>>("test", null!, metadata));
    }

    #endregion

    #region GetModel Tests

    [Fact]
    public void GetModel_WithoutVersion_ReturnsLatestVersion()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("multi-version-model", model, CreateTestMetadata("v1"));
        _registry.CreateModelVersion("multi-version-model", model, CreateTestMetadata("v2"), "Version 2");

        // Act
        var latestModel = _registry.GetModel("multi-version-model");

        // Assert
        Assert.NotNull(latestModel);
        Assert.Equal(2, latestModel.Version);
    }

    [Fact]
    public void GetModel_WithSpecificVersion_ReturnsCorrectVersion()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("version-test", model, CreateTestMetadata("v1"));
        _registry.CreateModelVersion("version-test", model, CreateTestMetadata("v2"), "Version 2");

        // Act
        var v1Model = _registry.GetModel("version-test", version: 1);

        // Assert
        Assert.NotNull(v1Model);
        Assert.Equal(1, v1Model.Version);
    }

    [Fact]
    public void GetModel_WithInvalidName_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _registry.GetModel("nonexistent-model"));
    }

    [Fact]
    public void GetModel_WithInvalidVersion_ThrowsArgumentException()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("single-version", model, CreateTestMetadata());

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _registry.GetModel("single-version", version: 99));
    }

    [Fact]
    public void GetLatestModel_ReturnsHighestVersionNumber()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("latest-test", model, CreateTestMetadata("v1"));
        _registry.CreateModelVersion("latest-test", model, CreateTestMetadata("v2"));
        _registry.CreateModelVersion("latest-test", model, CreateTestMetadata("v3"));

        // Act
        var latest = _registry.GetLatestModel("latest-test");

        // Assert
        Assert.NotNull(latest);
        Assert.Equal(3, latest.Version);
    }

    #endregion

    #region CreateModelVersion Tests

    [Fact]
    public void CreateModelVersion_IncrementsVersionNumber()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("increment-test", model, CreateTestMetadata());

        // Act
        var version2 = _registry.CreateModelVersion("increment-test", model, CreateTestMetadata(), "Version 2");
        var version3 = _registry.CreateModelVersion("increment-test", model, CreateTestMetadata(), "Version 3");

        // Assert
        Assert.Equal(2, version2);
        Assert.Equal(3, version3);
    }

    [Fact]
    public void CreateModelVersion_WithNonExistentModel_ThrowsArgumentException()
    {
        // Arrange
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _registry.CreateModelVersion("nonexistent", model, metadata));
    }

    #endregion

    #region Model Stage Transition Tests

    [Fact]
    public void TransitionModelStage_ToStaging_UpdatesStage()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("stage-test", model, CreateTestMetadata());

        // Act
        _registry.TransitionModelStage("stage-test", 1, ModelStage.Staging);

        // Assert
        var registeredModel = _registry.GetModel("stage-test", 1);
        Assert.Equal(ModelStage.Staging, registeredModel.Stage);
    }

    [Fact]
    public void TransitionModelStage_ToProduction_ArchivesPreviousProduction()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("archive-test", model, CreateTestMetadata());
        _registry.CreateModelVersion("archive-test", model, CreateTestMetadata());

        // Promote version 1 to production
        _registry.TransitionModelStage("archive-test", 1, ModelStage.Production);

        // Act - Promote version 2 to production
        _registry.TransitionModelStage("archive-test", 2, ModelStage.Production, archivePrevious: true);

        // Assert
        var v1 = _registry.GetModel("archive-test", 1);
        var v2 = _registry.GetModel("archive-test", 2);

        Assert.Equal(ModelStage.Archived, v1.Stage);
        Assert.Equal(ModelStage.Production, v2.Stage);
    }

    [Fact]
    public void GetModelByStage_ReturnsCorrectModel()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("by-stage-test", model, CreateTestMetadata());
        _registry.CreateModelVersion("by-stage-test", model, CreateTestMetadata());
        _registry.TransitionModelStage("by-stage-test", 1, ModelStage.Staging);
        _registry.TransitionModelStage("by-stage-test", 2, ModelStage.Production);

        // Act
        var stagingModel = _registry.GetModelByStage("by-stage-test", ModelStage.Staging);
        var productionModel = _registry.GetModelByStage("by-stage-test", ModelStage.Production);

        // Assert
        Assert.NotNull(stagingModel);
        Assert.Equal(1, stagingModel.Version);
        Assert.NotNull(productionModel);
        Assert.Equal(2, productionModel.Version);
    }

    [Fact]
    public void GetModelByStage_WhenNoModelInStage_ReturnsNull()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("no-stage-test", model, CreateTestMetadata());

        // Act
        var productionModel = _registry.GetModelByStage("no-stage-test", ModelStage.Production);

        // Assert
        Assert.Null(productionModel);
    }

    [Fact]
    public void ArchiveModel_TransitionsToArchived()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("archive-direct-test", model, CreateTestMetadata());

        // Act
        _registry.ArchiveModel("archive-direct-test", 1);

        // Assert
        var archivedModel = _registry.GetModel("archive-direct-test", 1);
        Assert.Equal(ModelStage.Archived, archivedModel.Stage);
    }

    #endregion

    #region List and Search Tests

    [Fact]
    public void ListModels_ReturnsAllModelNames()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("model-a", model, CreateTestMetadata());
        _registry.RegisterModel("model-b", model, CreateTestMetadata());
        _registry.RegisterModel("model-c", model, CreateTestMetadata());

        // Act
        var models = _registry.ListModels();

        // Assert
        Assert.Equal(3, models.Count);
        Assert.Contains("model-a", models);
        Assert.Contains("model-b", models);
        Assert.Contains("model-c", models);
    }

    [Fact]
    public void ListModels_WithFilter_ReturnsMatchingModels()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("classification-model", model, CreateTestMetadata());
        _registry.RegisterModel("regression-model", model, CreateTestMetadata());
        _registry.RegisterModel("classification-v2-model", model, CreateTestMetadata());

        // Act
        var models = _registry.ListModels("classification");

        // Assert
        Assert.Equal(2, models.Count);
        Assert.All(models, m => Assert.Contains("classification", m));
    }

    [Fact]
    public void ListModels_WithTags_ReturnsMatchingModels()
    {
        // Arrange
        var model = new MockModel();
        var productionTags = new Dictionary<string, string> { ["env"] = "production" };
        var devTags = new Dictionary<string, string> { ["env"] = "development" };

        _registry.RegisterModel("prod-model-1", model, CreateTestMetadata(), productionTags);
        _registry.RegisterModel("prod-model-2", model, CreateTestMetadata(), productionTags);
        _registry.RegisterModel("dev-model", model, CreateTestMetadata(), devTags);

        // Act
        var prodModels = _registry.ListModels(tags: productionTags);

        // Assert
        Assert.Equal(2, prodModels.Count);
        Assert.Contains("prod-model-1", prodModels);
        Assert.Contains("prod-model-2", prodModels);
    }

    [Fact]
    public void ListModelVersions_ReturnsAllVersions()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("versions-list-test", model, CreateTestMetadata());
        _registry.CreateModelVersion("versions-list-test", model, CreateTestMetadata());
        _registry.CreateModelVersion("versions-list-test", model, CreateTestMetadata());

        // Act
        var versions = _registry.ListModelVersions("versions-list-test");

        // Assert
        Assert.Equal(3, versions.Count);
        Assert.Contains(versions, v => v.Version == 1);
        Assert.Contains(versions, v => v.Version == 2);
        Assert.Contains(versions, v => v.Version == 3);
    }

    [Fact]
    public void SearchModels_ByNamePattern_ReturnsMatchingModels()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("search-test-a", model, CreateTestMetadata());
        _registry.RegisterModel("search-test-b", model, CreateTestMetadata());
        _registry.RegisterModel("other-model", model, CreateTestMetadata());

        // Act
        var criteria = new ModelSearchCriteria<double> { NamePattern = "search-test" };
        var results = _registry.SearchModels(criteria);

        // Assert
        Assert.Equal(2, results.Count);
    }

    [Fact]
    public void SearchModels_ByStage_ReturnsMatchingModels()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("stage-search-1", model, CreateTestMetadata());
        _registry.RegisterModel("stage-search-2", model, CreateTestMetadata());
        _registry.TransitionModelStage("stage-search-1", 1, ModelStage.Production);

        // Act
        var criteria = new ModelSearchCriteria<double> { Stage = ModelStage.Production };
        var results = _registry.SearchModels(criteria);

        // Assert
        Assert.Single(results);
        Assert.Equal("stage-search-1", results[0].Name);
    }

    #endregion

    #region Delete Tests

    [Fact]
    public void DeleteModelVersion_RemovesSpecificVersion()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("delete-version-test", model, CreateTestMetadata());
        _registry.CreateModelVersion("delete-version-test", model, CreateTestMetadata());

        // Act
        _registry.DeleteModelVersion("delete-version-test", 1);

        // Assert
        Assert.Throws<ArgumentException>(() => _registry.GetModel("delete-version-test", 1));
        // Version 2 should still exist
        var v2 = _registry.GetModel("delete-version-test", 2);
        Assert.NotNull(v2);
    }

    [Fact]
    public void DeleteModelVersion_LastVersion_RemovesEntireModel()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("delete-last-version", model, CreateTestMetadata());

        // Act
        _registry.DeleteModelVersion("delete-last-version", 1);

        // Assert
        var models = _registry.ListModels();
        Assert.DoesNotContain("delete-last-version", models);
    }

    [Fact]
    public void DeleteModel_RemovesAllVersions()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("delete-all-test", model, CreateTestMetadata());
        _registry.CreateModelVersion("delete-all-test", model, CreateTestMetadata());
        _registry.CreateModelVersion("delete-all-test", model, CreateTestMetadata());

        // Act
        _registry.DeleteModel("delete-all-test");

        // Assert
        var models = _registry.ListModels();
        Assert.DoesNotContain("delete-all-test", models);
    }

    #endregion

    #region Model Comparison Tests

    [Fact]
    public void CompareModels_ReturnsMetadataDifferences()
    {
        // Arrange
        var model = new MockModel();
        var metadata1 = CreateTestMetadata();
        metadata1.FeatureCount = 10;
        metadata1.Complexity = 5;

        var metadata2 = CreateTestMetadata();
        metadata2.FeatureCount = 15;
        metadata2.Complexity = 8;

        _registry.RegisterModel("compare-test", model, metadata1);
        _registry.CreateModelVersion("compare-test", model, metadata2);

        // Act
        var comparison = _registry.CompareModels("compare-test", 1, 2);

        // Assert
        Assert.NotNull(comparison);
        Assert.Equal(1, comparison.Version1);
        Assert.Equal(2, comparison.Version2);
        Assert.True(comparison.ArchitectureChanged); // FeatureCount changed
        Assert.True(comparison.MetadataDifferences.ContainsKey("FeatureCount"));
    }

    [Fact]
    public void GetModelLineage_ReturnsLineageInfo()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("lineage-test", model, CreateTestMetadata());

        // Act
        var lineage = _registry.GetModelLineage("lineage-test", 1);

        // Assert
        Assert.NotNull(lineage);
        Assert.Equal("lineage-test", lineage.ModelName);
        Assert.Equal(1, lineage.Version);
    }

    #endregion

    #region Update Tests

    [Fact]
    public void UpdateModelMetadata_UpdatesMetadata()
    {
        // Arrange
        var model = new MockModel();
        var initialMetadata = CreateTestMetadata();
        _registry.RegisterModel("update-metadata-test", model, initialMetadata);

        var updatedMetadata = CreateTestMetadata();
        updatedMetadata.Description = "Updated description";
        updatedMetadata.FeatureCount = 20;

        // Act
        _registry.UpdateModelMetadata("update-metadata-test", 1, updatedMetadata);

        // Assert
        var registeredModel = _registry.GetModel("update-metadata-test", 1);
        Assert.Equal("Updated description", registeredModel.Metadata.Description);
        Assert.Equal(20, registeredModel.Metadata.FeatureCount);
    }

    [Fact]
    public void UpdateModelTags_AddsOrUpdatesTags()
    {
        // Arrange
        var model = new MockModel();
        var tags = new Dictionary<string, string> { ["version"] = "1.0" };
        _registry.RegisterModel("update-tags-test", model, CreateTestMetadata(), tags);

        var newTags = new Dictionary<string, string>
        {
            ["version"] = "1.1",
            ["author"] = "test-user"
        };

        // Act
        _registry.UpdateModelTags("update-tags-test", 1, newTags);

        // Assert
        var registeredModel = _registry.GetModel("update-tags-test", 1);
        Assert.Equal("1.1", registeredModel.Tags["version"]);
        Assert.Equal("test-user", registeredModel.Tags["author"]);
    }

    #endregion

    #region Persistence Tests

    [Fact]
    public void Registry_PersistsModelsToDisk()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("persistence-test", model, CreateTestMetadata());

        // Act - Create new registry pointing to same directory
        var registry2 = new ModelRegistry<double, double[], double>(_testDirectory);
        var models = registry2.ListModels();

        // Assert
        Assert.Contains("persistence-test", models);
    }

    [Fact]
    public void Registry_PersistsVersionsToDisk()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("version-persistence-test", model, CreateTestMetadata());
        _registry.CreateModelVersion("version-persistence-test", model, CreateTestMetadata());

        // Act - Create new registry pointing to same directory
        var registry2 = new ModelRegistry<double, double[], double>(_testDirectory);
        var versions = registry2.ListModelVersions("version-persistence-test");

        // Assert
        Assert.Equal(2, versions.Count);
    }

    #endregion

    #region Storage Path Tests

    [Fact]
    public void GetModelStoragePath_ReturnsValidPath()
    {
        // Arrange
        var model = new MockModel();
        _registry.RegisterModel("storage-path-test", model, CreateTestMetadata());

        // Act
        var path = _registry.GetModelStoragePath("storage-path-test", 1);

        // Assert
        Assert.NotNull(path);
        Assert.NotEmpty(path);
        Assert.Contains("storage-path-test", path);
    }

    #endregion
}
