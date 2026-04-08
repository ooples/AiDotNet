using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.ModelRegistry;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ModelRegistry;

/// <summary>
/// Integration tests for the ModelRegistry module.
/// These tests verify model registration, versioning, stage transitions, and metadata management.
/// </summary>
public class ModelRegistryIntegrationTests : IDisposable
{
    private readonly string _testDirectory;

    public ModelRegistryIntegrationTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), "AiDotNet_ModelRegistry_Tests_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_testDirectory);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_testDirectory))
            {
                Directory.Delete(_testDirectory, true);
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
    }

    #region Constructor Tests

    [Fact(Timeout = 120000)]
    public async Task Constructor_WithValidDirectory_CreatesRegistry()
    {
        var registryDir = Path.Combine(_testDirectory, "registry1");
        var registry = new ModelRegistry<double, double[], double>(registryDir);

        Assert.True(Directory.Exists(registryDir));
    }

    [Fact(Timeout = 120000)]
    public async Task Constructor_WithNullDirectory_UsesDefault()
    {
        // This would create in current directory, so we skip to avoid pollution
        // Just verify it doesn't throw
        var cwd = Directory.GetCurrentDirectory();
        var expectedDir = Path.Combine(cwd, "model_registry");

        try
        {
            // Clean up if exists from previous test
            if (Directory.Exists(expectedDir))
            {
                Directory.Delete(expectedDir, true);
            }

            var registry = new ModelRegistry<double, double[], double>();
            Assert.True(Directory.Exists(expectedDir));
        }
        finally
        {
            // Clean up
            if (Directory.Exists(expectedDir))
            {
                Directory.Delete(expectedDir, true);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Constructor_CreatesDirectoryIfNotExists()
    {
        var registryDir = Path.Combine(_testDirectory, "new_registry");
        Assert.False(Directory.Exists(registryDir));

        var registry = new ModelRegistry<double, double[], double>(registryDir);

        Assert.True(Directory.Exists(registryDir));
    }

    #endregion

    #region RegisterModel Tests

    [Fact(Timeout = 120000)]
    public async Task RegisterModel_WithValidModel_ReturnsModelId()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_register");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        var modelId = registry.RegisterModel("test_model", model, metadata);

        Assert.NotNull(modelId);
        Assert.NotEmpty(modelId);
    }

    [Fact(Timeout = 120000)]
    public async Task RegisterModel_WithTags_StoresTags()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_tags");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();
        var tags = new Dictionary<string, string>
        {
            { "environment", "production" },
            { "framework", "AiDotNet" }
        };

        registry.RegisterModel("tagged_model", model, metadata, tags);
        var retrieved = registry.GetModel("tagged_model");

        Assert.Equal("production", retrieved.Tags["environment"]);
        Assert.Equal("AiDotNet", retrieved.Tags["framework"]);
    }

    [Fact(Timeout = 120000)]
    public async Task RegisterModel_NullName_ThrowsArgumentException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_null_name");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        Assert.Throws<ArgumentException>(() => registry.RegisterModel(null!, model, metadata));
    }

    [Fact(Timeout = 120000)]
    public async Task RegisterModel_EmptyName_ThrowsArgumentException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_empty_name");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        Assert.Throws<ArgumentException>(() => registry.RegisterModel("", model, metadata));
    }

    [Fact(Timeout = 120000)]
    public async Task RegisterModel_NullModel_ThrowsArgumentNullException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_null_model");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var metadata = CreateTestMetadata();

        Assert.Throws<ArgumentNullException>(() => registry.RegisterModel<object>("test", null!, metadata));
    }

    [Fact(Timeout = 120000)]
    public async Task RegisterModel_SameNameTwice_CreatesVersions()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_versions");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("versioned_model", model, metadata);
        registry.RegisterModel("versioned_model", model, metadata);

        var versions = registry.ListModelVersions("versioned_model");
        Assert.Equal(2, versions.Count);
    }

    #endregion

    #region CreateModelVersion Tests

    [Fact(Timeout = 120000)]
    public async Task CreateModelVersion_ExistingModel_ReturnsNewVersion()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_create_version");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("model_v1", model, metadata);
        var newVersion = registry.CreateModelVersion("model_v1", model, metadata, "Second version");

        Assert.Equal(2, newVersion);
    }

    [Fact(Timeout = 120000)]
    public async Task CreateModelVersion_NonExistentModel_ThrowsArgumentException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_create_nonexistent");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        Assert.Throws<ArgumentException>(() =>
            registry.CreateModelVersion("nonexistent", model, metadata));
    }

    [Fact(Timeout = 120000)]
    public async Task CreateModelVersion_WithDescription_StoresDescription()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_version_desc");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("model_desc", model, metadata);
        registry.CreateModelVersion("model_desc", model, metadata, "Improved accuracy");

        var versions = registry.ListModelVersions("model_desc");
        var v2 = versions.First(v => v.Version == 2);

        Assert.Equal("Improved accuracy", v2.Description);
    }

    #endregion

    #region GetModel Tests

    [Fact(Timeout = 120000)]
    public async Task GetModel_ExistingModel_ReturnsModel()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_get");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("get_test", model, metadata);
        var retrieved = registry.GetModel("get_test");

        Assert.NotNull(retrieved);
        Assert.Equal("get_test", retrieved.Name);
        Assert.Equal(1, retrieved.Version);
    }

    [Fact(Timeout = 120000)]
    public async Task GetModel_SpecificVersion_ReturnsCorrectVersion()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_get_version");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("multi_version", model, metadata);
        registry.CreateModelVersion("multi_version", model, metadata);
        registry.CreateModelVersion("multi_version", model, metadata);

        var v2 = registry.GetModel("multi_version", 2);

        Assert.Equal(2, v2.Version);
    }

    [Fact(Timeout = 120000)]
    public async Task GetModel_NonExistentModel_ThrowsArgumentException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_get_nonexistent");
        var registry = new ModelRegistry<double, double[], double>(registryDir);

        Assert.Throws<ArgumentException>(() => registry.GetModel("nonexistent"));
    }

    [Fact(Timeout = 120000)]
    public async Task GetModel_NonExistentVersion_ThrowsArgumentException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_get_bad_version");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("one_version", model, metadata);

        Assert.Throws<ArgumentException>(() => registry.GetModel("one_version", 999));
    }

    #endregion

    #region GetLatestModel Tests

    [Fact(Timeout = 120000)]
    public async Task GetLatestModel_MultipleVersions_ReturnsLatest()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_latest");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("latest_test", model, metadata);
        registry.CreateModelVersion("latest_test", model, metadata);
        registry.CreateModelVersion("latest_test", model, metadata);

        var latest = registry.GetLatestModel("latest_test");

        Assert.Equal(3, latest.Version);
    }

    #endregion

    #region GetModelByStage Tests

    [Fact(Timeout = 120000)]
    public async Task GetModelByStage_ExistingStage_ReturnsModel()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_stage");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("stage_test", model, metadata);
        registry.TransitionModelStage("stage_test", 1, ModelStage.Production);

        var production = registry.GetModelByStage("stage_test", ModelStage.Production);

        Assert.NotNull(production);
        Assert.Equal(ModelStage.Production, production.Stage);
    }

    [Fact(Timeout = 120000)]
    public async Task GetModelByStage_NoModelInStage_ReturnsNull()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_no_stage");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("no_prod", model, metadata);

        var production = registry.GetModelByStage("no_prod", ModelStage.Production);

        Assert.Null(production);
    }

    [Fact(Timeout = 120000)]
    public async Task GetModelByStage_NonExistentModel_ReturnsNull()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_nonexistent_stage");
        var registry = new ModelRegistry<double, double[], double>(registryDir);

        var result = registry.GetModelByStage("nonexistent", ModelStage.Production);

        Assert.Null(result);
    }

    #endregion

    #region TransitionModelStage Tests

    [Fact(Timeout = 120000)]
    public async Task TransitionModelStage_ToProduction_UpdatesStage()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_transition");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("transition_test", model, metadata);
        registry.TransitionModelStage("transition_test", 1, ModelStage.Production);

        var retrieved = registry.GetModel("transition_test", 1);
        Assert.Equal(ModelStage.Production, retrieved.Stage);
    }

    [Fact(Timeout = 120000)]
    public async Task TransitionModelStage_ArchivesPreviousInStage()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_archive_prev");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("archive_test", model, metadata);
        registry.CreateModelVersion("archive_test", model, metadata);

        registry.TransitionModelStage("archive_test", 1, ModelStage.Production);
        registry.TransitionModelStage("archive_test", 2, ModelStage.Production);

        var v1 = registry.GetModel("archive_test", 1);
        var v2 = registry.GetModel("archive_test", 2);

        Assert.Equal(ModelStage.Archived, v1.Stage);
        Assert.Equal(ModelStage.Production, v2.Stage);
    }

    [Fact(Timeout = 120000)]
    public async Task TransitionModelStage_ArchivePreviousFalse_DoesNotArchive()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_no_archive");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("no_archive_test", model, metadata);
        registry.CreateModelVersion("no_archive_test", model, metadata);

        registry.TransitionModelStage("no_archive_test", 1, ModelStage.Production, archivePrevious: false);
        registry.TransitionModelStage("no_archive_test", 2, ModelStage.Production, archivePrevious: false);

        var v1 = registry.GetModel("no_archive_test", 1);
        var v2 = registry.GetModel("no_archive_test", 2);

        // Both should be in Production since archivePrevious is false
        Assert.Equal(ModelStage.Production, v1.Stage);
        Assert.Equal(ModelStage.Production, v2.Stage);
    }

    #endregion

    #region ListModels Tests

    [Fact(Timeout = 120000)]
    public async Task ListModels_NoFilter_ReturnsAllModels()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_list");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("model_a", model, metadata);
        registry.RegisterModel("model_b", model, metadata);
        registry.RegisterModel("model_c", model, metadata);

        var models = registry.ListModels();

        Assert.Equal(3, models.Count);
        Assert.Contains("model_a", models);
        Assert.Contains("model_b", models);
        Assert.Contains("model_c", models);
    }

    [Fact(Timeout = 120000)]
    public async Task ListModels_WithFilter_ReturnsMatchingModels()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_list_filter");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("classifier_a", model, metadata);
        registry.RegisterModel("classifier_b", model, metadata);
        registry.RegisterModel("regressor_a", model, metadata);

        var classifiers = registry.ListModels(filter: "classifier");

        Assert.Equal(2, classifiers.Count);
        Assert.All(classifiers, m => Assert.Contains("classifier", m));
    }

    [Fact(Timeout = 120000)]
    public async Task ListModels_WithTags_ReturnsMatchingModels()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_list_tags");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("tagged_a", model, metadata, new Dictionary<string, string> { { "env", "prod" } });
        registry.RegisterModel("tagged_b", model, metadata, new Dictionary<string, string> { { "env", "dev" } });

        var prodModels = registry.ListModels(tags: new Dictionary<string, string> { { "env", "prod" } });

        Assert.Single(prodModels);
        Assert.Equal("tagged_a", prodModels[0]);
    }

    #endregion

    #region ListModelVersions Tests

    [Fact(Timeout = 120000)]
    public async Task ListModelVersions_ReturnsAllVersions()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_list_versions");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("versioned", model, metadata);
        registry.CreateModelVersion("versioned", model, metadata);
        registry.CreateModelVersion("versioned", model, metadata);

        var versions = registry.ListModelVersions("versioned");

        Assert.Equal(3, versions.Count);
        Assert.Contains(versions, v => v.Version == 1);
        Assert.Contains(versions, v => v.Version == 2);
        Assert.Contains(versions, v => v.Version == 3);
    }

    [Fact(Timeout = 120000)]
    public async Task ListModelVersions_NonExistentModel_ThrowsArgumentException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_list_nonexistent");
        var registry = new ModelRegistry<double, double[], double>(registryDir);

        Assert.Throws<ArgumentException>(() => registry.ListModelVersions("nonexistent"));
    }

    #endregion

    #region SearchModels Tests

    [Fact(Timeout = 120000)]
    public async Task SearchModels_ByNamePattern_ReturnsMatching()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_search_name");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("search_classifier", model, metadata);
        registry.RegisterModel("search_regressor", model, metadata);

        var criteria = new ModelSearchCriteria<double> { NamePattern = "classifier" };
        var results = registry.SearchModels(criteria);

        Assert.Single(results);
        Assert.Equal("search_classifier", results[0].Name);
    }

    [Fact(Timeout = 120000)]
    public async Task SearchModels_ByStage_ReturnsMatching()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_search_stage");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("stage_search_a", model, metadata);
        registry.RegisterModel("stage_search_b", model, metadata);
        registry.TransitionModelStage("stage_search_a", 1, ModelStage.Production);

        var criteria = new ModelSearchCriteria<double> { Stage = ModelStage.Production };
        var results = registry.SearchModels(criteria);

        Assert.Single(results);
        Assert.Equal("stage_search_a", results[0].Name);
    }

    [Fact(Timeout = 120000)]
    public async Task SearchModels_ByVersionRange_ReturnsMatching()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_search_version");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("version_search", model, metadata);
        registry.CreateModelVersion("version_search", model, metadata);
        registry.CreateModelVersion("version_search", model, metadata);
        registry.CreateModelVersion("version_search", model, metadata);

        var criteria = new ModelSearchCriteria<double> { MinVersion = 2, MaxVersion = 3 };
        var results = registry.SearchModels(criteria);

        Assert.Equal(2, results.Count);
        Assert.All(results, r => Assert.InRange(r.Version, 2, 3));
    }

    [Fact(Timeout = 120000)]
    public async Task SearchModels_NullCriteria_ThrowsArgumentNullException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_search_null");
        var registry = new ModelRegistry<double, double[], double>(registryDir);

        Assert.Throws<ArgumentNullException>(() => registry.SearchModels(null!));
    }

    #endregion

    #region UpdateModelMetadata Tests

    [Fact(Timeout = 120000)]
    public async Task UpdateModelMetadata_UpdatesMetadata()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_update_metadata");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("metadata_update", model, metadata);

        var newMetadata = new ModelMetadata<double>
        {
            FeatureCount = 20,
            Complexity = 8
        };

        registry.UpdateModelMetadata("metadata_update", 1, newMetadata);
        var retrieved = registry.GetModel("metadata_update", 1);

        Assert.Equal(20, retrieved.Metadata?.FeatureCount);
        Assert.Equal(8, retrieved.Metadata?.Complexity);
    }

    [Fact(Timeout = 120000)]
    public async Task UpdateModelMetadata_NullMetadata_ThrowsArgumentNullException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_update_null");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("null_metadata", model, metadata);

        Assert.Throws<ArgumentNullException>(() =>
            registry.UpdateModelMetadata("null_metadata", 1, null!));
    }

    #endregion

    #region UpdateModelTags Tests

    [Fact(Timeout = 120000)]
    public async Task UpdateModelTags_AddsTags()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_update_tags");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("tags_update", model, metadata);

        var newTags = new Dictionary<string, string>
        {
            { "author", "test" },
            { "version_type", "release" }
        };

        registry.UpdateModelTags("tags_update", 1, newTags);
        var retrieved = registry.GetModel("tags_update", 1);

        Assert.Equal("test", retrieved.Tags["author"]);
        Assert.Equal("release", retrieved.Tags["version_type"]);
    }

    [Fact(Timeout = 120000)]
    public async Task UpdateModelTags_NullTags_ThrowsArgumentNullException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_tags_null");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("null_tags", model, metadata);

        Assert.Throws<ArgumentNullException>(() =>
            registry.UpdateModelTags("null_tags", 1, null!));
    }

    #endregion

    #region DeleteModelVersion Tests

    [Fact(Timeout = 120000)]
    public async Task DeleteModelVersion_RemovesSpecificVersion()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_delete_version");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("delete_version", model, metadata);
        registry.CreateModelVersion("delete_version", model, metadata);

        registry.DeleteModelVersion("delete_version", 1);

        var versions = registry.ListModelVersions("delete_version");
        Assert.Single(versions);
        Assert.Equal(2, versions[0].Version);
    }

    [Fact(Timeout = 120000)]
    public async Task DeleteModelVersion_LastVersion_RemovesModel()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_delete_last");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("delete_last", model, metadata);
        registry.DeleteModelVersion("delete_last", 1);

        var models = registry.ListModels();
        Assert.DoesNotContain("delete_last", models);
    }

    [Fact(Timeout = 120000)]
    public async Task DeleteModelVersion_NonExistent_DoesNotThrow()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_delete_nonexistent");
        var registry = new ModelRegistry<double, double[], double>(registryDir);

        // Should not throw
        registry.DeleteModelVersion("nonexistent", 1);
    }

    #endregion

    #region DeleteModel Tests

    [Fact(Timeout = 120000)]
    public async Task DeleteModel_RemovesAllVersions()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_delete_all");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("delete_all", model, metadata);
        registry.CreateModelVersion("delete_all", model, metadata);
        registry.CreateModelVersion("delete_all", model, metadata);

        registry.DeleteModel("delete_all");

        var models = registry.ListModels();
        Assert.DoesNotContain("delete_all", models);
    }

    [Fact(Timeout = 120000)]
    public async Task DeleteModel_NonExistent_DoesNotThrow()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_delete_model_none");
        var registry = new ModelRegistry<double, double[], double>(registryDir);

        // Should not throw
        registry.DeleteModel("nonexistent");
    }

    #endregion

    #region CompareModels Tests

    [Fact(Timeout = 120000)]
    public async Task CompareModels_DifferentMetadata_ReportsChanges()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_compare");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();

        var metadata1 = new ModelMetadata<double>
        {
            FeatureCount = 10,
            Complexity = 3,
        };

        var metadata2 = new ModelMetadata<double>
        {
            FeatureCount = 20,
            Complexity = 8,
        };

        registry.RegisterModel("compare_model", model, metadata1);
        registry.CreateModelVersion("compare_model", model, metadata2);

        var comparison = registry.CompareModels("compare_model", 1, 2);

        Assert.Equal(1, comparison.Version1);
        Assert.Equal(2, comparison.Version2);
        Assert.True(comparison.ArchitectureChanged);
        Assert.True(comparison.MetadataDifferences.Count > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task CompareModels_SameMetadata_NoChanges()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_compare_same");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("compare_same", model, metadata);
        registry.CreateModelVersion("compare_same", model, metadata);

        var comparison = registry.CompareModels("compare_same", 1, 2);

        Assert.False(comparison.ArchitectureChanged);
    }

    #endregion

    #region GetModelLineage Tests

    [Fact(Timeout = 120000)]
    public async Task GetModelLineage_ReturnsLineageInfo()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_lineage");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("lineage_model", model, metadata);

        var lineage = registry.GetModelLineage("lineage_model", 1);

        Assert.Equal("lineage_model", lineage.ModelName);
        Assert.Equal(1, lineage.Version);
    }

    #endregion

    #region ArchiveModel Tests

    [Fact(Timeout = 120000)]
    public async Task ArchiveModel_SetsArchivedStage()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_archive");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("archive_model", model, metadata);
        registry.ArchiveModel("archive_model", 1);

        var retrieved = registry.GetModel("archive_model", 1);
        Assert.Equal(ModelStage.Archived, retrieved.Stage);
    }

    #endregion

    #region GetModelStoragePath Tests

    [Fact(Timeout = 120000)]
    public async Task GetModelStoragePath_ReturnsValidPath()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_path");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("path_model", model, metadata);

        var path = registry.GetModelStoragePath("path_model", 1);

        Assert.Contains("path_model", path);
        Assert.Contains("v1", path);
        Assert.EndsWith(".json", path);
    }

    #endregion

    #region ModelCard Tests

    [Fact(Timeout = 120000)]
    public async Task AttachModelCard_StoresModelCard()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_modelcard");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("modelcard_test", model, metadata);

        var modelCard = new AiDotNet.AdversarialRobustness.Documentation.ModelCard
        {
            ModelName = "modelcard_test",
            Version = "1.0.0",
            Developers = "Test Developer"
        };

        registry.AttachModelCard("modelcard_test", 1, modelCard);

        var retrieved = registry.GetModelCard("modelcard_test", 1);
        Assert.NotNull(retrieved);
        Assert.Equal("Test Developer", retrieved.Developers);
    }

    [Fact(Timeout = 120000)]
    public async Task AttachModelCard_NullModelCard_ThrowsArgumentNullException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_null_card");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("null_card", model, metadata);

        Assert.Throws<ArgumentNullException>(() =>
            registry.AttachModelCard("null_card", 1, null!));
    }

    [Fact(Timeout = 120000)]
    public async Task GetModelCard_NoCard_ReturnsNull()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_no_card");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("no_card", model, metadata);

        var card = registry.GetModelCard("no_card", 1);
        Assert.Null(card);
    }

    [Fact(Timeout = 120000)]
    public async Task GenerateModelCard_CreatesCard()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_generate_card");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("generate_card", model, metadata);

        var card = registry.GenerateModelCard("generate_card", 1, "Auto-Generated");

        Assert.NotNull(card);
        Assert.Equal("generate_card", card.ModelName);
        Assert.Equal("Auto-Generated", card.Developers);
    }

    [Fact(Timeout = 120000)]
    public async Task SaveModelCard_CreatesFile()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_save_card");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("save_card", model, metadata);

        var cardPath = Path.Combine(registryDir, "save_card", "modelcard.json");
        registry.SaveModelCard("save_card", 1, cardPath);

        Assert.True(File.Exists(cardPath));
    }

    [Fact(Timeout = 120000)]
    public async Task SaveModelCard_EmptyPath_ThrowsArgumentException()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_empty_path");
        var registry = new ModelRegistry<double, double[], double>(registryDir);
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        registry.RegisterModel("empty_path", model, metadata);

        Assert.Throws<ArgumentException>(() =>
            registry.SaveModelCard("empty_path", 1, ""));
    }

    #endregion

    #region Persistence Tests

    [Fact(Timeout = 120000)]
    public async Task Registry_PersistsAcrossInstances()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_persist");
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        // Create and register model
        var registry1 = new ModelRegistry<double, double[], double>(registryDir);
        registry1.RegisterModel("persisted_model", model, metadata);

        // Create new registry instance pointing to same directory
        var registry2 = new ModelRegistry<double, double[], double>(registryDir);
        var models = registry2.ListModels();

        Assert.Contains("persisted_model", models);
    }

    [Fact(Timeout = 120000)]
    public async Task Registry_LoadsExistingVersions()
    {
        var registryDir = Path.Combine(_testDirectory, "registry_load_versions");
        var model = new MockModel();
        var metadata = CreateTestMetadata();

        // Create and register multiple versions
        var registry1 = new ModelRegistry<double, double[], double>(registryDir);
        registry1.RegisterModel("multi_persist", model, metadata);
        registry1.CreateModelVersion("multi_persist", model, metadata);
        registry1.CreateModelVersion("multi_persist", model, metadata);

        // Create new registry instance and verify versions loaded
        var registry2 = new ModelRegistry<double, double[], double>(registryDir);
        var versions = registry2.ListModelVersions("multi_persist");

        Assert.Equal(3, versions.Count);
    }

    #endregion

    #region Helper Methods

    private static ModelMetadata<double> CreateTestMetadata()
    {
        return new ModelMetadata<double>
        {
            FeatureCount = 10,
            Complexity = 5, // 1-10 scale
        };
    }

    #endregion

    #region Mock Model Class

    /// <summary>
    /// Mock model implementation for testing.
    /// </summary>
    private class MockModel : IModel<double[], double, object>
    {
        public double Predict(double[] input) => 0.0;

        public void Train(double[] input, double expectedOutput) { }

        public object GetModelMetadata() => new object();
    }

    #endregion
}
