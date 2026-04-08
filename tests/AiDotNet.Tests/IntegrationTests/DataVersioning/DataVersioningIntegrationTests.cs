using AiDotNet.DataVersioning;
using Xunit;

// Alias to avoid conflict with AiDotNet.DataVersionControl namespace
using DVC = AiDotNet.DataVersioning.DataVersionControl;

namespace AiDotNet.Tests.IntegrationTests.DataVersioning;

/// <summary>
/// Integration tests for the DataVersioning module.
/// Tests the DVC-like data version control system with content-addressable storage.
/// </summary>
public class DataVersioningIntegrationTests : IDisposable
{
    private readonly string _testDir;
    private readonly string _storageDir;
    private readonly string _sourceDir;

    public DataVersioningIntegrationTests()
    {
        // Create unique test directories for each test run
        _testDir = Path.Combine(Path.GetTempPath(), $"DataVersioningTests_{Guid.NewGuid():N}");
        _storageDir = Path.Combine(_testDir, "storage");
        _sourceDir = Path.Combine(_testDir, "source");

        Directory.CreateDirectory(_testDir);
        Directory.CreateDirectory(_sourceDir);
    }

    public void Dispose()
    {
        // Clean up test directories
        if (Directory.Exists(_testDir))
        {
            try
            {
                Directory.Delete(_testDir, true);
            }
            catch
            {
                // Ignore cleanup errors in tests
            }
        }
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithStorageDirectory_CreatesDirectoryStructure()
    {
        // Act
        using var dvc = new DVC(_storageDir);

        // Assert
        Assert.Equal(_storageDir, dvc.StorageDirectory);
        Assert.True(Directory.Exists(_storageDir));
        Assert.True(Directory.Exists(Path.Combine(_storageDir, "datasets")));
        Assert.True(Directory.Exists(Path.Combine(_storageDir, "objects")));
        Assert.True(Directory.Exists(Path.Combine(_storageDir, "lineage")));
    }

    [Fact]
    public void Constructor_WithNullStorageDirectory_UsesDefaultDirectory()
    {
        var expectedDefault = Path.Combine(Directory.GetCurrentDirectory(), "data-versions");
        bool existedBefore = Directory.Exists(expectedDefault);

        // Act
        using var dvc = new DVC(null);

        // Assert
        Assert.Equal(expectedDefault, dvc.StorageDirectory);

        // Cleanup - only delete if we created it during this test, and only if empty or nearly empty
        if (!existedBefore && Directory.Exists(expectedDefault))
        {
            try
            {
                // Only delete if it's empty or contains only the test-created subdirectories
                var files = Directory.GetFiles(expectedDefault, "*", SearchOption.AllDirectories);
                if (files.Length == 0)
                {
                    Directory.Delete(expectedDefault, true);
                }
            }
            catch
            {
                // Ignore cleanup errors - test data may need manual cleanup
            }
        }
    }

    [Fact]
    public void Constructor_LoadsExistingData()
    {
        // Arrange - Create a dataset and version in first instance
        string datasetId;
        string versionId;
        CreateTestFile("test.txt", "test content");

        using (var dvc1 = new DVC(_storageDir))
        {
            datasetId = dvc1.CreateDataset("test-dataset", "Test description");
            var version = dvc1.AddVersion(datasetId, Path.Combine(_sourceDir, "test.txt"), "Initial version");
            versionId = version.VersionId;
        }

        // Act - Create new instance and verify data is loaded
        using var dvc2 = new DVC(_storageDir);

        // Assert
        var datasets = dvc2.ListDatasets();
        Assert.Single(datasets);
        Assert.Equal("test-dataset", datasets[0].Name);

        var versions = dvc2.ListVersions(datasetId);
        Assert.Single(versions);
        Assert.Equal(versionId, versions[0].VersionId);
    }

    #endregion

    #region CreateDataset Tests

    [Fact]
    public void CreateDataset_WithValidName_CreatesDataset()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);

        // Act
        var datasetId = dvc.CreateDataset("training-data", "Training images for model");

        // Assert
        Assert.False(string.IsNullOrEmpty(datasetId));
        Assert.Equal(12, datasetId.Length); // Generated IDs are 12 chars

        var datasets = dvc.ListDatasets();
        Assert.Single(datasets);
        Assert.Equal("training-data", datasets[0].Name);
        Assert.Equal("Training images for model", datasets[0].Description);
    }

    [Fact]
    public void CreateDataset_WithMetadata_StoresMetadata()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var metadata = new Dictionary<string, string>
        {
            ["source"] = "kaggle",
            ["format"] = "csv"
        };

        // Act
        var datasetId = dvc.CreateDataset("csv-data", "CSV dataset", metadata);

        // Assert
        var datasets = dvc.ListDatasets();
        Assert.Single(datasets);
        Assert.Equal("kaggle", datasets[0].Metadata["source"]);
        Assert.Equal("csv", datasets[0].Metadata["format"]);
    }

    [Fact]
    public void CreateDataset_WithDuplicateName_ReturnsExistingId()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var id1 = dvc.CreateDataset("test-dataset", "First description");

        // Act
        var id2 = dvc.CreateDataset("test-dataset", "Second description");

        // Assert - Should return the same ID
        Assert.Equal(id1, id2);
        Assert.Single(dvc.ListDatasets());
    }

    [Fact]
    public void CreateDataset_WithEmptyName_ThrowsArgumentException()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => dvc.CreateDataset(""));
        Assert.Contains("name", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void CreateDataset_WithWhitespaceName_ThrowsArgumentException()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => dvc.CreateDataset("   "));
    }

    #endregion

    #region AddVersion Tests

    [Fact]
    public void AddVersion_WithSingleFile_CreatesVersion()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "col1,col2\n1,2\n3,4");

        // Act
        var version = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Initial import");

        // Assert
        Assert.NotNull(version);
        Assert.Equal(datasetId, version.DatasetId);
        Assert.Equal(1, version.VersionNumber);
        Assert.Equal("Initial import", version.Message);
        Assert.Equal(1, version.FileCount);
        Assert.True(version.SizeBytes > 0);
        Assert.False(string.IsNullOrEmpty(version.ContentHash));
        Assert.Single(version.Files);
        Assert.Equal("data.csv", version.Files[0].RelativePath);
    }

    [Fact]
    public void AddVersion_WithDirectory_CreatesVersionWithAllFiles()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        var subDir = Path.Combine(_sourceDir, "dataset");
        Directory.CreateDirectory(subDir);
        File.WriteAllText(Path.Combine(subDir, "train.csv"), "a,b\n1,2");
        File.WriteAllText(Path.Combine(subDir, "test.csv"), "a,b\n3,4");
        Directory.CreateDirectory(Path.Combine(subDir, "nested"));
        File.WriteAllText(Path.Combine(subDir, "nested", "labels.txt"), "0\n1");

        // Act
        var version = dvc.AddVersion(datasetId, subDir, "Full dataset import");

        // Assert
        Assert.Equal(3, version.FileCount);
        Assert.Equal(3, version.Files.Count);

        var filePaths = version.Files.Select(f => f.RelativePath).ToList();
        Assert.Contains("train.csv", filePaths);
        Assert.Contains("test.csv", filePaths);
        // Nested file path includes subdirectory
        Assert.True(filePaths.Any(p => p.Contains("labels.txt")));
    }

    [Fact]
    public void AddVersion_WithUnchangedContent_ReturnsExistingVersion()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "fixed content");

        var v1 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "First version");

        // Act - Add same content again
        var v2 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Second version");

        // Assert - Should return the same version
        Assert.Equal(v1.VersionId, v2.VersionId);
        Assert.Equal(v1.ContentHash, v2.ContentHash);
        Assert.Single(dvc.ListVersions(datasetId));
    }

    [Fact]
    public void AddVersion_WithModifiedContent_CreatesNewVersion()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "version 1");

        var v1 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "First version");

        // Modify the file
        CreateTestFile("data.csv", "version 2");

        // Act
        var v2 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Second version");

        // Assert
        Assert.NotEqual(v1.VersionId, v2.VersionId);
        Assert.NotEqual(v1.ContentHash, v2.ContentHash);
        Assert.Equal(2, dvc.ListVersions(datasetId).Count);
        Assert.Equal(1, v1.VersionNumber);
        Assert.Equal(2, v2.VersionNumber);
    }

    [Fact]
    public void AddVersion_WithMetadata_StoresMetadata()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "content");
        var metadata = new Dictionary<string, string>
        {
            ["preprocessor"] = "normalize",
            ["split"] = "train"
        };

        // Act
        var version = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "With metadata", metadata);

        // Assert
        Assert.Equal("normalize", version.Metadata["preprocessor"]);
        Assert.Equal("train", version.Metadata["split"]);
    }

    [Fact]
    public void AddVersion_WithNonexistentPath_ThrowsFileNotFoundException()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() =>
            dvc.AddVersion(datasetId, "/nonexistent/path/data.csv", "Should fail"));
    }

    [Fact]
    public void AddVersion_WithNonexistentDataset_ThrowsArgumentException()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        CreateTestFile("data.csv", "content");

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() =>
            dvc.AddVersion("nonexistent-id", Path.Combine(_sourceDir, "data.csv"), "Should fail"));
        Assert.Contains("Dataset not found", ex.Message);
    }

    [Fact]
    public void AddVersion_WithEmptyDatasetId_ThrowsArgumentException()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        CreateTestFile("data.csv", "content");

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            dvc.AddVersion("", Path.Combine(_sourceDir, "data.csv"), "Should fail"));
    }

    #endregion

    #region GetVersion Tests

    [Fact]
    public void GetVersion_ByVersionId_ReturnsCorrectVersion()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("v1.csv", "v1");
        CreateTestFile("v2.csv", "v2");

        var v1 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "v1.csv"), "Version 1");
        CreateTestFile("v1.csv", "v1-modified"); // Create a different file to get new version
        var v2 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "v1.csv"), "Version 2");

        // Act
        var retrieved = dvc.GetVersion(datasetId, v1.VersionId);

        // Assert
        Assert.Equal(v1.VersionId, retrieved.VersionId);
        Assert.Equal(1, retrieved.VersionNumber);
    }

    [Fact]
    public void GetVersion_Latest_ReturnsLatestVersion()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "v1");
        dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 1");

        CreateTestFile("data.csv", "v2");
        var v2 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 2");

        // Act
        var latest = dvc.GetVersion(datasetId, "latest");

        // Assert
        Assert.Equal(v2.VersionId, latest.VersionId);
        Assert.Equal(2, latest.VersionNumber);
    }

    [Fact]
    public void GetVersion_ByVersionNumber_ReturnsCorrectVersion()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "v1");
        var v1 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 1");

        CreateTestFile("data.csv", "v2");
        dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 2");

        // Act
        var retrieved = dvc.GetVersion(datasetId, "1");

        // Assert
        Assert.Equal(v1.VersionId, retrieved.VersionId);
        Assert.Equal(1, retrieved.VersionNumber);
    }

    [Fact]
    public void GetVersion_WithNoVersions_ThrowsInvalidOperationException()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => dvc.GetVersion(datasetId, "latest"));
    }

    [Fact]
    public void GetVersion_NonexistentVersion_ThrowsArgumentException()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "content");
        dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 1");

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => dvc.GetVersion(datasetId, "nonexistent-id"));
        Assert.Contains("Version not found", ex.Message);
    }

    #endregion

    #region ListVersions and ListDatasets Tests

    [Fact]
    public void ListVersions_ReturnsVersionsInDescendingOrder()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        for (int i = 1; i <= 5; i++)
        {
            CreateTestFile("data.csv", $"version {i}");
            dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), $"Version {i}");
        }

        // Act
        var versions = dvc.ListVersions(datasetId);

        // Assert
        Assert.Equal(5, versions.Count);
        Assert.Equal(5, versions[0].VersionNumber); // Latest first
        Assert.Equal(4, versions[1].VersionNumber);
        Assert.Equal(3, versions[2].VersionNumber);
        Assert.Equal(2, versions[3].VersionNumber);
        Assert.Equal(1, versions[4].VersionNumber);
    }

    [Fact]
    public void ListVersions_EmptyDataset_ReturnsEmptyList()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        // Act
        var versions = dvc.ListVersions(datasetId);

        // Assert
        Assert.Empty(versions);
    }

    [Fact]
    public void ListDatasets_ReturnsAllDatasets()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        dvc.CreateDataset("dataset-1", "First");
        dvc.CreateDataset("dataset-2", "Second");
        dvc.CreateDataset("dataset-3", "Third");

        // Act
        var datasets = dvc.ListDatasets();

        // Assert
        Assert.Equal(3, datasets.Count);
        Assert.Contains(datasets, d => d.Name == "dataset-1");
        Assert.Contains(datasets, d => d.Name == "dataset-2");
        Assert.Contains(datasets, d => d.Name == "dataset-3");
    }

    [Fact]
    public void ListDatasets_OrderedByLastUpdated()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var id1 = dvc.CreateDataset("dataset-1");
        // Small delay to ensure distinct timestamps (avoids timestamp-tie flakes)
        System.Threading.Thread.Sleep(10);
        var id2 = dvc.CreateDataset("dataset-2");
        System.Threading.Thread.Sleep(10);

        // Add a version to dataset-1 to update its LastUpdatedAt
        CreateTestFile("data.csv", "content");
        dvc.AddVersion(id1, Path.Combine(_sourceDir, "data.csv"), "Version");

        // Act
        var datasets = dvc.ListDatasets();

        // Assert - dataset-1 should be first (most recently updated)
        Assert.Equal("dataset-1", datasets[0].Name);
        Assert.Equal("dataset-2", datasets[1].Name);
    }

    #endregion

    #region GetDataPath Tests

    [Fact]
    public void GetDataPath_ReturnsValidPath()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "content");
        var version = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version");

        // Act
        var dataPath = dvc.GetDataPath(datasetId, version.VersionId);

        // Assert
        Assert.True(Directory.Exists(dataPath));
        Assert.True(File.Exists(Path.Combine(dataPath, "data.csv")));
    }

    [Fact]
    public void GetDataPath_Latest_ReturnsLatestVersionPath()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        CreateTestFile("data.csv", "v1");
        dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 1");

        CreateTestFile("data.csv", "v2");
        var v2 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 2");

        // Act
        var dataPath = dvc.GetDataPath(datasetId, "latest");

        // Assert
        Assert.Equal(v2.DataPath, dataPath);
    }

    [Fact]
    public void GetDataPath_PreservesDirectoryStructure()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        var subDir = Path.Combine(_sourceDir, "dataset");
        Directory.CreateDirectory(Path.Combine(subDir, "train"));
        Directory.CreateDirectory(Path.Combine(subDir, "test"));
        File.WriteAllText(Path.Combine(subDir, "train", "data.csv"), "train");
        File.WriteAllText(Path.Combine(subDir, "test", "data.csv"), "test");

        var version = dvc.AddVersion(datasetId, subDir, "Structured dataset");

        // Act
        var dataPath = dvc.GetDataPath(datasetId, version.VersionId);

        // Assert
        Assert.True(Directory.Exists(Path.Combine(dataPath, "train")));
        Assert.True(Directory.Exists(Path.Combine(dataPath, "test")));
        Assert.True(File.Exists(Path.Combine(dataPath, "train", "data.csv")));
        Assert.True(File.Exists(Path.Combine(dataPath, "test", "data.csv")));
    }

    #endregion

    #region CompareVersions Tests

    [Fact]
    public void CompareVersions_IdenticalVersions_NoChanges()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "content");
        var v1 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 1");

        // Act - Compare version with itself
        var diff = dvc.CompareVersions(datasetId, v1.VersionId, v1.VersionId);

        // Assert
        Assert.Empty(diff.FilesAdded);
        Assert.Empty(diff.FilesRemoved);
        Assert.Empty(diff.FilesModified);
        Assert.Single(diff.FilesUnchanged);
        Assert.Equal(0, diff.SizeDelta);
    }

    [Fact]
    public void CompareVersions_AddedFiles_DetectsAdditions()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        var subDir = Path.Combine(_sourceDir, "dataset");
        Directory.CreateDirectory(subDir);
        File.WriteAllText(Path.Combine(subDir, "file1.csv"), "content1");
        var v1 = dvc.AddVersion(datasetId, subDir, "Version 1");

        // Add another file
        File.WriteAllText(Path.Combine(subDir, "file2.csv"), "content2");
        var v2 = dvc.AddVersion(datasetId, subDir, "Version 2");

        // Act
        var diff = dvc.CompareVersions(datasetId, v1.VersionId, v2.VersionId);

        // Assert
        Assert.Single(diff.FilesAdded);
        Assert.Equal("file2.csv", diff.FilesAdded[0].RelativePath);
        Assert.Empty(diff.FilesRemoved);
        Assert.Empty(diff.FilesModified);
        Assert.Single(diff.FilesUnchanged);
        Assert.True(diff.SizeDelta > 0);
    }

    [Fact]
    public void CompareVersions_RemovedFiles_DetectsRemovals()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        var subDir = Path.Combine(_sourceDir, "dataset");
        Directory.CreateDirectory(subDir);
        File.WriteAllText(Path.Combine(subDir, "file1.csv"), "content1");
        File.WriteAllText(Path.Combine(subDir, "file2.csv"), "content2");
        var v1 = dvc.AddVersion(datasetId, subDir, "Version 1");

        // Remove a file
        File.Delete(Path.Combine(subDir, "file2.csv"));
        var v2 = dvc.AddVersion(datasetId, subDir, "Version 2");

        // Act
        var diff = dvc.CompareVersions(datasetId, v1.VersionId, v2.VersionId);

        // Assert
        Assert.Empty(diff.FilesAdded);
        Assert.Single(diff.FilesRemoved);
        Assert.Equal("file2.csv", diff.FilesRemoved[0].RelativePath);
        Assert.Empty(diff.FilesModified);
        Assert.Single(diff.FilesUnchanged);
        Assert.True(diff.SizeDelta < 0);
    }

    [Fact]
    public void CompareVersions_ModifiedFiles_DetectsModifications()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        var subDir = Path.Combine(_sourceDir, "dataset");
        Directory.CreateDirectory(subDir);
        File.WriteAllText(Path.Combine(subDir, "file1.csv"), "content1");
        File.WriteAllText(Path.Combine(subDir, "file2.csv"), "content2");
        var v1 = dvc.AddVersion(datasetId, subDir, "Version 1");

        // Modify a file
        File.WriteAllText(Path.Combine(subDir, "file1.csv"), "modified content");
        var v2 = dvc.AddVersion(datasetId, subDir, "Version 2");

        // Act
        var diff = dvc.CompareVersions(datasetId, v1.VersionId, v2.VersionId);

        // Assert
        Assert.Empty(diff.FilesAdded);
        Assert.Empty(diff.FilesRemoved);
        Assert.Single(diff.FilesModified);
        Assert.Equal("file1.csv", diff.FilesModified[0].before.RelativePath);
        Assert.Single(diff.FilesUnchanged);
    }

    [Fact]
    public void CompareVersions_Summary_ReturnsCorrectFormat()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        var subDir = Path.Combine(_sourceDir, "dataset");
        Directory.CreateDirectory(subDir);
        File.WriteAllText(Path.Combine(subDir, "keep.csv"), "keep");
        File.WriteAllText(Path.Combine(subDir, "remove.csv"), "remove");
        File.WriteAllText(Path.Combine(subDir, "modify.csv"), "original");
        var v1 = dvc.AddVersion(datasetId, subDir, "Version 1");

        // Make changes
        File.WriteAllText(Path.Combine(subDir, "modify.csv"), "modified");
        File.Delete(Path.Combine(subDir, "remove.csv"));
        File.WriteAllText(Path.Combine(subDir, "add.csv"), "added");
        var v2 = dvc.AddVersion(datasetId, subDir, "Version 2");

        // Act
        var diff = dvc.CompareVersions(datasetId, v1.VersionId, v2.VersionId);

        // Assert
        Assert.Equal("Added: 1, Removed: 1, Modified: 1, Unchanged: 1", diff.Summary);
    }

    #endregion

    #region DeleteVersion Tests

    [Fact]
    public void DeleteVersion_RemovesVersion()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        CreateTestFile("data.csv", "v1");
        var v1 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 1");

        CreateTestFile("data.csv", "v2");
        var v2 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 2");

        // Act
        dvc.DeleteVersion(datasetId, v1.VersionId);

        // Assert
        var versions = dvc.ListVersions(datasetId);
        Assert.Single(versions);
        Assert.Equal(v2.VersionId, versions[0].VersionId);
    }

    [Fact]
    public void DeleteVersion_UpdatesDatasetMetadata()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        CreateTestFile("data.csv", "v1");
        var v1 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 1");

        CreateTestFile("data.csv", "v2");
        var v2 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 2");

        // Act
        dvc.DeleteVersion(datasetId, v2.VersionId);

        // Assert
        var dataset = dvc.ListDatasets().First();
        Assert.Equal(1, dataset.VersionCount);
        Assert.Equal(v1.VersionId, dataset.LatestVersionId);
    }

    [Fact]
    public void DeleteVersion_RemovesFilesFromDisk()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "content");
        var version = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version");

        var versionDir = version.DataPath;
        Assert.True(Directory.Exists(versionDir));

        // Act
        dvc.DeleteVersion(datasetId, version.VersionId);

        // Assert
        Assert.False(Directory.Exists(versionDir));
    }

    [Fact]
    public void DeleteVersion_NonexistentVersion_ThrowsArgumentException()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => dvc.DeleteVersion(datasetId, "nonexistent"));
    }

    #endregion

    #region DeleteDataset Tests

    [Fact]
    public void DeleteDataset_RemovesDatasetAndAllVersions()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        CreateTestFile("data.csv", "v1");
        dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 1");

        CreateTestFile("data.csv", "v2");
        dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 2");

        // Act
        dvc.DeleteDataset(datasetId);

        // Assert
        Assert.Empty(dvc.ListDatasets());
        Assert.Throws<ArgumentException>(() => dvc.ListVersions(datasetId));
    }

    [Fact]
    public void DeleteDataset_RemovesFilesFromDisk()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "content");
        dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version");

        var datasetDir = Path.Combine(_storageDir, "datasets", datasetId);
        Assert.True(Directory.Exists(datasetDir));

        // Act
        dvc.DeleteDataset(datasetId);

        // Assert
        Assert.False(Directory.Exists(datasetDir));
    }

    [Fact]
    public void DeleteDataset_AlsoDeletesLineageRecords()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "content");
        var version = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version");

        dvc.RecordLineage(datasetId, version.VersionId, new List<(string, string)>(), "import", null);

        // Act
        dvc.DeleteDataset(datasetId);

        // Assert - Verify lineage file is deleted
        var lineageFile = Path.Combine(_storageDir, "lineage", $"{datasetId}_{version.VersionId}.json");
        Assert.False(File.Exists(lineageFile));
    }

    [Fact]
    public void DeleteDataset_NonexistentDataset_ThrowsArgumentException()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => dvc.DeleteDataset("nonexistent"));
    }

    #endregion

    #region RecordLineage and GetLineage Tests

    [Fact]
    public void RecordLineage_StoresLineageInfo()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "content");
        var version = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version");

        var parameters = new Dictionary<string, object>
        {
            ["scale"] = 1.0,
            ["normalize"] = true
        };

        // Act
        dvc.RecordLineage(datasetId, version.VersionId, new List<(string, string)>(), "initial_import", parameters);

        // Assert
        var lineage = dvc.GetLineage(datasetId, version.VersionId);
        Assert.Equal(datasetId, lineage.DatasetId);
        Assert.Equal(version.VersionId, lineage.VersionId);
        Assert.Equal("initial_import", lineage.Transformation);
        Assert.Empty(lineage.Inputs);
        Assert.Equal(1.0, Convert.ToDouble(lineage.Parameters?["scale"]));
        Assert.True(Convert.ToBoolean(lineage.Parameters?["normalize"]));
    }

    [Fact]
    public void RecordLineage_WithInputDatasets_StoresInputs()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);

        var sourceId = dvc.CreateDataset("source-dataset");
        CreateTestFile("source.csv", "source");
        var sourceVersion = dvc.AddVersion(sourceId, Path.Combine(_sourceDir, "source.csv"), "Source");

        var outputId = dvc.CreateDataset("output-dataset");
        CreateTestFile("output.csv", "output");
        var outputVersion = dvc.AddVersion(outputId, Path.Combine(_sourceDir, "output.csv"), "Output");

        var inputs = new List<(string datasetId, string versionId)>
        {
            (sourceId, sourceVersion.VersionId)
        };

        // Act
        dvc.RecordLineage(outputId, outputVersion.VersionId, inputs, "transform", null);

        // Assert
        var lineage = dvc.GetLineage(outputId, outputVersion.VersionId);
        Assert.Single(lineage.Inputs);
        Assert.Equal(sourceId, lineage.Inputs[0].datasetId);
        Assert.Equal(sourceVersion.VersionId, lineage.Inputs[0].versionId);
    }

    [Fact]
    public void GetLineage_WithRecursiveUpstream_ResolvesFullLineage()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);

        // Create a chain: raw -> processed -> final
        var rawId = dvc.CreateDataset("raw-data");
        CreateTestFile("raw.csv", "raw");
        var rawVersion = dvc.AddVersion(rawId, Path.Combine(_sourceDir, "raw.csv"), "Raw");
        dvc.RecordLineage(rawId, rawVersion.VersionId, new List<(string, string)>(), "import", null);

        var processedId = dvc.CreateDataset("processed-data");
        CreateTestFile("processed.csv", "processed");
        var processedVersion = dvc.AddVersion(processedId, Path.Combine(_sourceDir, "processed.csv"), "Processed");
        dvc.RecordLineage(processedId, processedVersion.VersionId,
            new List<(string, string)> { (rawId, rawVersion.VersionId) },
            "preprocess", null);

        var finalId = dvc.CreateDataset("final-data");
        CreateTestFile("final.csv", "final");
        var finalVersion = dvc.AddVersion(finalId, Path.Combine(_sourceDir, "final.csv"), "Final");
        dvc.RecordLineage(finalId, finalVersion.VersionId,
            new List<(string, string)> { (processedId, processedVersion.VersionId) },
            "finalize", null);

        // Act
        var lineage = dvc.GetLineage(finalId, finalVersion.VersionId);

        // Assert
        Assert.Equal("finalize", lineage.Transformation);
        Assert.Single(lineage.UpstreamLineage);

        var processedLineage = lineage.UpstreamLineage[0];
        Assert.Equal("preprocess", processedLineage.Transformation);
        Assert.Single(processedLineage.UpstreamLineage);

        var rawLineage = processedLineage.UpstreamLineage[0];
        Assert.Equal("import", rawLineage.Transformation);
        Assert.Empty(rawLineage.UpstreamLineage);
    }

    [Fact]
    public void GetLineage_NoLineageRecorded_ReturnsEmptyLineage()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "content");
        var version = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version");

        // Act
        var lineage = dvc.GetLineage(datasetId, version.VersionId);

        // Assert
        Assert.Equal(datasetId, lineage.DatasetId);
        Assert.Equal(version.VersionId, lineage.VersionId);
        Assert.Null(lineage.Transformation);
        Assert.Empty(lineage.UpstreamLineage);
    }

    [Fact]
    public void GetLineage_HandlesCircularReferences()
    {
        // Arrange - Create a scenario that could cause infinite recursion
        using var dvc = new DVC(_storageDir);

        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "content");
        var version = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version");

        // Create a self-referential lineage (shouldn't happen normally, but tests robustness)
        dvc.RecordLineage(datasetId, version.VersionId,
            new List<(string, string)> { (datasetId, version.VersionId) },
            "self-transform", null);

        // Act - Should not hang or throw due to infinite recursion
        var lineage = dvc.GetLineage(datasetId, version.VersionId);

        // Assert - Should return without infinite recursion
        Assert.Equal("self-transform", lineage.Transformation);
        Assert.Single(lineage.UpstreamLineage);
        // The cyclic reference should be detected and return an empty lineage
        Assert.Empty(lineage.UpstreamLineage[0].UpstreamLineage);
    }

    #endregion

    #region Persistence Tests

    [Fact]
    public void Persistence_DatasetsSurviveRestart()
    {
        // Arrange
        string datasetId;
        using (var dvc1 = new DVC(_storageDir))
        {
            datasetId = dvc1.CreateDataset("persistent-dataset", "Test persistence", new Dictionary<string, string>
            {
                ["key"] = "value"
            });
        }

        // Act
        using var dvc2 = new DVC(_storageDir);

        // Assert
        var datasets = dvc2.ListDatasets();
        Assert.Single(datasets);
        Assert.Equal("persistent-dataset", datasets[0].Name);
        Assert.Equal("Test persistence", datasets[0].Description);
        Assert.Equal("value", datasets[0].Metadata["key"]);
    }

    [Fact]
    public void Persistence_VersionsSurviveRestart()
    {
        // Arrange
        string datasetId;
        string versionId;
        CreateTestFile("data.csv", "content");

        using (var dvc1 = new DVC(_storageDir))
        {
            datasetId = dvc1.CreateDataset("test-dataset");
            var version = dvc1.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Test version", new Dictionary<string, string>
            {
                ["meta"] = "data"
            });
            versionId = version.VersionId;
        }

        // Act
        using var dvc2 = new DVC(_storageDir);

        // Assert
        var versions = dvc2.ListVersions(datasetId);
        Assert.Single(versions);
        Assert.Equal(versionId, versions[0].VersionId);
        Assert.Equal("Test version", versions[0].Message);
        Assert.Equal("data", versions[0].Metadata["meta"]);
    }

    [Fact]
    public void Persistence_LineageSurvivesRestart()
    {
        // Arrange
        string datasetId;
        string versionId;
        CreateTestFile("data.csv", "content");

        using (var dvc1 = new DVC(_storageDir))
        {
            datasetId = dvc1.CreateDataset("test-dataset");
            var version = dvc1.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version");
            versionId = version.VersionId;
            dvc1.RecordLineage(datasetId, versionId, new List<(string, string)>(), "import",
                new Dictionary<string, object> { ["param"] = 42 });
        }

        // Act
        using var dvc2 = new DVC(_storageDir);
        var lineage = dvc2.GetLineage(datasetId, versionId);

        // Assert
        Assert.Equal("import", lineage.Transformation);
        Assert.NotNull(lineage.Parameters);
        Assert.Equal(42L, Convert.ToInt64(lineage.Parameters["param"])); // JSON deserializes to long
    }

    #endregion

    #region Model Classes Tests

    [Fact]
    public void DatasetInfo_DefaultValues()
    {
        // Arrange & Act
        var dataset = new DatasetInfo();

        // Assert
        Assert.Equal(string.Empty, dataset.DatasetId);
        Assert.Equal(string.Empty, dataset.Name);
        Assert.Null(dataset.Description);
        Assert.Equal(0, dataset.VersionCount);
        Assert.Null(dataset.LatestVersionId);
        Assert.NotNull(dataset.Metadata);
        Assert.Empty(dataset.Metadata);
    }

    [Fact]
    public void DataVersion_SizeFormatted_Bytes()
    {
        var version = new DataVersion { SizeBytes = 500 };
        Assert.Equal("500 B", version.SizeFormatted);
    }

    [Fact]
    public void DataVersion_SizeFormatted_Kilobytes()
    {
        var version = new DataVersion { SizeBytes = 2048 };
        Assert.Equal("2.0 KB", version.SizeFormatted);
    }

    [Fact]
    public void DataVersion_SizeFormatted_Megabytes()
    {
        var version = new DataVersion { SizeBytes = 5 * 1024 * 1024 };
        Assert.Equal("5.0 MB", version.SizeFormatted);
    }

    [Fact]
    public void DataVersion_SizeFormatted_Gigabytes()
    {
        var version = new DataVersion { SizeBytes = (long)(2.5 * 1024 * 1024 * 1024) };
        Assert.Equal("2.50 GB", version.SizeFormatted);
    }

    [Fact]
    public void DataFileInfo_DefaultValues()
    {
        // Arrange & Act
        var file = new DataFileInfo();

        // Assert
        Assert.Equal(string.Empty, file.RelativePath);
        Assert.Equal(0, file.SizeBytes);
        Assert.Equal(string.Empty, file.Hash);
    }

    [Fact]
    public void DataVersionDiff_SizeDelta_Calculation()
    {
        // Arrange
        var diff = new DataVersionDiff
        {
            Version1 = new DataVersion { SizeBytes = 1000 },
            Version2 = new DataVersion { SizeBytes = 1500 }
        };

        // Assert
        Assert.Equal(500, diff.SizeDelta);
    }

    [Fact]
    public void DataLineage_DefaultValues()
    {
        // Arrange & Act
        var lineage = new DataLineage();

        // Assert
        Assert.Equal(string.Empty, lineage.DatasetId);
        Assert.Equal(string.Empty, lineage.VersionId);
        Assert.Null(lineage.Transformation);
        Assert.Null(lineage.Parameters);
        Assert.NotNull(lineage.Inputs);
        Assert.Empty(lineage.Inputs);
        Assert.NotNull(lineage.UpstreamLineage);
        Assert.Empty(lineage.UpstreamLineage);
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public void ConcurrentCreateDatasets_ThreadSafe()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var tasks = new List<Task<string>>();

        // Act - Create 10 datasets concurrently
        for (int i = 0; i < 10; i++)
        {
            var index = i;
            tasks.Add(Task.Run(() => dvc.CreateDataset($"dataset-{index}")));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        var datasets = dvc.ListDatasets();
        Assert.Equal(10, datasets.Count);
    }

    [Fact]
    public void ConcurrentAddVersions_ThreadSafe()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        var tasks = new List<Task<DataVersion>>();

        // Create multiple source files
        for (int i = 0; i < 5; i++)
        {
            CreateTestFile($"data_{i}.csv", $"content {i}");
        }

        // Act - Add 5 versions concurrently (each with different content)
        for (int i = 0; i < 5; i++)
        {
            var index = i;
            tasks.Add(Task.Run(() =>
                dvc.AddVersion(datasetId, Path.Combine(_sourceDir, $"data_{index}.csv"), $"Version {index}")));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        var versions = dvc.ListVersions(datasetId);
        Assert.Equal(5, versions.Count);
    }

    [Fact]
    public void ConcurrentReadAndWrite_ThreadSafe()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");
        CreateTestFile("data.csv", "initial");
        dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Initial");

        var tasks = new List<Task>();

        // Act - Mix reads and writes
        for (int i = 0; i < 10; i++)
        {
            var index = i;
            if (i % 2 == 0)
            {
                // Read
                tasks.Add(Task.Run(() =>
                {
                    var versions = dvc.ListVersions(datasetId);
                    var latest = dvc.GetVersion(datasetId, "latest");
                }));
            }
            else
            {
                // Write
                CreateTestFile($"data_{i}.csv", $"content {index}");
                tasks.Add(Task.Run(() =>
                    dvc.AddVersion(datasetId, Path.Combine(_sourceDir, $"data_{index}.csv"), $"Version {index}")));
            }
        }

        // Should complete without deadlock or exceptions
        var completed = Task.WaitAll(tasks.ToArray(), TimeSpan.FromSeconds(30));

        // Assert
        Assert.True(completed, "Tasks should complete within timeout");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void AddVersion_LargeNumberOfFiles_HandledCorrectly()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        var subDir = Path.Combine(_sourceDir, "large-dataset");
        Directory.CreateDirectory(subDir);

        // Create 100 files
        for (int i = 0; i < 100; i++)
        {
            File.WriteAllText(Path.Combine(subDir, $"file_{i:D3}.csv"), $"content {i}");
        }

        // Act
        var version = dvc.AddVersion(datasetId, subDir, "Large dataset");

        // Assert
        Assert.Equal(100, version.FileCount);
        Assert.Equal(100, version.Files.Count);
    }

    [Fact]
    public void AddVersion_EmptyDirectory_CreatesVersionWithNoFiles()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        var emptyDir = Path.Combine(_sourceDir, "empty");
        Directory.CreateDirectory(emptyDir);

        // Act
        var version = dvc.AddVersion(datasetId, emptyDir, "Empty directory");

        // Assert
        Assert.Equal(0, version.FileCount);
        Assert.Empty(version.Files);
    }

    [Fact]
    public void AddVersion_SpecialCharactersInFilename_HandledCorrectly()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        // Create files with special characters (that are valid on Windows)
        CreateTestFile("data with spaces.csv", "content");
        CreateTestFile("data-with-dashes.csv", "content");
        CreateTestFile("data_with_underscores.csv", "content");

        var subDir = _sourceDir;

        // Act
        var version = dvc.AddVersion(datasetId, subDir, "Special chars");

        // Assert
        Assert.True(version.FileCount >= 3);
        Assert.Contains(version.Files, f => f.RelativePath.Contains("spaces"));
        Assert.Contains(version.Files, f => f.RelativePath.Contains("dashes"));
        Assert.Contains(version.Files, f => f.RelativePath.Contains("underscores"));
    }

    [Fact]
    public void ContentHash_Deterministic_SameContentSameHash()
    {
        // Arrange
        using var dvc = new DVC(_storageDir);
        var datasetId = dvc.CreateDataset("test-dataset");

        CreateTestFile("data.csv", "fixed content for hashing");
        var v1 = dvc.AddVersion(datasetId, Path.Combine(_sourceDir, "data.csv"), "Version 1");

        // Create another dataset with identical content
        var datasetId2 = dvc.CreateDataset("test-dataset-2");
        var v2 = dvc.AddVersion(datasetId2, Path.Combine(_sourceDir, "data.csv"), "Version 1 copy");

        // Assert - Same content should have same hash
        Assert.Equal(v1.ContentHash, v2.ContentHash);
        Assert.Equal(v1.Files[0].Hash, v2.Files[0].Hash);
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var dvc = new DVC(_storageDir);

        // Act & Assert - Should not throw
        dvc.Dispose();
        dvc.Dispose();
        dvc.Dispose();
    }

    #endregion

    #region Helper Methods

    private void CreateTestFile(string fileName, string content)
    {
        var filePath = Path.Combine(_sourceDir, fileName);
        var dir = Path.GetDirectoryName(filePath);
        if (dir != null && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }
        File.WriteAllText(filePath, content);
    }

    #endregion
}
