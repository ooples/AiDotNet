using AiDotNet.DataVersionControl;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DataVersionControl;

/// <summary>
/// Integration tests for the DataVersionControl module.
/// Tests the full implementation of dataset versioning, integrity verification,
/// lineage tracking, and multi-dataset snapshots.
/// </summary>
public class DataVersionControlIntegrationTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly string _storageDirectory;
    private readonly string _testDataDirectory;

    public DataVersionControlIntegrationTests()
    {
        // Create unique test directories for each test
        _testDirectory = Path.Combine(Path.GetTempPath(), $"dvc_test_{Guid.NewGuid():N}");
        _storageDirectory = Path.Combine(_testDirectory, "storage");
        _testDataDirectory = Path.Combine(_testDirectory, "data");

        Directory.CreateDirectory(_testDirectory);
        Directory.CreateDirectory(_storageDirectory);
        Directory.CreateDirectory(_testDataDirectory);
    }

    public void Dispose()
    {
        // Clean up test directories
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

    private string CreateTestFile(string filename, string content)
    {
        var path = Path.Combine(_testDataDirectory, filename);
        File.WriteAllText(path, content);
        return path;
    }

    private string CreateTestDirectory(string dirname)
    {
        var path = Path.Combine(_testDataDirectory, dirname);
        Directory.CreateDirectory(path);
        return path;
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithExplicitStorageDirectory_CreatesInstance()
    {
        // Test basic constructor with explicit storage directory
        var dvc = new DataVersionControl<double>(_storageDirectory);
        Assert.NotNull(dvc);
    }

    [Fact]
    public void Constructor_WithCustomStorageDirectory_CreatesDirectory()
    {
        var customStorage = Path.Combine(_testDirectory, "custom_storage");
        Assert.False(Directory.Exists(customStorage));

        var dvc = new DataVersionControl<double>(customStorage);

        Assert.True(Directory.Exists(customStorage));
    }

    #endregion

    #region CreateDatasetVersion Tests

    [Fact]
    public void CreateDatasetVersion_WithValidFile_ReturnsHash()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("test_data.csv", "col1,col2\n1,2\n3,4");

        var hash = dvc.CreateDatasetVersion("test_dataset", dataPath, "Initial version");

        Assert.NotNull(hash);
        Assert.NotEmpty(hash);
    }

    [Fact]
    public void CreateDatasetVersion_WithDescription_StoresDescription()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("test_data.csv", "col1,col2\n1,2");

        var hash = dvc.CreateDatasetVersion("test_dataset", dataPath, "My description");
        var version = dvc.GetDatasetVersion("test_dataset", hash);

        Assert.Equal("My description", version.Description);
    }

    [Fact]
    public void CreateDatasetVersion_MultipleVersions_IncrementsVersionNumber()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("data1.csv", "col1\n1\n2");
        var dataPath2 = CreateTestFile("data2.csv", "col1\n1\n2\n3");
        var dataPath3 = CreateTestFile("data3.csv", "col1\n1\n2\n3\n4");

        var hash1 = dvc.CreateDatasetVersion("test_dataset", dataPath1, "v1");
        var hash2 = dvc.CreateDatasetVersion("test_dataset", dataPath2, "v2");
        var hash3 = dvc.CreateDatasetVersion("test_dataset", dataPath3, "v3");

        var version1 = dvc.GetDatasetVersion("test_dataset", hash1);
        var version2 = dvc.GetDatasetVersion("test_dataset", hash2);
        var version3 = dvc.GetDatasetVersion("test_dataset", hash3);

        Assert.Equal(1, version1.Version);
        Assert.Equal(2, version2.Version);
        Assert.Equal(3, version3.Version);
    }

    [Fact]
    public void CreateDatasetVersion_WithTags_StoresTags()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("tagged_data.csv", "data");
        var tags = new Dictionary<string, string>
        {
            { "environment", "production" },
            { "source", "api" }
        };

        var hash = dvc.CreateDatasetVersion("tagged_dataset", dataPath, tags: tags);
        var version = dvc.GetDatasetVersion("tagged_dataset", hash);

        Assert.Equal("production", version.Tags["environment"]);
        Assert.Equal("api", version.Tags["source"]);
    }

    [Fact]
    public void CreateDatasetVersion_WithNullDatasetName_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("test.csv", "data");

        Assert.Throws<ArgumentException>(() =>
            dvc.CreateDatasetVersion(null!, dataPath));
    }

    [Fact]
    public void CreateDatasetVersion_WithEmptyDataPath_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);

        Assert.Throws<ArgumentException>(() =>
            dvc.CreateDatasetVersion("test", ""));
    }

    [Fact]
    public void CreateDatasetVersion_WithDirectory_ComputesHashCorrectly()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dirPath = CreateTestDirectory("multi_file_data");
        File.WriteAllText(Path.Combine(dirPath, "file1.txt"), "content1");
        File.WriteAllText(Path.Combine(dirPath, "file2.txt"), "content2");

        var hash = dvc.CreateDatasetVersion("dir_dataset", dirPath);

        Assert.NotNull(hash);
        Assert.NotEmpty(hash);
    }

    #endregion

    #region GetDatasetVersion Tests

    [Fact]
    public void GetDatasetVersion_WithoutVersionHash_ReturnsLatest()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("data1.csv", "old");
        var dataPath2 = CreateTestFile("data2.csv", "newer");

        dvc.CreateDatasetVersion("test_dataset", dataPath1, "v1");
        dvc.CreateDatasetVersion("test_dataset", dataPath2, "v2");

        var latest = dvc.GetDatasetVersion("test_dataset");

        Assert.Equal(2, latest.Version);
        Assert.Equal("v2", latest.Description);
    }

    [Fact]
    public void GetDatasetVersion_WithSpecificHash_ReturnsCorrectVersion()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("data1.csv", "version1");
        var dataPath2 = CreateTestFile("data2.csv", "version2");

        var hash1 = dvc.CreateDatasetVersion("test_dataset", dataPath1, "First");
        dvc.CreateDatasetVersion("test_dataset", dataPath2, "Second");

        var version = dvc.GetDatasetVersion("test_dataset", hash1);

        Assert.Equal(1, version.Version);
        Assert.Equal("First", version.Description);
    }

    [Fact]
    public void GetDatasetVersion_WithNonexistentDataset_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);

        Assert.Throws<ArgumentException>(() =>
            dvc.GetDatasetVersion("nonexistent"));
    }

    [Fact]
    public void GetDatasetVersion_WithNonexistentHash_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("test.csv", "data");
        dvc.CreateDatasetVersion("test_dataset", dataPath);

        Assert.Throws<ArgumentException>(() =>
            dvc.GetDatasetVersion("test_dataset", "invalid_hash"));
    }

    #endregion

    #region GetLatestDatasetVersion Tests

    [Fact]
    public void GetLatestDatasetVersion_ReturnsHighestVersionNumber()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("data1.csv", "v1");
        var dataPath2 = CreateTestFile("data2.csv", "v2");
        var dataPath3 = CreateTestFile("data3.csv", "v3");

        dvc.CreateDatasetVersion("dataset", dataPath1, "First");
        dvc.CreateDatasetVersion("dataset", dataPath2, "Second");
        dvc.CreateDatasetVersion("dataset", dataPath3, "Third");

        var latest = dvc.GetLatestDatasetVersion("dataset");

        Assert.Equal(3, latest.Version);
        Assert.Equal("Third", latest.Description);
    }

    #endregion

    #region ListDatasetVersions Tests

    [Fact]
    public void ListDatasetVersions_ReturnsAllVersionsDescending()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("d1.csv", "a");
        var dataPath2 = CreateTestFile("d2.csv", "b");
        var dataPath3 = CreateTestFile("d3.csv", "c");

        dvc.CreateDatasetVersion("dataset", dataPath1);
        dvc.CreateDatasetVersion("dataset", dataPath2);
        dvc.CreateDatasetVersion("dataset", dataPath3);

        var versions = dvc.ListDatasetVersions("dataset");

        Assert.Equal(3, versions.Count);
        Assert.Equal(3, versions[0].Version); // Most recent first
        Assert.Equal(2, versions[1].Version);
        Assert.Equal(1, versions[2].Version);
    }

    [Fact]
    public void ListDatasetVersions_IncludesVersionInfo()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "col1,col2\n1,2\n3,4");

        var hash = dvc.CreateDatasetVersion("dataset", dataPath, "Test description");
        var versions = dvc.ListDatasetVersions("dataset");

        var versionInfo = versions.First();
        Assert.Equal(hash, versionInfo.Hash);
        Assert.Equal("Test description", versionInfo.Description);
        Assert.True(versionInfo.SizeBytes > 0);
    }

    #endregion

    #region ListDatasets Tests

    [Fact]
    public void ListDatasets_ReturnsAllDatasetNames()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("d1.csv", "a");
        var dataPath2 = CreateTestFile("d2.csv", "b");
        var dataPath3 = CreateTestFile("d3.csv", "c");

        dvc.CreateDatasetVersion("dataset_alpha", dataPath1);
        dvc.CreateDatasetVersion("dataset_beta", dataPath2);
        dvc.CreateDatasetVersion("dataset_gamma", dataPath3);

        var datasets = dvc.ListDatasets();

        Assert.Equal(3, datasets.Count);
        Assert.Contains("dataset_alpha", datasets);
        Assert.Contains("dataset_beta", datasets);
        Assert.Contains("dataset_gamma", datasets);
    }

    [Fact]
    public void ListDatasets_WithFilter_ReturnsMatchingDatasets()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("d1.csv", "a");
        var dataPath2 = CreateTestFile("d2.csv", "b");
        var dataPath3 = CreateTestFile("d3.csv", "c");

        dvc.CreateDatasetVersion("training_data", dataPath1);
        dvc.CreateDatasetVersion("test_data", dataPath2);
        dvc.CreateDatasetVersion("validation_data", dataPath3);

        var datasets = dvc.ListDatasets(filter: "data");

        Assert.Equal(3, datasets.Count);

        var trainingDatasets = dvc.ListDatasets(filter: "training");
        Assert.Single(trainingDatasets);
        Assert.Equal("training_data", trainingDatasets[0]);
    }

    [Fact]
    public void ListDatasets_WithTagFilter_ReturnsMatchingDatasets()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("d1.csv", "a");
        var dataPath2 = CreateTestFile("d2.csv", "b");

        dvc.CreateDatasetVersion("prod_data", dataPath1, tags: new Dictionary<string, string> { { "env", "production" } });
        dvc.CreateDatasetVersion("dev_data", dataPath2, tags: new Dictionary<string, string> { { "env", "development" } });

        var prodDatasets = dvc.ListDatasets(tags: new Dictionary<string, string> { { "env", "production" } });

        Assert.Single(prodDatasets);
        Assert.Equal("prod_data", prodDatasets[0]);
    }

    #endregion

    #region ComputeDatasetHash Tests

    [Fact]
    public void ComputeDatasetHash_SameContent_ReturnsSameHash()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var content = "identical content";
        var path1 = CreateTestFile("file1.txt", content);
        var path2 = CreateTestFile("file2.txt", content);

        var hash1 = dvc.ComputeDatasetHash(path1);
        var hash2 = dvc.ComputeDatasetHash(path2);

        Assert.Equal(hash1, hash2);
    }

    [Fact]
    public void ComputeDatasetHash_DifferentContent_ReturnsDifferentHash()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var path1 = CreateTestFile("file1.txt", "content1");
        var path2 = CreateTestFile("file2.txt", "content2");

        var hash1 = dvc.ComputeDatasetHash(path1);
        var hash2 = dvc.ComputeDatasetHash(path2);

        Assert.NotEqual(hash1, hash2);
    }

    [Fact]
    public void ComputeDatasetHash_Directory_ComputesHashOfAllFiles()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dir = CreateTestDirectory("hash_test_dir");
        File.WriteAllText(Path.Combine(dir, "a.txt"), "file a");
        File.WriteAllText(Path.Combine(dir, "b.txt"), "file b");

        var hash = dvc.ComputeDatasetHash(dir);

        Assert.NotNull(hash);
        Assert.NotEmpty(hash);
    }

    [Fact]
    public void ComputeDatasetHash_NullPath_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);

        Assert.Throws<ArgumentException>(() =>
            dvc.ComputeDatasetHash(null!));
    }

    [Fact]
    public void ComputeDatasetHash_NonexistentPath_ThrowsFileNotFoundException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);

        Assert.Throws<FileNotFoundException>(() =>
            dvc.ComputeDatasetHash(Path.Combine(_testDirectory, "nonexistent.txt")));
    }

    #endregion

    #region VerifyDatasetIntegrity Tests

    [Fact]
    public void VerifyDatasetIntegrity_UnchangedFile_ReturnsTrue()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "original content");

        var hash = dvc.CreateDatasetVersion("dataset", dataPath);
        var isValid = dvc.VerifyDatasetIntegrity("dataset", hash, dataPath);

        Assert.True(isValid);
    }

    [Fact]
    public void VerifyDatasetIntegrity_ModifiedFile_ReturnsFalse()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "original content");

        var hash = dvc.CreateDatasetVersion("dataset", dataPath);

        // Modify the file
        File.WriteAllText(dataPath, "modified content");

        var isValid = dvc.VerifyDatasetIntegrity("dataset", hash, dataPath);

        Assert.False(isValid);
    }

    #endregion

    #region LinkDatasetToRun Tests

    [Fact]
    public void LinkDatasetToRun_CreatesLink()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "training data");
        var hash = dvc.CreateDatasetVersion("dataset", dataPath);

        dvc.LinkDatasetToRun("dataset", hash, "run_001", "model_v1");

        var runs = dvc.GetRunsUsingDataset("dataset", hash);
        Assert.Contains("run_001", runs);
    }

    [Fact]
    public void LinkDatasetToRun_WithInvalidRunId_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "data");
        var hash = dvc.CreateDatasetVersion("dataset", dataPath);

        Assert.Throws<ArgumentException>(() =>
            dvc.LinkDatasetToRun("dataset", hash, ""));
    }

    [Fact]
    public void GetDatasetForRun_ReturnsLinkedDataset()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "training data");
        var hash = dvc.CreateDatasetVersion("dataset", dataPath, "Training dataset");

        dvc.LinkDatasetToRun("dataset", hash, "run_001");

        var dataset = dvc.GetDatasetForRun("run_001");
        Assert.Equal("Training dataset", dataset.Description);
        Assert.Equal(hash, dataset.Hash);
    }

    [Fact]
    public void GetDatasetForRun_WithUnlinkedRun_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);

        Assert.Throws<ArgumentException>(() =>
            dvc.GetDatasetForRun("nonexistent_run"));
    }

    #endregion

    #region TagDatasetVersion Tests

    [Fact]
    public void TagDatasetVersion_CreatesTag()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "production data");
        var hash = dvc.CreateDatasetVersion("dataset", dataPath);

        dvc.TagDatasetVersion("dataset", hash, "production");

        var taggedVersion = dvc.GetDatasetByTag("dataset", "production");
        Assert.Equal(hash, taggedVersion.Hash);
    }

    [Fact]
    public void TagDatasetVersion_MultipleTagsOnSameVersion_AllTagsWork()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "data");
        var hash = dvc.CreateDatasetVersion("dataset", dataPath);

        dvc.TagDatasetVersion("dataset", hash, "latest");
        dvc.TagDatasetVersion("dataset", hash, "stable");
        dvc.TagDatasetVersion("dataset", hash, "v1.0");

        var v1 = dvc.GetDatasetByTag("dataset", "latest");
        var v2 = dvc.GetDatasetByTag("dataset", "stable");
        var v3 = dvc.GetDatasetByTag("dataset", "v1.0");

        Assert.Equal(hash, v1.Hash);
        Assert.Equal(hash, v2.Hash);
        Assert.Equal(hash, v3.Hash);
    }

    [Fact]
    public void TagDatasetVersion_WithEmptyTag_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "data");
        var hash = dvc.CreateDatasetVersion("dataset", dataPath);

        Assert.Throws<ArgumentException>(() =>
            dvc.TagDatasetVersion("dataset", hash, ""));
    }

    [Fact]
    public void GetDatasetByTag_WithNonexistentTag_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "data");
        dvc.CreateDatasetVersion("dataset", dataPath);

        Assert.Throws<ArgumentException>(() =>
            dvc.GetDatasetByTag("dataset", "nonexistent_tag"));
    }

    #endregion

    #region CompareDatasetVersions Tests

    [Fact]
    public void CompareDatasetVersions_IdenticalVersions_NoChanges()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "data");
        var hash = dvc.CreateDatasetVersion("dataset", dataPath);

        var comparison = dvc.CompareDatasetVersions("dataset", hash, hash);

        Assert.Equal(0, comparison.RecordsAdded);
        Assert.Equal(0, comparison.RecordsRemoved);
    }

    [Fact]
    public void CompareDatasetVersions_DifferentSizes_DetectsSchemaChange()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var smallPath = CreateTestFile("small.csv", "a");
        var largePath = CreateTestFile("large.csv", "a,b,c,d,e,f,g,h,i,j");

        var hash1 = dvc.CreateDatasetVersion("dataset", smallPath);
        var hash2 = dvc.CreateDatasetVersion("dataset", largePath);

        var comparison = dvc.CompareDatasetVersions("dataset", hash1, hash2);

        // Should detect size change
        Assert.NotEmpty(comparison.SchemaChanges);
    }

    [Fact]
    public void CompareDatasetVersions_DifferentHashes_DetectsModification()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var path1 = CreateTestFile("data1.csv", "content1");
        var path2 = CreateTestFile("data2.csv", "content2");

        var hash1 = dvc.CreateDatasetVersion("dataset", path1);
        var hash2 = dvc.CreateDatasetVersion("dataset", path2);

        var comparison = dvc.CompareDatasetVersions("dataset", hash1, hash2);

        Assert.True(comparison.RecordsModified > 0 || comparison.SchemaChanges.Count > 0);
    }

    #endregion

    #region RecordDatasetLineage Tests

    [Fact]
    public void RecordDatasetLineage_StoresLineageInfo()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "derived data");
        var hash = dvc.CreateDatasetVersion("derived_dataset", dataPath);

        var lineage = new DatasetLineage
        {
            DatasetName = "derived_dataset",
            Version = 1,
            SourceDataset = "original_dataset",
            SourceVersion = 1,
            Transformations = new List<string> { "Normalize", "Filter" },
            Creator = "test_user"
        };

        dvc.RecordDatasetLineage("derived_dataset", hash, lineage);

        var retrieved = dvc.GetDatasetLineage("derived_dataset", hash);
        Assert.Equal("original_dataset", retrieved.SourceDataset);
        Assert.Equal(1, retrieved.SourceVersion);
        Assert.Contains("Normalize", retrieved.Transformations);
        Assert.Contains("Filter", retrieved.Transformations);
        Assert.Equal("test_user", retrieved.Creator);
    }

    [Fact]
    public void GetDatasetLineage_WithNoRecordedLineage_ReturnsDefaultLineage()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "data");
        var hash = dvc.CreateDatasetVersion("dataset", dataPath);

        var lineage = dvc.GetDatasetLineage("dataset", hash);

        Assert.Equal("dataset", lineage.DatasetName);
        Assert.Equal(1, lineage.Version);
        Assert.Null(lineage.SourceDataset);
    }

    [Fact]
    public void RecordDatasetLineage_WithNullLineage_ThrowsArgumentNullException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "data");
        var hash = dvc.CreateDatasetVersion("dataset", dataPath);

        Assert.Throws<ArgumentNullException>(() =>
            dvc.RecordDatasetLineage("dataset", hash, null!));
    }

    #endregion

    #region DeleteDatasetVersion Tests

    [Fact]
    public void DeleteDatasetVersion_RemovesVersion()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("d1.csv", "a");
        var dataPath2 = CreateTestFile("d2.csv", "b");

        var hash1 = dvc.CreateDatasetVersion("dataset", dataPath1);
        var hash2 = dvc.CreateDatasetVersion("dataset", dataPath2);

        dvc.DeleteDatasetVersion("dataset", hash1);

        var versions = dvc.ListDatasetVersions("dataset");
        Assert.Single(versions);
        Assert.Equal(hash2, versions[0].Hash);
    }

    [Fact]
    public void DeleteDatasetVersion_RemovesAssociatedTags()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("d1.csv", "a");
        var dataPath2 = CreateTestFile("d2.csv", "b");

        var hash1 = dvc.CreateDatasetVersion("dataset", dataPath1);
        dvc.CreateDatasetVersion("dataset", dataPath2);

        dvc.TagDatasetVersion("dataset", hash1, "old_tag");
        dvc.DeleteDatasetVersion("dataset", hash1);

        Assert.Throws<ArgumentException>(() =>
            dvc.GetDatasetByTag("dataset", "old_tag"));
    }

    [Fact]
    public void DeleteDatasetVersion_LastVersion_RemovesDataset()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "data");

        var hash = dvc.CreateDatasetVersion("dataset", dataPath);
        dvc.DeleteDatasetVersion("dataset", hash);

        var datasets = dvc.ListDatasets();
        Assert.DoesNotContain("dataset", datasets);
    }

    [Fact]
    public void DeleteDatasetVersion_NonexistentVersion_DoesNotThrow()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "data");
        dvc.CreateDatasetVersion("dataset", dataPath);

        // Should not throw
        dvc.DeleteDatasetVersion("dataset", "nonexistent_hash");
    }

    #endregion

    #region GetDatasetStatistics Tests

    [Fact]
    public void GetDatasetStatistics_ReturnsBasicStats()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "col1,col2\n1,2\n3,4\n5,6");
        var hash = dvc.CreateDatasetVersion("dataset", dataPath);

        var stats = dvc.GetDatasetStatistics("dataset", hash);

        Assert.NotNull(stats);
        Assert.NotNull(stats.MissingValues);
        Assert.NotNull(stats.NumericStats);
        Assert.NotNull(stats.CategoricalStats);
    }

    #endregion

    #region Snapshot Tests

    [Fact]
    public void CreateDatasetSnapshot_ReturnsSnapshotId()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("train.csv", "training");
        var dataPath2 = CreateTestFile("test.csv", "testing");

        var hash1 = dvc.CreateDatasetVersion("training", dataPath1);
        var hash2 = dvc.CreateDatasetVersion("testing", dataPath2);

        var snapshotId = dvc.CreateDatasetSnapshot("experiment_v1", new Dictionary<string, string>
        {
            { "training", hash1 },
            { "testing", hash2 }
        }, "Initial experiment snapshot");

        Assert.NotNull(snapshotId);
        Assert.NotEmpty(snapshotId);
    }

    [Fact]
    public void GetDatasetSnapshot_ReturnsSnapshotMetadata()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("train.csv", "training");
        var dataPath2 = CreateTestFile("test.csv", "testing");

        var hash1 = dvc.CreateDatasetVersion("training", dataPath1);
        var hash2 = dvc.CreateDatasetVersion("testing", dataPath2);

        var snapshotId = dvc.CreateDatasetSnapshot("experiment_v1", new Dictionary<string, string>
        {
            { "training", hash1 },
            { "testing", hash2 }
        }, "My experiment");

        var snapshot = dvc.GetDatasetSnapshot("experiment_v1");

        Assert.Equal(snapshotId, snapshot.SnapshotId);
        Assert.Equal("experiment_v1", snapshot.DatasetName);
        Assert.Equal("My experiment", snapshot.Description);
    }

    [Fact]
    public void GetAllDatasetsInSnapshot_ReturnsAllDatasets()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath1 = CreateTestFile("train.csv", "training");
        var dataPath2 = CreateTestFile("test.csv", "testing");
        var dataPath3 = CreateTestFile("val.csv", "validation");

        var hash1 = dvc.CreateDatasetVersion("training", dataPath1);
        var hash2 = dvc.CreateDatasetVersion("testing", dataPath2);
        var hash3 = dvc.CreateDatasetVersion("validation", dataPath3);

        var snapshotId = dvc.CreateDatasetSnapshot("full_experiment", new Dictionary<string, string>
        {
            { "training", hash1 },
            { "testing", hash2 },
            { "validation", hash3 }
        });

        var (id, datasets, description, createdAt) = dvc.GetAllDatasetsInSnapshot("full_experiment");

        Assert.Equal(snapshotId, id);
        Assert.Equal(3, datasets.Count);
        Assert.Equal(hash1, datasets["training"]);
        Assert.Equal(hash2, datasets["testing"]);
        Assert.Equal(hash3, datasets["validation"]);
    }

    [Fact]
    public void CreateDatasetSnapshot_WithEmptyDatasets_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);

        Assert.Throws<ArgumentException>(() =>
            dvc.CreateDatasetSnapshot("empty_snapshot", new Dictionary<string, string>()));
    }

    [Fact]
    public void CreateDatasetSnapshot_WithNullName_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var dataPath = CreateTestFile("data.csv", "data");
        var hash = dvc.CreateDatasetVersion("dataset", dataPath);

        Assert.Throws<ArgumentException>(() =>
            dvc.CreateDatasetSnapshot(null!, new Dictionary<string, string> { { "dataset", hash } }));
    }

    [Fact]
    public void GetDatasetSnapshot_WithNonexistentSnapshot_ThrowsArgumentException()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);

        Assert.Throws<ArgumentException>(() =>
            dvc.GetDatasetSnapshot("nonexistent"));
    }

    #endregion

    #region Persistence Tests

    [Fact]
    public void DataVersionControl_PersistsAcrossInstances()
    {
        var dataPath = CreateTestFile("persist_test.csv", "persistent data");

        // Create version in first instance
        var dvc1 = new DataVersionControl<double>(_storageDirectory);
        var hash = dvc1.CreateDatasetVersion("persistent_dataset", dataPath, "Persisted version");
        dvc1.TagDatasetVersion("persistent_dataset", hash, "persisted_tag");

        // Verify in new instance
        var dvc2 = new DataVersionControl<double>(_storageDirectory);
        var versions = dvc2.ListDatasetVersions("persistent_dataset");
        Assert.Single(versions);
        Assert.Equal(hash, versions[0].Hash);
        Assert.Equal("Persisted version", versions[0].Description);

        var tagged = dvc2.GetDatasetByTag("persistent_dataset", "persisted_tag");
        Assert.Equal(hash, tagged.Hash);
    }

    [Fact]
    public void DataVersionControl_PersistsRunLinks()
    {
        var dataPath = CreateTestFile("run_link_test.csv", "data");

        // Create version and link in first instance
        var dvc1 = new DataVersionControl<double>(_storageDirectory);
        var hash = dvc1.CreateDatasetVersion("dataset", dataPath);
        dvc1.LinkDatasetToRun("dataset", hash, "persisted_run_001");

        // Verify in new instance
        var dvc2 = new DataVersionControl<double>(_storageDirectory);
        var runs = dvc2.GetRunsUsingDataset("dataset", hash);
        Assert.Contains("persisted_run_001", runs);

        var dataset = dvc2.GetDatasetForRun("persisted_run_001");
        Assert.Equal(hash, dataset.Hash);
    }

    [Fact]
    public void DataVersionControl_PersistsSnapshots()
    {
        var dataPath1 = CreateTestFile("snap1.csv", "data1");
        var dataPath2 = CreateTestFile("snap2.csv", "data2");

        // Create snapshot in first instance
        var dvc1 = new DataVersionControl<double>(_storageDirectory);
        var hash1 = dvc1.CreateDatasetVersion("dataset1", dataPath1);
        var hash2 = dvc1.CreateDatasetVersion("dataset2", dataPath2);

        var snapshotId = dvc1.CreateDatasetSnapshot("persisted_snapshot", new Dictionary<string, string>
        {
            { "dataset1", hash1 },
            { "dataset2", hash2 }
        }, "Persisted snapshot description");

        // Verify in new instance
        var dvc2 = new DataVersionControl<double>(_storageDirectory);
        var snapshot = dvc2.GetDatasetSnapshot("persisted_snapshot");
        Assert.Equal(snapshotId, snapshot.SnapshotId);
        Assert.Equal("Persisted snapshot description", snapshot.Description);

        var (_, datasets, _, _) = dvc2.GetAllDatasetsInSnapshot("persisted_snapshot");
        Assert.Equal(2, datasets.Count);
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public void DataVersionControl_ConcurrentVersionCreation_IsThreadSafe()
    {
        var dvc = new DataVersionControl<double>(_storageDirectory);
        var tasks = new List<Task<string>>();

        // Create 10 concurrent version creations
        for (int i = 0; i < 10; i++)
        {
            var index = i;
            var dataPath = CreateTestFile($"concurrent_{index}.csv", $"data_{index}");
            tasks.Add(Task.Run(() => dvc.CreateDatasetVersion("concurrent_dataset", dataPath, $"Version {index}")));
        }

        Task.WaitAll(tasks.ToArray());

        var versions = dvc.ListDatasetVersions("concurrent_dataset");
        Assert.Equal(10, versions.Count);

        // Verify version numbers are unique and sequential
        var versionNumbers = versions.Select(v => v.Version).OrderBy(v => v).ToList();
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(i + 1, versionNumbers[i]);
        }
    }

    #endregion

    #region Model Classes Tests

    [Fact]
    public void DatasetVersion_HasCorrectDefaults()
    {
        var version = new DatasetVersion<double>();

        Assert.NotNull(version.VersionId);
        Assert.NotEmpty(version.VersionId);
        Assert.Equal(string.Empty, version.DatasetName);
        Assert.Equal(0, version.Version);
        Assert.NotNull(version.Tags);
    }

    [Fact]
    public void DatasetLineage_HasCorrectDefaults()
    {
        var lineage = new DatasetLineage();

        Assert.Equal(string.Empty, lineage.DatasetName);
        Assert.Equal(0, lineage.Version);
        Assert.Null(lineage.SourceDataset);
        Assert.Null(lineage.SourceVersion);
        Assert.NotNull(lineage.Transformations);
        Assert.NotNull(lineage.UsedInRuns);
    }

    [Fact]
    public void DatasetComparison_HasCorrectDefaults()
    {
        var comparison = new DatasetComparison<double>();

        Assert.Equal(0, comparison.Version1);
        Assert.Equal(0, comparison.Version2);
        Assert.Equal(0, comparison.RecordsAdded);
        Assert.Equal(0, comparison.RecordsRemoved);
        Assert.Equal(0, comparison.RecordsModified);
        Assert.NotNull(comparison.SchemaChanges);
        Assert.NotNull(comparison.StatisticalChanges);
    }

    [Fact]
    public void DatasetStatistics_HasCorrectDefaults()
    {
        var stats = new DatasetStatistics<double>();

        Assert.Equal(0, stats.RecordCount);
        Assert.Equal(0, stats.ColumnCount);
        Assert.NotNull(stats.MissingValues);
        Assert.NotNull(stats.NumericStats);
        Assert.NotNull(stats.CategoricalStats);
    }

    [Fact]
    public void NumericColumnStats_SupportsNullableValues()
    {
        var stats = new NumericColumnStats<double>();

        // For unconstrained generic T?, when T is a value type like double,
        // the compiler treats T? as just T, so default is 0 not null
        Assert.Equal(0.0, stats.Min);
        Assert.Equal(0.0, stats.Max);
        Assert.Equal(0.0, stats.Mean);
        Assert.Equal(0.0, stats.StdDev);
        Assert.Equal(0.0, stats.Median);

        // Verify values can be set
        stats.Min = 10.0;
        stats.Max = 100.0;
        stats.Mean = 50.0;
        stats.StdDev = 15.0;
        stats.Median = 48.0;

        Assert.Equal(10.0, stats.Min);
        Assert.Equal(100.0, stats.Max);
        Assert.Equal(50.0, stats.Mean);
        Assert.Equal(15.0, stats.StdDev);
        Assert.Equal(48.0, stats.Median);
    }

    [Fact]
    public void CategoricalColumnStats_HasCorrectDefaults()
    {
        var stats = new CategoricalColumnStats();

        Assert.Equal(0, stats.UniqueCount);
        Assert.Null(stats.MostFrequent);
        Assert.Equal(0, stats.MostFrequentCount);
        Assert.NotNull(stats.ValueCounts);
    }

    [Fact]
    public void DatasetSnapshot_HasCorrectDefaults()
    {
        var snapshot = new DatasetSnapshot();

        Assert.NotNull(snapshot.SnapshotId);
        Assert.NotEmpty(snapshot.SnapshotId);
        Assert.Equal(string.Empty, snapshot.DatasetName);
        Assert.Equal(0, snapshot.Version);
    }

    #endregion
}
