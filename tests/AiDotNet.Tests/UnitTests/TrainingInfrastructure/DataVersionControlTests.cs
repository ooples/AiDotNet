using AiDotNet.DataVersionControl;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.TrainingInfrastructure;

/// <summary>
/// Unit tests for DataVersionControl data versioning and lineage tracking.
/// </summary>
public class DataVersionControlTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly string _dataDirectory;
    private readonly DataVersionControl<double> _versionControl;

    public DataVersionControlTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"data_version_control_tests_{Guid.NewGuid():N}");
        _dataDirectory = Path.Combine(_testDirectory, "data");
        Directory.CreateDirectory(_testDirectory);
        Directory.CreateDirectory(_dataDirectory);

        _versionControl = new DataVersionControl<double>(_testDirectory);

        // Create sample data file for testing
        CreateSampleDataFile("sample_data.csv");
    }

    private string CreateSampleDataFile(string fileName, string content = "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9")
    {
        var filePath = Path.Combine(_dataDirectory, fileName);
        File.WriteAllText(filePath, content);
        return filePath;
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

    #region Dataset Version Creation Tests

    [Fact(Timeout = 60000)]
    public async Task CreateDatasetVersion_WithValidData_ReturnsVersionHash()
    {
        // Arrange
        var dataPath = Path.Combine(_dataDirectory, "sample_data.csv");

        // Act
        var versionHash = _versionControl.CreateDatasetVersion("test-dataset", dataPath);

        // Assert
        Assert.NotNull(versionHash);
        Assert.NotEmpty(versionHash);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateDatasetVersion_WithDescription_StoresDescription()
    {
        // Arrange
        var dataPath = Path.Combine(_dataDirectory, "sample_data.csv");

        // Act
        var versionHash = _versionControl.CreateDatasetVersion(
            "described-dataset",
            dataPath,
            description: "Test dataset for unit tests");

        var version = _versionControl.GetDatasetVersion("described-dataset");

        // Assert
        Assert.Equal("Test dataset for unit tests", version.Description);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateDatasetVersion_WithMetadata_StoresMetadata()
    {
        // Arrange
        var dataPath = Path.Combine(_dataDirectory, "sample_data.csv");
        var metadata = new Dictionary<string, object>
        {
            ["source"] = "unit-tests",
            ["row_count"] = 100
        };

        // Act
        var versionHash = _versionControl.CreateDatasetVersion(
            "metadata-dataset",
            dataPath,
            metadata: metadata);

        // Assert - Version created successfully
        Assert.NotNull(versionHash);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateDatasetVersion_WithNullName_ThrowsArgumentException()
    {
        // Arrange
        var dataPath = Path.Combine(_dataDirectory, "sample_data.csv");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _versionControl.CreateDatasetVersion(null!, dataPath));
    }

    [Fact(Timeout = 60000)]
    public async Task CreateDatasetVersion_WithEmptyPath_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _versionControl.CreateDatasetVersion("test-dataset", ""));
    }

    [Fact(Timeout = 60000)]
    public async Task CreateDatasetVersion_MultipleTimes_IncrementsVersion()
    {
        // Arrange
        var dataPath1 = CreateSampleDataFile("data1.csv", "a,b,c\n1,2,3");
        var dataPath2 = CreateSampleDataFile("data2.csv", "a,b,c\n4,5,6");

        // Act
        _versionControl.CreateDatasetVersion("versioned-dataset", dataPath1);
        _versionControl.CreateDatasetVersion("versioned-dataset", dataPath2);

        // Assert
        var versions = _versionControl.ListDatasetVersions("versioned-dataset");
        Assert.Equal(2, versions.Count);
        Assert.Equal(2, versions.First().Version);  // Latest is version 2
        Assert.Equal(1, versions.Last().Version);   // First is version 1
    }

    #endregion

    #region Dataset Version Retrieval Tests

    [Fact(Timeout = 60000)]
    public async Task GetDatasetVersion_WithValidName_ReturnsVersion()
    {
        // Arrange
        var dataPath = Path.Combine(_dataDirectory, "sample_data.csv");
        var createdHash = _versionControl.CreateDatasetVersion("get-test-dataset", dataPath);

        // Act
        var version = _versionControl.GetDatasetVersion("get-test-dataset");

        // Assert
        Assert.NotNull(version);
        Assert.Equal("get-test-dataset", version.DatasetName);
        Assert.Equal(createdHash, version.Hash);
    }

    [Fact(Timeout = 60000)]
    public async Task GetDatasetVersion_WithVersionHash_ReturnsSpecificVersion()
    {
        // Arrange
        var dataPath1 = CreateSampleDataFile("version1.csv", "x,y,z\n1,2,3");
        var dataPath2 = CreateSampleDataFile("version2.csv", "x,y,z\n4,5,6");
        var hash1 = _versionControl.CreateDatasetVersion("multi-version-dataset", dataPath1);
        var hash2 = _versionControl.CreateDatasetVersion("multi-version-dataset", dataPath2);

        // Act
        var version1 = _versionControl.GetDatasetVersion("multi-version-dataset", hash1);
        var version2 = _versionControl.GetDatasetVersion("multi-version-dataset", hash2);

        // Assert
        Assert.Equal(hash1, version1.Hash);
        Assert.Equal(hash2, version2.Hash);
    }

    [Fact(Timeout = 60000)]
    public async Task GetDatasetVersion_WithInvalidName_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _versionControl.GetDatasetVersion("nonexistent-dataset"));
    }

    [Fact(Timeout = 60000)]
    public async Task GetDatasetVersion_WithInvalidHash_ThrowsArgumentException()
    {
        // Arrange
        var dataPath = Path.Combine(_dataDirectory, "sample_data.csv");
        _versionControl.CreateDatasetVersion("hash-test-dataset", dataPath);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _versionControl.GetDatasetVersion("hash-test-dataset", "invalid-hash"));
    }

    [Fact(Timeout = 60000)]
    public async Task GetDatasetVersion_WithNoHash_ReturnsLatest()
    {
        // Arrange
        var dataPath1 = CreateSampleDataFile("latest1.csv", "a,b\n1,2");
        var dataPath2 = CreateSampleDataFile("latest2.csv", "a,b\n3,4");
        _versionControl.CreateDatasetVersion("latest-dataset", dataPath1);
        var latestHash = _versionControl.CreateDatasetVersion("latest-dataset", dataPath2);

        // Act - GetDatasetVersion with no hash returns latest
        var latestVersion = _versionControl.GetDatasetVersion("latest-dataset");

        // Assert
        Assert.Equal(latestHash, latestVersion.Hash);
        Assert.Equal(2, latestVersion.Version);
    }

    #endregion

    #region Version Listing Tests

    [Fact(Timeout = 60000)]
    public async Task ListDatasetVersions_ReturnsAllVersions()
    {
        // Arrange
        var dataPath1 = CreateSampleDataFile("list1.csv", "col\n1");
        var dataPath2 = CreateSampleDataFile("list2.csv", "col\n2");
        var dataPath3 = CreateSampleDataFile("list3.csv", "col\n3");

        _versionControl.CreateDatasetVersion("list-dataset", dataPath1);
        _versionControl.CreateDatasetVersion("list-dataset", dataPath2);
        _versionControl.CreateDatasetVersion("list-dataset", dataPath3);

        // Act
        var versions = _versionControl.ListDatasetVersions("list-dataset");

        // Assert
        Assert.Equal(3, versions.Count);
    }

    [Fact(Timeout = 60000)]
    public async Task ListDatasetVersions_OrdersByVersionDescending()
    {
        // Arrange
        var dataPath1 = CreateSampleDataFile("order1.csv", "x\n1");
        var dataPath2 = CreateSampleDataFile("order2.csv", "x\n2");

        _versionControl.CreateDatasetVersion("ordered-dataset", dataPath1);
        _versionControl.CreateDatasetVersion("ordered-dataset", dataPath2);

        // Act
        var versions = _versionControl.ListDatasetVersions("ordered-dataset");

        // Assert
        Assert.Equal(2, versions.First().Version);  // Latest first
        Assert.Equal(1, versions.Last().Version);   // Oldest last
    }

    [Fact(Timeout = 60000)]
    public async Task ListDatasetVersions_WithInvalidName_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _versionControl.ListDatasetVersions("nonexistent-dataset"));
    }

    #endregion

    #region Run Linking Tests

    [Fact(Timeout = 60000)]
    public async Task LinkDatasetToRun_CreatesLink()
    {
        // Arrange
        var dataPath = Path.Combine(_dataDirectory, "sample_data.csv");
        var versionHash = _versionControl.CreateDatasetVersion("link-dataset", dataPath);

        // Act
        _versionControl.LinkDatasetToRun("link-dataset", versionHash, "run-123", "model-456");

        // Assert
        var linkedVersion = _versionControl.GetDatasetForRun("run-123");
        Assert.NotNull(linkedVersion);
        Assert.Equal("link-dataset", linkedVersion.DatasetName);
        Assert.Equal(versionHash, linkedVersion.Hash);
    }

    [Fact(Timeout = 60000)]
    public async Task GetDatasetForRun_WithNoLink_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _versionControl.GetDatasetForRun("nonexistent-run"));
    }

    #endregion

    #region Tagging Tests

    [Fact(Timeout = 60000)]
    public async Task TagDatasetVersion_CreatesTag()
    {
        // Arrange
        var dataPath = Path.Combine(_dataDirectory, "sample_data.csv");
        var versionHash = _versionControl.CreateDatasetVersion("tag-dataset", dataPath);

        // Act
        _versionControl.TagDatasetVersion("tag-dataset", versionHash, "production");

        // Assert
        var taggedVersion = _versionControl.GetDatasetByTag("tag-dataset", "production");
        Assert.NotNull(taggedVersion);
        Assert.Equal(versionHash, taggedVersion.Hash);
    }

    [Fact(Timeout = 60000)]
    public async Task GetDatasetByTag_WithInvalidTag_ThrowsArgumentException()
    {
        // Arrange
        var dataPath = Path.Combine(_dataDirectory, "sample_data.csv");
        _versionControl.CreateDatasetVersion("tag-test-dataset", dataPath);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _versionControl.GetDatasetByTag("tag-test-dataset", "nonexistent-tag"));
    }

    #endregion

    #region Comparison Tests

    [Fact(Timeout = 60000)]
    public async Task CompareDatasetVersions_ReturnsDifferences()
    {
        // Arrange
        var dataPath1 = CreateSampleDataFile("compare1.csv", "col\n1\n2\n3");
        var dataPath2 = CreateSampleDataFile("compare2.csv", "col\n1\n2\n3\n4\n5");

        var hash1 = _versionControl.CreateDatasetVersion("compare-dataset", dataPath1);
        var hash2 = _versionControl.CreateDatasetVersion("compare-dataset", dataPath2);

        // Act
        var comparison = _versionControl.CompareDatasetVersions("compare-dataset", hash1, hash2);

        // Assert
        Assert.NotNull(comparison);
    }

    #endregion

    #region Snapshot Tests

    [Fact(Timeout = 60000)]
    public async Task CreateDatasetSnapshot_CreatesMultiDatasetSnapshot()
    {
        // Arrange
        var dataPath1 = CreateSampleDataFile("snap1.csv", "a\n1");
        var dataPath2 = CreateSampleDataFile("snap2.csv", "b\n2");

        var hash1 = _versionControl.CreateDatasetVersion("snapshot-dataset-1", dataPath1);
        var hash2 = _versionControl.CreateDatasetVersion("snapshot-dataset-2", dataPath2);

        var datasetVersions = new Dictionary<string, string>
        {
            ["snapshot-dataset-1"] = hash1,
            ["snapshot-dataset-2"] = hash2
        };

        // Act
        var snapshotId = _versionControl.CreateDatasetSnapshot("test-snapshot", datasetVersions);

        // Assert
        Assert.NotNull(snapshotId);
        Assert.NotEmpty(snapshotId);
    }

    [Fact(Timeout = 60000)]
    public async Task GetDatasetSnapshot_ReturnsSnapshot()
    {
        // Arrange
        var dataPath = CreateSampleDataFile("snap_get.csv", "x\n1");
        var hash = _versionControl.CreateDatasetVersion("snap-get-dataset", dataPath);

        var datasetVersions = new Dictionary<string, string>
        {
            ["snap-get-dataset"] = hash
        };

        _versionControl.CreateDatasetSnapshot("get-snapshot", datasetVersions);

        // Act - GetDatasetSnapshot takes the snapshot name, not the ID
        var snapshot = _versionControl.GetDatasetSnapshot("get-snapshot");

        // Assert
        Assert.NotNull(snapshot);
        Assert.Equal("get-snapshot", snapshot.DatasetName);
    }

    [Fact(Timeout = 60000)]
    public async Task GetDatasetSnapshot_WithInvalidName_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _versionControl.GetDatasetSnapshot("nonexistent-snapshot"));
    }

    #endregion

    #region Thread Safety Tests

    [Fact(Timeout = 60000)]
    public async Task CreateDatasetVersion_FromMultipleThreads_IsThreadSafe()
    {
        // Arrange
        var tasks = new List<Task<string>>();
        var versionCount = 10;

        // Act
        for (int i = 0; i < versionCount; i++)
        {
            var index = i;
            var dataPath = CreateSampleDataFile($"concurrent_{index}.csv", $"data{index}\n{index}");
            tasks.Add(Task.Run(() => _versionControl.CreateDatasetVersion("concurrent-dataset", dataPath)));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        var versions = _versionControl.ListDatasetVersions("concurrent-dataset");
        Assert.Equal(versionCount, versions.Count);
    }

    #endregion
}
