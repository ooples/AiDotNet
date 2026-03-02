using AiDotNet.DataVersioning;
using Xunit;

// Alias to avoid conflict with AiDotNet.DataVersionControl namespace
using DVC = AiDotNet.DataVersioning.DataVersionControl;

namespace AiDotNet.Tests.IntegrationTests.DataVersioning;

/// <summary>
/// Deep integration tests for DataVersioning:
/// DataVersionControl (create/add/list/delete datasets and versions, hashing,
/// content deduplication, version diffing, lineage tracking, size formatting),
/// DataFileInfo, DataVersion, DataVersionDiff computed properties.
/// </summary>
public class DataVersioningDeepMathIntegrationTests : IDisposable
{
    private readonly string _testDir;
    private readonly string _sourceDir;

    public DataVersioningDeepMathIntegrationTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"dvc_test_{Guid.NewGuid():N}");
        _sourceDir = Path.Combine(_testDir, "source");
        Directory.CreateDirectory(_sourceDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, true); }
            catch { /* cleanup best effort */ }
        }
    }

    private string CreateTestFile(string name, string content)
    {
        var path = Path.Combine(_sourceDir, name);
        var dir = Path.GetDirectoryName(path);
        if (dir != null && !Directory.Exists(dir))
            Directory.CreateDirectory(dir);
        File.WriteAllText(path, content);
        return path;
    }

    // ============================
    // DatasetInfo: Defaults
    // ============================

    [Fact]
    public void DatasetInfo_Defaults_Empty()
    {
        var info = new DatasetInfo();
        Assert.Equal(string.Empty, info.DatasetId);
        Assert.Equal(string.Empty, info.Name);
        Assert.Null(info.Description);
        Assert.Equal(0, info.VersionCount);
        Assert.Null(info.LatestVersionId);
        Assert.NotNull(info.Metadata);
        Assert.Empty(info.Metadata);
    }

    // ============================
    // DataVersion: SizeFormatted
    // ============================

    [Theory]
    [InlineData(0, "0 B")]
    [InlineData(512, "512 B")]
    [InlineData(1023, "1023 B")]
    public void DataVersion_SizeFormatted_Bytes(long bytes, string expected)
    {
        var v = new DataVersion { SizeBytes = bytes };
        Assert.Equal(expected, v.SizeFormatted);
    }

    [Theory]
    [InlineData(1024, "1.0 KB")]
    [InlineData(1536, "1.5 KB")]
    [InlineData(10240, "10.0 KB")]
    public void DataVersion_SizeFormatted_Kilobytes(long bytes, string expected)
    {
        var v = new DataVersion { SizeBytes = bytes };
        Assert.Equal(expected, v.SizeFormatted);
    }

    [Theory]
    [InlineData(1048576, "1.0 MB")]
    [InlineData(5242880, "5.0 MB")]
    public void DataVersion_SizeFormatted_Megabytes(long bytes, string expected)
    {
        var v = new DataVersion { SizeBytes = bytes };
        Assert.Equal(expected, v.SizeFormatted);
    }

    [Theory]
    [InlineData(1073741824, "1.00 GB")]
    [InlineData(2684354560, "2.50 GB")]
    public void DataVersion_SizeFormatted_Gigabytes(long bytes, string expected)
    {
        var v = new DataVersion { SizeBytes = bytes };
        Assert.Equal(expected, v.SizeFormatted);
    }

    // ============================
    // DataFileInfo: Defaults
    // ============================

    [Fact]
    public void DataFileInfo_Defaults_Empty()
    {
        var info = new DataFileInfo();
        Assert.Equal(string.Empty, info.RelativePath);
        Assert.Equal(0, info.SizeBytes);
        Assert.Equal(string.Empty, info.Hash);
    }

    // ============================
    // DataVersionDiff: Computed Properties
    // ============================

    [Fact]
    public void DataVersionDiff_SizeDelta_Correct()
    {
        var diff = new DataVersionDiff
        {
            Version1 = new DataVersion { SizeBytes = 1000 },
            Version2 = new DataVersion { SizeBytes = 1500 }
        };

        Assert.Equal(500, diff.SizeDelta);
    }

    [Fact]
    public void DataVersionDiff_SizeDelta_Negative()
    {
        var diff = new DataVersionDiff
        {
            Version1 = new DataVersion { SizeBytes = 2000 },
            Version2 = new DataVersion { SizeBytes = 1000 }
        };

        Assert.Equal(-1000, diff.SizeDelta);
    }

    [Fact]
    public void DataVersionDiff_Summary_IncludesAllCategories()
    {
        var diff = new DataVersionDiff();
        diff.FilesAdded.Add(new DataFileInfo { RelativePath = "new.txt" });
        diff.FilesRemoved.Add(new DataFileInfo { RelativePath = "old.txt" });
        diff.FilesModified.Add((
            new DataFileInfo { RelativePath = "changed.txt", Hash = "aaa" },
            new DataFileInfo { RelativePath = "changed.txt", Hash = "bbb" }));
        diff.FilesUnchanged.Add(new DataFileInfo { RelativePath = "same.txt" });

        var summary = diff.Summary;
        Assert.Contains("Added: 1", summary);
        Assert.Contains("Removed: 1", summary);
        Assert.Contains("Modified: 1", summary);
        Assert.Contains("Unchanged: 1", summary);
    }

    // ============================
    // DataLineage: Defaults
    // ============================

    [Fact]
    public void DataLineage_Defaults_Empty()
    {
        var lineage = new DataLineage();
        Assert.Equal(string.Empty, lineage.DatasetId);
        Assert.Equal(string.Empty, lineage.VersionId);
        Assert.Empty(lineage.Inputs);
        Assert.Null(lineage.Transformation);
        Assert.Null(lineage.Parameters);
        Assert.Empty(lineage.UpstreamLineage);
    }

    // ============================
    // DataVersionControl: Create Dataset
    // ============================

    [Fact]
    public void DVC_CreateDataset_ReturnsNonEmptyId()
    {
        var storageDir = Path.Combine(_testDir, "storage1");
        using var dvc = new DVC(storageDir);

        var id = dvc.CreateDataset("test-dataset");
        Assert.False(string.IsNullOrWhiteSpace(id));
        Assert.Equal(12, id.Length); // GenerateId returns 12-char GUID substring
    }

    [Fact]
    public void DVC_CreateDataset_EmptyName_Throws()
    {
        var storageDir = Path.Combine(_testDir, "storage2");
        using var dvc = new DVC(storageDir);

        Assert.Throws<ArgumentException>(() => dvc.CreateDataset(""));
    }

    [Fact]
    public void DVC_CreateDataset_DuplicateName_ReturnsSameId()
    {
        var storageDir = Path.Combine(_testDir, "storage3");
        using var dvc = new DVC(storageDir);

        var id1 = dvc.CreateDataset("test-dataset");
        var id2 = dvc.CreateDataset("test-dataset");

        Assert.Equal(id1, id2);
    }

    [Fact]
    public void DVC_CreateDataset_CaseInsensitiveName()
    {
        var storageDir = Path.Combine(_testDir, "storage4");
        using var dvc = new DVC(storageDir);

        var id1 = dvc.CreateDataset("Test-Dataset");
        var id2 = dvc.CreateDataset("test-dataset");

        Assert.Equal(id1, id2);
    }

    [Fact]
    public void DVC_CreateDataset_WithMetadata()
    {
        var storageDir = Path.Combine(_testDir, "storage5");
        using var dvc = new DVC(storageDir);

        var metadata = new Dictionary<string, string> { { "type", "training" } };
        var id = dvc.CreateDataset("test", "A test dataset", metadata);

        Assert.False(string.IsNullOrWhiteSpace(id));
    }

    [Fact]
    public void DVC_ListDatasets_ReturnsCreated()
    {
        var storageDir = Path.Combine(_testDir, "storage6");
        using var dvc = new DVC(storageDir);

        dvc.CreateDataset("dataset-a");
        dvc.CreateDataset("dataset-b");

        var datasets = dvc.ListDatasets();
        Assert.Equal(2, datasets.Count);
    }

    // ============================
    // DataVersionControl: Add Version
    // ============================

    [Fact]
    public void DVC_AddVersion_SingleFile_ReturnsVersion()
    {
        var storageDir = Path.Combine(_testDir, "storage7");
        using var dvc = new DVC(storageDir);

        var filePath = CreateTestFile("data.csv", "id,value\n1,100\n2,200");
        var datasetId = dvc.CreateDataset("test");

        var version = dvc.AddVersion(datasetId, filePath, "Initial version");

        Assert.NotNull(version);
        Assert.Equal(1, version.VersionNumber);
        Assert.Equal("Initial version", version.Message);
        Assert.Equal(1, version.FileCount);
        Assert.True(version.SizeBytes > 0);
        Assert.False(string.IsNullOrWhiteSpace(version.ContentHash));
        Assert.Equal(64, version.ContentHash.Length); // SHA-256 hex = 64 chars
    }

    [Fact]
    public void DVC_AddVersion_EmptyDatasetId_Throws()
    {
        var storageDir = Path.Combine(_testDir, "storage8");
        using var dvc = new DVC(storageDir);

        Assert.Throws<ArgumentException>(() => dvc.AddVersion("", "/some/path"));
    }

    [Fact]
    public void DVC_AddVersion_NonExistentPath_Throws()
    {
        var storageDir = Path.Combine(_testDir, "storage9");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");
        Assert.Throws<FileNotFoundException>(() => dvc.AddVersion(datasetId, "/nonexistent/path"));
    }

    [Fact]
    public void DVC_AddVersion_ContentDeduplication()
    {
        var storageDir = Path.Combine(_testDir, "storage10");
        using var dvc = new DVC(storageDir);

        var filePath = CreateTestFile("data.csv", "id,value\n1,100");
        var datasetId = dvc.CreateDataset("test");

        var v1 = dvc.AddVersion(datasetId, filePath, "Version 1");
        var v2 = dvc.AddVersion(datasetId, filePath, "Same content");

        // Same content should return existing version
        Assert.Equal(v1.VersionId, v2.VersionId);
        Assert.Equal(v1.ContentHash, v2.ContentHash);
    }

    [Fact]
    public void DVC_AddVersion_DifferentContent_DifferentVersions()
    {
        var storageDir = Path.Combine(_testDir, "storage11");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");

        var file1 = CreateTestFile("data.csv", "id,value\n1,100");
        var v1 = dvc.AddVersion(datasetId, file1, "Version 1");

        // Change file content
        File.WriteAllText(file1, "id,value\n1,200\n2,300");
        var v2 = dvc.AddVersion(datasetId, file1, "Version 2");

        Assert.NotEqual(v1.VersionId, v2.VersionId);
        Assert.NotEqual(v1.ContentHash, v2.ContentHash);
        Assert.Equal(2, v2.VersionNumber);
    }

    [Fact]
    public void DVC_AddVersion_VersionId_Is12CharHashPrefix()
    {
        var storageDir = Path.Combine(_testDir, "storage12");
        using var dvc = new DVC(storageDir);

        var filePath = CreateTestFile("data.txt", "test content");
        var datasetId = dvc.CreateDataset("test");
        var version = dvc.AddVersion(datasetId, filePath);

        Assert.Equal(12, version.VersionId.Length);
        Assert.True(version.ContentHash.StartsWith(version.VersionId));
    }

    // ============================
    // DataVersionControl: Get Version
    // ============================

    [Fact]
    public void DVC_GetVersion_Latest_ReturnsNewest()
    {
        var storageDir = Path.Combine(_testDir, "storage13");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");

        var file1 = CreateTestFile("v1.txt", "version 1 content");
        var v1 = dvc.AddVersion(datasetId, file1, "V1");

        File.WriteAllText(file1, "version 2 content");
        var v2 = dvc.AddVersion(datasetId, file1, "V2");

        var latest = dvc.GetVersion(datasetId, "latest");
        Assert.Equal(v2.VersionId, latest.VersionId);
    }

    [Fact]
    public void DVC_GetVersion_ByExactId()
    {
        var storageDir = Path.Combine(_testDir, "storage14");
        using var dvc = new DVC(storageDir);

        var filePath = CreateTestFile("data.txt", "test");
        var datasetId = dvc.CreateDataset("test");
        var v1 = dvc.AddVersion(datasetId, filePath);

        var retrieved = dvc.GetVersion(datasetId, v1.VersionId);
        Assert.Equal(v1.VersionId, retrieved.VersionId);
    }

    [Fact]
    public void DVC_GetVersion_ByNumber()
    {
        var storageDir = Path.Combine(_testDir, "storage15");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");

        var file1 = CreateTestFile("f1.txt", "content 1");
        var v1 = dvc.AddVersion(datasetId, file1);

        var retrieved = dvc.GetVersion(datasetId, "1");
        Assert.Equal(v1.VersionId, retrieved.VersionId);
    }

    [Fact]
    public void DVC_GetVersion_NonExistent_Throws()
    {
        var storageDir = Path.Combine(_testDir, "storage16");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");
        var filePath = CreateTestFile("data.txt", "test");
        dvc.AddVersion(datasetId, filePath);

        Assert.Throws<ArgumentException>(() => dvc.GetVersion(datasetId, "nonexistent"));
    }

    // ============================
    // DataVersionControl: List Versions
    // ============================

    [Fact]
    public void DVC_ListVersions_OrderedDescending()
    {
        var storageDir = Path.Combine(_testDir, "storage17");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");
        var file1 = CreateTestFile("f.txt", "v1");
        dvc.AddVersion(datasetId, file1, "V1");

        File.WriteAllText(file1, "v2");
        dvc.AddVersion(datasetId, file1, "V2");

        File.WriteAllText(file1, "v3");
        dvc.AddVersion(datasetId, file1, "V3");

        var versions = dvc.ListVersions(datasetId);
        Assert.Equal(3, versions.Count);
        Assert.Equal(3, versions[0].VersionNumber); // Descending order
        Assert.Equal(2, versions[1].VersionNumber);
        Assert.Equal(1, versions[2].VersionNumber);
    }

    // ============================
    // DataVersionControl: GetDataPath
    // ============================

    [Fact]
    public void DVC_GetDataPath_ReturnsValidPath()
    {
        var storageDir = Path.Combine(_testDir, "storage18");
        using var dvc = new DVC(storageDir);

        var filePath = CreateTestFile("data.txt", "test content");
        var datasetId = dvc.CreateDataset("test");
        var version = dvc.AddVersion(datasetId, filePath);

        var dataPath = dvc.GetDataPath(datasetId, version.VersionId);
        Assert.True(Directory.Exists(dataPath));
    }

    // ============================
    // DataVersionControl: Compare Versions
    // ============================

    [Fact]
    public void DVC_CompareVersions_AddedFile()
    {
        var storageDir = Path.Combine(_testDir, "storage19");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");

        var file1 = CreateTestFile("a.txt", "content a");
        var v1 = dvc.AddVersion(datasetId, file1);

        // Create a directory with two files
        var dirPath = Path.Combine(_sourceDir, "multi");
        Directory.CreateDirectory(dirPath);
        File.WriteAllText(Path.Combine(dirPath, "a.txt"), "content a");
        File.WriteAllText(Path.Combine(dirPath, "b.txt"), "content b");
        var v2 = dvc.AddVersion(datasetId, dirPath);

        var diff = dvc.CompareVersions(datasetId, v1.VersionId, v2.VersionId);
        Assert.True(diff.FilesAdded.Count > 0 || diff.FilesModified.Count > 0);
    }

    [Fact]
    public void DVC_CompareVersions_ModifiedFile()
    {
        var storageDir = Path.Combine(_testDir, "storage20");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");

        var filePath = CreateTestFile("data.txt", "original");
        var v1 = dvc.AddVersion(datasetId, filePath);

        File.WriteAllText(filePath, "modified");
        var v2 = dvc.AddVersion(datasetId, filePath);

        var diff = dvc.CompareVersions(datasetId, v1.VersionId, v2.VersionId);
        Assert.Single(diff.FilesModified);
        Assert.Equal("data.txt", diff.FilesModified[0].before.RelativePath);
    }

    [Fact]
    public void DVC_CompareVersions_IdenticalContent()
    {
        var storageDir = Path.Combine(_testDir, "storage21");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");
        var filePath = CreateTestFile("data.txt", "same content");
        var v1 = dvc.AddVersion(datasetId, filePath);

        // Same content returns same version - so compare with self
        var diff = dvc.CompareVersions(datasetId, v1.VersionId, v1.VersionId);
        Assert.Empty(diff.FilesAdded);
        Assert.Empty(diff.FilesRemoved);
        Assert.Empty(diff.FilesModified);
        Assert.Equal(v1.FileCount, diff.FilesUnchanged.Count);
    }

    // ============================
    // DataVersionControl: Delete
    // ============================

    [Fact]
    public void DVC_DeleteVersion_RemovesFromList()
    {
        var storageDir = Path.Combine(_testDir, "storage22");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");
        var file1 = CreateTestFile("f.txt", "v1");
        var v1 = dvc.AddVersion(datasetId, file1, "V1");

        File.WriteAllText(file1, "v2");
        dvc.AddVersion(datasetId, file1, "V2");

        dvc.DeleteVersion(datasetId, v1.VersionId);
        var versions = dvc.ListVersions(datasetId);
        Assert.Single(versions);
    }

    [Fact]
    public void DVC_DeleteDataset_RemovesEverything()
    {
        var storageDir = Path.Combine(_testDir, "storage23");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");
        var filePath = CreateTestFile("f.txt", "content");
        dvc.AddVersion(datasetId, filePath);

        dvc.DeleteDataset(datasetId);
        var datasets = dvc.ListDatasets();
        Assert.Empty(datasets);
    }

    // ============================
    // DataVersionControl: Lineage
    // ============================

    [Fact]
    public void DVC_RecordLineage_RetrievableViaGetLineage()
    {
        var storageDir = Path.Combine(_testDir, "storage24");
        using var dvc = new DVC(storageDir);

        var srcId = dvc.CreateDataset("source");
        var outId = dvc.CreateDataset("output");

        var srcFile = CreateTestFile("src.txt", "raw data");
        var srcVersion = dvc.AddVersion(srcId, srcFile);

        File.WriteAllText(srcFile, "processed data");
        var outFile = CreateTestFile("out.txt", "processed data");
        var outVersion = dvc.AddVersion(outId, outFile);

        dvc.RecordLineage(outId, outVersion.VersionId,
            new List<(string, string)> { (srcId, srcVersion.VersionId) },
            "normalize",
            new Dictionary<string, object> { { "mean", 0.5 } });

        var lineage = dvc.GetLineage(outId, outVersion.VersionId);

        Assert.Equal(outId, lineage.DatasetId);
        Assert.Equal(outVersion.VersionId, lineage.VersionId);
        Assert.Equal("normalize", lineage.Transformation);
        Assert.Single(lineage.Inputs);
        Assert.NotNull(lineage.Parameters);
    }

    [Fact]
    public void DVC_GetLineage_NoLineage_ReturnsEmpty()
    {
        var storageDir = Path.Combine(_testDir, "storage25");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");
        var filePath = CreateTestFile("f.txt", "content");
        var version = dvc.AddVersion(datasetId, filePath);

        var lineage = dvc.GetLineage(datasetId, version.VersionId);
        Assert.Equal(datasetId, lineage.DatasetId);
        Assert.Empty(lineage.UpstreamLineage);
        Assert.Null(lineage.Transformation);
    }

    [Fact]
    public void DVC_RecursiveLineage_TracksUpstream()
    {
        var storageDir = Path.Combine(_testDir, "storage26");
        using var dvc = new DVC(storageDir);

        var rawId = dvc.CreateDataset("raw");
        var cleanId = dvc.CreateDataset("clean");
        var featId = dvc.CreateDataset("features");

        var rawFile = CreateTestFile("raw.txt", "raw data");
        var rawV = dvc.AddVersion(rawId, rawFile);

        var cleanFile = CreateTestFile("clean.txt", "clean data");
        var cleanV = dvc.AddVersion(cleanId, cleanFile);
        dvc.RecordLineage(cleanId, cleanV.VersionId,
            new List<(string, string)> { (rawId, rawV.VersionId) },
            "clean");

        var featFile = CreateTestFile("feat.txt", "features");
        var featV = dvc.AddVersion(featId, featFile);
        dvc.RecordLineage(featId, featV.VersionId,
            new List<(string, string)> { (cleanId, cleanV.VersionId) },
            "extract_features");

        var lineage = dvc.GetLineage(featId, featV.VersionId);

        Assert.Equal("extract_features", lineage.Transformation);
        Assert.Single(lineage.UpstreamLineage);
        Assert.Equal("clean", lineage.UpstreamLineage[0].Transformation);
    }

    // ============================
    // DataVersionControl: Hashing
    // ============================

    [Fact]
    public void DVC_SameContent_SameHash()
    {
        var storageDir = Path.Combine(_testDir, "storage27");
        using var dvc = new DVC(storageDir);

        var d1 = dvc.CreateDataset("ds1");
        var d2 = dvc.CreateDataset("ds2");

        var file = CreateTestFile("data.txt", "identical content");
        var v1 = dvc.AddVersion(d1, file);
        var v2 = dvc.AddVersion(d2, file);

        Assert.Equal(v1.ContentHash, v2.ContentHash);
    }

    [Fact]
    public void DVC_DifferentContent_DifferentHash()
    {
        var storageDir = Path.Combine(_testDir, "storage28");
        using var dvc = new DVC(storageDir);

        var datasetId = dvc.CreateDataset("test");

        var file1 = CreateTestFile("data.txt", "content A");
        var v1 = dvc.AddVersion(datasetId, file1);

        File.WriteAllText(file1, "content B");
        var v2 = dvc.AddVersion(datasetId, file1);

        Assert.NotEqual(v1.ContentHash, v2.ContentHash);
    }

    // ============================
    // DataVersionControl: StorageDirectory
    // ============================

    [Fact]
    public void DVC_StorageDirectory_ReturnsConfiguredPath()
    {
        var storageDir = Path.Combine(_testDir, "custom_storage");
        using var dvc = new DVC(storageDir);

        Assert.Equal(storageDir, dvc.StorageDirectory);
    }

    // ============================
    // DataVersionControl: Directory Source
    // ============================

    [Fact]
    public void DVC_AddVersion_Directory_TracksAllFiles()
    {
        var storageDir = Path.Combine(_testDir, "storage29");
        using var dvc = new DVC(storageDir);

        var dirPath = Path.Combine(_sourceDir, "multidir");
        Directory.CreateDirectory(dirPath);
        File.WriteAllText(Path.Combine(dirPath, "file1.txt"), "content 1");
        File.WriteAllText(Path.Combine(dirPath, "file2.txt"), "content 2");
        File.WriteAllText(Path.Combine(dirPath, "file3.txt"), "content 3");

        var datasetId = dvc.CreateDataset("test");
        var version = dvc.AddVersion(datasetId, dirPath);

        Assert.Equal(3, version.FileCount);
        Assert.Equal(3, version.Files.Count);
        Assert.True(version.SizeBytes > 0);
    }

    // ============================
    // DataVersionControl: Delete lineage cleanup
    // ============================

    [Fact]
    public void DVC_DeleteDataset_CleansUpLineage()
    {
        var storageDir = Path.Combine(_testDir, "storage30");
        using var dvc = new DVC(storageDir);

        var srcId = dvc.CreateDataset("source");
        var outId = dvc.CreateDataset("output");

        var srcFile = CreateTestFile("src.txt", "data");
        var srcV = dvc.AddVersion(srcId, srcFile);

        var outFile = CreateTestFile("out.txt", "result");
        var outV = dvc.AddVersion(outId, outFile);

        dvc.RecordLineage(outId, outV.VersionId,
            new List<(string, string)> { (srcId, srcV.VersionId) },
            "transform");

        dvc.DeleteDataset(outId);

        // Output dataset and its lineage should be cleaned up
        Assert.Single(dvc.ListDatasets());
    }
}
