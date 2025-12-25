using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

public class PointCloudDatasetLoaderIntegrationTests
{
    [Fact]
    public async Task ModelNet40Loader_LoadsPointCloudsAndLabels()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            string datasetRoot = Path.Combine(tempDir, "modelnet40_normal_resampled");
            Directory.CreateDirectory(datasetRoot);
            File.WriteAllLines(Path.Combine(datasetRoot, "modelnet40_shape_names.txt"), new[] { "chair", "table" });
            File.WriteAllLines(Path.Combine(datasetRoot, "modelnet40_train.txt"), new[] { "chair_0001", "table_0001" });

            string chairDir = Path.Combine(datasetRoot, "chair");
            string tableDir = Path.Combine(datasetRoot, "table");
            Directory.CreateDirectory(chairDir);
            Directory.CreateDirectory(tableDir);

            File.WriteAllLines(Path.Combine(chairDir, "chair_0001.txt"), new[]
            {
                "0 0 0 1 0 0",
                "1 0 0 0 1 0",
                "0 1 0 0 0 1",
                "1 1 1 0 0 1"
            });

            File.WriteAllLines(Path.Combine(tableDir, "table_0001.txt"), new[]
            {
                "0 0 1 1 0 0",
                "1 0 1 0 1 0",
                "0 1 1 0 0 1",
                "1 1 2 0 0 1"
            });

            var options = new ModelNet40ClassificationDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                Split = DatasetSplit.Train,
                PointsPerSample = 4,
                IncludeNormals = true,
                SamplingStrategy = PointSamplingStrategy.Sequential
            };

            var loader = new ModelNet40ClassificationDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(2, loader.TotalCount);
            Assert.Equal(4, loader.FeatureCount);
            Assert.Equal(6, loader.FeatureDimension);
            Assert.Equal(2, loader.OutputDimension);
            Assert.Equal(new[] { 2, 4, 6 }, loader.Features.Shape);
            Assert.Equal(new[] { 2, 2 }, loader.Labels.Shape);
            Assert.Equal(1.0, loader.Labels[0, 0], 6);
            Assert.Equal(1.0, loader.Labels[1, 1], 6);
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public async Task ShapeNetCoreLoader_LoadsSegmentationLabels()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            string datasetRoot = Path.Combine(tempDir, "shapenetcore_partanno_segmentation_benchmark_v0");
            Directory.CreateDirectory(datasetRoot);

            File.WriteAllLines(Path.Combine(datasetRoot, "synsetoffset2category.txt"), new[]
            {
                "Airplane 02691156"
            });

            string splitDir = Path.Combine(datasetRoot, "train_test_split");
            Directory.CreateDirectory(splitDir);
            File.WriteAllText(Path.Combine(splitDir, "shuffled_train_file_list.json"), "[\"02691156/0001\"]");

            string pointsDir = Path.Combine(datasetRoot, "points", "02691156");
            string labelsDir = Path.Combine(datasetRoot, "points_label", "02691156");
            Directory.CreateDirectory(pointsDir);
            Directory.CreateDirectory(labelsDir);

            File.WriteAllLines(Path.Combine(pointsDir, "0001.pts"), new[]
            {
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "0 0 1"
            });

            File.WriteAllLines(Path.Combine(labelsDir, "0001.seg"), new[]
            {
                "0",
                "1",
                "2",
                "3"
            });

            var options = new ShapeNetCorePartSegmentationDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                Split = DatasetSplit.Train,
                PointsPerSample = 4,
                IncludeNormals = false,
                NumClasses = 4,
                SamplingStrategy = PointSamplingStrategy.Sequential
            };

            var loader = new ShapeNetCorePartSegmentationDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(1, loader.TotalCount);
            Assert.Equal(4, loader.FeatureCount);
            Assert.Equal(3, loader.FeatureDimension);
            Assert.Equal(4, loader.OutputDimension);
            Assert.Equal(new[] { 1, 4, 3 }, loader.Features.Shape);
            Assert.Equal(new[] { 1, 4 }, loader.Labels.Shape);
            Assert.Equal(3.0, loader.Labels[0, 3], 6);
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public async Task ScanNetLoader_LoadsPreprocessedScene()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            File.WriteAllLines(Path.Combine(tempDir, "scannetv2_train.txt"), new[] { "scene0000_00" });

            string sceneDir = Path.Combine(tempDir, "scans", "scene0000_00");
            Directory.CreateDirectory(sceneDir);

            File.WriteAllLines(Path.Combine(sceneDir, "scene0000_00.txt"), new[]
            {
                "0 0 0 10 20 30 1",
                "1 0 0 20 30 40 2",
                "0 1 0 30 40 50 3",
                "0 0 1 40 50 60 4"
            });

            var options = new ScanNetSemanticSegmentationDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                Split = DatasetSplit.Train,
                PointsPerSample = 4,
                IncludeColors = true,
                IncludeNormals = false,
                InputFormat = ScanNetInputFormat.PreprocessedText,
                LabelMode = ScanNetLabelMode.Train20,
                IncludeUnknownClass = true,
                SamplingStrategy = PointSamplingStrategy.Sequential,
                AutoDetectLabelColumn = true
            };

            var loader = new ScanNetSemanticSegmentationDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(1, loader.TotalCount);
            Assert.Equal(4, loader.FeatureCount);
            Assert.Equal(6, loader.FeatureDimension);
            Assert.Equal(4, loader.OutputDimension);
            Assert.Equal(new[] { 1, 4, 6 }, loader.Features.Shape);
            Assert.Equal(new[] { 1, 4 }, loader.Labels.Shape);
            Assert.Equal(4.0, loader.Labels[0, 3], 6);
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    private static string CreateTempDirectory()
    {
        string root = Path.Combine(Path.GetTempPath(), "AiDotNetTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        return root;
    }

    private static void CleanupDirectory(string path)
    {
        if (Directory.Exists(path))
        {
            Directory.Delete(path, true);
        }
    }
}
