using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

public class VisionBenchmarkTests
{
    [Fact]
    public async Task EuroSatLoader_LoadsSyntheticData()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            // Create EuroSAT-style directory structure
            string datasetRoot = Path.Combine(tempDir, "EuroSAT");
            Directory.CreateDirectory(datasetRoot);

            string[] classes = { "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake" };

            // Create 2 synthetic images per class
            foreach (var className in classes)
            {
                string classDir = Path.Combine(datasetRoot, className);
                Directory.CreateDirectory(classDir);

                for (int i = 0; i < 2; i++)
                {
                    // Create minimal synthetic image data (just raw bytes, not real JPEG)
                    byte[] fakeImage = new byte[64 * 64 * 3];
                    new Random(42 + i).NextBytes(fakeImage);
                    File.WriteAllBytes(Path.Combine(classDir, $"{className}_{i}.jpg"), fakeImage);
                }
            }

            var options = new EuroSatDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                Normalize = true,
                MaxSamples = 10
            };

            var loader = new EuroSatDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(10, loader.TotalCount);
            Assert.Equal(64 * 64 * 3, loader.FeatureCount);
            Assert.Equal(10, loader.OutputDimension);
            Assert.Equal("EuroSAT", loader.Name);

            // Check tensor shapes
            Assert.Equal(new[] { 10, 64, 64, 3 }, loader.Features.Shape);
            Assert.Equal(new[] { 10, 10 }, loader.Labels.Shape);

            // Verify pixel values are normalized to [0, 1]
            for (int i = 0; i < 10; i++)
            {
                double val = loader.Features[i, 0, 0, 0];
                Assert.True(val >= 0.0 && val <= 1.0, $"Pixel value {val} is not normalized.");
            }
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public async Task ChestXray14Loader_LoadsSyntheticCsvData()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            // Create directory structure
            string imageDir = Path.Combine(tempDir, "images");
            Directory.CreateDirectory(imageDir);

            // Create CSV data entry file
            string csv = "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,OriginalImage[Width,OriginalImage[Height,OriginalImagePixelSpacing[x,OriginalImagePixelSpacing[y\n" +
                         "00000001_000.png,Atelectasis|Effusion,0,1,58,M,PA,2682,2749,0.143,0.143\n" +
                         "00000002_000.png,No Finding,0,2,65,F,AP,2500,2500,0.168,0.168\n" +
                         "00000003_000.png,Pneumonia|Consolidation,0,3,42,M,PA,2048,2048,0.175,0.175\n";

            File.WriteAllText(Path.Combine(tempDir, "Data_Entry_2017_v2020.csv"), csv);

            // Create synthetic grayscale images
            for (int i = 1; i <= 3; i++)
            {
                byte[] fakeImage = new byte[224 * 224];
                new Random(i).NextBytes(fakeImage);
                File.WriteAllBytes(Path.Combine(imageDir, $"{i:D8}_000.png"), fakeImage);
            }

            var options = new ChestXray14DataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                Normalize = true,
                ImageSize = 224
            };

            var loader = new ChestXray14DataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(3, loader.TotalCount);
            Assert.Equal(224 * 224, loader.FeatureCount);
            Assert.Equal(14, loader.OutputDimension);
            Assert.Equal("ChestX-ray14", loader.Name);

            // Check labels shape
            Assert.Equal(new[] { 3, 14 }, loader.Labels.Shape);

            // First sample: Atelectasis (0) + Effusion (2)
            Assert.Equal(1.0, loader.Labels[0, 0], 6); // Atelectasis
            Assert.Equal(1.0, loader.Labels[0, 2], 6); // Effusion

            // Second sample: No Finding (all zeros)
            for (int j = 0; j < 14; j++)
                Assert.Equal(0.0, loader.Labels[1, j], 6);

            // Third sample: Pneumonia (6) + Consolidation (8)
            Assert.Equal(1.0, loader.Labels[2, 6], 6); // Pneumonia
            Assert.Equal(1.0, loader.Labels[2, 8], 6); // Consolidation
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public async Task PascalVocLoader_LoadsSyntheticXmlAnnotations()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            // Create VOC directory structure
            string vocDir = Path.Combine(tempDir, "VOCdevkit", "VOC2012");
            string imageDir = Path.Combine(vocDir, "JPEGImages");
            string annDir = Path.Combine(vocDir, "Annotations");
            string setsDir = Path.Combine(vocDir, "ImageSets", "Main");
            Directory.CreateDirectory(imageDir);
            Directory.CreateDirectory(annDir);
            Directory.CreateDirectory(setsDir);

            // Create trainval list
            File.WriteAllText(Path.Combine(setsDir, "trainval.txt"), "000001\n000002\n");

            // Create synthetic images
            for (int i = 1; i <= 2; i++)
            {
                byte[] fakeImage = new byte[500 * 500 * 3];
                File.WriteAllBytes(Path.Combine(imageDir, $"{i:D6}.jpg"), fakeImage);
            }

            // Create XML annotations
            string ann1 = @"<annotation>
  <size><width>500</width><height>500</height><depth>3</depth></size>
  <object>
    <name>cat</name>
    <bndbox><xmin>100</xmin><ymin>100</ymin><xmax>300</xmax><ymax>300</ymax></bndbox>
  </object>
  <object>
    <name>dog</name>
    <bndbox><xmin>200</xmin><ymin>200</ymin><xmax>400</xmax><ymax>400</ymax></bndbox>
  </object>
</annotation>";

            string ann2 = @"<annotation>
  <size><width>500</width><height>500</height><depth>3</depth></size>
  <object>
    <name>person</name>
    <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>250</xmax><ymax>450</ymax></bndbox>
  </object>
</annotation>";

            File.WriteAllText(Path.Combine(annDir, "000001.xml"), ann1);
            File.WriteAllText(Path.Combine(annDir, "000002.xml"), ann2);

            var options = new PascalVocDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                Normalize = false,
                ImageSize = 500,
                MaxDetections = 10,
                Year = "2012"
            };

            var loader = new PascalVocDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(2, loader.TotalCount);
            Assert.Equal(500 * 500 * 3, loader.FeatureCount);
            Assert.Equal(10 * 5, loader.OutputDimension);
            Assert.Contains("Pascal-VOC", loader.Name);

            // Check labels shape: [N, MaxDetections, 5]
            Assert.Equal(new[] { 2, 10, 5 }, loader.Labels.Shape);

            // First image: cat (index 7) at normalized bbox
            double catClass = loader.Labels[0, 0, 0]; // class index
            Assert.Equal(7.0, catClass, 6); // "cat" is index 7

            // Second image: person (index 14)
            double personClass = loader.Labels[1, 0, 0];
            Assert.Equal(14.0, personClass, 6); // "person" is index 14
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public async Task SkinLesionLoader_LoadsSyntheticData()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            // Create CSV ground truth
            string csv = "image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC\n" +
                         "ISIC_0024306,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n" +
                         "ISIC_0024307,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n";

            File.WriteAllText(Path.Combine(tempDir, "ISIC_2019_Training_GroundTruth.csv"), csv);

            // Create image directory and images
            string imageDir = Path.Combine(tempDir, "ISIC_2019_Training_Input");
            Directory.CreateDirectory(imageDir);

            for (int i = 6; i <= 7; i++)
            {
                byte[] fakeImage = new byte[224 * 224 * 3];
                File.WriteAllBytes(Path.Combine(imageDir, $"ISIC_002430{i}.jpg"), fakeImage);
            }

            var options = new SkinLesionDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                Normalize = false,
                ImageSize = 224
            };

            var loader = new SkinLesionDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(2, loader.TotalCount);
            Assert.Equal(8, loader.OutputDimension);

            // First sample: NV (index 1)
            Assert.Equal(0.0, loader.Labels[0, 0], 6);
            Assert.Equal(1.0, loader.Labels[0, 1], 6);

            // Second sample: MEL (index 0)
            Assert.Equal(1.0, loader.Labels[1, 0], 6);
            Assert.Equal(0.0, loader.Labels[1, 1], 6);
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    [Fact]
    public void ImageNet1kLoader_DefaultOptions()
    {
        var loader = new ImageNet1kDataLoader<double>();
        Assert.Equal("ImageNet-1K", loader.Name);
        Assert.Equal(1000, loader.OutputDimension);
        Assert.Equal(224 * 224 * 3, loader.FeatureCount);
    }

    [Fact]
    public void Places365Loader_DefaultOptions()
    {
        var loader = new Places365DataLoader<double>();
        Assert.Equal("Places365", loader.Name);
        Assert.Equal(365, loader.OutputDimension);
        Assert.Equal(256 * 256 * 3, loader.FeatureCount);
    }

    [Fact]
    public void CocoDetectionLoader_DefaultOptions()
    {
        var options = new CocoDetectionDataLoaderOptions { MaxDetections = 50 };
        var loader = new CocoDetectionDataLoader<double>(options);
        Assert.Equal("COCO-Detection", loader.Name);
        Assert.Equal(50 * 5, loader.OutputDimension);
    }

    [Fact]
    public void RetinalFundusLoader_DefaultOptions()
    {
        var loader = new RetinalFundusDataLoader<double>();
        Assert.Equal("RetinalFundus-DR", loader.Name);
        Assert.Equal(5, loader.OutputDimension);
    }

    [Fact]
    public void FMoWLoader_DefaultOptions()
    {
        var loader = new FMoWDataLoader<double>();
        Assert.Equal("fMoW", loader.Name);
        Assert.Equal(62, loader.OutputDimension);
    }

    [Fact]
    public void BigEarthNetLoader_DefaultOptions()
    {
        var loader = new BigEarthNetDataLoader<double>();
        Assert.Equal("BigEarthNet", loader.Name);
        Assert.Equal(19, loader.OutputDimension); // Default 19-class scheme
    }

    [Fact]
    public void BigEarthNetLoader_43ClassScheme()
    {
        var options = new BigEarthNetDataLoaderOptions { Use19ClassScheme = false };
        var loader = new BigEarthNetDataLoader<double>(options);
        Assert.Equal(43, loader.OutputDimension);
    }

    [Fact]
    public async Task EuroSatLoader_SplitReturnsThreeSets()
    {
        string tempDir = CreateTempDirectory();

        try
        {
            string datasetRoot = Path.Combine(tempDir, "EuroSAT");
            Directory.CreateDirectory(datasetRoot);

            string[] classes = { "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake" };

            foreach (var className in classes)
            {
                string classDir = Path.Combine(datasetRoot, className);
                Directory.CreateDirectory(classDir);

                for (int i = 0; i < 3; i++)
                {
                    byte[] fakeImage = new byte[64 * 64 * 3];
                    File.WriteAllBytes(Path.Combine(classDir, $"{className}_{i}.jpg"), fakeImage);
                }
            }

            var options = new EuroSatDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                MaxSamples = 20
            };

            var loader = new EuroSatDataLoader<double>(options);
            await loader.LoadAsync();

            var (train, val, test) = loader.Split(0.7, 0.15, seed: 42);

            Assert.True(train.TotalCount > 0);
            Assert.True(val.TotalCount > 0);
            Assert.True(test.TotalCount > 0);
            Assert.Equal(loader.TotalCount, train.TotalCount + val.TotalCount + test.TotalCount);
        }
        finally
        {
            CleanupDirectory(tempDir);
        }
    }

    private static string CreateTempDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "AiDotNet_VisionBenchmarkTests_" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(path);
        return path;
    }

    private static void CleanupDirectory(string path)
    {
        try
        {
            if (Directory.Exists(path))
                Directory.Delete(path, true);
        }
        catch
        {
            // Best-effort cleanup
        }
    }
}
