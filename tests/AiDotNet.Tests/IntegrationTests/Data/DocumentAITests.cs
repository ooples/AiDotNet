using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

public class DocumentAITests
{
    [Fact]
    public void DocVqaOptions_DefaultValues()
    {
        var options = new DocVqaDataLoaderOptions();
        Assert.Equal(DatasetSplit.Train, options.Split);
        Assert.Equal(224, options.ImageWidth);
        Assert.Equal(64, options.MaxQuestionLength);
        Assert.Equal(128, options.MaxAnswerLength);
    }

    [Fact]
    public void PubLayNetOptions_DefaultValues()
    {
        var options = new PubLayNetDataLoaderOptions();
        Assert.Equal(DatasetSplit.Train, options.Split);
        Assert.Equal(50, options.MaxRegions);
        Assert.Equal(5, options.NumClasses);
    }

    [Fact]
    public void Ade20kOptions_DefaultValues()
    {
        var options = new Ade20kDataLoaderOptions();
        Assert.Equal(DatasetSplit.Train, options.Split);
        Assert.Equal(150, options.NumClasses);
        Assert.True(options.Normalize);
    }

    [Fact]
    public void CelebAOptions_DefaultValues()
    {
        var options = new CelebADataLoaderOptions();
        Assert.Equal(DatasetSplit.Train, options.Split);
        Assert.Equal(64, options.ImageWidth);
        Assert.Equal(40, options.NumAttributes);
        Assert.True(options.Normalize);
    }

    [Fact]
    public async Task DocVqaDataLoader_LoadsImageDataWithAnswers()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "docvqa_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string trainDir = Path.Combine(tempDir, "train");
            string docsDir = Path.Combine(trainDir, "documents");
            Directory.CreateDirectory(docsDir);

            // Create small synthetic PNG-like files
            for (int i = 0; i < 3; i++)
            {
                File.WriteAllBytes(Path.Combine(docsDir, $"doc_{i}.png"), new byte[8 * 8 * 3]);
            }

            // Create annotations JSON with answers
            string annotationsJson = @"{
                ""data"": [
                    { ""image"": ""doc_0.png"", ""question"": ""What is the title?"", ""answers"": [""Hello""] },
                    { ""image"": ""doc_1.png"", ""question"": ""What is the date?"", ""answers"": [""2024""] },
                    { ""image"": ""doc_2.png"", ""question"": ""Who signed?"", ""answers"": [""ABC""] }
                ]
            }";
            File.WriteAllText(Path.Combine(trainDir, "annotations.json"), annotationsJson);

            int maxAnswerLen = 16;
            var options = new DocVqaDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                ImageWidth = 8,
                ImageHeight = 8,
                MaxSamples = 3,
                MaxAnswerLength = maxAnswerLen
            };

            var loader = new DocVqaDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(3, loader.TotalCount);
            Assert.Equal(192, loader.FeatureCount); // 8 * 8 * 3
            Assert.Equal(maxAnswerLen, loader.OutputDimension);

            // Verify character-level answer encoding
            var batch = loader.GetBatches(batchSize: 3, shuffle: false).First();

            // doc_0.png -> "Hello": H=72, e=101, l=108, l=108, o=111
            Assert.Equal(72.0, batch.Labels[0, 0]); // 'H'
            Assert.Equal(101.0, batch.Labels[0, 1]); // 'e'
            Assert.Equal(108.0, batch.Labels[0, 2]); // 'l'
            Assert.Equal(108.0, batch.Labels[0, 3]); // 'l'
            Assert.Equal(111.0, batch.Labels[0, 4]); // 'o'
            Assert.Equal(0.0, batch.Labels[0, 5]); // padding

            // doc_1.png -> "2024": '2'=50, '0'=48, '2'=50, '4'=52
            Assert.Equal(50.0, batch.Labels[1, 0]); // '2'
            Assert.Equal(48.0, batch.Labels[1, 1]); // '0'
            Assert.Equal(50.0, batch.Labels[1, 2]); // '2'
            Assert.Equal(52.0, batch.Labels[1, 3]); // '4'
            Assert.Equal(0.0, batch.Labels[1, 4]); // padding
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task CelebADataLoader_LoadsFaceImages()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "celeba_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string imgDir = Path.Combine(tempDir, "img_align_celeba");
            Directory.CreateDirectory(imgDir);

            // Create small image files
            for (int i = 0; i < 4; i++)
            {
                File.WriteAllBytes(Path.Combine(imgDir, $"{i + 1:D6}.jpg"), new byte[4 * 4 * 3]);
            }

            // Create attributes file
            var attrLines = new List<string>
            {
                "4",
                "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald"
            };
            for (int i = 0; i < 4; i++)
            {
                attrLines.Add($"{i + 1:D6}.jpg  1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1");
            }
            File.WriteAllLines(Path.Combine(tempDir, "list_attr_celeba.txt"), attrLines);

            // Create partition file (all train)
            var partLines = Enumerable.Range(1, 4).Select(i => $"{i:D6}.jpg 0");
            File.WriteAllLines(Path.Combine(tempDir, "list_eval_partition.txt"), partLines);

            var options = new CelebADataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                ImageWidth = 4,
                ImageHeight = 4,
                MaxSamples = 4
            };

            var loader = new CelebADataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(4, loader.TotalCount);
            Assert.Equal(48, loader.FeatureCount); // 4 * 4 * 3
            Assert.Equal(40, loader.OutputDimension);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task Ade20kDataLoader_LoadsSegmentationData()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "ade20k_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string imgDir = Path.Combine(tempDir, "images", "training");
            string annDir = Path.Combine(tempDir, "annotations", "training");
            Directory.CreateDirectory(imgDir);
            Directory.CreateDirectory(annDir);

            for (int i = 0; i < 3; i++)
            {
                File.WriteAllBytes(Path.Combine(imgDir, $"ADE_train_{i:D8}.jpg"), new byte[4 * 4 * 3]);
                // Annotation mask: each byte is a class index
                byte[] maskData = Enumerable.Range(0, 16).Select(p => (byte)(p % 5)).ToArray();
                File.WriteAllBytes(Path.Combine(annDir, $"ADE_train_{i:D8}.png"), maskData);
            }

            var options = new Ade20kDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                ImageWidth = 4,
                ImageHeight = 4,
                NumClasses = 5,
                MaxSamples = 3
            };

            var loader = new Ade20kDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(3, loader.TotalCount);
            Assert.Equal(48, loader.FeatureCount); // 4 * 4 * 3
            Assert.Equal(16, loader.OutputDimension); // 4 * 4
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task PubLayNetDataLoader_LoadsLayoutData()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "publaynet_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string imgDir = Path.Combine(tempDir, "train", "images");
            Directory.CreateDirectory(imgDir);

            for (int i = 0; i < 3; i++)
            {
                File.WriteAllBytes(Path.Combine(imgDir, $"PMC{i:D7}.jpg"), new byte[4 * 4 * 3]);
            }

            var options = new PubLayNetDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                ImageWidth = 4,
                ImageHeight = 4,
                NumClasses = 5,
                MaxSamples = 3
            };

            var loader = new PubLayNetDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(3, loader.TotalCount);
            Assert.Equal(5, loader.OutputDimension);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }
}
