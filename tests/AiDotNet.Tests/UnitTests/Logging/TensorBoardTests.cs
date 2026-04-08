using AiDotNet.Logging;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Logging;

// These tests share the "runs" directory and must not execute in parallel.
[CollectionDefinition("TensorBoard filesystem", DisableParallelization = true)]
public sealed class TensorBoardFilesystemCollection
{
}

/// <summary>
/// Unit tests for TensorBoard logging functionality.
/// </summary>
public class TensorBoardWriterTests : IDisposable
{
    private readonly string _testDir;

    public TensorBoardWriterTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"tensorboard_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            Directory.Delete(_testDir, true);
        }
    }

    [Fact]
    public void TensorBoardWriter_CreatesEventFile()
    {
        // Arrange & Act
        using (var writer = new TensorBoardWriter(_testDir))
        {
            writer.WriteScalar("test", 1.0f, 0);
        }

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void TensorBoardWriter_WriteScalar_CreatesValidRecord()
    {
        // Arrange & Act
        using (var writer = new TensorBoardWriter(_testDir))
        {
            writer.WriteScalar("loss/train", 0.5f, 100);
            writer.WriteScalar("loss/val", 0.6f, 100);
        }

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);

        var fileInfo = new FileInfo(files[0]);
        Assert.True(fileInfo.Length > 0, "Event file should not be empty");
    }

    [Fact]
    public void TensorBoardWriter_WriteScalars_GroupsMultipleValues()
    {
        // Arrange
        var scalars = new Dictionary<string, float>
        {
            { "train", 0.5f },
            { "val", 0.6f },
            { "test", 0.55f }
        };

        // Act
        using (var writer = new TensorBoardWriter(_testDir))
        {
            writer.WriteScalars("loss", scalars, 10);
        }

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void TensorBoardWriter_WriteHistogram_CreatesValidRecord()
    {
        // Arrange
        var values = Enumerable.Range(0, 1000)
            .Select(i => (float)Math.Sin(i * 0.01) + (float)RandomHelper.CreateSeededRandom(i).NextDouble())
            .ToArray();

        // Act
        using (var writer = new TensorBoardWriter(_testDir))
        {
            writer.WriteHistogram("weights/layer1", values, 0);
        }

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
        Assert.True(new FileInfo(files[0]).Length > 100);
    }

    [Fact]
    public void TensorBoardWriter_WriteImage_CreatesValidRecord()
    {
        // Arrange - Create a simple 10x10 red image
        int height = 10, width = 10, channels = 3;
        var pixels = new byte[height * width * channels];
        for (int i = 0; i < pixels.Length; i += 3)
        {
            pixels[i] = 255;     // Red
            pixels[i + 1] = 0;   // Green
            pixels[i + 2] = 0;   // Blue
        }

        // Act
        using (var writer = new TensorBoardWriter(_testDir))
        {
            writer.WriteImageRaw("test_image", pixels, height, width, channels, 0);
        }

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void TensorBoardWriter_WriteText_CreatesValidRecord()
    {
        // Act
        using (var writer = new TensorBoardWriter(_testDir))
        {
            writer.WriteText("notes", "This is a test note", 0);
            writer.WriteText("config", "learning_rate: 0.001\nbatch_size: 32", 0);
        }

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void TensorBoardWriter_WriteEmbedding_CreatesFiles()
    {
        // Arrange
        var embeddings = new float[100, 128];
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 128; j++)
            {
                embeddings[i, j] = (float)random.NextDouble();
            }
        }

        var metadata = Enumerable.Range(0, 100).Select(i => $"item_{i}").ToArray();

        // Act
        using (var writer = new TensorBoardWriter(_testDir))
        {
            writer.WriteEmbedding("embedding", embeddings, metadata, 0);
        }

        // Assert
        Assert.True(File.Exists(Path.Combine(_testDir, "embedding_embeddings.tsv")));
        Assert.True(File.Exists(Path.Combine(_testDir, "embedding_metadata.tsv")));
        Assert.True(File.Exists(Path.Combine(_testDir, "projector_config.pbtxt")));
    }

    [Fact]
    public void TensorBoardWriter_Flush_WritesToDisk()
    {
        // Arrange
        using var writer = new TensorBoardWriter(_testDir);

        // Act
        writer.WriteScalar("test", 1.0f, 0);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
        Assert.True(new FileInfo(files[0]).Length > 0);
    }

    [Fact]
    public void TensorBoardWriter_MultipleWrites_IncreasesFileSize()
    {
        // Arrange
        using var writer = new TensorBoardWriter(_testDir);
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);

        writer.Flush();
        long sizeAfterInit = new FileInfo(files[0]).Length;

        // Act
        for (int i = 0; i < 100; i++)
        {
            writer.WriteScalar("loss", (float)Math.Exp(-i * 0.1), i);
        }
        writer.Flush();

        // Assert
        long sizeAfterWrites = new FileInfo(files[0]).Length;
        Assert.True(sizeAfterWrites > sizeAfterInit, "File size should increase after writes");
    }
}

/// <summary>
/// Unit tests for SummaryWriter (PyTorch-compatible API).
/// </summary>
[Collection("TensorBoard filesystem")]
public class SummaryWriterTests : IDisposable
{
    private readonly string _testDir;

    public SummaryWriterTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"summary_test_{Guid.NewGuid():N}");
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            Directory.Delete(_testDir, true);
        }
    }

    [Fact]
    public void SummaryWriter_CreatesLogDirectory()
    {
        // Act
        using var writer = new SummaryWriter(_testDir);

        // Assert
        Assert.True(Directory.Exists(_testDir));
    }

    [Fact]
    public void SummaryWriter_DefaultLogDir_CreatesRunsDirectory()
    {
        // Arrange
        string logDir = string.Empty;

        // Act
        using (var writer = new SummaryWriter())
        {
            logDir = writer.LogDir;

            // Assert
            Assert.False(string.IsNullOrWhiteSpace(writer.LogDir));

            var trimmedLogDir = writer.LogDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            var parentDir = Path.GetDirectoryName(trimmedLogDir);
            Assert.Equal("runs", Path.GetFileName(parentDir));
        }

        // Cleanup - only after writer is disposed
        if (Directory.Exists(logDir))
        {
            Directory.Delete(logDir, true);
        }
    }

    [Fact]
    public void SummaryWriter_AddScalar_Works()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);

        // Act
        writer.AddScalar("loss", 0.5f, 0);
        writer.AddScalar("loss", 0.4f, 1);
        writer.AddScalar("loss", 0.3f, 2);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void SummaryWriter_AddScalar_AutoIncrementsStep()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);

        // Act
        writer.AddScalar("metric1", 1.0f);
        writer.AddScalar("metric2", 2.0f);
        writer.AddScalar("metric3", 3.0f);

        // Assert - default step should have incremented
        Assert.Equal(3, writer.DefaultStep);
    }

    [Fact]
    public void SummaryWriter_AddScalars_GroupsMetrics()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);
        var metrics = new Dictionary<string, float>
        {
            { "train", 0.5f },
            { "val", 0.6f }
        };

        // Act
        writer.AddScalars("loss", metrics, 0);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void SummaryWriter_AddHistogram_FromArray()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);
        var weights = new float[1000];
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (float)random.NextGaussian();
        }

        // Act
        writer.AddHistogram("layer1/weights", weights, 0);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void SummaryWriter_AddHistogram_From2DArray()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);
        var matrix = new float[32, 64];
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 64; j++)
            {
                matrix[i, j] = (float)random.NextGaussian();
            }
        }

        // Act
        writer.AddHistogram("layer1/weights", matrix, 0);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void SummaryWriter_AddImage_FromFloatArray()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);
        var image = new float[3, 28, 28]; // CHW format
        var random = RandomHelper.CreateSeededRandom(42);
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < 28; h++)
            {
                for (int w = 0; w < 28; w++)
                {
                    image[c, h, w] = (float)random.NextDouble();
                }
            }
        }

        // Act
        writer.AddImage("sample", image, 0);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void SummaryWriter_AddImages_CreatesGrid()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);
        var images = new float[16, 1, 8, 8]; // 16 grayscale 8x8 images
        var random = RandomHelper.CreateSeededRandom(42);
        for (int n = 0; n < 16; n++)
        {
            for (int h = 0; h < 8; h++)
            {
                for (int w = 0; w < 8; w++)
                {
                    images[n, 0, h, w] = (float)random.NextDouble();
                }
            }
        }

        // Act
        writer.AddImages("samples", images, 0, nrow: 4);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void SummaryWriter_AddText_Works()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);

        // Act
        writer.AddText("experiment/notes", "Testing TensorBoard integration", 0);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void SummaryWriter_AddHparams_LogsConfig()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);
        var hparams = new Dictionary<string, object>
        {
            { "learning_rate", 0.001 },
            { "batch_size", 32 },
            { "optimizer", "Adam" }
        };
        var metrics = new Dictionary<string, float>
        {
            { "final_loss", 0.1f },
            { "final_accuracy", 0.95f }
        };

        // Act
        writer.AddHparams(hparams, metrics);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void SummaryWriter_AddEmbedding_CreatesFiles()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);
        var embeddings = new float[50, 64];
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < 50; i++)
        {
            for (int j = 0; j < 64; j++)
            {
                embeddings[i, j] = (float)random.NextDouble();
            }
        }
        var labels = Enumerable.Range(0, 50).Select(i => $"class_{i % 5}").ToArray();

        // Act
        writer.AddEmbedding("word_vectors", embeddings, labels, step: 0);
        writer.Flush();

        // Assert
        Assert.True(File.Exists(Path.Combine(_testDir, "word_vectors_embeddings.tsv")));
        Assert.True(File.Exists(Path.Combine(_testDir, "word_vectors_metadata.tsv")));
    }

    [Fact]
    public void SummaryWriter_AddPrCurve_Works()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);
        var random = RandomHelper.CreateSeededRandom(42);
        var labels = Enumerable.Range(0, 100).Select(_ => random.Next(2)).ToArray();
        var predictions = labels.Select(l => (float)(l * 0.7 + random.NextDouble() * 0.3)).ToArray();

        // Act
        writer.AddPrCurve("classifier/pr", labels, predictions, 0);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void SummaryWriter_LogTrainingStep_LogsAllMetrics()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);

        // Act
        for (int i = 0; i < 10; i++)
        {
            writer.LogTrainingStep(
                loss: (float)Math.Exp(-i * 0.1),
                accuracy: 0.5f + i * 0.05f,
                learningRate: 0.001f * (float)Math.Pow(0.95, i),
                step: i);
        }
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void SummaryWriter_LogValidationStep_Works()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);

        // Act
        writer.LogValidationStep(0.4f, 0.85f, 100);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void SummaryWriter_LogWeights_LogsStatistics()
    {
        // Arrange
        using var writer = new SummaryWriter(_testDir);
        var weights = new float[1000];
        var gradients = new float[1000];
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < 1000; i++)
        {
            weights[i] = (float)random.NextGaussian() * 0.1f;
            gradients[i] = (float)random.NextGaussian() * 0.01f;
        }

        // Act
        writer.LogWeights("dense1", weights, gradients, 0);
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(_testDir, "events.out.tfevents.*");
        Assert.Single(files);
    }
}

/// <summary>
/// Tests for TensorBoardTrainingContext.
/// </summary>
[Collection("TensorBoard filesystem")]
public class TensorBoardTrainingContextTests : IDisposable
{
    private readonly string _testDir;

    public TensorBoardTrainingContextTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"tb_context_test_{Guid.NewGuid():N}");
    }

    public void Dispose()
    {
        // Cleanup runs directory
        var runsDir = Path.Combine(Directory.GetCurrentDirectory(), "runs");
        if (Directory.Exists(runsDir))
        {
            try
            {
                Directory.Delete(runsDir, true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    [Fact]
    public void TensorBoardTrainingContext_CreatesFiles()
    {
        // Arrange & Act
        string logDir;
        using (var ctx = new TensorBoardTrainingContext("test_experiment", "run_1"))
        {
            logDir = ctx.Writer.LogDir;
            ctx.LogTrainStep(1.0f, 0.5f, 0.001f);
            ctx.LogTrainStep(0.8f, 0.6f, 0.001f);
            ctx.LogValStep(0.9f, 0.55f);
        }

        // Assert
        Assert.True(Directory.Exists(logDir));
        var files = Directory.GetFiles(logDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void TensorBoardTrainingContext_TracksGlobalStep()
    {
        // Arrange
        using var ctx = new TensorBoardTrainingContext("test_experiment", "run_2");

        // Act
        Assert.Equal(0, ctx.GlobalStep);
        ctx.LogTrainStep(1.0f);
        Assert.Equal(1, ctx.GlobalStep);
        ctx.LogTrainStep(0.9f);
        Assert.Equal(2, ctx.GlobalStep);
    }

    [Fact]
    public void TensorBoardTrainingContext_LogsHparams()
    {
        // Arrange
        var hparams = new Dictionary<string, object>
        {
            { "lr", 0.001 },
            { "batch_size", 32 }
        };

        // Act
        using var ctx = new TensorBoardTrainingContext("test_experiment", "run_3", hparams);
        ctx.LogTrainStep(1.0f);

        // Assert
        var files = Directory.GetFiles(ctx.Writer.LogDir, "events.out.tfevents.*");
        Assert.Single(files);
    }

    [Fact]
    public void TensorBoardTrainingContext_LogsElapsedTime()
    {
        // Arrange
        using var ctx = new TensorBoardTrainingContext("test_experiment", "run_4");

        // Act
        Thread.Sleep(10); // Small delay
        ctx.LogElapsedTime();

        // Assert
        Assert.True(ctx.Elapsed.TotalMilliseconds >= 10);
    }

    [Fact]
    public void TensorBoardTrainingContext_LogsModelWeights()
    {
        // Arrange
        using var ctx = new TensorBoardTrainingContext("test_experiment", "run_5");
        var weights = new Dictionary<string, float[]>
        {
            { "layer1", Enumerable.Range(0, 100).Select(i => (float)i / 100).ToArray() },
            { "layer2", Enumerable.Range(0, 200).Select(i => (float)i / 200).ToArray() }
        };

        // Act
        ctx.LogModelWeights(weights);

        // Assert
        var files = Directory.GetFiles(ctx.Writer.LogDir, "events.out.tfevents.*");
        Assert.Single(files);
    }
}

/// <summary>
/// Tests for TensorBoard extension methods.
/// </summary>
[Collection("TensorBoard filesystem")]
public class TensorBoardExtensionsTests : IDisposable
{
    public void Dispose()
    {
        // Cleanup runs directory
        var runsDir = Path.Combine(Directory.GetCurrentDirectory(), "runs");
        if (Directory.Exists(runsDir))
        {
            try
            {
                Directory.Delete(runsDir, true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    [Fact]
    public void CreateTensorBoardWriter_CreatesInRunsDirectory()
    {
        // Act
        using var writer = TensorBoardExtensions.CreateTensorBoardWriter("my_experiment", "run_1");

        // Assert
        Assert.Contains("runs", writer.LogDir);
        Assert.Contains("my_experiment", writer.LogDir);
        Assert.Contains("run_1", writer.LogDir);
    }

    [Fact]
    public void LogMetrics_WritesAllMetrics()
    {
        // Arrange
        using var writer = TensorBoardExtensions.CreateTensorBoardWriter("metrics_test");
        var metrics = new Dictionary<string, float>
        {
            { "loss", 0.5f },
            { "accuracy", 0.85f },
            { "f1_score", 0.82f }
        };

        // Act
        writer.LogMetrics(metrics, step: 100, prefix: "eval");
        writer.Flush();

        // Assert
        var files = Directory.GetFiles(writer.LogDir, "events.out.tfevents.*");
        Assert.Single(files);
    }
}

/// <summary>
/// Helper extension for generating Gaussian random numbers.
/// </summary>
internal static class RandomExtensions
{
    public static double NextGaussian(this Random random, double mean = 0, double stdDev = 1)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }
}
