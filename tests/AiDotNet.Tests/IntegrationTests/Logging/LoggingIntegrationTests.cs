using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Logging;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Logging;

/// <summary>
/// Integration tests for the Logging module.
/// These tests verify TensorBoard logging functionality including scalar, histogram, image, and text logging.
/// </summary>
public class LoggingIntegrationTests : IDisposable
{
    private readonly string _testDir;

    public LoggingIntegrationTests()
    {
        // Create a unique temp directory for each test
        _testDir = Path.Combine(Path.GetTempPath(), $"AiDotNet_LoggingTests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        // Clean up test directory
        if (Directory.Exists(_testDir))
        {
            try
            {
                Directory.Delete(_testDir, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    #region SummaryWriter Constructor Tests

    [Fact]
    public void SummaryWriter_Constructor_WithLogDir_CreatesDirectory()
    {
        var logDir = Path.Combine(_testDir, "test_run");

        using var writer = new SummaryWriter(logDir);

        Assert.True(Directory.Exists(logDir));
        Assert.Equal(logDir, writer.LogDir);
    }

    [Fact]
    public void SummaryWriter_Constructor_WithNullLogDir_GeneratesAutoDirectory()
    {
        string? logDir = null;
        SummaryWriter? writer = null;
        try
        {
            // The auto-generated directory is in "runs" folder
            writer = new SummaryWriter();
            logDir = writer.LogDir;
            Assert.NotNull(writer.LogDir);
            // Check the parent directory name is "runs" (path-agnostic, works with absolute paths)
            var parentDirName = Path.GetFileName(Path.GetDirectoryName(writer.LogDir));
            Assert.Equal("runs", parentDirName);
            Assert.True(Directory.Exists(writer.LogDir));
        }
        finally
        {
            // Dispose writer first to release file handles
            writer?.Dispose();
            // Then clean up directory
            CleanupRunsDirectory(logDir);
        }
    }

    [Fact]
    public void SummaryWriter_Constructor_WithComment_IncludesInDirectory()
    {
        string? logDir = null;
        SummaryWriter? writer = null;
        try
        {
            writer = new SummaryWriter(comment: "test_experiment");
            logDir = writer.LogDir;
            Assert.Contains("test_experiment", writer.LogDir);
        }
        finally
        {
            // Dispose writer first to release file handles
            writer?.Dispose();
            // Then clean up directory
            CleanupRunsDirectory(logDir);
        }
    }

    private static void CleanupRunsDirectory(string? logDir)
    {
        if (logDir == null) return;
        try
        {
            if (Directory.Exists(logDir))
            {
                Directory.Delete(logDir, true);
            }
            // Try to clean up parent directories up to "runs"
            var parent = Path.GetDirectoryName(logDir);
            while (parent != null && parent.Contains("runs") && Directory.Exists(parent))
            {
                try
                {
                    if (!Directory.EnumerateFileSystemEntries(parent).Any())
                    {
                        Directory.Delete(parent);
                        parent = Path.GetDirectoryName(parent);
                    }
                    else
                    {
                        break;
                    }
                }
                catch
                {
                    break;
                }
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
    }

    [Fact]
    public void SummaryWriter_DefaultStep_InitializesToZero()
    {
        var logDir = Path.Combine(_testDir, "test_run");
        using var writer = new SummaryWriter(logDir);

        Assert.Equal(0, writer.DefaultStep);
    }

    #endregion

    #region SummaryWriter AddScalar Tests

    [Fact]
    public void SummaryWriter_AddScalar_Float_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "scalar_test");
        using var writer = new SummaryWriter(logDir);

        writer.AddScalar("test/loss", 0.5f, 0);

        writer.Flush();
        Assert.True(Directory.GetFiles(logDir, "events.*").Length > 0);
    }

    [Fact]
    public void SummaryWriter_AddScalar_Double_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "scalar_test_double");
        using var writer = new SummaryWriter(logDir);

        writer.AddScalar("test/accuracy", 0.95, 0);

        writer.Flush();
        Assert.True(Directory.GetFiles(logDir, "events.*").Length > 0);
    }

    [Fact]
    public void SummaryWriter_AddScalar_WithoutStep_AutoIncrements()
    {
        var logDir = Path.Combine(_testDir, "scalar_auto_step");
        using var writer = new SummaryWriter(logDir);

        Assert.Equal(0, writer.DefaultStep);

        writer.AddScalar("test/loss", 0.5f);
        Assert.Equal(1, writer.DefaultStep);

        writer.AddScalar("test/loss", 0.4f);
        Assert.Equal(2, writer.DefaultStep);

        writer.AddScalar("test/loss", 0.3f);
        Assert.Equal(3, writer.DefaultStep);
    }

    [Theory]
    [InlineData(0.0f)]
    [InlineData(1.0f)]
    [InlineData(-1.0f)]
    [InlineData(float.MaxValue)]
    [InlineData(float.MinValue)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void SummaryWriter_AddScalar_VariousFloatValues_HandlesCorrectly(float value)
    {
        var logDir = Path.Combine(_testDir, $"scalar_value_{value.GetHashCode()}");
        using var writer = new SummaryWriter(logDir);

        // Should not throw
        writer.AddScalar("test/value", value, 0);
        writer.Flush();
    }

    #endregion

    #region SummaryWriter AddScalars Tests

    [Fact]
    public void SummaryWriter_AddScalars_MultipleTags_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "scalars_test");
        using var writer = new SummaryWriter(logDir);

        var values = new Dictionary<string, float>
        {
            { "train", 0.5f },
            { "val", 0.6f },
            { "test", 0.55f }
        };

        writer.AddScalars("loss", values, 0);

        writer.Flush();
        Assert.True(Directory.GetFiles(logDir, "events.*").Length > 0);
    }

    [Fact]
    public void SummaryWriter_AddScalars_EmptyDictionary_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "scalars_empty");
        using var writer = new SummaryWriter(logDir);

        writer.AddScalars("loss", new Dictionary<string, float>(), 0);

        writer.Flush();
    }

    #endregion

    #region SummaryWriter AddHistogram Tests

    [Fact]
    public void SummaryWriter_AddHistogram_FloatArray_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "histogram_test");
        using var writer = new SummaryWriter(logDir);

        var values = new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
        writer.AddHistogram("weights/layer1", values, 0);

        writer.Flush();
        Assert.True(Directory.GetFiles(logDir, "events.*").Length > 0);
    }

    [Fact]
    public void SummaryWriter_AddHistogram_2DArray_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "histogram_2d_test");
        using var writer = new SummaryWriter(logDir);

        var values = new float[,]
        {
            { 0.1f, 0.2f, 0.3f },
            { 0.4f, 0.5f, 0.6f },
            { 0.7f, 0.8f, 0.9f }
        };
        writer.AddHistogram("weights/layer2", values, 0);

        writer.Flush();
        Assert.True(Directory.GetFiles(logDir, "events.*").Length > 0);
    }

    [Fact]
    public void SummaryWriter_AddHistogram_LargeArray_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "histogram_large");
        using var writer = new SummaryWriter(logDir);

        var random = new Random(42);
        var values = Enumerable.Range(0, 10000).Select(_ => (float)random.NextDouble()).ToArray();
        writer.AddHistogram("weights/large_layer", values, 0);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_AddHistogram_NegativeValues_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "histogram_negative");
        using var writer = new SummaryWriter(logDir);

        var values = new float[] { -1.0f, -0.5f, 0.0f, 0.5f, 1.0f };
        writer.AddHistogram("weights/mixed", values, 0);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_AddHistogram_SingleValue_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "histogram_single");
        using var writer = new SummaryWriter(logDir);

        var values = new float[] { 0.5f };
        writer.AddHistogram("weights/single", values, 0);

        writer.Flush();
    }

    #endregion

    #region SummaryWriter AddImage Tests

    [Fact]
    public void SummaryWriter_AddImage_CHW_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "image_chw_test");
        using var writer = new SummaryWriter(logDir);

        // 3 channels, 4 height, 4 width
        var imageData = new float[3, 4, 4];
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < 4; h++)
            {
                for (int w = 0; w < 4; w++)
                {
                    imageData[c, h, w] = (float)(c + h + w) / 10f;
                }
            }
        }

        writer.AddImage("test/image", imageData, 0, "CHW");

        writer.Flush();
        Assert.True(Directory.GetFiles(logDir, "events.*").Length > 0);
    }

    [Fact]
    public void SummaryWriter_AddImage_HWC_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "image_hwc_test");
        using var writer = new SummaryWriter(logDir);

        // 4 height, 4 width, 3 channels
        var imageData = new float[4, 4, 3];
        for (int h = 0; h < 4; h++)
        {
            for (int w = 0; w < 4; w++)
            {
                for (int c = 0; c < 3; c++)
                {
                    imageData[h, w, c] = (float)(c + h + w) / 10f;
                }
            }
        }

        writer.AddImage("test/image_hwc", imageData, 0, "HWC");

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_AddImageRaw_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "image_raw_test");
        using var writer = new SummaryWriter(logDir);

        // 4x4 RGB image
        var pixels = new byte[4 * 4 * 3];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = (byte)(i % 256);
        }

        writer.AddImageRaw("test/raw_image", pixels, 4, 4, 3, 0);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_AddImages_Grid_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "images_grid_test");
        using var writer = new SummaryWriter(logDir);

        // 4 images, 3 channels, 8x8 each
        var images = new float[4, 3, 8, 8];
        for (int n = 0; n < 4; n++)
        {
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < 8; h++)
                {
                    for (int w = 0; w < 8; w++)
                    {
                        images[n, c, h, w] = (float)(n + c + h + w) / 20f;
                    }
                }
            }
        }

        writer.AddImages("test/image_grid", images, 0, nrow: 2);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_AddImages_Normalized_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "images_normalized");
        using var writer = new SummaryWriter(logDir);

        var images = new float[2, 3, 4, 4];
        // Use values outside [0, 1] to test normalization
        for (int n = 0; n < 2; n++)
        {
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < 4; h++)
                {
                    for (int w = 0; w < 4; w++)
                    {
                        images[n, c, h, w] = (float)(n - 1) * 2f; // Values in [-2, 0]
                    }
                }
            }
        }

        writer.AddImages("test/normalized", images, 0, normalize: true);

        writer.Flush();
    }

    #endregion

    #region SummaryWriter AddText Tests

    [Fact]
    public void SummaryWriter_AddText_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "text_test");
        using var writer = new SummaryWriter(logDir);

        writer.AddText("test/description", "This is a test description.", 0);

        writer.Flush();
        Assert.True(Directory.GetFiles(logDir, "events.*").Length > 0);
    }

    [Fact]
    public void SummaryWriter_AddText_MultiLine_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "text_multiline");
        using var writer = new SummaryWriter(logDir);

        var text = "Line 1\nLine 2\nLine 3\n\nParagraph 2";
        writer.AddText("test/multiline", text, 0);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_AddText_EmptyString_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "text_empty");
        using var writer = new SummaryWriter(logDir);

        writer.AddText("test/empty", "", 0);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_AddText_UnicodeCharacters_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "text_unicode");
        using var writer = new SummaryWriter(logDir);

        writer.AddText("test/unicode", "Unicode: \u00e9\u00e8\u00ea \u4e2d\u6587 \u0410\u0411\u0412", 0);

        writer.Flush();
    }

    #endregion

    #region SummaryWriter AddHparams Tests

    [Fact]
    public void SummaryWriter_AddHparams_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "hparams_test");
        using var writer = new SummaryWriter(logDir);

        var hparams = new Dictionary<string, object>
        {
            { "learning_rate", 0.001 },
            { "batch_size", 32 },
            { "optimizer", "Adam" }
        };

        var metrics = new Dictionary<string, float>
        {
            { "accuracy", 0.95f },
            { "loss", 0.05f }
        };

        writer.AddHparams(hparams, metrics);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_AddHparams_EmptyHparams_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "hparams_empty");
        using var writer = new SummaryWriter(logDir);

        writer.AddHparams(
            new Dictionary<string, object>(),
            new Dictionary<string, float>());

        writer.Flush();
    }

    #endregion

    #region SummaryWriter AddEmbedding Tests

    [Fact]
    public void SummaryWriter_AddEmbedding_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "embedding_test");
        using var writer = new SummaryWriter(logDir);

        // 5 samples, 3 dimensions
        var embeddings = new float[5, 3]
        {
            { 1.0f, 0.0f, 0.0f },
            { 0.0f, 1.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f },
            { 0.5f, 0.5f, 0.0f },
            { 0.0f, 0.5f, 0.5f }
        };

        // Use a simple tag without slashes to avoid path issues
        writer.AddEmbedding("embeddings", embeddings, step: 0);

        writer.Flush();

        // Check that TSV file was created
        Assert.True(File.Exists(Path.Combine(logDir, "embeddings_embeddings.tsv")));
    }

    [Fact]
    public void SummaryWriter_AddEmbedding_WithMetadata_WritesMetadataFile()
    {
        var logDir = Path.Combine(_testDir, "embedding_metadata");
        using var writer = new SummaryWriter(logDir);

        var embeddings = new float[3, 2]
        {
            { 1.0f, 0.0f },
            { 0.0f, 1.0f },
            { 0.5f, 0.5f }
        };

        var metadata = new[] { "class_a", "class_b", "class_c" };

        // Use a simple tag without slashes to avoid path issues
        writer.AddEmbedding("embeddings", embeddings, metadata, step: 0);

        writer.Flush();

        // Check that metadata file was created
        Assert.True(File.Exists(Path.Combine(logDir, "embeddings_metadata.tsv")));
    }

    #endregion

    #region SummaryWriter AddPrCurve Tests

    [Fact]
    public void SummaryWriter_AddPrCurve_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "pr_curve_test");
        using var writer = new SummaryWriter(logDir);

        var labels = new[] { 1, 0, 1, 1, 0, 0, 1, 0 };
        var predictions = new[] { 0.9f, 0.1f, 0.8f, 0.7f, 0.2f, 0.3f, 0.6f, 0.4f };

        writer.AddPrCurve("test/pr_curve", labels, predictions, step: 0);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_AddPrCurve_AllPositive_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "pr_curve_all_pos");
        using var writer = new SummaryWriter(logDir);

        var labels = new[] { 1, 1, 1, 1 };
        var predictions = new[] { 0.9f, 0.8f, 0.7f, 0.6f };

        writer.AddPrCurve("test/all_positive", labels, predictions, step: 0);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_AddPrCurve_AllNegative_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "pr_curve_all_neg");
        using var writer = new SummaryWriter(logDir);

        var labels = new[] { 0, 0, 0, 0 };
        var predictions = new[] { 0.1f, 0.2f, 0.3f, 0.4f };

        writer.AddPrCurve("test/all_negative", labels, predictions, step: 0);

        writer.Flush();
    }

    #endregion

    #region SummaryWriter LogTrainingStep Tests

    [Fact]
    public void SummaryWriter_LogTrainingStep_WritesLoss()
    {
        var logDir = Path.Combine(_testDir, "training_step");
        using var writer = new SummaryWriter(logDir);

        writer.LogTrainingStep(0.5f, step: 0);

        writer.Flush();
        Assert.True(Directory.GetFiles(logDir, "events.*").Length > 0);
    }

    [Fact]
    public void SummaryWriter_LogTrainingStep_WithAccuracy_WritesBoth()
    {
        var logDir = Path.Combine(_testDir, "training_step_acc");
        using var writer = new SummaryWriter(logDir);

        writer.LogTrainingStep(0.5f, accuracy: 0.85f, step: 0);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_LogTrainingStep_WithLearningRate_WritesAll()
    {
        var logDir = Path.Combine(_testDir, "training_step_lr");
        using var writer = new SummaryWriter(logDir);

        writer.LogTrainingStep(0.5f, accuracy: 0.85f, learningRate: 0.001f, step: 0);

        writer.Flush();
    }

    #endregion

    #region SummaryWriter LogValidationStep Tests

    [Fact]
    public void SummaryWriter_LogValidationStep_WritesLoss()
    {
        var logDir = Path.Combine(_testDir, "validation_step");
        using var writer = new SummaryWriter(logDir);

        writer.LogValidationStep(0.6f, step: 0);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_LogValidationStep_WithAccuracy_WritesBoth()
    {
        var logDir = Path.Combine(_testDir, "validation_step_acc");
        using var writer = new SummaryWriter(logDir);

        writer.LogValidationStep(0.6f, accuracy: 0.82f, step: 0);

        writer.Flush();
    }

    #endregion

    #region SummaryWriter LogWeights Tests

    [Fact]
    public void SummaryWriter_LogWeights_WritesHistogram()
    {
        var logDir = Path.Combine(_testDir, "log_weights");
        using var writer = new SummaryWriter(logDir);

        var weights = new float[] { 0.1f, 0.2f, -0.1f, 0.3f, -0.2f };
        writer.LogWeights("dense_1", weights, step: 0);

        writer.Flush();
    }

    [Fact]
    public void SummaryWriter_LogWeights_WithGradients_WritesBoth()
    {
        var logDir = Path.Combine(_testDir, "log_weights_grads");
        using var writer = new SummaryWriter(logDir);

        var weights = new float[] { 0.1f, 0.2f, -0.1f, 0.3f, -0.2f };
        var gradients = new float[] { 0.01f, 0.02f, -0.01f, 0.03f, -0.02f };
        writer.LogWeights("dense_1", weights, gradients, step: 0);

        writer.Flush();
    }

    #endregion

    #region SummaryWriter Flush and Dispose Tests

    [Fact]
    public void SummaryWriter_Flush_PersistsData()
    {
        var logDir = Path.Combine(_testDir, "flush_test");

        using (var writer = new SummaryWriter(logDir))
        {
            writer.AddScalar("test/value", 1.0f, 0);
            writer.Flush();
        }

        // After dispose, files should exist
        Assert.True(Directory.GetFiles(logDir, "events.*").Length > 0);
    }

    [Fact]
    public void SummaryWriter_Close_IsAliasForDispose()
    {
        var logDir = Path.Combine(_testDir, "close_test");
        var writer = new SummaryWriter(logDir);
        writer.AddScalar("test/value", 1.0f, 0);

        // Close is an alias for Dispose
        writer.Close();

        // Should not throw on second close/dispose
        writer.Dispose();
    }

    [Fact]
    public void SummaryWriter_MultipleDispose_DoesNotThrow()
    {
        var logDir = Path.Combine(_testDir, "multi_dispose");
        var writer = new SummaryWriter(logDir);

        writer.Dispose();
        writer.Dispose();
        writer.Dispose();
    }

    #endregion

    #region TensorBoardWriter Tests

    [Fact]
    public void TensorBoardWriter_Constructor_CreatesEventFile()
    {
        var logDir = Path.Combine(_testDir, "tb_writer_test");

        using var writer = new TensorBoardWriter(logDir);

        Assert.True(Directory.Exists(logDir));
        Assert.True(File.Exists(writer.FilePath));
        Assert.Equal(logDir, writer.LogDir);
    }

    [Fact]
    public void TensorBoardWriter_Constructor_WithNullLogDir_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new TensorBoardWriter(null!));
    }

    [Fact]
    public void TensorBoardWriter_WriteScalar_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "tb_scalar");
        using var writer = new TensorBoardWriter(logDir);

        writer.WriteScalar("test", 0.5f, 0);
        writer.Flush();
    }

    [Fact]
    public void TensorBoardWriter_WriteScalars_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "tb_scalars");
        using var writer = new TensorBoardWriter(logDir);

        writer.WriteScalars("metrics", new Dictionary<string, float>
        {
            { "loss", 0.5f },
            { "acc", 0.9f }
        }, 0);
        writer.Flush();
    }

    [Fact]
    public void TensorBoardWriter_WriteHistogram_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "tb_histogram");
        using var writer = new TensorBoardWriter(logDir);

        writer.WriteHistogram("weights", new float[] { 0.1f, 0.2f, 0.3f }, 0);
        writer.Flush();
    }

    [Fact]
    public void TensorBoardWriter_WriteHistogram_EmptyArray_DoesNotWrite()
    {
        var logDir = Path.Combine(_testDir, "tb_histogram_empty");
        using var writer = new TensorBoardWriter(logDir);

        // Empty array should not cause error
        writer.WriteHistogram("weights", Array.Empty<float>(), 0);
        writer.Flush();
    }

    [Fact]
    public void TensorBoardWriter_WriteText_WritesWithoutError()
    {
        var logDir = Path.Combine(_testDir, "tb_text");
        using var writer = new TensorBoardWriter(logDir);

        writer.WriteText("description", "Test text", 0);
        writer.Flush();
    }

    [Fact]
    public void TensorBoardWriter_FilePath_ContainsCorrectFormat()
    {
        var logDir = Path.Combine(_testDir, "tb_filepath");
        using var writer = new TensorBoardWriter(logDir);

        Assert.Contains("events.out.tfevents", writer.FilePath);
    }

    #endregion

    #region TensorBoardExtensions Tests

    [Fact]
    public void TensorBoardExtensions_CreateTensorBoardWriter_CreatesWriter()
    {
        string? logDir = null;
        SummaryWriter? writer = null;
        try
        {
            writer = TensorBoardExtensions.CreateTensorBoardWriter("test_experiment");
            logDir = writer.LogDir;

            Assert.NotNull(writer);
            Assert.Contains("test_experiment", writer.LogDir);
        }
        finally
        {
            // Dispose writer first to release file handles
            writer?.Dispose();
            // Then clean up directory
            CleanupRunsDirectory(logDir);
        }
    }

    [Fact]
    public void TensorBoardExtensions_CreateTensorBoardWriter_WithRunName_IncludesInPath()
    {
        string? logDir = null;
        SummaryWriter? writer = null;
        try
        {
            writer = TensorBoardExtensions.CreateTensorBoardWriter("test_experiment", "run_001");
            logDir = writer.LogDir;

            Assert.Contains("test_experiment", writer.LogDir);
            Assert.Contains("run_001", writer.LogDir);
        }
        finally
        {
            // Dispose writer first to release file handles
            writer?.Dispose();
            // Then clean up directory
            CleanupRunsDirectory(logDir);
        }
    }

    [Fact]
    public void TensorBoardExtensions_LogMetrics_WritesAllMetrics()
    {
        var logDir = Path.Combine(_testDir, "log_metrics");
        using var writer = new SummaryWriter(logDir);

        var metrics = new Dictionary<string, float>
        {
            { "loss", 0.5f },
            { "accuracy", 0.9f },
            { "f1_score", 0.85f }
        };

        writer.LogMetrics(metrics, 0, "train");

        writer.Flush();
    }

    [Fact]
    public void TensorBoardExtensions_LogMetrics_WithoutPrefix_WritesWithoutPrefix()
    {
        var logDir = Path.Combine(_testDir, "log_metrics_no_prefix");
        using var writer = new SummaryWriter(logDir);

        var metrics = new Dictionary<string, float>
        {
            { "loss", 0.5f }
        };

        writer.LogMetrics(metrics, 0);

        writer.Flush();
    }

    #endregion

    #region TensorBoardTrainingContext Tests

    [Fact]
    public void TensorBoardTrainingContext_Constructor_CreatesWriter()
    {
        string? logDir = null;
        TensorBoardTrainingContext? context = null;
        try
        {
            context = new TensorBoardTrainingContext("test_experiment");
            logDir = context.Writer.LogDir;

            Assert.NotNull(context.Writer);
            Assert.Equal(0, context.GlobalStep);
        }
        finally
        {
            context?.Dispose();
            CleanupRunsDirectory(logDir);
        }
    }

    [Fact]
    public void TensorBoardTrainingContext_LogTrainStep_IncrementsGlobalStep()
    {
        string? logDir = null;
        TensorBoardTrainingContext? context = null;
        try
        {
            context = new TensorBoardTrainingContext("step_test");
            logDir = context.Writer.LogDir;

            Assert.Equal(0, context.GlobalStep);

            context.LogTrainStep(0.5f);
            Assert.Equal(1, context.GlobalStep);

            context.LogTrainStep(0.4f);
            Assert.Equal(2, context.GlobalStep);
        }
        finally
        {
            context?.Dispose();
            CleanupRunsDirectory(logDir);
        }
    }

    [Fact]
    public void TensorBoardTrainingContext_LogValStep_DoesNotIncrementGlobalStep()
    {
        string? logDir = null;
        TensorBoardTrainingContext? context = null;
        try
        {
            context = new TensorBoardTrainingContext("val_step_test");
            logDir = context.Writer.LogDir;

            context.LogTrainStep(0.5f); // Step becomes 1
            Assert.Equal(1, context.GlobalStep);

            context.LogValStep(0.6f); // Step should stay 1
            Assert.Equal(1, context.GlobalStep);
        }
        finally
        {
            context?.Dispose();
            CleanupRunsDirectory(logDir);
        }
    }

    [Fact]
    public void TensorBoardTrainingContext_GlobalStep_CanBeSet()
    {
        string? logDir = null;
        TensorBoardTrainingContext? context = null;
        try
        {
            context = new TensorBoardTrainingContext("global_step_set");
            logDir = context.Writer.LogDir;

            context.GlobalStep = 100;
            Assert.Equal(100, context.GlobalStep);
        }
        finally
        {
            context?.Dispose();
            CleanupRunsDirectory(logDir);
        }
    }

    [Fact]
    public void TensorBoardTrainingContext_Elapsed_ReturnsPositiveTimeSpan()
    {
        string? logDir = null;
        TensorBoardTrainingContext? context = null;
        try
        {
            context = new TensorBoardTrainingContext("elapsed_test");
            logDir = context.Writer.LogDir;

            // Small delay
            System.Threading.Thread.Sleep(10);

            Assert.True(context.Elapsed.TotalMilliseconds >= 0);
        }
        finally
        {
            context?.Dispose();
            CleanupRunsDirectory(logDir);
        }
    }

    [Fact]
    public void TensorBoardTrainingContext_LogModelWeights_WritesWeights()
    {
        string? logDir = null;
        TensorBoardTrainingContext? context = null;
        try
        {
            context = new TensorBoardTrainingContext("model_weights");
            logDir = context.Writer.LogDir;

            var weights = new Dictionary<string, float[]>
            {
                { "layer1", new float[] { 0.1f, 0.2f, 0.3f } },
                { "layer2", new float[] { 0.4f, 0.5f, 0.6f } }
            };

            context.LogModelWeights(weights);
        }
        finally
        {
            context?.Dispose();
            CleanupRunsDirectory(logDir);
        }
    }

    [Fact]
    public void TensorBoardTrainingContext_LogModelWeights_WithGradients_WritesBoth()
    {
        string? logDir = null;
        TensorBoardTrainingContext? context = null;
        try
        {
            context = new TensorBoardTrainingContext("model_weights_grads");
            logDir = context.Writer.LogDir;

            var weights = new Dictionary<string, float[]>
            {
                { "layer1", new float[] { 0.1f, 0.2f, 0.3f } }
            };

            var gradients = new Dictionary<string, float[]>
            {
                { "layer1", new float[] { 0.01f, 0.02f, 0.03f } }
            };

            context.LogModelWeights(weights, gradients);
        }
        finally
        {
            context?.Dispose();
            CleanupRunsDirectory(logDir);
        }
    }

    [Fact]
    public void TensorBoardTrainingContext_LogElapsedTime_WritesScalar()
    {
        string? logDir = null;
        TensorBoardTrainingContext? context = null;
        try
        {
            context = new TensorBoardTrainingContext("elapsed_log");
            logDir = context.Writer.LogDir;

            context.LogElapsedTime();
        }
        finally
        {
            context?.Dispose();
            CleanupRunsDirectory(logDir);
        }
    }

    [Fact]
    public void TensorBoardTrainingContext_WithHparams_LogsHparams()
    {
        string? logDir = null;
        TensorBoardTrainingContext? context = null;
        try
        {
            var hparams = new Dictionary<string, object>
            {
                { "learning_rate", 0.001 },
                { "batch_size", 32 }
            };

            context = new TensorBoardTrainingContext("hparams_context", hparams: hparams);
            logDir = context.Writer.LogDir;

            Assert.NotNull(context.Writer);
        }
        finally
        {
            context?.Dispose();
            CleanupRunsDirectory(logDir);
        }
    }

    #endregion

    #region Integration Scenario Tests

    [Fact]
    public void SummaryWriter_FullTrainingLoop_WritesAllMetrics()
    {
        var logDir = Path.Combine(_testDir, "full_training_loop");
        using var writer = new SummaryWriter(logDir);

        var random = new Random(42);

        // Simulate training loop
        for (int epoch = 0; epoch < 5; epoch++)
        {
            float trainLoss = 1.0f - (epoch * 0.1f) + (float)random.NextDouble() * 0.1f;
            float trainAcc = 0.5f + (epoch * 0.08f) + (float)random.NextDouble() * 0.05f;
            float valLoss = 1.0f - (epoch * 0.08f) + (float)random.NextDouble() * 0.15f;
            float valAcc = 0.5f + (epoch * 0.06f) + (float)random.NextDouble() * 0.05f;

            writer.LogTrainingStep(trainLoss, trainAcc, learningRate: 0.001f, step: epoch);
            writer.LogValidationStep(valLoss, valAcc, step: epoch);

            // Log some weights
            var weights = Enumerable.Range(0, 100).Select(_ => (float)random.NextDouble() * 2 - 1).ToArray();
            writer.AddHistogram("layer1/weights", weights, epoch);
        }

        writer.Flush();

        // Verify files were created
        Assert.True(Directory.GetFiles(logDir, "events.*").Length > 0);
    }

    [Fact]
    public void TensorBoardTrainingContext_FullTrainingLoop_TracksMetrics()
    {
        string? logDir = null;
        TensorBoardTrainingContext? context = null;
        try
        {
            context = new TensorBoardTrainingContext("full_loop_context");
            logDir = context.Writer.LogDir;

            var random = new Random(42);

            // Simulate training
            for (int i = 0; i < 10; i++)
            {
                float loss = 1.0f / (i + 1);
                float acc = 0.5f + (i * 0.05f);

                context.LogTrainStep(loss, acc, 0.001f);
            }

            // Final validation
            context.LogValStep(0.1f, 0.9f);

            Assert.Equal(10, context.GlobalStep);
        }
        finally
        {
            context?.Dispose();
            CleanupRunsDirectory(logDir);
        }
    }

    #endregion
}
