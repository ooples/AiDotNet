using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Models;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Tokenization;

/// <summary>
/// Unit tests for AutoTokenizer (HuggingFace-style automatic loading).
/// </summary>
public class AutoTokenizerTests : IDisposable
{
    private readonly string _tempDir;

    public AutoTokenizerTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "auto_tokenizer_tests_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            try
            {
                Directory.Delete(_tempDir, true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    [Fact]
    public void FromPretrained_WithLocalPath_LoadsTokenizer()
    {
        // Arrange
        CreateLocalTokenizerFiles();

        // Act
        var tokenizer = AutoTokenizer.FromPretrained(_tempDir);

        // Assert
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void FromPretrained_WithEmptyPath_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => AutoTokenizer.FromPretrained(""));
        Assert.Throws<ArgumentException>(() => AutoTokenizer.FromPretrained("   "));
    }

    [Fact]
    public void GetDefaultCacheDir_ReturnsValidPath()
    {
        // Act
        var cacheDir = AutoTokenizer.GetDefaultCacheDir();

        // Assert
        Assert.NotNull(cacheDir);
        Assert.Contains("huggingface", cacheDir);
        Assert.Contains("tokenizers", cacheDir);
    }

    [Fact]
    public void IsCached_WithNonexistentModel_ReturnsFalse()
    {
        // Act
        var isCached = AutoTokenizer.IsCached("nonexistent-model-xyz", _tempDir);

        // Assert
        Assert.False(isCached);
    }

    [Fact]
    public void IsCached_WithCachedModel_ReturnsTrue()
    {
        // Arrange
        var modelName = "test-model";
        var modelDir = Path.Combine(_tempDir, modelName);
        Directory.CreateDirectory(modelDir);
        CreateTokenizerFilesIn(modelDir);

        // Act
        var isCached = AutoTokenizer.IsCached(modelName, _tempDir);

        // Assert
        Assert.True(isCached);
    }

    [Fact]
    public void IsCached_WithEmptyName_ReturnsFalse()
    {
        // Act
        var isCached = AutoTokenizer.IsCached("");

        // Assert
        Assert.False(isCached);
    }

    [Fact]
    public void ListCachedModels_ReturnsCorrectModels()
    {
        // Arrange
        var model1Dir = Path.Combine(_tempDir, "model1");
        var model2Dir = Path.Combine(_tempDir, "model2");
        Directory.CreateDirectory(model1Dir);
        Directory.CreateDirectory(model2Dir);

        // Act
        var models = AutoTokenizer.ListCachedModels(_tempDir);

        // Assert
        Assert.Equal(2, models.Length);
        Assert.Contains("model1", models);
        Assert.Contains("model2", models);
    }

    [Fact]
    public void ListCachedModels_EmptyCache_ReturnsEmptyArray()
    {
        // Arrange
        var emptyDir = Path.Combine(_tempDir, "empty_cache");
        Directory.CreateDirectory(emptyDir);

        // Act
        var models = AutoTokenizer.ListCachedModels(emptyDir);

        // Assert
        Assert.Empty(models);
    }

    [Fact]
    public void ListCachedModels_NonexistentDir_ReturnsEmptyArray()
    {
        // Arrange
        var nonexistentDir = Path.Combine(_tempDir, "nonexistent");

        // Act
        var models = AutoTokenizer.ListCachedModels(nonexistentDir);

        // Assert
        Assert.Empty(models);
    }

    [Fact]
    public void ClearCache_SpecificModel_RemovesOnlyThatModel()
    {
        // Arrange
        var model1Dir = Path.Combine(_tempDir, "model1");
        var model2Dir = Path.Combine(_tempDir, "model2");
        Directory.CreateDirectory(model1Dir);
        Directory.CreateDirectory(model2Dir);

        // Act
        AutoTokenizer.ClearCache("model1", _tempDir);

        // Assert
        Assert.False(Directory.Exists(model1Dir));
        Assert.True(Directory.Exists(model2Dir));
    }

    [Fact]
    public void ClearCache_AllModels_ClearsEntireCache()
    {
        // Arrange
        var cacheDir = Path.Combine(_tempDir, "cache_to_clear");
        var model1Dir = Path.Combine(cacheDir, "model1");
        var model2Dir = Path.Combine(cacheDir, "model2");
        Directory.CreateDirectory(model1Dir);
        Directory.CreateDirectory(model2Dir);

        // Act
        AutoTokenizer.ClearCache(null, cacheDir);

        // Assert
        Assert.True(Directory.Exists(cacheDir)); // Cache dir still exists
        Assert.Empty(Directory.GetDirectories(cacheDir)); // But is empty
    }

    [Fact]
    public void ListCachedModels_HandlesSlashInModelName()
    {
        // Arrange - Simulate org/model format stored as org--model
        var modelDir = Path.Combine(_tempDir, "org--model");
        Directory.CreateDirectory(modelDir);

        // Act
        var models = AutoTokenizer.ListCachedModels(_tempDir);

        // Assert
        Assert.Contains("org/model", models);
    }

    [Fact]
    public async System.Threading.Tasks.Task FromPretrainedAsync_WithLocalPath_LoadsTokenizer()
    {
        // Arrange
        CreateLocalTokenizerFiles();

        // Act
        var tokenizer = await AutoTokenizer.FromPretrainedAsync(_tempDir);

        // Assert
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public async System.Threading.Tasks.Task FromPretrainedAsync_WithEmptyPath_ThrowsArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => AutoTokenizer.FromPretrainedAsync(""));
    }

    private void CreateLocalTokenizerFiles()
    {
        CreateTokenizerFilesIn(_tempDir);
    }

    private void CreateTokenizerFilesIn(string dir)
    {
        // Create vocab.json
        var vocab = new Dictionary<string, int>
        {
            { "[UNK]", 0 },
            { "[PAD]", 1 },
            { "[CLS]", 2 },
            { "[SEP]", 3 },
            { "hello", 4 },
            { "world", 5 }
        };
        File.WriteAllText(Path.Combine(dir, "vocab.json"),
            JsonConvert.SerializeObject(vocab));

        // Create tokenizer_config.json
        var config = new TokenizerConfig
        {
            TokenizerClass = "BertTokenizer",
            UnkToken = "[UNK]",
            PadToken = "[PAD]",
            ClsToken = "[CLS]",
            SepToken = "[SEP]"
        };
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"),
            JsonConvert.SerializeObject(config));
    }
}
