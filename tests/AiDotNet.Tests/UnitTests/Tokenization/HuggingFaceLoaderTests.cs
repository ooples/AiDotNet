using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Models;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Tokenization;

/// <summary>
/// Unit tests for HuggingFace tokenizer loader.
/// </summary>
public class HuggingFaceLoaderTests : IDisposable
{
    private readonly string _tempDir;

    public HuggingFaceLoaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "hf_tokenizer_tests_" + Guid.NewGuid().ToString("N"));
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
    public void LoadFromDirectory_WithValidBpeFiles_LoadsTokenizer()
    {
        // Arrange
        CreateBpeTokenizerFiles();

        // Act
        var tokenizer = HuggingFaceTokenizerLoader.LoadFromDirectory(_tempDir);

        // Assert
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void LoadFromDirectory_WithVocabJson_LoadsVocabulary()
    {
        // Arrange
        CreateWordPieceTokenizerFiles();

        // Act
        var tokenizer = HuggingFaceTokenizerLoader.LoadFromDirectory(_tempDir);

        // Assert
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.Vocabulary.ContainsToken("hello"));
    }

    [Fact]
    public void LoadFromDirectory_MissingDirectory_ThrowsException()
    {
        // Arrange
        var nonExistentPath = Path.Combine(_tempDir, "nonexistent");

        // Act & Assert
        Assert.Throws<DirectoryNotFoundException>(() =>
            HuggingFaceTokenizerLoader.LoadFromDirectory(nonExistentPath));
    }

    [Fact]
    public void LoadFromDirectory_MissingConfigFile_ThrowsException()
    {
        // Arrange - empty directory with no tokenizer files

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() =>
            HuggingFaceTokenizerLoader.LoadFromDirectory(_tempDir));
    }

    [Fact]
    public void SaveToDirectory_CreatesVocabFile()
    {
        // Arrange
        var corpus = new List<string> { "Hello world", "Test text" };
        var tokenizer = AiDotNet.Tokenization.Algorithms.BpeTokenizer.Train(corpus, 100);
        var outputDir = Path.Combine(_tempDir, "output");

        // Act
        HuggingFaceTokenizerLoader.SaveToDirectory(tokenizer, outputDir);

        // Assert
        Assert.True(File.Exists(Path.Combine(outputDir, "vocab.json")));
        Assert.True(File.Exists(Path.Combine(outputDir, "tokenizer_config.json")));
    }

    [Fact]
    public void SaveToDirectory_ConfigContainsSpecialTokens()
    {
        // Arrange
        var corpus = new List<string> { "Hello world" };
        var tokenizer = AiDotNet.Tokenization.Algorithms.BpeTokenizer.Train(corpus, 100);
        var outputDir = Path.Combine(_tempDir, "output2");

        // Act
        HuggingFaceTokenizerLoader.SaveToDirectory(tokenizer, outputDir);

        // Assert
        var configPath = Path.Combine(outputDir, "tokenizer_config.json");
        var configJson = File.ReadAllText(configPath);
        var config = JsonConvert.DeserializeObject<TokenizerConfig>(configJson);

        Assert.NotNull(config);
        Assert.Equal("[UNK]", config.UnkToken);
        Assert.Equal("[PAD]", config.PadToken);
    }

    [Fact]
    public void LoadAndSave_Roundtrip_PreservesVocabulary()
    {
        // Arrange
        CreateBpeTokenizerFiles();
        var outputDir = Path.Combine(_tempDir, "roundtrip");

        // Act
        var loaded = HuggingFaceTokenizerLoader.LoadFromDirectory(_tempDir);
        HuggingFaceTokenizerLoader.SaveToDirectory(loaded, outputDir);
        var reloaded = HuggingFaceTokenizerLoader.LoadFromDirectory(outputDir);

        // Assert
        Assert.Equal(loaded.VocabularySize, reloaded.VocabularySize);
    }

    [Fact]
    public void LoadFromTokenizerJson_WithBpeModel_LoadsCorrectly()
    {
        // Arrange
        CreateTokenizerJsonFile("bpe");

        // Act
        var tokenizer = HuggingFaceTokenizerLoader.LoadFromTokenizerJson(
            Path.Combine(_tempDir, "tokenizer.json"));

        // Assert
        Assert.NotNull(tokenizer);
    }

    [Fact]
    public void LoadFromTokenizerJson_WithWordPieceModel_LoadsCorrectly()
    {
        // Arrange
        CreateTokenizerJsonFile("wordpiece");

        // Act
        var tokenizer = HuggingFaceTokenizerLoader.LoadFromTokenizerJson(
            Path.Combine(_tempDir, "tokenizer.json"));

        // Assert
        Assert.NotNull(tokenizer);
    }

    [Fact]
    public void LoadFromTokenizerJson_WithUnigramModel_LoadsCorrectly()
    {
        // Arrange
        CreateTokenizerJsonFile("unigram");

        // Act
        var tokenizer = HuggingFaceTokenizerLoader.LoadFromTokenizerJson(
            Path.Combine(_tempDir, "tokenizer.json"));

        // Assert
        Assert.NotNull(tokenizer);
    }

    [Fact]
    public void LoadFromTokenizerJson_ExtractsSpecialTokens()
    {
        // Arrange
        CreateTokenizerJsonFile("bpe");

        // Act
        var tokenizer = HuggingFaceTokenizerLoader.LoadFromTokenizerJson(
            Path.Combine(_tempDir, "tokenizer.json"));

        // Assert
        Assert.NotNull(tokenizer.SpecialTokens);
        Assert.NotEmpty(tokenizer.SpecialTokens.UnkToken);
    }

    private void CreateBpeTokenizerFiles()
    {
        // Create vocab.json
        var vocab = new Dictionary<string, int>
        {
            { "[UNK]", 0 },
            { "[PAD]", 1 },
            { "[CLS]", 2 },
            { "[SEP]", 3 },
            { "hello", 4 },
            { "world", 5 },
            { "he", 6 },
            { "llo", 7 }
        };
        File.WriteAllText(Path.Combine(_tempDir, "vocab.json"),
            JsonConvert.SerializeObject(vocab));

        // Create merges.txt
        File.WriteAllText(Path.Combine(_tempDir, "merges.txt"),
            "#version: 0.2\nh e\nl l\nhe llo\nwor ld");

        // Create tokenizer_config.json
        var config = new TokenizerConfig
        {
            TokenizerClass = "GPT2Tokenizer",
            UnkToken = "[UNK]",
            PadToken = "[PAD]",
            ClsToken = "[CLS]",
            SepToken = "[SEP]"
        };
        File.WriteAllText(Path.Combine(_tempDir, "tokenizer_config.json"),
            JsonConvert.SerializeObject(config));
    }

    private void CreateWordPieceTokenizerFiles()
    {
        // Create vocab.json
        var vocab = new Dictionary<string, int>
        {
            { "[UNK]", 0 },
            { "[PAD]", 1 },
            { "[CLS]", 2 },
            { "[SEP]", 3 },
            { "[MASK]", 4 },
            { "hello", 5 },
            { "world", 6 },
            { "##ing", 7 }
        };
        File.WriteAllText(Path.Combine(_tempDir, "vocab.json"),
            JsonConvert.SerializeObject(vocab));

        // Create tokenizer_config.json
        var config = new TokenizerConfig
        {
            TokenizerClass = "BertTokenizer",
            UnkToken = "[UNK]",
            PadToken = "[PAD]",
            ClsToken = "[CLS]",
            SepToken = "[SEP]",
            MaskToken = "[MASK]"
        };
        File.WriteAllText(Path.Combine(_tempDir, "tokenizer_config.json"),
            JsonConvert.SerializeObject(config));
    }

    private void CreateTokenizerJsonFile(string modelType)
    {
        object vocab;
        object model;

        if (modelType == "unigram")
        {
            vocab = new object[]
            {
                new object[] { "[UNK]", 0.0 },
                new object[] { "[PAD]", 0.0 },
                new object[] { "hello", -1.0 },
                new object[] { "world", -1.5 }
            };
            model = new { type = "Unigram", vocab };
        }
        else
        {
            vocab = new Dictionary<string, int>
            {
                { "[UNK]", 0 },
                { "[PAD]", 1 },
                { "hello", 2 },
                { "world", 3 }
            };
            model = new
            {
                type = modelType == "wordpiece" ? "WordPiece" : "BPE",
                vocab,
                merges = modelType == "bpe" ? new[] { "h e", "l l" } : null
            };
        }

        var tokenizerJson = new
        {
            model,
            added_tokens = new[]
            {
                new { id = 0, content = "[UNK]", special = true },
                new { id = 1, content = "[PAD]", special = true }
            }
        };

        File.WriteAllText(Path.Combine(_tempDir, "tokenizer.json"),
            JsonConvert.SerializeObject(tokenizerJson));
    }
}
