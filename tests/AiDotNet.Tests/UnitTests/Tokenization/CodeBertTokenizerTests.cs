using System.Collections.Generic;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.CodeTokenization;
using AiDotNet.Tokenization.Models;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Tokenization;

/// <summary>
/// Unit tests for CodeBertTokenizer.
/// </summary>
public class CodeBertTokenizerTests
{
    private readonly CodeBertTokenizer _tokenizer;
    private readonly WordPieceTokenizer _baseTokenizer;

    public CodeBertTokenizerTests()
    {
        var corpus = new List<string>
        {
            "def hello_world(): print('Hello')",
            "function test() { return 1; }",
            "public void Main() { Console.WriteLine(); }"
        };

        _baseTokenizer = WordPieceTokenizer.Train(corpus, 500);
        _tokenizer = new CodeBertTokenizer(_baseTokenizer.Vocabulary);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_WithVocabulary_Succeeds()
    {
        Assert.NotNull(_tokenizer);
    }

    [Fact(Timeout = 60000)]
    public async Task EncodeCodeAndNL_WithCodeOnly_ReturnsTokens()
    {
        // Arrange
        var code = "def test(): pass";

        // Act
        var result = _tokenizer.EncodeCodeAndNL(code);

        // Assert
        Assert.NotEmpty(result.TokenIds);
        Assert.NotEmpty(result.Tokens);
    }

    [Fact(Timeout = 60000)]
    public async Task EncodeCodeAndNL_WithCodeAndNaturalLanguage_CombinesBoth()
    {
        // Arrange
        var code = "def add(a, b): return a + b";
        var naturalLanguage = "Add two numbers together";

        // Act
        var result = _tokenizer.EncodeCodeAndNL(code, naturalLanguage);

        // Assert
        Assert.NotEmpty(result.TokenIds);
        Assert.NotEmpty(result.TokenTypeIds);
    }

    [Fact(Timeout = 60000)]
    public async Task EncodeCodeAndNL_AddsBertSpecialTokens()
    {
        // Arrange
        var code = "def test(): pass";

        // Act
        var result = _tokenizer.EncodeCodeAndNL(code);

        // Assert
        Assert.Contains("[CLS]", result.Tokens);
        Assert.Contains("[SEP]", result.Tokens);
    }

    [Fact(Timeout = 60000)]
    public async Task EncodeCodeAndNL_WithPadding_PadsToMaxLength()
    {
        // Arrange
        var code = "x = 1";
        var options = new EncodingOptions
        {
            Padding = true,
            MaxLength = 50
        };

        // Act
        var result = _tokenizer.EncodeCodeAndNL(code, options: options);

        // Assert
        Assert.Equal(50, result.TokenIds.Count);
    }

    [Fact(Timeout = 60000)]
    public async Task EncodeCodeAndNL_WithTruncation_TruncatesToMaxLength()
    {
        // Arrange
        var code = "def test(): return 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8";
        var options = new EncodingOptions
        {
            Truncation = true,
            MaxLength = 10
        };

        // Act
        var result = _tokenizer.EncodeCodeAndNL(code, options: options);

        // Assert
        Assert.True(result.TokenIds.Count <= 10);
    }

    [Fact(Timeout = 60000)]
    public async Task Decode_ReturnsDecodedText()
    {
        // Arrange
        var code = "def test";
        var encoded = _tokenizer.EncodeCodeAndNL(code);

        // Act
        var decoded = _tokenizer.Decode(encoded.TokenIds);

        // Assert
        Assert.NotEmpty(decoded);
    }

    [Fact(Timeout = 60000)]
    public async Task EncodeCodeAndNL_AttentionMask_HasCorrectLength()
    {
        // Arrange
        var code = "def test(): pass";

        // Act
        var result = _tokenizer.EncodeCodeAndNL(code);

        // Assert
        Assert.Equal(result.TokenIds.Count, result.AttentionMask.Count);
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenizer_Property_ReturnsUnderlyingTokenizer()
    {
        Assert.NotNull(_tokenizer.Tokenizer);
    }
}
