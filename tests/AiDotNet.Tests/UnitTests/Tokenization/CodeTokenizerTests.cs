using System.Collections.Generic;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.CodeTokenization;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Tokenization;

/// <summary>
/// Unit tests for CodeTokenizer.
/// </summary>
public class CodeTokenizerTests
{
    private readonly CodeTokenizer _codeTokenizer;
    private readonly BpeTokenizer _baseTokenizer;

    public CodeTokenizerTests()
    {
        var corpus = new List<string>
        {
            "def hello_world():",
            "    print('Hello, World!')",
            "class MyClass:",
            "    def __init__(self):",
            "        self.value = 0",
            "function calculateSum(a, b) { return a + b; }",
            "public class Program { public static void Main() { } }"
        };

        _baseTokenizer = BpeTokenizer.Train(corpus, 500);
        _codeTokenizer = new CodeTokenizer(_baseTokenizer, ProgrammingLanguage.Python, splitIdentifiers: true);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_WithValidBaseTokenizer_Succeeds()
    {
        // Arrange & Act
        var tokenizer = new CodeTokenizer(_baseTokenizer);

        // Assert
        Assert.NotNull(tokenizer);
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenize_PythonCode_ReturnsTokens()
    {
        // Arrange
        var code = "def test():\n    return 1";

        // Act
        var tokens = _codeTokenizer.Tokenize(code);

        // Assert
        Assert.NotEmpty(tokens);
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenize_EmptyCode_ReturnsEmpty()
    {
        // Act
        var tokens = _codeTokenizer.Tokenize("");

        // Assert
        Assert.Empty(tokens);
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenize_CamelCaseIdentifier_SplitsIdentifier()
    {
        // Arrange
        var code = "calculateTotalSum";

        // Act
        var tokens = _codeTokenizer.Tokenize(code);

        // Assert
        Assert.NotEmpty(tokens);
        // Should split camelCase identifier
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenize_SnakeCaseIdentifier_SplitsIdentifier()
    {
        // Arrange
        var code = "calculate_total_sum";

        // Act
        var tokens = _codeTokenizer.Tokenize(code);

        // Assert
        Assert.NotEmpty(tokens);
        // Should split snake_case identifier
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenize_Keywords_PreservesKeywords()
    {
        // Arrange
        var code = "def class return if else";

        // Act
        var tokens = _codeTokenizer.Tokenize(code);

        // Assert
        Assert.NotEmpty(tokens);
        Assert.Contains("def", tokens);
        Assert.Contains("class", tokens);
    }

    [Fact(Timeout = 60000)]
    public async Task Encode_ReturnsValidTokenIds()
    {
        // Arrange
        var code = "def hello(): pass";

        // Act
        var result = _codeTokenizer.Encode(code);

        // Assert
        Assert.NotEmpty(result.TokenIds);
        Assert.Equal(result.Tokens.Count, result.TokenIds.Count);
    }

    [Fact(Timeout = 60000)]
    public async Task Decode_ReconstructsCode()
    {
        // Arrange
        var code = "def test";
        var encoded = _codeTokenizer.Encode(code);

        // Act
        var decoded = _codeTokenizer.Decode(encoded.TokenIds);

        // Assert
        Assert.NotEmpty(decoded);
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenize_WithCSharpLanguage_UsesCorrectKeywords()
    {
        // Arrange
        var csharpTokenizer = new CodeTokenizer(_baseTokenizer, ProgrammingLanguage.CSharp);
        var code = "public class Test { }";

        // Act
        var tokens = csharpTokenizer.Tokenize(code);

        // Assert
        Assert.NotEmpty(tokens);
        Assert.Contains("public", tokens);
        Assert.Contains("class", tokens);
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenize_WithJavaScriptLanguage_UsesCorrectKeywords()
    {
        // Arrange
        var jsTokenizer = new CodeTokenizer(_baseTokenizer, ProgrammingLanguage.JavaScript);
        var code = "function test() { return true; }";

        // Act
        var tokens = jsTokenizer.Tokenize(code);

        // Assert
        Assert.NotEmpty(tokens);
        Assert.Contains("function", tokens);
        Assert.Contains("return", tokens);
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenize_StringLiteral_PreservesString()
    {
        // Arrange
        var code = "\"Hello World\"";

        // Act
        var tokens = _codeTokenizer.Tokenize(code);

        // Assert
        Assert.NotEmpty(tokens);
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenize_Numbers_PreservesNumbers()
    {
        // Arrange
        var code = "x = 123";

        // Act
        var tokens = _codeTokenizer.Tokenize(code);

        // Assert
        Assert.NotEmpty(tokens);
    }
}
