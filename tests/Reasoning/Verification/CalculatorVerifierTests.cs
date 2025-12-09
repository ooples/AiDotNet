using Xunit;
using AiDotNet.Reasoning.Verification;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Tests.Reasoning.Verification;

/// <summary>
/// Unit tests for CalculatorVerifier.
/// </summary>
public class CalculatorVerifierTests
{
    private readonly CalculatorVerifier<double> _verifier;

    public CalculatorVerifierTests()
    {
        _verifier = new CalculatorVerifier<double>();
    }

    [Theory]
    [InlineData("2 + 2", "4", true)]
    [InlineData("10 - 3", "7", true)]
    [InlineData("5 × 6", "30", true)]
    [InlineData("15 / 3", "5", true)]
    [InlineData("2 + 2", "5", false)]
    public async Task VerifyAsync_WithSimpleArithmetic_ReturnsCorrectValidation(
        string expression,
        string claimedAnswer,
        bool shouldBeValid)
    {
        // Arrange
        var chain = CreateChainWithCalculation(expression, claimedAnswer);

        // Act
        var result = await _verifier.VerifyAsync(chain, claimedAnswer);

        // Assert
        Assert.Equal(shouldBeValid, result.IsValid);
    }

    [Fact]
    public async Task VerifyAsync_WithMultiStepMath_ValidatesCorrectly()
    {
        // Arrange
        var chain = new ReasoningChain<double>
        {
            Steps = new List<ReasoningStep<double>>
            {
                new() { Content = "First calculate 5 + 3 = 8" },
                new() { Content = "Then multiply 8 × 2 = 16" },
                new() { Content = "Finally divide 16 / 4 = 4" }
            },
            FinalAnswer = "4"
        };

        // Act
        var result = await _verifier.VerifyAsync(chain, "4");

        // Assert
        Assert.True(result.IsValid);
        Assert.Contains("verified", result.Message.ToLowerInvariant());
    }

    [Fact]
    public async Task VerifyAsync_WithIncorrectCalculation_DetectsError()
    {
        // Arrange
        var chain = new ReasoningChain<double>
        {
            Steps = new List<ReasoningStep<double>>
            {
                new() { Content = "Calculate 7 × 8 = 54" } // Wrong! Should be 56
            },
            FinalAnswer = "54"
        };

        // Act
        var result = await _verifier.VerifyAsync(chain, "54");

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("incorrect", result.Message.ToLowerInvariant());
    }

    [Theory]
    [InlineData("2^3", "8")]
    [InlineData("10^2", "100")]
    [InlineData("5^0", "1")]
    public async Task VerifyAsync_WithExponents_ValidatesCorrectly(string expression, string answer)
    {
        // Arrange
        var chain = CreateChainWithCalculation(expression, answer);

        // Act
        var result = await _verifier.VerifyAsync(chain, answer);

        // Assert
        Assert.True(result.IsValid);
    }

    [Fact]
    public async Task VerifyAsync_WithNoMathematicalContent_ReturnsNotApplicable()
    {
        // Arrange
        var chain = new ReasoningChain<double>
        {
            Steps = new List<ReasoningStep<double>>
            {
                new() { Content = "This is just text without any math" }
            },
            FinalAnswer = "No calculation"
        };

        // Act
        var result = await _verifier.VerifyAsync(chain);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("no mathematical", result.Message.ToLowerInvariant());
    }

    [Fact]
    public async Task VerifyAsync_WithDecimalNumbers_HandlesCorrectly()
    {
        // Arrange
        var chain = CreateChainWithCalculation("3.14 × 2", "6.28");

        // Act
        var result = await _verifier.VerifyAsync(chain, "6.28");

        // Assert
        Assert.True(result.IsValid);
    }

    [Fact]
    public async Task VerifyAsync_WithParentheses_RespectsOrderOfOperations()
    {
        // Arrange
        var chain = CreateChainWithCalculation("(2 + 3) × 4", "20");

        // Act
        var result = await _verifier.VerifyAsync(chain, "20");

        // Assert
        Assert.True(result.IsValid);
    }

    [Fact]
    public void VerifierName_ReturnsCorrectName()
    {
        // Assert
        Assert.Equal("Calculator Verifier", _verifier.VerifierName);
    }

    [Fact]
    public void Description_ContainsRelevantKeywords()
    {
        // Assert
        var description = _verifier.Description.ToLowerInvariant();
        Assert.Contains("mathematical", description);
        Assert.Contains("calculation", description);
    }

    private ReasoningChain<double> CreateChainWithCalculation(string expression, string answer)
    {
        return new ReasoningChain<double>
        {
            Steps = new List<ReasoningStep<double>>
            {
                new() { Content = $"Calculate {expression} = {answer}" }
            },
            FinalAnswer = answer
        };
    }
}
