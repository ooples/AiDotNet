using Xunit;
using Moq;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Strategies;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Tests.Reasoning.Strategies;

/// <summary>
/// Unit tests for ChainOfThoughtStrategy.
/// </summary>
public class ChainOfThoughtStrategyTests
{
    private readonly Mock<IChatModel> _mockChatModel;
    private readonly ChainOfThoughtStrategy<double> _strategy;

    public ChainOfThoughtStrategyTests()
    {
        _mockChatModel = new Mock<IChatModel>();
        _strategy = new ChainOfThoughtStrategy<double>(_mockChatModel.Object);
    }

    [Fact]
    public async Task ReasonAsync_WithSimpleMathProblem_ReturnsSuccessfulResult()
    {
        // Arrange
        string query = "What is 2 + 2?";
        string mockResponse = @"Let me solve this step by step:

Step 1: Identify the operation
We need to add 2 and 2.

Step 2: Perform the addition
2 + 2 = 4

Final Answer: 4";

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(mockResponse);

        // Act
        var result = await _strategy.ReasonAsync(query);

        // Assert
        Assert.True(result.Success);
        Assert.NotNull(result.Chain);
        Assert.True(result.Chain.Steps.Count >= 2);
        Assert.Contains("4", result.FinalAnswer);
    }

    [Fact]
    public async Task ReasonAsync_WithConfiguration_RespectsMaxSteps()
    {
        // Arrange
        string query = "Count to 10";
        string mockResponse = @"Step 1: Start counting
Step 2: 1
Step 3: 2
Step 4: 3
Step 5: 4
Step 6: 5
Step 7: 6
Step 8: 7
Step 9: 8
Step 10: 9
Step 11: 10";

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(mockResponse);

        var config = new ReasoningConfig { MaxSteps = 5 };

        // Act
        var result = await _strategy.ReasonAsync(query, config);

        // Assert
        Assert.True(result.Success);
        Assert.NotNull(result.Chain);
        Assert.True(result.Chain.Steps.Count <= 5);
    }

    [Fact]
    public async Task ReasonAsync_WithEmptyQuery_ThrowsArgumentException()
    {
        // Arrange
        string emptyQuery = "";

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(
            async () => await _strategy.ReasonAsync(emptyQuery)
        );
    }

    [Fact]
    public async Task ReasonAsync_WithCancellation_ThrowsOperationCanceledException()
    {
        // Arrange
        string query = "What is 2 + 2?";
        var cts = new CancellationTokenSource();
        cts.Cancel();

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ThrowsAsync(new OperationCanceledException());

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await _strategy.ReasonAsync(query, cancellationToken: cts.Token)
        );
    }

    [Fact]
    public async Task ReasonAsync_WithJsonFormattedSteps_ParsesCorrectly()
    {
        // Arrange
        string query = "Solve 3 × 4";
        string mockResponse = @"```json
{
    ""steps"": [
        {
            ""step_number"": 1,
            ""content"": ""Multiply 3 by 4"",
            ""reasoning"": ""Basic multiplication""
        },
        {
            ""step_number"": 2,
            ""content"": ""Result is 12"",
            ""reasoning"": ""3 × 4 = 12""
        }
    ],
    ""final_answer"": ""12""
}
```";

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(mockResponse);

        // Act
        var result = await _strategy.ReasonAsync(query);

        // Assert
        Assert.True(result.Success);
        Assert.NotNull(result.Chain);
        Assert.Equal(2, result.Chain.Steps.Count);
        Assert.Contains("12", result.FinalAnswer);
    }

    [Fact]
    public void StrategyName_ReturnsCorrectName()
    {
        // Assert
        Assert.Equal("Chain-of-Thought", _strategy.StrategyName);
    }

    [Fact]
    public void Description_ContainsRelevantKeywords()
    {
        // Assert
        Assert.Contains("step-by-step", _strategy.Description.ToLowerInvariant());
    }

    [Theory]
    [InlineData("What is 5 + 7?", "12")]
    [InlineData("What is 10 - 3?", "7")]
    [InlineData("What is 6 × 8?", "48")]
    public async Task ReasonAsync_WithVariousMathProblems_ReturnsCorrectAnswers(
        string query,
        string expectedAnswer)
    {
        // Arrange
        string mockResponse = $@"Step 1: Calculate
The answer is {expectedAnswer}

Final Answer: {expectedAnswer}";

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(mockResponse);

        // Act
        var result = await _strategy.ReasonAsync(query);

        // Assert
        Assert.True(result.Success);
        Assert.Contains(expectedAnswer, result.FinalAnswer);
    }

    [Fact]
    public async Task ReasonAsync_WithFastConfig_CompletesQuickly()
    {
        // Arrange
        string query = "Quick question";
        string mockResponse = "Step 1: Answer quickly\nFinal: Done";

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(mockResponse);

        var config = ReasoningConfig.Fast;
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        // Act
        var result = await _strategy.ReasonAsync(query, config);
        stopwatch.Stop();

        // Assert
        Assert.True(result.Success);
        Assert.True(stopwatch.ElapsedMilliseconds < 5000); // Should be fast
    }
}
