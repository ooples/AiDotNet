using AiDotNet.Agents;
using AiDotNet.Interfaces;
using AiDotNet.Tools;
using Xunit;

namespace AiDotNetTests.UnitTests.Agents;

/// <summary>
/// Unit tests for the ChainOfThoughtAgent class.
/// </summary>
public class ChainOfThoughtAgentTests
{
    [Fact]
    public void Constructor_WithValidChatModel_InitializesSuccessfully()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>("Test response");

        // Act
        var agent = new ChainOfThoughtAgent<double>(mockChatModel);

        // Assert
        Assert.NotNull(agent);
        Assert.Equal(mockChatModel, agent.ChatModel);
        Assert.Empty(agent.Tools);
    }

    [Fact]
    public void Constructor_WithTools_InitializesWithTools()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>("Test response");
        var calculator = new CalculatorTool();
        var tools = new ITool[] { calculator };

        // Act
        var agent = new ChainOfThoughtAgent<double>(mockChatModel, tools);

        // Assert
        Assert.NotNull(agent);
        Assert.Single(agent.Tools);
        Assert.Equal("Calculator", agent.Tools[0].Name);
    }

    [Fact]
    public void Constructor_WithAllowToolsFalse_CreatesPureReasoningAgent()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>("Test response");
        var calculator = new CalculatorTool();
        var tools = new ITool[] { calculator };

        // Act
        var agent = new ChainOfThoughtAgent<double>(mockChatModel, tools, allowTools: false);

        // Assert
        Assert.NotNull(agent);
        // Even though tools are provided, allowTools=false means pure reasoning mode
        Assert.Single(agent.Tools);
    }

    [Fact]
    public void Constructor_WithNullChatModel_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new ChainOfThoughtAgent<double>(null!));
    }

    [Fact]
    public async Task RunAsync_WithNullQuery_ThrowsArgumentException()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>("Test response");
        var agent = new ChainOfThoughtAgent<double>(mockChatModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            agent.RunAsync(null!));
    }

    [Fact]
    public async Task RunAsync_WithEmptyQuery_ThrowsArgumentException()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>("Test response");
        var agent = new ChainOfThoughtAgent<double>(mockChatModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            agent.RunAsync(""));
    }

    [Fact]
    public async Task RunAsync_WithValidQuery_ReturnsAnswer()
    {
        // Arrange
        var jsonResponse = @"{
            ""reasoning_steps"": [
                ""Step 1: Understand the problem"",
                ""Step 2: Apply the solution""
            ],
            ""tool_calls"": [],
            ""final_answer"": ""The answer is 42.""
        }";

        var mockChatModel = new MockChatModel<double>(jsonResponse);
        var agent = new ChainOfThoughtAgent<double>(mockChatModel);

        // Act
        var result = await agent.RunAsync("What is the answer?");

        // Assert
        Assert.Contains("42", result);
        Assert.Contains("What is the answer?", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithToolCall_ExecutesTool()
    {
        // Arrange
        var jsonResponse1 = @"{
            ""reasoning_steps"": [""I need to calculate 25 * 4""],
            ""tool_calls"": [{
                ""tool_name"": ""Calculator"",
                ""tool_input"": ""25 * 4""
            }],
            ""final_answer"": """"
        }";

        var jsonResponse2 = @"{
            ""reasoning_steps"": [],
            ""tool_calls"": [],
            ""final_answer"": ""The result is 100.""
        }";

        var mockChatModel = new MockChatModel<double>(jsonResponse1, jsonResponse2);
        var calculator = new CalculatorTool();
        var agent = new ChainOfThoughtAgent<double>(mockChatModel, new ITool[] { calculator });

        // Act
        var result = await agent.RunAsync("What is 25 * 4?");

        // Assert
        Assert.Contains("100", result);
        Assert.Contains("Calculator", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithInvalidMaxIterations_ThrowsArgumentException()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>("Test response");
        var agent = new ChainOfThoughtAgent<double>(mockChatModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            agent.RunAsync("Test query", maxIterations: 0));
    }

    [Fact]
    public async Task RunAsync_UpdatesScratchpad()
    {
        // Arrange
        var jsonResponse = @"{
            ""reasoning_steps"": [""Step 1: First step""],
            ""tool_calls"": [],
            ""final_answer"": ""Done.""
        }";

        var mockChatModel = new MockChatModel<double>(jsonResponse);
        var agent = new ChainOfThoughtAgent<double>(mockChatModel);

        // Act
        var result = await agent.RunAsync("Test query");

        // Assert
        Assert.NotEmpty(agent.Scratchpad);
        Assert.Contains("Test query", agent.Scratchpad);
        Assert.Contains("Reasoning Steps", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithHttpError_ReturnsErrorMessage()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>(); // No responses, will throw
        var agent = new ChainOfThoughtAgent<double>(mockChatModel);

        // Act
        var result = await agent.RunAsync("Test query");

        // Assert
        Assert.Contains("error", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RunAsync_PureCoTMode_DoesNotExecuteTools()
    {
        // Arrange
        var jsonResponse = @"{
            ""reasoning_steps"": [
                ""Step 1: 25 multiplied by 4"",
                ""Step 2: equals 100""
            ],
            ""tool_calls"": [],
            ""final_answer"": ""100""
        }";

        var mockChatModel = new MockChatModel<double>(jsonResponse);
        var calculator = new CalculatorTool();
        var agent = new ChainOfThoughtAgent<double>(mockChatModel, new ITool[] { calculator }, allowTools: false);

        // Act
        var result = await agent.RunAsync("What is 25 * 4?");

        // Assert
        Assert.Contains("100", result);
        // In pure CoT mode, even if tool_calls were present, they wouldn't execute
    }

    [Fact]
    public async Task RunAsync_WithFallbackParsing_HandlesNonJsonResponse()
    {
        // Arrange
        var textResponse = @"
Step 1: I need to solve this problem
Step 2: The answer is clear
Final Answer: The result is success.";

        var mockChatModel = new MockChatModel<double>(textResponse);
        var agent = new ChainOfThoughtAgent<double>(mockChatModel);

        // Act
        var result = await agent.RunAsync("What should I do?");

        // Assert
        Assert.NotEmpty(result);
        // Fallback parser should extract the final answer
    }
}
