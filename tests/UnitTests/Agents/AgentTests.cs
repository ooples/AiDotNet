using AiDotNet.Agents;
using AiDotNet.Interfaces;
using AiDotNet.Tools;
using Xunit;

namespace AiDotNetTests.UnitTests.Agents;

/// <summary>
/// Unit tests for the Agent class.
/// </summary>
public class AgentTests
{
    [Fact]
    public void Constructor_WithValidParameters_InitializesSuccessfully()
    {
        // Arrange
        var mockModel = new MockChatModel<double>();
        var tools = new List<ITool> { new CalculatorTool() };

        // Act
        var agent = new Agent<double>(mockModel, tools);

        // Assert
        Assert.NotNull(agent);
        Assert.Equal(mockModel, agent.ChatModel);
        Assert.Single(agent.Tools);
    }

    [Fact]
    public void Constructor_WithNullChatModel_ThrowsArgumentNullException()
    {
        // Arrange
        IChatModel<double> nullModel = null!;
        var tools = new List<ITool> { new CalculatorTool() };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new Agent<double>(nullModel, tools));
    }

    [Fact]
    public void Constructor_WithNullTools_InitializesWithEmptyToolList()
    {
        // Arrange
        var mockModel = new MockChatModel<double>();

        // Act
        var agent = new Agent<double>(mockModel, null);

        // Assert
        Assert.NotNull(agent);
        Assert.Empty(agent.Tools);
    }

    [Fact]
    public async Task RunAsync_WithNullQuery_ThrowsArgumentException()
    {
        // Arrange
        var mockModel = new MockChatModel<double>();
        var agent = new Agent<double>(mockModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync(null!));
    }

    [Fact]
    public async Task RunAsync_WithEmptyQuery_ThrowsArgumentException()
    {
        // Arrange
        var mockModel = new MockChatModel<double>();
        var agent = new Agent<double>(mockModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync(""));
    }

    [Fact]
    public async Task RunAsync_WithWhitespaceQuery_ThrowsArgumentException()
    {
        // Arrange
        var mockModel = new MockChatModel<double>();
        var agent = new Agent<double>(mockModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync("   "));
    }

    [Fact]
    public async Task RunAsync_WithInvalidMaxIterations_ThrowsArgumentException()
    {
        // Arrange
        var mockModel = new MockChatModel<double>();
        var agent = new Agent<double>(mockModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync("test query", maxIterations: 0));
        await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync("test query", maxIterations: -1));
    }

    [Fact]
    public async Task RunAsync_WithFinalAnswerInFirstIteration_ReturnsFinalAnswer()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            "{\"thought\": \"I can answer this directly\", \"final_answer\": \"42\"}"
        );
        var agent = new Agent<double>(mockModel);

        // Act
        var result = await agent.RunAsync("What is the answer?");

        // Assert
        Assert.Equal("42", result);
        Assert.Contains("42", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithCalculatorTool_ExecutesToolAndReturnsFinalAnswer()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            // First iteration: use calculator
            "{\"thought\": \"I need to calculate\", \"action\": \"Calculator\", \"action_input\": \"5 + 3\"}",
            // Second iteration: provide final answer
            "{\"thought\": \"I have the result\", \"final_answer\": \"8\"}"
        );
        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<double>(mockModel, tools);

        // Act
        var result = await agent.RunAsync("What is 5 + 3?");

        // Assert
        Assert.Equal("8", result);
        Assert.Contains("Calculator", agent.Scratchpad);
        Assert.Contains("Observation: 8", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithMultipleToolCalls_ExecutesAllToolsCorrectly()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            // First: calculate 5 * 4
            "{\"thought\": \"First multiply\", \"action\": \"Calculator\", \"action_input\": \"5 * 4\"}",
            // Second: add 10
            "{\"thought\": \"Now add 10\", \"action\": \"Calculator\", \"action_input\": \"20 + 10\"}",
            // Third: final answer
            "{\"thought\": \"Done\", \"final_answer\": \"30\"}"
        );
        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<double>(mockModel, tools);

        // Act
        var result = await agent.RunAsync("Calculate (5 * 4) + 10");

        // Assert
        Assert.Equal("30", result);
        Assert.Contains("Observation: 20", agent.Scratchpad);
        Assert.Contains("Observation: 30", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithNonExistentTool_ReturnsToolNotFoundError()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            "{\"thought\": \"I'll use a tool\", \"action\": \"NonExistentTool\", \"action_input\": \"test\"}",
            "{\"thought\": \"Tool not found\", \"final_answer\": \"I couldn't complete the task\"}"
        );
        var agent = new Agent<double>(mockModel);

        // Act
        var result = await agent.RunAsync("Do something");

        // Assert
        Assert.Contains("I couldn't complete the task", result);
        Assert.Contains("Tool 'NonExistentTool' not found", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithSearchTool_ExecutesSearchAndReturnsAnswer()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            "{\"thought\": \"I'll search for this\", \"action\": \"Search\", \"action_input\": \"capital of France\"}",
            "{\"thought\": \"Found it\", \"final_answer\": \"Paris\"}"
        );
        var tools = new List<ITool> { new SearchTool() };
        var agent = new Agent<double>(mockModel, tools);

        // Act
        var result = await agent.RunAsync("What is the capital of France?");

        // Assert
        Assert.Equal("Paris", result);
        Assert.Contains("Paris", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithBothTools_CanUseBothSuccessfully()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            "{\"thought\": \"Search first\", \"action\": \"Search\", \"action_input\": \"speed of light\"}",
            "{\"thought\": \"Now calculate\", \"action\": \"Calculator\", \"action_input\": \"299792458 / 1000\"}",
            "{\"thought\": \"Done\", \"final_answer\": \"299792.458 km/s\"}"
        );
        var tools = new List<ITool> { new SearchTool(), new CalculatorTool() };
        var agent = new Agent<double>(mockModel, tools);

        // Act
        var result = await agent.RunAsync("Find the speed of light and divide by 1000");

        // Assert
        Assert.Equal("299792.458 km/s", result);
        Assert.Contains("Search", agent.Scratchpad);
        Assert.Contains("Calculator", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_ReachingMaxIterationsWithoutAnswer_ReturnsPartialResults()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            "{\"thought\": \"Step 1\", \"action\": \"Calculator\", \"action_input\": \"1 + 1\"}",
            "{\"thought\": \"Step 2\", \"action\": \"Calculator\", \"action_input\": \"2 + 2\"}",
            "{\"thought\": \"Step 3\", \"action\": \"Calculator\", \"action_input\": \"3 + 3\"}"
        );
        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<double>(mockModel, tools);

        // Act
        var result = await agent.RunAsync("Complex task", maxIterations: 3);

        // Assert
        Assert.Contains("maximum number of iterations", result);
        Assert.Contains("Observation: 2", agent.Scratchpad);
        Assert.Contains("Observation: 4", agent.Scratchpad);
        Assert.Contains("Observation: 6", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_ClearsScratchpadBetweenRuns()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            "{\"thought\": \"First run\", \"final_answer\": \"Result 1\"}",
            "{\"thought\": \"Second run\", \"final_answer\": \"Result 2\"}"
        );
        var agent = new Agent<double>(mockModel);

        // Act
        var result1 = await agent.RunAsync("Query 1");
        var scratchpad1 = agent.Scratchpad;
        var result2 = await agent.RunAsync("Query 2");
        var scratchpad2 = agent.Scratchpad;

        // Assert
        Assert.Equal("Result 1", result1);
        Assert.Equal("Result 2", result2);
        Assert.Contains("Query 1", scratchpad1);
        Assert.DoesNotContain("Query 1", scratchpad2);
        Assert.Contains("Query 2", scratchpad2);
    }

    [Fact]
    public async Task RunAsync_WithMarkdownWrappedJSON_ParsesCorrectly()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            "```json\n{\"thought\": \"Testing\", \"final_answer\": \"Success\"}\n```"
        );
        var agent = new Agent<double>(mockModel);

        // Act
        var result = await agent.RunAsync("Test query");

        // Assert
        Assert.Equal("Success", result);
    }

    [Fact]
    public async Task RunAsync_WithNonJSONResponse_ParsesWithRegex()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            "Thought: I can answer this\nFinal Answer: 42"
        );
        var agent = new Agent<double>(mockModel);

        // Act
        var result = await agent.RunAsync("What is the answer?");

        // Assert
        Assert.Equal("42", result);
    }

    [Fact]
    public async Task RunAsync_RecordsIterationNumbers()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            "{\"thought\": \"Step 1\", \"action\": \"Calculator\", \"action_input\": \"1 + 1\"}",
            "{\"thought\": \"Step 2\", \"final_answer\": \"Done\"}"
        );
        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<double>(mockModel, tools);

        // Act
        await agent.RunAsync("Test");

        // Assert
        Assert.Contains("=== Iteration 1 ===", agent.Scratchpad);
        Assert.Contains("=== Iteration 2 ===", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_RecordsThoughtsActionsAndObservations()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            "{\"thought\": \"I need to calculate this\", \"action\": \"Calculator\", \"action_input\": \"10 * 5\"}",
            "{\"thought\": \"Got the result\", \"final_answer\": \"50\"}"
        );
        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<double>(mockModel, tools);

        // Act
        await agent.RunAsync("What is 10 * 5?");

        // Assert
        Assert.Contains("Thought: I need to calculate this", agent.Scratchpad);
        Assert.Contains("Action: Calculator", agent.Scratchpad);
        Assert.Contains("Action Input: 10 * 5", agent.Scratchpad);
        Assert.Contains("Observation: 50", agent.Scratchpad);
        Assert.Contains("Thought: Got the result", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_SendsToolDescriptionsToModel()
    {
        // Arrange
        var mockModel = new MockChatModel<double>(
            "{\"final_answer\": \"Done\"}"
        );
        var tools = new List<ITool> { new CalculatorTool(), new SearchTool() };
        var agent = new Agent<double>(mockModel, tools);

        // Act
        await agent.RunAsync("Test query");

        // Assert
        var sentPrompt = mockModel.ReceivedPrompts[0];
        Assert.Contains("Calculator", sentPrompt);
        Assert.Contains("Search", sentPrompt);
        Assert.Contains("mathematical", sentPrompt, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RunAsync_WithFloatType_WorksCorrectly()
    {
        // Arrange
        var mockModel = new MockChatModel<float>(
            "{\"thought\": \"Calculate\", \"action\": \"Calculator\", \"action_input\": \"3.14 * 2\"}",
            "{\"final_answer\": \"6.28\"}"
        );
        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<float>(mockModel, tools);

        // Act
        var result = await agent.RunAsync("Double pi");

        // Assert
        Assert.Equal("6.28", result);
    }

    [Fact]
    public async Task RunAsync_WithDecimalType_WorksCorrectly()
    {
        // Arrange
        var mockModel = new MockChatModel<decimal>(
            "{\"final_answer\": \"Success\"}"
        );
        var agent = new Agent<decimal>(mockModel);

        // Act
        var result = await agent.RunAsync("Test");

        // Assert
        Assert.Equal("Success", result);
    }

    [Fact]
    public void ChatModel_Property_ReturnsCorrectModel()
    {
        // Arrange
        var mockModel = new MockChatModel<double>();
        var agent = new Agent<double>(mockModel);

        // Act
        var chatModel = agent.ChatModel;

        // Assert
        Assert.Same(mockModel, chatModel);
    }

    [Fact]
    public void Tools_Property_ReturnsReadOnlyList()
    {
        // Arrange
        var mockModel = new MockChatModel<double>();
        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<double>(mockModel, tools);

        // Act
        var agentTools = agent.Tools;

        // Assert
        Assert.IsAssignableFrom<IReadOnlyList<ITool>>(agentTools);
        Assert.Single(agentTools);
    }

    [Fact]
    public void Scratchpad_InitiallyEmpty()
    {
        // Arrange
        var mockModel = new MockChatModel<double>();
        var agent = new Agent<double>(mockModel);

        // Act
        var scratchpad = agent.Scratchpad;

        // Assert
        Assert.Empty(scratchpad);
    }
}
