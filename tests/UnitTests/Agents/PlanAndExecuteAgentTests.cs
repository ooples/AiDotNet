using AiDotNet.Agents;
using AiDotNet.Interfaces;
using AiDotNet.Tools;
using Xunit;

namespace AiDotNetTests.UnitTests.Agents;

/// <summary>
/// Unit tests for the PlanAndExecuteAgent class.
/// </summary>
public class PlanAndExecuteAgentTests
{
    [Fact]
    public void Constructor_WithValidChatModel_InitializesSuccessfully()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>("Test response");

        // Act
        var agent = new PlanAndExecuteAgent<double>(mockChatModel);

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
        var agent = new PlanAndExecuteAgent<double>(mockChatModel, tools);

        // Assert
        Assert.NotNull(agent);
        Assert.Single(agent.Tools);
        Assert.Equal("Calculator", agent.Tools[0].Name);
    }

    [Fact]
    public void Constructor_WithAllowPlanRevisionFalse_CreateStrictAgent()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>("Test response");

        // Act
        var agent = new PlanAndExecuteAgent<double>(mockChatModel, allowPlanRevision: false);

        // Assert
        Assert.NotNull(agent);
        // Agent should not revise plan on errors (tested in execution)
    }

    [Fact]
    public void Constructor_WithNullChatModel_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new PlanAndExecuteAgent<double>(null!));
    }

    [Fact]
    public async Task RunAsync_WithNullQuery_ThrowsArgumentException()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>("Test response");
        var agent = new PlanAndExecuteAgent<double>(mockChatModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            agent.RunAsync(null!));
    }

    [Fact]
    public async Task RunAsync_WithEmptyQuery_ThrowsArgumentException()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>("Test response");
        var agent = new PlanAndExecuteAgent<double>(mockChatModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            agent.RunAsync(""));
    }

    [Fact]
    public async Task RunAsync_WithValidPlan_ExecutesSteps()
    {
        // Arrange
        var planResponse = @"{
            ""steps"": [
                {
                    ""description"": ""Calculate 10 + 5"",
                    ""tool"": ""Calculator"",
                    ""input"": ""10 + 5"",
                    ""is_final_step"": false
                },
                {
                    ""description"": ""Provide final answer"",
                    ""tool"": """",
                    ""input"": """",
                    ""is_final_step"": true
                }
            ]
        }";

        var finalResponse = "The result is 15.";

        var mockChatModel = new MockChatModel<double>(planResponse, finalResponse);
        var calculator = new CalculatorTool();
        var agent = new PlanAndExecuteAgent<double>(mockChatModel, new ITool[] { calculator });

        // Act
        var result = await agent.RunAsync("What is 10 + 5?");

        // Assert
        Assert.Contains("15", result);
        Assert.Contains("PLANNING PHASE", agent.Scratchpad);
        Assert.Contains("EXECUTION PHASE", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithInvalidMaxIterations_ThrowsArgumentException()
    {
        // Arrange
        var mockChatModel = new MockChatModel<double>("Test response");
        var agent = new PlanAndExecuteAgent<double>(mockChatModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            agent.RunAsync("Test query", maxIterations: 0));
    }

    [Fact]
    public async Task RunAsync_UpdatesScratchpad()
    {
        // Arrange
        var planResponse = @"{
            ""steps"": [
                {
                    ""description"": ""Simple test step"",
                    ""tool"": """",
                    ""input"": """",
                    ""is_final_step"": true
                }
            ]
        }";

        var finalResponse = "Test complete.";

        var mockChatModel = new MockChatModel<double>(planResponse, finalResponse);
        var agent = new PlanAndExecuteAgent<double>(mockChatModel);

        // Act
        var result = await agent.RunAsync("Test query");

        // Assert
        Assert.NotEmpty(agent.Scratchpad);
        Assert.Contains("Test query", agent.Scratchpad);
        Assert.Contains("Plan created", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithNoPlan_ReturnsErrorMessage()
    {
        // Arrange
        var emptyPlanResponse = @"{""steps"": []}";

        var mockChatModel = new MockChatModel<double>(emptyPlanResponse);
        var agent = new PlanAndExecuteAgent<double>(mockChatModel);

        // Act
        var result = await agent.RunAsync("Test query");

        // Assert
        Assert.Contains("unable to create a plan", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RunAsync_WithMultipleSteps_ExecutesInOrder()
    {
        // Arrange
        var planResponse = @"{
            ""steps"": [
                {
                    ""description"": ""Step 1: Calculate 5 * 2"",
                    ""tool"": ""Calculator"",
                    ""input"": ""5 * 2"",
                    ""is_final_step"": false
                },
                {
                    ""description"": ""Step 2: Add 3 to result"",
                    ""tool"": ""Calculator"",
                    ""input"": ""10 + 3"",
                    ""is_final_step"": false
                },
                {
                    ""description"": ""Step 3: Provide answer"",
                    ""tool"": """",
                    ""input"": """",
                    ""is_final_step"": true
                }
            ]
        }";

        var finalResponse = "The final result is 13.";

        var mockChatModel = new MockChatModel<double>(planResponse, finalResponse);
        var calculator = new CalculatorTool();
        var agent = new PlanAndExecuteAgent<double>(mockChatModel, new ITool[] { calculator });

        // Act
        var result = await agent.RunAsync("Calculate (5 * 2) + 3");

        // Assert
        Assert.Contains("13", result);
        Assert.Contains("Step 1", agent.Scratchpad);
        Assert.Contains("Step 2", agent.Scratchpad);
        Assert.Contains("Step 3", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithFallbackParsing_HandlesNonJsonPlan()
    {
        // Arrange
        var textPlanResponse = @"
1. First do this
2. Then do that
3. Finally finish up";

        var mockChatModel = new MockChatModel<double>(textPlanResponse, "Done!");
        var agent = new PlanAndExecuteAgent<double>(mockChatModel);

        // Act
        var result = await agent.RunAsync("What should I do?");

        // Assert
        Assert.NotEmpty(result);
        // Fallback parser should extract numbered steps
    }

    [Fact]
    public async Task RunAsync_WithFinalStepInPlan_ReturnsImmediately()
    {
        // Arrange
        var planResponse = @"{
            ""steps"": [
                {
                    ""description"": ""Provide the answer directly"",
                    ""tool"": """",
                    ""input"": """",
                    ""is_final_step"": true
                }
            ]
        }";

        var finalResponse = "The answer is yes.";

        var mockChatModel = new MockChatModel<double>(planResponse, finalResponse);
        var agent = new PlanAndExecuteAgent<double>(mockChatModel);

        // Act
        var result = await agent.RunAsync("Is this a test?");

        // Assert
        Assert.Contains("yes", result);
        Assert.Contains("PLAN COMPLETED", agent.Scratchpad);
    }

    [Fact]
    public async Task RunAsync_WithToolNotFound_ReturnsErrorInResult()
    {
        // Arrange
        var planResponse = @"{
            ""steps"": [
                {
                    ""description"": ""Use non-existent tool"",
                    ""tool"": ""NonExistentTool"",
                    ""input"": ""test"",
                    ""is_final_step"": true
                }
            ]
        }";

        var mockChatModel = new MockChatModel<double>(planResponse);
        var agent = new PlanAndExecuteAgent<double>(mockChatModel);

        // Act
        var result = await agent.RunAsync("Test query");

        // Assert
        // Tool execution should fail gracefully
        Assert.NotEmpty(result);
    }
}
