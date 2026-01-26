using AiDotNet.Agents;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Agents;

/// <summary>
/// Integration tests for the Agents module.
/// Tests the various agent types: Agent (ReAct), ChainOfThoughtAgent, PlanAndExecuteAgent, and RAGAgent.
/// </summary>
public class AgentsIntegrationTests
{
    #region Mock Implementations

    /// <summary>
    /// Mock chat model that returns configurable responses for testing.
    /// </summary>
    private class MockChatModel : IChatModel<double>
    {
        private readonly Queue<string> _responses;
        private readonly Func<string, string>? _dynamicResponse;

        public string ModelName => "mock-model";
        public int MaxContextTokens => 4096;
        public int MaxGenerationTokens => 1024;

        public int CallCount { get; private set; }
        public List<string> ReceivedPrompts { get; } = new();

        public MockChatModel(params string[] responses)
        {
            _responses = new Queue<string>(responses);
        }

        public MockChatModel(Func<string, string> dynamicResponse)
        {
            _responses = new Queue<string>();
            _dynamicResponse = dynamicResponse;
        }

        public Task<string> GenerateResponseAsync(string prompt, CancellationToken cancellationToken = default)
        {
            CallCount++;
            ReceivedPrompts.Add(prompt);

            if (_dynamicResponse != null)
            {
                return Task.FromResult(_dynamicResponse(prompt));
            }

            return Task.FromResult(_responses.Count > 0 ? _responses.Dequeue() : "No more responses configured");
        }

        public Task<string> GenerateAsync(string prompt, CancellationToken cancellationToken = default)
        {
            return GenerateResponseAsync(prompt, cancellationToken);
        }

        public string Generate(string prompt)
        {
            return GenerateResponseAsync(prompt).GetAwaiter().GetResult();
        }
    }

    /// <summary>
    /// Mock chat model that throws exceptions for testing error handling.
    /// </summary>
    private class ErrorChatModel : IChatModel<double>
    {
        private readonly Exception _exception;

        public string ModelName => "error-model";
        public int MaxContextTokens => 4096;
        public int MaxGenerationTokens => 1024;

        public ErrorChatModel(Exception exception)
        {
            _exception = exception;
        }

        public Task<string> GenerateResponseAsync(string prompt, CancellationToken cancellationToken = default)
        {
            throw _exception;
        }

        public Task<string> GenerateAsync(string prompt, CancellationToken cancellationToken = default)
        {
            throw _exception;
        }

        public string Generate(string prompt)
        {
            throw _exception;
        }
    }

    /// <summary>
    /// Mock tool for testing agent tool execution.
    /// </summary>
    private class MockTool : ITool
    {
        private readonly Func<string, string>? _executeFunc;

        public string Name { get; }
        public string Description { get; }
        public int ExecuteCount { get; private set; }
        public List<string> ReceivedInputs { get; } = new();

        public MockTool(string name, string description, Func<string, string>? executeFunc = null)
        {
            Name = name;
            Description = description;
            _executeFunc = executeFunc;
        }

        public string Execute(string input)
        {
            ExecuteCount++;
            ReceivedInputs.Add(input);

            if (_executeFunc != null)
            {
                return _executeFunc(input);
            }

            return $"Result for: {input}";
        }
    }

    /// <summary>
    /// Mock tool that throws exceptions for testing error handling.
    /// </summary>
    private class ErrorTool : ITool
    {
        public string Name => "ErrorTool";
        public string Description => "A tool that always throws an error";

        public string Execute(string input)
        {
            throw new InvalidOperationException("Tool execution failed");
        }
    }

    /// <summary>
    /// Mock retriever for RAGAgent testing.
    /// </summary>
    private class MockRetriever : IRetriever<double>
    {
        private readonly List<Document<double>> _documents;

        public int DefaultTopK => 10;

        public MockRetriever(params Document<double>[] documents)
        {
            _documents = documents.ToList();
        }

        public IEnumerable<Document<double>> Retrieve(string query)
        {
            return Retrieve(query, DefaultTopK);
        }

        public IEnumerable<Document<double>> Retrieve(string query, int topK)
        {
            return _documents.Take(topK);
        }

        public IEnumerable<Document<double>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            // For testing, we just ignore metadata filters and return documents
            return Retrieve(query, topK);
        }
    }

    /// <summary>
    /// Mock retriever that returns no documents.
    /// </summary>
    private class EmptyRetriever : IRetriever<double>
    {
        public int DefaultTopK => 10;

        public IEnumerable<Document<double>> Retrieve(string query)
        {
            return Enumerable.Empty<Document<double>>();
        }

        public IEnumerable<Document<double>> Retrieve(string query, int topK)
        {
            return Enumerable.Empty<Document<double>>();
        }

        public IEnumerable<Document<double>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            return Enumerable.Empty<Document<double>>();
        }
    }

    /// <summary>
    /// Mock reranker for RAGAgent testing.
    /// </summary>
    private class MockReranker : IReranker<double>
    {
        public bool ModifiesScores => true;

        public IEnumerable<Document<double>> Rerank(string query, IEnumerable<Document<double>> documents)
        {
            // Simple rerank: reverse the order
            return documents.Reverse();
        }

        public IEnumerable<Document<double>> Rerank(string query, IEnumerable<Document<double>> documents, int topK)
        {
            return Rerank(query, documents).Take(topK);
        }
    }

    /// <summary>
    /// Mock generator for RAGAgent testing.
    /// </summary>
    private class MockGenerator : IGenerator<double>
    {
        public int MaxContextTokens => 4096;
        public int MaxGenerationTokens => 1024;

        public string Generate(string prompt)
        {
            return $"Generated response for: {prompt}";
        }

        public GroundedAnswer<double> GenerateGrounded(string query, IEnumerable<Document<double>> documents)
        {
            var docList = documents.ToList();
            return new GroundedAnswer<double>
            {
                Answer = $"Generated answer for: {query} using {docList.Count} documents",
                Citations = docList.Select(d => d.Id).ToList(),
                ConfidenceScore = 0.85
            };
        }
    }

    #endregion

    #region Agent (ReAct) Tests

    [Fact]
    public void Agent_Constructor_WithValidChatModel_CreatesInstance()
    {
        // Arrange
        var chatModel = new MockChatModel("test response");

        // Act
        var agent = new Agent<double>(chatModel);

        // Assert
        Assert.NotNull(agent);
        Assert.Same(chatModel, agent.ChatModel);
        Assert.Empty(agent.Tools);
    }

    [Fact]
    public void Agent_Constructor_WithNullChatModel_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new Agent<double>(null!));
    }

    [Fact]
    public void Agent_Constructor_WithTools_StoresTools()
    {
        // Arrange
        var chatModel = new MockChatModel("test response");
        var tools = new[]
        {
            new MockTool("Calculator", "Performs calculations"),
            new MockTool("Search", "Searches for information")
        };

        // Act
        var agent = new Agent<double>(chatModel, tools);

        // Assert
        Assert.Equal(2, agent.Tools.Count);
        Assert.Contains(agent.Tools, t => t.Name == "Calculator");
        Assert.Contains(agent.Tools, t => t.Name == "Search");
    }

    [Fact]
    public async Task Agent_RunAsync_WithFinalAnswerInFirstResponse_ReturnsAnswer()
    {
        // Arrange
        var jsonResponse = @"{
            ""thought"": ""I can answer this directly"",
            ""action"": """",
            ""action_input"": """",
            ""final_answer"": ""The answer is 42""
        }";
        var chatModel = new MockChatModel(jsonResponse);
        var agent = new Agent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("What is the answer to life?");

        // Assert
        Assert.Equal("The answer is 42", result);
        Assert.Contains("The answer is 42", agent.Scratchpad);
    }

    [Fact]
    public async Task Agent_RunAsync_WithToolAction_ExecutesTool()
    {
        // Arrange
        var iteration1Response = @"{
            ""thought"": ""I need to calculate this"",
            ""action"": ""Calculator"",
            ""action_input"": ""2 + 2"",
            ""final_answer"": """"
        }";
        var iteration2Response = @"{
            ""thought"": ""I have the answer now"",
            ""action"": """",
            ""action_input"": """",
            ""final_answer"": ""The result is 4""
        }";

        var chatModel = new MockChatModel(iteration1Response, iteration2Response);
        var calculator = new MockTool("Calculator", "Performs calculations", input => "4");
        var agent = new Agent<double>(chatModel, new[] { calculator });

        // Act
        var result = await agent.RunAsync("What is 2 + 2?");

        // Assert
        Assert.Equal("The result is 4", result);
        Assert.Equal(1, calculator.ExecuteCount);
        Assert.Contains("2 + 2", calculator.ReceivedInputs);
    }

    [Fact]
    public async Task Agent_RunAsync_WithNullQuery_ThrowsArgumentException()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var agent = new Agent<double>(chatModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync(null!));
    }

    [Fact]
    public async Task Agent_RunAsync_WithEmptyQuery_ThrowsArgumentException()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var agent = new Agent<double>(chatModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync("   "));
    }

    [Fact]
    public async Task Agent_RunAsync_WithZeroMaxIterations_ThrowsArgumentException()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var agent = new Agent<double>(chatModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync("query", maxIterations: 0));
    }

    [Fact]
    public async Task Agent_RunAsync_ReachingMaxIterations_ReturnsFallbackMessage()
    {
        // Arrange - No final answer, always wants to use a tool
        var neverEndingResponse = @"{
            ""thought"": ""I need to search"",
            ""action"": ""Search"",
            ""action_input"": ""query"",
            ""final_answer"": """"
        }";
        var chatModel = new MockChatModel(neverEndingResponse, neverEndingResponse, neverEndingResponse);
        var search = new MockTool("Search", "Searches", _ => "Found something");
        var agent = new Agent<double>(chatModel, new[] { search });

        // Act
        var result = await agent.RunAsync("Find information", maxIterations: 2);

        // Assert
        Assert.Contains("maximum number of iterations", result);
        Assert.Contains("2", result);
    }

    [Fact]
    public async Task Agent_RunAsync_WithNonExistentTool_ReturnsError()
    {
        // Arrange
        var response = @"{
            ""thought"": ""I need to use a tool"",
            ""action"": ""NonExistentTool"",
            ""action_input"": ""input"",
            ""final_answer"": """"
        }";
        var finalResponse = @"{
            ""thought"": ""I see the tool doesn't exist"",
            ""action"": """",
            ""action_input"": """",
            ""final_answer"": ""Could not find the tool""
        }";
        var chatModel = new MockChatModel(response, finalResponse);
        var agent = new Agent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("Do something");

        // Assert
        Assert.Contains("NonExistentTool", agent.Scratchpad);
        Assert.Contains("not found", agent.Scratchpad);
    }

    [Fact]
    public async Task Agent_RunAsync_WithHttpException_ReturnsErrorMessage()
    {
        // Arrange
        var chatModel = new ErrorChatModel(new System.Net.Http.HttpRequestException("Network error"));
        var agent = new Agent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("query");

        // Assert
        Assert.Contains("error", result.ToLower());
        Assert.Contains("Network error", result);
    }

    [Fact]
    public async Task Agent_RunAsync_WithToolExecutionError_ReturnsErrorObservation()
    {
        // Arrange
        var response = @"{
            ""thought"": ""I need to use the error tool"",
            ""action"": ""ErrorTool"",
            ""action_input"": ""input"",
            ""final_answer"": """"
        }";
        var finalResponse = @"{
            ""thought"": ""The tool errored"",
            ""action"": """",
            ""action_input"": """",
            ""final_answer"": ""Tool failed""
        }";
        var chatModel = new MockChatModel(response, finalResponse);
        var errorTool = new ErrorTool();
        var agent = new Agent<double>(chatModel, new[] { errorTool });

        // Act
        var result = await agent.RunAsync("Do something");

        // Assert
        Assert.Contains("Error executing tool", agent.Scratchpad);
    }

    [Fact]
    public async Task Agent_RunAsync_ParsesRegexFallback()
    {
        // Arrange - Non-JSON response that uses the regex fallback parser
        var response = @"
            Thought: I can answer this
            Action:
            Action Input:
            Final Answer: The answer is 42";
        var chatModel = new MockChatModel(response);
        var agent = new Agent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("What is the answer?");

        // Assert
        Assert.Equal("The answer is 42", result);
    }

    [Fact]
    public async Task Agent_Scratchpad_ClearsOnNewRun()
    {
        // Arrange
        var response = @"{""final_answer"": ""answer""}";
        var chatModel = new MockChatModel(response, response);
        var agent = new Agent<double>(chatModel);

        // Act
        await agent.RunAsync("First query");
        var scratchpadAfterFirst = agent.Scratchpad;

        await agent.RunAsync("Second query");
        var scratchpadAfterSecond = agent.Scratchpad;

        // Assert
        Assert.Contains("First query", scratchpadAfterFirst);
        Assert.DoesNotContain("First query", scratchpadAfterSecond);
        Assert.Contains("Second query", scratchpadAfterSecond);
    }

    #endregion

    #region ChainOfThoughtAgent Tests

    [Fact]
    public void ChainOfThoughtAgent_Constructor_WithValidChatModel_CreatesInstance()
    {
        // Arrange
        var chatModel = new MockChatModel("response");

        // Act
        var agent = new ChainOfThoughtAgent<double>(chatModel);

        // Assert
        Assert.NotNull(agent);
        Assert.Same(chatModel, agent.ChatModel);
    }

    [Fact]
    public void ChainOfThoughtAgent_Constructor_WithNullChatModel_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ChainOfThoughtAgent<double>(null!));
    }

    [Fact]
    public async Task ChainOfThoughtAgent_RunAsync_WithFinalAnswer_ReturnsAnswer()
    {
        // Arrange
        var response = @"{
            ""reasoning_steps"": [
                ""Step 1: Analyze the problem"",
                ""Step 2: Apply the formula""
            ],
            ""tool_calls"": [],
            ""final_answer"": ""The result is 100""
        }";
        var chatModel = new MockChatModel(response);
        var agent = new ChainOfThoughtAgent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("Calculate something");

        // Assert
        Assert.Equal("The result is 100", result);
        Assert.Contains("Step 1", agent.Scratchpad);
        Assert.Contains("Step 2", agent.Scratchpad);
    }

    [Fact]
    public async Task ChainOfThoughtAgent_RunAsync_WithToolCalls_ExecutesTools()
    {
        // Arrange
        var response = @"{
            ""reasoning_steps"": [
                ""Step 1: I need to calculate""
            ],
            ""tool_calls"": [
                {
                    ""tool_name"": ""Calculator"",
                    ""tool_input"": ""5 * 5""
                }
            ],
            ""final_answer"": """"
        }";
        var refinementResponse = @"{
            ""final_answer"": ""The result is 25""
        }";
        var chatModel = new MockChatModel(response, refinementResponse);
        var calculator = new MockTool("Calculator", "Calculates", _ => "25");
        var agent = new ChainOfThoughtAgent<double>(chatModel, new[] { calculator }, allowTools: true);

        // Act
        var result = await agent.RunAsync("What is 5 times 5?");

        // Assert
        Assert.Contains("25", result);
        Assert.Equal(1, calculator.ExecuteCount);
    }

    [Fact]
    public async Task ChainOfThoughtAgent_RunAsync_WithAllowToolsFalse_DoesNotExecuteTools()
    {
        // Arrange
        var response = @"{
            ""reasoning_steps"": [
                ""Step 1: Calculate 5 * 5 = 25""
            ],
            ""tool_calls"": [
                {
                    ""tool_name"": ""Calculator"",
                    ""tool_input"": ""5 * 5""
                }
            ],
            ""final_answer"": ""The result is 25""
        }";
        var chatModel = new MockChatModel(response);
        var calculator = new MockTool("Calculator", "Calculates", _ => "25");
        var agent = new ChainOfThoughtAgent<double>(chatModel, new[] { calculator }, allowTools: false);

        // Act
        var result = await agent.RunAsync("What is 5 times 5?");

        // Assert
        Assert.Equal("The result is 25", result);
        Assert.Equal(0, calculator.ExecuteCount); // Tool not executed
    }

    [Fact]
    public async Task ChainOfThoughtAgent_RunAsync_TruncatesStepsExceedingMaxIterations()
    {
        // Arrange - Response with more steps than maxIterations allows
        var response = @"{
            ""reasoning_steps"": [
                ""Step 1"",
                ""Step 2"",
                ""Step 3"",
                ""Step 4"",
                ""Step 5""
            ],
            ""tool_calls"": [],
            ""final_answer"": ""Answer""
        }";
        var chatModel = new MockChatModel(response);
        var agent = new ChainOfThoughtAgent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("Query", maxIterations: 3);

        // Assert
        Assert.Contains("truncating to 3", agent.Scratchpad);
    }

    [Fact]
    public async Task ChainOfThoughtAgent_RunAsync_WithEmptyQuery_ThrowsArgumentException()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var agent = new ChainOfThoughtAgent<double>(chatModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync(""));
    }

    [Fact]
    public async Task ChainOfThoughtAgent_RunAsync_WithHttpException_ReturnsErrorMessage()
    {
        // Arrange
        var chatModel = new ErrorChatModel(new System.Net.Http.HttpRequestException("Network error"));
        var agent = new ChainOfThoughtAgent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("query");

        // Assert
        Assert.Contains("error", result.ToLower());
    }

    [Fact]
    public async Task ChainOfThoughtAgent_RunAsync_ParsesRegexFallback()
    {
        // Arrange - Non-JSON response
        var response = @"
            Step 1: First step
            Step 2: Second step
            Final Answer: The result is 42";
        var chatModel = new MockChatModel(response);
        var agent = new ChainOfThoughtAgent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("What is the answer?");

        // Assert
        Assert.Contains("42", result);
    }

    #endregion

    #region PlanAndExecuteAgent Tests

    [Fact]
    public void PlanAndExecuteAgent_Constructor_WithValidChatModel_CreatesInstance()
    {
        // Arrange
        var chatModel = new MockChatModel("response");

        // Act
        var agent = new PlanAndExecuteAgent<double>(chatModel);

        // Assert
        Assert.NotNull(agent);
        Assert.Same(chatModel, agent.ChatModel);
    }

    [Fact]
    public void PlanAndExecuteAgent_Constructor_WithNullChatModel_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new PlanAndExecuteAgent<double>(null!));
    }

    [Fact]
    public async Task PlanAndExecuteAgent_RunAsync_ExecutesPlan()
    {
        // Arrange
        var planResponse = @"{
            ""steps"": [
                {
                    ""description"": ""Search for information"",
                    ""tool"": ""Search"",
                    ""input"": ""test query"",
                    ""is_final_step"": false
                },
                {
                    ""description"": ""Provide the final answer"",
                    ""tool"": """",
                    ""input"": """",
                    ""is_final_step"": true
                }
            ]
        }";
        var finalAnswer = "The answer based on search results";

        var chatModel = new MockChatModel(planResponse, finalAnswer);
        var search = new MockTool("Search", "Searches", _ => "Search result");
        var agent = new PlanAndExecuteAgent<double>(chatModel, new[] { search });

        // Act
        var result = await agent.RunAsync("Find information");

        // Assert
        Assert.Contains("answer", result.ToLower());
        Assert.Equal(1, search.ExecuteCount);
        Assert.Contains("PLANNING PHASE", agent.Scratchpad);
        Assert.Contains("EXECUTION PHASE", agent.Scratchpad);
    }

    [Fact]
    public async Task PlanAndExecuteAgent_RunAsync_WithEmptyPlan_ReturnsErrorMessage()
    {
        // Arrange
        var emptyPlanResponse = @"{ ""steps"": [] }";
        var chatModel = new MockChatModel(emptyPlanResponse);
        var agent = new PlanAndExecuteAgent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("Do something");

        // Assert
        Assert.Contains("unable to create a plan", result);
    }

    [Fact]
    public async Task PlanAndExecuteAgent_RunAsync_WithPlanRevision_RevisesPlan()
    {
        // Arrange - First plan has a step that will fail, then revision succeeds
        var firstPlan = @"{
            ""steps"": [
                {
                    ""description"": ""Use error tool"",
                    ""tool"": ""ErrorTool"",
                    ""input"": ""input"",
                    ""is_final_step"": false
                }
            ]
        }";
        var revisedPlan = @"{
            ""steps"": [
                {
                    ""description"": ""Provide answer directly"",
                    ""tool"": """",
                    ""input"": """",
                    ""is_final_step"": true
                }
            ]
        }";
        var finalAnswer = "Final answer after revision";

        var chatModel = new MockChatModel(firstPlan, revisedPlan, finalAnswer);
        var errorTool = new MockTool("ErrorTool", "Always errors", _ => throw new System.Net.Http.HttpRequestException("Error"));
        var agent = new PlanAndExecuteAgent<double>(chatModel, new[] { errorTool }, allowPlanRevision: true);

        // Act
        var result = await agent.RunAsync("Do something");

        // Assert
        Assert.Contains("Attempting to revise plan", agent.Scratchpad);
        Assert.Contains("Plan revised", agent.Scratchpad);
    }

    [Fact]
    public async Task PlanAndExecuteAgent_RunAsync_WithPlanRevisionDisabled_DoesNotRevise()
    {
        // Arrange
        var plan = @"{
            ""steps"": [
                {
                    ""description"": ""Use error tool"",
                    ""tool"": ""ErrorTool"",
                    ""input"": ""input"",
                    ""is_final_step"": false
                }
            ]
        }";

        var chatModel = new MockChatModel(plan);
        var errorTool = new MockTool("ErrorTool", "Always errors", _ => throw new System.Net.Http.HttpRequestException("Error"));
        var agent = new PlanAndExecuteAgent<double>(chatModel, new[] { errorTool }, allowPlanRevision: false);

        // Act
        var result = await agent.RunAsync("Do something");

        // Assert
        Assert.Contains("encountered an error", result);
        Assert.DoesNotContain("Attempting to revise plan", agent.Scratchpad);
    }

    [Fact]
    public async Task PlanAndExecuteAgent_RunAsync_WithMaxRevisionsReached_StopsRevising()
    {
        // Arrange - Plans that always fail
        var failingPlan = @"{
            ""steps"": [
                {
                    ""description"": ""Use error tool"",
                    ""tool"": ""ErrorTool"",
                    ""input"": ""input"",
                    ""is_final_step"": false
                }
            ]
        }";

        // Return failing plans repeatedly
        var chatModel = new MockChatModel(failingPlan, failingPlan, failingPlan, failingPlan, failingPlan);
        var errorTool = new MockTool("ErrorTool", "Always errors", _ => throw new System.Net.Http.HttpRequestException("Error"));
        var agent = new PlanAndExecuteAgent<double>(chatModel, new[] { errorTool }, allowPlanRevision: true);

        // Act
        var result = await agent.RunAsync("Do something", maxIterations: 2);

        // Assert
        Assert.Contains("maximum number of plan revisions", result);
    }

    [Fact]
    public async Task PlanAndExecuteAgent_RunAsync_WithEmptyQuery_ThrowsArgumentException()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var agent = new PlanAndExecuteAgent<double>(chatModel);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync(""));
    }

    #endregion

    #region RAGAgent Tests

    [Fact]
    public void RAGAgent_Constructor_WithValidComponents_CreatesInstance()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var retriever = new MockRetriever();
        var generator = new MockGenerator();

        // Act
        var agent = new RAGAgent<double>(chatModel, retriever, generator);

        // Assert
        Assert.NotNull(agent);
    }

    [Fact]
    public void RAGAgent_Constructor_WithNullRetriever_ThrowsArgumentNullException()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var generator = new MockGenerator();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new RAGAgent<double>(chatModel, null!, generator));
    }

    [Fact]
    public void RAGAgent_Constructor_WithNullGenerator_ThrowsArgumentNullException()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var retriever = new MockRetriever();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new RAGAgent<double>(chatModel, retriever, null!));
    }

    [Fact]
    public void RAGAgent_Constructor_WithInvalidRetrievalTopK_ThrowsArgumentException()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var retriever = new MockRetriever();
        var generator = new MockGenerator();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new RAGAgent<double>(chatModel, retriever, generator, retrievalTopK: 0));
    }

    [Fact]
    public void RAGAgent_Constructor_WithInvalidRerankTopK_ThrowsArgumentException()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var retriever = new MockRetriever();
        var generator = new MockGenerator();
        var reranker = new MockReranker();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new RAGAgent<double>(chatModel, retriever, generator, reranker, rerankTopK: 0));
    }

    [Fact]
    public async Task RAGAgent_RunAsync_RetrievesAndGenerates()
    {
        // Arrange
        var documents = new[]
        {
            new Document<double> { Id = "doc1", Content = "Content 1" },
            new Document<double> { Id = "doc2", Content = "Content 2" }
        };

        var chatModel = new MockChatModel("refined query");
        var retriever = new MockRetriever(documents);
        var generator = new MockGenerator();
        var agent = new RAGAgent<double>(chatModel, retriever, generator, retrievalTopK: 5);

        // Act
        var result = await agent.RunAsync("What is this about?");

        // Assert
        Assert.Contains("Generated answer", result);
        Assert.Contains("2 documents", result);
        Assert.Contains("RETRIEVAL PHASE", agent.Scratchpad);
        Assert.Contains("GENERATION PHASE", agent.Scratchpad);
    }

    [Fact]
    public async Task RAGAgent_RunAsync_WithReranker_ReranksDocuments()
    {
        // Arrange
        var documents = new[]
        {
            new Document<double> { Id = "doc1", Content = "Content 1" },
            new Document<double> { Id = "doc2", Content = "Content 2" }
        };

        var chatModel = new MockChatModel("refined query");
        var retriever = new MockRetriever(documents);
        var generator = new MockGenerator();
        var reranker = new MockReranker();
        var agent = new RAGAgent<double>(chatModel, retriever, generator, reranker, rerankTopK: 1);

        // Act
        var result = await agent.RunAsync("Query");

        // Assert
        Assert.Contains("RERANKING PHASE", agent.Scratchpad);
        Assert.Contains("1 documents after reranking", agent.Scratchpad);
    }

    [Fact]
    public async Task RAGAgent_RunAsync_WithNoDocuments_ReturnsNoDocumentsMessage()
    {
        // Arrange
        var chatModel = new MockChatModel("refined query");
        var retriever = new EmptyRetriever();
        var generator = new MockGenerator();
        var agent = new RAGAgent<double>(chatModel, retriever, generator);

        // Act
        var result = await agent.RunAsync("Query about nothing");

        // Assert
        Assert.Contains("couldn't find any relevant information", result);
    }

    [Fact]
    public async Task RAGAgent_RunAsync_WithCitations_IncludesCitations()
    {
        // Arrange
        var documents = new[]
        {
            new Document<double> { Id = "doc1", Content = "Content 1" }
        };

        var chatModel = new MockChatModel("refined query");
        var retriever = new MockRetriever(documents);
        var generator = new MockGenerator();
        var agent = new RAGAgent<double>(chatModel, retriever, generator, includeCitations: true);

        // Act
        var result = await agent.RunAsync("Query");

        // Assert
        Assert.Contains("Sources:", result);
        Assert.Contains("[1]", result);
    }

    [Fact]
    public async Task RAGAgent_RunAsync_WithoutCitations_DoesNotIncludeCitations()
    {
        // Arrange
        var documents = new[]
        {
            new Document<double> { Id = "doc1", Content = "Content 1" }
        };

        var chatModel = new MockChatModel("refined query");
        var retriever = new MockRetriever(documents);
        var generator = new MockGenerator();
        var agent = new RAGAgent<double>(chatModel, retriever, generator, includeCitations: false);

        // Act
        var result = await agent.RunAsync("Query");

        // Assert
        Assert.DoesNotContain("Sources:", result);
    }

    [Fact]
    public async Task RAGAgent_RunAsync_WithQueryRefinement_RefinesQuery()
    {
        // Arrange
        var documents = new[]
        {
            new Document<double> { Id = "doc1", Content = "Content 1" }
        };

        var chatModel = new MockChatModel("How do I reset my password?"); // Refined query
        var retriever = new MockRetriever(documents);
        var generator = new MockGenerator();
        var agent = new RAGAgent<double>(chatModel, retriever, generator, allowQueryRefinement: true);

        // Act
        var result = await agent.RunAsync("How do I reset it?");

        // Assert
        Assert.Contains("QUERY ANALYSIS", agent.Scratchpad);
    }

    [Fact]
    public async Task RAGAgent_RunAsync_WithEmptyQuery_ThrowsArgumentException()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var retriever = new MockRetriever();
        var generator = new MockGenerator();
        var agent = new RAGAgent<double>(chatModel, retriever, generator);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync(""));
    }

    [Fact]
    public void RAGAgent_GetPipelineInfo_ReturnsConfiguration()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var retriever = new MockRetriever();
        var generator = new MockGenerator();
        var reranker = new MockReranker();
        var agent = new RAGAgent<double>(chatModel, retriever, generator, reranker,
            retrievalTopK: 10, rerankTopK: 5, includeCitations: true);

        // Act
        var info = agent.GetPipelineInfo();

        // Assert
        Assert.Contains("RAG Pipeline Configuration", info);
        Assert.Contains("Retrieval TopK: 10", info);
        Assert.Contains("Rerank TopK: 5", info);
        Assert.Contains("Include Citations: True", info);
    }

    #endregion

    #region AgentBase Tests (via concrete implementations)

    [Fact]
    public void AgentBase_Tools_ReturnsReadOnlyList()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var tools = new[] { new MockTool("Tool", "Description") };
        var agent = new Agent<double>(chatModel, tools);

        // Act
        var toolList = agent.Tools;

        // Assert
        Assert.IsAssignableFrom<IReadOnlyList<ITool>>(toolList);
        Assert.Single(toolList);
    }

    [Fact]
    public void AgentBase_Tools_WithNullTools_ReturnsEmptyList()
    {
        // Arrange
        var chatModel = new MockChatModel("response");
        var agent = new Agent<double>(chatModel, null);

        // Act
        var toolList = agent.Tools;

        // Assert
        Assert.Empty(toolList);
    }

    [Fact]
    public async Task AgentBase_Scratchpad_TracksReasoningHistory()
    {
        // Arrange
        var response = @"{""final_answer"": ""answer""}";
        var chatModel = new MockChatModel(response);
        var agent = new Agent<double>(chatModel);

        // Act
        await agent.RunAsync("Test query");

        // Assert
        var scratchpad = agent.Scratchpad;
        Assert.Contains("Query: Test query", scratchpad);
        Assert.Contains("Iteration 1", scratchpad);
        Assert.Contains("Final Answer", scratchpad);
    }

    #endregion

    #region JSON Parsing Edge Cases

    [Fact]
    public async Task Agent_RunAsync_ParsesJsonInMarkdownCodeBlock()
    {
        // Arrange
        var response = @"```json
{
    ""thought"": ""I can answer this"",
    ""action"": """",
    ""action_input"": """",
    ""final_answer"": ""The answer is 42""
}
```";
        var chatModel = new MockChatModel(response);
        var agent = new Agent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("What is the answer?");

        // Assert
        Assert.Equal("The answer is 42", result);
    }

    [Fact]
    public async Task Agent_RunAsync_ParsesJsonWithoutCodeBlock()
    {
        // Arrange
        var response = @"Here is my response:
{
    ""thought"": ""I can answer this"",
    ""action"": """",
    ""action_input"": """",
    ""final_answer"": ""The answer is 42""
}";
        var chatModel = new MockChatModel(response);
        var agent = new Agent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("What is the answer?");

        // Assert
        Assert.Equal("The answer is 42", result);
    }

    [Fact]
    public async Task Agent_RunAsync_HandlesEmptyResponse()
    {
        // Arrange
        var response = "";
        var finalResponse = @"{""final_answer"": ""Recovered answer""}";
        var chatModel = new MockChatModel(response, finalResponse);
        var agent = new Agent<double>(chatModel);

        // Act
        var result = await agent.RunAsync("Query", maxIterations: 2);

        // Assert
        // Either recovers or returns max iterations message
        Assert.True(result.Contains("Recovered") || result.Contains("maximum"));
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public async Task Agent_ConcurrentRuns_AreIsolated()
    {
        // Arrange
        var chatModel = new MockChatModel(prompt =>
        {
            // Return different answers based on the query
            if (prompt.Contains("Query A"))
                return @"{""final_answer"": ""Answer A""}";
            else
                return @"{""final_answer"": ""Answer B""}";
        });

        var agentA = new Agent<double>(chatModel);
        var agentB = new Agent<double>(chatModel);

        // Act
        var taskA = agentA.RunAsync("Query A");
        var taskB = agentB.RunAsync("Query B");

        var results = await Task.WhenAll(taskA, taskB);

        // Assert
        Assert.Equal("Answer A", results[0]);
        Assert.Equal("Answer B", results[1]);
    }

    #endregion
}
