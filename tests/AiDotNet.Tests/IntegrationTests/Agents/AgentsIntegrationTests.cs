using AiDotNet.Agents;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Agents
{
    /// <summary>
    /// Comprehensive integration tests for all Agent types in the AiDotNet library.
    /// Tests agent initialization, workflow orchestration, tool execution, and error handling.
    /// Uses mock implementations to avoid external API calls.
    /// </summary>
    public class AgentsIntegrationTests
    {
        #region Mock Implementations

        /// <summary>
        /// Mock chat model that returns predefined responses for testing.
        /// </summary>
        private class MockChatModel<T> : IChatModel<T>
        {
            private readonly Queue<string> _responses;
            private readonly bool _shouldThrowError;
            private readonly Exception? _errorToThrow;

            public string ModelName { get; } = "mock-model";
            public int MaxContextTokens { get; } = 4096;
            public int MaxGenerationTokens { get; } = 1024;

            public MockChatModel(params string[] responses)
            {
                _responses = new Queue<string>(responses);
                _shouldThrowError = false;
            }

            public MockChatModel(Exception error)
            {
                _responses = new Queue<string>();
                _shouldThrowError = true;
                _errorToThrow = error;
            }

            public Task<string> GenerateResponseAsync(string prompt)
            {
                if (_shouldThrowError && _errorToThrow != null)
                {
                    throw _errorToThrow;
                }

                if (_responses.Count == 0)
                {
                    return Task.FromResult("Default response");
                }

                return Task.FromResult(_responses.Dequeue());
            }

            public Task<string> GenerateAsync(string prompt) => GenerateResponseAsync(prompt);
            public string Generate(string prompt) => GenerateResponseAsync(prompt).GetAwaiter().GetResult();
        }

        /// <summary>
        /// Mock tool for testing tool execution in agents.
        /// </summary>
        private class MockTool : ITool
        {
            public string Name { get; }
            public string Description { get; }
            private readonly Func<string, string> _executeFunc;

            public MockTool(string name, string description, Func<string, string> executeFunc)
            {
                Name = name;
                Description = description;
                _executeFunc = executeFunc;
            }

            public string Execute(string input) => _executeFunc(input);
        }

        /// <summary>
        /// Calculator tool for testing mathematical operations.
        /// </summary>
        private class CalculatorTool : ITool
        {
            public string Name => "Calculator";
            public string Description => "Performs mathematical calculations. Input should be a valid expression.";

            public string Execute(string input)
            {
                try
                {
                    // Simple calculator - handles basic operations
                    if (input.Contains("+"))
                    {
                        var parts = input.Split('+');
                        var sum = parts.Select(p => double.Parse(p.Trim())).Sum();
                        return sum.ToString();
                    }
                    if (input.Contains("*"))
                    {
                        var parts = input.Split('*');
                        var product = parts.Select(p => double.Parse(p.Trim())).Aggregate(1.0, (a, b) => a * b);
                        return product.ToString();
                    }
                    if (input.Contains("sqrt"))
                    {
                        var num = double.Parse(input.Replace("sqrt", "").Replace("(", "").Replace(")", "").Trim());
                        return Math.Sqrt(num).ToString();
                    }
                    return input;
                }
                catch (Exception ex)
                {
                    return $"Error: {ex.Message}";
                }
            }
        }

        /// <summary>
        /// Mock retriever for RAG agent testing.
        /// </summary>
        private class MockRetriever<T> : IRetriever<T>
        {
            private readonly List<Document<T>> _documents;

            public int DefaultTopK { get; } = 5;

            public MockRetriever(List<Document<T>> documents)
            {
                _documents = documents;
            }

            public IEnumerable<Document<T>> Retrieve(string query)
            {
                return Retrieve(query, DefaultTopK);
            }

            public IEnumerable<Document<T>> Retrieve(string query, int topK)
            {
                // Simple keyword matching for testing
                return _documents
                    .Where(d => d.Content.Contains(query, StringComparison.OrdinalIgnoreCase) ||
                                query.Split(' ').Any(word => d.Content.Contains(word, StringComparison.OrdinalIgnoreCase)))
                    .Take(topK);
            }

            public IEnumerable<Document<T>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
            {
                return Retrieve(query, topK);
            }
        }

        /// <summary>
        /// Mock reranker for RAG agent testing.
        /// </summary>
        private class MockReranker<T> : IReranker<T>
        {
            public bool ModifiesScores { get; } = true;

            public IEnumerable<Document<T>> Rerank(string query, IEnumerable<Document<T>> documents)
            {
                // Just reverse the order for testing
                return documents.Reverse();
            }

            public IEnumerable<Document<T>> Rerank(string query, IEnumerable<Document<T>> documents, int topK)
            {
                return Rerank(query, documents).Take(topK);
            }
        }

        /// <summary>
        /// Mock generator for RAG agent testing.
        /// </summary>
        private class MockGenerator<T> : IGenerator<T>
        {
            public int MaxContextTokens { get; } = 4096;
            public int MaxGenerationTokens { get; } = 1024;

            public string Generate(string prompt)
            {
                return "Generated response for: " + prompt.Substring(0, Math.Min(50, prompt.Length));
            }

            public GroundedAnswer<T> GenerateGrounded(string query, IEnumerable<Document<T>> context)
            {
                var docs = context.ToList();
                var answer = $"Based on {docs.Count} documents: " + string.Join(", ", docs.Select(d => d.Content.Substring(0, Math.Min(50, d.Content.Length))));
                var citations = docs.Select(d => $"Document {d.Id}").ToList();

                return new GroundedAnswer<T>
                {
                    Answer = answer,
                    Citations = citations,
                    ConfidenceScore = 0.85
                };
            }
        }

        #endregion

        #region Agent (ReAct) Tests

        [Fact]
        public async Task Agent_Initialization_Success()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();
            var tools = new List<ITool> { new CalculatorTool() };

            // Act
            var agent = new Agent<double>(chatModel, tools);

            // Assert
            Assert.NotNull(agent);
            Assert.Same(chatModel, agent.ChatModel);
            Assert.Single(agent.Tools);
            Assert.Equal("Calculator", agent.Tools[0].Name);
        }

        [Fact]
        public async Task Agent_Initialization_WithoutTools_Success()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();

            // Act
            var agent = new Agent<double>(chatModel);

            // Assert
            Assert.NotNull(agent);
            Assert.Empty(agent.Tools);
        }

        [Fact]
        public async Task Agent_SimpleQueryWithFinalAnswer_ReturnsAnswer()
        {
            // Arrange
            var response = @"{
                ""thought"": ""The user is asking for a simple greeting."",
                ""action"": """",
                ""action_input"": """",
                ""final_answer"": ""Hello! How can I help you today?""
            }";
            var chatModel = new MockChatModel<double>(response);
            var agent = new Agent<double>(chatModel);

            // Act
            var result = await agent.RunAsync("Say hello");

            // Assert
            Assert.Contains("Hello", result);
            Assert.Contains("Iteration 1", agent.Scratchpad);
        }

        [Fact]
        public async Task Agent_QueryWithToolExecution_ExecutesTool()
        {
            // Arrange
            var response1 = @"{
                ""thought"": ""I need to calculate 5 + 3."",
                ""action"": ""Calculator"",
                ""action_input"": ""5 + 3"",
                ""final_answer"": """"
            }";
            var response2 = @"{
                ""thought"": ""I have the result."",
                ""action"": """",
                ""action_input"": """",
                ""final_answer"": ""The answer is 8""
            }";
            var chatModel = new MockChatModel<double>(response1, response2);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new Agent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("What is 5 + 3?");

            // Assert
            Assert.Contains("8", result);
            Assert.Contains("Calculator", agent.Scratchpad);
            Assert.Contains("Observation: 8", agent.Scratchpad);
        }

        [Fact]
        public async Task Agent_MultiStepCalculation_ExecutesMultipleTools()
        {
            // Arrange
            var response1 = @"{
                ""thought"": ""First calculate sqrt(16)."",
                ""action"": ""Calculator"",
                ""action_input"": ""sqrt(16)"",
                ""final_answer"": """"
            }";
            var response2 = @"{
                ""thought"": ""Now add 4 to the result."",
                ""action"": ""Calculator"",
                ""action_input"": ""4 + 6"",
                ""final_answer"": """"
            }";
            var response3 = @"{
                ""thought"": ""I have the final answer."",
                ""action"": """",
                ""action_input"": """",
                ""final_answer"": ""The answer is 10""
            }";
            var chatModel = new MockChatModel<double>(response1, response2, response3);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new Agent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("What is sqrt(16) + 6?");

            // Assert
            Assert.Contains("10", result);
            Assert.Contains("Iteration 1", agent.Scratchpad);
            Assert.Contains("Iteration 2", agent.Scratchpad);
        }

        [Fact]
        public async Task Agent_ToolNotFound_ReturnsErrorObservation()
        {
            // Arrange
            var response = @"{
                ""thought"": ""I'll try to use a search tool."",
                ""action"": ""Search"",
                ""action_input"": ""test query"",
                ""final_answer"": """"
            }";
            var chatModel = new MockChatModel<double>(response, response);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new Agent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("Search for something");

            // Assert
            Assert.Contains("Tool 'Search' not found", agent.Scratchpad);
        }

        [Fact]
        public async Task Agent_MaxIterationsReached_ReturnsPartialResult()
        {
            // Arrange
            var response = @"{
                ""thought"": ""I'm still thinking."",
                ""action"": ""Calculator"",
                ""action_input"": ""1 + 1"",
                ""final_answer"": """"
            }";
            var chatModel = new MockChatModel<double>(response, response, response, response, response);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new Agent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("Complex query", maxIterations: 3);

            // Assert
            Assert.Contains("maximum number of iterations", result);
            Assert.Contains("Iteration 3", agent.Scratchpad);
        }

        [Fact]
        public async Task Agent_NullOrWhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();
            var agent = new Agent<double>(chatModel);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync(""));
            await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync("   "));
        }

        [Fact]
        public async Task Agent_InvalidMaxIterations_ThrowsArgumentException()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();
            var agent = new Agent<double>(chatModel);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync("test", maxIterations: 0));
            await Assert.ThrowsAsync<ArgumentException>(() => agent.RunAsync("test", maxIterations: -1));
        }

        [Fact]
        public async Task Agent_HttpRequestException_ReturnsErrorMessage()
        {
            // Arrange
            var chatModel = new MockChatModel<double>(new System.Net.Http.HttpRequestException("Connection failed"));
            var agent = new Agent<double>(chatModel);

            // Act
            var result = await agent.RunAsync("Test query");

            // Assert
            Assert.Contains("error while thinking", result);
            Assert.Contains("Connection failed", result);
        }

        [Fact]
        public async Task Agent_FallbackRegexParsing_WorksWithoutJSON()
        {
            // Arrange - Response without JSON format
            var response = @"Thought: I need to calculate this
Action: Calculator
Action Input: 2 + 2
Final Answer: The result is 4";
            var chatModel = new MockChatModel<double>(response);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new Agent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("What is 2 + 2?");

            // Assert
            Assert.Contains("4", result);
        }

        [Fact]
        public async Task Agent_ScratchpadTracking_RecordsAllSteps()
        {
            // Arrange
            var response = @"{
                ""thought"": ""Calculate the sum"",
                ""action"": ""Calculator"",
                ""action_input"": ""10 + 20"",
                ""final_answer"": """"
            }";
            var response2 = @"{
                ""thought"": ""Done"",
                ""final_answer"": ""30""
            }";
            var chatModel = new MockChatModel<double>(response, response2);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new Agent<double>(chatModel, tools);

            // Act
            await agent.RunAsync("Calculate 10 + 20");

            // Assert
            var scratchpad = agent.Scratchpad;
            Assert.Contains("Query:", scratchpad);
            Assert.Contains("Thought: Calculate the sum", scratchpad);
            Assert.Contains("Action: Calculator", scratchpad);
            Assert.Contains("Action Input: 10 + 20", scratchpad);
            Assert.Contains("Observation: 30", scratchpad);
            Assert.Contains("Final Answer", scratchpad);
        }

        #endregion

        #region ChainOfThoughtAgent Tests

        [Fact]
        public async Task ChainOfThoughtAgent_Initialization_Success()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();

            // Act
            var agent = new ChainOfThoughtAgent<double>(chatModel);

            // Assert
            Assert.NotNull(agent);
            Assert.Same(chatModel, agent.ChatModel);
        }

        [Fact]
        public async Task ChainOfThoughtAgent_PureReasoning_NoTools()
        {
            // Arrange
            var response = @"{
                ""reasoning_steps"": [
                    ""Step 1: Identify that this is a logical deduction problem"",
                    ""Step 2: Apply the rule that all humans are mortal"",
                    ""Step 3: Socrates is a human, therefore Socrates is mortal""
                ],
                ""tool_calls"": [],
                ""final_answer"": ""Socrates is mortal""
            }";
            var chatModel = new MockChatModel<double>(response);
            var agent = new ChainOfThoughtAgent<double>(chatModel, allowTools: false);

            // Act
            var result = await agent.RunAsync("If all humans are mortal and Socrates is human, what can we conclude?");

            // Assert
            Assert.Contains("Socrates is mortal", result);
            Assert.Contains("Step 1:", agent.Scratchpad);
            Assert.Contains("Step 2:", agent.Scratchpad);
            Assert.Contains("Step 3:", agent.Scratchpad);
        }

        [Fact]
        public async Task ChainOfThoughtAgent_WithTools_ExecutesToolsAndRefines()
        {
            // Arrange
            var response1 = @"{
                ""reasoning_steps"": [
                    ""Step 1: Break down the calculation"",
                    ""Step 2: Calculate 5 * 5 first""
                ],
                ""tool_calls"": [
                    {
                        ""tool_name"": ""Calculator"",
                        ""tool_input"": ""5 * 5""
                    }
                ],
                ""final_answer"": """"
            }";
            var response2 = @"{
                ""final_answer"": ""The result is 25""
            }";
            var chatModel = new MockChatModel<double>(response1, response2);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new ChainOfThoughtAgent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("What is 5 * 5?");

            // Assert
            Assert.Contains("25", result);
            Assert.Contains("Step 1:", agent.Scratchpad);
            Assert.Contains("Tool: Calculator", agent.Scratchpad);
        }

        [Fact]
        public async Task ChainOfThoughtAgent_MaxSteps_TruncatesExcessiveSteps()
        {
            // Arrange
            var response = @"{
                ""reasoning_steps"": [
                    ""Step 1"", ""Step 2"", ""Step 3"", ""Step 4"", ""Step 5"",
                    ""Step 6"", ""Step 7"", ""Step 8"", ""Step 9"", ""Step 10""
                ],
                ""tool_calls"": [],
                ""final_answer"": ""Done""
            }";
            var chatModel = new MockChatModel<double>(response);
            var agent = new ChainOfThoughtAgent<double>(chatModel);

            // Act
            await agent.RunAsync("Test", maxIterations: 5);

            // Assert
            Assert.Contains("truncating to 5", agent.Scratchpad);
        }

        [Fact]
        public async Task ChainOfThoughtAgent_FallbackRegexParsing_WorksWithoutJSON()
        {
            // Arrange
            var response = @"Step 1: First understand the problem
Step 2: Apply the formula
Step 3: Calculate the result
Final Answer: 42";
            var chatModel = new MockChatModel<double>(response);
            var agent = new ChainOfThoughtAgent<double>(chatModel);

            // Act
            var result = await agent.RunAsync("Test query");

            // Assert
            Assert.Contains("42", result);
        }

        [Fact]
        public async Task ChainOfThoughtAgent_NoFinalAnswer_ReturnsDefault()
        {
            // Arrange
            var response = @"{
                ""reasoning_steps"": [""Step 1: Thinking""],
                ""tool_calls"": [],
                ""final_answer"": """"
            }";
            var chatModel = new MockChatModel<double>(response);
            var agent = new ChainOfThoughtAgent<double>(chatModel);

            // Act
            var result = await agent.RunAsync("Test");

            // Assert
            Assert.Contains("unable to determine a final answer", result);
        }

        #endregion

        #region PlanAndExecuteAgent Tests

        [Fact]
        public async Task PlanAndExecuteAgent_Initialization_Success()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();

            // Act
            var agent = new PlanAndExecuteAgent<double>(chatModel);

            // Assert
            Assert.NotNull(agent);
            Assert.Same(chatModel, agent.ChatModel);
        }

        [Fact]
        public async Task PlanAndExecuteAgent_SimplePlan_ExecutesAllSteps()
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
                        ""description"": ""Provide the final answer"",
                        ""tool"": """",
                        ""input"": """",
                        ""is_final_step"": true
                    }
                ]
            }";
            var finalResponse = "The sum is 15";
            var chatModel = new MockChatModel<double>(planResponse, finalResponse);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new PlanAndExecuteAgent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("What is 10 + 5?");

            // Assert
            Assert.Contains("15", result);
            Assert.Contains("PLANNING PHASE", agent.Scratchpad);
            Assert.Contains("EXECUTION PHASE", agent.Scratchpad);
            Assert.Contains("Step 1/2", agent.Scratchpad);
        }

        [Fact]
        public async Task PlanAndExecuteAgent_MultiStepPlan_ExecutesInOrder()
        {
            // Arrange
            var planResponse = @"{
                ""steps"": [
                    {
                        ""description"": ""Calculate sqrt(16)"",
                        ""tool"": ""Calculator"",
                        ""input"": ""sqrt(16)"",
                        ""is_final_step"": false
                    },
                    {
                        ""description"": ""Add 10 to result"",
                        ""tool"": ""Calculator"",
                        ""input"": ""4 + 10"",
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
            var finalResponse = "The result is 14";
            var chatModel = new MockChatModel<double>(planResponse, finalResponse);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new PlanAndExecuteAgent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("Calculate sqrt(16) + 10");

            // Assert
            Assert.Contains("14", result);
            Assert.Contains("Step 1/3", agent.Scratchpad);
            Assert.Contains("Step 2/3", agent.Scratchpad);
        }

        [Fact]
        public async Task PlanAndExecuteAgent_PlanRevision_RevisesOnError()
        {
            // Arrange
            var initialPlan = @"{
                ""steps"": [
                    {
                        ""description"": ""Use nonexistent tool"",
                        ""tool"": ""NonexistentTool"",
                        ""input"": ""test"",
                        ""is_final_step"": true
                    }
                ]
            }";
            var revisedPlan = @"{
                ""steps"": [
                    {
                        ""description"": ""Use calculator instead"",
                        ""tool"": ""Calculator"",
                        ""input"": ""5 + 5"",
                        ""is_final_step"": true
                    }
                ]
            }";
            var finalResponse = "10";
            var chatModel = new MockChatModel<double>(initialPlan, revisedPlan, finalResponse);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new PlanAndExecuteAgent<double>(chatModel, tools, allowPlanRevision: true);

            // Act
            var result = await agent.RunAsync("Calculate something");

            // Assert
            Assert.Contains("10", result);
        }

        [Fact]
        public async Task PlanAndExecuteAgent_NoPlanRevision_FailsOnError()
        {
            // Arrange
            var plan = @"{
                ""steps"": [
                    {
                        ""description"": ""Use nonexistent tool"",
                        ""tool"": ""NonexistentTool"",
                        ""input"": ""test"",
                        ""is_final_step"": true
                    }
                ]
            }";
            var chatModel = new MockChatModel<double>(plan);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new PlanAndExecuteAgent<double>(chatModel, tools, allowPlanRevision: false);

            // Act
            var result = await agent.RunAsync("Test");

            // Assert
            Assert.Contains("Error", result);
        }

        [Fact]
        public async Task PlanAndExecuteAgent_EmptyPlan_ReturnsError()
        {
            // Arrange
            var planResponse = @"{""steps"": []}";
            var chatModel = new MockChatModel<double>(planResponse);
            var agent = new PlanAndExecuteAgent<double>(chatModel);

            // Act
            var result = await agent.RunAsync("Test query");

            // Assert
            Assert.Contains("unable to create a plan", result);
        }

        [Fact]
        public async Task PlanAndExecuteAgent_MaxRevisions_StopsAfterLimit()
        {
            // Arrange - Create responses that will always fail
            var failingPlan = @"{
                ""steps"": [
                    {
                        ""description"": ""Failing step"",
                        ""tool"": ""NonexistentTool"",
                        ""input"": ""test"",
                        ""is_final_step"": true
                    }
                ]
            }";
            var chatModel = new MockChatModel<double>(
                failingPlan, failingPlan, failingPlan, failingPlan, failingPlan,
                failingPlan, failingPlan, failingPlan);
            var agent = new PlanAndExecuteAgent<double>(chatModel, null, allowPlanRevision: true);

            // Act
            var result = await agent.RunAsync("Test", maxIterations: 3);

            // Assert
            Assert.Contains("maximum number of plan revisions", result);
        }

        #endregion

        #region RAGAgent Tests

        [Fact]
        public void RAGAgent_Initialization_Success()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();
            var documents = new List<Document<double>>
            {
                new Document<double>("1", "Python is a programming language"),
                new Document<double>("2", "Python is used for data science")
            };
            var retriever = new MockRetriever<double>(documents);
            var generator = new MockGenerator<double>();

            // Act
            var agent = new RAGAgent<double>(chatModel, retriever, generator);

            // Assert
            Assert.NotNull(agent);
            Assert.Same(chatModel, agent.ChatModel);
        }

        [Fact]
        public async Task RAGAgent_SimpleQuery_RetrievesAndGenerates()
        {
            // Arrange
            var chatModel = new MockChatModel<double>("What is Python programming language?");
            var documents = new List<Document<double>>
            {
                new Document<double>("1", "Python is a high-level programming language created by Guido van Rossum"),
                new Document<double>("2", "Python is widely used for web development, data analysis, and machine learning"),
                new Document<double>("3", "Java is an object-oriented programming language")
            };
            var retriever = new MockRetriever<double>(documents);
            var generator = new MockGenerator<double>();
            var agent = new RAGAgent<double>(chatModel, retriever, generator, retrievalTopK: 2);

            // Act
            var result = await agent.RunAsync("What is Python?");

            // Assert
            Assert.Contains("Based on", result);
            Assert.Contains("RETRIEVAL PHASE", agent.Scratchpad);
            Assert.Contains("GENERATION PHASE", agent.Scratchpad);
        }

        [Fact]
        public async Task RAGAgent_WithReranker_UsesReranking()
        {
            // Arrange
            var chatModel = new MockChatModel<double>("refined query");
            var documents = new List<Document<double>>
            {
                new Document<double>("1", "Document about Python"),
                new Document<double>("2", "Document about Java"),
                new Document<double>("3", "Document about C++")
            };
            var retriever = new MockRetriever<double>(documents);
            var reranker = new MockReranker<double>();
            var generator = new MockGenerator<double>();
            var agent = new RAGAgent<double>(chatModel, retriever, generator, reranker, retrievalTopK: 3, rerankTopK: 2);

            // Act
            var result = await agent.RunAsync("Tell me about Python");

            // Assert
            Assert.Contains("RERANKING PHASE", agent.Scratchpad);
            Assert.Contains("Kept top", agent.Scratchpad);
        }

        [Fact]
        public async Task RAGAgent_WithCitations_IncludesCitations()
        {
            // Arrange
            var chatModel = new MockChatModel<double>("query");
            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Information about topic")
            };
            var retriever = new MockRetriever<double>(documents);
            var generator = new MockGenerator<double>();
            var agent = new RAGAgent<double>(chatModel, retriever, generator, includeCitations: true);

            // Act
            var result = await agent.RunAsync("Query");

            // Assert
            Assert.Contains("Sources:", result);
            Assert.Contains("[1]", result);
        }

        [Fact]
        public async Task RAGAgent_NoCitations_ExcludesCitations()
        {
            // Arrange
            var chatModel = new MockChatModel<double>("query");
            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Information")
            };
            var retriever = new MockRetriever<double>(documents);
            var generator = new MockGenerator<double>();
            var agent = new RAGAgent<double>(chatModel, retriever, generator, includeCitations: false);

            // Act
            var result = await agent.RunAsync("Query");

            // Assert
            Assert.DoesNotContain("Sources:", result);
        }

        [Fact]
        public async Task RAGAgent_NoDocumentsFound_ReturnsNotFoundMessage()
        {
            // Arrange
            var chatModel = new MockChatModel<double>("query");
            var documents = new List<Document<double>>();
            var retriever = new MockRetriever<double>(documents);
            var generator = new MockGenerator<double>();
            var agent = new RAGAgent<double>(chatModel, retriever, generator);

            // Act
            var result = await agent.RunAsync("Nonexistent topic");

            // Assert
            Assert.Contains("couldn't find any relevant information", result);
        }

        [Fact]
        public async Task RAGAgent_QueryRefinement_RefinesQuery()
        {
            // Arrange
            var chatModel = new MockChatModel<double>("How do I install Python programming language?");
            var documents = new List<Document<double>>
            {
                new Document<double>("1", "Python installation guide for Windows")
            };
            var retriever = new MockRetriever<double>(documents);
            var generator = new MockGenerator<double>();
            var agent = new RAGAgent<double>(chatModel, retriever, generator, allowQueryRefinement: true);

            // Act
            var result = await agent.RunAsync("How do I install it?");

            // Assert
            Assert.Contains("QUERY ANALYSIS", agent.Scratchpad);
        }

        [Fact]
        public async Task RAGAgent_NoQueryRefinement_SkipsRefinement()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();
            var documents = new List<Document<double>>
            {
                new Document<double>("1", "Content")
            };
            var retriever = new MockRetriever<double>(documents);
            var generator = new MockGenerator<double>();
            var agent = new RAGAgent<double>(chatModel, retriever, generator, allowQueryRefinement: false);

            // Act
            await agent.RunAsync("Query");

            // Assert
            Assert.DoesNotContain("QUERY ANALYSIS", agent.Scratchpad);
        }

        [Fact]
        public void RAGAgent_GetPipelineInfo_ReturnsConfiguration()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();
            var documents = new List<Document<double>>();
            var retriever = new MockRetriever<double>(documents);
            var reranker = new MockReranker<double>();
            var generator = new MockGenerator<double>();
            var agent = new RAGAgent<double>(chatModel, retriever, generator, reranker,
                retrievalTopK: 15, rerankTopK: 7, includeCitations: true);

            // Act
            var info = agent.GetPipelineInfo();

            // Assert
            Assert.Contains("RAG Pipeline Configuration", info);
            Assert.Contains("Retriever: MockRetriever", info);
            Assert.Contains("Generator: MockGenerator", info);
            Assert.Contains("Reranker: MockReranker", info);
            Assert.Contains("Retrieval TopK: 15", info);
            Assert.Contains("Rerank TopK: 7", info);
            Assert.Contains("Include Citations: True", info);
        }

        [Fact]
        public void RAGAgent_InvalidTopK_ThrowsArgumentException()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();
            var documents = new List<Document<double>>();
            var retriever = new MockRetriever<double>(documents);
            var generator = new MockGenerator<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new RAGAgent<double>(chatModel, retriever, generator, retrievalTopK: 0));
            Assert.Throws<ArgumentException>(() =>
                new RAGAgent<double>(chatModel, retriever, generator, rerankTopK: 0));
        }

        #endregion

        #region AgentKeyResolver Tests

        [Fact]
        public void AgentKeyResolver_ExplicitKey_UsesExplicitKey()
        {
            // Arrange
            var explicitKey = "explicit-key";
            var storedConfig = new AgentConfiguration<double> { ApiKey = "stored-key" };

            // Act
            var result = AgentKeyResolver.ResolveApiKey(explicitKey, storedConfig, LLMProvider.OpenAI);

            // Assert
            Assert.Equal("explicit-key", result);
        }

        [Fact]
        public void AgentKeyResolver_StoredConfig_UsesStoredKey()
        {
            // Arrange
            var storedConfig = new AgentConfiguration<double> { ApiKey = "stored-key" };

            // Act
            var result = AgentKeyResolver.ResolveApiKey(storedConfig: storedConfig, provider: LLMProvider.OpenAI);

            // Assert
            Assert.Equal("stored-key", result);
        }

        [Fact]
        public void AgentKeyResolver_GlobalConfig_UsesGlobalKey()
        {
            // Arrange
            AgentGlobalConfiguration.Configure(config => config.ConfigureOpenAI("global-key"));

            try
            {
                // Act
                var result = AgentKeyResolver.ResolveApiKey<double>(provider: LLMProvider.OpenAI);

                // Assert
                Assert.Equal("global-key", result);
            }
            finally
            {
                // Cleanup - reset global configuration
                AgentGlobalConfiguration.Configure(config => { });
            }
        }

        [Fact]
        public void AgentKeyResolver_EnvironmentVariable_UsesEnvKey()
        {
            // Arrange
            var originalKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            Environment.SetEnvironmentVariable("OPENAI_API_KEY", "env-key");

            try
            {
                // Act
                var result = AgentKeyResolver.ResolveApiKey<double>(provider: LLMProvider.OpenAI);

                // Assert
                Assert.Equal("env-key", result);
            }
            finally
            {
                // Cleanup
                Environment.SetEnvironmentVariable("OPENAI_API_KEY", originalKey);
            }
        }

        [Fact]
        public void AgentKeyResolver_NoKeyFound_ThrowsInvalidOperationException()
        {
            // Arrange - ensure no keys are set
            var originalKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            Environment.SetEnvironmentVariable("OPENAI_API_KEY", null);

            try
            {
                // Act & Assert
                var exception = Assert.Throws<InvalidOperationException>(() =>
                    AgentKeyResolver.ResolveApiKey<double>(provider: LLMProvider.OpenAI));

                Assert.Contains("No API key found for OpenAI", exception.Message);
                Assert.Contains("Explicit parameter", exception.Message);
                Assert.Contains("Global config", exception.Message);
                Assert.Contains("Environment variable", exception.Message);
            }
            finally
            {
                Environment.SetEnvironmentVariable("OPENAI_API_KEY", originalKey);
            }
        }

        [Fact]
        public void AgentKeyResolver_PriorityOrder_ExplicitOverridesAll()
        {
            // Arrange
            var originalKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            Environment.SetEnvironmentVariable("OPENAI_API_KEY", "env-key");
            AgentGlobalConfiguration.Configure(config => config.ConfigureOpenAI("global-key"));
            var storedConfig = new AgentConfiguration<double> { ApiKey = "stored-key" };

            try
            {
                // Act
                var result = AgentKeyResolver.ResolveApiKey("explicit-key", storedConfig, LLMProvider.OpenAI);

                // Assert - explicit key wins
                Assert.Equal("explicit-key", result);
            }
            finally
            {
                // Cleanup
                Environment.SetEnvironmentVariable("OPENAI_API_KEY", originalKey);
                AgentGlobalConfiguration.Configure(config => { });
            }
        }

        [Fact]
        public void AgentKeyResolver_AnthropicProvider_UsesCorrectEnvVar()
        {
            // Arrange
            var originalKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY");
            Environment.SetEnvironmentVariable("ANTHROPIC_API_KEY", "anthropic-key");

            try
            {
                // Act
                var result = AgentKeyResolver.ResolveApiKey<double>(provider: LLMProvider.Anthropic);

                // Assert
                Assert.Equal("anthropic-key", result);
            }
            finally
            {
                Environment.SetEnvironmentVariable("ANTHROPIC_API_KEY", originalKey);
            }
        }

        [Fact]
        public void AgentKeyResolver_AzureProvider_UsesCorrectEnvVar()
        {
            // Arrange
            var originalKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY");
            Environment.SetEnvironmentVariable("AZURE_OPENAI_KEY", "azure-key");

            try
            {
                // Act
                var result = AgentKeyResolver.ResolveApiKey<double>(provider: LLMProvider.AzureOpenAI);

                // Assert
                Assert.Equal("azure-key", result);
            }
            finally
            {
                Environment.SetEnvironmentVariable("AZURE_OPENAI_KEY", originalKey);
            }
        }

        #endregion

        #region AgentGlobalConfiguration Tests

        [Fact]
        public void AgentGlobalConfiguration_ConfigureOpenAI_SetsApiKey()
        {
            // Arrange & Act
            AgentGlobalConfiguration.Configure(config => config.ConfigureOpenAI("test-key"));

            try
            {
                // Assert
                Assert.True(AgentGlobalConfiguration.ApiKeys.ContainsKey(LLMProvider.OpenAI));
                Assert.Equal("test-key", AgentGlobalConfiguration.ApiKeys[LLMProvider.OpenAI]);
            }
            finally
            {
                // Cleanup
                AgentGlobalConfiguration.Configure(config => { });
            }
        }

        [Fact]
        public void AgentGlobalConfiguration_ConfigureAnthropic_SetsApiKey()
        {
            // Arrange & Act
            AgentGlobalConfiguration.Configure(config => config.ConfigureAnthropic("anthropic-key"));

            try
            {
                // Assert
                Assert.True(AgentGlobalConfiguration.ApiKeys.ContainsKey(LLMProvider.Anthropic));
                Assert.Equal("anthropic-key", AgentGlobalConfiguration.ApiKeys[LLMProvider.Anthropic]);
            }
            finally
            {
                AgentGlobalConfiguration.Configure(config => { });
            }
        }

        [Fact]
        public void AgentGlobalConfiguration_ConfigureMultipleProviders_SetsAllKeys()
        {
            // Arrange & Act
            AgentGlobalConfiguration.Configure(config => config
                .ConfigureOpenAI("openai-key")
                .ConfigureAnthropic("anthropic-key")
                .ConfigureAzureOpenAI("azure-key"));

            try
            {
                // Assert
                Assert.Equal(3, AgentGlobalConfiguration.ApiKeys.Count);
                Assert.Equal("openai-key", AgentGlobalConfiguration.ApiKeys[LLMProvider.OpenAI]);
                Assert.Equal("anthropic-key", AgentGlobalConfiguration.ApiKeys[LLMProvider.Anthropic]);
                Assert.Equal("azure-key", AgentGlobalConfiguration.ApiKeys[LLMProvider.AzureOpenAI]);
            }
            finally
            {
                AgentGlobalConfiguration.Configure(config => { });
            }
        }

        [Fact]
        public void AgentGlobalConfiguration_DefaultProvider_SetsCorrectly()
        {
            // Arrange
            var originalProvider = AgentGlobalConfiguration.DefaultProvider;

            try
            {
                // Act
                AgentGlobalConfiguration.Configure(config => config.UseDefaultProvider(LLMProvider.Anthropic));

                // Assert
                Assert.Equal(LLMProvider.Anthropic, AgentGlobalConfiguration.DefaultProvider);
            }
            finally
            {
                AgentGlobalConfiguration.DefaultProvider = originalProvider;
            }
        }

        [Fact]
        public void AgentGlobalConfiguration_ApiKeys_ReturnsReadOnlyCopy()
        {
            // Arrange
            AgentGlobalConfiguration.Configure(config => config.ConfigureOpenAI("test-key"));

            try
            {
                // Act
                var keys = AgentGlobalConfiguration.ApiKeys;

                // Assert - should not be able to modify the returned dictionary
                Assert.IsAssignableFrom<IReadOnlyDictionary<LLMProvider, string>>(keys);
            }
            finally
            {
                AgentGlobalConfiguration.Configure(config => { });
            }
        }

        [Fact]
        public void AgentGlobalConfiguration_ThreadSafety_MultipleConfigureCalls()
        {
            // Arrange
            var tasks = new List<Task>();

            // Act - configure from multiple threads
            for (int i = 0; i < 10; i++)
            {
                int index = i;
                tasks.Add(Task.Run(() =>
                {
                    AgentGlobalConfiguration.Configure(config =>
                        config.ConfigureOpenAI($"key-{index}"));
                }));
            }

            Task.WaitAll(tasks.ToArray());

            try
            {
                // Assert - should have a key set (exact value doesn't matter due to race)
                Assert.True(AgentGlobalConfiguration.ApiKeys.ContainsKey(LLMProvider.OpenAI));
            }
            finally
            {
                AgentGlobalConfiguration.Configure(config => { });
            }
        }

        #endregion

        #region AgentBase Tests

        [Fact]
        public void AgentBase_ScratchpadTracking_StartsEmpty()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();
            var agent = new Agent<double>(chatModel);

            // Assert
            Assert.Equal("", agent.Scratchpad);
        }

        [Fact]
        public async Task AgentBase_ToolsCollection_ReturnsReadOnlyList()
        {
            // Arrange
            var chatModel = new MockChatModel<double>();
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new Agent<double>(chatModel, tools);

            // Assert
            Assert.IsAssignableFrom<IReadOnlyList<ITool>>(agent.Tools);
            Assert.Single(agent.Tools);
        }

        [Fact]
        public void AgentBase_NullChatModel_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new Agent<double>(null!));
        }

        [Fact]
        public async Task AgentBase_MultipleRuns_ClearsScratchpadBetweenRuns()
        {
            // Arrange
            var response = @"{""final_answer"": ""Answer""}";
            var chatModel = new MockChatModel<double>(response, response);
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

        #region Error Handling and Edge Cases

        [Fact]
        public async Task Agent_IOExceptionDuringGeneration_ReturnsErrorMessage()
        {
            // Arrange
            var chatModel = new MockChatModel<double>(new System.IO.IOException("IO error"));
            var agent = new Agent<double>(chatModel);

            // Act
            var result = await agent.RunAsync("Test");

            // Assert
            Assert.Contains("IO error", result);
        }

        [Fact]
        public async Task Agent_TaskCanceledExceptionDuringGeneration_ReturnsErrorMessage()
        {
            // Arrange
            var chatModel = new MockChatModel<double>(new TaskCanceledException("Timeout"));
            var agent = new Agent<double>(chatModel);

            // Act
            var result = await agent.RunAsync("Test");

            // Assert
            Assert.Contains("timeout", result);
        }

        [Fact]
        public async Task ChainOfThoughtAgent_ErrorDuringGeneration_ReturnsErrorMessage()
        {
            // Arrange
            var chatModel = new MockChatModel<double>(new System.Net.Http.HttpRequestException("Network error"));
            var agent = new ChainOfThoughtAgent<double>(chatModel);

            // Act
            var result = await agent.RunAsync("Test");

            // Assert
            Assert.Contains("error while reasoning", result);
        }

        [Fact]
        public async Task Agent_ToolExecutionThrowsException_ReturnsErrorObservation()
        {
            // Arrange
            var response = @"{
                ""action"": ""FailingTool"",
                ""action_input"": ""test""
            }";
            var chatModel = new MockChatModel<double>(response, response);
            var failingTool = new MockTool("FailingTool", "A tool that fails",
                input => throw new InvalidOperationException("Tool failed"));
            var tools = new List<ITool> { failingTool };
            var agent = new Agent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("Test");

            // Assert
            Assert.Contains("Error executing tool", agent.Scratchpad);
        }

        [Fact]
        public async Task Agent_EmptyResponse_HandlesGracefully()
        {
            // Arrange
            var chatModel = new MockChatModel<double>("", "");
            var agent = new Agent<double>(chatModel);

            // Act
            var result = await agent.RunAsync("Test", maxIterations: 2);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public async Task PlanAndExecuteAgent_PlanParsingFails_HandlesGracefully()
        {
            // Arrange
            var chatModel = new MockChatModel<double>("Invalid JSON {{{");
            var agent = new PlanAndExecuteAgent<double>(chatModel);

            // Act
            var result = await agent.RunAsync("Test");

            // Assert
            Assert.Contains("unable to create a plan", result);
        }

        #endregion

        #region Integration Scenarios

        [Fact]
        public async Task IntegrationScenario_ComplexMathProblem_AgentSolvesStepByStep()
        {
            // Arrange - Complex calculation: sqrt(144) + 10 * 2
            var response1 = @"{
                ""thought"": ""First calculate sqrt(144)"",
                ""action"": ""Calculator"",
                ""action_input"": ""sqrt(144)""
            }";
            var response2 = @"{
                ""thought"": ""Now calculate 10 * 2"",
                ""action"": ""Calculator"",
                ""action_input"": ""10 * 2""
            }";
            var response3 = @"{
                ""thought"": ""Add the results: 12 + 20"",
                ""action"": ""Calculator"",
                ""action_input"": ""12 + 20""
            }";
            var response4 = @"{
                ""thought"": ""I have the answer"",
                ""final_answer"": ""The result is 32""
            }";
            var chatModel = new MockChatModel<double>(response1, response2, response3, response4);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new Agent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("Calculate sqrt(144) + 10 * 2");

            // Assert
            Assert.Contains("32", result);
            Assert.Contains("Iteration 1", agent.Scratchpad);
            Assert.Contains("Iteration 2", agent.Scratchpad);
            Assert.Contains("Iteration 3", agent.Scratchpad);
        }

        [Fact]
        public async Task IntegrationScenario_RAGWithMultipleDocuments_FindsRelevantInfo()
        {
            // Arrange
            var chatModel = new MockChatModel<double>("Python programming language");
            var documents = new List<Document<double>>
            {
                new Document<double>("1", "Python was created by Guido van Rossum in 1991"),
                new Document<double>("2", "Python is known for its simple, readable syntax"),
                new Document<double>("3", "Python is used in web development, data science, and AI"),
                new Document<double>("4", "Java is a compiled programming language"),
                new Document<double>("5", "C++ is used for system programming")
            };
            var retriever = new MockRetriever<double>(documents);
            var generator = new MockGenerator<double>();
            var agent = new RAGAgent<double>(chatModel, retriever, generator,
                retrievalTopK: 5, includeCitations: true);

            // Act
            var result = await agent.RunAsync("Tell me about Python");

            // Assert
            Assert.Contains("Based on", result);
            Assert.Contains("Sources:", result);
            Assert.Contains("Python", agent.Scratchpad);
        }

        [Fact]
        public async Task IntegrationScenario_ChainOfThoughtWithTools_CombinesReasoningAndExecution()
        {
            // Arrange
            var response1 = @"{
                ""reasoning_steps"": [
                    ""Step 1: Identify we need to calculate the area of a square"",
                    ""Step 2: The formula for square area is side * side"",
                    ""Step 3: Calculate 5 * 5""
                ],
                ""tool_calls"": [{
                    ""tool_name"": ""Calculator"",
                    ""tool_input"": ""5 * 5""
                }],
                ""final_answer"": """"
            }";
            var response2 = @"{""final_answer"": ""The area is 25 square units""}";
            var chatModel = new MockChatModel<double>(response1, response2);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new ChainOfThoughtAgent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("What is the area of a square with side 5?");

            // Assert
            Assert.Contains("25", result);
            Assert.Contains("Step 1:", agent.Scratchpad);
            Assert.Contains("Step 2:", agent.Scratchpad);
            Assert.Contains("Step 3:", agent.Scratchpad);
            Assert.Contains("Tool: Calculator", agent.Scratchpad);
        }

        [Fact]
        public async Task IntegrationScenario_PlanAndExecuteMultiStep_CompletesFullWorkflow()
        {
            // Arrange
            var planResponse = @"{
                ""steps"": [
                    {
                        ""description"": ""Calculate the square root"",
                        ""tool"": ""Calculator"",
                        ""input"": ""sqrt(16)"",
                        ""is_final_step"": false
                    },
                    {
                        ""description"": ""Multiply by 2"",
                        ""tool"": ""Calculator"",
                        ""input"": ""4 * 2"",
                        ""is_final_step"": false
                    },
                    {
                        ""description"": ""Provide answer"",
                        ""tool"": """",
                        ""input"": """",
                        ""is_final_step"": true
                    }
                ]
            }";
            var finalResponse = "The answer is 8";
            var chatModel = new MockChatModel<double>(planResponse, finalResponse);
            var tools = new List<ITool> { new CalculatorTool() };
            var agent = new PlanAndExecuteAgent<double>(chatModel, tools);

            // Act
            var result = await agent.RunAsync("Calculate sqrt(16) * 2");

            // Assert
            Assert.Contains("8", result);
            Assert.Contains("PLANNING PHASE", agent.Scratchpad);
            Assert.Contains("Step 1/3", agent.Scratchpad);
            Assert.Contains("Step 2/3", agent.Scratchpad);
            Assert.Contains("PLAN COMPLETED", agent.Scratchpad);
        }

        #endregion
    }

    #region Helper Classes for Global Configuration

    /// <summary>
    /// Builder for AgentGlobalConfiguration (mock for testing).
    /// </summary>
    public class AgentGlobalConfigurationBuilder
    {
        private readonly Dictionary<LLMProvider, string> _keys = new();
        private LLMProvider? _defaultProvider;

        public AgentGlobalConfigurationBuilder ConfigureOpenAI(string apiKey)
        {
            _keys[LLMProvider.OpenAI] = apiKey;
            return this;
        }

        public AgentGlobalConfigurationBuilder ConfigureAnthropic(string apiKey)
        {
            _keys[LLMProvider.Anthropic] = apiKey;
            return this;
        }

        public AgentGlobalConfigurationBuilder ConfigureAzureOpenAI(string apiKey)
        {
            _keys[LLMProvider.AzureOpenAI] = apiKey;
            return this;
        }

        public AgentGlobalConfigurationBuilder UseDefaultProvider(LLMProvider provider)
        {
            _defaultProvider = provider;
            return this;
        }

        public void Apply()
        {
            foreach (var kvp in _keys)
            {
                AgentGlobalConfiguration.SetApiKey(kvp.Key, kvp.Value);
            }

            if (_defaultProvider.HasValue)
            {
                AgentGlobalConfiguration.DefaultProvider = _defaultProvider.Value;
            }
        }
    }

    #endregion
}
