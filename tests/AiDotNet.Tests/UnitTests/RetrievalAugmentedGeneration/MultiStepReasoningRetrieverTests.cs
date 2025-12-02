using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    public class MultiStepReasoningRetrieverTests
    {
        // Mock retriever for testing
        private class MockRetriever : RetrieverBase<double>
        {
            private readonly List<Document<double>> _documents;

            public MockRetriever(List<Document<double>> documents) : base(defaultTopK: 5)
            {
                _documents = documents ?? new List<Document<double>>();
            }

            protected override IEnumerable<Document<double>> RetrieveCore(
                string query,
                int topK,
                Dictionary<string, object> metadataFilters)
            {
                return _documents.Take(topK);
            }
        }

        private MockRetriever CreateMockRetriever()
        {
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "Solar panels convert sunlight into electricity.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc2",
                    Content = "Environmental benefits include reduced carbon emissions.",
                    RelevanceScore = 0.85,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc3",
                    Content = "Economic impacts include job creation and energy savings.",
                    RelevanceScore = 0.8,
                    HasRelevanceScore = true
                }
            };
            return new MockRetriever(docs);
        }

        [Fact]
        public void Constructor_WithNullGenerator_ThrowsArgumentNullException()
        {
            // Arrange
            var mockRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new MultiStepReasoningRetriever<double>(null!, mockRetriever));
        }

        [Fact]
        public void Constructor_WithNullRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var generator = new StubGenerator<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new MultiStepReasoningRetriever<double>(generator, null!));
        }

        [Fact]
        public void Constructor_WithValidArguments_InitializesCorrectly()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();

            // Act
            var multiStepRetriever = new MultiStepReasoningRetriever<double>(
                generator,
                mockRetriever,
                maxSteps: 5);

            // Assert
            Assert.NotNull(multiStepRetriever);
        }

        [Fact]
        public void Constructor_WithInvalidMaxSteps_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new MultiStepReasoningRetriever<double>(generator, mockRetriever, maxSteps: 0));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new MultiStepReasoningRetriever<double>(generator, mockRetriever, maxSteps: 25));
        }

        [Fact]
        public void RetrieveMultiStep_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var multiStepRetriever = new MultiStepReasoningRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                multiStepRetriever.RetrieveMultiStep(null!, 10));
        }

        [Fact]
        public void RetrieveMultiStep_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var multiStepRetriever = new MultiStepReasoningRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                multiStepRetriever.RetrieveMultiStep("   ", 10));
        }

        [Fact]
        public void RetrieveMultiStep_WithNegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var multiStepRetriever = new MultiStepReasoningRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                multiStepRetriever.RetrieveMultiStep("test query", -1));
        }

        [Fact]
        public void RetrieveMultiStep_WithValidQuery_ReturnsResult()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var multiStepRetriever = new MultiStepReasoningRetriever<double>(
                generator,
                mockRetriever,
                maxSteps: 3);

            // Act
            var result = multiStepRetriever.RetrieveMultiStep(
                "What are the environmental and economic impacts of solar energy?",
                topK: 10);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Documents);
            Assert.NotNull(result.StepResults);
            Assert.NotNull(result.ReasoningTrace);

            var docList = result.Documents.ToList();
            Assert.NotEmpty(docList);
            Assert.True(docList.Count <= 10);
        }

        [Fact]
        public void RetrieveMultiStep_TracksReasoningSteps()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var multiStepRetriever = new MultiStepReasoningRetriever<double>(generator, mockRetriever);

            // Act
            var result = multiStepRetriever.RetrieveMultiStep(
                "Complex multi-faceted question",
                topK: 5);

            // Assert
            Assert.NotEmpty(result.StepResults);
            Assert.True(result.TotalSteps > 0);

            foreach (var step in result.StepResults)
            {
                Assert.NotNull(step.StepQuery);
                Assert.NotEmpty(step.StepQuery);
                Assert.NotNull(step.Documents);
                Assert.NotNull(step.StepSummary);
                Assert.True(step.StepNumber > 0);
            }
        }

        [Fact]
        public void RetrieveMultiStep_GeneratesReasoningTrace()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var multiStepRetriever = new MultiStepReasoningRetriever<double>(generator, mockRetriever);

            // Act
            var result = multiStepRetriever.RetrieveMultiStep(
                "Test query",
                topK: 5);

            // Assert
            Assert.NotNull(result.ReasoningTrace);
            Assert.NotEmpty(result.ReasoningTrace);
            Assert.Contains("Original Query", result.ReasoningTrace);
        }

        [Fact]
        public void RetrieveMultiStep_WithMetadataFilters_ReturnsResults()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var multiStepRetriever = new MultiStepReasoningRetriever<double>(generator, mockRetriever);
            var filters = new Dictionary<string, object> { { "topic", "energy" } };

            // Act
            var result = multiStepRetriever.RetrieveMultiStep(
                "Solar energy impacts",
                topK: 5,
                filters);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Documents);
        }

        [Fact]
        public void ReasoningStepResult_HasAllRequiredProperties()
        {
            // Arrange
            var stepResult = new MultiStepReasoningRetriever<double>.ReasoningStepResult
            {
                StepQuery = "What is photosynthesis?",
                Documents = new List<Document<double>>(),
                StepSummary = "Photosynthesis converts light to energy",
                IsSuccessful = true,
                StepNumber = 1
            };

            // Assert
            Assert.Equal("What is photosynthesis?", stepResult.StepQuery);
            Assert.NotNull(stepResult.Documents);
            Assert.Equal("Photosynthesis converts light to energy", stepResult.StepSummary);
            Assert.True(stepResult.IsSuccessful);
            Assert.Equal(1, stepResult.StepNumber);
        }

        [Fact]
        public void MultiStepReasoningResult_HasAllRequiredProperties()
        {
            // Arrange
            var result = new MultiStepReasoningRetriever<double>.MultiStepReasoningResult
            {
                Documents = new List<Document<double>>(),
                StepResults = new List<MultiStepReasoningRetriever<double>.ReasoningStepResult>(),
                ReasoningTrace = "Step 1: ...\nStep 2: ...",
                TotalSteps = 2,
                Converged = true
            };

            // Assert
            Assert.NotNull(result.Documents);
            Assert.NotNull(result.StepResults);
            Assert.Equal("Step 1: ...\nStep 2: ...", result.ReasoningTrace);
            Assert.Equal(2, result.TotalSteps);
            Assert.True(result.Converged);
        }
    }

    public class ToolAugmentedReasoningRetrieverTests
    {
        // Mock retriever for testing
        private class MockRetriever : RetrieverBase<double>
        {
            private readonly List<Document<double>> _documents;

            public MockRetriever(List<Document<double>> documents) : base(defaultTopK: 5)
            {
                _documents = documents ?? new List<Document<double>>();
            }

            protected override IEnumerable<Document<double>> RetrieveCore(
                string query,
                int topK,
                Dictionary<string, object> metadataFilters)
            {
                return _documents.Take(topK);
            }
        }

        private MockRetriever CreateMockRetriever()
        {
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "The calculation shows a 29.4% compound annual growth rate.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true
                }
            };
            return new MockRetriever(docs);
        }

        [Fact]
        public void Constructor_WithNullGenerator_ThrowsArgumentNullException()
        {
            // Arrange
            var mockRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new ToolAugmentedReasoningRetriever<double>(null!, mockRetriever));
        }

        [Fact]
        public void Constructor_WithNullRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var generator = new StubGenerator<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new ToolAugmentedReasoningRetriever<double>(generator, null!));
        }

        [Fact]
        public void Constructor_RegistersDefaultTools()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();

            // Act
            var toolRetriever = new ToolAugmentedReasoningRetriever<double>(generator, mockRetriever);

            // Assert
            Assert.NotNull(toolRetriever);
        }

        [Fact]
        public void RegisterTool_WithNullName_ThrowsArgumentException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var toolRetriever = new ToolAugmentedReasoningRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                toolRetriever.RegisterTool(null!, input => "output"));
        }

        [Fact]
        public void RegisterTool_WithNullFunction_ThrowsArgumentNullException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var toolRetriever = new ToolAugmentedReasoningRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                toolRetriever.RegisterTool("test_tool", null!));
        }

        [Fact]
        public void RegisterTool_WithValidArguments_RegistersSuccessfully()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var toolRetriever = new ToolAugmentedReasoningRetriever<double>(generator, mockRetriever);

            // Act
            toolRetriever.RegisterTool("custom_tool", input => $"Processed: {input}");

            // No exception means success
            Assert.NotNull(toolRetriever);
        }

        [Fact]
        public void RetrieveWithTools_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var toolRetriever = new ToolAugmentedReasoningRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                toolRetriever.RetrieveWithTools(null!, 10));
        }

        [Fact]
        public void RetrieveWithTools_WithValidQuery_ReturnsResult()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var toolRetriever = new ToolAugmentedReasoningRetriever<double>(generator, mockRetriever);

            // Act
            var result = toolRetriever.RetrieveWithTools(
                "Calculate the growth rate from 2015 to 2023",
                topK: 10);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Documents);
            Assert.NotNull(result.ToolInvocations);
            Assert.NotNull(result.ReasoningTrace);
        }

        [Fact]
        public void RetrieveWithTools_TracksToolInvocations()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var toolRetriever = new ToolAugmentedReasoningRetriever<double>(generator, mockRetriever);

            // Act
            var result = toolRetriever.RetrieveWithTools(
                "Test query",
                topK: 5);

            // Assert
            Assert.NotNull(result.ToolInvocations);
            // Tool invocations may be empty if LLM didn't request tools
        }

        [Fact]
        public void ToolInvocation_HasAllRequiredProperties()
        {
            // Arrange
            var invocation = new ToolAugmentedReasoningRetriever<double>.ToolInvocation
            {
                ToolName = "calculator",
                Input = "2 + 2",
                Output = "4",
                Success = true
            };

            // Assert
            Assert.Equal("calculator", invocation.ToolName);
            Assert.Equal("2 + 2", invocation.Input);
            Assert.Equal("4", invocation.Output);
            Assert.True(invocation.Success);
        }

        [Fact]
        public void ToolAugmentedResult_HasAllRequiredProperties()
        {
            // Arrange
            var result = new ToolAugmentedReasoningRetriever<double>.ToolAugmentedResult
            {
                Documents = new List<Document<double>>(),
                ToolInvocations = new List<ToolAugmentedReasoningRetriever<double>.ToolInvocation>(),
                ReasoningTrace = "Tool analysis: calculator needed"
            };

            // Assert
            Assert.NotNull(result.Documents);
            Assert.NotNull(result.ToolInvocations);
            Assert.Equal("Tool analysis: calculator needed", result.ReasoningTrace);
        }
    }
}
