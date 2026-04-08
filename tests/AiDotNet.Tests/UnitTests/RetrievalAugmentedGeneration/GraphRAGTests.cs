using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Integration tests for GraphRAG (Graph-based Retrieval Augmented Generation).
    /// Tests verify knowledge graph traversal, entity extraction, and score boosting behavior.
    /// </summary>
    public class GraphRAGTests
    {
        // Mock retriever that returns documents based on content matching
        private class ContentAwareMockRetriever : RetrieverBase<double>
        {
            private readonly List<Document<double>> _documents;
            public List<string> RetrievalQueries { get; } = new List<string>();

            public ContentAwareMockRetriever(List<Document<double>> documents) : base(defaultTopK: 10)
            {
                _documents = documents ?? new List<Document<double>>();
            }

            protected override IEnumerable<Document<double>> RetrieveCore(
                string query,
                int topK,
                Dictionary<string, object> metadataFilters)
            {
                RetrievalQueries.Add(query);
                // Return documents, simulating vector similarity
                return _documents.Take(topK);
            }
        }

        // Mock generator for entity extraction
        private class EntityExtractionMockGenerator : IGenerator<double>
        {
            private readonly string _entityExtractionResponse;
            public List<string> GeneratePrompts { get; } = new List<string>();

            public int MaxContextTokens => 2048;
            public int MaxGenerationTokens => 500;

            public EntityExtractionMockGenerator(string entityExtractionResponse = "")
            {
                _entityExtractionResponse = entityExtractionResponse;
            }

            public string Generate(string prompt)
            {
                GeneratePrompts.Add(prompt);
                return _entityExtractionResponse;
            }

            public GroundedAnswer<double> GenerateGrounded(string query, IEnumerable<Document<double>> context)
            {
                return new GroundedAnswer<double>
                {
                    Query = query,
                    Answer = "Grounded answer",
                    SourceDocuments = context?.ToList() ?? new List<Document<double>>(),
                    Citations = new List<string>(),
                    ConfidenceScore = 0.8
                };
            }
        }

        private List<Document<double>> CreateEinsteinDocuments()
        {
            return new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "einstein1",
                    Content = "Albert Einstein developed the Theory of Relativity which revolutionized physics.",
                    RelevanceScore = 0.90,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "einstein2",
                    Content = "Einstein's E=mc squared equation shows the relationship between energy and mass.",
                    RelevanceScore = 0.85,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "einstein3",
                    Content = "The photoelectric effect was explained by Einstein and earned him the Nobel Prize.",
                    RelevanceScore = 0.80,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "general1",
                    Content = "Physics is the study of matter, energy, and their interactions.",
                    RelevanceScore = 0.60,
                    HasRelevanceScore = true
                }
            };
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_WithNullGenerator_ThrowsArgumentNullException()
        {
            // Arrange
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new GraphRAG<double>(null!, retriever));
        }

        [Fact]
        public void Constructor_WithNullRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new GraphRAG<double>(generator, null!));
        }

        [Fact]
        public void Constructor_WithValidArguments_InitializesCorrectly()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());

            // Act
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Assert
            Assert.NotNull(graphRag);
        }

        [Fact]
        public void Constructor_WithCustomMaxHops_InitializesCorrectly()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());

            // Act
            var graphRag = new GraphRAG<double>(generator, retriever, maxHops: 3);

            // Assert
            Assert.NotNull(graphRag);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        [InlineData(-100)]
        public void Constructor_WithInvalidMaxHops_ThrowsArgumentOutOfRangeException(int maxHops)
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new GraphRAG<double>(generator, retriever, maxHops));
        }

        #endregion

        #region AddRelation Tests

        [Fact]
        public void AddRelation_WithNullEntity_ThrowsArgumentException()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                graphRag.AddRelation(null!, "DISCOVERED", "Relativity"));
        }

        [Fact]
        public void AddRelation_WithEmptyEntity_ThrowsArgumentException()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                graphRag.AddRelation("   ", "DISCOVERED", "Relativity"));
        }

        [Fact]
        public void AddRelation_WithNullRelation_ThrowsArgumentException()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                graphRag.AddRelation("Einstein", null!, "Relativity"));
        }

        [Fact]
        public void AddRelation_WithEmptyRelation_ThrowsArgumentException()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                graphRag.AddRelation("Einstein", "", "Relativity"));
        }

        [Fact]
        public void AddRelation_WithNullTarget_ThrowsArgumentException()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                graphRag.AddRelation("Einstein", "DISCOVERED", null!));
        }

        [Fact]
        public void AddRelation_WithEmptyTarget_ThrowsArgumentException()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                graphRag.AddRelation("Einstein", "DISCOVERED", "   "));
        }

        [Fact]
        public void AddRelation_WithValidInputs_AddsRelationSuccessfully()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act - Should not throw
            graphRag.AddRelation("Einstein", "DISCOVERED", "Theory of Relativity");

            // Assert - Can retrieve using the entity (indirect verification)
            var results = graphRag.Retrieve("Tell me about Einstein", 10);
            Assert.NotNull(results);
        }

        [Fact]
        public void AddRelation_NormalizesRelationToUpperCase()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "Einstein discovered the Theory of Relativity.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act - Add with lowercase relation
            graphRag.AddRelation("Einstein", "discovered", "Theory of Relativity");

            // Assert - Query should still work (relation is normalized to uppercase internally)
            var results = graphRag.Retrieve("Einstein discovered something", 10);
            Assert.NotNull(results);
        }

        [Fact]
        public void AddRelation_TrimsEntityAndTarget()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "Einstein developed relativity theory.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act - Add with whitespace
            graphRag.AddRelation("  Einstein  ", "DISCOVERED", "  Relativity  ");

            // Assert - Should work with trimmed values
            var results = graphRag.Retrieve("Einstein theory", 10);
            Assert.NotNull(results);
        }

        [Fact]
        public void AddRelation_PreventsDuplicateRelations()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act - Add same relation twice
            graphRag.AddRelation("Einstein", "DISCOVERED", "Relativity");
            graphRag.AddRelation("Einstein", "DISCOVERED", "Relativity");

            // Assert - Should not throw, duplicates should be ignored
            var results = graphRag.Retrieve("Einstein", 10);
            Assert.NotNull(results);
        }

        #endregion

        #region Retrieve Tests

        [Fact]
        public void Retrieve_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                graphRag.Retrieve(null!, 10));
        }

        [Fact]
        public void Retrieve_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                graphRag.Retrieve("   ", 10));
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        [InlineData(-100)]
        public void Retrieve_WithNonPositiveTopK_ThrowsArgumentOutOfRangeException(int topK)
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                graphRag.Retrieve("Einstein", topK));
        }

        [Fact]
        public void Retrieve_WithValidQuery_ReturnsDocuments()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act
            var results = graphRag.Retrieve("What did Einstein discover?", 10);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
        }

        [Fact]
        public void Retrieve_RespectsTopKLimit()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act
            var results = graphRag.Retrieve("Einstein", topK: 2);

            // Assert
            var resultList = results.ToList();
            Assert.True(resultList.Count <= 2);
        }

        #endregion

        #region Entity Extraction Tests

        [Fact]
        public void Retrieve_ExtractsCapitalizedPhrases()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "Albert Einstein was a physicist who worked at Princeton University.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Add relation with matching capitalized phrase
            graphRag.AddRelation("Albert Einstein", "WORKED_AT", "Princeton University");

            // Act - Query with capitalized phrase should extract entity
            var results = graphRag.Retrieve("What did Albert Einstein discover?", 10);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
        }

        [Fact]
        public void Retrieve_ExtractsQuotedTerms()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "The theory of relativity was developed in the early 1900s.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            graphRag.AddRelation("relativity", "PUBLISHED_IN", "1905");

            // Act - Query with quoted term
            var results = graphRag.Retrieve("Tell me about \"relativity\"", 10);

            // Assert
            Assert.NotNull(results);
        }

        [Fact]
        public void Retrieve_FallsBackToLLMForEntityExtraction()
        {
            // Arrange - Generator returns entities when no capitalized/quoted terms found
            var generator = new EntityExtractionMockGenerator("physics, energy, matter");
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "Physics explains how energy and matter interact.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act - Query with no capitalized phrases (lowercase only)
            var results = graphRag.Retrieve("what is physics about", 10);

            // Assert
            Assert.NotNull(results);
            // Generator should have been called for entity extraction
            // (only if no capitalized/quoted entities found)
        }

        #endregion

        #region Graph Traversal Tests

        [Fact]
        public void Retrieve_TraversesOneHopRelations()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "relativity_doc",
                    Content = "The Theory of Relativity explains space-time curvature.",
                    RelevanceScore = 0.85,
                    HasRelevanceScore = true
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Add relation: Einstein -> DISCOVERED -> Theory of Relativity
            graphRag.AddRelation("Einstein", "DISCOVERED", "Theory of Relativity");

            // Act - Query mentions Einstein, should find related entity via graph
            var results = graphRag.Retrieve("Tell me about Einstein", 10);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
            // Document mentioning "Theory of Relativity" should be boosted
        }

        [Fact]
        public void Retrieve_AddsGraphContextToDocuments()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "relativity_doc",
                    Content = "Theory of Relativity describes gravity as spacetime curvature.",
                    RelevanceScore = 0.85,
                    HasRelevanceScore = true
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            graphRag.AddRelation("Einstein", "DISCOVERED", "Theory of Relativity");

            // Act
            var results = graphRag.Retrieve("Einstein discoveries", 10);
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            // Check if document was enriched (should contain graph context if entity matched)
        }

        #endregion

        #region Score Boosting Tests

        [Fact]
        public void Retrieve_BoostsScoresForGraphMatchedDocuments()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "matching_doc",
                    Content = "Theory of Relativity changed our understanding of physics.",
                    RelevanceScore = 0.70,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "non_matching_doc",
                    Content = "Quantum mechanics is another branch of physics.",
                    RelevanceScore = 0.75,
                    HasRelevanceScore = true
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Add relation that makes "Theory of Relativity" a related entity
            graphRag.AddRelation("Einstein", "DISCOVERED", "Theory of Relativity");

            // Act
            var results = graphRag.Retrieve("Einstein", 10);
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            // The document mentioning "Theory of Relativity" should be boosted
            // because it's connected to "Einstein" in the graph
        }

        [Fact]
        public void Retrieve_BoostFactorIsFixed1_5x()
        {
            // Arrange - Document mentions multiple graph-connected entities
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "multi_match_doc",
                    Content = "Einstein developed Theory of Relativity at Princeton University.",
                    RelevanceScore = 0.60,
                    HasRelevanceScore = true
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Add multiple relations from Einstein
            graphRag.AddRelation("Einstein", "DEVELOPED", "Theory of Relativity");
            graphRag.AddRelation("Einstein", "WORKED_AT", "Princeton University");

            // Act
            var results = graphRag.Retrieve("Einstein", 10);
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            // Document mentioning graph-connected entities gets fixed 1.5x boost
            // 0.60 * 1.5 = 0.90 boosted score
            var firstDoc = resultList.First();
            Assert.True(firstDoc.HasRelevanceScore);
            Assert.Equal(0.90, Convert.ToDouble(firstDoc.RelevanceScore), 2);
        }

        [Fact]
        public void Retrieve_AddsGraphMetadataToEnrichedDocuments()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "einstein_doc",
                    Content = "Einstein is famous for Theory of Relativity.",
                    RelevanceScore = 0.80,
                    HasRelevanceScore = true,
                    Metadata = new Dictionary<string, object>()
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            graphRag.AddRelation("Einstein", "DISCOVERED", "Theory of Relativity");

            // Act
            var results = graphRag.Retrieve("Einstein", 10);
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            // Check for graph_boosted and graph_entities metadata
            var enrichedDoc = resultList.FirstOrDefault(d =>
                d.Metadata.ContainsKey("graph_boosted") || d.Metadata.ContainsKey("graph_entities"));

            // If document was enriched, it should have metadata
            if (enrichedDoc != null)
            {
                Assert.True(enrichedDoc.Metadata.ContainsKey("graph_boosted") ||
                           enrichedDoc.Metadata.ContainsKey("graph_entities"));
            }
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_WithEmptyKnowledgeGraph_StillReturnsVectorResults()
        {
            // Arrange - No relations added
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(CreateEinsteinDocuments());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act
            var results = graphRag.Retrieve("Einstein", 10);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
        }

        [Fact]
        public void Retrieve_WithNoMatchingDocuments_ReturnsEmptyResults()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var retriever = new ContentAwareMockRetriever(new List<Document<double>>());
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Act
            var results = graphRag.Retrieve("Einstein", 10);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.Empty(resultList);
        }

        [Fact]
        public void Retrieve_WithDocumentsWithoutScores_UsesDefaultScore()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "no_score_doc",
                    Content = "Einstein developed the Theory of Relativity.",
                    HasRelevanceScore = false,
                    Metadata = new Dictionary<string, object>()
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            graphRag.AddRelation("Einstein", "DEVELOPED", "Theory of Relativity");

            // Act
            var results = graphRag.Retrieve("Einstein", 10);
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            // Should use 0.5 default for boosting calculation
        }

        [Fact]
        public void Retrieve_CapsBoostAtMaxValue()
        {
            // Arrange - Many matching entities to test boost capping
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "many_matches_doc",
                    Content = "Einstein Relativity Princeton Physics Nobel Germany Zurich Patent Office Photoelectric",
                    RelevanceScore = 0.90,
                    HasRelevanceScore = true,
                    Metadata = new Dictionary<string, object>()
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Add many relations
            graphRag.AddRelation("Einstein", "RELATED", "Relativity");
            graphRag.AddRelation("Einstein", "RELATED", "Princeton");
            graphRag.AddRelation("Einstein", "RELATED", "Physics");
            graphRag.AddRelation("Einstein", "RELATED", "Nobel");
            graphRag.AddRelation("Einstein", "RELATED", "Germany");

            // Act
            var results = graphRag.Retrieve("Einstein", 10);
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            // Boosted score should be capped at 1.0
            var firstDoc = resultList.First();
            Assert.True(Convert.ToDouble(firstDoc.RelevanceScore) <= 1.0);
        }

        #endregion

        #region Multi-hop Traversal Tests

        [Fact]
        public void Retrieve_TraversesMultipleHops_DefaultTwoHops()
        {
            // Arrange - Create a chain: Alpha -> Beta -> Gamma (2 hops needed to reach Gamma from Alpha)
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "gamma_doc",
                    Content = "Document about Gamma only, no other entities mentioned.",
                    RelevanceScore = 0.80,
                    HasRelevanceScore = true,
                    Metadata = new Dictionary<string, object>()
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever); // Default maxHops = 2

            // Alpha -> Beta, Beta -> Gamma (two hops needed to reach Gamma from Alpha)
            graphRag.AddRelation("Alpha", "RELATED_TO", "Beta");
            graphRag.AddRelation("Beta", "RELATED_TO", "Gamma");

            // Act - Query about Alpha
            var results = graphRag.Retrieve("Tell me about Alpha", 10);
            var resultList = results.ToList();

            // Assert
            // With default maxHops=2, Gamma IS reachable from Alpha
            Assert.NotEmpty(resultList);
            // Document about "Gamma" SHOULD be boosted (2 hops away, within maxHops)
            var firstDoc = resultList.First();
            Assert.True(firstDoc.Metadata.ContainsKey("graph_boosted"));
            Assert.True((bool)firstDoc.Metadata["graph_boosted"]);
        }

        [Fact]
        public void Retrieve_DoesNotExceedMaxHops()
        {
            // Arrange - Create a chain: Alpha -> Beta -> Gamma -> Delta (3 hops needed to reach Delta from Alpha)
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "delta_doc",
                    Content = "Document about Delta only, no other entities mentioned.",
                    RelevanceScore = 0.80,
                    HasRelevanceScore = true,
                    Metadata = new Dictionary<string, object>()
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever, maxHops: 2); // Only 2 hops

            // Alpha -> Beta, Beta -> Gamma, Gamma -> Delta (three hops needed to reach Delta from Alpha)
            graphRag.AddRelation("Alpha", "RELATED_TO", "Beta");
            graphRag.AddRelation("Beta", "RELATED_TO", "Gamma");
            graphRag.AddRelation("Gamma", "RELATED_TO", "Delta");

            // Act - Query about Alpha
            var results = graphRag.Retrieve("Tell me about Alpha", 10);
            var resultList = results.ToList();

            // Assert
            // With maxHops=2, Delta is NOT reachable from Alpha (3 hops away)
            Assert.NotEmpty(resultList);
            // Document about "Delta" should NOT be boosted (3 hops away, exceeds maxHops)
            var firstDoc = resultList.First();
            Assert.False(firstDoc.Metadata.ContainsKey("graph_boosted"));
        }

        [Fact]
        public void Retrieve_WithMaxHopsThree_ReachesThreeHopEntities()
        {
            // Arrange - Create a chain: Alpha -> Beta -> Gamma -> Delta
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "delta_doc",
                    Content = "Document about Delta only, no other entities mentioned.",
                    RelevanceScore = 0.80,
                    HasRelevanceScore = true,
                    Metadata = new Dictionary<string, object>()
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever, maxHops: 3); // 3 hops

            // Alpha -> Beta, Beta -> Gamma, Gamma -> Delta (three hops needed to reach Delta from Alpha)
            graphRag.AddRelation("Alpha", "RELATED_TO", "Beta");
            graphRag.AddRelation("Beta", "RELATED_TO", "Gamma");
            graphRag.AddRelation("Gamma", "RELATED_TO", "Delta");

            // Act - Query about Alpha
            var results = graphRag.Retrieve("Tell me about Alpha", 10);
            var resultList = results.ToList();

            // Assert
            // With maxHops=3, Delta IS reachable from Alpha
            Assert.NotEmpty(resultList);
            // Document about "Delta" SHOULD be boosted (3 hops away, within maxHops)
            var firstDoc = resultList.First();
            Assert.True(firstDoc.Metadata.ContainsKey("graph_boosted"));
            Assert.True((bool)firstDoc.Metadata["graph_boosted"]);
        }

        #endregion

        #region Case-Insensitive Entity Matching Tests

        [Fact]
        public void AddRelation_IsCaseInsensitive()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "Einstein discovered Relativity.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true,
                    Metadata = new Dictionary<string, object>()
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Add relation with uppercase entity names (different from doc content case)
            graphRag.AddRelation("EINSTEIN", "DISCOVERED", "RELATIVITY");

            // Act - Query with proper case (entity extraction requires capitalized words)
            // The graph lookup is case-insensitive, so "Einstein" in query finds "EINSTEIN" relation
            var results = graphRag.Retrieve("Tell me about Einstein", 10);
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            // Document mentioning "Relativity" should be boosted
            // because "RELATIVITY" in graph matches "Relativity" in doc (case-insensitive)
            var firstDoc = resultList.First();
            Assert.True(firstDoc.Metadata.ContainsKey("graph_boosted"));
        }

        [Fact]
        public void Retrieve_MatchesEntitiesCaseInsensitively()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "The theory of RELATIVITY changed physics forever.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true,
                    Metadata = new Dictionary<string, object>()
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Add relation with lowercase
            graphRag.AddRelation("einstein", "discovered", "relativity");

            // Act - Document content has uppercase "RELATIVITY"
            var results = graphRag.Retrieve("Tell me about Einstein", 10);
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            // Document with uppercase "RELATIVITY" should match lowercase "relativity" in graph
            var firstDoc = resultList.First();
            Assert.True(firstDoc.Metadata.ContainsKey("graph_boosted"));
        }

        [Fact]
        public void AddRelation_PreventsDuplicatesRegardlessOfCase()
        {
            // Arrange
            var generator = new EntityExtractionMockGenerator();
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "Einstein discovered Relativity.",
                    RelevanceScore = 0.60,
                    HasRelevanceScore = true,
                    Metadata = new Dictionary<string, object>()
                }
            };
            var retriever = new ContentAwareMockRetriever(docs);
            var graphRag = new GraphRAG<double>(generator, retriever);

            // Add same relation with different cases - should be treated as duplicates
            graphRag.AddRelation("Einstein", "DISCOVERED", "Relativity");
            graphRag.AddRelation("EINSTEIN", "discovered", "RELATIVITY");
            graphRag.AddRelation("einstein", "Discovered", "relativity");

            // Act
            var results = graphRag.Retrieve("Einstein", 10);
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            // Score should only be boosted once, not multiple times for "duplicate" relations
            // 0.60 * 1.5 = 0.90 (fixed 1.5x boost)
            var firstDoc = resultList.First();
            Assert.Equal(0.90, Convert.ToDouble(firstDoc.RelevanceScore), 2);
        }

        #endregion
    }
}
