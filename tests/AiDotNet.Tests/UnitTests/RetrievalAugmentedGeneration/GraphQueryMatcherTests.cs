using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    public class GraphQueryMatcherTests
    {
        private readonly KnowledgeGraph<double> _graph;
        private readonly GraphQueryMatcher<double> _matcher;

        public GraphQueryMatcherTests()
        {
            _graph = new KnowledgeGraph<double>();
            _matcher = new GraphQueryMatcher<double>(_graph);

            SetupTestData();
        }

        private GraphNode<double> CreateNode(string id, string label, Dictionary<string, object>? properties = null)
        {
            var node = new GraphNode<double>(id, label);
            if (properties != null)
            {
                foreach (var kvp in properties)
                {
                    node.SetProperty(kvp.Key, kvp.Value);
                }
            }
            return node;
        }

        private GraphEdge<double> CreateEdge(string sourceId, string relationType, string targetId, double weight = 1.0)
        {
            return new GraphEdge<double>(sourceId, targetId, relationType, weight);
        }

        private void SetupTestData()
        {
            // Create people
            _graph.AddNode(CreateNode("alice", "Person", new Dictionary<string, object>
            {
                { "name", "Alice" },
                { "age", 30 }
            }));

            _graph.AddNode(CreateNode("bob", "Person", new Dictionary<string, object>
            {
                { "name", "Bob" },
                { "age", 35 }
            }));

            _graph.AddNode(CreateNode("charlie", "Person", new Dictionary<string, object>
            {
                { "name", "Charlie" },
                { "age", 28 }
            }));

            // Create companies
            _graph.AddNode(CreateNode("google", "Company", new Dictionary<string, object>
            {
                { "name", "Google" },
                { "industry", "Tech" }
            }));

            _graph.AddNode(CreateNode("microsoft", "Company", new Dictionary<string, object>
            {
                { "name", "Microsoft" },
                { "industry", "Tech" }
            }));

            // Create relationships
            _graph.AddEdge(CreateEdge("alice", "KNOWS", "bob"));
            _graph.AddEdge(CreateEdge("bob", "KNOWS", "charlie"));
            _graph.AddEdge(CreateEdge("alice", "WORKS_AT", "google"));
            _graph.AddEdge(CreateEdge("bob", "WORKS_AT", "microsoft"));
            _graph.AddEdge(CreateEdge("charlie", "WORKS_AT", "google"));
        }

        #region FindNodes Tests

        [Fact]
        public void FindNodes_ByLabel_ReturnsAllMatchingNodes()
        {
            // Act
            var people = _matcher.FindNodes("Person");

            // Assert
            Assert.Equal(3, people.Count);
            Assert.All(people, p => Assert.Equal("Person", p.Label));
        }

        [Fact]
        public void FindNodes_ByLabelAndProperty_ReturnsFilteredNodes()
        {
            // Arrange
            var props = new Dictionary<string, object> { { "name", "Alice" } };

            // Act
            var results = _matcher.FindNodes("Person", props);

            // Assert
            Assert.Single(results);
            Assert.Equal("alice", results[0].Id);
        }

        [Fact]
        public void FindNodes_MultipleProperties_FiltersCorrectly()
        {
            // Arrange
            var props = new Dictionary<string, object>
            {
                { "name", "Alice" },
                { "age", 30 }
            };

            // Act
            var results = _matcher.FindNodes("Person", props);

            // Assert
            Assert.Single(results);
            Assert.Equal("alice", results[0].Id);
        }

        [Fact]
        public void FindNodes_NoMatches_ReturnsEmptyList()
        {
            // Arrange
            var props = new Dictionary<string, object> { { "name", "NonExistent" } };

            // Act
            var results = _matcher.FindNodes("Person", props);

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void FindNodes_InvalidLabel_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _matcher.FindNodes(null!));
            Assert.Throws<ArgumentException>(() => _matcher.FindNodes(""));
            Assert.Throws<ArgumentException>(() => _matcher.FindNodes("   "));
        }

        #endregion

        #region FindPaths Tests

        [Fact]
        public void FindPaths_SimplePattern_ReturnsMatchingPaths()
        {
            // Act
            var paths = _matcher.FindPaths("Person", "KNOWS", "Person");

            // Assert
            Assert.Equal(2, paths.Count); // Alice->Bob and Bob->Charlie
            Assert.All(paths, p =>
            {
                Assert.Equal("Person", p.SourceNode.Label);
                Assert.Equal("KNOWS", p.Edge.RelationType);
                Assert.Equal("Person", p.TargetNode.Label);
            });
        }

        [Fact]
        public void FindPaths_WithSourceFilter_ReturnsFilteredPaths()
        {
            // Arrange
            var sourceProps = new Dictionary<string, object> { { "name", "Alice" } };

            // Act
            var paths = _matcher.FindPaths("Person", "KNOWS", "Person", sourceProps);

            // Assert
            Assert.Single(paths); // Only Alice->Bob
            Assert.Equal("alice", paths[0].SourceNode.Id);
            Assert.Equal("bob", paths[0].TargetNode.Id);
        }

        [Fact]
        public void FindPaths_WithTargetFilter_ReturnsFilteredPaths()
        {
            // Arrange
            var targetProps = new Dictionary<string, object> { { "name", "Charlie" } };

            // Act
            var paths = _matcher.FindPaths("Person", "KNOWS", "Person", null, targetProps);

            // Assert
            Assert.Single(paths); // Only Bob->Charlie
            Assert.Equal("bob", paths[0].SourceNode.Id);
            Assert.Equal("charlie", paths[0].TargetNode.Id);
        }

        [Fact]
        public void FindPaths_DifferentLabels_WorksCorrectly()
        {
            // Act
            var paths = _matcher.FindPaths("Person", "WORKS_AT", "Company");

            // Assert
            Assert.Equal(3, paths.Count); // Alice->Google, Bob->Microsoft, Charlie->Google
            Assert.All(paths, p =>
            {
                Assert.Equal("Person", p.SourceNode.Label);
                Assert.Equal("WORKS_AT", p.Edge.RelationType);
                Assert.Equal("Company", p.TargetNode.Label);
            });
        }

        [Fact]
        public void FindPaths_WithBothFilters_ReturnsSpecificPath()
        {
            // Arrange
            var sourceProps = new Dictionary<string, object> { { "name", "Alice" } };
            var targetProps = new Dictionary<string, object> { { "name", "Google" } };

            // Act
            var paths = _matcher.FindPaths("Person", "WORKS_AT", "Company", sourceProps, targetProps);

            // Assert
            Assert.Single(paths);
            Assert.Equal("alice", paths[0].SourceNode.Id);
            Assert.Equal("google", paths[0].TargetNode.Id);
        }

        [Fact]
        public void FindPaths_InvalidArguments_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _matcher.FindPaths(null!, "KNOWS", "Person"));

            Assert.Throws<ArgumentException>(() =>
                _matcher.FindPaths("Person", null!, "Person"));

            Assert.Throws<ArgumentException>(() =>
                _matcher.FindPaths("Person", "KNOWS", null!));
        }

        #endregion

        #region FindPathsOfLength Tests

        [Fact]
        public void FindPathsOfLength_Length1_ReturnsDirectNeighbors()
        {
            // Act
            var paths = _matcher.FindPathsOfLength("alice", 1);

            // Assert
            Assert.Equal(2, paths.Count); // Alice->Bob and Alice->Google
            Assert.All(paths, p =>
            {
                Assert.Equal(2, p.Count); // Source + Target
                Assert.Equal("alice", p[0].Id);
            });
        }

        [Fact]
        public void FindPathsOfLength_Length2_ReturnsDistantNodes()
        {
            // Act
            var paths = _matcher.FindPathsOfLength("alice", 2);

            // Assert
            Assert.NotEmpty(paths);
            Assert.All(paths, p => Assert.Equal(3, p.Count)); // Source + Intermediate + Target
        }

        [Fact]
        public void FindPathsOfLength_WithRelationshipFilter_FiltersCorrectly()
        {
            // Act
            var paths = _matcher.FindPathsOfLength("alice", 1, "KNOWS");

            // Assert
            Assert.Single(paths); // Only Alice->Bob via KNOWS
            Assert.Equal("bob", paths[0][1].Id);
        }

        [Fact]
        public void FindPathsOfLength_AvoidsCycles()
        {
            // Act - This would create a cycle if not handled
            var paths = _matcher.FindPathsOfLength("alice", 5);

            // Assert - Should not contain cycles (same node appearing twice)
            Assert.All(paths, p =>
            {
                var nodeIds = p.Select(n => n.Id).ToList();
                Assert.Equal(nodeIds.Count, nodeIds.Distinct().Count());
            });
        }

        [Fact]
        public void FindPathsOfLength_InvalidArguments_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _matcher.FindPathsOfLength(null!, 1));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                _matcher.FindPathsOfLength("alice", 0));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                _matcher.FindPathsOfLength("alice", -1));
        }

        #endregion

        #region FindShortestPaths Tests

        [Fact]
        public void FindShortestPaths_DirectConnection_ReturnsShortestPath()
        {
            // Act
            var paths = _matcher.FindShortestPaths("alice", "bob");

            // Assert
            Assert.Single(paths);
            Assert.Equal(2, paths[0].Count); // Alice -> Bob
            Assert.Equal("alice", paths[0][0].Id);
            Assert.Equal("bob", paths[0][1].Id);
        }

        [Fact]
        public void FindShortestPaths_IndirectConnection_FindsPath()
        {
            // Act
            var paths = _matcher.FindShortestPaths("alice", "charlie");

            // Assert
            Assert.NotEmpty(paths);
            var shortestPath = paths[0];
            Assert.Equal("alice", shortestPath[0].Id);
            Assert.Equal("charlie", shortestPath[^1].Id);
        }

        [Fact]
        public void FindShortestPaths_SameNode_ReturnsSingleNodePath()
        {
            // Act
            var paths = _matcher.FindShortestPaths("alice", "alice");

            // Assert
            Assert.Single(paths);
            Assert.Single(paths[0]);
            Assert.Equal("alice", paths[0][0].Id);
        }

        [Fact]
        public void FindShortestPaths_NoConnection_ReturnsEmpty()
        {
            // Arrange - Add isolated node
            _graph.AddNode(CreateNode("isolated", "Person", new Dictionary<string, object> { { "name", "Isolated" } }));

            // Act
            var paths = _matcher.FindShortestPaths("alice", "isolated");

            // Assert
            Assert.Empty(paths);
        }

        [Fact]
        public void FindShortestPaths_InvalidArguments_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _matcher.FindShortestPaths(null!, "bob"));

            Assert.Throws<ArgumentException>(() =>
                _matcher.FindShortestPaths("alice", null!));
        }

        [Fact]
        public void FindShortestPaths_RespectsMaxDepth()
        {
            // Act - Set maxDepth to 1, so can't reach Charlie from Alice
            var paths = _matcher.FindShortestPaths("alice", "charlie", maxDepth: 1);

            // Assert - Charlie is 2 hops away, so shouldn't be found
            Assert.Empty(paths);
        }

        #endregion

        #region ExecutePattern Tests

        [Fact]
        public void ExecutePattern_SimplePattern_ReturnsMatches()
        {
            // Act
            var paths = _matcher.ExecutePattern("(Person)-[KNOWS]->(Person)");

            // Assert
            Assert.Equal(2, paths.Count);
        }

        [Fact]
        public void ExecutePattern_WithSourceProperty_FiltersCorrectly()
        {
            // Act
            var paths = _matcher.ExecutePattern("(Person {name: \"Alice\"})-[KNOWS]->(Person)");

            // Assert
            Assert.Single(paths);
            Assert.Equal("alice", paths[0].SourceNode.Id);
        }

        [Fact]
        public void ExecutePattern_WithTargetProperty_FiltersCorrectly()
        {
            // Act
            var paths = _matcher.ExecutePattern("(Person)-[WORKS_AT]->(Company {name: \"Google\"})");

            // Assert
            Assert.Equal(2, paths.Count); // Alice and Charlie work at Google
            Assert.All(paths, p => Assert.Equal("google", p.TargetNode.Id));
        }

        [Fact]
        public void ExecutePattern_WithBothProperties_FindsSpecificPath()
        {
            // Act
            var paths = _matcher.ExecutePattern("(Person {name: \"Alice\"})-[WORKS_AT]->(Company {name: \"Google\"})");

            // Assert
            Assert.Single(paths);
            Assert.Equal("alice", paths[0].SourceNode.Id);
            Assert.Equal("google", paths[0].TargetNode.Id);
        }

        [Fact]
        public void ExecutePattern_InvalidFormat_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _matcher.ExecutePattern("invalid pattern"));

            Assert.Throws<ArgumentException>(() =>
                _matcher.ExecutePattern("(Person)"));

            Assert.Throws<ArgumentException>(() =>
                _matcher.ExecutePattern(null!));
        }

        #endregion

        #region GraphPath ToString Tests

        [Fact]
        public void GraphPath_ToString_FormatsCorrectly()
        {
            // Arrange
            var paths = _matcher.FindPaths("Person", "KNOWS", "Person");

            // Act
            var pathString = paths[0].ToString();

            // Assert
            Assert.Contains("Person:", pathString);
            Assert.Contains("-[KNOWS]->", pathString);
        }

        #endregion

        #region Complex Scenario Tests

        [Fact]
        public void FindPaths_ComplexQuery_HandlesMultipleCriteria()
        {
            // Arrange - Find people older than 30 who work at tech companies
            var sourceProps = new Dictionary<string, object> { { "age", 35 } };
            var targetProps = new Dictionary<string, object> { { "industry", "Tech" } };

            // Act
            var paths = _matcher.FindPaths("Person", "WORKS_AT", "Company", sourceProps, targetProps);

            // Assert
            Assert.Single(paths); // Only Bob (age 35) works at Microsoft (Tech)
            Assert.Equal("bob", paths[0].SourceNode.Id);
        }

        [Fact]
        public void FindPathsOfLength_ComplexGraph_FindsAllPaths()
        {
            // Arrange - Add more connections to create multiple paths
            _graph.AddEdge(CreateEdge("alice", "KNOWS", "charlie"));

            // Act - Find 1-hop paths from Alice
            var paths = _matcher.FindPathsOfLength("alice", 1);

            // Assert - Should now have 3 paths (Bob, Charlie, Google)
            Assert.Equal(3, paths.Count);
        }

        #endregion

        #region Numeric Property Tests

        [Fact]
        public void FindNodes_NumericProperty_ComparesCorrectly()
        {
            // Arrange
            var props = new Dictionary<string, object> { { "age", 30 } };

            // Act
            var results = _matcher.FindNodes("Person", props);

            // Assert
            Assert.Single(results);
            Assert.Equal("alice", results[0].Id);
        }

        [Fact]
        public void ExecutePattern_NumericProperty_ParsesCorrectly()
        {
            // Act - Pattern with numeric property
            var paths = _matcher.ExecutePattern("(Person {age: 30})-[KNOWS]->(Person)");

            // Assert
            Assert.Single(paths);
            Assert.Equal("alice", paths[0].SourceNode.Id);
        }

        #endregion
    }
}
