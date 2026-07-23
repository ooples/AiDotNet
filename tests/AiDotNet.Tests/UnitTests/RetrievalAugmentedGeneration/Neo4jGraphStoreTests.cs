#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for the Neo4j property-graph backend (AiDotNet.Storage.Neo4j). Split into two groups:
    /// (a) pure-logic tests over <see cref="Neo4jCypher"/> — the Cypher/param/mapping builders — which need
    /// no database and always run; and (b) integration tests over <see cref="Neo4jGraphStore{T}"/> that hit
    /// a live Neo4j and are SKIPPED (not failed) unless NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD are set, mirroring
    /// the Postgres/Redis checkpointer integration tests. The project is net10-only (the driver's TFM).
    /// </summary>
    public class Neo4jGraphStoreTests
    {
        // ---------- (a) Pure-logic Cypher/mapping builder tests (no database). ----------

        [Fact]
        public void EscapeIdentifier_WrapsInBackticks()
        {
            Assert.Equal("`Entity`", Neo4jCypher.EscapeIdentifier("Entity"));
        }

        [Fact]
        public void EscapeIdentifier_DoublesEmbeddedBackticks_PreventingBreakout()
        {
            // A malicious "identifier" that tries to close the backtick and inject Cypher must be neutralized
            // by doubling the internal backtick, keeping the whole thing a single identifier token.
            var malicious = "Entity`) DETACH DELETE (x) //";
            var escaped = Neo4jCypher.EscapeIdentifier(malicious);
            Assert.Equal("`Entity``) DETACH DELETE (x) //`", escaped);
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("   ")]
        public void EscapeIdentifier_Throws_OnNullOrWhitespace(string? identifier)
        {
            Assert.Throws<ArgumentException>(() => Neo4jCypher.EscapeIdentifier(identifier!));
        }

        [Fact]
        public void EscapeIdentifier_Throws_OnNulChar()
        {
            Assert.Throws<ArgumentException>(() => Neo4jCypher.EscapeIdentifier("Ent\0ity"));
        }

        [Fact]
        public void Constructor_EscapesLabelAndRelationshipType()
        {
            var cypher = new Neo4jCypher("Person", "KNOWS");
            Assert.Equal("`Person`", cypher.NodeLabel);
            Assert.Equal("`KNOWS`", cypher.RelationshipType);
        }

        [Fact]
        public void Constructor_UsesDefaults()
        {
            var cypher = new Neo4jCypher();
            Assert.Equal("`Entity`", cypher.NodeLabel);
            Assert.Equal("`RELATED`", cypher.RelationshipType);
        }

        [Fact]
        public void UpsertNode_MergesByIdAndSetsAllColumns_Parameterized()
        {
            var cypher = new Neo4jCypher();
            var q = cypher.UpsertNode();

            Assert.Contains("MERGE (n:`Entity` {id: $id})", q);
            Assert.Contains("n.label = $label", q);
            Assert.Contains("n.props = $props", q);
            Assert.Contains("n.embedding = $embedding", q);
            Assert.Contains("n.createdAt = $createdAt", q);
            Assert.Contains("n.updatedAt = $updatedAt", q);
        }

        [Fact]
        public void UpsertEdge_MatchesBothEndpoints_MergesRelationship_AndReturnsIt()
        {
            var cypher = new Neo4jCypher();
            var q = cypher.UpsertEdge();

            Assert.Contains("MATCH (s:`Entity` {id: $sourceId})", q);
            Assert.Contains("MATCH (t:`Entity` {id: $targetId})", q);
            Assert.Contains("MERGE (s)-[r:`RELATED` {id: $id}]->(t)", q);
            Assert.Contains("r.relationType = $relationType", q);
            Assert.Contains("r.weight = $weight", q);
            Assert.Contains("r.validFrom = $validFrom", q);
            Assert.Contains("r.validUntil = $validUntil", q);
            Assert.EndsWith("RETURN r", q);
        }

        [Fact]
        public void RemoveNode_DetachDeletes_AndReturnsCount()
        {
            var q = new Neo4jCypher().RemoveNode();
            Assert.Contains("MATCH (n:`Entity` {id: $id})", q);
            Assert.Contains("DETACH DELETE n", q);
            Assert.Contains("count(*) AS deleted", q);
        }

        [Fact]
        public void GetOutgoingAndIncoming_TraverseInCorrectDirection()
        {
            var cypher = new Neo4jCypher();
            Assert.Contains("(s:`Entity` {id: $id})-[r:`RELATED`]->(t:`Entity`)", cypher.OutgoingEdges());
            Assert.Contains("(s:`Entity`)-[r:`RELATED`]->(t:`Entity` {id: $id})", cypher.IncomingEdges());
        }

        [Fact]
        public void NodesByLabel_FiltersOnTheLabelProperty()
        {
            var q = new Neo4jCypher().NodesByLabel();
            Assert.Contains("MATCH (n:`Entity` {label: $label})", q);
        }

        [Fact]
        public void CountQueries_UseAggregates()
        {
            var cypher = new Neo4jCypher();
            Assert.Contains("count(n) AS c", cypher.NodeCount());
            Assert.Contains("count(r) AS c", cypher.EdgeCount());
        }

        [Fact]
        public void CustomLabel_FlowsIntoEveryQuery()
        {
            var cypher = new Neo4jCypher("Movie", "ACTED_IN");
            Assert.Contains(":`Movie`", cypher.AllNodes());
            Assert.Contains(":`ACTED_IN`", cypher.AllEdges());
            Assert.Contains(":`Movie`", cypher.Clear());
        }

        [Fact]
        public void SerializeProperties_NullOrEmpty_ReturnsEmptyObject()
        {
            Assert.Equal("{}", Neo4jCypher.SerializeProperties(null));
            Assert.Equal("{}", Neo4jCypher.SerializeProperties(new Dictionary<string, object>()));
        }

        [Fact]
        public void Properties_RoundTrip_LosslesslyForStringsAndBooleans()
        {
            var original = new Dictionary<string, object>
            {
                ["name"] = "Einstein",
                ["active"] = true,
                ["score"] = 3.5,
            };

            var json = Neo4jCypher.SerializeProperties(original);
            var restored = Neo4jCypher.DeserializeProperties(json);

            Assert.Equal("Einstein", restored["name"]);
            Assert.Equal(true, restored["active"]);
            Assert.Equal(3.5, Convert.ToDouble(restored["score"]));
        }

        [Fact]
        public void DeserializeProperties_EmptyOrNull_ReturnsEmptyDictionary()
        {
            Assert.Empty(Neo4jCypher.DeserializeProperties(null));
            Assert.Empty(Neo4jCypher.DeserializeProperties(""));
            Assert.Empty(Neo4jCypher.DeserializeProperties("{}"));
        }

        [Fact]
        public void EpochMillis_RoundTripsUtc()
        {
            var when = new DateTime(2026, 7, 19, 12, 30, 45, DateTimeKind.Utc);
            var millis = Neo4jCypher.ToEpochMillis(when);
            var restored = Neo4jCypher.FromEpochMillis(millis);
            Assert.Equal(when, restored);
        }

        [Fact]
        public void EpochMillis_Nullable_PreservesNull()
        {
            Assert.Null(Neo4jCypher.ToEpochMillis((DateTime?)null));
            Assert.Null(Neo4jCypher.FromEpochMillis((long?)null));

            long? millis = Neo4jCypher.ToEpochMillis((DateTime?)new DateTime(2000, 1, 1, 0, 0, 0, DateTimeKind.Utc));
            Assert.NotNull(millis);
            Assert.Equal(new DateTime(2000, 1, 1, 0, 0, 0, DateTimeKind.Utc), Neo4jCypher.FromEpochMillis(millis));
        }

        // ---------- (b) Integration tests (require a live Neo4j; skipped when env vars are unset). ----------

        private const string UriEnv = "NEO4J_URI";
        private const string UserEnv = "NEO4J_USER";
        private const string PasswordEnv = "NEO4J_PASSWORD";

        private static Neo4jGraphStore<double>? CreateStoreOrSkip(out string reason)
        {
            var uri = Environment.GetEnvironmentVariable(UriEnv);
            reason = $"Set {UriEnv}/{UserEnv}/{PasswordEnv} to run the Neo4j integration tests.";
            if (string.IsNullOrWhiteSpace(uri))
            {
                return null;
            }

            var user = Environment.GetEnvironmentVariable(UserEnv);
            var password = Environment.GetEnvironmentVariable(PasswordEnv);
            // Unique per-run label/relationship type so the test partitions its own data and Clear() only
            // touches this run's nodes, never any real graph in the target database.
            var suffix = Guid.NewGuid().ToString("N");
            return new Neo4jGraphStore<double>(uri, user, password, nodeLabel: "AidnTest_" + suffix, relationshipType: "REL_" + suffix);
        }

        [SkippableFact(Timeout = 120000)]
        [Trait("Category", "Integration")]
        public async Task RoundTrips_Nodes_Edges_Traversal_And_Clear()
        {
            var store = CreateStoreOrSkip(out var reason);
            Skip.If(store is null, reason);

            using (store)
            {
                try
                {
                    var alice = new GraphNode<double>("alice", "PERSON");
                    alice.SetProperty("name", "Alice");
                    alice.SetProperty("age", 30);
                    alice.Embedding = new Vector<double>(new[] { 0.1, 0.2, 0.3 });

                    var acme = new GraphNode<double>("acme", "COMPANY");
                    acme.SetProperty("name", "Acme");

                    await store!.AddNodeAsync(alice);
                    await store.AddNodeAsync(acme);

                    var edge = new GraphEdge<double>("alice", "acme", "WORKS_FOR", 0.9);
                    edge.SetProperty("since", "2020");
                    await store.AddEdgeAsync(edge);

                    Assert.Equal(2, store.NodeCount);
                    Assert.Equal(1, store.EdgeCount);

                    // Get node round-trips label, properties, and embedding.
                    var fetched = await store.GetNodeAsync("alice");
                    Assert.NotNull(fetched);
                    Assert.Equal("PERSON", fetched!.Label);
                    Assert.Equal("Alice", fetched.GetProperty<string>("name"));
                    Assert.Equal(30, fetched.GetProperty<int>("age"));
                    Assert.NotNull(fetched.Embedding);
                    Assert.Equal(3, fetched.Embedding!.Length);
                    Assert.Equal(0.2, fetched.Embedding[1], 6);

                    // Get edge round-trips endpoints, type, weight, and properties.
                    var fetchedEdge = await store.GetEdgeAsync(edge.Id);
                    Assert.NotNull(fetchedEdge);
                    Assert.Equal("WORKS_FOR", fetchedEdge!.RelationType);
                    Assert.Equal(0.9, fetchedEdge.Weight, 6);
                    Assert.Equal("2020", fetchedEdge.GetProperty<string>("since"));

                    // Traversal.
                    var outgoing = (await store.GetOutgoingEdgesAsync("alice")).ToList();
                    Assert.Single(outgoing);
                    Assert.Equal("acme", outgoing[0].TargetId);

                    var incoming = (await store.GetIncomingEdgesAsync("acme")).ToList();
                    Assert.Single(incoming);
                    Assert.Equal("alice", incoming[0].SourceId);

                    // Label index.
                    var people = (await store.GetNodesByLabelAsync("PERSON")).ToList();
                    Assert.Single(people);
                    Assert.Equal("alice", people[0].Id);

                    // Upsert semantics: re-adding updates in place, not duplicates.
                    var aliceUpdated = new GraphNode<double>("alice", "PERSON");
                    aliceUpdated.SetProperty("name", "Alice Smith");
                    await store.AddNodeAsync(aliceUpdated);
                    Assert.Equal(2, store.NodeCount);
                    Assert.Equal("Alice Smith", (await store.GetNodeAsync("alice"))!.GetProperty<string>("name"));

                    // Removals.
                    Assert.True(await store.RemoveEdgeAsync(edge.Id));
                    Assert.Equal(0, store.EdgeCount);
                    Assert.True(await store.RemoveNodeAsync("acme"));
                    Assert.False(await store.RemoveNodeAsync("does-not-exist"));
                    Assert.Equal(1, store.NodeCount);
                }
                finally
                {
                    await store!.ClearAsync();
                }
            }
        }

        [SkippableFact(Timeout = 120000)]
        [Trait("Category", "Integration")]
        public async Task AddEdge_Throws_WhenEndpointMissing()
        {
            var store = CreateStoreOrSkip(out var reason);
            Skip.If(store is null, reason);

            using (store)
            {
                try
                {
                    await store!.AddNodeAsync(new GraphNode<double>("only", "PERSON"));
                    var dangling = new GraphEdge<double>("only", "ghost", "KNOWS");
                    await Assert.ThrowsAsync<InvalidOperationException>(() => store.AddEdgeAsync(dangling));
                }
                finally
                {
                    await store!.ClearAsync();
                }
            }
        }
    }
}
#endif
