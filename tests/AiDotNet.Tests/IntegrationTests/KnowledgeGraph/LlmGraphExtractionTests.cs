using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Construction;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.KnowledgeGraph;

/// <summary>
/// Tests for the LLM-based (Microsoft GraphRAG parity) entity/relation extraction and
/// community-report paths in <see cref="KGConstructor{T}"/> and <see cref="CommunitySummarizer{T}"/>.
/// All tests are CI-runnable and use a scripted fake generator (no network).
/// </summary>
public class LlmGraphExtractionTests
{
    /// <summary>
    /// A fake <see cref="IGenerator{T}"/> that returns scripted text and records the prompts it saw.
    /// </summary>
    private sealed class ScriptedGenerator : IGenerator<double>
    {

        public System.Threading.Tasks.Task<string> GenerateAsync(string prompt, System.Threading.CancellationToken cancellationToken = default) { cancellationToken.ThrowIfCancellationRequested(); return System.Threading.Tasks.Task.FromResult(Generate(prompt)); }
        public System.Threading.Tasks.Task<GroundedAnswer<double>> GenerateGroundedAsync(string query, IEnumerable<Document<double>> context, System.Threading.CancellationToken cancellationToken = default) { cancellationToken.ThrowIfCancellationRequested(); return System.Threading.Tasks.Task.FromResult(GenerateGrounded(query, context)); }
        private readonly string _response;
        public List<string> Prompts { get; } = new List<string>();

        public int MaxContextTokens => 8192;
        public int MaxGenerationTokens => 1024;

        public ScriptedGenerator(string response)
        {
            _response = response;
        }

        public string Generate(string prompt)
        {
            Prompts.Add(prompt);
            return _response;
        }

        public GroundedAnswer<double> GenerateGrounded(string query, IEnumerable<Document<double>> context)
        {
            return new GroundedAnswer<double>
            {
                Query = query,
                Answer = _response,
                SourceDocuments = context?.ToList() ?? new List<Document<double>>(),
                Citations = new List<string>(),
                ConfidenceScore = 1.0
            };
        }
    }

    private const string ValidExtractionJson = @"{
        ""entities"": [
            { ""name"": ""Albert Einstein"", ""type"": ""PERSON"", ""description"": ""Theoretical physicist."" },
            { ""name"": ""Princeton University"", ""type"": ""ORGANIZATION"", ""description"": ""University in New Jersey."" },
            { ""name"": ""Ulm"", ""type"": ""LOCATION"", ""description"": ""City in Germany."" }
        ],
        ""relations"": [
            { ""source"": ""Albert Einstein"", ""relation"": ""WORKS_AT"", ""target"": ""Princeton University"", ""description"": ""Einstein worked at Princeton."" },
            { ""source"": ""Albert Einstein"", ""relation"": ""BORN_IN"", ""target"": ""Ulm"", ""description"": ""Einstein was born in Ulm."" }
        ]
    }";

    [Fact(Timeout = 120000)]
    public async Task KGConstructor_WithLlmGenerator_PopulatesGraphFromJson()
    {
        var generator = new ScriptedGenerator(ValidExtractionJson);
        var constructor = new KGConstructor<double>(generator);

        // Text content is deliberately terse; the LLM (scripted) supplies the structure.
        var graph = constructor.ConstructFromText("Einstein biography.");

        // (a) LLM-extracted entities populate the graph.
        Assert.True(generator.Prompts.Count > 0, "LLM generator should have been invoked.");
        var nodeIds = graph.GetAllNodes().Select(n => n.Id).ToList();
        Assert.Contains("albert_einstein", nodeIds);
        Assert.Contains("princeton_university", nodeIds);
        Assert.Contains("ulm", nodeIds);

        var einstein = graph.GetNode("albert_einstein");
        Assert.NotNull(einstein);
        Assert.Equal("PERSON", einstein!.Label);
        Assert.Equal("Theoretical physicist.", einstein.GetProperty<string>("description"));

        // (a) LLM-extracted relations populate the graph.
        var edges = graph.GetAllEdges().ToList();
        Assert.Contains(edges, e => e.RelationType == "WORKS_AT" &&
                                    e.SourceId == "albert_einstein" &&
                                    e.TargetId == "princeton_university");
        Assert.Contains(edges, e => e.RelationType == "BORN_IN");
    }

    [Fact(Timeout = 120000)]
    public async Task KGConstructor_MalformedJson_DegradesToRegexFallback()
    {
        // Not valid JSON — the LLM path must transparently fall back to the regex extractor.
        var generator = new ScriptedGenerator("Sorry, I cannot help with that. {this is : not, valid json");
        var constructor = new KGConstructor<double>(generator);

        var graph = constructor.ConstructFromText(
            "Albert Einstein was born in Ulm, Germany. He worked at Princeton University.");

        Assert.True(generator.Prompts.Count > 0, "LLM generator should still have been invoked.");

        // (b) Regex fallback still populates the graph with capitalized entities.
        var nodeIds = graph.GetAllNodes().Select(n => n.Id).ToList();
        Assert.NotEmpty(nodeIds);
        Assert.Contains(nodeIds, id => id.Contains("einstein") || id.Contains("albert"));
        Assert.Contains(nodeIds, id => id.Contains("princeton"));
    }

    [Fact(Timeout = 120000)]
    public async Task KGConstructor_JsonWrappedInMarkdownFences_IsParsed()
    {
        var fenced = "Here is the extraction:\n```json\n" + ValidExtractionJson + "\n```\nDone.";
        var generator = new ScriptedGenerator(fenced);
        var constructor = new KGConstructor<double>(generator);

        var graph = constructor.ConstructFromText("Einstein biography.");

        var nodeIds = graph.GetAllNodes().Select(n => n.Id).ToList();
        Assert.Contains("albert_einstein", nodeIds);
        Assert.Contains("princeton_university", nodeIds);
    }

    [Fact(Timeout = 120000)]
    public async Task KGConstructor_NoGenerator_UsesRegexFallback()
    {
        var constructor = new KGConstructor<double>(); // no generator injected
        var graph = constructor.ConstructFromText(
            "Albert Einstein was born in Germany. Marie Curie worked at the University of Paris.");

        Assert.NotEmpty(graph.GetAllNodes());
        Assert.NotEmpty(graph.GetAllEdges());
    }

    [Fact(Timeout = 120000)]
    public async Task KGConstructor_UseLlmExtractionDisabled_UsesRegexEvenWithGenerator()
    {
        // A generator that would throw if actually used — proves the regex path is taken.
        var generator = new ScriptedGenerator(ValidExtractionJson);
        var constructor = new KGConstructor<double>(generator);
        var opts = new KGConstructionOptions { UseLlmExtraction = false };

        var graph = constructor.ConstructFromText(
            "Albert Einstein was born in Ulm, Germany.", options: opts);

        Assert.Empty(generator.Prompts); // LLM never called
        Assert.NotEmpty(graph.GetAllNodes());
    }

    [Fact(Timeout = 120000)]
    public async Task KGConstructor_WithClaimsExtraction_AttachesClaimsToNodes()
    {
        const string jsonWithClaims = @"{
            ""entities"": [
                { ""name"": ""Acme Corp"", ""type"": ""ORGANIZATION"", ""description"": ""A company."" }
            ],
            ""relations"": [],
            ""claims"": [
                { ""subject"": ""Acme Corp"", ""object"": ""NONE"", ""type"": ""REGULATORY_VIOLATION"", ""description"": ""Fined for pollution."", ""status"": ""TRUE"" }
            ]
        }";

        var generator = new ScriptedGenerator(jsonWithClaims);
        var constructor = new KGConstructor<double>(generator);
        var opts = new KGConstructionOptions { ExtractClaims = true };

        var graph = constructor.ConstructFromText("Acme Corp news.", options: opts);

        var node = graph.GetNode("acme_corp");
        Assert.NotNull(node);
        var claims = node!.GetProperty<List<string>>("claims");
        Assert.NotNull(claims);
        Assert.Contains(claims!, c => c.Contains("REGULATORY_VIOLATION") && c.Contains("pollution"));
    }

    [Fact(Timeout = 120000)]
    public async Task CommunitySummarizer_WithGenerator_UsesLlmReport()
    {
        var graph = BuildPhysicsGraph();
        var partition = new Dictionary<string, int>
        {
            ["einstein"] = 0,
            ["bohr"] = 0,
            ["physics"] = 0
        };

        const string report = "REPORT_SENTINEL: This community centers on physics pioneers Einstein and Bohr.";
        var summarizer = new CommunitySummarizer<double>(new ScriptedGenerator(report));

        var summaries = summarizer.SummarizePartition(graph, partition, level: 0);

        // (c) LLM community report is used when a generator is present.
        Assert.NotEmpty(summaries);
        Assert.All(summaries, s => Assert.Contains("REPORT_SENTINEL", s.Description));
    }

    [Fact(Timeout = 120000)]
    public async Task CommunitySummarizer_NoGenerator_UsesExtractiveFallback()
    {
        var graph = BuildPhysicsGraph();
        var partition = new Dictionary<string, int>
        {
            ["einstein"] = 0,
            ["bohr"] = 0,
            ["physics"] = 0
        };

        var summarizer = new CommunitySummarizer<double>(); // no generator
        var summaries = summarizer.SummarizePartition(graph, partition, level: 0);

        Assert.NotEmpty(summaries);
        Assert.All(summaries, s => Assert.Contains("Community of", s.Description));
    }

    [Fact(Timeout = 120000)]
    public async Task CommunitySummarizer_EmptyLlmResponse_FallsBackToExtractive()
    {
        var graph = BuildPhysicsGraph();
        var partition = new Dictionary<string, int>
        {
            ["einstein"] = 0,
            ["bohr"] = 0,
            ["physics"] = 0
        };

        var summarizer = new CommunitySummarizer<double>(new ScriptedGenerator("   "));
        var summaries = summarizer.SummarizePartition(graph, partition, level: 0);

        Assert.NotEmpty(summaries);
        Assert.All(summaries, s => Assert.Contains("Community of", s.Description));
    }

    private static KnowledgeGraph<double> BuildPhysicsGraph()
    {
        var graph = new KnowledgeGraph<double>();

        var einstein = new GraphNode<double>("einstein", "PERSON");
        einstein.SetProperty("name", "Albert Einstein");
        var bohr = new GraphNode<double>("bohr", "PERSON");
        bohr.SetProperty("name", "Niels Bohr");
        var physics = new GraphNode<double>("physics", "CONCEPT");
        physics.SetProperty("name", "Physics");

        graph.AddNode(einstein);
        graph.AddNode(bohr);
        graph.AddNode(physics);

        graph.AddEdge(new GraphEdge<double>("einstein", "bohr", "COLLABORATED_WITH", 0.9));
        graph.AddEdge(new GraphEdge<double>("einstein", "physics", "CONTRIBUTED_TO", 0.9));
        graph.AddEdge(new GraphEdge<double>("bohr", "physics", "CONTRIBUTED_TO", 0.9));

        return graph;
    }
}
