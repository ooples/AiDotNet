using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Xunit;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Graph;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Verifies the composite <c>retrievalAugmentedGeneration:</c> YAML section wires the FULL RAG pipeline
/// through the source-generated applier: retriever, reranker, generator, query processors, document
/// store, graph store, embedding model, similarity metric, and knowledge-graph options are all applied
/// to the builder rather than only the retriever being kept and everything else dropped.
/// </summary>
public class RagYamlCompositeTests
{
    private static IReadOnlyDictionary<string, Dictionary<string, Type>> Registries =>
        YamlTypeRegistry<double, Matrix<double>, Vector<double>>.GetAllRegistries();

    /// <summary>
    /// Finds the first registered type name in <paramref name="section"/> that both instantiates
    /// without throwing and is assignable to <paramref name="expected"/>. Keeps the test robust as the
    /// concrete implementation set evolves, and avoids heavy/failing constructors.
    /// </summary>
    private static string? PickInstantiable(string section, Type expected, int maxAttempts = 25)
    {
        if (!Registries.TryGetValue(section, out var types)) return null;

        var attempts = 0;
        foreach (var name in types.Keys)
        {
            if (attempts++ >= maxAttempts) break;
            try
            {
                var instance = YamlTypeRegistry<double, Matrix<double>, Vector<double>>
                    .CreateInstance<object>(section, name);
                if (instance is not null && expected.IsInstanceOfType(instance))
                {
                    return name;
                }
            }
            catch
            {
                // Skip types whose constructors need resources we can't supply in a unit test.
            }
        }

        return null;
    }

    private static object? GetPrivateField(object target, string fieldName)
    {
        var field = target.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.True(field is not null, $"Expected private field '{fieldName}' on {target.GetType().Name}.");
        return field!.GetValue(target);
    }

    /// <summary>
    /// Parsing a composite RAG section must retain every sub-component (nothing dropped at deserialize).
    /// </summary>
    [Fact]
    public void LoadFromString_CompositeRagSection_RetainsAllSubComponents()
    {
        var yaml = @"
retrievalAugmentedGeneration:
  retriever:
    type: BM25Retriever
  reranker:
    type: IdentityReranker
  generator:
    type: StubGenerator
  queryProcessors:
    - type: IdentityQueryProcessor
  documentStore:
    type: InMemoryDocumentStore
  graphStore:
    type: MemoryGraphStore
  embeddingModel:
    type: Word2Vec
  similarityMetric:
    type: CosineSimilarityMetric
  knowledgeGraph:
    params:
      trainEmbeddings: true
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.RetrievalAugmentedGeneration);
        var rag = config.RetrievalAugmentedGeneration!;
        Assert.Equal("BM25Retriever", rag.Retriever!.Type);
        Assert.Equal("IdentityReranker", rag.Reranker!.Type);
        Assert.Equal("StubGenerator", rag.Generator!.Type);
        Assert.Single(rag.QueryProcessors);
        Assert.Equal("IdentityQueryProcessor", rag.QueryProcessors[0].Type);
        Assert.Equal("InMemoryDocumentStore", rag.DocumentStore!.Type);
        Assert.Equal("MemoryGraphStore", rag.GraphStore!.Type);
        Assert.Equal("Word2Vec", rag.EmbeddingModel!.Type);
        Assert.Equal("CosineSimilarityMetric", rag.SimilarityMetric!.Type);
        Assert.NotNull(rag.KnowledgeGraph);
        Assert.True(rag.KnowledgeGraph!.Params.ContainsKey("trainEmbeddings"));
    }

    /// <summary>
    /// The generated <c>YamlRagSection</c> exposes a property for every RAG sub-component — proof the
    /// generator produced a composite section rather than a single-type retriever-only section.
    /// </summary>
    [Fact]
    public void YamlModelConfig_RetrievalAugmentedGeneration_IsCompositeSection()
    {
        var prop = typeof(YamlModelConfig).GetProperty("RetrievalAugmentedGeneration");
        Assert.NotNull(prop);
        Assert.Equal("YamlRagSection", prop!.PropertyType.Name);

        var ragType = prop.PropertyType;
        foreach (var expectedMember in new[]
                 {
                     "Retriever", "Reranker", "Generator", "QueryProcessors",
                     "DocumentStore", "GraphStore", "EmbeddingModel", "SimilarityMetric", "KnowledgeGraph",
                 })
        {
            Assert.True(ragType.GetProperty(expectedMember) is not null,
                $"YamlRagSection is missing composite sub-component '{expectedMember}'.");
        }
    }

    /// <summary>
    /// Applying the composite section wires EVERY sub-component to the builder — retriever, reranker,
    /// generator, query processors, graph store (+ derived knowledge graph and hybrid retriever),
    /// embedding model, similarity metric — and calls ConfigureKnowledgeGraph. Type names are chosen
    /// dynamically from the registry so the test tracks the real implementation set.
    /// </summary>
    [Fact]
    public void Apply_CompositeRagSection_WiresAllSubComponentsOntoBuilder()
    {
        var retriever = PickInstantiable("Retriever", typeof(IRetriever<double>));
        var reranker = PickInstantiable("Reranker", typeof(IReranker<double>));
        var generator = PickInstantiable("Generator", typeof(IGenerator<double>));
        var queryProcessor = PickInstantiable("QueryProcessor", typeof(IQueryProcessor));
        var graphStore = PickInstantiable("GraphStore", typeof(IGraphStore<double>));
        var documentStore = PickInstantiable("DocumentStore", typeof(IDocumentStore<double>));
        var embeddingModel = PickInstantiable("EmbeddingModel", typeof(IEmbeddingModel<double>));
        var similarityMetric = PickInstantiable(
            "SimilarityMetric",
            typeof(AiDotNet.RetrievalAugmentedGeneration.VectorSearch.ISimilarityMetric<double>));

        // The components that prove the OLD "keep only the retriever, drop everything else" bug is fixed
        // must be registry-constructible. (Retrievers require an injected IDocumentStore constructor arg
        // that the registry cannot synthesize, so retriever wiring is exercised opportunistically only —
        // and retriever was never the dropped component; it was the sole survivor of the old bug.)
        Assert.NotNull(reranker);
        Assert.NotNull(generator);
        Assert.NotNull(queryProcessor);
        Assert.NotNull(graphStore);
        Assert.NotNull(documentStore);
        Assert.NotNull(similarityMetric);

        var yaml = new StringBuilder();
        yaml.AppendLine("retrievalAugmentedGeneration:");
        if (retriever is not null)
        {
            yaml.AppendLine("  retriever:");
            yaml.AppendLine($"    type: {retriever}");
        }
        yaml.AppendLine("  reranker:");
        yaml.AppendLine($"    type: {reranker}");
        yaml.AppendLine("  generator:");
        yaml.AppendLine($"    type: {generator}");
        yaml.AppendLine("  queryProcessors:");
        yaml.AppendLine($"    - type: {queryProcessor}");
        yaml.AppendLine("  graphStore:");
        yaml.AppendLine($"    type: {graphStore}");
        yaml.AppendLine("  documentStore:");
        yaml.AppendLine($"    type: {documentStore}");
        if (embeddingModel is not null)
        {
            yaml.AppendLine("  embeddingModel:");
            yaml.AppendLine($"    type: {embeddingModel}");
        }
        yaml.AppendLine("  similarityMetric:");
        yaml.AppendLine($"    type: {similarityMetric}");
        yaml.AppendLine("  knowledgeGraph:");
        yaml.AppendLine("    params:");
        yaml.AppendLine("      trainEmbeddings: true");

        var config = YamlConfigLoader.LoadFromString(yaml.ToString());
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        var exception = Record.Exception(() =>
            YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder));
        Assert.Null(exception);

        // Standard RAG components that were previously dropped are now wired.
        Assert.NotNull(GetPrivateField(builder, "_ragReranker"));
        Assert.NotNull(GetPrivateField(builder, "_ragGenerator"));

        var queryProcessors = GetPrivateField(builder, "_queryProcessors") as IEnumerable;
        Assert.NotNull(queryProcessors);
        Assert.True(queryProcessors!.Cast<object>().Any(), "_queryProcessors should contain at least one processor.");

        // Graph RAG components derived from graphStore + documentStore (documentStore was previously dropped).
        Assert.NotNull(GetPrivateField(builder, "_graphStore"));
        Assert.NotNull(GetPrivateField(builder, "_knowledgeGraph"));
        Assert.NotNull(GetPrivateField(builder, "_hybridGraphRetriever"));

        // Similarity metric wired via ConfigureSimilarityMetric (previously dropped).
        Assert.NotNull(GetPrivateField(builder, "_configuredSimilarityMetric"));

        // Retriever wired when a registry-constructible implementation was available.
        if (retriever is not null)
        {
            Assert.NotNull(GetPrivateField(builder, "_ragRetriever"));
        }

        // Embedding model wired via ConfigureEmbeddingModel (when a lightweight impl was resolvable).
        if (embeddingModel is not null)
        {
            Assert.NotNull(GetPrivateField(builder, "_configuredEmbeddingModel"));
        }

        // ConfigureKnowledgeGraph applied with the supplied params (previously a TODO / dropped).
        var kgOptions = GetPrivateField(builder, "_knowledgeGraphOptions") as KnowledgeGraphOptions;
        Assert.NotNull(kgOptions);
        Assert.True(kgOptions!.TrainEmbeddings == true,
            "ConfigureKnowledgeGraph should have applied trainEmbeddings=true from the composite section.");
    }

    /// <summary>
    /// The registry must expose a section for every RAG sub-component the composite applier resolves,
    /// including the newly-marked QueryProcessor section. Without these the composite wiring could not
    /// instantiate the named types.
    /// </summary>
    [Fact]
    public void TypeRegistry_HasSectionsForAllRagSubComponents()
    {
        foreach (var section in new[]
                 {
                     "Retriever", "Reranker", "Generator", "QueryProcessor",
                     "DocumentStore", "GraphStore", "EmbeddingModel", "SimilarityMetric",
                 })
        {
            Assert.True(Registries.ContainsKey(section), $"Type registry is missing RAG section '{section}'.");
            Assert.True(Registries[section].Count > 0, $"RAG section '{section}' has zero implementations.");
        }
    }
}
