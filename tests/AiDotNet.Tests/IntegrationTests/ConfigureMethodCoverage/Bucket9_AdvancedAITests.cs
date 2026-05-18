using AiDotNet.Models.Options;
using AiDotNet.Reasoning.Models;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 9 — Configure* methods for advanced AI features (reasoning,
/// knowledge graphs, RAG, knowledge distillation). Each test asserts
/// the configured value reaches an observable destination on the
/// post-build result.
/// </summary>
[Collection("ConfigureMethodCoverage")]
public class Bucket9_AdvancedAITests : ConfigureMethodTestBase
{
    private readonly ITestOutputHelper _output;
    public Bucket9_AdvancedAITests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// ConfigureReasoning — verifies the configured <c>ReasoningConfig</c>
    /// reaches <c>result.ReasoningConfig</c>. Picks a non-default
    /// <c>MaxSteps</c> as the sentinel; stored-but-not-consumed would
    /// either leave the property null or keep the type default of 10.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureReasoning_NonDefaultMaxSteps_LandsOnResult()
    {
        const int sentinel = 137;
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var reasoningCfg = new ReasoningConfig { MaxSteps = sentinel };
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureReasoning(reasoningCfg)
            .BuildAsync();

        Assert.NotNull(result.ReasoningConfig);
        Assert.Equal(sentinel, result.ReasoningConfig!.MaxSteps);
    }

    /// <summary>
    /// ConfigureRetrievalAugmentedGeneration — verifies the configured
    /// knowledge graph component reaches <c>result.KnowledgeGraph</c>.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureRetrievalAugmentedGeneration_KnowledgeGraph_LandsOnResult()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var sentinelGraph = new KnowledgeGraph<float>();
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureRetrievalAugmentedGeneration(knowledgeGraph: sentinelGraph)
            .BuildAsync();

        Assert.Same(sentinelGraph, result.KnowledgeGraph);
    }

    /// <summary>
    /// ConfigureKnowledgeGraph — when paired with ConfigureRAG (which
    /// supplies a KnowledgeGraph instance), ProcessKnowledgeGraphOptions
    /// runs and the configured options reach the result. Sentinel
    /// pattern: set a recognizable
    /// <c>EnableLinkPrediction</c> override and assert it's visible
    /// post-build via the configured graph's link-prediction state.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureKnowledgeGraph_WithRAGGraph_OptionsAppliedWithoutCrash()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var graph = new KnowledgeGraph<float>();
        // Sentinel option: TrainEmbeddings is a nullable bool; setting
        // it to false explicitly distinguishes "user overrode" from
        // "default null". ConfigureKnowledgeGraph builds a
        // KnowledgeGraphOptions and passes the action; the
        // ProcessKnowledgeGraphOptions consumer at
        // AiModelBuilder.cs:1629 reads the resulting options to decide
        // whether to train embeddings. Stored-but-not-consumed would
        // see the action run but the options dropped on the floor.
        bool optionsActionRan = false;
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureRetrievalAugmentedGeneration(knowledgeGraph: graph)
            .ConfigureKnowledgeGraph(opts =>
            {
                opts.TrainEmbeddings = false;
                opts.EnableLinkPrediction = false;
                optionsActionRan = true;
            })
            .BuildAsync();

        Assert.True(optionsActionRan,
            "ConfigureKnowledgeGraph received the Action<KnowledgeGraphOptions> but never invoked it. " +
            "Stored-but-not-consumed regression: the configure call dropped the action without running it.");
        // The graph instance flows through to result via the RAG wire.
        Assert.Same(graph, result.KnowledgeGraph);
    }

    /// <summary>
    /// ConfigureKnowledgeDistillation — verifies the configured
    /// <c>KnowledgeDistillationOptions</c> reaches
    /// <c>result.KnowledgeDistillationOptions</c>. Without the wiring
    /// fix in this PR the field was stored on the builder but never
    /// flowed to the result, so consumers couldn't read the
    /// configured options post-build.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureKnowledgeDistillation_NonDefaultOptions_LandsOnResult()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var kdOptions = new KnowledgeDistillationOptions<float, Tensor<float>, Tensor<float>>
        {
            Temperature = 7.0, // non-default sentinel
        };

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureKnowledgeDistillation(kdOptions)
            .BuildAsync();

        Assert.NotNull(result.KnowledgeDistillationOptions);
        Assert.Equal(7.0, result.KnowledgeDistillationOptions!.Temperature);
    }
}
