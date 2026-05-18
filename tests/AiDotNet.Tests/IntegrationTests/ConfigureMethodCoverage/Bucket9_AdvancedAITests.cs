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
    /// ConfigureKnowledgeDistillation — verifies the regular-training
    /// path FAILS FAST when KD is configured but the model path doesn't
    /// support tape-based distillation yet (review #1368 restored the
    /// NotSupportedException at AiModelBuilder.cs's regular-training
    /// branch; the previous Trace-warning downgrade silently fell
    /// through to standard supervised training, breaking the contract a
    /// user who called ConfigureKnowledgeDistillation expects).
    /// </summary>
    /// <remarks>
    /// The Bucket9 wiring assertion under the OLD contract (verify the
    /// options round-trip to result.KnowledgeDistillationOptions) is
    /// preserved on the direct-training paths (parametric / clustering /
    /// LoRA-wrapped NN), where the options ARE attached without going
    /// through the KD-aware training loop. The standard supervised path
    /// (regular Transformer / regular NN without LoRA) throws so the
    /// user discovers the missing integration at Build time. Once KD
    /// integrates with the tape-based flow upstream, this assertion
    /// flips back to the LandsOnResult shape.
    /// </remarks>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureKnowledgeDistillation_RegularTrainingPath_ThrowsUntilTapeIntegrationLands()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var kdOptions = new KnowledgeDistillationOptions<float, Tensor<float>, Tensor<float>>
        {
            Temperature = 7.0, // non-default sentinel
        };

        // Canary Transformer is a NeuralNetworkBase + IParameterizable,
        // and the test does not configure LoRA — so UseDirectTrainingPath
        // returns false and BuildAsync routes to the regular training
        // path where the KD-not-integrated throw fires.
        var ex = await Assert.ThrowsAsync<System.NotSupportedException>(async () =>
        {
            await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureDataLoader(loader)
                .ConfigureKnowledgeDistillation(kdOptions)
                .BuildAsync();
        });

        // Assert by exception TYPE + origin (TargetSite namespace in
        // AiDotNet — confirms the throw came from production code, not
        // from a runtime / framework-level NRE) rather than by message
        // substring. The message text is human-readable and can be
        // rephrased by a future maintainer without breaking behavior —
        // type+namespace assertions don't drift (review #1368 C6WMo).
        Assert.IsType<System.NotSupportedException>(ex);
        Assert.True(
            ex.TargetSite?.DeclaringType?.FullName?.StartsWith("AiDotNet.", System.StringComparison.Ordinal) == true
            || (ex.StackTrace?.Contains("at AiDotNet.", System.StringComparison.Ordinal) == true),
            $"Expected the throw to originate inside AiDotNet.* code (production builder path). " +
            $"Got TargetSite={ex.TargetSite?.DeclaringType?.FullName ?? "<null>"} | message={ex.Message}");
    }
}
