using AiDotNet.ProgramSynthesis.Options;
using AiDotNet.ProgramSynthesis.Serving;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 13 — ConfigureProgramSynthesis + ConfigureProgramSynthesisServing.
/// Each test verifies the configured value reaches the matching
/// internal property on the post-build <see cref="AiModelResult{T, TInput, TOutput}"/>.
/// </summary>
[Collection("ConfigureMethodCoverage")]
public class Bucket13_ProgramSynthesisTests : ConfigureMethodTestBase
{
    private readonly ITestOutputHelper _output;
    public Bucket13_ProgramSynthesisTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// ConfigureProgramSynthesis — verifies the call constructs a
    /// <see cref="AiDotNet.NeuralNetworks.ProgramSynthesisModel{T}"/>
    /// from the supplied options and stores it on the builder so the
    /// inference-only build path at <c>AiModelBuilder.cs:1522</c>
    /// dispatches to <c>BuildProgramSynthesisInferenceOnlyResult</c>.
    /// Stored-but-not-consumed would leave the result's
    /// <c>ProgramSynthesisModel</c> property null.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureProgramSynthesis_DefaultOptions_LandsOnResult()
    {
        var model = MakeCanaryModel();

        // ConfigureProgramSynthesis defaults VocabularySize to 50000 to
        // satisfy the default tokenizer's vocab-size invariant — we just
        // pass an empty options and let defaults apply.
        var psOptions = new ProgramSynthesisOptions
        {
            MaxSequenceLength = 32,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
        };

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureProgramSynthesis(psOptions)
            .BuildAsync();

        Assert.NotNull(result.ProgramSynthesisModel);
    }

    /// <summary>
    /// ConfigureProgramSynthesisServing — verifies the configured serving
    /// client options reach
    /// <c>result.ProgramSynthesisServingClientOptions</c>. Uses a
    /// custom <c>BaseAddress</c> as the sentinel to distinguish from
    /// the default <c>http://localhost:52432/</c> the no-args overload
    /// installs.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureProgramSynthesisServing_CustomOptions_LandsOnResult()
    {
        var model = MakeCanaryModel();
        var psOptions = new ProgramSynthesisOptions { MaxSequenceLength = 32, NumEncoderLayers = 1, NumDecoderLayers = 1 };

        var sentinelEndpoint = new System.Uri("http://program-synthesis-sentinel.local:9999/");
        var servingOptions = new ProgramSynthesisServingClientOptions
        {
            BaseAddress = sentinelEndpoint,
        };

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureProgramSynthesis(psOptions)
            .ConfigureProgramSynthesisServing(servingOptions)
            .BuildAsync();

        Assert.NotNull(result.ProgramSynthesisServingClientOptions);
        Assert.Equal(sentinelEndpoint, result.ProgramSynthesisServingClientOptions!.BaseAddress);
    }

    /// <summary>
    /// ConfigureProgramSynthesisServing with a pre-constructed client —
    /// asserts the EXACT instance flows through to the result without
    /// being re-instantiated.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureProgramSynthesisServing_PreBuiltClient_LandsOnResultUnchanged()
    {
        var model = MakeCanaryModel();
        var psOptions = new ProgramSynthesisOptions { MaxSequenceLength = 32, NumEncoderLayers = 1, NumDecoderLayers = 1 };

        var customClient = new ProgramSynthesisServingClient(new ProgramSynthesisServingClientOptions
        {
            BaseAddress = new System.Uri("http://test-sentinel.local:1/"),
        });

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureProgramSynthesis(psOptions)
            .ConfigureProgramSynthesisServing(client: customClient)
            .BuildAsync();

        Assert.Same(customClient, result.ProgramSynthesisServingClient);
    }
}
