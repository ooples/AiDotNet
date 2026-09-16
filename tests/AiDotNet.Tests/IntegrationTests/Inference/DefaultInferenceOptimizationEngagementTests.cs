using AiDotNet.Configuration;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Inference;

/// <summary>
/// Pins the default inference-optimization contract — #1632's first checklist item, which asked
/// for a test "proving what actually engages when the builder is left alone".
///
/// #1632 catalogued the inference stack as dormant: <c>_inferenceOptimizationConfig</c> defaulted
/// to <c>null</c>, so nothing engaged unless the caller opted in with
/// <c>ConfigureInferenceOptimizations()</c>. That default has since been flipped, which is a
/// silent, easily-reverted one-line change: setting it back to <c>null</c> would return the whole
/// stack to dormant with no test failing anywhere. These tests are that guard.
///
/// <para><b>Scope, stated honestly.</b> This pins the CONFIGURATION contract — which optimizations
/// the builder resolves when untouched, and that opting out still works. It does not assert that
/// each component executes inside a forward pass; that needs per-component instrumentation and is
/// tracked separately on #1632.</para>
/// </summary>
public class DefaultInferenceOptimizationEngagementTests
{
    private static InferenceOptimizationConfig? ResolveDefault()
    {
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>();
        var view = (IConfiguredView<double, Tensor<double>, Tensor<double>>)builder;
        return view.ConfiguredInferenceOptimizations;
    }

    [Fact]
    public void UntouchedBuilder_ResolvesAnInferenceConfig_RatherThanLeavingTheStackDormant()
    {
        // The regression this guards: AiModelBuilder.cs previously left this null, so every model
        // ran with the entire built-and-verified inference stack switched off.
        Assert.NotNull(ResolveDefault());
    }

    [Fact]
    public void UntouchedBuilder_EngagesTheDocumentedOptimizationSet()
    {
        var config = ResolveDefault();
        Assert.NotNull(config);

        // What IS on by default.
        Assert.True(config!.EnableKVCache, "KV cache should be on by default.");
        Assert.True(config.EnablePagedKVCache, "Paged KV cache should be on by default.");
        Assert.True(config.EnableFlashAttention, "Flash attention should be on by default.");
        Assert.True(config.EnableLayerFusion, "Layer fusion should be on by default.");
        Assert.True(config.EnableBatching, "Batching should be on by default.");

        // What is deliberately NOT on by default. Speculative decoding needs a draft model, so
        // enabling it implicitly would either fail or silently do nothing. Asserting the negative
        // keeps the documented default honest — #1632 assumed the whole set was uniformly off,
        // and the truth is that this one component alone stays opt-in.
        Assert.NotNull(config.SpeculativeDecoding);
        Assert.False(config.SpeculativeDecoding!.Enabled,
            "Speculative decoding requires a caller-supplied draft model and stays opt-in.");
    }

    [Fact]
    public void ExplicitConfiguration_StillOverridesTheDefault()
    {
        // Control arm. If the default were pinned but no longer overridable, the two assertions
        // above could pass while ConfigureInferenceOptimizations had become a no-op — which is the
        // exact bug class AIDN090/091 was added for (a Configure* that stores and never reads).
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureInferenceOptimizations(new InferenceOptimizationConfig
            {
                EnableKVCache = false,
                EnablePagedKVCache = false,
                EnableFlashAttention = false,
                EnableLayerFusion = false,
                EnableBatching = false,
            });

        var view = (IConfiguredView<double, Tensor<double>, Tensor<double>>)builder;
        var config = view.ConfiguredInferenceOptimizations;

        Assert.NotNull(config);
        Assert.False(config!.EnableKVCache);
        Assert.False(config.EnablePagedKVCache);
        Assert.False(config.EnableFlashAttention);
        Assert.False(config.EnableLayerFusion);
        Assert.False(config.EnableBatching);
    }
}
