using AiDotNet.Augmentation;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 8 — ConfigureAugmentation. Verifies that a user-supplied
/// custom augmenter, wired via the new
/// <see cref="AugmentationConfig.CustomAugmenter"/> slot, is actually
/// invoked on training data during BuildAsync.
/// </summary>
/// <remarks>
/// Before the source fix in this PR the entire ConfigureAugmentation
/// surface was a no-op: <c>_augmentationConfig</c> was set by the
/// configure call, flowed through to
/// <c>AiModelResultOptions.AugmentationConfig</c>, but no consumer
/// anywhere in src/ read it. The
/// <c>ImageSettings</c> / <c>TabularSettings</c> / etc. properties on
/// AugmentationConfig still have no factory translating them into
/// IAugmentation instances (deeper follow-up work); the
/// <c>CustomAugmenter</c> slot bridges the gap for advanced users who
/// construct their own IAugmentation from the existing
/// <c>src/Augmentation/*</c> augmenter zoo.
/// </remarks>
[Collection("ConfigureMethodCoverage")]
public class Bucket8_AugmentationTests : ConfigureMethodTestBase
{
    private readonly ITestOutputHelper _output;
    public Bucket8_AugmentationTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// ConfigureAugmentation — wires a recording IAugmentation through the
    /// new CustomAugmenter slot and asserts BuildAsync invoked Apply on
    /// the training data. A stored-but-not-consumed regression would
    /// leave ApplyCalls at 0.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureAugmentation_CustomAugmenter_ActuallyInvokesApply()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var recorder = new RecordingAugmenter();
        var augCfg = new AugmentationConfig
        {
            IsEnabled = true,
            CustomAugmenter = recorder,
        };

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureAugmentation(augCfg)
            .BuildAsync();

        Assert.True(recorder.ApplyCalls > 0,
            $"ConfigureAugmentation wired a custom augmenter but BuildAsync never invoked Apply on it (calls={recorder.ApplyCalls}). Stored-but-not-consumed regression on the augmentation surface.");
    }

    /// <summary>
    /// ConfigureAugmentation with IsEnabled=false must NOT invoke Apply —
    /// the gate at AiModelBuilder.cs prevents the wiring from firing when
    /// the user explicitly disabled augmentation.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureAugmentation_Disabled_DoesNotInvokeApply()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var recorder = new RecordingAugmenter();
        var augCfg = new AugmentationConfig
        {
            IsEnabled = false,
            CustomAugmenter = recorder,
        };

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureAugmentation(augCfg)
            .BuildAsync();

        Assert.Equal(0, recorder.ApplyCalls);
    }

    /// <summary>
    /// Identity augmenter that records every Apply call. Returns the input
    /// unchanged so the model's training trajectory is undisturbed — the
    /// test screens for wiring, not for augmentation behaviour.
    /// </summary>
    private sealed class RecordingAugmenter : IAugmentation<float, Tensor<float>>
    {
        public int ApplyCalls;
        public string Name => nameof(RecordingAugmenter);
        public double Probability => 1.0;
        public bool IsTrainingOnly => true;
        public bool IsEnabled { get; set; } = true;

        public Tensor<float> Apply(Tensor<float> data, AugmentationContext<float>? context = null)
        {
            ApplyCalls++;
            return data;
        }

        public System.Collections.Generic.IDictionary<string, object> GetParameters()
            => new System.Collections.Generic.Dictionary<string, object>();
    }
}
