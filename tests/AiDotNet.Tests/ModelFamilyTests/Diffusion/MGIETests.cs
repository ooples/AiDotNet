using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Interfaces;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.Editing;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// MGIE (Fu et al. 2024, MLLM-guided image editing) against the diffusion family contract, at a scale the
/// 120-second budget can actually run.
/// </summary>
/// <remarks>
/// <para>
/// The generated fixture timed out and took the whole test host down with it, which is not a capacity
/// problem but a semantic one: <c>Predict</c> on a latent diffusion model is not a forward pass. It
/// forwards the input as the initial sample to <c>Generate</c>, and MGIE's <c>NumDiffusionSteps</c>
/// defaults to 50 (EditingVLMOptions), so a single "forward pass" ran fifty passes over an SD-1.5-scale
/// U-Net — <c>BASE_CHANNELS = 320</c>, multipliers <c>{1, 2, 4, 4}</c>, two residual blocks per level —
/// plus a 128-channel VAE. Predict_ShouldBeDeterministic calls it twice, which is why several MGIE tests
/// hung rather than one.
/// </para>
/// <para>
/// So the inference setting is what shrinks, not the architecture. NumDiffusionSteps drops to 1, matching
/// UpscaleAVideoModelTests' <c>numInferenceSteps: 1</c>, and the injected U-Net and VAE keep every
/// paper-defined shape that carries meaning — <c>inputChannels: 8</c> and <c>outputChannels: 4</c>
/// (the base pads the 4-channel latent up to the U-Net's 8 for the concatenated source image),
/// <c>contextDim: 768</c> (the edit head emits CROSS_ATTENTION_DIM-wide context), and the 0.18215 latent
/// scale — while base channels and level counts come down. The edit head keeps its paper depth and head
/// count. Production defaults are untouched and fully user-customizable; only this fixture is small.
/// </para>
/// <para>
/// THE CLASS NAME IS LOAD-BEARING. The generator suppresses its own emission by simple name: it builds
/// <c>StripBacktick(ClassName) + "Tests"</c> and skips generation when a test class of that name already
/// exists, which for the model <c>MGIE</c> is exactly <c>MGIETests</c>. Naming this file's class anything
/// else — <c>MGIEModelTests</c>, say — leaves the generated fixture in place, and both run: the manual one
/// passes in seconds while the generated one still drives the 50-step sampler into the 120-second gate and
/// takes the test host with it. UpscaleAVideoModelTests works for the same reason and only by coincidence:
/// its model type is <c>UpscaleAVideoModel</c>, so the generated name already matches its manual class.
/// </para>
/// </remarks>
public class MGIETests : DiffusionModelTestBase<float>
{
    // A latent-shaped input keeps Predict on the latent path: LatentDiffusionModelBase.Generate reads a
    // latent-channel shape as PyTorch's output_type='latent' and skips the VAE decode, which dotnet-trace
    // showed dominates wall clock. The decode is still exercised through EditImage's own pixel-space path.
    protected override int[] InputShape => [1, 4, 8, 8];

    protected override int[] OutputShape => [1, 4, 8, 8];

    protected override int TrainingIterations => 1;

    protected override IDiffusionModel<float> CreateModel()
        => new MGIE<float>(
            options: new MGIEOptions
            {
                // The one inference setting that made a smoke test into a fifty-step sampler.
                NumDiffusionSteps = 1,
            },
            unet: new UNetNoisePredictor<float>(
                inputChannels: 8,
                outputChannels: 4,
                baseChannels: 32,
                channelMultipliers: [1, 2],
                numResBlocks: 1,
                attentionResolutions: [1],
                contextDim: 768,
                numHeads: 8,
                inputHeight: 8,
                seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3,
                latentChannels: 4,
                baseChannels: 8,
                channelMultipliers: [1, 2],
                numResBlocksPerLevel: 1,
                latentScaleFactor: 0.18215,
                seed: 42),
            seed: 42);
}
