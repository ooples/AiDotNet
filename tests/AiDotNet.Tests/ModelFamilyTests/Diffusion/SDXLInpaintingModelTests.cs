using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// SDXLInpainting test scaffold — inherits from the generic
/// <see cref="DiffusionModelTestBase{TNum}"/> with <c>TNum = float</c> instead
/// of the default-double <see cref="DiffusionModelTestBase"/> shim because
/// SDXLInpainting at paper-scale defaults (UNet baseChannels=320,
/// channelMultipliers=[1,2,4], contextDim=2048, ≈2.6 B parameters; VAE
/// baseChannels=128 channelMultipliers=[1,2,4,4]) is too large for FP64 on a
/// 16 GB CI host — <c>SDXLInpaintingModel&lt;double&gt;</c> OOMs in the 1.28 B-element
/// kernel allocation during <c>UNetNoisePredictor.CreateDefaultEncoderBlocks</c>
/// (verified via testconsole/SdxlMemProfile). The same model at FP32 fits in
/// ≈3.8 GB managed heap and completes one Predict() in ≈38 s, well within the
/// 120 s per-test timeout.
/// </summary>
/// <remarks>
/// FP32 is also the production-canonical numeric type for diffusion-model
/// weights — SD / SDXL / Flux / SD3 paper checkpoints are FP32 master / FP16
/// working precision. Testing against <see cref="IDiffusionModel{T}"/> with
/// <c>T = float</c> therefore mirrors the actual deployment configuration
/// rather than an FP64 test-only path whose memory cost and numerics would
/// silently diverge from any real SDXL pipeline. Per the
/// <see cref="AiDotNet.AiModelBuilder{T, TInput, TOutput}.ConfigureMixedPrecision"/>
/// contract, mixed-precision training is float-only too (the facade rejects
/// <c>T = double</c> at configure-time), so this is also the only numeric
/// type compatible with the documented production training path.
/// </remarks>
public class SDXLInpaintingModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion: 4 channels, 64x64 latent (512x512 images / 8x VAE).
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new SDXLInpaintingModel<float>(seed: 42);
}
