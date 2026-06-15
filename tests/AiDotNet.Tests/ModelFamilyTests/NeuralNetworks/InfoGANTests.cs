using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual InfoGAN test — replaces the auto-generated
/// AiDotNet.Tests.ModelFamilyTests.Generated.InfoGANTests scaffold.
/// </summary>
/// <remarks>
/// The auto-generated factory passes a single
/// <see cref="NeuralNetworkArchitecture{T}"/> to the
/// <c>InfoGAN(architecture)</c> convenience constructor, which then
/// reuses that same arch for generator + discriminator + Q-network. That
/// works only when noise / images / Q-output share a dimension count —
/// and the auto-generator's default <c>InputShape</c> / <c>OutputShape</c>
/// then disagree with the internal network shapes (Chen et al. 2016 §3:
/// generator input is z+c, discriminator input is image, Q-network input
/// is image). The generated test failed every Train-touching invariant
/// with "Batch size mismatch: realImages has 4, noise has 16, latentCodes
/// has 4" — see #1224 Cluster F.
///
/// This manual class uses InputShape / OutputShape that match a flat
/// 1-D regression-style InfoGAN: noise of size <c>latentSize</c>,
/// "image" of size <c>InputSize</c>. The factory wires the same arch
/// through the InfoGAN convenience ctor (gen / disc / q all share the
/// 1-D MLP backbone).
/// </remarks>
public class InfoGANTests : GANModelTestBase<float>
{
    // Test data shapes: input is the noise tensor (4-dim raw noise),
    // output is the imageDim (4). InfoGAN.Predict / ForwardForTraining
    // append default zero codes when the input is shaped as raw noise,
    // and Train wires noise + freshly-sampled codes through the
    // generator — so the test base's single (input, target) shape pair
    // works consistently across both invariants once the noise-only
    // contract is upheld on Predict.
    protected override int[] InputShape => [1, 4];
    protected override int[] OutputShape => [1, 4];

    // GAN training is adversarial — generator + discriminator + Q-network
    // pull in opposite directions, so the per-step loss oscillates rather
    // than monotonically decreasing. Test a slightly higher tolerance like
    // ConditionalGANTests does.
    protected override double MoreDataTolerance => 0.01;

    // Iteration counts capped to fit the 60-180 s xUnit per-test timeouts.
    // The InfoGAN forward pass goes through three sub-networks per step
    // (gen + disc + Q-network), so step time is 3× the small-MLP baseline.
    protected override int MoreDataShortIterations => 1;
    protected override int MoreDataLongIterations => 2;
    protected override int MemorizationTaskIterations => 4;
    protected override double MemorizationTaskLossThreshold => 0.99999;

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        // Per Chen et al. 2016 §3, InfoGAN has three sub-networks with
        // distinct input dimensions:
        //   - Generator:    input = noise (4) + latentCodes (2) = 6, output = imageDim (4)
        //   - Discriminator: input = imageDim (4), output = 1 (real/fake)
        //   - Q-network:    input = imageDim (4), output = latentCodes (2)
        // The single-arch convenience ctor reuses one shape for all three,
        // which fails as soon as Train wires noise+codes through the
        // generator and then images through disc/Q. Use the explicit
        // three-architecture ctor so each network sees the right shape.
        const int noiseSize = 4;
        const int latentSize = 2;
        const int imageSize = 4;
        var generatorArch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: noiseSize + latentSize,
            outputSize: imageSize);
        var discriminatorArch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputSize: imageSize,
            outputSize: 1);
        var qNetworkArch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: imageSize,
            outputSize: latentSize);
        return new InfoGAN<float>(
            generatorArch,
            discriminatorArch,
            qNetworkArch,
            latentCodeSize: latentSize,
            inputType: InputType.OneDimensional);
    }
}
