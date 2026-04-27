using AiDotNet.ComputerVision.Segmentation.Medical;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Segmentation;

/// <summary>
/// Manual factory for <see cref="SwinUNETR{T}"/>. The auto-generator emits a
/// vision-default <c>OutputShape => new[] {{ 4 }}</c> that does not match the
/// per-pixel segmentation logits this model produces, so the inherited
/// <c>Training_*</c> tests fail with a "target rank must be exactly one less
/// than predicted rank" loss-shape mismatch before any gradient flows.
/// </summary>
/// <remarks>
/// Per Hatamizadeh et al. 2022 ("Swin UNETR: Swin Transformers for Semantic
/// Segmentation of Brain Tumors in MRI Images"), the BTCV multi-organ
/// configuration uses 14 segmentation classes and the network outputs a
/// per-pixel class logit map at the input resolution. The 2D variant of the
/// architecture (used here for smoke-test speed) takes <c>[C, H, W]</c>
/// with H, W divisible by 32 — the encoder downsamples 32× across its four
/// stages (initial 4× patch projection plus three 2× patch merges), and
/// the decoder restores resolution via five 2× upsampling stages. We use
/// 64×64 spatial dims (the smallest multiple of 32 that exercises every
/// upsampling stage) and 1 input channel (BTCV's CT modality).
/// </remarks>
public class SwinUNETRTests : SegmentationTestBase
{
    private const int NumClasses = 14;
    private const int Height = 64;
    private const int Width = 64;
    private const int InputChannels = 1;

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: Height,
            inputWidth: Width,
            inputDepth: InputChannels,
            outputSize: NumClasses);
        return new SwinUNETR<double>(arch, numClasses: NumClasses);
    }

    protected override int[] InputShape => new[] { InputChannels, Height, Width };

    /// <summary>
    /// One-hot target shape: predicted is <c>[B, NumClasses, H, W]</c> and
    /// <c>CrossEntropyLoss</c> requires the target to share that rank (one-hot
    /// encoding along the class dim). The model adds the batch dimension
    /// internally inside Train.
    /// </summary>
    protected override int[] OutputShape => new[] { NumClasses, Height, Width };
}
