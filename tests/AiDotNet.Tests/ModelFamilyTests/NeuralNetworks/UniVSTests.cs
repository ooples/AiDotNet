using AiDotNet.ComputerVision.Segmentation.Video;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for UniVS (Li et al. 2024, "UniVS: Unified and Universal
/// Video Segmentation with Prompts as Queries"). The auto-generator is told to
/// skip UniVS (<c>ExcludedClassNames</c>) so this hand-written scaffold is
/// authoritative.
/// </summary>
/// <remarks>
/// <b>Why a reduced-scale config:</b> UniVS's default backbone is a full ResNet-50
/// (R50) feeding a Mask2Former-style decoder over 80 classes at 480x480 — genuine
/// heavy conv+attention compute whose forward+backward, times the training
/// invariants' iterations, exceeds the 120/180s CI budget on CPU. These invariants
/// validate the <i>architecture's code paths</i> (backbone stages, pixel decoder,
/// transformer decoder, per-pixel classification, backprop, optimizer step, clone)
/// — not paper-scale numerical behaviour. A 4-class, 64x64 config exercises every
/// path in seconds while keeping the architecture faithful.
/// </remarks>
public class UniVSTests : SegmentationTestBase
{
    private const int NumClasses = 4;
    private const int Height = 64;
    private const int Width = 64;
    private const int Channels = 3;

    protected override int[] InputShape => [Channels, Height, Width];

    protected override int[] OutputShape => [NumClasses, Height, Width];

    // The ResNet-50 backbone downsamples 32x, so at 64x64 the deepest stage is 2x2 —
    // BatchNorm's batch-1 statistics over 4 spatial samples are noisy, and a long
    // memorization run drifts the eval loss slightly upward. Trim MoreData's iteration
    // counts (base defaults 50/200) the same way SwinUNETR does for its heavy Swin
    // encoder: the "more data must not degrade" invariant still holds at 10/40, it just
    // trains for fewer, budget-safe, stability-safe steps.
    protected override int MoreDataShortIterations => 10;
    protected override int MoreDataLongIterations => 40;

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: Height,
            inputWidth: Width,
            inputDepth: Channels,
            outputSize: NumClasses);

        return new UniVS<double>(architecture, numClasses: NumClasses, dropRate: 0.0);
    }
}
