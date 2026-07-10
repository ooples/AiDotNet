using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for IconVSR (Chan et al., CVPR 2021, "BasicVSR: The
/// Search for Essential Components in Video Super-Resolution and Beyond",
/// arXiv:2012.02181). The auto-generator is told to skip IconVSR
/// (<c>ExcludedClassNames</c>) so this hand-written scaffold is authoritative.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a reduced-scale config:</b> IconVSR's production defaults are 30
/// residual blocks with 4x pixel-shuffle upsampling (NumResBlocks=30,
/// ScaleFactor=4, NumFeatures=64). That is genuine heavy conv compute — a single
/// forward+backward over the test clip, times the training invariants' up-to-250
/// iterations, exceeds the 120/180s per-test CI budget on CPU (verified: the
/// training invariants time out at 180000ms).
/// </para>
/// <para>
/// These invariants validate the <i>architecture's code paths</i> (initial feature
/// conv, residual blocks, pixel-shuffle upsampling, linear reconstruction head,
/// backprop, optimizer step, clone) — not paper-scale numerical behaviour. A
/// reduced config (4 residual blocks, 2x upscale, 16 features, small clip)
/// exercises every path in seconds while keeping the residual-block +
/// pixel-shuffle architecture faithful.
/// </para>
/// </remarks>
public class IconVSRTests : VideoSuperResolutionTestBase
{
    // Small video clip: [frames, channels, height, width]. 2x upscale keeps the
    // output ([2, 3, 32, 32]) larger than the input for the SR invariants.
    protected override int[] InputShape => [2, 3, 16, 16];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 16,
            inputWidth: 16,
            inputDepth: 3,
            outputSize: 3);

        var options = new IconVSROptions
        {
            NumFeatures = 16,
            NumResBlocks = 4,
            ScaleFactor = 2,
            DropoutRate = 0.0,
        };
        return new IconVSR<double>(architecture, options);
    }
}
