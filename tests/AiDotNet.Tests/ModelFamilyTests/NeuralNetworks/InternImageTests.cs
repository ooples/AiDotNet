using AiDotNet.ComputerVision.Segmentation.Semantic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for InternImage (Wang et al. 2023, "InternImage: Exploring
/// Large-Scale Vision Foundation Models with Deformable Convolutions",
/// arXiv:2211.05778). The auto-generator is told to skip InternImage
/// (<c>ExcludedClassNames</c>) so this hand-written scaffold is authoritative.
/// </summary>
/// <remarks>
/// <b>Why a reduced-scale config:</b> even InternImage's smallest variant (Tiny) is a
/// 30-block deformable-convolution backbone ([4,4,18,4] DCNv3 blocks across 4 stages,
/// [64,128,256,512] channels) at 512x512 — genuine heavy vision-foundation compute
/// whose forward+backward, times the training invariants' iterations, far exceeds the
/// 120s CI budget on CPU (the tests timed out). These invariants validate the
/// <i>architecture's code paths</i> (patch-embed stem, the 4 DCNv3 stages, downsample
/// transitions, segmentation head, backprop, optimizer step, clone) — not paper-scale
/// numerical behaviour. A 4-class, 32x32 input keeps the full Tiny architecture (all
/// 30 blocks) while shrinking only the spatial resolution, so every path runs in a
/// budget-safe time. MoreData iterations are trimmed like the other heavy-backbone
/// segmentation scaffolds (UniVS/SwinUNETR).
/// </remarks>
// Runs in <float> (not the SegmentationTestBase default <double>): even at reduced spatial
// scale the 30-block DCNv3 Tiny backbone's multi-iteration training invariants overran the
// 180 s gate at <double>. <float> halves per-step compute + the tape/activation footprint
// while keeping the full Tiny architecture and the self-relative invariants intact — the same
// lever the generated heavy-backbone scaffolds use via Fp32TestClassNames.
public class InternImageTests : SegmentationTestBase<float>
{
    private const int NumClasses = 4;
    private const int Height = 32;
    private const int Width = 32;
    private const int Channels = 3;

    protected override int[] InputShape => [Channels, Height, Width];

    protected override int[] OutputShape => [NumClasses, Height, Width];

    protected override int MoreDataShortIterations => 1;
    protected override int MoreDataLongIterations => 2;

    // TrainingError runs three times this value. Two base iterations therefore
    // retain a six-step fit check without spending the entire 120-second gate
    // on repeated full-backbone backward passes.
    protected override int TrainingIterations => 2;

    // One full Tiny-backbone backward pass is intentionally expensive: the
    // shared 100-step memorization probe exceeds 180 seconds on the 4-CPU
    // runner. Ten repeated steps still verify loss reduction through all 30
    // DCNv3 blocks while keeping the invariant within its execution budget.
    protected override int MemorizationTaskIterations => 10;

    // Same reason as every cap above, for the target-dependence probe: its default is
    // TargetDependenceStepCount (3) x TargetDependenceRepeatCount (3) = nine full backward passes
    // through all 30 DCNv3 blocks, which alone exceeds the 120-second gate.
    //
    // ONE repeat, not fewer steps. The repeats average away seed noise in the update-direction
    // cosine, whereas the three STEPS are what carry the signal: a single Adam step is sign(g), so
    // two targets that differ only in magnitude produce identical first steps and the probe would
    // report a false pass. Cutting the axis that costs the same but proves less is the whole point
    // of the cap rung.
    protected override int TargetDependenceRepeatCount => 1;

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: Height,
            inputWidth: Width,
            inputDepth: Channels,
            outputSize: NumClasses);

        return new InternImage<float>(architecture, numClasses: NumClasses,
            modelSize: InternImageModelSize.Tiny, dropRate: 0.0);
    }
}
