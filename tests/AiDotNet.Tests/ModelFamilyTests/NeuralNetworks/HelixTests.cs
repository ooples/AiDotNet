using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for Helix (Figure AI 2025, "Helix: A Vision-Language-
/// Action Model for Generalist Humanoid Control", arXiv:2502.07092). The
/// auto-generator is told to skip Helix (<c>ExcludedClassNames</c>) so this
/// hand-written scaffold is authoritative.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a reduced-scale config (Janus precedent).</b> Helix's paper defaults
/// make it a ~6.7B-parameter dual-system VLA (System-2 VLM DecoderDim=4096 × 32
/// layers + System-1 visuomotor transformer). A single full-model Adam step at
/// that scale cannot complete in the 120 s CI budget on CPU at <i>any</i>
/// precision — profiled at &gt;580 s/step in fp64 (a hard memory/IO wall on a
/// 64 GB box even with weight streaming) and still &gt;120 s in float. That is a
/// property of running a 7B step on CPU, not a bug: the memory-bounded streaming
/// training path (8-bit Adam optimizer-in-backward, see
/// <see cref="NeuralNetworkBase{T}.StreamingTraining"/>) makes such a step
/// <i>possible</i> where it would otherwise OOM, but does not make it
/// <i>fast</i> enough for a unit-test budget.
/// </para>
/// <para>
/// These model-family invariants validate the <i>architecture's code paths</i>
/// (the dual-system vision→S2→S1 chain, attention/FFN wiring, backprop,
/// optimizer step, clone) — not paper-scale numerical behaviour. A smaller
/// float config exercises every one of those paths in seconds while keeping the
/// architecture's SHAPE faithful; the dims below are scaled down ~4-8×, the
/// wiring is unchanged. The streaming training subsystem is exercised at a scale
/// that actually engages it by the dedicated streaming integration tests, not
/// here.
/// </para>
/// </remarks>
public class HelixTests : VisionLanguageTestBase<float>
{
    // Post-patch-embedding token features [batch, num_tokens, VisionDim] — Helix's
    // native chain begins with LayerNorm + vision attention, so it consumes token
    // tensors, not raw pixels. VisionDim is the reduced 256 (see CreateNetwork).
    protected override int[] InputShape => [1, 4, 256];

    // Full dual-system chain ends in the action head (ActionDimension = 35).
    protected override int[] OutputShape => [1, 4, 35];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 224,
            inputWidth: 224,
            inputDepth: 3,
            outputSize: 4);

        // Reduced-scale config (see <remarks>): same dual-system architecture
        // shape as the ~6.7B paper model, ~4-8× smaller dims so all invariants
        // fit the CI budget. Dropout off so the memorization invariants see clean
        // convergence.
        var options = new HelixOptions
        {
            VisionDim = 256,
            DecoderDim = 512,
            NumVisionLayers = 4,
            NumDecoderLayers = 4,
            NumHeads = 8,
            System2LatentDim = 128,
            System1HiddenDim = 96,
            System1NumLayers = 2,
            System1NumHeads = 4,
            ActionDimension = 35,
            DropoutRate = 0.0,
        };

        return new Helix<float>(architecture, options);
    }
}
