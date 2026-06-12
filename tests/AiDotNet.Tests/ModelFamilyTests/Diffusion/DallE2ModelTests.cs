using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Models.Options;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// DALL·E 2 test scaffold — see <see cref="DiffusionModelTestBase{TNum}"/>
/// for the FP32 rationale. DALL·E 2 at paper defaults OOMs in fresh-process
/// probes at FP64 on the 16 GB CI host.
/// </summary>
/// <remarks>
/// <c>DefaultInferenceSteps = 4</c>: <c>Predict</c> runs the full denoising
/// loop through BOTH paper-scale U-Nets (prior + decoder) per step, so the
/// option default of 10 steps × 2 Predicts per determinism test exceeds the
/// 120 s per-test budget on CPU (dotnet-trace profile, 2026-06-12 — after the
/// DiffusionResBlock ctor probe-Forward removal, sampling is all that
/// remains). The invariants asserted here (determinism, shape, input
/// sensitivity, finiteness) hold at any step count; per the
/// <see cref="DiffusionModelOptions{T}.DefaultInferenceSteps"/> contract,
/// quality-sensitive callers pass a step count to <c>Generate</c> directly.
/// </remarks>
public class DallE2ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new DallE2Model<float>(
            options: new DiffusionModelOptions<float>
            {
                // Model defaults from DallE2Model's ctor, unchanged:
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear,
                // Test-budget override (see class remarks):
                DefaultInferenceSteps = 4,
            },
            seed: 42);
}
