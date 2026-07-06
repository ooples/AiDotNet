using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Phase E (#1624) mixed-precision consumer wiring: the SGD path
/// (CompiledTapeTrainingStep.StepMixedPrecision) must loss-scale via a GradScaler, matching the
/// Adam/generic paths, so FP16 backward gradients don't underflow to zero before reaching the FP32
/// master weights. That depends on the Tensors 0.102.0+ GradScaler API being resolvable through
/// MixedPrecisionReflection (the consumer is pinned to 0.102.2). These tests pin that the API resolves
/// on the pinned package — if a future downgrade drops it, StepSgd would silently fall back to the
/// unscaled 2-arg Step and FP16 SGD would underflow, so failing loudly here is the intended guard.
/// </summary>
public class MixedPrecisionPhaseETests
{
    [Fact]
    public void GradScaler_ResolvesOnPinnedTensorsPackage()
    {
        // CreateGradScaler probes for AiDotNet.Tensors.Engines.Autodiff.GradScaler and constructs one.
        // Non-null proves the 0.102.0+ loss-scaling API is present on the pinned package (0.102.2),
        // which is the prerequisite for StepSgd to bind the 3-param loss-scaled Step.
        var scaler = MixedPrecisionReflection.CreateGradScaler(1024f);
        Assert.NotNull(scaler);
    }

    [Fact]
    public void GradScaler_DistinctInstancesPerCall()
    {
        // Each consumer optimizer path persists its own scaler; the factory must mint fresh instances
        // (a shared singleton would cross-contaminate dynamic loss scales between models).
        var a = MixedPrecisionReflection.CreateGradScaler(512f);
        var b = MixedPrecisionReflection.CreateGradScaler(512f);
        Assert.NotNull(a);
        Assert.NotNull(b);
        Assert.NotSame(a, b);
    }
}
