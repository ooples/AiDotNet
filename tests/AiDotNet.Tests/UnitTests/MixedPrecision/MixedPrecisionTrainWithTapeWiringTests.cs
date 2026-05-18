using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.MixedPrecision;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNetVector = AiDotNet.Tensors.LinearAlgebra.Vector<float>;

namespace AiDotNetTests.UnitTests.MixedPrecision;

/// <summary>
/// Unit tests for the AiDotNet#1354 wire-up: <c>TrainWithTape</c> consults
/// <c>_mixedPrecisionContext</c> during the per-sample <c>model.Train(x, y)</c>
/// path. Mixed-precision is configured ONLY through the facade
/// (<c>AiModelBuilder.ConfigureMixedPrecision + BuildAsync</c>); the underlying
/// <c>EnableMixedPrecision</c> remains <c>internal virtual</c> to preserve the
/// facade pattern. These tests reach it via <c>InternalsVisibleTo</c> grant.
///
/// <para>The wire-up contract verified here:</para>
/// <list type="number">
///   <item>Master weights stay in FP32 across calls. After many <c>Train</c> calls
///         the parameter vector reflects FP32 precision updates, not FP16-rounded
///         increments compounded.</item>
///   <item>Per-Train low-precision round-trip is applied to the working weights
///         for the forward pass: immediately after a <c>Train</c> call the FP32
///         master values stored in the layer are NOT the same as the FP16 round-
///         tripped values (proving the master was restored before opt.Step).</item>
///   <item>FP16 + loss-scaling: with a large initial scale, scaled gradients drive
///         the same effective FP32 update as non-mixed-precision training at the
///         same configuration (within numerical noise from the rounded forward).</item>
///   <item>BF16: works without loss scaling (default scale = 1.0).</item>
///   <item><c>BuildAsync</c> applies the stored <c>_mixedPrecisionConfig</c> to
///         the constructed neural-network model via internal
///         <c>EnableMixedPrecision</c> (existing call at AiModelBuilder.cs:2502).</item>
/// </list>
/// </summary>
public class MixedPrecisionTrainWithTapeWiringTests
{
    private const int InputSize = 4;
    private const int OutputSize = 3;

    private static NeuralNetwork<float> BuildTinyDeterministicNetwork()
    {
        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<float>>
        {
            new InputLayer<float>(InputSize),
            new DenseLayer<float>(OutputSize, activationFunction: new IdentityActivation<float>())
        };

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: InputSize,
            outputSize: OutputSize,
            layers: layers);

        var model = new NeuralNetwork<float>(arch, lossFunction: new MeanSquaredErrorLoss<float>());

        // Seed parameters deterministically — independent of the random init.
        var p = model.GetParameters();
        var det = new float[p.Length];
        for (int i = 0; i < det.Length; i++)
        {
            det[i] = ((i % 7) - 3) * 0.1f;
        }
        model.UpdateParameters(new AiDotNetVector(det));
        return model;
    }

    private static Tensor<float> MakeInput(float[] data)
    {
        return new Tensor<float>(new[] { 1, data.Length }, new AiDotNetVector(data));
    }

    [Fact(Timeout = 30000)]
    public async Task EnableMixedPrecision_Internal_AcceptsConfig()
    {
        // This is an INTERNAL-access test reached via InternalsVisibleTo.
        // The public facade path is covered by
        // MixedPrecisionFacadeBuildAsyncTests in IntegrationTests/MixedPrecision.
        // Naming this test "PublicSurface" was misleading — it's testing the
        // internal EnableMixedPrecision call directly, not the facade
        // (review #1362).
        await Task.Yield();
        var model = BuildTinyDeterministicNetwork();
        model.EnableMixedPrecision(new MixedPrecisionConfig
        {
            PrecisionType = MixedPrecisionType.FP16,
            InitialLossScale = 128.0
        });

        Assert.True(model.IsMixedPrecisionEnabled);
        var ctx = model.GetMixedPrecisionContext();
        Assert.NotNull(ctx);
        Assert.Equal(128.0, ctx!.LossScaler.Scale);

        model.DisableMixedPrecision();
        Assert.False(model.IsMixedPrecisionEnabled);
        Assert.Null(model.GetMixedPrecisionContext());
    }

    [Fact(Timeout = 60000)]
    public async Task EnableMixedPrecision_FP16_TrainStillUpdatesMasterWeightsInFP32()
    {
        await Task.Yield();
        var model = BuildTinyDeterministicNetwork();
        model.EnableMixedPrecision(new MixedPrecisionConfig
        {
            PrecisionType = MixedPrecisionType.FP16,
            InitialLossScale = 1.0, // disable scaling so we can compare directly
            EnableDynamicScaling = false
        });

        var paramsBefore = model.GetParameters();
        var x = MakeInput(new float[] { 0.5f, -0.3f, 0.2f, 0.1f });
        var y = MakeInput(new float[] { 1.0f, 0.0f, -0.5f });

        model.Train(x, y);

        var paramsAfter = model.GetParameters();
        Assert.Equal(paramsBefore.Length, paramsAfter.Length);

        // At least one parameter must have changed — otherwise the optimizer
        // skipped the step (which only happens on overflow at this scale).
        bool anyChanged = false;
        for (int i = 0; i < paramsBefore.Length; i++)
        {
            if (paramsBefore[i] != paramsAfter[i]) { anyChanged = true; break; }
        }
        Assert.True(anyChanged, "Train under MP must update master parameters when no overflow");

        // Master weights are FP32: at least one param must retain low
        // mantissa bits that an FP16/BF16 round-trip would have zeroed.
        // Use the public MixedPrecisionContext.HasFullFP32Precision
        // verification API instead of touching BitConverterHelper
        // directly (which is internal per the facade pattern).
        bool anyHasLowMantissaBits = false;
        for (int i = 0; i < paramsAfter.Length; i++)
        {
            if (AiDotNet.MixedPrecision.MixedPrecisionContext.HasFullFP32Precision(paramsAfter[i]))
            {
                anyHasLowMantissaBits = true;
                break;
            }
        }
        Assert.True(anyHasLowMantissaBits,
            "After Train under MP, at least one master weight should retain low mantissa bits (proves FP32 master, not stuck at FP16-rounded values)");
    }

    [Fact(Timeout = 60000)]
    public async Task EnableMixedPrecision_BF16_NoLossScalingNeeded_TrainConverges()
    {
        await Task.Yield();
        var model = BuildTinyDeterministicNetwork();
        model.EnableMixedPrecision(MixedPrecisionConfig.ForBF16());

        var x = MakeInput(new float[] { 0.5f, -0.3f, 0.2f, 0.1f });
        var y = MakeInput(new float[] { 1.0f, 0.0f, -0.5f });

        var paramsBefore = model.GetParameters();
        // Train multiple steps — for BF16 default scale = 1.0 so no scaling occurs.
        for (int step = 0; step < 5; step++)
        {
            model.Train(x, y);
        }
        var paramsAfter = model.GetParameters();

        bool anyChanged = false;
        for (int i = 0; i < paramsBefore.Length; i++)
        {
            if (System.Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-6f)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged, "BF16 mixed-precision training must update parameters over 5 steps");

        // Check loss scaler state — BF16 default config has dynamic scaling off
        // and scale = 1.0. The LossScaler's update-counter still ticks however
        // (we call UnscaleGradientsAndCheck on every step).
        var ctx = model.GetMixedPrecisionContext();
        Assert.NotNull(ctx);
        Assert.Equal(1.0, ctx!.LossScaler.Scale);
    }

    [Fact(Timeout = 60000)]
    public async Task EnableMixedPrecision_FP16_LossScaleNonOne_GradientUnscaledCorrectly()
    {
        await Task.Yield();
        // Build two identical networks. Train one with MP scale=1.0, one with
        // MP scale=1024. The parameter deltas after one Train step should be
        // close (within FP16 quantization noise on the forward) — proving the
        // loss-scaler's "scale up before backward, scale down before update"
        // contract holds.
        var modelA = BuildTinyDeterministicNetwork();
        modelA.EnableMixedPrecision(new MixedPrecisionConfig
        {
            PrecisionType = MixedPrecisionType.FP16,
            InitialLossScale = 1.0,
            EnableDynamicScaling = false
        });

        var modelB = BuildTinyDeterministicNetwork();
        modelB.EnableMixedPrecision(new MixedPrecisionConfig
        {
            PrecisionType = MixedPrecisionType.FP16,
            InitialLossScale = 1024.0,
            EnableDynamicScaling = false
        });

        var x = MakeInput(new float[] { 0.5f, -0.3f, 0.2f, 0.1f });
        var y = MakeInput(new float[] { 1.0f, 0.0f, -0.5f });

        var beforeA = modelA.GetParameters();
        var beforeB = modelB.GetParameters();

        modelA.Train(x, y);
        modelB.Train(x, y);

        var afterA = modelA.GetParameters();
        var afterB = modelB.GetParameters();

        // Both should have changed.
        bool anyChangedA = false, anyChangedB = false;
        for (int i = 0; i < beforeA.Length; i++)
        {
            if (beforeA[i] != afterA[i]) anyChangedA = true;
            if (beforeB[i] != afterB[i]) anyChangedB = true;
        }
        Assert.True(anyChangedA, "MP scale=1.0 must update parameters");
        Assert.True(anyChangedB, "MP scale=1024 must update parameters");

        // Deltas should be in the same ballpark — the loss scaler is doing
        // its job. The forward pass is rounded identically (same FP16
        // round-trip), so the gradient direction is identical. After
        // scale/unscale, the magnitudes must match within float noise.
        //
        // Track how many comparable (non-near-zero) deltas we measured;
        // if every deltaA is below 1e-8, the loop skips all comparisons
        // and maxRelDiff stays 0 — yielding a false pass with no signal
        // (review #1362).
        // Symmetric combined-tolerance form: a single rule that handles both
        // the near-zero and non-near-zero regimes consistently —
        //   |deltaA - deltaB| < ABS_TOL + REL_TOL * max(|deltaA|, |deltaB|)
        // Avoids the previous flaky pattern where deltaA < 1e-8 forced an
        // absolute threshold on deltaB that scale=1024-induced FP16 rounding
        // in the backward could legitimately exceed (review #1362 follow-up).
        const double absTol = 1e-6;
        const double relTol = 0.05;
        int compared = 0;
        for (int i = 0; i < beforeA.Length; i++)
        {
            float deltaA = afterA[i] - beforeA[i];
            float deltaB = afterB[i] - beforeB[i];
            double diff = System.Math.Abs(deltaB - deltaA);
            double mag = System.Math.Max(System.Math.Abs(deltaA), System.Math.Abs(deltaB));
            double tol = absTol + relTol * mag;
            // Always count the comparison — the symmetric tolerance is
            // meaningful at every magnitude, including near zero.
            compared++;
            Assert.True(diff <= tol,
                $"Per-parameter delta mismatch at index {i}: deltaA={deltaA:G6}, " +
                $"deltaB={deltaB:G6}, |diff|={diff:G6} > tol={tol:G6} " +
                $"(absTol={absTol:G3}, relTol*mag={relTol * mag:G6}).");
        }
        Assert.True(compared > 0,
            "No parameter deltas were compared; the invariance check was not exercised.");
    }

    [Fact(Timeout = 60000)]
    public async Task EnableMixedPrecision_AlreadyEnabled_ThrowsInvalidOp()
    {
        await Task.Yield();
        var model = BuildTinyDeterministicNetwork();
        model.EnableMixedPrecision();
        Assert.Throws<System.InvalidOperationException>(() => model.EnableMixedPrecision());
    }

    [Fact(Timeout = 60000)]
    public async Task EnableMixedPrecision_DisableThenReEnable_Works()
    {
        await Task.Yield();
        var model = BuildTinyDeterministicNetwork();
        model.EnableMixedPrecision(new MixedPrecisionConfig { InitialLossScale = 64.0 });
        model.DisableMixedPrecision();
        model.EnableMixedPrecision(new MixedPrecisionConfig { InitialLossScale = 256.0 });
        var ctx = model.GetMixedPrecisionContext();
        Assert.NotNull(ctx);
        Assert.Equal(256.0, ctx!.LossScaler.Scale);
    }

    /// <summary>
    /// Overflow path: when the loss-scaled gradient produces NaN/Inf the
    /// dynamic loss-scaler must (a) keep the optimizer step from updating
    /// master parameters AND (b) back the scale off so the next step has
    /// a chance to succeed. This is a core MP correctness contract that
    /// the rest of the suite doesn't cover directly.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task EnableMixedPrecision_FP16_Overflow_SkipsStepAndBacksOffScale()
    {
        await Task.Yield();
        var model = BuildTinyDeterministicNetwork();
        const double initialScale = 65536.0;
        model.EnableMixedPrecision(new MixedPrecisionConfig
        {
            PrecisionType = MixedPrecisionType.FP16,
            InitialLossScale = initialScale,
            EnableDynamicScaling = true
        });

        var before = model.GetParameters();
        // float.MaxValue inputs produce an overflow that catches the END-TO-END
        // skip-and-back-off contract: FP16 round-trip saturates to +Inf, MSE
        // loss is Inf, scaled loss × 65536 stays Inf, the backward emits NaN,
        // overflow detection fires, optimizer step is skipped, and the loss
        // scaler backs off. This exercises forward-saturation AND scaled-loss
        // overflow together — both arrive at the same skip-and-back-off
        // outcome, so the test verifies the SHARED contract. A more isolated
        // "scaled-loss overflow only" test would need a fixture where the
        // FP32 forward produces a loss large enough that scale × loss
        // overflows FP32 (~3.4e38) without inputs that saturate FP16; on a
        // 1-layer DenseLayer with modest inputs the forward loss is too
        // small for any scale ≤ float.MaxValue to push the scaled loss over
        // FP32 max. Constructing that fixture is its own follow-up
        // (review #1362).
        var x = MakeInput(new[] { float.MaxValue, float.MaxValue, float.MaxValue, float.MaxValue });
        var y = MakeInput(new[] { float.MaxValue, float.MaxValue, float.MaxValue });

        model.Train(x, y);

        var after = model.GetParameters();
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
        {
            Assert.Equal(before[i], after[i]); // step should be skipped on overflow
        }

        var ctx = model.GetMixedPrecisionContext();
        Assert.NotNull(ctx);
        Assert.True(ctx!.LossScaler.Scale < initialScale,
            $"Expected loss scale to back off from {initialScale} after overflow, " +
            $"but scale is still {ctx.LossScaler.Scale}.");
    }

    /// <summary>
    /// CastWeightsToBF16 round-trips the registered master weights through
    /// the BF16 representation (low 16 mantissa bits cleared with round-
    /// to-nearest-even) and returns the FP32-encoded result. Verifies the
    /// public helper (review #1362 flagged it as untested) by checking
    /// (a) NaN/Inf inputs pass through unchanged, (b) finite inputs have
    /// the low 16 mantissa bits cleared in their IEEE binary representation,
    /// and (c) the returned vector length matches the registered master.
    /// </summary>
    [Fact(Timeout = 30000)]
    public async Task CastWeightsToBF16_RegisteredMaster_ReturnsBf16RoundTripped()
    {
        await Task.Yield();
        // Build a context with deterministic master weights spanning a few
        // numeric categories: a zero, a normal positive, a normal negative,
        // a denormal-near value, a NaN, a positive infinity.
        var master = new AiDotNetVector(new float[] {
            0.0f,
            0.10000000149011612f,    // 0x3DCCCCCD — has non-zero low 16 bits
            -1.5f,                   // 0xBFC00000 — low 16 bits are zero
            1.0e-30f,                // 0x0C24EB55 — small normal, non-zero low bits
            float.NaN,
            float.PositiveInfinity,
        });

        var ctx = new MixedPrecisionContext(new MixedPrecisionConfig
        {
            PrecisionType = MixedPrecisionType.BF16,
        });
        ctx.Initialize(master);

        var result = ctx.CastWeightsToBF16();

        Assert.Equal(master.Length, result.Length);

        // (a) NaN propagates (test specifically for NaN; equality on NaN
        // is always false so we use float.IsNaN).
        Assert.True(float.IsNaN(result[4]),
            $"Expected NaN at index 4, got {result[4]}.");
        // (b) +Inf propagates.
        Assert.True(float.IsPositiveInfinity(result[5]),
            $"Expected +Infinity at index 5, got {result[5]}.");

        // (c) Finite values have the low 16 mantissa bits cleared. Verify
        // by checking the IEEE binary representation directly.
        for (int i = 0; i < 4; i++)
        {
            int bits = AiDotNet.MixedPrecision.BitConverterHelper.SingleToInt32Bits(result[i]);
            uint low16 = (uint)bits & 0x0000FFFFu;
            Assert.Equal(0u, low16);
        }

        // 0.0 round-trips to 0.0.
        Assert.Equal(0.0f, result[0]);
        // -1.5 round-trips to -1.5 (its low 16 bits are already zero).
        Assert.Equal(-1.5f, result[2]);
    }
}
