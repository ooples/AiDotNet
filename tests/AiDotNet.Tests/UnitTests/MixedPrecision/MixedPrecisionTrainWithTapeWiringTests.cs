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
/// Unit tests for the AiDotNet#1354 wire-up: <c>NeuralNetworkBase.EnableMixedPrecision</c>
/// is now public AND <c>TrainWithTape</c> consults <c>_mixedPrecisionContext</c>
/// during the per-sample <c>model.Train(x, y)</c> path.
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
///   <item>EnableMixedPrecision is invokable from outside the assembly (public).</item>
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
    public async Task EnableMixedPrecision_PublicSurface_AcceptsConfig()
    {
        await Task.Yield();
        var model = BuildTinyDeterministicNetwork();
        // BEFORE issue #1354: EnableMixedPrecision was `internal virtual` — this
        // line was a compile error from outside the assembly. Asserting it
        // compiles + executes proves the public-surface change shipped.
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
        double maxRelDiff = 0;
        for (int i = 0; i < beforeA.Length; i++)
        {
            float deltaA = afterA[i] - beforeA[i];
            float deltaB = afterB[i] - beforeB[i];
            if (System.Math.Abs(deltaA) < 1e-8f) continue;
            double rel = System.Math.Abs((deltaB - deltaA) / deltaA);
            if (rel > maxRelDiff) maxRelDiff = rel;
        }
        Assert.True(maxRelDiff < 0.05,
            $"Per-parameter delta should match across scale=1 and scale=1024 within 5% relative; got maxRel={maxRelDiff:G4}");
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
        // Float.MaxValue inputs guarantee an overflow under any non-trivial
        // forward + loss-scaled backward: 3.4e38 * 65536 in the scaled grad
        // is +inf, the unscale-and-check step catches it, and the optimizer
        // step is skipped.
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
}
