using AiDotNet.Tensors.Engines.Compilation;
using PrecisionMode = AiDotNet.Tensors.Engines.Gpu.PrecisionMode;

namespace AiDotNet.Training;

/// <summary>
/// Mixed precision for a fused compiled training step, the way PyTorch's <c>torch.autocast</c> + <c>GradScaler</c> run it:
/// FP32 master weights and activations, matrix products multiplied in <see cref="MatMulPrecision"/> (FP16 or BF16 tensor
/// cores, FP32 accumulation) inside the compiled plan, and, when <see cref="LossScaling"/> is set, the loss scaled into
/// the backward and the gradients unscaled, checked and the scale updated inside the plan (on the device on CUDA).
/// </summary>
internal sealed class FusedMixedPrecision
{
    /// <summary>The precision matrix products multiply in.</summary>
    public required PrecisionMode MatMulPrecision { get; init; }

    /// <summary>The plan's loss scaling, or null for none (BF16 needs none).</summary>
    public LossScalingOptions? LossScaling { get; init; }

    /// <summary>The model's host loss scaler: the plan starts from its state and it reads the plan's state back.</summary>
    public MixedPrecision.LossScaler<float>? HostScaler { get; init; }

    /// <summary>Identifies the configuration, so a change after the plan committed is refused rather than half-applied.</summary>
    public string Signature => LossScaling is { } s
        ? $"{MatMulPrecision}|scale {s.InitialScale} x{s.GrowthFactor}/{s.BackoffFactor} every {s.GrowthInterval} in [{s.MinScale},{s.MaxScale}] dynamic={s.Dynamic}"
        : $"{MatMulPrecision}|no scaling";

    /// <summary>
    /// The fused-path form of <paramref name="config"/>, or null with <paramref name="reason"/> when it has none (FP8,
    /// which this path does not multiply in).
    /// </summary>
    public static FusedMixedPrecision? From(MixedPrecision.MixedPrecisionContext context, out string? reason)
    {
        var config = context.Config;
        reason = null;
        PrecisionMode precision;
        switch (config.PrecisionType)
        {
            case Enums.MixedPrecisionType.FP16: precision = PrecisionMode.Float16; break;
            case Enums.MixedPrecisionType.BF16: precision = PrecisionMode.BFloat16; break;
            default:
                reason = $"mixed precision {config.PrecisionType} has no fused form (FP16 and BF16 do)";
                return null;
        }
        var scaler = context.LossScaler;
        // From the configuration, not scaler.Scale: once a plan owns the scaler, reading its scale synchronizes with the
        // device, and this runs every step.
        bool scales = config.EnableDynamicScaling || config.InitialLossScale != 1.0;
        return new FusedMixedPrecision
        {
            MatMulPrecision = precision,
            HostScaler = scaler,
            LossScaling = scales
                ? new LossScalingOptions
                {
                    InitialScale = (float)config.InitialLossScale,
                    GrowthFactor = (float)scaler.GrowthFactor,
                    BackoffFactor = (float)scaler.BackoffFactor,
                    GrowthInterval = scaler.GrowthInterval,
                    MinScale = (float)scaler.MinScale,
                    MaxScale = (float)scaler.MaxScale,
                    Dynamic = scaler.DynamicScaling,
                }
                : null,
        };
    }

    /// <summary>The host scaler's state, as the plan's starting state.</summary>
    public LossScaleState? HostState()
    {
        if (HostScaler is not { } scaler) return null;
        var (scale, tracker, steps, overflows) = scaler.HostState;
        return new LossScaleState((float)scale, tracker, overflows, steps);
    }

    /// <summary>Makes <paramref name="plan"/> the source of the host scaler's state.</summary>
    public void AttachHostScaler(ICompiledTrainingPlanLossScaling plan)
    {
        if (HostScaler is not { } scaler) return;
        var weak = new WeakReference<ICompiledTrainingPlanLossScaling>(plan);
        scaler.AttachDeviceState(() =>
        {
            if (!weak.TryGetTarget(out var live) || live.ReadLossScaleState() is not { } state) return null;
            return (state.Scale, state.GrowthTracker, state.Steps, state.OverflowSteps);
        });
    }
}
