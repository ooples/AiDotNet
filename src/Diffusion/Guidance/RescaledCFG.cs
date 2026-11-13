using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Attributes;

namespace AiDotNet.Diffusion.Guidance;

/// <summary>
/// Rescaled Classifier-Free Guidance to prevent over-saturation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Rescales the guided noise prediction to match the standard deviation of the
/// conditional prediction. This prevents the color saturation and contrast
/// blowout that occurs at high CFG scales (e.g., > 10).
/// </para>
/// <para>
/// <b>For Beginners:</b> When you use a very high guidance scale, images can become
/// oversaturated with unnaturally bright colors. This fix automatically adjusts
/// the brightness back to normal while keeping the guidance effect.
/// </para>
/// <para>
/// Reference: Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed", 2023
/// </para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class RescaledCFG<T> : IGuidanceMethod<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _rescaleWeight;

    /// <inheritdoc />
    public GuidanceType GuidanceType => GuidanceType.RescaledCFG;

    /// <summary>
    /// Initializes a new Rescaled CFG instance.
    /// </summary>
    /// <param name="rescaleWeight">Blending weight between standard and rescaled CFG. 0=standard, 1=fully rescaled. Default: 0.7.</param>
    public RescaledCFG(double rescaleWeight = 0.7)
    {
        _rescaleWeight = rescaleWeight;
    }

    /// <inheritdoc />
    public Tensor<T> Apply(Tensor<T> unconditional, Tensor<T> conditional, double scale, double timestep)
    {
        // Half cannot represent the degeneracy epsilon and can overflow variance reductions.
#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(Half))
            return new RescaledCFG<double>(_rescaleWeight)
                .Apply(unconditional.Cast<double>(), conditional.Cast<double>(), scale, timestep).Cast<T>();
#endif
        var engine = AiDotNetEngine.Current;

        // Step 1: Standard CFG: guided = uncond + scale * (cond - uncond)
        var scaleT = NumOps.FromDouble(scale);
        var diff = engine.TensorSubtract<T>(conditional, unconditional);
        var scaled = engine.TensorMultiplyScalar<T>(diff, scaleT);
        var guided = engine.TensorAdd<T>(unconditional, scaled);

        // Step 2: Standard deviations per sample (every axis but the batch axis), as in the paper and diffusers'
        // rescale_noise_cfg
        var axes = DiffusionVectorOps.SampleAxes(guided.Shape.Length);
        var condStd = DiffusionVectorOps.Std(conditional, axes);
        var guidedStd = DiffusionVectorOps.Std(guided, axes);

        // Step 3: Rescale to match conditional std
        // result = (1 - weight) * guided + weight * (rescaleFactor * guided)
        // Simplifies to: guided * ((1 - weight) + weight * rescaleFactor); a sample whose guided std is below
        // 1e-8 is left unscaled.
        var ones = Tensor<T>.CreateDefault(guidedStd._shape, NumOps.One);
        var degenerate = engine.TensorLessThan(guidedStd, NumOps.FromDouble(1e-8));
        var safeStd = engine.TensorWhere(degenerate, ones, guidedStd);
        var rescaleFactor = engine.TensorDivide(condStd, safeStd);
        var blendedScale = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(rescaleFactor, NumOps.FromDouble(_rescaleWeight)), NumOps.FromDouble(1.0 - _rescaleWeight));
        blendedScale = engine.TensorWhere(degenerate, ones, blendedScale);
        return engine.TensorMultiply<T>(guided, engine.TensorBroadcastTo(blendedScale, guided._shape));
    }
}
