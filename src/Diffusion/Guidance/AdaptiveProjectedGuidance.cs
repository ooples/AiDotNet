using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Attributes;

namespace AiDotNet.Diffusion.Guidance;

/// <summary>
/// Adaptive Projected Guidance (APG) for diffusion model inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// APG projects the guidance direction to reduce components that cause artifacts.
/// It decomposes the guidance vector (per sample) into parallel and orthogonal components
/// relative to the conditional prediction; the parallel component causes oversaturation and is
/// down-weighted, while the orthogonal component, which improves quality, is kept.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard guidance can sometimes push the image in bad
/// directions, causing artifacts. APG is smarter — it only keeps the "useful"
/// part of the guidance while discarding the part that causes problems.
/// </para>
/// <para>
/// Reference: Sadat et al., "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion
/// Models", 2024 (APG). The update is d_perp + eta d_par with eta = 1 - projectionStrength; the paper's eta = 0
/// corresponds to projectionStrength = 1.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class AdaptiveProjectedGuidance<T> : IGuidanceMethod<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _projectionStrength;

    /// <inheritdoc />
    public GuidanceType GuidanceType => GuidanceType.AdaptiveProjected;

    /// <summary>
    /// Initializes a new Adaptive Projected Guidance instance.
    /// </summary>
    /// <param name="projectionStrength">How much of the parallel (oversaturating) component is removed. 0=standard CFG, 1=full projection (the paper's eta = 0). Default: 0.5.</param>
    public AdaptiveProjectedGuidance(double projectionStrength = 0.5)
    {
        _projectionStrength = projectionStrength;
    }

    /// <inheritdoc />
    public Tensor<T> Apply(Tensor<T> unconditional, Tensor<T> conditional, double scale, double timestep)
    {
        // The projection accumulates in double: Half squares overflow for ordinary finite logits, and float
        // products and their sums overflow for large ones (1e20 squared).
        bool narrow = typeof(T) == typeof(float);
#if NET5_0_OR_GREATER
        narrow |= typeof(T) == typeof(Half);
#endif
        if (narrow)
            return TensorCopyHelper.CastLogical<double, T>(new AdaptiveProjectedGuidance<double>(_projectionStrength)
                .Apply(TensorCopyHelper.CastLogical<T, double>(unconditional),
                    TensorCopyHelper.CastLogical<T, double>(conditional), scale, timestep));
        var engine = AiDotNetEngine.Current;
        int[] shape = conditional._shape;
        var axes = DiffusionVectorOps.SampleAxes(shape.Length);

        // Compute guidance direction: d = cond - uncond
        var direction = engine.TensorSubtract(conditional, unconditional);

        // Project the direction onto each sample's conditional prediction: parallel = (<d, c> / <c, c>) c
        // (0 when |c|^2 <= 1e-10)
        var dotProduct = engine.ReduceSum(engine.TensorMultiply(direction, conditional), axes, true);
        var condNormSq = engine.ReduceSum(engine.TensorSquare(conditional), axes, true);
        var nonDegenerate = engine.TensorGreaterThan(condNormSq, NumOps.FromDouble(1e-10));
        var safeNormSq = engine.TensorWhere(nonDegenerate, condNormSq, Tensor<T>.CreateDefault(condNormSq._shape, NumOps.One));
        var projScale = engine.TensorMultiply(engine.TensorDivide(dotProduct, safeNormSq), nonDegenerate);
        var parallel = engine.TensorMultiply(engine.TensorBroadcastTo(projScale, shape), conditional);

        // Orthogonal component (kept) and parallel component (causes oversaturation, down-weighted by the
        // projection strength): update = d_perp + (1 - projectionStrength) d_par, as in APG with eta = 1 - strength
        var perpendicular = engine.TensorSubtract(direction, parallel);
        var update = engine.TensorAdd(perpendicular, engine.TensorMultiplyScalar(parallel, NumOps.FromDouble(1.0 - _projectionStrength)));

        return engine.TensorAdd(unconditional, engine.TensorMultiplyScalar(update, NumOps.FromDouble(scale)));
    }
}
