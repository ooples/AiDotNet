using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Guidance;

/// <summary>
/// Adaptive Projected Guidance (APG) for diffusion model inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// APG projects the guidance direction to reduce components that cause artifacts.
/// It decomposes the guidance vector into parallel and perpendicular components
/// relative to the conditional prediction, keeping only the beneficial part.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard guidance can sometimes push the image in bad
/// directions, causing artifacts. APG is smarter — it only keeps the "useful"
/// part of the guidance while discarding the part that causes problems.
/// </para>
/// <para>
/// Reference: Ahn et al., "Adaptive Projected Guidance", 2024
/// </para>
/// </remarks>
public class AdaptiveProjectedGuidance<T> : IGuidanceMethod<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _projectionStrength;

    /// <inheritdoc />
    public GuidanceType GuidanceType => GuidanceType.AdaptiveProjected;

    /// <summary>
    /// Initializes a new Adaptive Projected Guidance instance.
    /// </summary>
    /// <param name="projectionStrength">Strength of perpendicular projection removal. 0=standard CFG, 1=full projection. Default: 0.5.</param>
    public AdaptiveProjectedGuidance(double projectionStrength = 0.5)
    {
        _projectionStrength = projectionStrength;
    }

    /// <inheritdoc />
    public Tensor<T> Apply(Tensor<T> unconditional, Tensor<T> conditional, double scale, double timestep)
    {
        var uncondSpan = unconditional.AsSpan();
        var condSpan = conditional.AsSpan();

        // Compute guidance direction: d = cond - uncond
        int len = uncondSpan.Length;
        var direction = new double[len];
        double dotProduct = 0;
        double condNormSq = 0;

        for (int i = 0; i < len; i++)
        {
            double c = NumOps.ToDouble(condSpan[i]);
            double u = NumOps.ToDouble(uncondSpan[i]);
            direction[i] = c - u;
            dotProduct += direction[i] * c;
            condNormSq += c * c;
        }

        // Project direction onto conditional prediction
        double projScale = condNormSq > 1e-10 ? dotProduct / condNormSq : 0;

        var result = new Tensor<T>(unconditional.Shape);
        var resultSpan = result.AsWritableSpan();

        for (int i = 0; i < len; i++)
        {
            double c = NumOps.ToDouble(condSpan[i]);

            // Parallel component (aligned with conditional)
            double parallel = projScale * c;

            // Perpendicular component (causes artifacts)
            double perpendicular = direction[i] - parallel;

            // Remove perpendicular component based on projection strength
            double projectedDirection = parallel + perpendicular * (1.0 - _projectionStrength);

            resultSpan[i] = NumOps.FromDouble(
                NumOps.ToDouble(uncondSpan[i]) + scale * projectedDirection);
        }

        return result;
    }
}
