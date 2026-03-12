using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Interval Score Matching (ISM) for improved 3D score distillation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ISM reduces the variance of SDS gradients by using deterministic DDIM inversion between
/// two timesteps instead of random noise injection. Given a rendered view, ISM deterministically
/// noises it to timestep t2, then denoises to timestep t1, and uses the difference as the
/// gradient signal. This "interval" approach produces much lower-variance gradients than SDS.
/// </para>
/// <para>
/// <b>For Beginners:</b> SDS gradients are noisy because they compare random noise with the
/// model's prediction. ISM is cleverer — it uses a deterministic process to add and remove
/// noise over a small interval. This gives much smoother, more reliable feedback for 3D
/// optimization, resulting in cleaner 3D models with fewer artifacts.
/// </para>
/// <para>
/// Reference: Liang et al., "LucidDreamer: Towards High-Fidelity Text-to-3D Generation via
/// Interval Score Matching", CVPR 2024
/// </para>
/// </remarks>
public class IntervalScoreMatching<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _diffusionModel;
    private readonly double _guidanceScale;
    private readonly int _intervalSteps;

    /// <summary>
    /// Gets the guidance scale.
    /// </summary>
    public double GuidanceScale => _guidanceScale;

    /// <summary>
    /// Gets the number of DDIM steps in the interval.
    /// </summary>
    public int IntervalSteps => _intervalSteps;

    /// <summary>
    /// Initializes a new ISM instance.
    /// </summary>
    /// <param name="diffusionModel">Pretrained 2D diffusion model.</param>
    /// <param name="guidanceScale">CFG scale (default: 7.5).</param>
    /// <param name="intervalSteps">Number of DDIM steps between t1 and t2 (default: 1).</param>
    public IntervalScoreMatching(
        IDiffusionModel<T> diffusionModel,
        double guidanceScale = 7.5,
        int intervalSteps = 1)
    {
        _diffusionModel = diffusionModel;
        _guidanceScale = guidanceScale;
        _intervalSteps = intervalSteps;
    }

    /// <summary>
    /// Computes the ISM gradient between two timestep points.
    /// </summary>
    /// <param name="denoisedAtT1">DDIM-denoised result at timestep t1 (closer to clean).</param>
    /// <param name="invertedAtT2">DDIM-inverted result at timestep t2 (more noisy).</param>
    /// <param name="originalRender">Original rendered view.</param>
    /// <param name="timestepWeight">Weight for this interval.</param>
    /// <returns>ISM gradient.</returns>
    public Vector<T> ComputeGradient(
        Vector<T> denoisedAtT1, Vector<T> invertedAtT2,
        Vector<T> originalRender, double timestepWeight)
    {
        var gradient = new Vector<T>(originalRender.Length);
        var weight = NumOps.FromDouble(timestepWeight);

        for (int i = 0; i < gradient.Length; i++)
        {
            var target = i < denoisedAtT1.Length ? denoisedAtT1[i] : NumOps.Zero;
            var current = originalRender[i];
            var diff = NumOps.Subtract(target, current);
            gradient[i] = NumOps.Multiply(weight, diff);
        }

        return gradient;
    }
}
