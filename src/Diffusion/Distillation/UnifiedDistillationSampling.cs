using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Unified Distillation Sampling (UDS) framework unifying SDS, VSD, CSD, and ISM variants.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// UDS provides a unified theoretical framework that encompasses SDS, VSD, CSD, ISM, and DSD
/// as special cases. It parameterizes the score distillation loss with configurable weights
/// for different gradient components, allowing smooth interpolation between methods and
/// enabling new hybrid approaches.
/// </para>
/// <para>
/// <b>For Beginners:</b> Different score distillation methods (SDS, VSD, etc.) each have
/// strengths and weaknesses. UDS is a unified framework that can act as any of them by
/// adjusting its settings. It's like a Swiss Army knife for 3D generation — you can dial
/// in the exact balance of quality, speed, and diversity that you need.
/// </para>
/// <para>
/// Reference: Katzir et al., "A Unified Framework for Score Distillation", 2024
/// </para>
/// </remarks>
public class UnifiedDistillationSampling<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _diffusionModel;
    private readonly double _pretrainedWeight;
    private readonly double _particleWeight;
    private readonly double _noiseWeight;
    private readonly double _guidanceScale;

    /// <summary>
    /// Gets the weight for the pretrained model score component.
    /// </summary>
    public double PretrainedWeight => _pretrainedWeight;

    /// <summary>
    /// Gets the weight for the particle/LoRA model score component.
    /// </summary>
    public double ParticleWeight => _particleWeight;

    /// <summary>
    /// Gets the weight for the noise baseline component.
    /// </summary>
    public double NoiseWeight => _noiseWeight;

    /// <summary>
    /// Initializes a new UDS instance.
    /// </summary>
    /// <param name="diffusionModel">Pretrained 2D diffusion model.</param>
    /// <param name="pretrainedWeight">Weight for pretrained score (default: 1.0 = SDS-like).</param>
    /// <param name="particleWeight">Weight for particle score (default: 0.0; set to 1.0 for VSD-like).</param>
    /// <param name="noiseWeight">Weight for noise baseline (default: 1.0).</param>
    /// <param name="guidanceScale">CFG scale (default: 50.0).</param>
    public UnifiedDistillationSampling(
        IDiffusionModel<T> diffusionModel,
        double pretrainedWeight = 1.0,
        double particleWeight = 0.0,
        double noiseWeight = 1.0,
        double guidanceScale = 50.0)
    {
        _diffusionModel = diffusionModel;
        _pretrainedWeight = pretrainedWeight;
        _particleWeight = particleWeight;
        _noiseWeight = noiseWeight;
        _guidanceScale = guidanceScale;
    }

    /// <summary>
    /// Computes the unified gradient combining all score components.
    /// </summary>
    /// <param name="pretrainedScore">Score from pretrained model.</param>
    /// <param name="particleScore">Score from particle/LoRA model (can be null if not used).</param>
    /// <param name="addedNoise">Noise that was added to the rendered view.</param>
    /// <param name="timestepWeight">Timestep-dependent weight.</param>
    /// <returns>Unified gradient.</returns>
    public Vector<T> ComputeGradient(
        Vector<T> pretrainedScore, Vector<T>? particleScore,
        Vector<T> addedNoise, double timestepWeight)
    {
        var gradient = new Vector<T>(pretrainedScore.Length);
        var w = NumOps.FromDouble(timestepWeight);

        for (int i = 0; i < gradient.Length; i++)
        {
            var grad = NumOps.Zero;

            // Pretrained score component
            grad = NumOps.Add(grad, NumOps.Multiply(
                NumOps.FromDouble(_pretrainedWeight), pretrainedScore[i]));

            // Particle score component (if available)
            if (particleScore != null && i < particleScore.Length)
            {
                grad = NumOps.Subtract(grad, NumOps.Multiply(
                    NumOps.FromDouble(_particleWeight), particleScore[i]));
            }

            // Noise baseline component
            if (i < addedNoise.Length)
            {
                grad = NumOps.Subtract(grad, NumOps.Multiply(
                    NumOps.FromDouble(_noiseWeight), addedNoise[i]));
            }

            gradient[i] = NumOps.Multiply(w, grad);
        }

        return gradient;
    }
}
