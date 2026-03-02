using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Alignment;

/// <summary>
/// Reward-guided sampling for inference-time alignment of diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Reward guidance modifies the sampling process at inference time by incorporating
/// gradients from a reward model. Unlike RLHF which fine-tunes the model, reward
/// guidance keeps the base model frozen and steers the denoising trajectory toward
/// higher-reward regions using the reward model's gradient signal.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of permanently changing the model (like RLHF does),
/// reward guidance acts like a compass during image generation. At each denoising step,
/// it asks the reward model "which direction leads to better images?" and nudges the
/// generation that way. This is flexible — you can change the reward model or guidance
/// strength without retraining.
/// </para>
/// <para>
/// Reference: Xu et al., "Imagereward: Learning and evaluating human preferences for text-to-image generation", NeurIPS 2023
/// </para>
/// </remarks>
public class RewardGuidance<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _guidanceScale;
    private readonly double _truncationTimestep;
    private readonly double _gradientClipNorm;

    /// <summary>
    /// Initializes a new reward guidance module.
    /// </summary>
    /// <param name="guidanceScale">Scale for reward gradient guidance (default: 5.0).</param>
    /// <param name="truncationTimestep">Stop guidance after this fraction of steps for stability (default: 0.5).</param>
    /// <param name="gradientClipNorm">Maximum L2 norm for gradient clipping (default: 1.0).</param>
    public RewardGuidance(
        double guidanceScale = 5.0,
        double truncationTimestep = 0.5,
        double gradientClipNorm = 1.0)
    {
        _guidanceScale = guidanceScale;
        _truncationTimestep = truncationTimestep;
        _gradientClipNorm = gradientClipNorm;
    }

    /// <summary>
    /// Applies reward guidance to a noise prediction by incorporating the reward gradient.
    /// </summary>
    /// <param name="noisePrediction">Original noise prediction from the diffusion model.</param>
    /// <param name="rewardGradient">Gradient of the reward model with respect to the latent.</param>
    /// <param name="currentTimestepFraction">Current timestep as fraction of total (0.0 = start, 1.0 = end).</param>
    /// <returns>Modified noise prediction guided toward higher reward.</returns>
    public Vector<T> ApplyGuidance(
        Vector<T> noisePrediction,
        Vector<T> rewardGradient,
        double currentTimestepFraction)
    {
        // Skip guidance after truncation point for stability
        if (currentTimestepFraction > _truncationTimestep)
        {
            return noisePrediction;
        }

        // Clip gradient norm
        var clippedGradient = ClipGradientNorm(rewardGradient);

        // Guided prediction: eps - scale * grad_reward
        var result = new Vector<T>(noisePrediction.Length);
        var scale = NumOps.FromDouble(_guidanceScale);

        for (int i = 0; i < result.Length; i++)
        {
            var grad = i < clippedGradient.Length ? clippedGradient[i] : NumOps.Zero;
            var guidanceTerm = NumOps.Multiply(scale, grad);
            result[i] = NumOps.Subtract(noisePrediction[i], guidanceTerm);
        }

        return result;
    }

    /// <summary>
    /// Clips the gradient vector to have maximum L2 norm.
    /// </summary>
    /// <param name="gradient">The gradient vector to clip.</param>
    /// <returns>The clipped gradient vector.</returns>
    private Vector<T> ClipGradientNorm(Vector<T> gradient)
    {
        // Compute L2 norm
        double normSq = 0;
        for (int i = 0; i < gradient.Length; i++)
        {
            var val = NumOps.ToDouble(gradient[i]);
            normSq += val * val;
        }
        var norm = Math.Sqrt(normSq);

        if (norm <= _gradientClipNorm)
        {
            return gradient;
        }

        // Scale down
        var clipScale = NumOps.FromDouble(_gradientClipNorm / norm);
        var clipped = new Vector<T>(gradient.Length);
        for (int i = 0; i < gradient.Length; i++)
        {
            clipped[i] = NumOps.Multiply(clipScale, gradient[i]);
        }

        return clipped;
    }

    /// <summary>
    /// Gets the guidance scale.
    /// </summary>
    public double GuidanceScale => _guidanceScale;

    /// <summary>
    /// Gets the truncation timestep fraction.
    /// </summary>
    public double TruncationTimestep => _truncationTimestep;

    /// <summary>
    /// Gets the gradient clipping norm.
    /// </summary>
    public double GradientClipNorm => _gradientClipNorm;
}
