using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Reward-weighted Score Distillation Sampling (RewardSDS) for preference-aligned 3D generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RewardSDS augments standard SDS gradients with reward model feedback. Instead of using
/// only the diffusion model's score, it weights the gradient by a reward signal that measures
/// alignment with human preferences (aesthetics, text-image correspondence, etc.). This
/// produces 3D objects that not only look realistic but also match desired aesthetic qualities.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard SDS creates realistic-looking 3D objects, but they may not
/// match what people actually want aesthetically. RewardSDS adds a "quality judge" (reward model)
/// that steers the generation toward more visually appealing results. It's like having both a
/// realism expert and an art critic guiding the 3D artist simultaneously.
/// </para>
/// <para>
/// Reference: Adapted from DreamReward (Ye et al., 2024) combining human preference alignment
/// with score distillation for text-to-3D generation
/// </para>
/// </remarks>
public class RewardScoreDistillation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _diffusionModel;
    private readonly double _guidanceScale;
    private readonly double _rewardWeight;

    /// <summary>
    /// Gets the guidance scale for score distillation.
    /// </summary>
    public double GuidanceScale => _guidanceScale;

    /// <summary>
    /// Gets the reward signal weight.
    /// </summary>
    public double RewardWeight => _rewardWeight;

    /// <summary>
    /// Initializes a new RewardSDS instance.
    /// </summary>
    /// <param name="diffusionModel">Pretrained 2D diffusion model.</param>
    /// <param name="guidanceScale">CFG scale for score computation (default: 100.0).</param>
    /// <param name="rewardWeight">Weight for the reward gradient component (default: 10.0).</param>
    public RewardScoreDistillation(
        IDiffusionModel<T> diffusionModel,
        double guidanceScale = 100.0,
        double rewardWeight = 10.0)
    {
        _diffusionModel = diffusionModel;
        _guidanceScale = guidanceScale;
        _rewardWeight = rewardWeight;
    }

    /// <summary>
    /// Computes the reward-weighted SDS gradient.
    /// </summary>
    /// <param name="predictedNoise">Diffusion model's noise prediction.</param>
    /// <param name="addedNoise">Noise that was added.</param>
    /// <param name="rewardGradient">Gradient from the reward model.</param>
    /// <param name="timestepWeight">Timestep-dependent weight.</param>
    /// <returns>Combined SDS + reward gradient.</returns>
    public Vector<T> ComputeGradient(
        Vector<T> predictedNoise, Vector<T> addedNoise,
        Vector<T> rewardGradient, double timestepWeight)
    {
        var gradient = new Vector<T>(predictedNoise.Length);
        var sdsWeight = NumOps.FromDouble(timestepWeight);
        var rwdWeight = NumOps.FromDouble(_rewardWeight);

        for (int i = 0; i < gradient.Length; i++)
        {
            // SDS component
            var sdsDiff = NumOps.Subtract(predictedNoise[i],
                i < addedNoise.Length ? addedNoise[i] : NumOps.Zero);
            var sdsGrad = NumOps.Multiply(sdsWeight, sdsDiff);

            // Reward component
            var rwdGrad = NumOps.Multiply(rwdWeight,
                i < rewardGradient.Length ? rewardGradient[i] : NumOps.Zero);

            gradient[i] = NumOps.Add(sdsGrad, rwdGrad);
        }

        return gradient;
    }
}
