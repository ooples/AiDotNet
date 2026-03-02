using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Alignment;

/// <summary>
/// Reinforcement Learning from Human Feedback (RLHF) adapted for diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Diffusion-RLHF uses a trained reward model to provide feedback signals that guide
/// the diffusion model toward generating outputs aligned with human preferences. The
/// reward model is trained on human preference data, then used to fine-tune the diffusion
/// model via policy gradient methods with KL regularization against a reference model.
/// </para>
/// <para>
/// <b>For Beginners:</b> RLHF is a two-step process: first, train a "reward model" that
/// learns what humans prefer (like a judge). Then use that judge to give feedback to the
/// diffusion model during training. The diffusion model learns to generate images the
/// judge would rate highly. KL regularization prevents the model from "cheating" by
/// exploiting the judge's blind spots.
/// </para>
/// <para>
/// Reference: Black et al., "Training Diffusion Models with Reinforcement Learning", 2024
/// </para>
/// </remarks>
public class DiffusionRLHF<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _model;
    private readonly IDiffusionModel<T> _referenceModel;
    private readonly double _klWeight;
    private readonly double _rewardScale;
    private readonly double _clipRange;

    /// <summary>
    /// Initializes a new Diffusion-RLHF trainer.
    /// </summary>
    /// <param name="model">The diffusion model to align.</param>
    /// <param name="referenceModel">Frozen reference model for KL divergence constraint.</param>
    /// <param name="klWeight">Weight for KL divergence penalty (default: 0.01).</param>
    /// <param name="rewardScale">Scaling factor for reward signals (default: 1.0).</param>
    /// <param name="clipRange">PPO-style clipping range for policy updates (default: 0.2).</param>
    public DiffusionRLHF(
        IDiffusionModel<T> model,
        IDiffusionModel<T> referenceModel,
        double klWeight = 0.01,
        double rewardScale = 1.0,
        double clipRange = 0.2)
    {
        _model = model;
        _referenceModel = referenceModel;
        _klWeight = klWeight;
        _rewardScale = rewardScale;
        _clipRange = clipRange;
    }

    /// <summary>
    /// Computes the RLHF objective: reward minus KL penalty.
    /// </summary>
    /// <param name="reward">Reward from the reward model for a generated sample.</param>
    /// <param name="modelLogProb">Log-probability of the sample under the current model.</param>
    /// <param name="refLogProb">Log-probability of the sample under the reference model.</param>
    /// <returns>The RLHF objective value (higher is better).</returns>
    public T ComputeRLHFObjective(T reward, T modelLogProb, T refLogProb)
    {
        // Objective: reward_scale * reward - kl_weight * KL(pi || ref)
        // KL approximation: log_pi - log_ref
        var scaledReward = NumOps.Multiply(NumOps.FromDouble(_rewardScale), reward);
        var klDiv = NumOps.Subtract(modelLogProb, refLogProb);
        var klPenalty = NumOps.Multiply(NumOps.FromDouble(_klWeight), klDiv);

        return NumOps.Subtract(scaledReward, klPenalty);
    }

    /// <summary>
    /// Computes the PPO-clipped policy gradient loss for diffusion denoising steps.
    /// </summary>
    /// <param name="advantage">Advantage estimate (reward - baseline).</param>
    /// <param name="ratio">Probability ratio pi(a|s) / pi_old(a|s).</param>
    /// <returns>The clipped surrogate loss (negated for minimization).</returns>
    public T ComputePPOLoss(T advantage, T ratio)
    {
        // L_CLIP = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        var unclipped = NumOps.Multiply(ratio, advantage);

        var ratioVal = NumOps.ToDouble(ratio);
        var clippedRatioVal = Math.Max(1.0 - _clipRange, Math.Min(ratioVal, 1.0 + _clipRange));
        var clipped = NumOps.Multiply(NumOps.FromDouble(clippedRatioVal), advantage);

        var unclippedVal = NumOps.ToDouble(unclipped);
        var clippedVal = NumOps.ToDouble(clipped);
        var minVal = Math.Min(unclippedVal, clippedVal);

        // Negate for minimization (we want to maximize the objective)
        return NumOps.Negate(NumOps.FromDouble(minVal));
    }

    /// <summary>
    /// Gets the model being aligned.
    /// </summary>
    public IDiffusionModel<T> Model => _model;

    /// <summary>
    /// Gets the frozen reference model.
    /// </summary>
    public IDiffusionModel<T> ReferenceModel => _referenceModel;

    /// <summary>
    /// Gets the KL divergence weight.
    /// </summary>
    public double KLWeight => _klWeight;

    /// <summary>
    /// Gets the reward scaling factor.
    /// </summary>
    public double RewardScale => _rewardScale;

    /// <summary>
    /// Gets the PPO clipping range.
    /// </summary>
    public double ClipRange => _clipRange;
}
