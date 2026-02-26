using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Alignment;

/// <summary>
/// Direct Preference Optimization (DPO) adapted for diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Diffusion-DPO adapts the DPO framework from language models to diffusion models.
/// Given pairs of preferred and dispreferred images, it directly optimizes the diffusion
/// model's policy to prefer generating the preferred outputs without needing a separate
/// reward model.
/// </para>
/// <para>
/// <b>For Beginners:</b> DPO teaches the model by showing it pairs of images: one that
/// humans preferred and one they didn't. Instead of training a separate "judge" model first
/// (like RLHF does), DPO directly adjusts the diffusion model to produce more of what
/// humans like. It's simpler and often more stable than RLHF for alignment.
/// </para>
/// <para>
/// Reference: Wallace et al., "Diffusion Model Alignment Using Direct Preference Optimization", CVPR 2024
/// </para>
/// </remarks>
public class DiffusionDPO<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _model;
    private readonly IDiffusionModel<T> _referenceModel;
    private readonly double _beta;
    private readonly double _labelSmoothing;

    /// <summary>
    /// Initializes a new Diffusion-DPO trainer.
    /// </summary>
    /// <param name="model">The diffusion model to align.</param>
    /// <param name="referenceModel">Frozen reference model for KL regularization.</param>
    /// <param name="beta">Temperature parameter controlling deviation from reference (default: 5000.0).</param>
    /// <param name="labelSmoothing">Label smoothing factor for robust training (default: 0.0).</param>
    public DiffusionDPO(
        IDiffusionModel<T> model,
        IDiffusionModel<T> referenceModel,
        double beta = 5000.0,
        double labelSmoothing = 0.0)
    {
        _model = model;
        _referenceModel = referenceModel;
        _beta = beta;
        _labelSmoothing = labelSmoothing;
    }

    /// <summary>
    /// Computes the DPO loss given preferred and dispreferred noise predictions.
    /// </summary>
    /// <param name="preferredModelLogProb">Log-probability of preferred sample under current model.</param>
    /// <param name="disPreferredModelLogProb">Log-probability of dispreferred sample under current model.</param>
    /// <param name="preferredRefLogProb">Log-probability of preferred sample under reference model.</param>
    /// <param name="disPreferredRefLogProb">Log-probability of dispreferred sample under reference model.</param>
    /// <returns>The DPO loss value.</returns>
    public T ComputeDPOLoss(
        T preferredModelLogProb,
        T disPreferredModelLogProb,
        T preferredRefLogProb,
        T disPreferredRefLogProb)
    {
        // DPO loss: -log(sigmoid(beta * (log_pi(y_w) - log_pi(y_l) - log_ref(y_w) + log_ref(y_l))))
        var prefDiff = NumOps.Subtract(preferredModelLogProb, preferredRefLogProb);
        var disPrefDiff = NumOps.Subtract(disPreferredModelLogProb, disPreferredRefLogProb);
        var logitDiff = NumOps.Subtract(prefDiff, disPrefDiff);
        var scaled = NumOps.Multiply(NumOps.FromDouble(_beta), logitDiff);

        // Sigmoid: 1 / (1 + exp(-x))
        var negScaled = NumOps.Negate(scaled);
        var expNeg = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(negScaled)));
        var sigmoid = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));

        // Apply label smoothing
        if (_labelSmoothing > 0)
        {
            var smooth = NumOps.FromDouble(_labelSmoothing);
            var oneMinusSmooth = NumOps.FromDouble(1.0 - _labelSmoothing);
            sigmoid = NumOps.Add(
                NumOps.Multiply(oneMinusSmooth, sigmoid),
                NumOps.Multiply(smooth, NumOps.FromDouble(0.5)));
        }

        // -log(sigmoid)
        var logSigmoid = NumOps.FromDouble(Math.Log(Math.Max(NumOps.ToDouble(sigmoid), 1e-10)));
        return NumOps.Negate(logSigmoid);
    }

    /// <summary>
    /// Computes the implicit reward for a sample.
    /// </summary>
    /// <param name="modelLogProb">Log-probability under the current model.</param>
    /// <param name="refLogProb">Log-probability under the reference model.</param>
    /// <returns>The implicit reward value.</returns>
    public T ComputeImplicitReward(T modelLogProb, T refLogProb)
    {
        // r(x) = beta * (log_pi(x) - log_ref(x))
        var diff = NumOps.Subtract(modelLogProb, refLogProb);
        return NumOps.Multiply(NumOps.FromDouble(_beta), diff);
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
    /// Gets the beta temperature parameter.
    /// </summary>
    public double Beta => _beta;

    /// <summary>
    /// Gets the label smoothing factor.
    /// </summary>
    public double LabelSmoothing => _labelSmoothing;
}
