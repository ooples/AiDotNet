using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Variational Score Distillation (VSD) for high-fidelity text-to-3D generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VSD improves upon SDS by reducing the over-saturation and over-smoothing artifacts. It
/// maintains a separate LoRA-adapted diffusion model trained on rendered views and uses the
/// difference between the pretrained model's score and the LoRA model's score as the gradient.
/// This effectively performs variational inference in the diffusion model's data space.
/// </para>
/// <para>
/// <b>For Beginners:</b> SDS sometimes produces washed-out or over-saturated 3D models. VSD
/// fixes this by training a second model that learns what the 3D model's renders look like.
/// By comparing the "ideal" score (from the pretrained model) with the "current" score (from
/// the adapted model), it gives more precise, less noisy feedback for 3D optimization.
/// </para>
/// <para>
/// Reference: Wang et al., "ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation
/// with Variational Score Distillation", NeurIPS 2023
/// </para>
/// </remarks>
public class VariationalScoreDistillation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _pretrainedModel;
    private readonly double _guidanceScale;
    private readonly double _loraLearningRate;
    private readonly int _loraRank;

    /// <summary>
    /// Gets the guidance scale for the pretrained model.
    /// </summary>
    public double GuidanceScale => _guidanceScale;

    /// <summary>
    /// Gets the LoRA rank for the particle model.
    /// </summary>
    public int LoRARank => _loraRank;

    /// <summary>
    /// Initializes a new VSD instance.
    /// </summary>
    /// <param name="pretrainedModel">Pretrained 2D diffusion model.</param>
    /// <param name="guidanceScale">CFG scale for pretrained model (default: 7.5).</param>
    /// <param name="loraLearningRate">Learning rate for the LoRA particle model (default: 1e-4).</param>
    /// <param name="loraRank">LoRA rank for the particle model adaptation (default: 4).</param>
    public VariationalScoreDistillation(
        IDiffusionModel<T> pretrainedModel,
        double guidanceScale = 7.5,
        double loraLearningRate = 1e-4,
        int loraRank = 4)
    {
        _pretrainedModel = pretrainedModel;
        _guidanceScale = guidanceScale;
        _loraLearningRate = loraLearningRate;
        _loraRank = loraRank;
    }

    /// <summary>
    /// Computes the VSD gradient using both pretrained and particle model scores.
    /// </summary>
    /// <param name="pretrainedScore">Noise prediction from pretrained model.</param>
    /// <param name="particleScore">Noise prediction from LoRA particle model.</param>
    /// <param name="timestepWeight">Weighting factor for this timestep.</param>
    /// <returns>VSD gradient: w(t) * (pretrained_score - particle_score).</returns>
    public Vector<T> ComputeGradient(Vector<T> pretrainedScore, Vector<T> particleScore, double timestepWeight)
    {
        var gradient = new Vector<T>(pretrainedScore.Length);
        var weight = NumOps.FromDouble(timestepWeight);

        for (int i = 0; i < gradient.Length; i++)
        {
            var diff = NumOps.Subtract(
                pretrainedScore[i],
                i < particleScore.Length ? particleScore[i] : NumOps.Zero);
            gradient[i] = NumOps.Multiply(weight, diff);
        }

        return gradient;
    }

    /// <summary>
    /// Computes the LoRA training loss for the particle model.
    /// </summary>
    /// <param name="particlePrediction">Particle model's noise prediction.</param>
    /// <param name="targetNoise">Actual noise added to the rendered view.</param>
    /// <returns>MSE loss for LoRA training.</returns>
    public T ComputeParticleLoss(Vector<T> particlePrediction, Vector<T> targetNoise)
    {
        var loss = NumOps.Zero;
        int len = Math.Min(particlePrediction.Length, targetNoise.Length);
        for (int i = 0; i < len; i++)
        {
            var diff = NumOps.Subtract(particlePrediction[i], targetNoise[i]);
            loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
        }
        return NumOps.Divide(loss, NumOps.FromDouble(len));
    }
}
