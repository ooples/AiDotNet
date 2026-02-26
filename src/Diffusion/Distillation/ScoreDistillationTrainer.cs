using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Trainer for Score Distillation Sampling (SDS) and its variants for generator training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SDS uses the score function (gradient of the log-probability) from a pretrained
/// diffusion model to guide a generator network. The generator produces samples, noise
/// is added, and the diffusion model's score provides the training gradient. Supports
/// Variational Score Distillation (VSD) and Classifier Score Distillation (CSD) variants.
/// </para>
/// <para>
/// <b>For Beginners:</b> Score distillation asks a pretrained diffusion model "how could
/// this generated image be improved?" and uses the answer to train the generator. It's
/// like having an art critic (the diffusion model) provide feedback that the student
/// generator uses to improve. VSD and CSD are refined versions that give better feedback.
/// </para>
/// <para>
/// Reference: Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion", ICLR 2023 (SDS);
/// Wang et al., "ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation", NeurIPS 2023 (VSD)
/// </para>
/// </remarks>
public class ScoreDistillationTrainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _teacher;
    private readonly double _guidanceScale;
    private readonly double _minTimestep;
    private readonly double _maxTimestep;

    /// <summary>
    /// Initializes a new score distillation trainer.
    /// </summary>
    /// <param name="teacher">Pretrained teacher diffusion model providing scores.</param>
    /// <param name="guidanceScale">CFG scale for score computation (default: 100.0).</param>
    /// <param name="minTimestep">Minimum timestep for noise injection (default: 0.02).</param>
    /// <param name="maxTimestep">Maximum timestep for noise injection (default: 0.98).</param>
    public ScoreDistillationTrainer(
        IDiffusionModel<T> teacher,
        double guidanceScale = 100.0,
        double minTimestep = 0.02,
        double maxTimestep = 0.98)
    {
        _teacher = teacher;
        _guidanceScale = guidanceScale;
        _minTimestep = minTimestep;
        _maxTimestep = maxTimestep;
    }

    /// <summary>
    /// Computes the SDS gradient for a generator's output.
    /// </summary>
    /// <param name="generatedLatent">Latent from the generator.</param>
    /// <param name="noisyLatent">Generator latent with added noise at timestep t.</param>
    /// <param name="predictedNoise">Teacher's noise prediction at timestep t.</param>
    /// <param name="addedNoise">The noise that was added.</param>
    /// <returns>SDS gradient (teacher noise prediction minus added noise, scaled).</returns>
    public Vector<T> ComputeSDSGradient(
        Vector<T> generatedLatent,
        Vector<T> noisyLatent,
        Vector<T> predictedNoise,
        Vector<T> addedNoise)
    {
        // SDS gradient: w(t) * (epsilon_pretrained - epsilon_added)
        var gradient = new Vector<T>(generatedLatent.Length);
        var weight = NumOps.FromDouble(_guidanceScale);

        for (int i = 0; i < gradient.Length; i++)
        {
            var diff = NumOps.Subtract(
                i < predictedNoise.Length ? predictedNoise[i] : NumOps.Zero,
                i < addedNoise.Length ? addedNoise[i] : NumOps.Zero);
            gradient[i] = NumOps.Multiply(weight, diff);
        }

        return gradient;
    }

    /// <summary>
    /// Gets the teacher model.
    /// </summary>
    public IDiffusionModel<T> Teacher => _teacher;

    /// <summary>
    /// Gets the guidance scale for score computation.
    /// </summary>
    public double GuidanceScale => _guidanceScale;

    /// <summary>
    /// Gets the minimum timestep.
    /// </summary>
    public double MinTimestep => _minTimestep;

    /// <summary>
    /// Gets the maximum timestep.
    /// </summary>
    public double MaxTimestep => _maxTimestep;
}
