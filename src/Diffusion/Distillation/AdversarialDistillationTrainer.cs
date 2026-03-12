using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Trainer for Adversarial Diffusion Distillation (ADD) as used in SD/SDXL Turbo.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ADD combines a diffusion denoising loss with an adversarial loss from a pretrained
/// discriminator. The student generates a sample in 1-4 steps, a discriminator evaluates
/// realism, and the diffusion teacher provides structure guidance. This dual-loss approach
/// produces the highest quality single-step generators.
/// </para>
/// <para>
/// <b>For Beginners:</b> ADD uses two teachers: (1) a diffusion model that ensures the
/// student captures the right image structure, and (2) a discriminator that ensures the
/// output looks realistic. This dual supervision produces remarkably good single-step
/// images — the student learns both "what" to generate and "how real" it should look.
/// </para>
/// <para>
/// Reference: Sauer et al., "Adversarial Diffusion Distillation", 2023
/// </para>
/// </remarks>
public class AdversarialDistillationTrainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _teacher;
    private readonly double _adversarialWeight;
    private readonly double _diffusionWeight;

    /// <summary>
    /// Initializes a new adversarial distillation trainer.
    /// </summary>
    /// <param name="teacher">Pretrained teacher diffusion model.</param>
    /// <param name="adversarialWeight">Weight for adversarial loss (default: 1.0).</param>
    /// <param name="diffusionWeight">Weight for diffusion loss (default: 2.5).</param>
    public AdversarialDistillationTrainer(
        IDiffusionModel<T> teacher,
        double adversarialWeight = 1.0,
        double diffusionWeight = 2.5)
    {
        _teacher = teacher;
        _adversarialWeight = adversarialWeight;
        _diffusionWeight = diffusionWeight;
    }

    /// <summary>
    /// Computes the combined ADD loss.
    /// </summary>
    /// <param name="studentOutput">Student model output (generated image latent).</param>
    /// <param name="teacherOutput">Teacher model denoised output.</param>
    /// <param name="discriminatorScore">Discriminator realness score for student output (0-1).</param>
    /// <returns>Combined adversarial + diffusion loss.</returns>
    public T ComputeADDLoss(Vector<T> studentOutput, Vector<T> teacherOutput, T discriminatorScore)
    {
        // Diffusion loss: L2 between student and teacher outputs
        var diffLoss = NumOps.Zero;
        for (int i = 0; i < studentOutput.Length && i < teacherOutput.Length; i++)
        {
            var diff = NumOps.Subtract(studentOutput[i], teacherOutput[i]);
            diffLoss = NumOps.Add(diffLoss, NumOps.Multiply(diff, diff));
        }
        diffLoss = NumOps.Divide(diffLoss, NumOps.FromDouble(studentOutput.Length));

        // Adversarial loss: -log(D(student_output))
        var advLoss = NumOps.Negate(NumOps.Log(NumOps.Add(discriminatorScore, NumOps.FromDouble(1e-8))));

        // Combined loss
        var weightedDiff = NumOps.Multiply(NumOps.FromDouble(_diffusionWeight), diffLoss);
        var weightedAdv = NumOps.Multiply(NumOps.FromDouble(_adversarialWeight), advLoss);

        return NumOps.Add(weightedDiff, weightedAdv);
    }

    /// <summary>
    /// Gets the teacher model.
    /// </summary>
    public IDiffusionModel<T> Teacher => _teacher;

    /// <summary>
    /// Gets the adversarial loss weight.
    /// </summary>
    public double AdversarialWeight => _adversarialWeight;

    /// <summary>
    /// Gets the diffusion loss weight.
    /// </summary>
    public double DiffusionWeight => _diffusionWeight;
}
