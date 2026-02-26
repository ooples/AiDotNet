using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Distribution Matching Distillation (DMD) trainer for single-step generation via distribution alignment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DMD distills a diffusion model into a single-step generator by matching the output distribution
/// of the generator to that of the teacher diffusion model. Combines a regression loss (MSE between
/// generator output and teacher's denoised output) with a distribution matching loss that ensures
/// the generator's output distribution matches the teacher's. DMD2 improves this by removing the
/// need for a GAN discriminator.
/// </para>
/// <para>
/// <b>For Beginners:</b> DMD trains a fast single-step generator by ensuring its outputs "look like"
/// the outputs of the slow multi-step teacher. It checks both individual quality (regression loss:
/// "does each image look right?") and overall diversity (distribution loss: "does the collection of
/// generated images look like what the teacher would produce?").
/// </para>
/// <para>
/// Reference: Yin et al., "One-step Diffusion with Distribution Matching Distillation", CVPR 2024;
/// Yin et al., "Improved Distribution Matching Distillation for Fast Image Synthesis", NeurIPS 2024
/// </para>
/// </remarks>
public class DistributionMatchingDistiller<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _teacher;
    private readonly double _regressionWeight;
    private readonly double _distributionWeight;
    private readonly double _guidanceScale;

    /// <summary>
    /// Gets the regression loss weight.
    /// </summary>
    public double RegressionWeight => _regressionWeight;

    /// <summary>
    /// Gets the distribution matching loss weight.
    /// </summary>
    public double DistributionWeight => _distributionWeight;

    /// <summary>
    /// Initializes a new DMD trainer.
    /// </summary>
    /// <param name="teacher">Pretrained teacher diffusion model.</param>
    /// <param name="regressionWeight">Weight for the regression (MSE) loss (default: 1.0).</param>
    /// <param name="distributionWeight">Weight for the distribution matching loss (default: 1.0).</param>
    /// <param name="guidanceScale">CFG scale for teacher's denoising (default: 7.5).</param>
    public DistributionMatchingDistiller(
        IDiffusionModel<T> teacher,
        double regressionWeight = 1.0,
        double distributionWeight = 1.0,
        double guidanceScale = 7.5)
    {
        _teacher = teacher;
        _regressionWeight = regressionWeight;
        _distributionWeight = distributionWeight;
        _guidanceScale = guidanceScale;
    }

    /// <summary>
    /// Computes the regression loss between generator output and teacher's denoised output.
    /// </summary>
    /// <param name="generatorOutput">Single-step generator prediction.</param>
    /// <param name="teacherDenoised">Teacher's multi-step denoised output.</param>
    /// <returns>MSE regression loss.</returns>
    public T ComputeRegressionLoss(Vector<T> generatorOutput, Vector<T> teacherDenoised)
    {
        var loss = NumOps.Zero;
        int len = Math.Min(generatorOutput.Length, teacherDenoised.Length);

        for (int i = 0; i < len; i++)
        {
            var diff = NumOps.Subtract(generatorOutput[i], teacherDenoised[i]);
            loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
        }

        return NumOps.Multiply(NumOps.FromDouble(_regressionWeight),
            NumOps.Divide(loss, NumOps.FromDouble(len)));
    }

    /// <summary>
    /// Computes the distribution matching loss using score differences.
    /// </summary>
    /// <param name="fakeScore">Score (gradient of log-density) from teacher on generated samples.</param>
    /// <param name="realScore">Score from teacher on real data samples.</param>
    /// <returns>Distribution matching loss.</returns>
    public T ComputeDistributionLoss(Vector<T> fakeScore, Vector<T> realScore)
    {
        var loss = NumOps.Zero;
        int len = Math.Min(fakeScore.Length, realScore.Length);

        for (int i = 0; i < len; i++)
        {
            var diff = NumOps.Subtract(fakeScore[i], realScore[i]);
            loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
        }

        return NumOps.Multiply(NumOps.FromDouble(_distributionWeight),
            NumOps.Divide(loss, NumOps.FromDouble(len)));
    }

    /// <summary>
    /// Computes the combined DMD loss (regression + distribution matching).
    /// </summary>
    /// <param name="generatorOutput">Generator's single-step prediction.</param>
    /// <param name="teacherDenoised">Teacher's denoised output.</param>
    /// <param name="fakeScore">Score on generated samples.</param>
    /// <param name="realScore">Score on real samples.</param>
    /// <returns>Total DMD loss.</returns>
    public T ComputeTotalLoss(
        Vector<T> generatorOutput, Vector<T> teacherDenoised,
        Vector<T> fakeScore, Vector<T> realScore)
    {
        var regLoss = ComputeRegressionLoss(generatorOutput, teacherDenoised);
        var distLoss = ComputeDistributionLoss(fakeScore, realScore);
        return NumOps.Add(regLoss, distLoss);
    }
}
