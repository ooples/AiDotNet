using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Denoised Score Distillation (DSD) for artifact-free 3D generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DSD addresses the mode-seeking bias of SDS by using a fully denoised reference image
/// instead of comparing noisy predictions. At each optimization step, it runs the full
/// diffusion denoising chain to get a clean reference, then uses the difference between
/// this reference and the rendered view as the gradient. While more expensive per step,
/// DSD produces significantly sharper and more diverse 3D results.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular SDS compares noise predictions, which can be imprecise.
/// DSD takes a different approach — it fully generates a clean reference image and compares
/// it to what the 3D model renders. This is like having the teacher paint a complete example
/// for every critique, giving much clearer feedback to the 3D model.
/// </para>
/// <para>
/// Reference: Hertz et al., "Denoised Score Distillation", 2024
/// </para>
/// </remarks>
public class DenoisedScoreDistillation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _diffusionModel;
    private readonly double _guidanceScale;
    private readonly int _denoisingSteps;

    /// <summary>
    /// Gets the guidance scale.
    /// </summary>
    public double GuidanceScale => _guidanceScale;

    /// <summary>
    /// Gets the number of full denoising steps for reference generation.
    /// </summary>
    public int DenoisingSteps => _denoisingSteps;

    /// <summary>
    /// Initializes a new DSD instance.
    /// </summary>
    /// <param name="diffusionModel">Pretrained 2D diffusion model.</param>
    /// <param name="guidanceScale">CFG scale for reference generation (default: 7.5).</param>
    /// <param name="denoisingSteps">Steps for generating clean reference (default: 50).</param>
    public DenoisedScoreDistillation(
        IDiffusionModel<T> diffusionModel,
        double guidanceScale = 7.5,
        int denoisingSteps = 50)
    {
        _diffusionModel = diffusionModel;
        _guidanceScale = guidanceScale;
        _denoisingSteps = denoisingSteps;
    }

    /// <summary>
    /// Computes the DSD gradient from a fully denoised reference.
    /// </summary>
    /// <param name="denoisedReference">Fully denoised clean reference image latent.</param>
    /// <param name="renderedView">Current rendered view from the 3D model.</param>
    /// <param name="weight">Gradient weight (default: 1.0).</param>
    /// <returns>DSD gradient.</returns>
    public Vector<T> ComputeGradient(Vector<T> denoisedReference, Vector<T> renderedView, double weight = 1.0)
    {
        var gradient = new Vector<T>(renderedView.Length);
        var w = NumOps.FromDouble(weight);

        for (int i = 0; i < gradient.Length; i++)
        {
            var target = i < denoisedReference.Length ? denoisedReference[i] : NumOps.Zero;
            var diff = NumOps.Subtract(target, renderedView[i]);
            gradient[i] = NumOps.Multiply(w, diff);
        }

        return gradient;
    }
}
