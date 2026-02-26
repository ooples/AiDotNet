using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Consistency Distillation Sampling (CSD) for 3D generation with consistency constraints.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CSD combines score distillation with consistency model training for faster and higher-quality
/// 3D generation. It enforces that the optimized 3D representation produces consistent results
/// across different noise levels, reducing the multi-step dependency and artifacts common in SDS.
/// Produces sharper, more coherent 3D objects with fewer Janus artifacts.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular SDS can produce 3D objects with problems like "two faces"
/// (Janus effect) or blurry textures. CSD fixes this by also requiring that the 3D model
/// looks consistent no matter how much noise is added — this extra constraint produces much
/// cleaner, more coherent 3D results.
/// </para>
/// <para>
/// Reference: Kim et al., "Consistency Trajectory Models: Learning Probability Flow ODE
/// Trajectory of Diffusion", ICLR 2024; adapted for 3D score distillation
/// </para>
/// </remarks>
public class ConsistencyDistillationSampling<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _diffusionModel;
    private readonly double _guidanceScale;
    private readonly double _consistencyWeight;

    /// <summary>
    /// Gets the guidance scale.
    /// </summary>
    public double GuidanceScale => _guidanceScale;

    /// <summary>
    /// Gets the consistency loss weight.
    /// </summary>
    public double ConsistencyWeight => _consistencyWeight;

    /// <summary>
    /// Initializes a new CSD instance.
    /// </summary>
    /// <param name="diffusionModel">Pretrained 2D diffusion model.</param>
    /// <param name="guidanceScale">CFG scale (default: 50.0).</param>
    /// <param name="consistencyWeight">Weight for consistency constraint (default: 1.0).</param>
    public ConsistencyDistillationSampling(
        IDiffusionModel<T> diffusionModel,
        double guidanceScale = 50.0,
        double consistencyWeight = 1.0)
    {
        _diffusionModel = diffusionModel;
        _guidanceScale = guidanceScale;
        _consistencyWeight = consistencyWeight;
    }

    /// <summary>
    /// Computes the combined SDS + consistency gradient.
    /// </summary>
    /// <param name="predictedNoise">Model's noise prediction.</param>
    /// <param name="addedNoise">Noise that was added.</param>
    /// <param name="consistencyTarget">Target from adjacent timestep for consistency.</param>
    /// <param name="currentPrediction">Current model prediction for consistency loss.</param>
    /// <param name="timestepWeight">Timestep-dependent weight.</param>
    /// <returns>Combined gradient.</returns>
    public Vector<T> ComputeGradient(
        Vector<T> predictedNoise, Vector<T> addedNoise,
        Vector<T> consistencyTarget, Vector<T> currentPrediction,
        double timestepWeight)
    {
        var gradient = new Vector<T>(predictedNoise.Length);
        var sdsWeight = NumOps.FromDouble(timestepWeight);
        var consWeight = NumOps.FromDouble(_consistencyWeight);

        for (int i = 0; i < gradient.Length; i++)
        {
            // SDS component
            var sdsDiff = NumOps.Subtract(predictedNoise[i],
                i < addedNoise.Length ? addedNoise[i] : NumOps.Zero);
            var sdsGrad = NumOps.Multiply(sdsWeight, sdsDiff);

            // Consistency component
            var consDiff = NumOps.Subtract(
                i < currentPrediction.Length ? currentPrediction[i] : NumOps.Zero,
                i < consistencyTarget.Length ? consistencyTarget[i] : NumOps.Zero);
            var consGrad = NumOps.Multiply(consWeight, consDiff);

            gradient[i] = NumOps.Add(sdsGrad, consGrad);
        }

        return gradient;
    }
}
