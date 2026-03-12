using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Score Distillation Sampling (SDS) for text-to-3D and generator optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SDS computes gradients from a pretrained 2D diffusion model to optimize a 3D representation
/// (NeRF, mesh, etc.). It adds noise to rendered views, asks the diffusion model to predict
/// the noise, and uses the difference between added and predicted noise as a gradient signal.
/// This enables text-to-3D generation without any 3D training data.
/// </para>
/// <para>
/// <b>For Beginners:</b> SDS uses a 2D image generator as a "critic" for 3D models. It renders
/// the 3D model from a random angle, asks the 2D model "how could this image be improved?",
/// and uses that feedback to update the 3D model. Repeating this from many angles creates
/// a 3D object that looks good from every direction.
/// </para>
/// <para>
/// Reference: Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion", ICLR 2023
/// </para>
/// </remarks>
public class ScoreDistillationSampling<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _diffusionModel;
    private readonly double _guidanceScale;
    private readonly double _minTimestep;
    private readonly double _maxTimestep;
    private readonly double _gradientScale;

    /// <summary>
    /// Gets the classifier-free guidance scale.
    /// </summary>
    public double GuidanceScale => _guidanceScale;

    /// <summary>
    /// Initializes a new SDS instance.
    /// </summary>
    /// <param name="diffusionModel">Pretrained 2D diffusion model providing score estimates.</param>
    /// <param name="guidanceScale">CFG scale for score computation (default: 100.0).</param>
    /// <param name="minTimestep">Minimum timestep fraction for noise (default: 0.02).</param>
    /// <param name="maxTimestep">Maximum timestep fraction for noise (default: 0.98).</param>
    /// <param name="gradientScale">Global gradient scaling factor (default: 1.0).</param>
    public ScoreDistillationSampling(
        IDiffusionModel<T> diffusionModel,
        double guidanceScale = 100.0,
        double minTimestep = 0.02,
        double maxTimestep = 0.98,
        double gradientScale = 1.0)
    {
        _diffusionModel = diffusionModel;
        _guidanceScale = guidanceScale;
        _minTimestep = minTimestep;
        _maxTimestep = maxTimestep;
        _gradientScale = gradientScale;
    }

    /// <summary>
    /// Computes the SDS gradient for a rendered view.
    /// </summary>
    /// <param name="predictedNoise">Diffusion model's noise prediction at timestep t.</param>
    /// <param name="addedNoise">The noise that was actually added to the rendered view.</param>
    /// <param name="timestepWeight">Weighting factor w(t) for this timestep.</param>
    /// <returns>SDS gradient: w(t) * (epsilon_pred - epsilon_added).</returns>
    public Vector<T> ComputeGradient(Vector<T> predictedNoise, Vector<T> addedNoise, double timestepWeight)
    {
        var gradient = new Vector<T>(predictedNoise.Length);
        var weight = NumOps.FromDouble(timestepWeight * _gradientScale);

        for (int i = 0; i < gradient.Length; i++)
        {
            var diff = NumOps.Subtract(
                predictedNoise[i],
                i < addedNoise.Length ? addedNoise[i] : NumOps.Zero);
            gradient[i] = NumOps.Multiply(weight, diff);
        }

        return gradient;
    }

    /// <summary>
    /// Samples a random timestep within the configured range.
    /// </summary>
    /// <param name="numTimesteps">Total number of timesteps in the schedule.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>Sampled timestep.</returns>
    public int SampleTimestep(int numTimesteps, Random random)
    {
        int minT = (int)(_minTimestep * numTimesteps);
        int maxT = (int)(_maxTimestep * numTimesteps);
        return random.Next(minT, maxT + 1);
    }
}
