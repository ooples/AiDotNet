using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Alignment;

/// <summary>
/// Refiner stage for late-stage noise-add-then-denoise detail improvement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The refiner stage takes a nearly-finished generated image, adds a small amount of noise
/// back, and re-denoises using a specialized refiner model (or the same model with different
/// conditioning). This late-stage refinement enhances fine details, textures, and overall
/// coherence. Used in SDXL's two-stage pipeline and other multi-stage generation approaches.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine you've painted a picture but want to add more detail. The
/// refiner slightly blurs the image (adds noise), then carefully sharpens it back. This
/// re-processing enhances fine details like skin texture, fabric patterns, and small objects
/// that the main generation pass might have missed.
/// </para>
/// <para>
/// Reference: Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution
/// Image Synthesis", ICLR 2024 (refiner stage)
/// </para>
/// </remarks>
public class RefinerStage<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _refinerModel;
    private readonly double _strength;
    private readonly double _guidanceScale;
    private readonly int _refinerSteps;

    /// <summary>
    /// Gets the noise strength for refinement (0.0 = no change, 1.0 = full re-generation).
    /// </summary>
    public double Strength => _strength;

    /// <summary>
    /// Gets the classifier-free guidance scale for the refiner.
    /// </summary>
    public double GuidanceScale => _guidanceScale;

    /// <summary>
    /// Gets the number of denoising steps the refiner uses.
    /// </summary>
    public int RefinerSteps => _refinerSteps;

    /// <summary>
    /// Initializes a new refiner stage.
    /// </summary>
    /// <param name="refinerModel">The refiner diffusion model (can be same model or specialized refiner).</param>
    /// <param name="strength">Noise strength — fraction of schedule to re-denoise (default: 0.3).</param>
    /// <param name="guidanceScale">CFG scale for the refiner (default: 7.5).</param>
    /// <param name="refinerSteps">Number of denoising steps for refinement (default: 20).</param>
    public RefinerStage(
        IDiffusionModel<T> refinerModel,
        double strength = 0.3,
        double guidanceScale = 7.5,
        int refinerSteps = 20)
    {
        _refinerModel = refinerModel;
        _strength = Math.Min(1.0, Math.Max(0.0, strength));
        _guidanceScale = guidanceScale;
        _refinerSteps = refinerSteps;
    }

    /// <summary>
    /// Adds noise to a latent at the specified strength level.
    /// </summary>
    /// <param name="cleanLatent">The base generation's latent output.</param>
    /// <param name="noise">Random noise to add.</param>
    /// <returns>Noised latent ready for refinement denoising.</returns>
    public Vector<T> AddNoiseForRefinement(Vector<T> cleanLatent, Vector<T> noise)
    {
        var strengthT = NumOps.FromDouble(_strength);
        var oneMinusStrength = NumOps.Subtract(NumOps.One, strengthT);
        var result = new Vector<T>(cleanLatent.Length);

        for (int i = 0; i < cleanLatent.Length; i++)
        {
            var signal = NumOps.Multiply(oneMinusStrength, cleanLatent[i]);
            var noiseComponent = NumOps.Multiply(strengthT,
                i < noise.Length ? noise[i] : NumOps.Zero);
            result[i] = NumOps.Add(signal, noiseComponent);
        }

        return result;
    }

    /// <summary>
    /// Computes the starting timestep for the refiner based on strength.
    /// </summary>
    /// <param name="totalTimesteps">Total timesteps in the noise schedule.</param>
    /// <returns>Starting timestep for the refiner (lower strength = later start = less change).</returns>
    public int GetStartTimestep(int totalTimesteps)
    {
        return (int)(totalTimesteps * _strength);
    }

    /// <summary>
    /// Computes the refinement quality improvement estimate.
    /// </summary>
    /// <param name="beforeRefinement">Latent before refinement.</param>
    /// <param name="afterRefinement">Latent after refinement.</param>
    /// <returns>L2 distance indicating amount of change (lower = less change).</returns>
    public T ComputeRefinementDelta(Vector<T> beforeRefinement, Vector<T> afterRefinement)
    {
        var sum = NumOps.Zero;
        int len = Math.Min(beforeRefinement.Length, afterRefinement.Length);
        for (int i = 0; i < len; i++)
        {
            var diff = NumOps.Subtract(beforeRefinement[i], afterRefinement[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return NumOps.Sqrt(NumOps.Divide(sum, NumOps.FromDouble(len)));
    }
}
