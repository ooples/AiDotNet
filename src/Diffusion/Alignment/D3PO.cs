using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Alignment;

/// <summary>
/// Direct Preference for Denoising Diffusion Policy Optimization (D3PO).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// D3PO extends DPO to operate directly on the denoising process of diffusion models.
/// Rather than comparing final generated images, it applies preference optimization at each
/// denoising step, treating the denoising trajectory as a Markov Decision Process. This
/// enables finer-grained alignment by optimizing the generation policy at the step level.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular DPO teaches a model by comparing two finished images. D3PO
/// goes deeper — it compares images at every step of the generation process. This is like
/// having a coach who gives feedback on every brush stroke rather than just rating the final
/// painting, leading to more precise control over image quality.
/// </para>
/// <para>
/// Reference: Yang et al., "Using Human Feedback to Fine-tune Diffusion Models without Any
/// Reward Model", CVPR 2024
/// </para>
/// </remarks>
public class D3PO<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _model;
    private readonly IDiffusionModel<T> _referenceModel;
    private readonly double _beta;
    private readonly double _stepWeight;

    /// <summary>
    /// Gets the temperature parameter beta.
    /// </summary>
    public double Beta => _beta;

    /// <summary>
    /// Gets the per-step weighting factor.
    /// </summary>
    public double StepWeight => _stepWeight;

    /// <summary>
    /// Initializes a new D3PO trainer.
    /// </summary>
    /// <param name="model">The diffusion model to align.</param>
    /// <param name="referenceModel">Frozen reference model for KL regularization.</param>
    /// <param name="beta">Temperature parameter controlling preference strength (default: 5000.0).</param>
    /// <param name="stepWeight">Weight for per-step preference signal (default: 1.0).</param>
    public D3PO(
        IDiffusionModel<T> model,
        IDiffusionModel<T> referenceModel,
        double beta = 5000.0,
        double stepWeight = 1.0)
    {
        _model = model;
        _referenceModel = referenceModel;
        _beta = beta;
        _stepWeight = stepWeight;
    }

    /// <summary>
    /// Computes the D3PO per-step loss at a single denoising timestep.
    /// </summary>
    /// <param name="preferredNoisePred">Model's noise prediction for the preferred trajectory at this step.</param>
    /// <param name="disPreferredNoisePred">Model's noise prediction for the dispreferred trajectory at this step.</param>
    /// <param name="refPreferredNoisePred">Reference model's noise prediction for preferred trajectory.</param>
    /// <param name="refDisPreferredNoisePred">Reference model's noise prediction for dispreferred trajectory.</param>
    /// <returns>Per-step D3PO loss.</returns>
    public T ComputeStepLoss(
        Vector<T> preferredNoisePred,
        Vector<T> disPreferredNoisePred,
        Vector<T> refPreferredNoisePred,
        Vector<T> refDisPreferredNoisePred)
    {
        // Log-likelihood ratio at this step
        var prefLogRatio = ComputeLogLikelihoodRatio(preferredNoisePred, refPreferredNoisePred);
        var disPrefLogRatio = ComputeLogLikelihoodRatio(disPreferredNoisePred, refDisPreferredNoisePred);

        var logitDiff = NumOps.Subtract(prefLogRatio, disPrefLogRatio);
        var scaled = NumOps.Multiply(NumOps.FromDouble(_beta * _stepWeight), logitDiff);

        // -log(sigmoid(scaled))
        var negScaled = NumOps.Negate(scaled);
        var expNeg = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(negScaled)));
        var sigmoid = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));

        var logSigmoid = NumOps.FromDouble(Math.Log(Math.Max(1e-10, NumOps.ToDouble(sigmoid))));
        return NumOps.Negate(logSigmoid);
    }

    private T ComputeLogLikelihoodRatio(Vector<T> modelPred, Vector<T> refPred)
    {
        var sum = NumOps.Zero;
        int len = Math.Min(modelPred.Length, refPred.Length);
        for (int i = 0; i < len; i++)
        {
            var diff = NumOps.Subtract(modelPred[i], refPred[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return NumOps.Negate(NumOps.Divide(sum, NumOps.FromDouble(2.0 * len)));
    }
}
