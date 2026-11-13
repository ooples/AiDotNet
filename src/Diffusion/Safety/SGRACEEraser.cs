using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Safety;

/// <summary>
/// S-GRACE: Style-aware GRACE for erasing artistic styles from diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// S-GRACE extends the GRACE (GRadient-based Concept Erasure) method to handle artistic
/// style concepts, which are more distributed across model weights than object concepts.
/// It uses style-specific gradient directions and multi-layer editing to effectively
/// remove artistic style associations while preserving content generation quality.
/// </para>
/// <para>
/// <b>For Beginners:</b> Removing an artistic style from a model is harder than removing
/// an object because styles affect how everything looks, not just what appears. S-GRACE
/// is designed specifically for this challenge — it finds and removes the "style knowledge"
/// distributed throughout the model's layers, so the model can't mimic the erased style
/// but still generates high-quality images in other styles.
/// </para>
/// <para>
/// Reference: Pham et al., "Robust Concept Erasure Using Task Vectors", 2024
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class SGRACEEraser<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _erasureRate;
    private readonly int _numIterations;
    private readonly double _preservationThreshold;
    private readonly List<Vector<T>> _styleVectors;

    /// <summary>
    /// Initializes a new S-GRACE eraser.
    /// </summary>
    /// <param name="erasureRate">Learning rate for gradient-based erasure (default: 1e-5).</param>
    /// <param name="numIterations">Number of erasure iterations (default: 1000).</param>
    /// <param name="preservationThreshold">Threshold for preserving non-target capabilities (default: 0.01).</param>
    public SGRACEEraser(
        double erasureRate = 1e-5,
        int numIterations = 1000,
        double preservationThreshold = 0.01)
    {
        _erasureRate = erasureRate;
        _numIterations = numIterations;
        _preservationThreshold = preservationThreshold;
        _styleVectors = [];
    }

    /// <summary>
    /// Registers a style task vector for erasure.
    /// </summary>
    /// <param name="styleVector">The task vector representing the style to erase
    /// (difference between fine-tuned and base model parameters).</param>
    public void AddStyleVector(Vector<T> styleVector)
    {
        _styleVectors.Add(styleVector);
    }

    /// <summary>
    /// Computes the erasure update for model parameters using the task vector negation approach.
    /// </summary>
    /// <param name="currentParameters">Current model parameters.</param>
    /// <param name="baseParameters">Base model parameters (before style fine-tuning).</param>
    /// <returns>Updated parameters with the style concept erased.</returns>
    public Vector<T> ComputeErasureUpdate(Vector<T> currentParameters, Vector<T> baseParameters)
    {
        var engine = AiDotNetEngine.Current;
        var result = currentParameters.Clone();
        var rate = NumOps.FromDouble(_erasureRate);

        foreach (var styleVec in _styleVectors)
        {
            // Task vector negation: theta_erased = theta - alpha * tau_style
            // where tau_style = theta_finetuned - theta_base (zero-padded / truncated to the parameters)
            result = engine.Subtract(result, engine.Multiply(DiffusionVectorOps.Fit(styleVec, result.Length), rate));
        }

        // Apply preservation: don't deviate too far from base. Over the common prefix, an element whose deviation
        // from the base exceeds the threshold is clamped to base +/- threshold.
        if (_preservationThreshold > 0)
        {
            var threshold = NumOps.FromDouble(_preservationThreshold);
            int len = Math.Min(result.Length, baseParameters.Length);
            var resultPrefix = DiffusionVectorOps.Fit(result, len).AsTensor();
            var basePrefix = DiffusionVectorOps.Fit(baseParameters, len).AsTensor();
            var diff = engine.TensorSubtract(resultPrefix, basePrefix);
            var exceeds = engine.TensorGreaterThan(engine.TensorAbs(diff), threshold);
            var clamped = engine.TensorAdd(basePrefix, engine.TensorClamp(diff, NumOps.Negate(threshold), threshold));
            var preserved = engine.TensorWhere(exceeds, clamped, resultPrefix).ToVector();
            result = len < result.Length
                ? Vector<T>.Concatenate(preserved, result.GetSubVector(len, result.Length - len))
                : preserved;
        }

        return result;
    }

    /// <summary>
    /// Measures style presence as cosine similarity between parameter delta and style vector.
    /// </summary>
    /// <param name="currentParameters">Current model parameters.</param>
    /// <param name="baseParameters">Base model parameters.</param>
    /// <param name="styleIndex">Index of the style vector to check.</param>
    /// <returns>Style presence score (0 = absent, 1 = fully present).</returns>
    public T MeasureStylePresence(Vector<T> currentParameters, Vector<T> baseParameters, int styleIndex)
    {
        if (styleIndex < 0 || styleIndex >= _styleVectors.Count)
        {
            return NumOps.Zero;
        }

        // |cos(delta, style)| over the common prefix, delta = current - base; 0 when either norm vanishes
        var engine = AiDotNetEngine.Current;
        var styleVec = _styleVectors[styleIndex];
        var len = Math.Min(Math.Min(currentParameters.Length, baseParameters.Length), styleVec.Length);
#if NET5_0_OR_GREATER
        // Half dot products accumulate in Half and overflow for ordinary parameter vectors: reduce in double, as the
        // converted loop did, and return the cosine as Half.
        if (typeof(T) == typeof(Half))
        {
            var wide = new SGRACEEraser<double>();
            wide.AddStyleVector(DiffusionVectorOps.Fit(styleVec, len).AsTensor().Cast<double>().ToVector());
            return NumOps.FromDouble(wide.MeasureStylePresence(
                DiffusionVectorOps.Fit(currentParameters, len).AsTensor().Cast<double>().ToVector(),
                DiffusionVectorOps.Fit(baseParameters, len).AsTensor().Cast<double>().ToVector(), 0));
        }
#endif
        var delta = engine.Subtract(DiffusionVectorOps.Fit(currentParameters, len), DiffusionVectorOps.Fit(baseParameters, len));
        var style = DiffusionVectorOps.Fit(styleVec, len);
        var dotProduct = engine.DotProduct(delta, style);
        var denom = NumOps.Multiply(NumOps.Sqrt(engine.DotProduct(delta, delta)), NumOps.Sqrt(engine.DotProduct(style, style)));
        return NumOps.GreaterThan(denom, NumOps.FromDouble(1e-10)) ? NumOps.Divide(NumOps.Abs(dotProduct), denom) : NumOps.Zero;
    }

    /// <summary>
    /// Gets the number of registered style vectors.
    /// </summary>
    public int StyleCount => _styleVectors.Count;

    /// <summary>
    /// Gets the erasure rate.
    /// </summary>
    public double ErasureRate => _erasureRate;

    /// <summary>
    /// Gets the number of erasure iterations.
    /// </summary>
    public int NumIterations => _numIterations;

    /// <summary>
    /// Gets the preservation threshold.
    /// </summary>
    public double PreservationThreshold => _preservationThreshold;
}
