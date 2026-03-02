using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

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
        var result = new Vector<T>(currentParameters.Length);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = currentParameters[i];
        }

        foreach (var styleVec in _styleVectors)
        {
            // Task vector negation: theta_erased = theta - alpha * tau_style
            // where tau_style = theta_finetuned - theta_base
            var rate = NumOps.FromDouble(_erasureRate);

            for (int i = 0; i < result.Length && i < styleVec.Length; i++)
            {
                var update = NumOps.Multiply(rate, styleVec[i]);
                result[i] = NumOps.Subtract(result[i], update);
            }
        }

        // Apply preservation: don't deviate too far from base
        if (_preservationThreshold > 0)
        {
            var threshold = NumOps.FromDouble(_preservationThreshold);
            for (int i = 0; i < result.Length && i < baseParameters.Length; i++)
            {
                var diff = NumOps.Subtract(result[i], baseParameters[i]);
                var absDiff = NumOps.FromDouble(Math.Abs(NumOps.ToDouble(diff)));
                if (NumOps.ToDouble(absDiff) > NumOps.ToDouble(threshold))
                {
                    // Clamp deviation
                    var sign = NumOps.ToDouble(diff) >= 0 ? 1.0 : -1.0;
                    result[i] = NumOps.Add(baseParameters[i],
                        NumOps.FromDouble(sign * NumOps.ToDouble(threshold)));
                }
            }
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
    public double MeasureStylePresence(Vector<T> currentParameters, Vector<T> baseParameters, int styleIndex)
    {
        if (styleIndex < 0 || styleIndex >= _styleVectors.Count)
        {
            return 0.0;
        }

        var styleVec = _styleVectors[styleIndex];
        double dotProduct = 0, normDelta = 0, normStyle = 0;
        var len = Math.Min(Math.Min(currentParameters.Length, baseParameters.Length), styleVec.Length);

        for (int i = 0; i < len; i++)
        {
            var delta = NumOps.ToDouble(NumOps.Subtract(currentParameters[i], baseParameters[i]));
            var style = NumOps.ToDouble(styleVec[i]);
            dotProduct += delta * style;
            normDelta += delta * delta;
            normStyle += style * style;
        }

        var denom = Math.Sqrt(normDelta) * Math.Sqrt(normStyle);
        return denom > 1e-10 ? Math.Abs(dotProduct) / denom : 0.0;
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
