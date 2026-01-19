using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Models.Results;

/// <summary>
/// Represents a prediction result augmented with uncertainty information.
/// </summary>
/// <typeparam name="T">The numeric type used for uncertainty computations (e.g., float, double).</typeparam>
/// <typeparam name="TOutput">The model's output type (e.g., <c>Tensor&lt;T&gt;</c>).</typeparam>
/// <remarks>
/// <para>
/// This type is returned by the facade method <c>AiModelResult.PredictWithUncertainty(...)</c>.
/// </para>
/// <para><b>For Beginners:</b> This lets you ask the model both:
/// - "What is the prediction?"
/// - "How uncertain is that prediction?"</para>
/// </remarks>
public sealed class UncertaintyPredictionResult<T, TOutput>
{
    /// <summary>
    /// Gets the uncertainty method that was used to produce this result.
    /// </summary>
    public UncertaintyQuantificationMethod MethodUsed { get; }

    /// <summary>
    /// Gets the point prediction (mean / expected prediction).
    /// </summary>
    public TOutput Prediction { get; }

    /// <summary>
    /// Gets the per-output predictive variance (when available).
    /// </summary>
    public TOutput? Variance { get; }

    /// <summary>
    /// Gets additional uncertainty diagnostics.
    /// </summary>
    /// <remarks>
    /// Keys are stable so downstream consumers do not need to branch on missing keys.
    /// </remarks>
    public IReadOnlyDictionary<string, Tensor<T>> Metrics { get; }

    /// <summary>
    /// Gets an optional conformal regression interval, when configured and supported.
    /// </summary>
    public RegressionConformalInterval<TOutput>? RegressionInterval { get; }

    /// <summary>
    /// Gets an optional conformal classification prediction set, when configured and supported.
    /// </summary>
    public ClassificationConformalPredictionSet? ClassificationSet { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="UncertaintyPredictionResult{T, TOutput}"/> class.
    /// </summary>
    public UncertaintyPredictionResult(
        UncertaintyQuantificationMethod methodUsed,
        TOutput prediction,
        TOutput? variance,
        IReadOnlyDictionary<string, Tensor<T>> metrics,
        RegressionConformalInterval<TOutput>? regressionInterval = null,
        ClassificationConformalPredictionSet? classificationSet = null)
    {
        MethodUsed = methodUsed;
        Prediction = prediction;
        Variance = variance;
        Metrics = metrics ?? throw new ArgumentNullException(nameof(metrics));
        RegressionInterval = regressionInterval;
        ClassificationSet = classificationSet;
    }
}

