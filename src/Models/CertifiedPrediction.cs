using AiDotNet.Helpers;

namespace AiDotNet.Models;

/// <summary>
/// Represents a certified prediction with robustness guarantees.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class CertifiedPrediction<T>
{
    /// <summary>
    /// Gets or sets the predicted class label.
    /// </summary>
    public int PredictedClass { get; set; }

    /// <summary>
    /// Gets or sets the certified robustness radius.
    /// </summary>
    /// <remarks>
    /// The prediction is guaranteed to remain the same for all inputs within this radius.
    /// </remarks>
    public required T CertifiedRadius { get; set; }

    /// <summary>
    /// Gets or sets whether the prediction could be certified.
    /// </summary>
    public bool IsCertified { get; set; }

    /// <summary>
    /// Gets or sets the confidence score for the prediction.
    /// </summary>
    public T Confidence { get; set; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets or sets the lower bound on the probability of the predicted class.
    /// </summary>
    public T LowerBound { get; set; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets or sets the upper bound on the probability of the predicted class.
    /// </summary>
    public T UpperBound { get; set; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets or sets additional certification information.
    /// </summary>
    public Dictionary<string, object> CertificationDetails { get; set; } = new();
}
