namespace AiDotNet.UncertaintyQuantification.Interfaces;

using AiDotNet.Models.Results;

/// <summary>
/// Defines the contract for models that can estimate prediction uncertainty.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Uncertainty estimation helps you understand how confident a model is in its predictions.
///
/// Think of it like a weather forecast that not only predicts rain but also tells you how sure it is:
/// - "90% chance of rain" shows high confidence
/// - "50% chance of rain" shows high uncertainty
///
/// This interface is for models that can provide both a prediction and an estimate of how uncertain that prediction is.
/// This is crucial for safety-critical applications like medical diagnosis or autonomous vehicles.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("UncertaintyEstimator")]
public interface IUncertaintyEstimator<T>
{
    /// <summary>
    /// Predicts output with uncertainty estimates for a single input.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A prediction result augmented with uncertainty information.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method returns two values:
    /// - mean: The model's best guess for the prediction
    /// - uncertainty: How confident the model is (lower = more confident)
    /// </remarks>
    UncertaintyPredictionResult<T, Tensor<T>> PredictWithUncertainty(Tensor<T> input);

    /// <summary>
    /// Estimates aleatoric uncertainty (data noise) for the given input.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The aleatoric uncertainty estimate.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Aleatoric uncertainty comes from inherent randomness in the data.
    /// For example, two identical medical scans might have slightly different measurements due to sensor noise.
    /// This type of uncertainty cannot be reduced by collecting more data.
    /// </remarks>
    Tensor<T> EstimateAleatoricUncertainty(Tensor<T> input);

    /// <summary>
    /// Estimates epistemic uncertainty (model uncertainty) for the given input.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The epistemic uncertainty estimate.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Epistemic uncertainty comes from the model not having enough knowledge.
    /// For example, if your model was trained on cats and dogs, it will have high epistemic uncertainty
    /// when shown a horse. This type of uncertainty can be reduced by collecting more training data.
    /// </remarks>
    Tensor<T> EstimateEpistemicUncertainty(Tensor<T> input);
}
