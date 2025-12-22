namespace AiDotNet.Interfaces;

using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Defines the contract for certified defense mechanisms that provide provable robustness guarantees.
/// </summary>
/// <remarks>
/// Certified defenses provide mathematical guarantees that a model's predictions won't change
/// within a specified perturbation radius, unlike heuristic defenses.
///
/// <b>For Beginners:</b> Think of certified defenses as "guaranteed protection" for your model.
/// While regular defenses make models harder to fool, certified defenses can mathematically prove
/// that no attack within certain limits can trick the model.
///
/// Common certified defense methods include:
/// - Randomized Smoothing: Uses random noise to create certified predictions
/// - Interval Bound Propagation: Tracks ranges of possible values through the network
/// - CROWN: Computes certified bounds for neural network outputs
///
/// Why certified defenses matter:
/// - They provide provable security guarantees
/// - They're essential for safety-critical applications
/// - They help meet regulatory requirements
/// - They give confidence bounds for predictions
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type for the model (e.g., Vector&lt;T&gt;, string).</typeparam>
/// <typeparam name="TOutput">The output data type for the model (e.g., Vector&lt;T&gt;, int).</typeparam>
public interface ICertifiedDefense<T, TInput, TOutput> : IModelSerializer
{
    /// <summary>
    /// Computes a certified prediction with robustness guarantees.
    /// </summary>
    /// <remarks>
    /// This method provides a prediction that is guaranteed to be correct for all inputs
    /// within a specified perturbation radius.
    ///
    /// <b>For Beginners:</b> This is like making a prediction with a "warranty".
    /// The method tells you: "I predict this class, and I guarantee that even if someone
    /// changes the input slightly (within specified limits), my prediction won't change."
    ///
    /// The process typically involves:
    /// 1. Analyzing the input and model
    /// 2. Computing bounds on what the model could output
    /// 3. If all possible outputs within the bounds agree, the prediction is certified
    /// 4. Returning both the prediction and the guaranteed radius of correctness
    /// </remarks>
    /// <param name="input">The input to make a certified prediction for.</param>
    /// <param name="model">The model to certify.</param>
    /// <returns>Certified prediction result with robustness radius.</returns>
    CertifiedPrediction<T> CertifyPrediction(TInput input, IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Computes certified predictions for a batch of inputs.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the same as CertifyPrediction, but processes
    /// multiple inputs at once for efficiency.
    /// </remarks>
    /// <param name="inputs">The batch of inputs to certify.</param>
    /// <param name="model">The model to certify.</param>
    /// <returns>Batch of certified prediction results.</returns>
    CertifiedPrediction<T>[] CertifyBatch(TInput[] inputs, IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Computes the maximum perturbation radius that can be certified for an input.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This calculates how much an input can be changed while
    /// still guaranteeing the model's prediction stays the same. It's like measuring
    /// the "safety zone" around your input.
    ///
    /// For example, if the radius is 0.1, you can change pixel values by up to 0.1
    /// and the model is guaranteed to give the same answer.
    /// </remarks>
    /// <param name="input">The input to analyze.</param>
    /// <param name="model">The model being certified.</param>
    /// <returns>The maximum certified robustness radius.</returns>
    T ComputeCertifiedRadius(TInput input, IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Evaluates certified accuracy on a dataset.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This measures what percentage of predictions can be
    /// certified as robust. Higher certified accuracy means more predictions have
    /// guaranteed robustness.
    /// </remarks>
    /// <param name="testData">The test data to evaluate on.</param>
    /// <param name="labels">The true labels.</param>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="radius">The perturbation radius to certify.</param>
    /// <returns>Certified accuracy metrics.</returns>
    CertifiedAccuracyMetrics<T> EvaluateCertifiedAccuracy(
        TInput[] testData,
        TOutput[] labels,
        IFullModel<T, TInput, TOutput> model,
        T radius);

    /// <summary>
    /// Gets the configuration options for the certified defense.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are the settings that control how certification works,
    /// like the number of samples to use or the tightness of bounds.
    /// </remarks>
    /// <returns>The configuration options for the certified defense.</returns>
    CertifiedDefenseOptions<T> GetOptions();

    /// <summary>
    /// Resets the certified defense state.
    /// </summary>
    void Reset();
}
