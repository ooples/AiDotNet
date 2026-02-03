using AiDotNet.Preprocessing;

namespace AiDotNet.Models.Inputs;

/// <summary>
/// Represents the input data required for evaluating a machine learning model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of the input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of the output data from the model.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class acts as a container for all the information needed to evaluate a model.
/// It includes the model itself, the data to evaluate it with, and information about how the data is preprocessed.
/// </para>
/// <para>
/// - The Model property holds the actual machine learning model to be evaluated.
/// - The InputData property contains the data used for evaluation, including inputs and expected outputs.
/// - The PreprocessingInfo property holds information about how the data has been transformed, which is important for
///   interpreting the results correctly.
/// </para>
/// </remarks>
public class ModelEvaluationInput<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the machine learning model to be evaluated.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? Model { get; set; }

    /// <summary>
    /// Gets or sets the input data used for model evaluation.
    /// </summary>
    /// <remarks>
    /// This includes both the input features and the expected outputs for evaluation.
    /// </remarks>
    public OptimizationInputData<T, TInput, TOutput> InputData { get; set; } = new();

    /// <summary>
    /// Gets or sets the preprocessing information for the input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is crucial for correctly interpreting the model's outputs and calculating accurate metrics.
    /// </para>
    /// <para><b>For Beginners:</b> This stores all the data transformations applied during training
    /// (like scaling, encoding, imputation) so the same transformations can be applied to new data.
    /// </para>
    /// </remarks>
    public PreprocessingInfo<T, TInput, TOutput>? PreprocessingInfo { get; set; }

    /// <summary>
    /// Gets or sets an optional override for the prediction type used when calculating metrics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If this is not provided, evaluators may infer the prediction type from the target values or model configuration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells the evaluator what kind of problem you're solving (classification vs regression),
    /// so it can compute the right metrics by default.
    /// </para>
    /// </remarks>
    public PredictionType? PredictionTypeOverride { get; set; }
}
