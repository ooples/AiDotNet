using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Abstract base class for teacher models used in knowledge distillation.
/// Provides common functionality for extracting logits, soft predictions, and features from trained models.
/// </summary>
/// <typeparam name="TInput">The input data type (e.g., Vector, Matrix, Tensor).</typeparam>
/// <typeparam name="TOutput">The output data type (typically logits or probabilities as Vector or Matrix).</typeparam>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides common functionality that all teacher models need.
/// Instead of each teacher implementation writing the same code for softmax and temperature scaling,
/// they inherit these capabilities from this base class.</para>
///
/// <para><b>Why use a base class?</b>
/// - **Code Reuse**: Common operations like softmax are implemented once
/// - **Consistency**: All teachers behave the same way for standard operations
/// - **Extensibility**: New teacher types only need to implement model-specific logic
/// - **Maintainability**: Bug fixes in base class benefit all implementations</para>
///
/// <para><b>Design Pattern:</b> This implements the Template Method pattern, where the base class
/// defines the algorithm structure (e.g., GetSoftPredictions), and subclasses fill in specific steps
/// (e.g., GetLogits implementation).</para>
/// </remarks>
public abstract class TeacherModelBase<TInput, TOutput, T> : ITeacherModel<TInput, TOutput>
{
    /// <summary>
    /// Numeric operations for the specified type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the number of output dimensions (e.g., number of classes for classification).
    /// </summary>
    public abstract int OutputDimension { get; }

    /// <summary>
    /// Initializes the base teacher model.
    /// </summary>
    protected TeacherModelBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the teacher's raw logits (pre-softmax outputs) for the given input.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <returns>Raw logits before applying softmax.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this method to extract logits from your specific model type.
    /// Ensure you return pre-activation outputs, not probabilities.</para>
    /// </remarks>
    public abstract TOutput GetLogits(TInput input);

    /// <summary>
    /// Gets the teacher's soft predictions (probabilities) with temperature scaling.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <param name="temperature">Softmax temperature (default 1.0). Higher values produce softer distributions.</param>
    /// <returns>Probability distribution with temperature scaling applied.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies temperature scaling to logits and converts them
    /// to probabilities. The base class handles this automatically - subclasses just need to provide logits.</para>
    /// </remarks>
    public virtual TOutput GetSoftPredictions(TInput input, double temperature = 1.0)
    {
        if (temperature <= 0)
            throw new ArgumentException("Temperature must be positive", nameof(temperature));

        var logits = GetLogits(input);
        return ApplyTemperatureSoftmax(logits, temperature);
    }

    /// <summary>
    /// Gets intermediate layer features from the teacher model.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <param name="layerName">The name of the layer to extract features from.</param>
    /// <returns>Feature map from the specified layer, or null if not supported.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this method if your model supports feature extraction.
    /// Return null if feature extraction is not applicable for your model type.</para>
    /// </remarks>
    public virtual object? GetFeatures(TInput input, string layerName)
    {
        // Default implementation: feature extraction not supported
        return null;
    }

    /// <summary>
    /// Gets attention weights from the teacher model (for transformer architectures).
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <param name="layerName">The name of the attention layer to extract weights from.</param>
    /// <returns>Attention weight matrix from the specified layer, or null if not supported.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this method if your model has attention mechanisms.
    /// This is primarily for transformer-based models like BERT, GPT, etc.</para>
    /// </remarks>
    public virtual object? GetAttentionWeights(TInput input, string layerName)
    {
        // Default implementation: attention extraction not supported
        return null;
    }

    /// <summary>
    /// Applies temperature-scaled softmax to logits. Must be implemented by subclasses
    /// based on their output type (Vector, Matrix, etc.).
    /// </summary>
    /// <param name="logits">Raw model outputs.</param>
    /// <param name="temperature">Temperature for scaling.</param>
    /// <returns>Probability distribution.</returns>
    protected abstract TOutput ApplyTemperatureSoftmax(TOutput logits, double temperature);

    /// <summary>
    /// Validates that the input is not null.
    /// </summary>
    /// <param name="input">Input to validate.</param>
    /// <param name="paramName">Parameter name for exception message.</param>
    protected void ValidateInput(TInput? input, string paramName = "input")
    {
        if (input == null) throw new ArgumentNullException(paramName);
    }

    /// <summary>
    /// Validates that the layer name is not null or empty.
    /// </summary>
    /// <param name="layerName">Layer name to validate.</param>
    protected void ValidateLayerName(string? layerName)
    {
        if (string.IsNullOrWhiteSpace(layerName)) throw new ArgumentException("Value cannot be null or whitespace.", nameof(layerName));
    }
}
