using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Abstract base class for teacher models used in knowledge distillation.
/// Provides common functionality and utilities for teacher model implementations.
/// </summary>
/// <typeparam name="TInput">The input data type (e.g., Vector, Matrix, Tensor).</typeparam>
/// <typeparam name="TOutput">The output data type (typically logits as Vector or Matrix).</typeparam>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides common functionality that all teacher models need,
/// such as numeric operations and input validation. It's a lightweight foundation that derived classes
/// build upon.</para>
///
/// <para><b>Why use a base class?</b>
/// - **Code Reuse**: Common utilities like numeric operations are available to all implementations
/// - **Consistency**: All teachers have access to the same helper methods
/// - **Extensibility**: New teacher types inherit core functionality automatically
/// - **Maintainability**: Updates to common utilities benefit all implementations</para>
///
/// <para><b>Architecture Note:</b> This base class is intentionally minimal. Complex operations like
/// temperature scaling are handled by distillation strategies, not teachers. Teachers are responsible
/// only for providing raw logits.</para>
/// </remarks>
public abstract class TeacherModelBase<TInput, TOutput, T> : ITeacherModel<TInput, TOutput>
{
    /// <summary>
    /// Numeric operations for the specified type T.
    /// </summary>
    /// <remarks>
    /// <para>Provides type-specific arithmetic operations (add, multiply, etc.) that work
    /// with any numeric type (double, float, decimal, etc.).</para>
    /// </remarks>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the number of output dimensions (e.g., number of classes for classification).
    /// </summary>
    /// <remarks>
    /// <para><b>For Implementers:</b> Return the size of the output vector. For example,
    /// a 10-class classifier should return 10.</para>
    /// </remarks>
    public abstract int OutputDimension { get; }

    /// <summary>
    /// Initializes the base teacher model and sets up numeric operations.
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
    ///
    /// <para><b>Important:</b> Temperature scaling and softmax conversion are handled by the
    /// distillation strategy, not by the teacher. Just return raw logits here.</para>
    /// </remarks>
    public abstract TOutput GetLogits(TInput input);

    /// <summary>
    /// Validates that the input is not null.
    /// </summary>
    /// <param name="input">Input to validate.</param>
    /// <param name="paramName">Parameter name for exception message.</param>
    /// <remarks>
    /// <para>Helper method for derived classes to validate inputs before processing.</para>
    /// </remarks>
    protected void ValidateInput(TInput? input, string paramName = "input")
    {
        if (input == null) throw new ArgumentNullException(paramName);
    }

    /// <summary>
    /// Applies temperature-scaled softmax to logits.
    /// </summary>
    /// <param name="logits">Raw logits to convert to probabilities.</param>
    /// <param name="temperature">Temperature parameter for softening (default: 1.0).</param>
    /// <returns>Probability distribution over classes.</returns>
    /// <remarks>
    /// <para><b>Temperature Effects:</b>
    /// - temperature = 1.0: Standard softmax (sharp distribution)
    /// - temperature > 1.0: Softened distribution (more uniform, better for distillation)
    /// - temperature < 1.0: Sharpened distribution (more peaked)</para>
    ///
    /// <para><b>Stability:</b> Uses LogSumExp trick to prevent numerical overflow/underflow.</para>
    /// </remarks>
    protected virtual Vector<T> Softmax(Vector<T> logits, double temperature = 1.0)
    {
        if (logits == null) throw new ArgumentNullException(nameof(logits));
        if (temperature <= 0) throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be positive");

        var scaledLogits = new T[logits.Length];
        var tempValue = NumOps.FromDouble(temperature);

        for (int i = 0; i < logits.Length; i++)
        {
            scaledLogits[i] = NumOps.Divide(logits[i], tempValue);
        }

        // Find max logit for numerical stability
        var maxLogit = scaledLogits[0];
        for (int i = 1; i < scaledLogits.Length; i++)
        {
            if (NumOps.GreaterThanOrEquals(scaledLogits[i], maxLogit))
            {
                maxLogit = scaledLogits[i];
            }
        }

        // Compute exp(logit - maxLogit) and sum
        T sumExp = NumOps.Zero;
        var expValues = new T[scaledLogits.Length];
        for (int i = 0; i < scaledLogits.Length; i++)
        {
            var shifted = NumOps.Subtract(scaledLogits[i], maxLogit);
            expValues[i] = NumOps.Exp(shifted);
            sumExp = NumOps.Add(sumExp, expValues[i]);
        }

        // Normalize to get probabilities
        var probabilities = new T[scaledLogits.Length];
        for (int i = 0; i < scaledLogits.Length; i++)
        {
            probabilities[i] = NumOps.Divide(expValues[i], sumExp);
        }

        return new Vector<T>(probabilities);
    }    
    /// <summary>
    /// Applies temperature-scaled softmax to logits. Must be implemented by subclasses
    /// based on their output type (Vector, Matrix, etc.).
    /// </summary>
    /// <param name="logits">Raw model outputs.</param>
    /// <param name="temperature">Temperature for scaling.</param>
    /// <returns>Probability distribution.</returns>
    protected virtual TOutput ApplyTemperatureSoftmax(TOutput logits, double temperature)
    {
        // Default implementation for Vector<T>
        if (logits is not Vector<T> vectorLogits)
            throw new NotSupportedException(
                $"Default softmax implementation only supports Vector<T>. Override ApplyTemperatureSoftmax for {typeof(TOutput).Name}.");

        return (TOutput)(object)SoftmaxVector(vectorLogits, temperature);
    }

    private Vector<T> SoftmaxVector(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);
        var scaled = new T[n];

        for (int i = 0; i < n; i++)
            scaled[i] = NumOps.FromDouble(Convert.ToDouble(logits[i]) / temperature);

        T maxLogit = scaled[0];
        for (int i = 1; i < n; i++)
            if (NumOps.GreaterThan(scaled[i], maxLogit))
                maxLogit = scaled[i];

        T sum = NumOps.Zero;
        var expValues = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = Convert.ToDouble(NumOps.Subtract(scaled[i], maxLogit));
            expValues[i] = NumOps.FromDouble(Math.Exp(val));
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < n; i++)
            result[i] = NumOps.Divide(expValues[i], sum);

        return result;
    }

}
