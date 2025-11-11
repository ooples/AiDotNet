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
}
