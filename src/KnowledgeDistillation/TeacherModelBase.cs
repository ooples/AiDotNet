using AiDotNet.Autodiff;

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
public abstract class TeacherModelBase<TInput, TOutput, T> : ITeacherModel<TInput, TOutput>, IJitCompilable<T>
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

    #region IJitCompilable Implementation

    /// <summary>
    /// Gets whether this teacher model supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> if the teacher model can be JIT compiled; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// Teacher models that wrap other models should delegate to the wrapped model's JIT support.
    /// Teacher models using function delegates or cached predictions may not support JIT.
    /// </para>
    /// <para><b>For Implementers:</b> Return <c>true</c> if your teacher model can export its
    /// computation as a graph. Models wrapping IJitCompilable implementations should return
    /// the wrapped model's SupportsJitCompilation value.
    /// </para>
    /// </remarks>
    public abstract bool SupportsJitCompilation { get; }

    /// <summary>
    /// Exports the teacher model's computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the teacher's logits.</returns>
    /// <remarks>
    /// <para>
    /// For teacher models that wrap other models, this should delegate to the wrapped model's
    /// ExportComputationGraph method. For models using function delegates, this may not be
    /// supported and should throw NotSupportedException.
    /// </para>
    /// <para><b>For Implementers:</b> If your teacher wraps a model implementing IJitCompilable,
    /// delegate to that model's ExportComputationGraph. Otherwise, implement the computation
    /// graph directly or throw NotSupportedException with a clear explanation.
    /// </para>
    /// </remarks>
    /// <exception cref="NotSupportedException">
    /// Thrown when the teacher model does not support JIT compilation.
    /// </exception>
    public abstract ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes);

    #endregion

    #region JIT Helper Methods

    /// <summary>
    /// Checks if a wrapped teacher model supports JIT compilation.
    /// </summary>
    /// <param name="wrappedModel">The wrapped teacher model to check.</param>
    /// <returns>
    /// <c>true</c> if the wrapped model implements IJitCompilable and supports JIT; otherwise, <c>false</c>.
    /// </returns>
    /// <remarks>
    /// <para>Use this helper method in derived classes that wrap another ITeacherModel to implement
    /// the SupportsJitCompilation property.</para>
    /// <para>Example:
    /// <code>
    /// public override bool SupportsJitCompilation => CheckWrappedModelJitSupport(_baseTeacher);
    /// </code>
    /// </para>
    /// </remarks>
    protected static bool CheckWrappedModelJitSupport(ITeacherModel<TInput, TOutput> wrappedModel)
    {
        return wrappedModel is IJitCompilable<T> jitCompilable && jitCompilable.SupportsJitCompilation;
    }

    /// <summary>
    /// Delegates JIT compilation export to a wrapped teacher model.
    /// </summary>
    /// <param name="wrappedModel">The wrapped teacher model to delegate to.</param>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <param name="wrapperTypeName">Name of the wrapper type (for error messages).</param>
    /// <returns>The output computation node from the wrapped model.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when the wrapped model does not implement IJitCompilable or does not support JIT.
    /// </exception>
    /// <remarks>
    /// <para>Use this helper method in derived classes that wrap another ITeacherModel to implement
    /// the ExportComputationGraph method.</para>
    /// <para>Example:
    /// <code>
    /// public override ComputationNode&lt;T&gt; ExportComputationGraph(List&lt;ComputationNode&lt;T&gt;&gt; inputNodes)
    ///     => DelegateJitExport(_baseTeacher, inputNodes, nameof(AdaptiveTeacherModel&lt;T&gt;));
    /// </code>
    /// </para>
    /// </remarks>
    protected static ComputationNode<T> DelegateJitExport(
        ITeacherModel<TInput, TOutput> wrappedModel,
        List<ComputationNode<T>> inputNodes,
        string wrapperTypeName)
    {
        if (wrappedModel is not IJitCompilable<T> jitCompilable)
        {
            throw new NotSupportedException(
                $"{wrapperTypeName} cannot export computation graph because the wrapped model " +
                $"({wrappedModel.GetType().Name}) does not implement IJitCompilable<T>.");
        }

        if (!jitCompilable.SupportsJitCompilation)
        {
            throw new NotSupportedException(
                $"{wrapperTypeName} cannot export computation graph because the wrapped model " +
                $"({wrappedModel.GetType().Name}) does not support JIT compilation.");
        }

        return jitCompilable.ExportComputationGraph(inputNodes);
    }

    /// <summary>
    /// Throws a standardized NotSupportedException for teacher models that cannot support JIT compilation.
    /// </summary>
    /// <param name="teacherTypeName">Name of the teacher type.</param>
    /// <param name="reason">Reason why JIT is not supported.</param>
    /// <returns>Never returns (always throws).</returns>
    /// <exception cref="NotSupportedException">Always thrown.</exception>
    /// <remarks>
    /// <para>Use this helper method in derived classes that cannot support JIT compilation.</para>
    /// <para>Example:
    /// <code>
    /// public override ComputationNode&lt;T&gt; ExportComputationGraph(List&lt;ComputationNode&lt;T&gt;&gt; inputNodes)
    ///     => ThrowJitNotSupported(nameof(PretrainedTeacherModel&lt;T&gt;),
    ///         "it uses a function delegate which cannot be exported as a computation graph");
    /// </code>
    /// </para>
    /// </remarks>
    protected static ComputationNode<T> ThrowJitNotSupported(string teacherTypeName, string reason)
    {
        throw new NotSupportedException(
            $"{teacherTypeName} does not support JIT compilation because {reason}.");
    }

    #endregion

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
