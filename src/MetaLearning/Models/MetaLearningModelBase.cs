using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// Abstract base class for meta-learning adapted models that wrap a base model with task-specific parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Extends <see cref="ModelWrapperBase{T, TInput, TOutput}"/> with meta-learning-specific behavior:
/// training is not supported directly (use the meta-learning algorithm instead), and parameter
/// management is abstract so each adapted model can store its own task-specific parameters.
/// </para>
/// <para><b>For Beginners:</b> Meta-learning adapted models are created by meta-learning algorithms
/// (like MAML, ProtoNets, etc.) after adapting to a new task. They wrap a base neural network
/// with task-specific parameters. This base class handles common concerns like serialization,
/// gradient computation, and feature awareness by delegating to the wrapped model.</para>
/// </remarks>
public abstract class MetaLearningModelBase<T, TInput, TOutput> : ModelWrapperBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MetaLearningModelBase{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="baseModel">The underlying model used for feature extraction.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="baseModel"/> is null.</exception>
    protected MetaLearningModelBase(IFullModel<T, TInput, TOutput> baseModel)
        : base(baseModel)
    {
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Direct training is not supported for meta-learning adapted models. Use the corresponding
    /// meta-learning algorithm to adapt the model to a new task.
    /// </remarks>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        throw new NotSupportedException(
            $"Direct training is not supported for {GetType().Name}. " +
            "Use the corresponding meta-learning algorithm to adapt the model.");
    }

    /// <inheritdoc/>
    public abstract override Vector<T> GetParameters();

    /// <inheritdoc/>
    public abstract override void SetParameters(Vector<T> parameters);

    /// <inheritdoc/>
    public override int ParameterCount => GetParameters().Length;

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var parameters = GetParameters();
        if (gradients.Length != parameters.Length)
        {
            throw new ArgumentException(
                $"Gradient length mismatch: expected {parameters.Length}, got {gradients.Length}.",
                nameof(gradients));
        }

        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] = NumOps.Subtract(parameters[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        SetParameters(parameters);
    }

    /// <summary>
    /// Helper method to convert a vector output to the expected TOutput type.
    /// </summary>
    /// <param name="logits">The logit vector to convert.</param>
    /// <returns>The converted output.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the conversion to <typeparamref name="TOutput"/> is not supported.
    /// </exception>
    protected static TOutput ConvertVectorToOutput(Vector<T> logits)
    {
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)logits;
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)Tensor<T>.FromVector(logits);
        }

        if (typeof(TOutput) == typeof(T[]))
        {
            return (TOutput)(object)logits.ToArray();
        }

        throw new InvalidOperationException(
            $"Cannot convert Vector<{typeof(T).Name}> to {typeof(TOutput).Name}. " +
            $"Supported types: Vector<T>, Tensor<T>, T[]");
    }

    /// <summary>
    /// Helper method to extract features from the base model output as a vector.
    /// </summary>
    /// <param name="input">The input to process.</param>
    /// <param name="featureDim">The expected feature dimension (for fallback).</param>
    /// <returns>The feature vector extracted from the base model output.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the base model produces unsupported output.
    /// </exception>
    protected Vector<T> ExtractFeaturesFromBaseModel(TInput input, int featureDim)
    {
        var output = BaseModel.Predict(input);

        if (output is Vector<T> vec)
        {
            return vec;
        }

        if (output is Tensor<T> tensor)
        {
            return tensor.ToVector();
        }

        if (output is Matrix<T> matrix)
        {
            if (matrix.Rows == 1)
            {
                var result = new Vector<T>(matrix.Columns);
                for (int j = 0; j < matrix.Columns; j++)
                {
                    result[j] = matrix[0, j];
                }
                return result;
            }
            else
            {
                var result = new Vector<T>(matrix.Rows * matrix.Columns);
                int idx = 0;
                for (int i = 0; i < matrix.Rows; i++)
                {
                    for (int j = 0; j < matrix.Columns; j++)
                    {
                        result[idx++] = matrix[i, j];
                    }
                }
                return result;
            }
        }

        if (output is T[] array)
        {
            return new Vector<T>(array);
        }

        throw new InvalidOperationException(
            $"Base model returned unsupported output type '{output?.GetType().Name ?? "null"}'. " +
            $"Expected Vector<{typeof(T).Name}>, Tensor<{typeof(T).Name}>, Matrix<{typeof(T).Name}>, or {typeof(T).Name}[].");
    }
}
