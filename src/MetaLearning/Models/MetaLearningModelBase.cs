using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// Abstract base class for meta-learning adapted models that implement <see cref="IFullModel{T, TInput, TOutput}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This base class provides default implementations for most <see cref="IFullModel{T, TInput, TOutput}"/>
/// interface members by delegating to the wrapped base model. Subclasses only need to implement
/// prediction logic and parameter management specific to their adaptation strategy.
/// </para>
/// <para><b>For Beginners:</b> Meta-learning adapted models are created by meta-learning algorithms
/// (like MAML, ProtoNets, etc.) after adapting to a new task. They wrap a base neural network
/// with task-specific parameters. This base class handles common concerns like serialization,
/// gradient computation, and feature awareness by delegating to the wrapped model.</para>
/// </remarks>
public abstract class MetaLearningModelBase<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The underlying full model used for feature extraction and prediction.
    /// </summary>
    protected IFullModel<T, TInput, TOutput> BaseModel { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="MetaLearningModelBase{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="baseModel">The underlying model used for feature extraction.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="baseModel"/> is null.</exception>
    protected MetaLearningModelBase(IFullModel<T, TInput, TOutput> baseModel)
    {
        Guard.NotNull(baseModel);
        BaseModel = baseModel;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    public abstract TOutput Predict(TInput input);

    /// <inheritdoc/>
    public virtual void Train(TInput input, TOutput expectedOutput)
    {
        throw new NotSupportedException(
            $"Direct training is not supported for {GetType().Name}. " +
            "Use the corresponding meta-learning algorithm to adapt the model.");
    }

    /// <inheritdoc/>
    public virtual ModelMetadata<T> GetModelMetadata() => Metadata;

    // --- IParameterizable ---

    /// <inheritdoc/>
    public abstract Vector<T> GetParameters();

    /// <inheritdoc/>
    public abstract void SetParameters(Vector<T> parameters);

    /// <inheritdoc/>
    public virtual int ParameterCount => GetParameters().Length;

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;

    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

    /// <inheritdoc/>
    public abstract IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);

    // --- ICloneable<IFullModel<T, TInput, TOutput>> ---

    /// <inheritdoc/>
    public abstract IFullModel<T, TInput, TOutput> DeepCopy();

    /// <inheritdoc/>
    public virtual IFullModel<T, TInput, TOutput> Clone() => DeepCopy();

    // --- IFullModel ---

    /// <inheritdoc/>
    public virtual ILossFunction<T> DefaultLossFunction => BaseModel.DefaultLossFunction;

    // --- IGradientComputable ---

    /// <inheritdoc/>
    public virtual Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
    {
        return BaseModel.ComputeGradients(input, target, lossFunction ?? DefaultLossFunction);
    }

    /// <inheritdoc/>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
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

    // --- IModelSerializer ---

    /// <inheritdoc/>
    public virtual byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        using (ModelPersistenceGuard.InternalOperation())
        {
            return BaseModel.Serialize();
        }
    }

    /// <inheritdoc/>
    public virtual void Deserialize(byte[] data)
    {
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        Guard.NotNull(data);
        using (ModelPersistenceGuard.InternalOperation())
        {
            BaseModel.Deserialize(data);
        }
    }

    /// <inheritdoc/>
    public virtual void SaveModel(string filePath) => BaseModel.SaveModel(filePath);

    /// <inheritdoc/>
    public virtual void LoadModel(string filePath) => BaseModel.LoadModel(filePath);

    // --- ICheckpointableModel ---

    /// <inheritdoc/>
    public virtual void SaveState(Stream stream) => BaseModel.SaveState(stream);

    /// <inheritdoc/>
    public virtual void LoadState(Stream stream) => BaseModel.LoadState(stream);

    // --- IFeatureAware ---

    /// <inheritdoc/>
    public virtual IEnumerable<int> GetActiveFeatureIndices() => BaseModel.GetActiveFeatureIndices();

    /// <inheritdoc/>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        => BaseModel.SetActiveFeatureIndices(featureIndices);

    /// <inheritdoc/>
    public virtual bool IsFeatureUsed(int featureIndex) => BaseModel.IsFeatureUsed(featureIndex);

    // --- IFeatureImportance ---

    /// <inheritdoc/>
    public virtual Dictionary<string, T> GetFeatureImportance() => BaseModel.GetFeatureImportance();

    // --- IJitCompilable ---

    /// <inheritdoc/>
    public virtual bool SupportsJitCompilation => false;

    /// <inheritdoc/>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            $"JIT compilation is not supported for {GetType().Name}.");
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
