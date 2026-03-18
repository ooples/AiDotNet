using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Validation;

namespace AiDotNet.Models;

/// <summary>
/// Abstract base class for model wrappers that delegate to an underlying <see cref="IFullModel{T, TInput, TOutput}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Provides default implementations for most <see cref="IFullModel{T, TInput, TOutput}"/>
/// interface members by delegating to the wrapped base model. Subclasses only need to override
/// prediction logic and parameter management specific to their wrapping strategy.
/// </para>
/// <para><b>For Beginners:</b> Some models work by wrapping another model and adding extra behavior.
/// For example, a transfer-learning model wraps a pre-trained model with a feature mapper,
/// or an adversarial defense wraps a model with input preprocessing. This base class handles
/// all the common delegation so wrapper classes only implement what's different.
/// </para>
/// </remarks>
public abstract class ModelWrapperBase<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The underlying full model being wrapped.
    /// </summary>
    protected IFullModel<T, TInput, TOutput> BaseModel { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelWrapperBase{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="baseModel">The underlying model to wrap.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="baseModel"/> is null.</exception>
    protected ModelWrapperBase(IFullModel<T, TInput, TOutput> baseModel)
    {
        Guard.NotNull(baseModel);
        BaseModel = baseModel;
    }

    /// <inheritdoc/>
    public virtual ILossFunction<T> DefaultLossFunction => BaseModel.DefaultLossFunction;

    /// <inheritdoc/>
    public abstract TOutput Predict(TInput input);

    /// <inheritdoc/>
    public virtual void Train(TInput input, TOutput expectedOutput)
        => BaseModel.Train(input, expectedOutput);

    /// <inheritdoc/>
    public virtual ModelMetadata<T> GetModelMetadata() => BaseModel.GetModelMetadata();

    // --- IParameterizable ---

    /// <inheritdoc/>
    public virtual Vector<T> GetParameters() => BaseModel.GetParameters();

    /// <inheritdoc/>
    public virtual void SetParameters(Vector<T> parameters) => BaseModel.SetParameters(parameters);

    /// <inheritdoc/>
    public virtual int ParameterCount => BaseModel.ParameterCount;

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;

    /// <inheritdoc/>
    public abstract IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);

    // --- ICloneable ---

    /// <inheritdoc/>
    public abstract IFullModel<T, TInput, TOutput> DeepCopy();

    /// <inheritdoc/>
    public virtual IFullModel<T, TInput, TOutput> Clone() => DeepCopy();

    // --- IGradientComputable ---

    /// <inheritdoc/>
    public virtual Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
        => BaseModel.ComputeGradients(input, target, lossFunction ?? DefaultLossFunction);

    /// <inheritdoc/>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
        => BaseModel.ApplyGradients(gradients, learningRate);

    // --- IModelSerializer ---

    /// <inheritdoc/>
    public virtual byte[] Serialize() => BaseModel.Serialize();

    /// <inheritdoc/>
    public virtual void Deserialize(byte[] data)
    {
        Guard.NotNull(data);
        BaseModel.Deserialize(data);
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
}
