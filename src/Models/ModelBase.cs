using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;

namespace AiDotNet.Models;

/// <summary>
/// Abstract base class for standalone models that directly implement <see cref="IFullModel{T, TInput, TOutput}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Provides common infrastructure and sensible defaults for standalone model implementations
/// that are not wrappers around other models. Subclasses must implement core model behavior:
/// prediction, training, parameter management, loss function, and cloning.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for building standalone machine learning models.
/// Models like linear regression, expression trees, gradient boosting, and ensembles all inherit
/// from this class. It handles boilerplate like serialization and feature tracking so each model
/// only needs to implement its core prediction and training logic.
/// </para>
/// </remarks>
public abstract class ModelBase<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public abstract ILossFunction<T> DefaultLossFunction { get; }

    /// <inheritdoc/>
    public abstract TOutput Predict(TInput input);

    /// <inheritdoc/>
    public abstract void Train(TInput input, TOutput expectedOutput);

    /// <inheritdoc/>
    public virtual ModelMetadata<T> GetModelMetadata() => new();

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
    public abstract IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);

    // --- ICloneable ---

    /// <inheritdoc/>
    public abstract IFullModel<T, TInput, TOutput> DeepCopy();

    /// <inheritdoc/>
    public virtual IFullModel<T, TInput, TOutput> Clone() => DeepCopy();

    // --- IGradientComputable ---

    /// <inheritdoc/>
    public virtual Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
    {
        throw new NotSupportedException(
            $"Gradient computation is not supported for {GetType().Name}. " +
            "Override ComputeGradients to provide an implementation.");
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
        throw new NotSupportedException(
            $"Serialization is not supported for {GetType().Name}. Override Serialize to provide an implementation.");
    }

    /// <inheritdoc/>
    public virtual void Deserialize(byte[] data)
    {
        throw new NotSupportedException(
            $"Deserialization is not supported for {GetType().Name}. Override Deserialize to provide an implementation.");
    }

    /// <inheritdoc/>
    public virtual void SaveModel(string filePath)
    {
        File.WriteAllBytes(filePath, Serialize());
    }

    /// <inheritdoc/>
    public virtual void LoadModel(string filePath)
    {
        Deserialize(File.ReadAllBytes(filePath));
    }

    // --- ICheckpointableModel ---

    /// <inheritdoc/>
    public virtual void SaveState(Stream stream)
    {
        var data = Serialize();
        stream.Write(data, 0, data.Length);
        stream.Flush();
    }

    /// <inheritdoc/>
    public virtual void LoadState(Stream stream)
    {
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        Deserialize(ms.ToArray());
    }

    // --- IFeatureAware ---

    /// <inheritdoc/>
    public virtual IEnumerable<int> GetActiveFeatureIndices() => Array.Empty<int>();

    /// <inheritdoc/>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }

    /// <inheritdoc/>
    public virtual bool IsFeatureUsed(int featureIndex) => false;

    // --- IFeatureImportance ---

    /// <inheritdoc/>
    public virtual Dictionary<string, T> GetFeatureImportance() => new(StringComparer.Ordinal);

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
