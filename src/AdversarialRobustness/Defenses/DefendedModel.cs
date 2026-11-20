using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.AdversarialRobustness.Defenses;

/// <summary>
/// The model as an attack sees it when a defense is configured: predictions (and gradients computed from
/// them) include the defense's input preprocessing; everything else is the wrapped model's.
/// </summary>
[AiDotNet.Attributes.ModelMetadataExempt]
internal sealed class DefendedModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    private readonly IFullModel<T, TInput, TOutput> _inner;
    private readonly IAdversarialDefense<T, TInput, TOutput> _defense;

    public DefendedModel(IFullModel<T, TInput, TOutput> inner, IAdversarialDefense<T, TInput, TOutput> defense)
    {
        _inner = inner;
        _defense = defense;
    }

    public ILossFunction<T> DefaultLossFunction => _inner.DefaultLossFunction;

    public long ParameterCount => InterfaceGuard.Parameterizable(_inner).ParameterCount;

    public bool SupportsParameterInitialization => InterfaceGuard.Parameterizable(_inner).SupportsParameterInitialization;

    public Vector<T> SanitizeParameters(Vector<T> parameters) => InterfaceGuard.Parameterizable(_inner).SanitizeParameters(parameters);

    public TOutput Predict(TInput input) => _inner.Predict(_defense.PreprocessInput(input));

    public void Train(TInput input, TOutput expectedOutput) => _inner.Train(_defense.PreprocessInput(input), expectedOutput);

    public ModelMetadata<T> GetModelMetadata() => _inner.GetModelMetadata();

    public byte[] Serialize() => _inner.Serialize();

    public void Deserialize(byte[] data) => _inner.Deserialize(data);

    public void SaveModel(string filePath) => _inner.SaveModel(filePath);

    public void LoadModel(string filePath) => _inner.LoadModel(filePath);

    public void SaveState(Stream stream) => _inner.SaveState(stream);

    public void LoadState(Stream stream) => _inner.LoadState(stream);

    public Vector<T> GetParameters() => InterfaceGuard.Parameterizable(_inner).GetParameters();

    public void SetParameters(Vector<T> parameters) => InterfaceGuard.Parameterizable(_inner).SetParameters(parameters);

    public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
        => new DefendedModel<T, TInput, TOutput>(InterfaceGuard.Parameterizable(_inner).WithParameters(parameters), _defense);

    public IEnumerable<int> GetActiveFeatureIndices() => InterfaceGuard.FeatureAware(_inner).GetActiveFeatureIndices();

    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) => InterfaceGuard.FeatureAware(_inner).SetActiveFeatureIndices(featureIndices);

    public bool IsFeatureUsed(int featureIndex) => InterfaceGuard.FeatureAware(_inner).IsFeatureUsed(featureIndex);

    public Dictionary<string, T> GetFeatureImportance() => _inner.GetFeatureImportance();

    public IFullModel<T, TInput, TOutput> DeepCopy() => new DefendedModel<T, TInput, TOutput>(_inner.DeepCopy(), _defense);

    public IFullModel<T, TInput, TOutput> Clone() => new DefendedModel<T, TInput, TOutput>(_inner.Clone(), _defense);

    public Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
        => InterfaceGuard.GradientComputable(_inner).ComputeGradients(_defense.PreprocessInput(input), target, lossFunction);

    public void ApplyGradients(Vector<T> gradients, T learningRate) => InterfaceGuard.GradientComputable(_inner).ApplyGradients(gradients, learningRate);

    // The wrapped model belongs to the result, which disposes it; the view owns nothing.
    public void Dispose()
    {
        System.GC.SuppressFinalize(this);
    }
}
