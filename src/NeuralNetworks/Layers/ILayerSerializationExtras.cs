using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Provides additional, optional parameter blocks for serialization that are not part of <see cref="ILayer{T}.ParameterCount"/>.
/// </summary>
/// <typeparam name="T">Numeric type for the layer.</typeparam>
/// <remarks>
/// This exists to support layers where <see cref="ILayer{T}.ParameterCount"/> intentionally reflects trainable parameters
/// (e.g., frozen base weights in LoRA adapters) but full model serialization/cloning must still preserve non-trainable state.
/// </remarks>
internal interface ILayerSerializationExtras<T>
{
    int ExtraParameterCount { get; }

    Vector<T> GetExtraParameters();

    void SetExtraParameters(Vector<T> extraParameters);
}

