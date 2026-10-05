using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A layer that refers to another layer of the same network, such as a language-model head tied to its token
/// embedding, and must be pointed at that layer's current instance whenever the network's layer list is built or
/// replaced (construction, clone, deserialization).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal interface ILayerGraphBinding<T>
{
    /// <summary>Resolves this layer's reference against the network's final layer list.</summary>
    /// <param name="layers">The network's canonical layers, in order.</param>
    void BindToLayerGraph(IReadOnlyList<ILayer<T>> layers);
}