using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// The architecture every detection backbone hands to <see cref="NeuralNetworkBase{T}"/>.
/// </summary>
/// <remarks>
/// A backbone is itself a <see cref="NeuralNetworkBase{T}"/>, and that constructor restarts the
/// layer-initialization seed scope from its architecture's seed. Built inside a detector's constructor,
/// a backbone with no seed of its own therefore reset the detector's armed scope to nothing: none of its
/// layers took a seed, and two detectors built from equal options differed in 96-99.9% of their weights
/// (#2201). Drawing the backbone's seed from the enclosing scope makes it seed exactly as a layer does:
/// derived from the detector's seed when one is armed, and unseeded when none is.
/// </remarks>
internal static class DetectionBackboneArchitecture<T>
{
    /// <summary>Creates a spatially dynamic NCHW architecture seeded from the enclosing scope.</summary>
    /// <param name="inChannels">The number of input image channels.</param>
    internal static NeuralNetworkArchitecture<T> Create(int inChannels)
    {
        var architecture = NeuralNetworkArchitecture<T>.CreateDynamicSpatial(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            channels: inChannels,
            outputSize: 1);
        architecture.RandomSeed = LayerInitializationSeedScope.NextSeedOrNull();
        return architecture;
    }
}