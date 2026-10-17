using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// UNet3D runs a caller-supplied layer list as given. The encoder/decoder skip topology belongs to the layers it generates
/// itself, and is chosen by that origin, not by the list's length: a caller list that happens to have the generated layer
/// count (24 at two encoder blocks) must not be indexed as encoder units, pools and skip concatenations.
/// </summary>
public class UNet3DCallerLayersTests
{
    [Fact]
    public void CallerLayers_WithTheGeneratedLayerCount_RunSequentially()
    {
        const int generatedCountAtTwoBlocks = (3 * 2 * 2) + (2 - 1) + 3 + (7 * (2 - 1)) + 1;
        var layers = new List<ILayer<float>>();
        for (int i = 0; i < generatedCountAtTwoBlocks; i++)
            layers.Add(new ActivationLayer<float>((IActivationFunction<float>)new ReLUActivation<float>()));
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputHeight: 4,
            inputWidth: 4,
            inputDepth: 4,
            outputSize: 2,
            layers: layers);
        var network = new UNet3D<float>(architecture,
            options: new UNet3DOptions { VoxelResolution = 4, NumEncoderBlocks = 2, BaseFilters = 4 });

        var input = new Tensor<float>(new[] { 2, 4, 4, 4 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)System.Math.Sin(i * 0.61);
        var output = network.Forward(input);

        // Twenty-four ReLUs in a row are one ReLU. Read as the generated topology, the skip concatenation would have
        // doubled the channel axis.
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        for (int i = 0; i < input.Length; i++)
            Assert.Equal(System.Math.Max(0f, input[i]), output[i]);
    }
}
