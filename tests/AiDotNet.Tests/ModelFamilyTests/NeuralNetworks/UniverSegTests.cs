using AiDotNet.ComputerVision.Segmentation.Medical;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for UniverSeg. UniverSeg (Butoi et al. 2023) is <c>f(x_t, S)</c>, defined only for a
/// support set of at least one labelled pair, so a model without one rejects predictions; this scaffold gives every
/// fixture a fixed two-pair support set (part of the model's saved state, so copies and reloads keep it).
/// </summary>
public class UniverSegTests : NeuralNetworkModelTestBase<float>
{
    private const int Channels = 1;
    private const int Height = 32;
    private const int Width = 32;
    private const int SupportSize = 2;

    protected override int[] InputShape => [1, Channels, Height, Width];
    protected override int[] OutputShape => [1, 1, Height, Width];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: Height,
            inputWidth: Width,
            inputDepth: Channels,
            outputSize: 1);
        var model = new UniverSeg<float>(arch);
        model.SetSupportSet(Pattern(11, SupportSize, Channels, Height, Width), Pattern(12, SupportSize, Height, Width));
        return model;
    }

    private static Tensor<float> Pattern(int seed, params int[] shape)
    {
        int total = 1;
        foreach (int s in shape) total *= s;
        var data = new float[total];
        var random = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < total; i++) data[i] = (float)random.NextDouble();
        return new Tensor<float>(shape, new Vector<float>(data));
    }
}
