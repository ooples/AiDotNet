using AiDotNet.ComputerVision.Segmentation.Panoptic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for ODISE. The auto-generator emits NotImplementedException
/// for ctor-required models; this manual class supplies the ctor args explicitly.
/// Per Xu et al. 2023 §3 ("Open-Vocabulary Panoptic Segmentation with Text-to-
/// Image Diffusion Models") ODISE takes an NCHW image and produces a per-pixel
/// class map.
/// </summary>
public class ODISETests : NeuralNetworkModelTestBase
{
    // Smaller spatial dim (32 vs paper's 512) keeps the test fast while still
    // exercising the diffusion-feature + mask-classifier pipeline.
    private const int Channels = 3;
    private const int Height = 32;
    private const int Width = 32;
    private const int NumClasses = 8;

    protected override int[] InputShape => [1, Channels, Height, Width];
    protected override int[] OutputShape => [1, NumClasses, Height, Width];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: Height,
            inputWidth: Width,
            inputDepth: Channels,
            outputSize: NumClasses);
        return new ODISE<double>(arch, numClasses: NumClasses);
    }
}
