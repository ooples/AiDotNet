using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// #2087 / #1706: out of the HeavyTimeout lane. ViT-Base at 224x224 is GEMM-bound (~44 s per
// iteration), past the 120 s gate. Per the #1706 out-of-memory ladder the test caps the image at
// 32x32 and shrinks the encoder to 64 wide, 2 layers, 2 heads and a 256 MLP; the paper's 16x16
// patches, 3 channels and 1,000-class head are unchanged.
public class VisionTransformerTests : NeuralNetworkModelTestBase<float>
{
    // ViT default architecture is [batch, 3, 224, 224] (paper-faithful
    // ImageNet-1k geometry). The parameterless ctor hard-codes this and
    // rejects any other resolution at Predict, so the test base must
    // submit a 224x224 input to match.
    protected override int[] InputShape => [1, 3, ImageSize, ImageSize];
    protected override int[] OutputShape => [1000];

    private const int ImageSize = 32;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new VisionTransformer<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.MultiClassClassification,
                inputHeight: ImageSize,
                inputWidth: ImageSize,
                inputDepth: 3,
                outputSize: 1000),
            imageHeight: ImageSize, imageWidth: ImageSize, channels: 3, patchSize: 16, numClasses: 1000,
            options: new VisionTransformerOptions { HiddenDim = 64, NumLayers = 2, NumHeads = 2, MlpDim = 256 });
}
