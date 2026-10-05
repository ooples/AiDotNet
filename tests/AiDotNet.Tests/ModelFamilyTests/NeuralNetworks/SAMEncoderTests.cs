using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// The vision-language model-family invariants for <see cref="SAM{T}"/> in AiDotNet.VisionLanguage.Encoders.
/// </summary>
/// <remarks>
/// The generated "SAMTests" class belongs to AiDotNet.ComputerVision.Segmentation.Foundation.SAM (see the
/// generator's CollisionOwners), so this namesake gets no generated class. This scaffold covers it by type
/// through its CreateNetwork override. Its name deliberately does not match "SAM" + a test suffix, so it never
/// claims the owner's name and the segmentation model's generated suite keeps running.
/// </remarks>
public class SAMEncoderTests : VisionLanguageTestBase<double>
{
    protected override int[] InputShape => [1, 3, 32, 32];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 32);

        // The paper's ViT-B image encoder (1024px, 768 wide, 12 layers, 12 heads, 16px patches) at test scale:
        // the same patch-embedding ViT, with 4 patches per side.
        var options = new SAMOptions
        {
            ImageSize = 32,
            PatchSize = 8,
            EmbeddingDim = 32,
            NumLayers = 2,
            NumHeads = 4,
            DropoutRate = 0.0,
        };
        return new SAM<double>(architecture, options);
    }
}
