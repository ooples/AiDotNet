using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.Unified;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for Transfusion (Zhou et al. 2024, "Transfusion:
/// Predict the Next Token and Diffuse Images with One Multi-Modal Model").
/// The auto-generator can't emit this scaffold because Transfusion's
/// constructors require either an ONNX model file or an explicitly-built
/// <see cref="NeuralNetworkArchitecture{T}"/> + <see cref="TransfusionOptions"/>;
/// neither path satisfies the parameterless-ctor or all-defaulted-ctor
/// rule the generator uses for auto-construction.
/// </summary>
/// <remarks>
/// Production defaults remain the paper's 4096-wide, 32-layer configuration. This conformance
/// fixture uses the same patch/fusion/decoder topology at public-options smoke scale; the exact-model
/// performance census records latency and memory independently. The former full-scale fixture reached
/// 17.3 GiB and never completed one cold forward in 90 seconds, so it tested no behavior at all.
/// </remarks>
public class TransfusionTests : VisionLanguageTestBase<float>
{
    protected override int[] InputShape => [1, 3, 32, 32];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 64);

        return new Transfusion<float>(architecture, new TransfusionOptions
        {
            ImageSize = 32,
            OutputImageSize = 32,
            VisionDim = 32,
            DecoderDim = 32,
            NumVisionLayers = 0,
            NumDecoderLayers = 2,
            NumHeads = 4,
            VocabSize = 64,
            MaxSequenceLength = 16,
            MaxGenerationLength = 8,
            NumVisualTokens = 16,
            DropoutRate = 0.0,
        });
    }
}
