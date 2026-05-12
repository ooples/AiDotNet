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
/// Paper-faithful configuration: <see cref="TransfusionOptions"/>'s own
/// defaults reflect the paper's hyperparameters (ImageSize=224, DecoderDim,
/// NumVisionLayers, NumHeads, VocabSize, etc.). Do not override them —
/// per the project's never-shrink-tests rule, slow or saturating tests
/// at paper scale are model-side performance bugs to be fixed in the
/// model code, not papered over here.
/// </remarks>
public class TransfusionTests : VisionLanguageTestBase
{
    // Paper-faithful image size (224×224 RGB per Transfusion paper §3).
    // Transfusion's VisionLanguageModelBase contract treats input as
    // [batch, channels=3, height, width].
    protected override int[] InputShape => [1, 3, 224, 224];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        // VisionLanguageModelBase wants a real NeuralNetworkArchitecture
        // even in native mode. Match the architecture's image dims to
        // TransfusionOptions's paper defaults (ImageSize=224, 3 channels).
        // OutputSize is left at the paper's VocabSize-derived next-token
        // classification head; the architecture's actual layer
        // construction is delegated to TransfusionOptions.
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 224,
            inputWidth: 224,
            inputDepth: 3,
            outputSize: 512);

        // Defaults intentional — see <remarks> above.
        return new Transfusion<double>(architecture, new TransfusionOptions());
    }
}
