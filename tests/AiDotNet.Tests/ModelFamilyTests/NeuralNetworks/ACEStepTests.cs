using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for ACE-Step (Chen et al. 2024, "ACE-Step: A Step
/// Towards Music Generation Foundation Model"). The auto-generator can't
/// emit this scaffold because ACEStep's only constructors require either
/// an ONNX model path (file existence check) or a configured
/// <see cref="NeuralNetworkArchitecture{T}"/> + <see cref="ACEStepOptions"/>.
/// </summary>
/// <remarks>
/// Paper-faithful configuration: <see cref="ACEStepOptions"/>'s own
/// defaults already match the paper (SampleRate=44100, NumChannels=2,
/// LatentDim=128, UNetDim=512, NumUNetLayers=4, NumSteps=4,
/// TextEncoderDim=768). Do not override them — per the project's
/// never-shrink-tests rule, slow or saturating tests at paper scale
/// are model-side performance bugs to be fixed in the model code, not
/// papered over here.
/// </remarks>
public class ACEStepTests : AudioNNModelTestBase
{
    // Paper-faithful 1-second stereo audio at 44.1 kHz. The model
    // contract is [batch, channels, samples] per AudioNeuralNetworkBase.
    // One second is the minimum meaningful evaluation clip — anything
    // smaller doesn't exercise the diffusion stack's temporal context.
    protected override int[] InputShape => [1, 2, 44100];
    protected override int[] OutputShape => [1, 2, 44100];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        // Architecture matches the audio I/O contract; ACEStepOptions
        // carries the paper-default diffusion-stack hyperparameters via
        // its own field initializers.
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 44100,
            outputSize: 44100);

        // Defaults intentional — see <remarks> above.
        return new ACEStep<double>(architecture, new ACEStepOptions());
    }
}
