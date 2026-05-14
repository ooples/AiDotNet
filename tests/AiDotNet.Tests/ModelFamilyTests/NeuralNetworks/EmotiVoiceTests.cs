using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech.StyleEmotion;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for EmotiVoice (style-/emotion-controlled
/// end-to-end TTS). The auto-generator can't emit this scaffold
/// because EmotiVoice's constructors require either an ONNX model file
/// or an explicit <see cref="NeuralNetworkArchitecture{T}"/> +
/// <see cref="EmotiVoiceOptions"/>; neither satisfies the
/// parameterless-ctor rule.
/// </summary>
/// <remarks>
/// Paper-faithful configuration: <see cref="EmotiVoiceOptions"/>'s own
/// defaults reflect the published TTS hyperparameters
/// (NumEncoderLayers=6, NumDecoderLayers=4, NumHeads=2, DropoutRate=0.1,
/// EmotionDim=128, NumEmotionLayers=3) on top of the
/// <see cref="AiDotNet.TextToSpeech.EndToEnd.EndToEndTtsOptions"/>
/// shared TTS defaults (EncoderDim=192, DecoderDim=192, InterChannels=192,
/// FilterChannels=768). Do not override them — per the project's
/// never-shrink-tests rule, slow or saturating tests at paper scale
/// are model-side performance bugs to fix in the model code.
/// </remarks>
public class EmotiVoiceTests : TTSModelTestBase
{
    // EmotiVoice's text-encoder front end expects a tokenized text
    // sequence projected to EncoderDim=192. Use a paper-aligned shape
    // [batch, sequence_len, encoderDim] = [1, 128, 192] — 128 tokens
    // is the common phoneme-length context, 192 is the encoder
    // embedding dim shared across EmotiVoice / VITS / VITS2 TTS
    // variants per EndToEndTtsOptions.EncoderDim.
    protected override int[] InputShape => [1, 128, 192];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        // Architecture's input size mirrors the encoder context length;
        // outputSize is the mel-channel count (80 per paper convention,
        // matching TtsModelBase.MelChannels).
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 128 * 192,
            outputSize: 80);

        // Defaults intentional — see <remarks> above.
        return new EmotiVoice<double>(architecture, new EmotiVoiceOptions());
    }
}
