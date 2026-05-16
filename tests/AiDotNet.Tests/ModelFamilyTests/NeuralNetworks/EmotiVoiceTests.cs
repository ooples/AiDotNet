using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech.StyleEmotion;
using AiDotNet.Tensors;
using Xunit;

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
/// defaults mirror the upstream EmotiVoice joint config: segment_size=32,
/// n_mels=80, encoder_n_hidden=384, decoder_n_hidden=384,
/// encoder_n_heads=8, decoder_n_heads=8, encoder_n_layers=4,
/// and decoder_n_layers=4.
/// Do not override them; slow or saturating tests at paper scale are
/// model-side performance bugs to fix in the model code.
/// </remarks>
public class EmotiVoiceTests : TTSModelTestBase
{
    // Regression for #1311: mel/prosody-width input (80 channels) must be
    // projected into EmotiVoice's 384-d hidden state before the first MHA.
    protected override int[] InputShape => [1, 32, 80];
    protected override int[] OutputShape => [1, 32, 80];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        // Architecture's input size mirrors the mel/prosody feature grid;
        // outputSize is the mel-channel count (80 per upstream config).
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 32 * 80,
            outputSize: 80);

        return new EmotiVoice<double>(architecture, new EmotiVoiceOptions());
    }

    [Fact]
    public void DefaultOptions_MatchUpstreamEmotiVoiceConfig()
    {
        var options = new EmotiVoiceOptions();

        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(80, options.MelChannels);
        Assert.Equal(384, options.HiddenDim);
        Assert.Equal(384, options.EncoderDim);
        Assert.Equal(384, options.DecoderDim);
        Assert.Equal(384, options.EmotionDim);
        Assert.Equal(4, options.NumEncoderLayers);
        Assert.Equal(4, options.NumDecoderLayers);
        Assert.Equal(8, options.NumHeads);
        Assert.Equal(0.2, options.DropoutRate, precision: 6);
        Assert.Equal(1.25e-5, options.LearningRate, precision: 10);
        Assert.Equal(0.0, options.WeightDecay, precision: 10);
        Assert.Equal(0.5, options.OptimizerBeta1, precision: 6);
        Assert.Equal(0.9, options.OptimizerBeta2, precision: 6);
        Assert.Equal(1e-9, options.OptimizerEpsilon, precision: 12);
        Assert.Equal(0.999875, options.LearningRateSchedulerGamma, precision: 6);
    }

    [Fact]
    public void Predict_MelWidthInput_ProjectsToHiddenAndReturnsMelChannels()
    {
        using var network = CreateNetwork();
        var input = new Tensor<double>(InputShape);

        var output = network.Predict(input);

        Assert.Equal(3, output.Rank);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(32, output.Shape[1]);
        Assert.Equal(80, output.Shape[2]);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(output[i]), $"Output[{i}] is infinite.");
        }
    }
}
