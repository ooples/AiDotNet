using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Complete four-codebook editor, real EnCodec and source-recorded phonemes for smoke utterances.</summary>
internal static class VoiceCraftNativeFixture
{
    internal static VoiceCraft<T> Create<T>()
    {
        var options = VoiceCraftPaperTests.SmallOptions();
        options.MaxTextLength = 128;
        options.MaxCodecFrames = 16;
        var model = new VoiceCraft<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 640) { RandomSeed = 2294 }, options);
        var operations = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var recording = new Tensor<T>(new[] { 640 }, new Vector<T>(Enumerable.Range(0, 640)
            .Select(index => operations.FromDouble(.25 * Math.Sin(index * .07) + .05 * Math.Cos(index * .031))).ToArray()));
        model.Voice = model.CreateVoice(recording, "hello");
        return model;
    }
}
