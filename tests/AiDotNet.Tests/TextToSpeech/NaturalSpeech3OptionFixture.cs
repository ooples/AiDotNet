using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class NaturalSpeech3OptionFixture
{
    internal static bool IsCodecOption(string option) => option is
        "MelBins" or "MelWindows" or "SpeakerReversalWeight" or "DetachProsodyContent" or
        "ReconstructionWeight" or "PhoneWeight" or "PhoneReversalWeight" or "CodecLearningRate" or
        "AdversarialWeight" or "CodecBeta1" or "CodecBeta2" or "F0Weight" or "FeatureWeight" or
        "F0ReversalWeight" or "CodebookWeight" or "DetailDropout" or "CommitmentWeight";

    internal static void PrepareOptions(NaturalSpeech3Options options, string option)
    {
        if (IsCodecOption(option)) options.CurrentStage = NaturalSpeech3Stage.Codec;
    }

    internal static void Train<T>(NaturalSpeech3<T> model)
    {
        var options = (NaturalSpeech3Options)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        const int frames = 20;
        var audio = new Tensor<T>(new[] { frames * options.HopSize });
        for (int i = 0; i < audio.Length; i++)
            audio[i] = ops.FromDouble(.3 * Math.Sin(2 * Math.PI * .05 * i) + .1 * Math.Sin(2 * Math.PI * .21 * i));
        var tokens = new Tensor<T>(new[] { 8 });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = ops.FromDouble(1 + i);
        model.Train(new TtsTrainingSample<T>
        {
            Tokens = tokens,
            Durations = new[] { 3, 2, 4, 1, 3, 2, 3, 2 },
            Audio = audio,
            Pitch = Enumerable.Range(0, frames).Select(f => f % 5 == 4 ? 0.0 : 120 + 10 * Math.Sin(f)).ToArray(),
            SpeakerId = 1,
        });
    }
}
