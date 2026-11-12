using AiDotNet.TextToSpeech.MultiModal;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.TextToSpeech;

internal static class GlmVoiceOptionFixture
{
    internal static bool IsTokenizerOption(string option) => option.StartsWith("Asr", StringComparison.Ordinal)
        || option is "EmaDecay" or "QuantizeLossScale" or "CommitCoefficient" or "SpeechTokenizerLayers" or "NumEncoderLayers" or "RestartInterval";

    internal static void PrepareOptions(GLM4VoiceOptions options, string option)
    {
        if (option is "LearningRate" or "FineTuningDropout" or "TextEpochs" or "SpeechEpochs") options.Stage = GlmVoiceTrainingStage.FineTuning;
        if (IsTokenizerOption(option)) options.IncludeSpeechTokenizerTraining = true;
        if (option == "TrainingCfgRate") options.TrainingCfgRate = 0;
    }

    internal static bool TryTrain<T>(GLM4Voice<T> model, string option)
    {
        if (IsTokenizerOption(option))
        {
            model.TrainSpeechTokenizer(GLM4VoiceFixture.Tones<T>(option == "RestartInterval" ? 1.2 : 0.6, 16000), new[] { 5, 9, 3, 7, 2 });
            return true;
        }
        if (option is "TextEpochs" or "SpeechEpochs")
        {
            model.TrainDialogue(new GlmVoiceDialogueTurn
            {
                UserText = "hello", ResponseText = "hello there", ResponseSpeechTokens = new[] { 1, 2, 3 },
            }, epoch: 1);
            return true;
        }
        if (option is "FlowLoss" or "StreamingTruncation" or "TrainingCfgRate")
        {
            model.TrainSpeechDecoder(Enumerable.Range(0, 12).Select(i => i % GLM4VoiceFixture.Codes).ToArray(),
                GLM4VoiceFixture.Tones<T>(1.0, 22050));
            return true;
        }
        return false;
    }

    internal static long? Observe<T>(GLM4Voice<T> model, string option)
    {
        if (!IsTokenizerOption(option)) return null;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        // EMA counts and accumulated vectors determine the codebook used by the speech frontend.
        return OptionSensitivityHarness.Fingerprint(model.SpeechTokenizer.Codebook.GetParameters().Select(ops.ToDouble)
            .Concat(model.SpeechTokenizer.EmaCount).Concat(model.SpeechTokenizer.EmaWeight));
    }
}
