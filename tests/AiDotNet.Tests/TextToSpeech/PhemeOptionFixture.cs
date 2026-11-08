using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.Tests.TextToSpeech;

internal static class PhemeOptionFixture
{
    internal static bool IsTrainingOption(string option) => option is "TextWarmupSteps" or "TextTrainingSteps"
        or "AcousticWarmupSteps" or "AcousticTrainingSteps" or "WeightDecay" or "SpeakerEmbeddingDropout";

    internal static void PrepareOptions(PhemeOptions options, string option)
    {
        if (!IsTrainingOption(option)) return;
        options.TextWarmupSteps = options.AcousticWarmupSteps = 2;
        options.TextTrainingSteps = options.AcousticTrainingSteps = 4;
        options.UseSpeakerEmbedding = option == "SpeakerEmbeddingDropout";
        if (option is "TextTrainingSteps" or "AcousticTrainingSteps")
            options.TextWarmupSteps = options.AcousticWarmupSteps = 0;
    }

    internal static void Train<T>(Pheme<T> model, string option)
    {
        model.CurrentStage = option is "TextWarmupSteps" or "TextTrainingSteps"
            ? PhemeTrainingStage.TextToSemantic : PhemeTrainingStage.SemanticToAcoustic;
        var options = (PhemeOptions)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var codes = new Tensor<T>(new[] { 6, options.AcousticCodebooks + 1 });
        for (int f = 0; f < codes.Shape[0]; f++)
            for (int q = 0; q < codes.Shape[1]; q++) codes[f, q] = ops.FromDouble((3 * f + 5 * q) % options.CodebookSize);
        var phonemes = model.EncodePhonemes(EnglishG2P.Default.Phonemize("hello there"));
        var recording = new Tensor<T>(new[] { 8000 });
        for (int i = 0; i < recording.Length; i++) recording[i] = ops.FromDouble(.4 * Math.Sin(2 * Math.PI * 180 * i / options.SampleRate));
        model.Train(new TtsTrainingSample<T>
        {
            Tokens = new Tensor<T>(new[] { phonemes.Length }, new Vector<T>(phonemes.Select(id => ops.FromDouble(id)).ToArray())),
            CodecTokens = codes, SpeakerReference = recording,
        });
    }
}
