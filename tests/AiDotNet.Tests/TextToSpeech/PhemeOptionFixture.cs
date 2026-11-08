using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.Tests.TextToSpeech;

internal static class PhemeOptionFixture
{
    internal static bool IsTrainingOption(string option) => option is "TextWarmupSteps" or "TextTrainingSteps"
        or "AcousticWarmupSteps" or "AcousticTrainingSteps" or "WeightDecay" or "SpeakerEmbeddingDropout";

    internal static bool IsInferenceOption(string option) => option is "AcousticDecoding" or "MaskGitSteps" or "MaxTextLength";

    internal static void PrepareOptions(PhemeOptions options, string option)
    {
        if (IsInferenceOption(option))
        {
            options.UseSpeakerEmbedding = false;
            if (option == "MaxTextLength") options.MaxTextLength = 2;
            return;
        }
        if (!IsTrainingOption(option)) return;
        options.TextWarmupSteps = options.AcousticWarmupSteps = 2;
        options.TextTrainingSteps = options.AcousticTrainingSteps = 4;
        options.UseSpeakerEmbedding = option == "SpeakerEmbeddingDropout";
        if (option is "TextTrainingSteps" or "AcousticTrainingSteps")
            options.TextWarmupSteps = options.AcousticWarmupSteps = 0;
    }

    internal static long? Observe<T>(Pheme<T> model, string option)
    {
        if (!IsInferenceOption(option)) return null;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option == "MaxTextLength")
            return TtsPaperNumerics.Fingerprint(model.TextToTokens("hello there").Select(ops.ToDouble));
        var options = (PhemeOptions)model.GetOptions();
        var prompt = new int[options.AcousticCodebooks + 1, 3];
        for (int q = 0; q < prompt.GetLength(0); q++)
            for (int f = 0; f < prompt.GetLength(1); f++) prompt[q, f] = (q + f) % options.CodebookSize;
        var semantic = Enumerable.Range(0, 8).Select(i => i % options.SemanticCodes).ToArray();
        var generate = typeof(Pheme<T>).GetMethod("GenerateAcoustic", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        bool training = model.IsTrainingMode;
        model.SetTrainingMode(false);
        try
        {
            var values = new List<double>();
            for (int seed = 0; seed < 16; seed++)
            {
                var codes = (int[,])generate.Invoke(model, new object?[] { prompt, semantic, null, new Random(options.SamplingSeed + seed) })!;
                values.Add(codes.GetLength(1));
                values.AddRange(codes.Cast<int>().Select(code => (double)code));
            }
            return TtsPaperNumerics.Fingerprint(values);
        }
        finally { model.SetTrainingMode(training); }
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
