using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class SoundStormOptionFixture
{
    internal static bool IsTrainingOption(string option) => option is "DropoutRate" or "WeightDecay"
        or "WarmupStartLearningRate" or "WarmupSteps" or "TrainingSteps";

    internal static bool IsInferenceOption(string option) => option is "TopK" or "Temperature" or "SamplingSeed" or "FirstLevelIterations";

    internal static void PrepareOptions(SoundStormOptions options, string option)
    {
        if (!IsTrainingOption(option)) return;
        options.WarmupSteps = option == "TrainingSteps" ? 0 : 2;
        options.TrainingSteps = 4;
    }

    internal static void Train<T>(SoundStorm<T> model, string option)
    {
        var options = (SoundStormOptions)model.GetOptions();
        model.CurrentStage = option == "DropoutRate" ? SoundStormTrainingStage.SemanticToAcoustic : SoundStormTrainingStage.TextToSemantic;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var codes = new Tensor<T>(new[] { 12, options.NumCodebooks + 1 });
        for (int f = 0; f < codes.Shape[0]; f++)
            for (int q = 0; q < codes.Shape[1]; q++)
                codes[f, q] = ops.FromDouble((3 * f + 5 * q + 1) % (q == 0 ? options.SemanticCodes : options.CodebookSize));
        model.Train(new TtsTrainingSample<T> { Tokens = model.TextToTokens("hello there"), CodecTokens = codes });
    }

    internal static long? Observe<T>(SoundStorm<T> model, string option)
    {
        if (!IsInferenceOption(option)) return null;
        var options = (SoundStormOptions)model.GetOptions();
        var values = new List<double>();
        bool training = model.IsTrainingMode;
        model.SetTrainingMode(false);
        try
        {
            using var scope = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            for (int seed = 0; seed < 16; seed++)
            {
                var random = new Random(options.SamplingSeed + seed);
                if (option == "FirstLevelIterations")
                {
                    var codes = new int[options.NumCodebooks, 12];
                    for (int q = 0; q < codes.GetLength(0); q++)
                        for (int f = 0; f < codes.GetLength(1); f++) codes[q, f] = model.AcousticCore!.MaskId;
                    var semantic = Enumerable.Range(0, 12).Select(i => i % options.SemanticCodes).ToArray();
                    model.AcousticCore!.Decode(codes, semantic, 0, level => level == 0 ? options.FirstLevelIterations : 1, null, random);
                    values.AddRange(codes.Cast<int>().Select(code => (double)code));
                }
                else
                {
                    var tokens = model.TextModel!.Generate(model.TextToTokens("hello there"), Array.Empty<int>(),
                        options.Temperature, options.TopK, options.MaxNewSemanticTokens, random,
                        (token, step) => token >= 384 || (token == 1 && step > 0));
                    values.Add(tokens.Count);
                    values.AddRange(tokens.Select(token => (double)token));
                }
            }
            return TtsPaperNumerics.Fingerprint(values);
        }
        finally { model.SetTrainingMode(training); }
    }
}
