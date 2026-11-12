using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class SeedTtsOptionFixture
{
    private static bool IsTokenizerOption(string option) => option is "TrainingMelWindows" or "TrainingMelBins"
        or "DiscriminatorWidthDivisor" or "DiscriminatorChannels" or "DiscriminatorWindows";

    private static bool IsDiffusionOption(string option) => option is "DiffusionSteps" or "DiffusionTrainingSteps"
        or "DiffusionBetaStart" or "DiffusionBetaEnd";

    internal static void PrepareOptions(SeedTTSOptions options, string option)
    {
        options.SampleRate = 800;
        options.CodecFrameRate = 100;
        options.CodecRates = new[] { 2, 2, 2 };
        options.TokenizerChannels = 4;
        options.TokenizerCodeDimension = 2;
        options.AcousticDimension = 8;
        options.DiffusionLayers = 1;
        options.VocoderHiddenDimension = 16;
        options.TrainingMelWindows = new[] { 16, 32 };
        options.TrainingMelBins = new[] { 4, 8 };
        options.DiscriminatorWindows = new[] { 64 };
        options.DiscriminatorChannels = 2;
        options.DiscriminatorWidthDivisor = 32;
        options.TrainingComponent = IsTokenizerOption(option) ? SeedTTSTrainingComponent.SpeechTokenizer
            : IsDiffusionOption(option) ? SeedTTSTrainingComponent.TokenDiffusion : SeedTTSTrainingComponent.TokenLanguageModel;
        if (option is "FramesPerTextToken" or "MinimumGeneratedCodecFrames") options.MaxCodecFrames = 64;
        if (option == "TopK") options.TopK = 4;
        if (option == "MaxPromptCodecFrames") options.MaxPromptCodecFrames = 4;
    }

    internal static void Train<T>(SeedTTS<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        Tensor<T> Ids(params int[] ids) => new(new[] { ids.Length }, new Vector<T>(ids.Select(id => ops.FromDouble(id)).ToArray()));
        var audio = new Tensor<T>(new[] { 24 });
        for (int i = 0; i < audio.Length; i++) audio[i] = ops.FromDouble(.3 * Math.Sin(.4 * i) + .1 * Math.Cos(.17 * i));
        model.Train(new TtsTrainingSample<T>
        {
            Text = "abc", Tokens = Ids(3, 4, 5), CodecTokens = Ids(1, 2, 3),
            PromptCodecTokens = Ids(4, 5, 6), Audio = audio, SpeakerReference = audio,
        });
    }

    internal static long? Observe<T>(SeedTTS<T> model, string option)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (IsDiffusionOption(option))
        {
            var codes = new Tensor<T>(new[] { 3 }, new Vector<T>(new[] { ops.One, ops.FromDouble(2), ops.FromDouble(3) }));
            return TtsPaperNumerics.Fingerprint(model.DecodeFromTokens(codes).Select(ops.ToDouble));
        }
        if (option is not ("Temperature" or "TopK" or "FramesPerTextToken" or "MinimumGeneratedCodecFrames" or "MaxPromptCodecFrames")) return null;
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var language = (SeedTTSTokenLanguageModel<T>)typeof(SeedTTS<T>).GetField("_tokenLanguageModel", flags)!.GetValue(model)!;
        var values = new List<double>();
        for (int seed = 0; seed < 16; seed++)
        {
            var tokens = language.Generate(new[] { 3, 4, 5 }, option == "MaxPromptCodecFrames" ? new[] { 1, 2, 3, 4, 5, 6 } : null, new Random(1337 + seed));
            values.Add(tokens.Length);
            values.AddRange(tokens.Select(token => (double)token));
        }
        return TtsPaperNumerics.Fingerprint(values);
    }
}
