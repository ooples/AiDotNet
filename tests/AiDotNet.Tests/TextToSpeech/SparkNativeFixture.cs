using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Native Qwen, XLSR and BiCodec, with explicit grammar-constrained smoke sampling.</summary>
internal static class SparkNativeFixture
{
    internal static bool IsAnnotatedOption(string option) => option is
        "GenderTokenOffset" or "PitchLabelOffset" or "SpeedLabelOffset" or "PitchValueOffset" or "SpeedValueOffset"
        or "TaskControllableTtsToken" or "StartStyleLabelToken" or "EndStyleLabelToken"
        or "StartStylePromptToken" or "EndStylePromptToken";

    internal static void PrepareOptions(SparkTTSOptions options, string option)
    {
        if (option.StartsWith("Discriminator", StringComparison.Ordinal)
            || option.StartsWith("TrainingMel", StringComparison.Ordinal)
            || option is "MelLossWeight" or "FeatureMatchingWeight" or "SemanticCodebookWeight"
                or "SemanticCommitmentWeight" or "GlobalTokenizerWarmupSteps")
            options.TrainingComponent = SparkTTSTrainingComponent.BiCodec;
        if (option == "GlobalTokenizerWarmupSteps") options.GlobalTokenizerWarmupSteps = 0;
    }

    internal static Tensor<T> Recording<T>()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var recording = new Tensor<T>(new[] { 320 });
        for (int i = 0; i < recording.Length; i++)
            recording[i] = ops.FromDouble(0.4 * Math.Sin(0.17 * i) + 0.1 * Math.Cos(0.037 * i));
        return recording;
    }

    internal static void Prepare<T>(SparkTTS<T> model) => model.EncodeToTokens(Recording<T>());

    internal static long? Observe<T>(SparkTTS<T> model, string option)
    {
        if (option != "Wav2VecHeads") return null;
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var codec = typeof(SparkTTS<T>).GetField("_biCodec", flags)!.GetValue(model)!;
        var features = (Tensor<T>)codec.GetType().GetMethod("MixedWav2VecFeatures", flags)!.Invoke(codec, new object[] { Recording<T>() })!;
        var ops = MathHelper.GetNumericOperations<T>();
        return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(features.Select(ops.ToDouble));
    }

    internal static SparkTTS<T> Create<T>()
    {
        var options = SparkTTSPaperTests.SmallOptions();
        options.MaxTextLength = 128;
        options.DropoutRate = 0;
        return new SparkTTS<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 320) { RandomSeed = 2294 }, options);
    }
}
