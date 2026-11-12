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
        if (option == "MaxTextLength") options.MaxTextLength = 4;
        if (option is "TopK" or "TopP" or "Temperature" or "GenerationSeed" or "MaxCodecFrames"
            or "ConstrainedSpeechDecoding" or "MinimumGeneratedCodecFrames")
            options.MaxCodecFrames = 8;
        if (option == "MinimumGeneratedCodecFrames") { options.TopK = 0; options.TopP = 1; }
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
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        if (option == "WaveDecoderRates")
        {
            var audio = Recording<T>();
            var waveform = model.DecodeFromTokens(model.EncodeToTokens(audio), model.EncodeGlobalTokens(audio));
            var waveOps = MathHelper.GetNumericOperations<T>();
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(waveform.Select(waveOps.ToDouble));
        }
        if (option == "MaxTextLength")
        {
            var accepted = new List<double>();
            for (int length = 1; length <= 8; length++)
            {
                try { accepted.Add(model.TextToTokens(new string('a', length)).Length); }
                catch (ArgumentException error) when (error.Message.Contains("exceeds MaxTextLength")) { accepted.Add(-1); }
            }
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(accepted);
        }
        if (option is "TopK" or "TopP" or "Temperature" or "GenerationSeed" or "MaxCodecFrames"
            or "ConstrainedSpeechDecoding" or "MinimumGeneratedCodecFrames")
        {
            var languageModel = typeof(SparkTTS<T>).GetField("_languageModel", flags)!.GetValue(model)!;
            var generate = languageModel.GetType().GetMethod(option is "ConstrainedSpeechDecoding" or "MinimumGeneratedCodecFrames"
                ? "GenerateSemantic" : "GenerateAll")!;
            var values = new List<double>();
            for (int seed = 0; seed < 16; seed++)
            {
                try
                {
                    int samplingSeed = unchecked(((SparkTTSOptions)model.GetOptions()).GenerationSeed + seed);
                    var ids = (int[])generate.Invoke(languageModel, new object[] { new[] { 3, 4, 5 }, new Random(samplingSeed) })!;
                    values.Add(ids.Length);
                    values.AddRange(ids.Select(id => (double)id));
                }
                catch (System.Reflection.TargetInvocationException error) when
                    (error.InnerException is InvalidOperationException native && native.Message.Contains("no semantic speech tokens"))
                {
                    // The released unconstrained path can emit no speech at random initialization.
                    values.Add(-1);
                }
            }
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(values);
        }
        bool speakerFrontend = option is "ReferenceSegmentSeconds" or "MelFftSize" or "MelWindowSize" or "MelMinimumFrequency";
        if (option != "Wav2VecHeads" && !speakerFrontend) return null;
        var codec = typeof(SparkTTS<T>).GetField("_biCodec", flags)!.GetValue(model)!;
        Tensor<T> features;
        if (speakerFrontend)
        {
            var speaker = codec.GetType().GetProperty("Speaker")!.GetValue(codec)!;
            var coordinates = speaker.GetType().GetMethod("Coordinates", flags)!.Invoke(speaker, new object[] { Recording<T>() })!;
            features = (Tensor<T>)coordinates.GetType().GetField("Item2")!.GetValue(coordinates)!;
        }
        else features = (Tensor<T>)codec.GetType().GetMethod("MixedWav2VecFeatures", flags)!.Invoke(codec, new object[] { Recording<T>() })!;
        var ops = MathHelper.GetNumericOperations<T>();
        return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(features.Select(ops.ToDouble));
    }

    internal static SparkTTS<T> Create<T>()
    {
        var options = SparkTTSPaperTests.SmallOptions();
        options.MaxTextLength = 128;
        return new SparkTTS<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 320) { RandomSeed = 2294 }, options);
    }
}
