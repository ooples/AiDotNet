using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>A bounded native v3 graph: semantic GPT, flow synthesizer, BigVGAN, HuBERT and BERT.</summary>
internal static class GptSoVitsFixture
{
    internal static long Observe<T>(GPTSoVITS<T> model, Tensor<T> input, string option)
    {
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var options = (GPTSoVITSOptions)model.GetOptions();
        model.SetTrainingMode(false);
        if (option == "PromptLanguage")
        {
            var promptRecording = new Tensor<T>(new[] { 9600 });
            for (int i = 0; i < promptRecording.Length; i++) promptRecording[i] = ops.FromDouble(.3 * Math.Sin(i * .13));
            try
            {
                var voice = model.CreateVoice(promptRecording, model.SampleRate, "hello world");
                var prompt = model.GetType().GetMethod("Prompt", flags)!.Invoke(model, new object[] { voice })!;
                var phones = (int[])prompt.GetType().GetField("Phones")!.GetValue(prompt)!;
                return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(phones.Select(phone => (double)phone));
            }
            catch (InvalidOperationException error) when (MissingLanguageAsset(error)) { return -1; }
        }
        if (option is "SamplingSeed" or "Temperature" or "TopK" or "TopP" or "RepetitionPenalty"
            or "PauseSeconds" or "Speed" or "NoiseScale" or "MaxSeconds" or "SpeakerChannels" or "SpeakerBlocks")
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(model.Synthesize("hello world").Select(ops.ToDouble));
        if (option == "TextLanguage")
        {
            try
            {
                return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(
                    model.TextToTokens("hello world").Select(value => ops.ToDouble(value)));
            }
            catch (InvalidOperationException error) when (error.Message.StartsWith(
                "Reading mixed-language text needs fastText's lid.176.bin", StringComparison.Ordinal))
            {
                // English is self-contained; every other release language mode requires its real segmenter asset.
                return -1;
            }
        }
        if (option == "SynthesizerDropout")
        {
            var synthesizer = model.GetType().GetProperty("SoVits", flags)!.GetValue(model)!;
            var quantized = (Tensor<T>)synthesizer.GetType().GetMethod("Dequantize")!.Invoke(synthesizer, new object[] { new[] { 1, 2, 3 } })!;
            var reference = new Tensor<T>(new[] { 1, 1025, 8 });
            for (int i = 0; i < reference.Length; i++) reference[i] = ops.FromDouble(0.3 * Math.Sin(i * 0.13));
            var style = (Tensor<T>)synthesizer.GetType().GetMethod("Style")!.Invoke(synthesizer, new object?[] { reference, null, false, new Random(7) })!;
            var (mean, logs) = ((Tensor<T>, Tensor<T>))synthesizer.GetType().GetMethod("Encode")!.Invoke(synthesizer,
                new object[] { quantized, new[] { 1, 2, 3 }, style, 1.0, true, new Random(7) })!;
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(mean.Concat(logs).Select(value => ops.ToDouble(value)));
        }
        if (option is "VocoderUpsampleRates" or "VocoderUpsampleKernels" or "VocoderInitialChannels" or "VocoderResblockKernels")
        {
            var vocoder = model.GetType().GetField("_bigVgan", flags)!.GetValue(model)
                ?? model.GetType().GetField("_hifiGan", flags)!.GetValue(model)!;
            var mel = new Tensor<T>(new[] { 1, 100, 5 });
            for (int i = 0; i < mel.Length; i++) mel[i] = ops.FromDouble(0.3 * Math.Sin(i * 0.17));
            var audio = (Tensor<T>)vocoder.GetType().GetMethod("Forward")!.Invoke(vocoder, new object?[] { mel, null })!;
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(audio.Select(value => ops.ToDouble(value)));
        }
        if (option == "Cut")
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(GptSoVitsText.Sentences(
                "This first sentence is long enough to exercise the split. This second sentence is also long enough. And a third sentence follows. Finally this is the fourth sentence. A fifth sentence remains.", options.Cut)
                .SelectMany(sentence => new[] { (double)sentence.Length }.Concat(sentence.Select(c => (double)c))));
        if (option == "BertHeads")
        {
            var features = (double[][])model.GetType().GetMethod("BertFeatures", flags)!.Invoke(model, new object[] { "你好世界", new[] { 1, 1, 1, 1 } })!;
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(features.SelectMany(row => row));
        }
        if (option == "HubertHeads")
        {
            var audio = Enumerable.Range(0, 6400).Select(i => 0.3 * Math.Sin(i * 0.13) + 0.1 * Math.Cos(i * 0.037)).ToArray();
            var features = (Tensor<T>)model.GetType().GetMethod("SslFeatures", flags)!.Invoke(model, new object[] { audio })!;
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(features.Select(value => ops.ToDouble(value)));
        }
        if (option is "SampleSteps" or "CfgRate" or "DitDim" or "DitDepth" or "DitHeads" or "DitHeadDim" or "DitConvLayers" or "BridgeChannels")
        {
            var synthesizer = model.GetType().GetProperty("FlowSynthesizer", flags)!.GetValue(model)!;
            Tensor<T> Values(int[] shape)
            {
                var tensor = new Tensor<T>(shape);
                for (int i = 0; i < tensor.Length; i++) tensor[i] = ops.FromDouble(0.3 * Math.Sin(i * 0.17) + 0.2 * Math.Cos(i * 0.037));
                return tensor;
            }
            var mel = (Tensor<T>)synthesizer.GetType().GetMethod("SynthesizeMel")!.Invoke(synthesizer, new object[]
            {
                Values(new[] { 1, options.BridgeChannels, 3 }), Values(new[] { 1, 100, 3 }), Values(new[] { 1, options.BridgeChannels, 5 }),
                options.SampleSteps ?? 2, options.CfgRate ?? 0, (Func<int[], Tensor<T>>)Values,
            })!;
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(mel.Select(value => ops.ToDouble(value)));
        }
        var recording = new Tensor<T>(new[] { 9600 });
        for (int i = 0; i < recording.Length; i++) recording[i] = ops.FromDouble(0.3 * Math.Sin(i * 0.13) + 0.1 * Math.Cos(i * 0.037));
        return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(new[] { ops.ToDouble(model.EvaluateTrainingObjective(
            new AiDotNet.TextToSpeech.TtsTrainingSample<T> { Tokens = input, Audio = recording })) });
    }

    internal static void PrepareOptions(GPTSoVITSOptions options, string option)
    {
        if (option is "Beta1" or "Beta2" or "Epsilon" or "SynthesizerLearningRate" or "UpdatesPerEpoch" or "LearningRateDecay")
        {
            options.CurrentStage = GptSoVitsStage.FlowMatching;
            options.SynthesizerLearningRate = 0.003;
            options.UpdatesPerEpoch = 1;
            options.LearningRateDecay = 0.5;
        }
        if (option is "TextLowLearningRate" or "SegmentSize" or "SynthesizerDropout" or "UpsampleRates" or "UpsampleKernels" or "DiscriminatorWidthDivisor"
            or "MelLossWeight" or "KlLossWeight" or "NoiseScale"
            or "UpsampleInitialChannels" or "ResblockKernels" or "ResblockDilations")
        {
            options.Version = GptSoVitsVersion.V2;
            options.CurrentStage = GptSoVitsStage.SoVits;
            options.SegmentSize = 2560;
            if (option == "SynthesizerDropout") options.SynthesizerDropout = 0.1;
            if (option == "UpsampleKernels") options.UpsampleKernels = new[] { 16, 16, 8, 2, 2 };
        }
        if (option is "SpeakerChannels" or "SpeakerBlocks" or "StyleChannels" or "MrteChannels")
        {
            options.Version = GptSoVitsVersion.V2Pro;
            options.CurrentStage = GptSoVitsStage.SoVits;
            options.SegmentSize = 2560;
        }
        if (option == "CfgRate")
        {
            options.Version = GptSoVitsVersion.V5Dev;
            options.CfgRate = 0.2;
        }
        if (option == "VocoderUpsampleRates") options.VocoderUpsampleRates = new[] { 4, 4, 2, 2, 2, 2 };
        if (option == "VocoderUpsampleKernels") options.VocoderUpsampleKernels = new[] { 8, 8, 4, 4, 4, 4 };
    }

    internal static GPTSoVITSOptions Options() => new()
    {
        Version = GptSoVitsVersion.V3, DiscriminatorWidthDivisor = 16,
        HiddenDim = 16, NumHeads = 2, T2SLayers = 1, SemanticCodebookSize = 32,
        InterChannels = 8, SynthesizerHidden = 16, SynthesizerFilter = 32, SynthesizerHeads = 2, SynthesizerLayers = 2,
        StyleChannels = 16, MrteChannels = 16, StyleHidden = 16, UpsampleInitialChannels = 32,
        ResblockKernels = new[] { 3 }, ResblockDilations = new[] { new[] { 1, 3, 5 } },
        HubertHidden = 16, HubertLayers = 1, HubertHeads = 2, HubertIntermediate = 32,
        HubertConvDims = new[] { 8, 8, 8, 8, 8, 8, 8 }, HubertPositionKernel = 8, HubertPositionGroups = 2,
        BertHidden = 16, BertLayers = 3, BertHeads = 2, BertIntermediate = 32,
        SpeakerChannels = 8, SpeakerBlocks = new[] { 1, 1, 1, 1 },
        TextLanguage = GptSoVitsTextLanguage.English, PromptLanguage = GptSoVitsTextLanguage.English,
        MaxSeconds = 1, T2SAccumulation = 1,
        DitDim = 16, DitDepth = 1, DitHeads = 2, DitHeadDim = 8, DitConvLayers = 1, BridgeChannels = 16,
        LoraRank = 0, SampleSteps = 2, VocoderInitialChannels = 64,
        VocoderResblockKernels = new[] { 3 }, VocoderResblockDilations = new[] { new[] { 1, 3, 5 } },
    };

    internal static void Prepare<T>(GPTSoVITS<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var recording = new Tensor<T>(new[] { 9600 });
        for (int i = 0; i < recording.Length; i++) recording[i] = ops.FromDouble(0.3 * Math.Sin(i * 0.13) + 0.1 * Math.Cos(i * 0.037));
        try { model.Voice = model.CreateVoice(recording, model.SampleRate, "hello world"); }
        catch (InvalidOperationException error) when (MissingLanguageAsset(error)
            && ((GPTSoVITSOptions)model.GetOptions()).PromptLanguage != GptSoVitsTextLanguage.English)
        {
            // The prompt-language observer executes CreateVoice and records this exact asset requirement.
        }
    }

    private static bool MissingLanguageAsset(InvalidOperationException error) => error.Message.StartsWith(
        "Reading mixed-language text needs fastText's lid.176.bin", StringComparison.Ordinal);

    internal static GPTSoVITS<T> Create<T>()
    {
        var model = new GPTSoVITS<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), Options());
        try { Prepare(model); return model; }
        catch { model.Dispose(); throw; }
    }
}
