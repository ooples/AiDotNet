using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech;

namespace AiDotNet.Tests.TextToSpeech;

internal static class VevoOptionFixture
{
    internal static bool IsSampling(string option) => option is "MaxLength" or "RepetitionPenalty";

    internal static void PrepareOptions(VevoOptions options, string option = "MaxLength")
    {
        if (option is "TrainingSeed" or "FlowLearningRate" or "FlowPromptDropProbability" or "FlowSigma")
            options.CurrentStage = VevoStage.Acoustic;
        if (option == "VoiceArHeads")
        {
            options.CurrentStage = VevoStage.ContentToContentStyle;
            options.VoiceConversion = true;
        }
        if (!IsSampling(option)) return;
        // Four phones, two input delimiters and the output start token consume seven positions.
        options.MaxLength = option == "RepetitionPenalty" ? 39 : 11;
        options.MinNewTokens = option == "RepetitionPenalty" ? 24 : 4;
        options.ConstrainToContentStyleTokens = true;
    }

    internal static void Reconcile(VevoOptions options, string option)
    {
        long stride = options.HubertConvStrides.Aggregate(1L, (a, b) => a * b);
        if (option is "SampleRate" or "ContentSampleRate" && options.ContentSampleRate > 0)
        {
            long numerator = stride * options.SampleRate;
            if (numerator > 0 && numerator % options.ContentSampleRate == 0 && numerator / options.ContentSampleRate <= int.MaxValue)
                options.HopSize = (int)(numerator / options.ContentSampleRate);
        }
        else if (option == "HopSize" && stride > 0)
        {
            long numerator = (long)options.HopSize * options.ContentSampleRate;
            if (numerator > 0 && numerator % stride == 0 && numerator / stride <= int.MaxValue)
                options.SampleRate = (int)(numerator / stride);
        }
    }

    internal static Tensor<T> Recording<T>()
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return new Tensor<T>(new[] { 1024 }, new Vector<T>(Enumerable.Range(0, 1024)
            .Select(i => ops.FromDouble(.4 * Math.Sin(2 * Math.PI * (120 + i * .1) * i / 24000))).ToArray()));
    }

    private static Tensor<T> Phones<T>()
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return new Tensor<T>(new[] { 4 }, new Vector<T>(new[] { 3, 7, 11, 4 }.Select(v => ops.FromDouble(v)).ToArray()));
    }

    internal static void Train<T>(Vevo<T> model) => model.Train(new TtsTrainingSample<T>
    {
        Tokens = Phones<T>(), Audio = Recording<T>(),
    });

    internal static long Observe<T>(Vevo<T> model, string option = "MaxLength")
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var options = (VevoOptions)model.GetOptions();
        if (option is "HubertHeads" or "NumEncoderLayers")
            return OptionSensitivityHarness.Fingerprint(model.HubertFeatures(model.AtContentRate(Recording<T>())).ToArray().Select(ops.ToDouble));
        if (option is "SampleRate" or "ContentSampleRate" or "HopSize")
            return OptionSensitivityHarness.Combine(
                OptionSensitivityHarness.Fingerprint(model.AtContentRate(Recording<T>()).ToArray().Select(ops.ToDouble)),
                OptionSensitivityHarness.Fingerprint(model.Mel(Recording<T>()).ToArray().Select(ops.ToDouble)));
        if (option == "VoiceArHeads")
        {
            var wave = model.AtContentRate(Recording<T>());
            var content = model.ContentCodes(wave).Select(code => code + options.PhoneVocabulary).ToArray();
            return OptionSensitivityHarness.Fingerprint(model.ContentToContentStyle.Logits(content, model.ContentStyleCodes(wave)).ToArray().Select(ops.ToDouble));
        }
        if (!IsSampling(option))
            return OptionSensitivityHarness.Fingerprint(model.Predict(Phones<T>()).ToArray().Select(ops.ToDouble));
        var decoding = new VevoDecoding(options.Sample, options.Temperature, options.TopK, options.TopP,
            options.RepetitionPenalty, options.MinNewTokens, options.MaxLength, options.ConstrainToContentStyleTokens);
        var prompt = option == "RepetitionPenalty" ? model.ContentStyleCodes(model.AtContentRate(Recording<T>())).Take(8).ToArray() : Array.Empty<int>();
        var codes = model.TextToContentStyle.Generate(new[] { 3, 7, 11, 4 }, prompt, decoding,
            new Random(options.SamplingSeed));
        return TtsPaperNumerics.Fingerprint(codes.Select(value => (double)value));
    }
}
