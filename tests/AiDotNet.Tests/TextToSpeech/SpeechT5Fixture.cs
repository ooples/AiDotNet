using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.Tests.TextToSpeech;

internal static class SpeechT5Fixture
{
    internal static SpeechT5Options Options() => new()
    {
        HiddenDim = 16, FeedForwardDim = 32, NumHeads = 2, NumEncoderLayers = 2, NumDecoderLayers = 2,
        MaxRelativePosition = 8, MaxTextLength = 64, MaxMelLength = 400, MelChannels = 16,
        SpeakerDim = 8, PrenetUnits = 8, PostnetChannels = 8,
        FeatureLayers = new[] { (8, 10, 5), (8, 3, 2) }, ConvPositionKernel = 8,
        ConvPositionGroups = 2, FinalDim = 8, HubertClasses = new[] { 16 }, LatentVars = 4,
        SpeakerEncoderChannels = new[] { 8, 8, 8, 8, 16 }, VocoderInitialChannels = 64,
        VocoderResBlockKernels = new[] { 3 }, VocoderResBlockDilations = new[] { new[] { 1 } },
        WarmupUpdates = 1, FreezeEncoderUpdates = 0, MaxLengthRatio = 4,
        LearningRate = .001, BertInitialization = false,
    };

    internal static SpeechT5<T> Create<T>(SpeechT5Options options) => new(
        new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Generative,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    internal static bool IsPretraining(string option) => option is
        "PretrainEpsilon" or "MaskId" or "PretrainUpdates" or "FeaturePenaltyWeight"
        or "PretrainWeightDecay" or "BosId" or "EosId" or "PretrainLearningRate"
        or "DiversityWeight" or "TextRandomRatio" or "PretrainMaxGradientNorm"
        or "CodebookProbability" or "MaskProbability" or "PretrainWarmupFraction"
        or "TextMaskRatio" or "MaxSpeechSamples" or "PoissonLambda" or "LabelRate" or "MaskLength";

    internal static bool IsText(string option) => option is
        "MaskId" or "BosId" or "EosId" or "TextRandomRatio" or "TextMaskRatio" or "PoissonLambda";

    internal static void PrepareOptions(SpeechT5Options options, string option)
    {
        if (option is "StopCriterion" or "MaxLengthRatio") options.StopThreshold = .75;
        if (option == "MaxMelLength") options.MaxMelLength = 8;
        if (!IsPretraining(option)) return;
        options.CurrentStage = SpeechT5Stage.Pretraining;
        // Exercise warmup and decay within the same bounded native training run.
        options.PretrainUpdates = 20;
        options.PretrainWarmupFraction = .1;
        // The waveform exceeds this limit so the real aligned crop is exercised.
        options.MaxSpeechSamples = 2400;
        if (option == "PretrainMaxGradientNorm") options.PretrainMaxGradientNorm = .001;
    }

    internal static long? Observe<T>(SpeechT5<T> model, string option)
    {
        var o = (SpeechT5Options)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option is "StopCriterion" or "MaxLengthRatio")
        {
            // A real stop head at p=.5 separates any-frame and summed-pair rules at .75.
            var stop = model.Core.SpeechPostnet.ProbabilityOut;
            stop.SetParameters(new Vector<T>(checked((int)stop.ParameterCount)));
            var mel = model.GenerateSpectrogram(new[] { 3, 4, 2 }, new TtsVoice<T>
            {
                SpeakerEmbedding = new Tensor<T>(new[] { o.SpeakerDim }),
            });
            return AiDotNet.Tests.ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(mel.Select(ops.ToDouble));
        }
        if (option is "MaxTextLength" or "MaxMelLength")
        {
            var sample = Sample<T>(string.Empty);
            if (option == "MaxTextLength")
            {
                var tokens = new Tensor<T>(new[] { 65 });
                for (int i = 0; i < 65; i++) tokens[i] = ops.FromDouble(3);
                sample = new TtsTrainingSample<T> { Tokens = tokens, Audio = sample.Audio };
            }
            else
            {
                sample = new TtsTrainingSample<T>
                {
                    Tokens = sample.Tokens, SpeakerReference = sample.Audio,
                    Mel = new Tensor<T>(new[] { 18, o.MelChannels }),
                };
            }
            try
            {
                var loss = model.EvaluateTrainingObjective(sample);
                return AiDotNet.Tests.ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(new[] { ops.ToDouble(loss) });
            }
            catch (ArgumentException ex) when (
                option == "MaxTextLength" && ex.Message.StartsWith("There are more than", StringComparison.Ordinal)
                || option == "MaxMelLength" && ex.Message.StartsWith("The recording has more decoder steps", StringComparison.Ordinal))
            {
                return long.MinValue;
            }
        }
        return null;
    }

    internal static TtsTrainingSample<T> Sample<T>(string option)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var tokens = new Tensor<T>(new[] { 18 });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = ops.FromDouble(3 + i % 20);
        if (IsText(option)) return new TtsTrainingSample<T> { Tokens = tokens };
        var audio = new Tensor<T>(new[] { 3200 });
        for (int i = 0; i < audio.Length; i++)
            audio[i] = ops.FromDouble(.3 * Math.Sin(2 * Math.PI * 220 * i / 16000.0)
                + .1 * Math.Sin(2 * Math.PI * 1330 * i / 16000.0 + .4));
        var labels = new Tensor<T>(new[] { 10 });
        for (int i = 0; i < labels.Length; i++) labels[i] = ops.FromDouble((7 * i + 3) % 16);
        return new TtsTrainingSample<T> { Tokens = tokens, Audio = audio, CodecTokens = labels };
    }
}
