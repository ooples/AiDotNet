using AiDotNet.Audio.Effects;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class ZonosFixture
{
    internal static void PrepareOptions(ZonosOptions options, string option)
    {
        if (option is "StateDim" or "ConvWidth" or "Expand" or "MambaHeadDim" or "Groups" or "MambaIntermediate" or "ChunkSize")
        {
            options.Backbone = ZonosBackbone.Hybrid;
            options.AttentionLayers = new[] { 0 };
            options.StateDim = 8;
            options.MambaHeadDim = 8;
            options.ChunkSize = 3;
            options.MambaIntermediate = 32;
        }
        if (option is "CtcLoss" or "DnsmosOverall" or "VqScore" or "SpeakerNoised")
        {
            options.Conditioners = ZonosConditioner.Hybrid.ToList();
            options.UnconditionalInputs.Clear();
        }
        if (option == "ConditionerDropout") options.ConditionerDropout = 0.4;
        if (option == "MaxGradNorm") options.MaxGradNorm = 1e-5;
        if (option is "RepetitionPenalty" or "RepetitionWindow" or "UnifiedConfidence" or "UnifiedQuadratic")
        {
            options.Temperature = 1;
            options.MinP = 0;
            options.TopP = 1;
            options.TopK = 0;
            if (option is "UnifiedConfidence" or "UnifiedQuadratic")
            {
                options.UnifiedLinear = 0.5;
                options.UnifiedConfidence = 0.3;
                options.UnifiedQuadratic = 0.05;
            }
        }
    }

    internal static bool TryTrain<T>(Zonos<T> model, Tensor<T> input, string option)
    {
        if (option is not ("ConditionerDropout" or "MaxGradNorm")) return false;
        var options = (ZonosOptions)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var codes = new Tensor<T>(new[] { 12, options.Codebooks });
        for (int i = 0; i < codes.Length; i++) codes[i] = ops.FromDouble(i % options.Codec.CodebookSize);
        model.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<T> { Tokens = input, CodecTokens = codes, LanguageId = 0 });
        return true;
    }

    internal static long? Observe<T>(Zonos<T> model, Tensor<T> input, string option)
    {
        var options = (ZonosOptions)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option is "RepetitionPenalty" or "RepetitionWindow" or "UnifiedConfidence" or "UnifiedQuadratic")
        {
            var logits = new double[options.Codebooks, options.Codec.CodebookSize + 1];
            var history = new int[options.Codebooks, 6];
            int[] repeated = { 1, 1, 2, 3, 0, 1 };
            for (int q = 0; q < options.Codebooks; q++)
            {
                for (int v = 0; v < logits.GetLength(1); v++) logits[q, v] = v < 8 ? 2.5 - 0.4 * v : -12;
                for (int t = 0; t < 6; t++) history[q, t] = repeated[t];
            }
            var values = new System.Collections.Generic.List<double>();
            for (int seed = 0; seed < 32; seed++) values.AddRange(model.SampleFrame(logits, history, 6, new Random(seed)).Select(code => (double)code));
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(values);
        }
        if (option is "CtcLoss" or "DnsmosOverall" or "VqScore" or "SpeakerNoised" or "ChunkSize")
        {
            var ids = input.Select(value => (int)ops.ToDouble(value)).ToArray();
            var prefix = model.Conditioner.Forward(model.ConditionInputs(ids, null));
            var values = option == "ChunkSize" ? model.Network.Forward(prefix, null) : prefix;
            return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(values.Select(value => ops.ToDouble(value)));
        }
        return null;
    }

    internal static ZonosOptions Options() => new()
    {
        HiddenDim = 32, Layers = 2, Heads = 4, KeyValueHeads = 2, AttentionIntermediate = 128,
        RotaryDim = 8, StateDim = 8, MambaHeadDim = 8, ChunkSize = 3,
        MaxNewTokens = 24, SampleRate = 16000,
        Codec = new DACOptions
        {
            SampleRate = 16000, EncoderDim = 4, EncoderRates = new[] { 2, 4 }, DecoderDim = 16,
            DecoderRates = new[] { 4, 2 }, NumQuantizers = 9, CodebookSize = 1024, CodebookDim = 2,
            TargetBandwidthKbps = 180,
        },
    };

    internal static Zonos<T> Create<T>() => new(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
        NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), Options());
}
