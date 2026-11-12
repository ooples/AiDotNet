using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class VoiceCraftOptionFixture
{
    internal static void PrepareOptions(VoiceCraftOptions options, string option)
    {
        if (option is "EdenWarmupSteps" or "EdenStep" or "EdenEpoch" or "PseudoEpochSteps")
        {
            options.UseScaledAdam = true;
            options.EdenWarmupSteps = 2;
            options.EdenStep = 4;
            options.EdenEpoch = 1;
            options.PseudoEpochSteps = 2;
        }
        if (option == "MinimumGapFrames")
        {
            options.SamplingSeed = 0;
            options.MinimumGapFrames = 8;
            options.MaxMaskSpans = 3;
            options.MaximumMaskFrames = 4;
        }
        if (option == "MaxTextLength") options.MaxTextLength = 2;
        if (option is "MinimumGeneratedFrames" or "StopSilenceRepetition" or "SilenceTokens" or "SampleBatchSize" or "MaxCodecFrames")
        {
            options.TopK = 1;
            options.TopP = 1;
            options.MaximumFramesPerPhone = 8;
            if (option != "MaxCodecFrames") options.MaxCodecFrames = 32;
            if (option is "StopSilenceRepetition" or "SilenceTokens")
            {
                options.SilenceTokens = new[] { 1 };
                options.StopSilenceRepetition = 3;
            }
            else options.SilenceTokens = Array.Empty<int>();
        }
        if (option == "SampleBatchSize")
        {
            options.TopK = 0;
            options.SampleBatchSize = 3;
            options.MinimumGeneratedFrames = 0;
            options.MaxCodecFrames = 8;
        }
    }

    internal static void Train<T>(VoiceCraft<T> model)
    {
        var options = (VoiceCraftOptions)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var codes = new Tensor<T>(new[] { 32, options.NumCodebooks });
        for (int f = 0; f < codes.Shape[0]; f++)
            for (int q = 0; q < codes.Shape[1]; q++) codes[f, q] = ops.FromDouble((3 * f + 5 * q) % options.CodebookSize);
        model.Train(new TtsTrainingSample<T> { Text = "hello", Tokens = model.TokenizeText("hello"), CodecTokens = codes });
    }

    internal static long? Observe<T>(VoiceCraft<T> model, string option)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option == "MaxTextLength")
            return TtsPaperNumerics.Fingerprint(model.TokenizeText("hello").Select(ops.ToDouble));
        if (option is not ("MinimumGeneratedFrames" or "StopSilenceRepetition" or "SilenceTokens" or "SampleBatchSize" or "MaxCodecFrames")) return null;
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var options = (VoiceCraftOptions)typeof(VoiceCraft<T>).GetField("_options", flags)!.GetValue(model)!;
        var network = typeof(VoiceCraft<T>).GetField("_network", flags)!.GetValue(model)!;
        // Materialize input-dependent dense weights before saving and calibrating the readout.
        model.Synthesize("hello");
        var heads = ((DenseLayer<T> Hidden, DenseLayer<T> Output)[])network.GetType().GetField("_predictionHeads", flags)!.GetValue(network)!;
        var parameters = heads.SelectMany(head => new[]
        {
            (Tensor<T>)typeof(DenseLayer<T>).GetField("_weights", flags)!.GetValue(head.Output)!,
            (Tensor<T>)typeof(DenseLayer<T>).GetField("_biases", flags)!.GetValue(head.Output)!,
        }).ToArray();
        var saved = parameters.Select(tensor => tensor.ToArray()).ToArray();
        int originalSeed = options.SamplingSeed;
        try
        {
            for (int q = 0; q < heads.Length; q++)
            {
                var tensors = parameters.Skip(2 * q).Take(2);
                foreach (var tensor in tensors)
                {
                    for (int i = 0; i < tensor.Length; i++) tensor[i] = ops.Zero;
                    if (tensor.Rank != 1) continue;
                    for (int i = 0; i < tensor.Length; i++) tensor[i] = ops.FromDouble(-10000);
                    tensor[1] = ops.Zero;
                    if (q == 0 && option is "StopSilenceRepetition" or "SilenceTokens")
                    {
                        tensor[1] = ops.One;
                        tensor[options.EndOfUtteranceToken] = ops.FromDouble(.75);
                    }
                    if (q == 0 && option == "MinimumGeneratedFrames") tensor[options.EndOfUtteranceToken] = ops.FromDouble(100);
                    if (q == 0 && option == "SampleBatchSize") tensor[options.EndOfUtteranceToken] = ops.FromDouble(-.1);
                }
            }
            var lengths = new List<double>();
            for (int seed = 0; seed < (option == "SampleBatchSize" ? 16 : 1); seed++)
            {
                options.SamplingSeed = originalSeed + seed;
                lengths.Add(model.Synthesize("hello").Length);
            }
            return TtsPaperNumerics.Fingerprint(lengths);
        }
        finally
        {
            options.SamplingSeed = originalSeed;
            for (int p = 0; p < parameters.Length; p++)
                for (int i = 0; i < parameters[p].Length; i++) parameters[p][i] = saved[p][i];
        }
    }
}
