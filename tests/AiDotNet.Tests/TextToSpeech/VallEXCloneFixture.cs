using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.VoiceCloning;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>The paper tests' bounded native bilingual AR/NAR and 75 Hz EnCodec cloning facade.</summary>
internal static class VallEXCloneFixture
{
    private sealed class FirstCodeRandom : Random
    {
        public override double NextDouble() => 0;
    }

    internal static void PrepareOptions(VALLEXOptions options, string option)
    {
        if (option == "MaxTextLength") options.MaxTextLength = 4;
        if (option == "SynthesisLanguage")
        {
            options.UseCommunityPhoneTokenizer = true;
            options.TextTokens = 512;
            options.LanguageCount = 3;
            options.LanguagePlacement = VallELanguagePlacement.TextTokens;
            options.SynthesisLanguage = VallEXLanguage.English;
        }
        if (option == "UseCommunityPhoneTokenizer") options.TextTokens = 512;
    }

    internal static long? Observe<T>(VALLEX<T> model, string option, Tensor<T>? input = null)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option == "Seed" && input is not null)
        {
            try
            {
                return TtsPaperNumerics.Fingerprint(model.Predict(input).ToArray().Select(value => ops.ToDouble(value)));
            }
            catch (InvalidOperationException error) when (error.Message.StartsWith(
                "VALL-E's AR model ended the utterance before its first code", StringComparison.Ordinal))
            {
                // A different seed can legitimately draw immediate EOS from an untrained AR model.
                // Keep that native sampling outcome observable rather than treating it as a harness crash.
                return -1;
            }
        }
        if (option == "StopOnMostLikelyEndToken")
        {
            model.SetTrainingMode(false);
            const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            var core = typeof(VallEModelBase<T>).GetProperty("PaperCore", flags)!.GetValue(model)!;
            var head = core.GetType().GetProperty("ArPredict")!.GetValue(core)!;
            var weights = (Tensor<T>)head.GetType().GetProperty("Weights", flags)!.GetValue(head)!;
            var saved = weights.ToArray();
            var text = model.TextToTokens("hello").Select(value => (int)ops.ToDouble(value)).ToArray();
            var boundaryPrompt = new[] { 1, 2, 3 };
            var logitsMethod = core.GetType().GetMethod("ArLogits")!;
            var logits = (Tensor<T>)logitsMethod.Invoke(core, new object?[] { text, boundaryPrompt, false, new Random(7), null, new int[boundaryPrompt.Length] })!;
            int column = Enumerable.Range(0, weights.Shape[1]).OrderByDescending(v => Math.Abs(ops.ToDouble(logits[boundaryPrompt.Length - 1, v]))).First();
            double scale = 0.1 / ops.ToDouble(logits[boundaryPrompt.Length - 1, column]);
            try
            {
                for (int i = 0; i < weights.Length; i++) weights[i] = ops.Zero;
                for (int d = 0; d < weights.Shape[0]; d++)
                    weights[d, weights.Shape[1] - 1] = ops.FromDouble(ops.ToDouble(saved[d * weights.Shape[1] + column]) * scale);
                var boundary = (Tensor<T>)logitsMethod.Invoke(core, new object?[] { text, boundaryPrompt, false, new Random(7), null, new int[boundaryPrompt.Length] })!;
                if (ops.ToDouble(boundary[boundaryPrompt.Length - 1, weights.Shape[1] - 1]) <= 0)
                    throw new InvalidOperationException("The native EOS boundary must rank EOS above every zero-logit code.");
                try
                {
                    var codes = (System.Collections.Generic.List<int>)core.GetType().GetMethod("ArGenerate")!.Invoke(core,
                        new object?[] { text, boundaryPrompt, 1.0, 0, 2, new FirstCodeRandom(), null, 0, 0 })!;
                    return codes.Count;
                }
                catch (System.Reflection.TargetInvocationException error) when (error.InnerException is InvalidOperationException ended
                    && ended.Message.StartsWith("VALL-E's AR model ended the utterance before its first code", StringComparison.Ordinal))
                { return -1; }
            }
            finally { for (int i = 0; i < weights.Length; i++) weights[i] = saved[i]; }
        }
        if (option is "MaxTextLength" or "UseCommunityPhoneTokenizer")
            return TtsPaperNumerics.Fingerprint(
                model.TextToTokens("hello world, this is a longer sentence").Select(value => ops.ToDouble(value)));
        if (option == "MinReferenceDurationSec" && model is VALLEXClone<T> clone)
        {
            var audio = new Tensor<T>(new[] { 320 });
            for (int i = 0; i < audio.Length; i++) audio[i] = ops.FromDouble(0.3 * Math.Sin(i * 0.17));
            try { return clone.ExtractSpeakerEmbedding(audio).Length; }
            catch (ArgumentException) { return -1; }
        }
        if (option != "SynthesisLanguage") return null;
        model.SetTrainingMode(false);
        var target = model.TextToTokens("hello world").Select(value => (int)ops.ToDouble(value)).ToArray();
        var enrolled = model.TextToTokens("a prompt").Select(value => (int)ops.ToDouble(value)).ToArray();
        var options = (VALLEXOptions)model.GetOptions();
        var prompt = new int[3, options.NumCodebooks];
        for (int f = 0; f < 3; f++) for (int q = 0; q < options.NumCodebooks; q++) prompt[f, q] = (f * 5 + q * 11) % options.CodebookSize;
        var generate = typeof(VALLEX<T>).GetMethod("Generate", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        var values = new System.Collections.Generic.List<double>();
        for (int seed = 0; seed < 8; seed++)
        {
            var codes = (int[,])generate.Invoke(model, new object[] { target, enrolled, prompt, new TtsVoice<T> { LanguageId = 0 }, new Random(seed) })!;
            values.Add(codes.GetLength(0));
            values.AddRange(codes.Cast<int>().Select(code => (double)code));
        }
        return TtsPaperNumerics.Fingerprint(values);
    }

    internal static VALLEXCloneOptions Options(string referenceText = "a prompt") => new()
    {
        HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, NumDecoderLayers = 1, FeedForwardDim = 32,
        TextTokens = 192, NumCodebooks = 3, CodebookSize = 64, MaxCodesPerTextToken = 2,
        LearningRate = 3e-3, WarmupSteps = 0, DropoutRate = 0, MinReferenceDurationSec = 0.01,
        ReferenceText = referenceText,
        Codec = new EnCodecOptions
        {
            SampleRate = 24000, NumQuantizers = 3, CodebookSize = 64, Filters = 2,
            Ratios = [8, 5, 4, 2], Dimension = 8, ResidualKernelSizes = [3, 1],
            TargetBandwidthKbps = 1.35, TargetBandwidths = [1.35], SegmentSize = 640,
            IncludeDiscriminators = false, KMeansIterations = 0, DeadCodeThreshold = 0, LstmLayers = 1,
        }
    };

    internal static VALLEXClone<T> Create<T>() =>
        new(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, Options());
}
