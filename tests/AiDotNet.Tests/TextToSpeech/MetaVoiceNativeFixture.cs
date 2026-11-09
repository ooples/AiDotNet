using System.Text.Json;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech.VoiceCloning;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Compact native assets for framework invariants; released-weight parity is tested separately.</summary>
internal static class MetaVoiceNativeFixture
{
    private sealed class InitialWeights { internal long Hash; }
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, InitialWeights> Initial = new();
    internal static long InitialFingerprint<T>(MetaVoice1B<T> model) => Initial.GetValue(model,
        _ => throw new InvalidOperationException("The fixture has not recorded native initialization.")).Hash;
    private static string FilePath(string name) => Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);

    internal static MetaVoice1B<T> CreateUnloaded<T>()
    {
        var architecture = new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 4, outputSize: 640);
        return new MetaVoice1B<T>(architecture, new MetaVoice1BOptions
        {
            FirstStageDim = 8, NumFirstStageLayers = 2, NumHeads = 2, FirstStageContext = 128,
            SecondStageDim = 8, NumSecondStageLayers = 2, SecondStageHeads = 2, SecondStageContext = 128,
            SpeakerHiddenDim = 8, SpeakerEmbeddingDim = 3, SpeakerMelChannels = 5, SwiGLUMultipleOf = 4,
            MaximumNewTokens = 8, FirstStageTopP = 1, TrainingDropout = 0, FineTuneLastBlocks = 1,
            MinReferenceDurationSec = .001, RawDecoderOutput = true, SynthesisSeed = 2294
        });
    }

    internal static MetaVoice1B<T> Create<T>()
    {
        var model = CreateUnloaded<T>();
        Prepare(model);
        return model;
    }

    internal static void Prepare<T>(MetaVoice1B<T> model)
    {
        var initial = model.GetParameters();
        var numeric = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        Initial.Remove(model);
        Initial.Add(model, new InitialWeights { Hash = OptionSensitivityHarness.Fingerprint(
            Enumerable.Range(0, initial.Length).Select(i => numeric.ToDouble(initial[i]))) });
        var engine = AiDotNetEngine.Current;
        model.LoadLanguageModelCheckpoints(FilePath("metavoice_invariant_first.pt"), FilePath("metavoice_invariant_second.pt"));
        model.LoadSpeakerEncoderCheckpoint(FilePath("metavoice_public_speaker.pt"));
        var options = EnCodecOptions.OfficialCheckpoint24kHz();
        options.Filters = 2;
        options.NumQuantizers = 8;
        options.IncludeDiscriminators = false;
        var codec = new EnCodec<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 2294 }, options);
        codec.LoadPretrainedWeights(FilePath("metavoice_invariant_codec.safetensors"));
        var decoder = new MetaVoiceMultibandDecoder<T>(engine, codec);
        using var processorReference = JsonDocument.Parse(File.ReadAllText(FilePath("metavoice_band_processor.json")));
        var operations = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int band = 0; band < 4; band++)
        {
            // Preserve four resolutions, both residual blocks, original codec width,
            // all four bands and the complete released diffusion schedule.
            var network = new MetaVoiceDiffusionUnet<T>(engine, hidden: 2, depth: 4, growth: 1,
                codecDimension: 128, groups: 1, maximumChannels: 2);
            int layerIndex = 0;
            foreach (var layer in network.Layers)
            {
                var values = layer.GetParameters();
                for (int i = 0; i < values.Length; i++)
                    values[i] = operations.FromDouble(.03 * Math.Sin(i * .13 + layerIndex * .37 + band));
                layer.SetParameters(values);
                layerIndex++;
            }
            Dictionary<string, double[]>? fitted = null;
            if (band >= 2)
            {
                var source = processorReference.RootElement.GetProperty("cases").EnumerateArray().Single(value =>
                    value.GetProperty("band").GetInt32() == band && value.GetProperty("trained").GetBoolean()).GetProperty("state");
                fitted = source.EnumerateObject().ToDictionary(pair => pair.Name,
                    pair => pair.Value.GetProperty("values").EnumerateArray().Select(value => value.GetDouble()).ToArray());
            }
            decoder.InstallBand(band, network, fitted);
        }
        model.InstallDecoder(codec, decoder, options);
    }
}

internal static class MetaVoiceOptionFixture
{
    internal static long ParameterCount<T>(MetaVoice1B<T> model)
    {
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var speaker = (MetaVoiceSpeakerEncoder<T>)typeof(MetaVoice1B<T>).GetField("_speaker", flags)!.GetValue(model)!;
        var options = (MetaVoice1BOptions)model.GetOptions();
        // Dense speaker layers allocate their input weights on the first forward pass.
        speaker.ForwardMels(new Tensor<T>(new[] { 1, 1, options.SpeakerMelChannels }));
        return model.GetParameters().Length;
    }

    internal static bool Handles(string option) => option is "Seed" or "SecondStageHeads" or "SpeakerGuidance"
        or "PromptGuidance" or "RawDecoderOutput" or "EnhanceOutput";

    internal static void Prepare(MetaVoice1BOptions options, string option)
    {
        if (option == "RawDecoderOutput") options.EnhanceOutput = false;
        if (option == "EnhanceOutput")
        {
            options.RawDecoderOutput = false;
            options.EnhanceOutput = false;
            options.MaximumNewTokens = 100;
        }
    }

    internal static long Observe<T>(MetaVoice1B<T> model, Tensor<T> input, string option)
    {
        if (option == "Seed") return MetaVoiceNativeFixture.InitialFingerprint(model);
        var numeric = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option is "RawDecoderOutput" or "EnhanceOutput")
        {
            try { return OptionSensitivityHarness.Fingerprint(model.Predict(input).Select(value => numeric.ToDouble(value))); }
            catch (InvalidOperationException error) when (error.Message.StartsWith("Load the original DeepFilterNet3 checkpoint", StringComparison.Ordinal))
            { return OptionSensitivityHarness.Fingerprint(Array.Empty<double>()); }
        }
        var options = (MetaVoice1BOptions)model.GetOptions();
        var audio = new Tensor<T>(new[] { 1280 });
        for (int i = 0; i < audio.Length; i++) audio[i] = numeric.FromDouble(0.2 * Math.Sin(i * 0.13));
        var speaker = AiDotNetEngine.Current.Reshape(model.ExtractSpeakerEmbedding(audio), new[] { 1, options.SpeakerEmbeddingDim });
        model.SetTrainingMode(false);
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var modelType = typeof(MetaVoice1B<T>);
        if (option == "SecondStageHeads")
        {
            var decoder = (MetaVoiceGpt<T>)modelType.GetField("_second", flags)!.GetValue(model)!;
            var tokenizer = (MetaVoiceTokenizer)modelType.GetField("_secondTokenizer", flags)!.GetValue(model)!;
            var tokens = MetaVoiceGeneration.SecondStageInput(tokenizer.Encode("test"),
                new[] { new[] { 1, 2, 3, 4 }, new[] { 2, 3, 4, 5 } }, Math.Min(16, options.SecondStageContext));
            return OptionSensitivityHarness.Combine(decoder.Forward(tokens, speaker).Logits.Select(logits =>
                OptionSensitivityHarness.Fingerprint(logits.Select(value => numeric.ToDouble(value)))).ToArray());
        }
        var first = (MetaVoiceGpt<T>)modelType.GetField("_first", flags)!.GetValue(model)!;
        var textTokenizer = (MetaVoiceTokenizer)modelType.GetField("_firstTokenizer", flags)!.GetValue(model)!;
        long observation = 0;
        MetaVoiceGeneration.FirstStage(textTokenizer.Encode("test"), 1, options.FirstStageContext,
            (prefix, conditioned) => first.Forward(new[] { prefix }, conditioned ? speaker : null, lastOnly: true)
                .Logits[0].Select(value => numeric.ToDouble(value)).ToArray(), _ => 0.5, true,
            options.FirstStageTemperature, options.FirstStageTopK, options.FirstStageTopP,
            (options.SpeakerGuidance, options.PromptGuidance),
            observeProbabilities: (_, probabilities) => observation = OptionSensitivityHarness.Fingerprint(probabilities));
        return observation;
    }
}
