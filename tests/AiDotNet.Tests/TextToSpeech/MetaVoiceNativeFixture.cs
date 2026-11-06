using System.Text.Json;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.VoiceCloning;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Compact native assets for framework invariants; released-weight parity is tested separately.</summary>
internal static class MetaVoiceNativeFixture
{
    private static string FilePath(string name) => Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);

    internal static MetaVoice1B<T> CreateUnloaded<T>()
    {
        var architecture = new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 4, outputSize: 640) { RandomSeed = 2294 };
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
