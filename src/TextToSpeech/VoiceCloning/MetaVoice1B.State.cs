using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.VoiceCloning;

public partial class MetaVoice1B<T>
{
    // The byte-rank inventories are materialized checkpoint data. Runtime BPE
    // encoders contain services and are rebuilt from these owned dictionaries.
    private Dictionary<string, int>? _firstTokenizerRanks;
    private Dictionary<string, int>? _secondTokenizerRanks;
    private string? _decoderRuntimeState;
    private string? _decoderCodecOptions;
    private int[] _decoderLayerIndices = Array.Empty<int>();

    internal void InstallDecoder(EnCodec<T> codec, MetaVoiceMultibandDecoder<T> decoder, EnCodecOptions options)
    {
        string runtime = decoder.SaveRuntimeState();
        string codecOptions = Newtonsoft.Json.JsonConvert.SerializeObject(options);
        foreach (var layer in _decoderLayers) ComponentLayers.Remove(layer);
        _decoderLayers.Clear();
        _decoderLayers.AddRange(codec.Layers.Cast<LayerBase<T>>());
        _decoderLayers.AddRange(decoder.DiffusionLayers);
        ComponentLayers.AddRange(_decoderLayers);
        if (!ReferenceEquals(_codec, codec)) _codec?.Dispose();
        _codec = codec;
        _decoder = decoder;
        _decoderRuntimeState = runtime;
        _decoderCodecOptions = codecOptions;
        UpdateDecoderLayerIndices();
        InvalidateParameterCountCache();
        SetTrainingMode(false);
    }

    internal void InstallEnhancer(MetaVoiceDeepFilterEnhancer<T> enhancer)
    {
        if (_enhancer is not null) ComponentLayers.Remove(_enhancer.Network);
        _enhancer = enhancer;
        ComponentLayers.Add(enhancer.Network);
        enhancer.Network.SetTrainingMode(false);
        UpdateDecoderLayerIndices();
        InvalidateParameterCountCache();
    }

    private void UpdateDecoderLayerIndices()
    {
        _decoderLayerIndices = _decoderLayers.Select(layer => ComponentLayers.IndexOf(layer)).ToArray();
        if (_decoderLayerIndices.Any(index => index < 0))
            throw new InvalidOperationException("A decoder layer is missing from its owning model.");
    }

    /// <inheritdoc />
    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        if (_useNativeMode && Architecture.Layers is not { Count: > 0 })
        {
            if (_secondLayerCount <= 0 || _speakerLayerCount <= 0 || _secondLayerCount + _speakerLayerCount > ComponentLayers.Count)
                throw new InvalidDataException("Missing owned MetaVoice actor or speaker layers.");
            // These helpers are views of the canonical component list, not a
            // second copy of its parameters. Rebind every noncausal head and LSTM.
            _second = new MetaVoiceGpt<T>(Engine, SecondStageConfiguration(), (Architecture.RandomSeed ?? 2294) + 1,
                ComponentLayers.Take(_secondLayerCount).ToArray());
            _speaker = new MetaVoiceSpeakerEncoder<T>(Engine, _options.SpeakerMelChannels, _options.SpeakerHiddenDim,
                _options.SpeakerEmbeddingDim, _options.SpeakerEncoderLayers,
                ComponentLayers.Skip(_secondLayerCount).Take(_speakerLayerCount).ToArray());
        }
        _firstTokenizer = _firstTokenizerRanks is null ? null : MetaVoiceTokenizer.FromByteRanks(_firstTokenizerRanks, 2049);
        _secondTokenizer = _secondTokenizerRanks is null ? null : MetaVoiceTokenizer.FromByteRanks(_secondTokenizerRanks, 1025);
        RestoreDecoderRuntime();
        var enhancer = ComponentLayers.OfType<MetaVoiceDeepFilterNet<T>>().SingleOrDefault();
        _enhancer = enhancer is null ? null : new MetaVoiceDeepFilterEnhancer<T>(enhancer);
    }

    private void RestoreDecoderRuntime()
    {
        if (_decoderRuntimeState is null) return;
        if (_decoderCodecOptions is null || _decoderLayerIndices.Length == 0 ||
            _decoderLayerIndices.Any(index => index < 0 || index >= ComponentLayers.Count) ||
            _decoderLayerIndices.Distinct().Count() != _decoderLayerIndices.Length)
            throw new InvalidDataException("The saved MetaVoice decoder has no valid owned-layer mapping.");
        var options = Newtonsoft.Json.JsonConvert.DeserializeObject<EnCodecOptions>(_decoderCodecOptions)
            ?? throw new InvalidDataException("Missing saved EnCodec architecture.");
        var architecture = new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = Architecture.RandomSeed };
        var codec = new EnCodec<T>(architecture, options);
        try
        {
            var codecLayers = codec.Layers.Cast<LayerBase<T>>().ToArray();
            if (codecLayers.Length >= _decoderLayerIndices.Length)
                throw new InvalidDataException("The saved decoder is missing its diffusion layers.");
            for (int i = 0; i < codecLayers.Length; i++)
            {
                var saved = ComponentLayers[_decoderLayerIndices[i]];
                if (saved.GetType() != codecLayers[i].GetType())
                    throw new InvalidDataException("The saved codec layer differs at index " + i + ".");
                // Carry persistent buffers as well as parameters (RVQ tables and
                // normalization state are not interchangeable with a flat vector).
                using var stream = new MemoryStream();
                using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true)) saved.Serialize(writer);
                stream.Position = 0;
                using var reader = new BinaryReader(stream);
                codecLayers[i].Deserialize(reader);
            }
            var bands = _decoderLayerIndices.Skip(codecLayers.Length).Select(index => ComponentLayers[index]).ToArray();
            var decoder = MetaVoiceMultibandDecoder<T>.RestoreRuntime(Engine, codec, _decoderRuntimeState, bands);
            for (int i = 0; i < codecLayers.Length; i++) ComponentLayers[_decoderLayerIndices[i]] = codecLayers[i];
            _decoderLayers.Clear();
            _decoderLayers.AddRange(codecLayers);
            _decoderLayers.AddRange(bands);
            if (!ReferenceEquals(_codec, codec)) _codec?.Dispose();
            _codec = codec;
            _decoder = decoder;
            InvalidateParameterCountCache();
        }
        catch { codec.Dispose(); throw; }
    }
}
