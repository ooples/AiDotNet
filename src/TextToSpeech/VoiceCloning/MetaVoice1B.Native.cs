using AiDotNet.Audio.Generation;
using AiDotNet.Audio.Resampling;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

public partial class MetaVoice1B<T>
{
    private MetaVoiceGpt<T>? _first;
    [AiDotNet.Attributes.Scratch] private MetaVoiceGpt<T>? _second;
    [AiDotNet.Attributes.Scratch] private MetaVoiceSpeakerEncoder<T>? _speaker;
    private int _secondLayerCount, _speakerLayerCount;
    private MetaVoiceTokenizer? _firstTokenizer, _secondTokenizer;
    [AiDotNet.Attributes.Scratch] private EnCodec<T>? _codec;
    [AiDotNet.Attributes.Scratch] private MetaVoiceMultibandDecoder<T>? _decoder;
    [AiDotNet.Attributes.Scratch] private MetaVoiceDeepFilterEnhancer<T>? _enhancer;
    [AiDotNet.Attributes.Scratch] private readonly List<LayerBase<T>> _decoderLayers = new();
    private bool _firstLoaded, _secondLoaded, _speakerLoaded;
    internal Func<double>? SamplingUniformForAudit { get; set; }
    internal Func<int[,], Tensor<T>>? DecoderForAudit { get; set; }
    internal Func<int[], Tensor<T>>? DiffusionGaussianForAudit { get; set; }
    internal Action<string, Tensor<T>>? ObserveOutputForAudit { get; set; }

    private int InitializationSeed() => (Architecture.HasExplicitRandomSeed
        ? Architecture.RandomSeed : _options.Seed ?? Architecture.RandomSeed) ?? 2294;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is { Count: > 0 }) { Layers.AddRange(Architecture.Layers); return; }
        if (_options.FirstStageDim <= 0 || _options.NumHeads <= 0 || _options.FirstStageDim % _options.NumHeads != 0 ||
            _options.SecondStageDim <= 0 || _options.SecondStageHeads <= 0 || _options.SecondStageDim % _options.SecondStageHeads != 0 ||
            _options.NumFirstStageLayers <= 0 || _options.NumSecondStageLayers <= 0 || _options.FirstStageContext <= 1 ||
            _options.SecondStageContext <= 0 || _options.NumCodebooks != 8 || _options.FirstStageCodebooks != 2 ||
            _options.CodebookSize != 1024 || _options.CodecLatentDim != 128 || _options.SampleRate != 24000 || _options.HopSize != 320)
            throw new ArgumentException("MetaVoice dimensions must describe its causal/noncausal GPT and released 24 kHz eight-codebook codec.");
        int seed = InitializationSeed();
        _first = new MetaVoiceGpt<T>(Engine, new MetaVoiceGptConfiguration
        {
            Dimension = _options.FirstStageDim, Layers = _options.NumFirstStageLayers, Heads = _options.NumHeads,
            Context = _options.FirstStageContext, SpeakerDimension = _options.SpeakerEmbeddingDim,
            SwiGluMultiple = _options.SwiGLUMultipleOf, Dropout = _options.TrainingDropout
        }, seed);
        _second = new MetaVoiceGpt<T>(Engine, SecondStageConfiguration(), seed + 1);
        _speaker = new MetaVoiceSpeakerEncoder<T>(Engine, _options.SpeakerMelChannels, _options.SpeakerHiddenDim,
            _options.SpeakerEmbeddingDim, _options.SpeakerEncoderLayers);
        _secondLayerCount = _second.Layers.Count;
        _speakerLayerCount = _speaker.Layers.Count;
        Layers.AddRange(_first.Layers); ComponentLayers.AddRange(_second.Layers); ComponentLayers.AddRange(_speaker.Layers);
    }

    private MetaVoiceGptConfiguration SecondStageConfiguration() => new()
    {
        Dimension = _options.SecondStageDim, Layers = _options.NumSecondStageLayers, Heads = _options.SecondStageHeads,
        Context = _options.SecondStageContext, SpeakerDimension = _options.SpeakerEmbeddingDim,
        Causal = false, RmsNorm = false, SwiGlu = false, SpeakerDropout = 0,
        InputVocabularies = new[] { 1538, 1025 }, OutputVocabularies = Enumerable.Repeat(1025, 6).ToArray()
    };

    /// <summary>Encodes normalized text using the actual first-stage checkpoint tokenizer.</summary>
    public int[] EncodeText(string text) => (_firstTokenizer ?? throw new InvalidOperationException("Load the first-stage tokenizer checkpoint."))
        .Encode(MetaVoiceText.Normalize(text));

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => HasPaperPipeline
        ? LayerInputDomain.Indices(2049, 2562) : base.GetInputDomain(inputShape);

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperPipeline) return base.PreprocessText(text);
        var ids = EncodeText(text);
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(id => NumOps.FromDouble(id))));
    }

    private Tensor<T>? ResolveVoice()
    {
        if (Voice?.SpeakerEmbedding is { } embedding)
        {
            if (embedding.Length != SpeakerEmbeddingDim) throw new ArgumentException("Speaker embedding dimension differs.");
            return Engine.Reshape(embedding, new[] { 1, SpeakerEmbeddingDim });
        }
        return Voice?.ReferenceAudio is { } audio ? ExtractSpeakerEmbedding(audio, _options.SampleRate) : null;
    }

    /// <summary>Runs the original speaker frontend and LSTM, without imposing the synthesis UI's duration threshold.</summary>
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio, int sampleRate)
    {
        ThrowIfDisposed();
        if (_speaker is null || !_speakerLoaded) throw new InvalidOperationException("Load the released speaker encoder checkpoint.");
        if (sampleRate <= 0 || referenceAudio.Length == 0) throw new ArgumentException("A nonempty mono reference and valid sample rate are required.", nameof(referenceAudio));
        using var noGrad = new NoGradScope<T>();
        var values = referenceAudio.ToArray().Select(x => NumOps.ToDouble(x)).ToArray();
        if (sampleRate != 16000) values = SoxrResampler.LibrosaResample(values, sampleRate, 16000);
        values = ChatterboxVoiceEncoder<T>.TrimSilence(values, 20);
        return Engine.Reshape(_speaker.EmbedPreprocessed(new Tensor<T>(new[] { values.Length },
            new Vector<T>(values.Select(x => NumOps.FromDouble(x))))), new[] { SpeakerEmbeddingDim });
    }

    private double Uniform(Random random) => SamplingUniformForAudit?.Invoke() ?? random.NextDouble();

    private Tensor<T> SynthesizeNative(string text, Tensor<T>? speaker, int[]? firstStageTextTokens = null)
    {
        if (_first is null || _second is null || !_firstLoaded || !_secondLoaded || _firstTokenizer is null || _secondTokenizer is null)
            throw new InvalidOperationException("Load both original MetaVoice language-model checkpoints before synthesis.");
        if (_decoder is null && DecoderForAudit is null) throw new InvalidOperationException("Load the EnCodec and four-band diffusion checkpoints before synthesis.");
        if (!_options.RawDecoderOutput && _options.EnhanceOutput && _enhancer is null)
            throw new InvalidOperationException("Load the original DeepFilterNet3 checkpoint before released-output synthesis, or explicitly select raw decoder output.");
        using var noGrad = new NoGradScope<T>(); SetTrainingMode(false);
        var random = _options.SynthesisSeed is { } seed ? new Random(seed) : _random;
        // String synthesis owns normalization. Predict already receives tokenizer IDs:
        // preserve those IDs, including byte sequences that are not normalized prose.
        if (firstStageTextTokens is null) text = MetaVoiceText.Normalize(text);
        if (speaker is not null) speaker = Engine.Reshape(speaker, new[] { 1, SpeakerEmbeddingDim });
        var firstTokens = MetaVoiceGeneration.FirstStage(firstStageTextTokens ?? _firstTokenizer.Encode(text), _options.MaximumNewTokens,
            _options.FirstStageContext, (prefix, conditioned) => _first.Forward(new[] { prefix }, conditioned ? speaker : null,
                lastOnly: true).Logits[0].ToArray().Select(x => NumOps.ToDouble(x)).ToArray(), _ => Uniform(random), speaker is not null,
            _options.FirstStageTemperature, _options.FirstStageTopK, _options.FirstStageTopP,
            speaker is null ? null : (_options.SpeakerGuidance, _options.PromptGuidance));
        var firstAudio = MetaVoiceTokenAdapters.Decode(new[] { firstTokens }, true).Audio;
        if (firstAudio[0].Length == 0) throw new InvalidOperationException("MetaVoice generated no paired audio codebooks.");
        var input = MetaVoiceGeneration.SecondStageInput(_secondTokenizer.Encode(text), firstAudio, _options.SecondStageContext);
        var logits = _second.Forward(input, speaker).Logits;
        var arrays = logits.Select(x => x.ToArray().Select(value => NumOps.ToDouble(value)).ToArray()).ToArray();
        var predicted = MetaVoiceGeneration.NonCausal(6, _options.SecondStageContext,
            (head, frame) => arrays[head].Skip(frame * 1025).Take(1025).ToArray(), (_, _) => Uniform(random),
            _options.SecondStageTemperature, _options.SecondStageTopK);
        var audio = MetaVoiceTokenAdapters.Decode(input.Concat(predicted).ToArray(), false).Audio;
        if (audio[0].Length == 0) throw new InvalidOperationException("MetaVoice generated no complete eight-codebook frames.");
        var codes = new int[8, audio[0].Length];
        for (int q = 0; q < 8; q++) for (int t = 0; t < audio[0].Length; t++) codes[q, t] = audio[q][t];
        Tensor<T> Gaussian(int[] shape) => DiffusionGaussianForAudit?.Invoke(shape)
            ?? new Tensor<T>(shape, TorchParameters.Gaussian<T>(random, shape.Aggregate(1, (a, b) => a * b), 1));
        var waveform = DecoderForAudit is { } audit ? audit(codes) : _decoder!.Decode(codes,
            (_, shape) => Gaussian(shape), (_, _, shape) => Gaussian(shape));
        waveform = Engine.Reshape(waveform, new[] { waveform.Length });
        if (_options.RawDecoderOutput) return waveform;
        var normalized = MetaVoiceOutputPolicy.Normalize(waveform.ToArray().Select(x => NumOps.ToDouble(x)).ToArray());
        ObserveOutputForAudit?.Invoke("normalized", new Tensor<T>(new[] { normalized.Length }, new Vector<T>(normalized.Select(x => NumOps.FromDouble(x)))));
        var pcm = MetaVoiceOutputPolicy.DecoderPcm(normalized);
        waveform = new Tensor<T>(new[] { pcm.Length }, new Vector<T>(pcm.Select(x => NumOps.FromDouble(x / 32768.0))));
        if (!_options.EnhanceOutput) return waveform;
        // The pinned legacy sinc_fast alias effectively selects Kaiser in Torchaudio 2.1.
        waveform = new MetaVoiceSincFastResample<T>(Engine).Forward(waveform);
        ObserveOutputForAudit?.Invoke("resampled", waveform);
        var enhanced = _enhancer!.Enhance48k(waveform);
        ObserveOutputForAudit?.Invoke("enhanced", enhanced);
        var finalPcm = MetaVoiceOutputPolicy.EnhancedPcm(enhanced.ToArray().Select(x => NumOps.ToDouble(x)).ToArray());
        return new Tensor<T>(new[] { finalPcm.Length }, new Vector<T>(finalPcm.Select(x => NumOps.FromDouble(x / 32768.0))));
    }

    private int[] IntegerIds(Tensor<T> values)
    {
        var result = new int[values.Length];
        for (int i = 0; i < result.Length; i++)
        {
            double value = NumOps.ToDouble(values[i]);
            if (double.IsNaN(value) || double.IsInfinity(value) || value != Math.Truncate(value) || value < 0 || value > int.MaxValue)
                throw new ArgumentException("Expected integer token IDs.", nameof(values));
            result[i] = (int)value;
        }
        return result;
    }
}
