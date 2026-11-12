using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>MetaVoice's learned-position causal GPT, noncausal codebook GPT, speaker LSTM
/// and EnCodec-conditioned four-band diffusion waveform generator.</summary>
/// <remarks>The released output includes loudness normalization and DeepFilterNet3 enhancement
/// at 48 kHz. RawDecoderOutput selects the original 24 kHz neural decoder waveform.</remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MetaVoice-1B: 1.2B Parameter Voice Cloning Model", "https://github.com/metavoiceio/metavoice-src")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-5, Beta1 = .9, Beta2 = .95, Epsilon = 1e-8,
    WeightDecay = .1, Provenance = RecipeProvenance.ReleasedCode,
    Source = "MetaVoice fam/llm/config/finetune_params.py and finetune.py: AdamW; final norm and last transformer blocks only.")]
public partial class MetaVoice1B<T> : TtsModelBase<T>, IEndToEndTts<T>, IVoiceCloner<T>
{
    private readonly MetaVoice1BOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly bool _useNativeMode;
    private readonly bool _usesDefaultOptimizer;
    private readonly Random _random;
    private bool _disposed;

    /// <summary>Loads an exported ONNX graph.</summary>
    public MetaVoice1B(NeuralNetworkArchitecture<T> architecture, string modelPath, MetaVoice1BOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MetaVoice1BOptions(); Options = _options;
        _random = new Random(InitializationSeed());
        ConfigureProperties();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException("ONNX model not found.", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    /// <summary>Constructs the actual native transformer and speaker graphs. Load the released
    /// actor and decoder checkpoints before waveform synthesis.</summary>
    public MetaVoice1B(NeuralNetworkArchitecture<T> architecture, MetaVoice1BOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MetaVoice1BOptions(); Options = _options;
        _useNativeMode = true; _usesDefaultOptimizer = optimizer is null; _random = new Random(InitializationSeed());
        _optimizer = optimizer ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        ConfigureProperties(); InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int HiddenDim => _options.FirstStageDim;
    public int NumFlowSteps => 0;
    public double MinReferenceDuration => _options.MinReferenceDurationSec;
    public int SpeakerEmbeddingDim => _options.SpeakerEmbeddingDim;
    /// <summary>Whether the actual native language and speaker graphs are present.</summary>
    public bool HasPaperPipeline => _first is not null && _second is not null && _speaker is not null;

    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperPipeline)
        {
            var waveform = PredictCore(PreprocessText(text));
            return Engine.Reshape(waveform, new[] { waveform.Length });
        }
        return SynthesizeNative(text, ResolveVoice());
    }

    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio)
        => SynthesizeWithVoice(text, referenceAudio, _options.ReferenceSampleRate);

    /// <summary>Clones a mono recording at an explicitly specified sample rate.</summary>
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio, int sampleRate)
    {
        ThrowIfDisposed();
        if (!HasPaperPipeline) throw new NotSupportedException("Voice cloning requires the native MetaVoice graphs.");
        if (sampleRate <= 0 || referenceAudio.Length / (double)sampleRate < MinReferenceDuration)
            throw new ArgumentException($"The released synthesis frontend requires at least {MinReferenceDuration} seconds of reference audio.", nameof(referenceAudio));
        return SynthesizeNative(text, ExtractSpeakerEmbedding(referenceAudio, sampleRate));
    }

    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
        => ExtractSpeakerEmbedding(referenceAudio, _options.ReferenceSampleRate);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperPipeline)
        {
            var output = input;
            foreach (var layer in Layers) output = layer.Forward(output);
            return output;
        }
        var tokenizer = _firstTokenizer ?? throw new InvalidOperationException("Load the first-stage tokenizer checkpoint.");
        var ids = IntegerIds(input);
        // The first EOT terminates the text; any remaining values are padding.
        int end = Array.IndexOf(ids, tokenizer.EndToken);
        if (end >= 0) ids = ids.Take(end).ToArray();
        return SynthesizeNative(tokenizer.Decode(ids), ResolveVoice(), ids.Concat(new[] { tokenizer.EndToken }).ToArray());
    }

    protected override bool SupportsParameterMutation => _useNativeMode;

    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var layer in ComponentLayers) layer.SetTrainingMode(false);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MetaVoice1B-Native" : "MetaVoice1B-ONNX",
            Description = "MetaVoice causal/noncausal GPT and four-band diffusion",
            FeatureCount = _options.FirstStageDim
        };
        metadata.AdditionalInfo["Architecture"] = "MetaVoice-1B";
        metadata.AdditionalInfo["NativePaperGraph"] = HasPaperPipeline;
        metadata.AdditionalInfo["PositionEncoding"] = "Learned absolute";
        metadata.AdditionalInfo["Decoder"] = "Eight-codebook EnCodec, four-band diffusion, trained processors, 32-band re-equalization";
        metadata.AdditionalInfo["SampleRate"] = SampleRate;
        metadata.AdditionalInfo["NeuralDecoderSampleRate"] = _options.SampleRate;
        metadata.AdditionalInfo["RawDecoderOutput"] = _options.RawDecoderOutput;
        metadata.AdditionalInfo["ExternalEnhancementEnabled"] = !_options.RawDecoderOutput && _options.EnhanceOutput;
        return metadata;
    }

    private void ConfigureProperties()
    {
        base.SampleRate = _useNativeMode && Architecture.Layers is not { Count: > 0 } &&
            !_options.RawDecoderOutput && _options.EnhanceOutput ? 48000 : _options.SampleRate;
        base.MelChannels = _options.SpeakerMelChannels;
        base.HopSize = _options.HopSize; base.HiddenDim = _options.FirstStageDim;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MetaVoice1B<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) _codec?.Dispose();
        base.Dispose(disposing);
    }
}
