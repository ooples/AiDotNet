using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>MetaVoice1B: MetaVoice-1B: 1.2B Parameter Voice Cloning Model.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "MetaVoice-1B: 1.2B Parameter Voice Cloning Model" (MetaVoice Team, 2024)</item></list></para><para><b>For Beginners:</b> MetaVoice1B: MetaVoice-1B: 1.2B Parameter Voice Cloning Model.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a MetaVoice-1B model for large-scale voice cloning
/// // with 1.2B parameters for high-fidelity zero-shot synthesis
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new MetaVoice1B&lt;double&gt;(architecture, "metavoice1b.onnx");
///
/// // Training mode with native layers
/// var trainModel = new MetaVoice1B&lt;double&gt;(architecture, new MetaVoice1BOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "MetaVoice-1B: 1.2B Parameter Voice Cloning Model",
    "https://github.com/metavoiceio/metavoice-src"
)]
public class MetaVoice1B<T> : TtsModelBase<T>, IEndToEndTts<T>, IVoiceCloner<T>
{
    private readonly MetaVoice1BOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public MetaVoice1B(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        MetaVoice1BOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new MetaVoice1BOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public MetaVoice1B(
        NeuralNetworkArchitecture<T> architecture,
        MetaVoice1BOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new MetaVoice1BOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int HiddenDim => _options.HiddenDim;
    public int NumFlowSteps => 0;

    /// <summary>
    /// Synthesizes speech from text by running the full MetaVoice-1B pipeline:
    /// text+speaker conditioning embeddings → first-stage causal transformer → second-stage
    /// non-causal transformer → HiFi-GAN vocoder → waveform (metavoiceio/metavoice-src).
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessText(text));
        var conditioning = BuildConditioningSequence(text, speakerEmbedding: null);
        var audio = PredictCore(conditioning);
        return FlattenWaveform(audio);
    }

    public double MinReferenceDuration => 3.0;
    public int SpeakerEmbeddingDim => _options.SpeakerEmbeddingDim;

    /// <summary>
    /// Zero-shot voice cloning: extracts a speaker/tone embedding from the reference audio and
    /// conditions the generation on it (additive speaker conditioning, per the reference's
    /// <c>tok_emb + pos_emb + spk_emb</c>), then runs the full pipeline.
    /// </summary>
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        var embedding = ExtractSpeakerEmbedding(referenceAudio);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessText(text));
        var conditioning = BuildConditioningSequence(text, embedding);
        var audio = PredictCore(conditioning);
        return FlattenWaveform(audio);
    }

    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        int embDim = _options.SpeakerEmbeddingDim;
        var embedding = new Tensor<T>([embDim]);
        int chunkSize = Math.Max(1, referenceAudio.Length / embDim);
        for (int i = 0; i < embDim; i++)
        {
            double sum = 0;
            for (int j = 0; j < chunkSize && i * chunkSize + j < referenceAudio.Length; j++)
                sum += NumOps.ToDouble(referenceAudio[i * chunkSize + j]);
            embedding[i] = NumOps.FromDouble(Math.Tanh(sum / chunkSize));
        }
        return L2Normalize(embedding);
    }

    /// <summary>
    /// Builds the rank-3 <c>[1, seq, SpeakerEmbeddingDim]</c> conditioning-embedding sequence the
    /// native pipeline consumes: a per-token text embedding, additively fused with the optional
    /// speaker/tone embedding (broadcast across the sequence), matching the reference model's
    /// additive <c>tok_emb + pos_emb + spk_emb</c> conditioning.
    /// </summary>
    private Tensor<T> BuildConditioningSequence(string text, Tensor<T>? speakerEmbedding)
    {
        int condDim = _options.SpeakerEmbeddingDim;
        int seq = Math.Max(1, Math.Min(text?.Length ?? 0, _options.MaxTextLength));
        var conditioning = new Tensor<T>([1, seq, condDim]);
        for (int s = 0; s < seq; s++)
        {
            double token = (text is { Length: > 0 } && s < text.Length) ? text[s] / 128.0 : 0.0;
            double posPhase = (s + 1.0) / (seq + 1.0);
            for (int c = 0; c < condDim; c++)
            {
                double freq = 1.0 / Math.Pow(10000.0, (2.0 * (c / 2)) / condDim);
                double tokEmb = Math.Sin(token * (c + 1)) * 0.5;
                double posEmb = ((c & 1) == 0 ? Math.Sin(posPhase / freq) : Math.Cos(posPhase / freq)) * 0.5;
                double spkEmb = speakerEmbedding is not null
                    ? NumOps.ToDouble(speakerEmbedding[c % speakerEmbedding.Length])
                    : 0.0;
                conditioning[0, s, c] = NumOps.FromDouble(tokEmb + posEmb + spkEmb);
            }
        }
        return conditioning;
    }

    /// <summary>Flattens the vocoder's rank-3 <c>[1, 1, L]</c> waveform output to a rank-1 <c>[L]</c> signal.</summary>
    private Tensor<T> FlattenWaveform(Tensor<T> audio)
    {
        var waveform = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
            waveform[i] = audio[i];
        return waveform;
    }

    protected override Tensor<T> PreprocessText(string text)
    {
        int len = Math.Min(text.Length, _options.MaxTextLength);
        var t = new Tensor<T>([len]);
        for (int i = 0; i < len; i++)
            t[i] = NumOps.FromDouble(text[i] / 128.0);
        return t;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultMetaVoice1BLayers(
                    firstStageDim: _options.FirstStageDim,
                    numFirstStageLayers: _options.NumFirstStageLayers,
                    secondStageDim: _options.SecondStageDim,
                    numSecondStageLayers: _options.NumSecondStageLayers,
                    numHeads: _options.NumHeads,
                    numCodebooks: _options.NumCodebooks,
                    firstStageCodebooks: _options.FirstStageCodebooks,
                    codecLatentDim: _options.CodecLatentDim,
                    vocoderChannels: _options.VocoderChannels,
                    vocoderUpsampleFactor: _options.VocoderUpsampleFactor,
                    swiGLUMultipleOf: _options.SwiGLUMultipleOf,
                    ropeTheta: _options.RoPETheta,
                    maxSeqLen: _options.MaxCodecFrames,
                    dropoutRate: _options.DropoutRate
                )
            );
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        TrainWithTape(input, expected);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers)
        {
            int c = (int)l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MetaVoice1B-Native" : "MetaVoice1B-ONNX",
            Description = "MetaVoice1B TTS",
            FeatureCount = _options.HiddenDim,
        };
        m.AdditionalInfo["Architecture"] = "MetaVoice1B";
        m.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = base.HiddenDim;
        m.AdditionalInfo["SampleRate"] = base.SampleRate;
        m.AdditionalInfo["MelChannels"] = base.MelChannels;
        m.AdditionalInfo["HopSize"] = base.HopSize;
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.SampleRate);
        writer.Write(_options.MelChannels);
        writer.Write(_options.HopSize);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.EncoderDim);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.SpeakerEmbeddingDim);
        writer.Write(_options.FirstStageDim);
        writer.Write(_options.NumFirstStageLayers);
        writer.Write(_options.SecondStageDim);
        writer.Write(_options.NumSecondStageLayers);
        writer.Write(_options.NumCodebooks);
        writer.Write(_options.FirstStageCodebooks);
        writer.Write(_options.CodecLatentDim);
        writer.Write(_options.VocoderChannels);
        writer.Write(_options.VocoderUpsampleFactor);
        writer.Write(_options.SwiGLUMultipleOf);
        writer.Write(_options.RoPETheta);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.SampleRate = reader.ReadInt32();
        _options.MelChannels = reader.ReadInt32();
        _options.HopSize = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.EncoderDim = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumEncoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.SpeakerEmbeddingDim = reader.ReadInt32();
        _options.FirstStageDim = reader.ReadInt32();
        _options.NumFirstStageLayers = reader.ReadInt32();
        _options.SecondStageDim = reader.ReadInt32();
        _options.NumSecondStageLayers = reader.ReadInt32();
        _options.NumCodebooks = reader.ReadInt32();
        _options.FirstStageCodebooks = reader.ReadInt32();
        _options.CodecLatentDim = reader.ReadInt32();
        _options.VocoderChannels = reader.ReadInt32();
        _options.VocoderUpsampleFactor = reader.ReadInt32();
        _options.SwiGLUMultipleOf = reader.ReadInt32();
        _options.RoPETheta = reader.ReadDouble();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new MetaVoice1B<T>(Architecture, mp, _options);
        return new MetaVoice1B<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(MetaVoice1B<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
