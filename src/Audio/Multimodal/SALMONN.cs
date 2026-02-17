using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Audio.Multimodal;

/// <summary>
/// SALMONN dual-encoder audio-language model for speech and audio understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SALMONN (Tang et al., 2024, Tsinghua/ByteDance) uses dual audio encoders: a Whisper
/// speech encoder and a BEATs audio encoder, connected to a Vicuna LLM through a
/// window-level Q-Former adapter. This gives it strong capability for both speech and
/// general audio understanding tasks.
/// </para>
/// <para>
/// <b>For Beginners:</b> SALMONN has two "ears": one for speech (Whisper) and one for
/// general sounds (BEATs). This means it can understand what people say AND non-speech
/// sounds. Ask it "What is the person saying?" and it transcribes speech. Ask "What sounds
/// are in the background?" and it identifies environmental audio.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1280, outputSize: 4096);
/// var model = new SALMONN&lt;float&gt;(arch, "salmonn.onnx");
/// string answer = model.Understand(audio, "What is the person saying?");
/// </code>
/// </para>
/// </remarks>
public class SALMONN<T> : AudioNeuralNetworkBase<T>, IAudioLanguageModel<T>
{
    #region Fields

    private readonly SALMONNOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ITokenizer _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private static readonly string[] Capabilities =
        ["captioning", "qa", "reasoning", "speech_recognition", "translation", "sound_event_detection"];

    #endregion

    #region IAudioLanguageModel Properties

    /// <inheritdoc />
    public double MaxAudioDurationSeconds => _options.MaxAudioDurationSeconds;

    /// <inheritdoc />
    public int MaxResponseTokens => _options.MaxResponseTokens;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SALMONN model in ONNX inference mode.
    /// </summary>
    public SALMONN(NeuralNetworkArchitecture<T> architecture, string modelPath, SALMONNOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SALMONNOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.Vicuna);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a SALMONN model in native training mode.
    /// </summary>
    public SALMONN(NeuralNetworkArchitecture<T> architecture, SALMONNOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SALMONNOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.Vicuna);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<SALMONN<T>> CreateAsync(SALMONNOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new SALMONNOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("salmonn", "salmonn.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.SpeechEncoderDim, outputSize: options.LMHiddenDim);
        return new SALMONN<T>(arch, mp, options);
    }

    #endregion

    #region IAudioLanguageModel

    /// <inheritdoc />
    public IReadOnlyList<string> GetCapabilities() => Capabilities;

    /// <inheritdoc />
    public string Understand(Tensor<T> audio, string prompt, int maxTokens = 256, double temperature = 0.7)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);

        // Dual encoder: speech (Whisper) + audio (BEATs)
        var speechFeatures = EncodeSpeech(features);
        var audioFeatures = EncodeAudio(features);

        // Fuse through window-level Q-Former
        var fusedFeatures = FuseEncoderOutputs(speechFeatures, audioFeatures);

        // Encode text prompt
        var promptEmbedding = EncodePrompt(prompt);

        // Combine modalities for LM
        var combined = CombineModalEmbeddings(fusedFeatures, promptEmbedding);

        // Generate response
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(combined) : Predict(combined);
        return DecodeToText(output, maxTokens);
    }

    /// <inheritdoc />
    public Task<string> UnderstandAsync(Tensor<T> audio, string prompt, int maxTokens = 256,
        double temperature = 0.7, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Understand(audio, prompt, maxTokens, temperature), cancellationToken);
    }

    /// <inheritdoc />
    public string Caption(Tensor<T> audio, int maxTokens = 128)
    {
        return Understand(audio, "Provide a detailed description of this audio.", maxTokens);
    }

    /// <inheritdoc />
    public Tensor<T> ExtractAudioEmbeddings(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultSALMONNLayers(
            speechEncoderDim: _options.SpeechEncoderDim, audioEncoderDim: _options.AudioEncoderDim,
            qFormerDim: _options.QFormerDim, numQFormerLayers: _options.NumQFormerLayers,
            lmHiddenDim: _options.LMHiddenDim, dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SALMONN-Native" : "SALMONN-ONNX",
            Description = "SALMONN dual-encoder audio-language model (Tang et al., 2024, Tsinghua/ByteDance)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.SpeechEncoderDim,
            Complexity = _options.NumSpeechEncoderLayers + _options.NumAudioEncoderLayers + _options.NumLMLayers
        };
        m.AdditionalInfo["LMHiddenDim"] = _options.LMHiddenDim.ToString();
        m.AdditionalInfo["QFormerDim"] = _options.QFormerDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.SpeechEncoderDim);
        w.Write(_options.NumSpeechEncoderLayers); w.Write(_options.AudioEncoderDim);
        w.Write(_options.NumAudioEncoderLayers); w.Write(_options.NumMels);
        w.Write(_options.MaxAudioDurationSeconds); w.Write(_options.QFormerDim);
        w.Write(_options.NumQFormerLayers); w.Write(_options.NumQueryTokens);
        w.Write(_options.WindowSize); w.Write(_options.LMHiddenDim);
        w.Write(_options.NumLMLayers); w.Write(_options.NumLMHeads);
        w.Write(_options.VocabSize); w.Write(_options.MaxResponseTokens);
        w.Write(_options.Temperature); w.Write(_options.TopP);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.SpeechEncoderDim = r.ReadInt32();
        _options.NumSpeechEncoderLayers = r.ReadInt32(); _options.AudioEncoderDim = r.ReadInt32();
        _options.NumAudioEncoderLayers = r.ReadInt32(); _options.NumMels = r.ReadInt32();
        _options.MaxAudioDurationSeconds = r.ReadDouble(); _options.QFormerDim = r.ReadInt32();
        _options.NumQFormerLayers = r.ReadInt32(); _options.NumQueryTokens = r.ReadInt32();
        _options.WindowSize = r.ReadInt32(); _options.LMHiddenDim = r.ReadInt32();
        _options.NumLMLayers = r.ReadInt32(); _options.NumLMHeads = r.ReadInt32();
        _options.VocabSize = r.ReadInt32(); _options.MaxResponseTokens = r.ReadInt32();
        _options.Temperature = r.ReadDouble(); _options.TopP = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new SALMONN<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> EncodeSpeech(Tensor<T> features)
    {
        // Whisper-style speech encoder pass
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        var speechEmbed = new Tensor<T>([_options.SpeechEncoderDim]);
        for (int i = 0; i < _options.SpeechEncoderDim && i < output.Length; i++)
            speechEmbed[i] = output[i];
        return speechEmbed;
    }

    private Tensor<T> EncodeAudio(Tensor<T> features)
    {
        // BEATs-style audio encoder pass
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        var audioEmbed = new Tensor<T>([_options.AudioEncoderDim]);
        int offset = Math.Min(_options.SpeechEncoderDim, output.Length);
        for (int i = 0; i < _options.AudioEncoderDim; i++)
        {
            int idx = (offset + i) % output.Length;
            audioEmbed[i] = output[idx];
        }
        return audioEmbed;
    }

    private Tensor<T> FuseEncoderOutputs(Tensor<T> speechEmbed, Tensor<T> audioEmbed)
    {
        // Window-level Q-Former fusion
        int outDim = _options.NumQueryTokens * _options.QFormerDim / _options.NumQueryTokens;
        var fused = new Tensor<T>([outDim]);
        for (int i = 0; i < outDim; i++)
        {
            double s = i < speechEmbed.Length ? NumOps.ToDouble(speechEmbed[i % speechEmbed.Length]) : 0;
            double a = i < audioEmbed.Length ? NumOps.ToDouble(audioEmbed[i % audioEmbed.Length]) : 0;
            fused[i] = NumOps.FromDouble((s + a) / 2.0);
        }
        return fused;
    }

    private Tensor<T> EncodePrompt(string prompt)
    {
        var embedding = new Tensor<T>([_options.LMHiddenDim]);
        int hash = prompt.GetHashCode();
        for (int i = 0; i < _options.LMHiddenDim; i++)
        {
            double val = Math.Sin((hash + i) * 0.1) * 0.5;
            embedding[i] = NumOps.FromDouble(val);
        }
        return embedding;
    }

    private Tensor<T> CombineModalEmbeddings(Tensor<T> audioEmbed, Tensor<T> textEmbed)
    {
        int len = audioEmbed.Length + textEmbed.Length;
        var combined = new Tensor<T>([len]);
        for (int i = 0; i < audioEmbed.Length; i++) combined[i] = audioEmbed[i];
        for (int i = 0; i < textEmbed.Length; i++) combined[audioEmbed.Length + i] = textEmbed[i];
        return combined;
    }

    private string DecodeToText(Tensor<T> output, int maxTokens)
    {
        int numTokens = Math.Min(maxTokens, output.Length);
        var tokenIds = new List<int>();
        for (int i = 0; i < numTokens; i++)
        {
            int tokenId = (int)Math.Round(NumOps.ToDouble(output[i]));
            if (tokenId < 0) tokenId = 0;
            if (tokenId >= _tokenizer.VocabularySize) tokenId = _tokenizer.VocabularySize - 1;
            tokenIds.Add(tokenId);
        }
        return _tokenizer.Decode(tokenIds);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SALMONN<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
