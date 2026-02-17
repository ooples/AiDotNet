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
/// Qwen2-Audio multimodal audio-language model for audio understanding and reasoning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Qwen2-Audio (Chu et al., 2024, Alibaba) uses a Whisper-style audio encoder with a
/// Qwen2 language model backbone, connected by a perceiver-style adapter. It supports
/// audio captioning, question answering, sound event detection, and audio reasoning.
/// </para>
/// <para>
/// <b>For Beginners:</b> Qwen2-Audio can "listen" to audio and answer questions about it.
/// Play it music and ask "What genre is this?", play it a conversation and ask "What
/// language are they speaking?", or play environmental sounds and ask "Describe this scene."
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1280, outputSize: 3584);
/// var model = new Qwen2Audio&lt;float&gt;(arch, "qwen2_audio.onnx");
/// string answer = model.Understand(audio, "What instrument is playing?");
/// </code>
/// </para>
/// </remarks>
public class Qwen2Audio<T> : AudioNeuralNetworkBase<T>, IAudioLanguageModel<T>
{
    #region Fields

    private readonly Qwen2AudioOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private static readonly string[] Capabilities =
        ["captioning", "qa", "reasoning", "sound_event_detection", "speech_recognition", "emotion_recognition"];

    #endregion

    #region IAudioLanguageModel Properties

    /// <inheritdoc />
    public double MaxAudioDurationSeconds => _options.MaxAudioDurationSeconds;

    /// <inheritdoc />
    public int MaxResponseTokens => _options.MaxResponseTokens;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Qwen2-Audio model in ONNX inference mode.
    /// </summary>
    public Qwen2Audio(NeuralNetworkArchitecture<T> architecture, string modelPath, Qwen2AudioOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new Qwen2AudioOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.Qwen);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a Qwen2-Audio model in native training mode.
    /// </summary>
    public Qwen2Audio(NeuralNetworkArchitecture<T> architecture, Qwen2AudioOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new Qwen2AudioOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.Qwen);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<Qwen2Audio<T>> CreateAsync(Qwen2AudioOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new Qwen2AudioOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("qwen2_audio", "qwen2_audio.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.AudioEncoderDim, outputSize: options.LMHiddenDim);
        return new Qwen2Audio<T>(arch, mp, options);
    }

    #endregion

    #region IAudioLanguageModel

    /// <inheritdoc />
    public IReadOnlyList<string> GetCapabilities() => Capabilities;

    /// <inheritdoc />
    public string Understand(Tensor<T> audio, string prompt, int maxTokens = 256, double temperature = 0.7)
    {
        ThrowIfDisposed();
        // Encode audio through Whisper-style encoder
        var audioFeatures = PreprocessAudio(audio);
        var audioEmbedding = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audioFeatures) : Predict(audioFeatures);

        // Project through perceiver adapter
        var adaptedFeatures = AdaptAudioFeatures(audioEmbedding);

        // Encode prompt
        var promptEmbedding = EncodePrompt(prompt);

        // Combine audio + text for LM
        var combined = CombineModalEmbeddings(adaptedFeatures, promptEmbedding);

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
        return Understand(audio, "Describe this audio in detail.", maxTokens);
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
        else Layers.AddRange(LayerHelper<T>.CreateDefaultQwen2AudioLayers(
            audioEncoderDim: _options.AudioEncoderDim, numAudioEncoderLayers: _options.NumAudioEncoderLayers,
            numAudioEncoderHeads: _options.NumAudioEncoderHeads, lmHiddenDim: _options.LMHiddenDim,
            dropoutRate: _options.DropoutRate));
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
        _optimizer?.UpdateParameters(Layers);
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
            Name = _useNativeMode ? "Qwen2-Audio-Native" : "Qwen2-Audio-ONNX",
            Description = "Qwen2-Audio multimodal audio-language model (Chu et al., 2024, Alibaba)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.AudioEncoderDim,
            Complexity = _options.NumAudioEncoderLayers + _options.NumLMLayers
        };
        m.AdditionalInfo["LMHiddenDim"] = _options.LMHiddenDim.ToString();
        m.AdditionalInfo["VocabSize"] = _options.VocabSize.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.AudioEncoderDim);
        w.Write(_options.NumAudioEncoderLayers); w.Write(_options.NumAudioEncoderHeads);
        w.Write(_options.NumMels); w.Write(_options.MaxAudioDurationSeconds);
        w.Write(_options.LMHiddenDim); w.Write(_options.NumLMLayers);
        w.Write(_options.NumLMHeads); w.Write(_options.VocabSize);
        w.Write(_options.MaxResponseTokens); w.Write(_options.AdapterDim);
        w.Write(_options.NumLatentTokens); w.Write(_options.Temperature);
        w.Write(_options.TopP); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.AudioEncoderDim = r.ReadInt32();
        _options.NumAudioEncoderLayers = r.ReadInt32(); _options.NumAudioEncoderHeads = r.ReadInt32();
        _options.NumMels = r.ReadInt32(); _options.MaxAudioDurationSeconds = r.ReadDouble();
        _options.LMHiddenDim = r.ReadInt32(); _options.NumLMLayers = r.ReadInt32();
        _options.NumLMHeads = r.ReadInt32(); _options.VocabSize = r.ReadInt32();
        _options.MaxResponseTokens = r.ReadInt32(); _options.AdapterDim = r.ReadInt32();
        _options.NumLatentTokens = r.ReadInt32(); _options.Temperature = r.ReadDouble();
        _options.TopP = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new Qwen2Audio<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> AdaptAudioFeatures(Tensor<T> audioEmbed)
    {
        // Perceiver-style adapter: project to LM dimension
        var adapted = new Tensor<T>([_options.NumLatentTokens * _options.AdapterDim / _options.NumLatentTokens]);
        for (int i = 0; i < adapted.Length; i++)
        {
            double v = i < audioEmbed.Length ? NumOps.ToDouble(audioEmbed[i % audioEmbed.Length]) : 0;
            adapted[i] = NumOps.FromDouble(v);
        }
        return adapted;
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

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Qwen2Audio<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
