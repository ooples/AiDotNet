using AiDotNet.Attributes;
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
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Qwen2-Audio Technical Report", "https://doi.org/10.48550/arXiv.2407.10759", Year = 2024, Authors = "Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, Jingren Zhou")]
public partial class Qwen2Audio<T> : AudioNeuralNetworkBase<T>, IAudioLanguageModel<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Traced from output construction: PredictCore folds over Layers, and the last layer
    /// CreateDefaultQwen2AudioLayers emits is the output projection
    /// <c>FullyConnectedLayer&lt;T&gt;(lmHiddenDim)</c>, wired from <c>_options.LMHiddenDim</c>
    /// (3584). Explicitly NOT VocabSize (151936): the default stack stops at the LM embedding space
    /// and never ties an unembedding head, so reading the vocabulary field would overstate the width
    /// by ~42x.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.LMHiddenDim;

    #region Fields

    private readonly Qwen2AudioOptions _options;
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
        Options = _options;
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
        Options = _options;
        _useNativeMode = true;
        // The paper rate from this model's own options. Built bare, AdamW would use its
        // own 1e-3 default instead and LearningRate would be configuration nobody reads —
        // the defect that diverged MusicFlamingo's training.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            { InitialLearningRate = _options.LearningRate });
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

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Qwen2-Audio-Native" : "Qwen2-Audio-ONNX",
            Description = "Qwen2-Audio multimodal audio-language model (Chu et al., 2024, Alibaba)",
            Complexity = _options.NumAudioEncoderLayers + _options.NumLMLayers
        };
        m.AdditionalInfo["LMHiddenDim"] = _options.LMHiddenDim.ToString();
        m.AdditionalInfo["VocabSize"] = _options.VocabSize.ToString();
        return m;
    }





    #endregion

    #region Private Helpers

    private Tensor<T> AdaptAudioFeatures(Tensor<T> audioEmbed)
    {
        // Perceiver-style adapter: project to LM dimension (the first values, zero-padded; a fresh tensor)
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return AudioEngineOps.FitLength(Engine, audioEmbed, _options.NumLatentTokens * _options.AdapterDim / _options.NumLatentTokens).Clone();
    }

    private Tensor<T> EncodePrompt(string prompt)
    {
        var embedding = new Tensor<T>([_options.LMHiddenDim]);
        // A stable hash: string.GetHashCode() changes per process, so the same prompt would encode differently each run (#2361).
        int hash = AiDotNet.Helpers.StableHash.Seed(prompt);
        for (int i = 0; i < _options.LMHiddenDim; i++)
        {
            double val = Math.Sin((hash + i) * 0.1) * 0.5;
            embedding[i] = NumOps.FromDouble(val);
        }
        return embedding;
    }

    private Tensor<T> CombineModalEmbeddings(Tensor<T> audioEmbed, Tensor<T> textEmbed)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Engine.Concat(new[] { AudioEngineOps.Flatten(Engine, audioEmbed), AudioEngineOps.Flatten(Engine, textEmbed) }, 0);
    }

    private string DecodeToText(Tensor<T> output, int maxTokens)
    {
        int numTokens = Math.Min(maxTokens, output.Length);
        if (numTokens <= 0)
            return _tokenizer.Decode(new List<int>());

        // Token ids: round to nearest (banker's), clamped to [0, vocab - 1]; NaN maps to 0
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var rounded = Engine.TensorRound(AudioEngineOps.FitLength(Engine, output, numTokens));
        var zeros = new Tensor<T>([numTokens]);
        T maxId = NumOps.FromDouble(_tokenizer.VocabularySize - 1);
        var clamped = Engine.TensorWhere(Engine.TensorLessThan(rounded, NumOps.Zero), zeros, rounded);
        clamped = Engine.TensorWhere(
            Engine.TensorGreaterThan(clamped, maxId), Tensor<T>.CreateDefault([numTokens], maxId), clamped);
        clamped = Engine.TensorWhere(Engine.TensorIsNan(clamped), zeros, clamped);
        return _tokenizer.Decode(clamped.Cast<int>().ToArray().ToList());
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Qwen2Audio<T>)); }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
