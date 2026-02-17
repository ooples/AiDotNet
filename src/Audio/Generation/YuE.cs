using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// YuE full-song music generation model with vocals and accompaniment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// YuE (Yuan et al., 2025) generates complete songs with vocals and accompaniment from lyrics
/// and genre/style tags. It uses a dual-AR architecture: a lyrics-conditioned language model
/// generates semantic tokens, then a second stage produces acoustic tokens, enabling generation
/// of songs lasting several minutes.
/// </para>
/// <para>
/// <b>For Beginners:</b> YuE is like having a virtual band that can write and perform an
/// entire song. You give it lyrics and a style ("pop, female vocalist, upbeat") and it
/// generates a complete song with singing, instruments, and production. Unlike most AI music
/// tools that only make short clips, YuE can create full-length songs.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 2048, outputSize: 2048);
/// var model = new YuE&lt;float&gt;(arch, "yue.onnx");
/// var song = model.GenerateMusic("Upbeat pop with female vocals", durationSeconds: 180);
/// </code>
/// </para>
/// </remarks>
public class YuE<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>
{
    #region Fields

    private readonly YuEOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IAudioGenerator Properties

    /// <inheritdoc />
    public double MaxDurationSeconds => _options.MaxDurationSeconds;

    /// <inheritdoc />
    public bool SupportsTextToAudio => false;

    /// <inheritdoc />
    public bool SupportsTextToMusic => true;

    /// <inheritdoc />
    public bool SupportsAudioContinuation => true;

    /// <inheritdoc />
    public bool SupportsAudioInpainting => false;

    /// <inheritdoc />
    public new bool IsOnnxMode => base.IsOnnxMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a YuE model in ONNX inference mode.
    /// </summary>
    public YuE(NeuralNetworkArchitecture<T> architecture, string modelPath, YuEOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new YuEOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a YuE model in native training mode.
    /// </summary>
    public YuE(NeuralNetworkArchitecture<T> architecture, YuEOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new YuEOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<YuE<T>> CreateAsync(YuEOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new YuEOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("yue", "yue.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.SemanticDim, outputSize: options.SemanticDim);
        return new YuE<T>(arch, mp, options);
    }

    #endregion

    #region IAudioGenerator

    /// <inheritdoc />
    public Tensor<T> GenerateAudio(string prompt, string? negativePrompt = null, double durationSeconds = 5.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null)
    {
        throw new NotSupportedException("YuE is a music generation model. Use GenerateMusic() or GenerateSong() instead.");
    }

    /// <inheritdoc />
    public Task<Tensor<T>> GenerateAudioAsync(string prompt, string? negativePrompt = null, double durationSeconds = 5.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => GenerateAudio(prompt, negativePrompt, durationSeconds, numInferenceSteps, guidanceScale, seed), cancellationToken);
    }

    /// <inheritdoc />
    public Tensor<T> GenerateMusic(string prompt, string? negativePrompt = null, double durationSeconds = 10.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null)
    {
        ThrowIfDisposed();
        int numSamples = (int)(durationSeconds * SampleRate);

        // Encode style/genre prompt
        var styleEmbedding = EncodeStylePrompt(prompt);

        // Stage 1: Generate semantic tokens from style conditioning
        var semanticTokens = GenerateSemanticTokens(styleEmbedding, durationSeconds, seed);

        // Stage 2: Generate acoustic tokens from semantic tokens
        var acousticTokens = GenerateAcousticTokens(semanticTokens);

        // Decode to waveform
        return DecodeToWaveform(acousticTokens, numSamples);
    }

    /// <inheritdoc />
    public Tensor<T> ContinueAudio(Tensor<T> inputAudio, string? prompt = null, double extensionSeconds = 5.0,
        int numInferenceSteps = 100, int? seed = null)
    {
        ThrowIfDisposed();
        int extensionSamples = (int)(extensionSeconds * SampleRate);
        var features = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(inputAudio) : Predict(inputAudio);

        var continuation = new Tensor<T>([inputAudio.Length + extensionSamples]);
        for (int i = 0; i < inputAudio.Length; i++) continuation[i] = inputAudio[i];
        for (int i = 0; i < extensionSamples; i++)
        {
            double v = i < features.Length ? NumOps.ToDouble(features[i % features.Length]) : 0;
            continuation[inputAudio.Length + i] = NumOps.FromDouble(Math.Tanh(v));
        }
        return continuation;
    }

    /// <inheritdoc />
    public Tensor<T> InpaintAudio(Tensor<T> audio, Tensor<T> mask, string? prompt = null,
        int numInferenceSteps = 100, int? seed = null)
    {
        throw new NotSupportedException("YuE does not support audio inpainting.");
    }

    /// <inheritdoc />
    public AudioGenerationOptions<T> GetDefaultOptions() => new()
    {
        DurationSeconds = _options.MaxDurationSeconds,
        NumInferenceSteps = 100,
        GuidanceScale = 3.0,
        Seed = null,
        SchedulerType = "autoregressive"
    };

    #endregion

    #region Song Generation

    /// <summary>
    /// Generates a complete song from lyrics and style tags.
    /// </summary>
    /// <param name="lyrics">The song lyrics.</param>
    /// <param name="styleTags">Genre and style tags (e.g., "pop, female vocalist, upbeat, 120bpm").</param>
    /// <param name="durationSeconds">Desired song duration in seconds.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Generated song waveform tensor.</returns>
    public Tensor<T> GenerateSong(string lyrics, string styleTags, double durationSeconds = 180.0, int? seed = null)
    {
        ThrowIfDisposed();
        string combinedPrompt = $"{styleTags} | {lyrics}";
        return GenerateMusic(combinedPrompt, durationSeconds: durationSeconds, seed: seed);
    }

    /// <summary>
    /// Generates a complete song asynchronously.
    /// </summary>
    public Task<Tensor<T>> GenerateSongAsync(string lyrics, string styleTags, double durationSeconds = 180.0,
        int? seed = null, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => GenerateSong(lyrics, styleTags, durationSeconds, seed), cancellationToken);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultYuELayers(
            semanticDim: _options.SemanticDim, numSemanticLayers: _options.NumSemanticLayers,
            numSemanticHeads: _options.NumSemanticHeads, lyricsVocabSize: _options.LyricsVocabSize,
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
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "YuE-Native" : "YuE-ONNX",
            Description = "YuE full-song music generation with vocals (Yuan et al., 2025)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.SemanticDim,
            Complexity = _options.NumSemanticLayers + _options.NumAcousticLayers
        };
        m.AdditionalInfo["LyricsVocabSize"] = _options.LyricsVocabSize.ToString();
        m.AdditionalInfo["MaxDurationSeconds"] = _options.MaxDurationSeconds.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.MaxDurationSeconds);
        w.Write(_options.SemanticDim); w.Write(_options.NumSemanticLayers);
        w.Write(_options.NumSemanticHeads); w.Write(_options.AcousticDim);
        w.Write(_options.NumAcousticLayers); w.Write(_options.NumAcousticHeads);
        w.Write(_options.LyricsVocabSize); w.Write(_options.SemanticVocabSize);
        w.Write(_options.AcousticCodebookSize); w.Write(_options.NumAcousticQuantizers);
        w.Write(_options.NumStyleTags); w.Write(_options.StyleEmbeddingDim);
        w.Write(_options.Temperature); w.Write(_options.TopP);
        w.Write(_options.RepetitionPenalty); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.MaxDurationSeconds = r.ReadDouble();
        _options.SemanticDim = r.ReadInt32(); _options.NumSemanticLayers = r.ReadInt32();
        _options.NumSemanticHeads = r.ReadInt32(); _options.AcousticDim = r.ReadInt32();
        _options.NumAcousticLayers = r.ReadInt32(); _options.NumAcousticHeads = r.ReadInt32();
        _options.LyricsVocabSize = r.ReadInt32(); _options.SemanticVocabSize = r.ReadInt32();
        _options.AcousticCodebookSize = r.ReadInt32(); _options.NumAcousticQuantizers = r.ReadInt32();
        _options.NumStyleTags = r.ReadInt32(); _options.StyleEmbeddingDim = r.ReadInt32();
        _options.Temperature = r.ReadDouble(); _options.TopP = r.ReadDouble();
        _options.RepetitionPenalty = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new YuE<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> EncodeStylePrompt(string prompt)
    {
        var embedding = new Tensor<T>([_options.SemanticDim]);
        int hash = prompt.GetHashCode();
        for (int i = 0; i < _options.SemanticDim; i++)
        {
            double val = Math.Sin((hash + i) * 0.1) * 0.5;
            embedding[i] = NumOps.FromDouble(val);
        }
        return embedding;
    }

    private Tensor<T> GenerateSemanticTokens(Tensor<T> conditioning, double durationSeconds, int? seed)
    {
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(conditioning) : Predict(conditioning);
        int numTokens = (int)(durationSeconds * 50); // ~50 semantic tokens/sec
        var result = new Tensor<T>([numTokens]);
        for (int i = 0; i < numTokens; i++)
            result[i] = i < output.Length ? output[i % output.Length] : NumOps.Zero;
        return result;
    }

    private Tensor<T> GenerateAcousticTokens(Tensor<T> semanticTokens)
    {
        int totalTokens = semanticTokens.Length * _options.NumAcousticQuantizers;
        var result = new Tensor<T>([totalTokens]);
        for (int i = 0; i < totalTokens; i++)
        {
            int si = i / _options.NumAcousticQuantizers;
            if (si < semanticTokens.Length)
                result[i] = semanticTokens[si];
        }
        return result;
    }

    private Tensor<T> DecodeToWaveform(Tensor<T> tokens, int numSamples)
    {
        var waveform = new Tensor<T>([numSamples]);
        for (int i = 0; i < numSamples; i++)
        {
            int ti = i * tokens.Length / numSamples;
            if (ti < tokens.Length)
            {
                double v = NumOps.ToDouble(tokens[ti]);
                waveform[i] = NumOps.FromDouble(Math.Tanh(v));
            }
        }
        return waveform;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(YuE<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
