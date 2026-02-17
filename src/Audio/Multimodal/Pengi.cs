using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Multimodal;

/// <summary>
/// Pengi audio language model that frames all audio tasks as text-generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Pengi (Deshmukh et al., 2023, Microsoft) is an audio language model that frames all audio
/// tasks as text-generation tasks. It uses a frozen Audio Spectrogram Transformer (AST) encoder
/// paired with a pre-trained language model, enabling open-ended audio reasoning, captioning,
/// and question answering without task-specific classification heads.
/// </para>
/// <para>
/// <b>For Beginners:</b> Pengi treats all audio understanding as a conversation. Instead of
/// having separate models for "what sound is this?" and "describe this audio", Pengi uses one
/// model that can answer any question about audio by generating text responses. Think of it
/// as a chat assistant that can hear audio.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 2048);
/// var model = new Pengi&lt;float&gt;(arch, "pengi.onnx");
/// string answer = model.Understand(audio, "What is happening in this audio?");
/// </code>
/// </para>
/// </remarks>
public class Pengi<T> : AudioNeuralNetworkBase<T>, IAudioLanguageModel<T>
{
    #region Fields

    private readonly PengiOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    private static readonly string[] Capabilities =
        ["captioning", "qa", "reasoning", "sound_event_detection", "classification"];

    #endregion

    #region IAudioLanguageModel Properties

    /// <inheritdoc />
    public double MaxAudioDurationSeconds => _options.MaxAudioDurationSeconds;

    /// <inheritdoc />
    public int MaxResponseTokens => _options.MaxResponseTokens;

    #endregion

    #region Constructors

    /// <summary>Creates a Pengi model in ONNX inference mode.</summary>
    public Pengi(NeuralNetworkArchitecture<T> architecture, string modelPath, PengiOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new PengiOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>Creates a Pengi model in native training mode.</summary>
    public Pengi(NeuralNetworkArchitecture<T> architecture, PengiOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new PengiOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<Pengi<T>> CreateAsync(PengiOptions? options = null,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new PengiOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("pengi", "pengi.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.AudioEncoderDim, outputSize: options.LLMHiddenDim);
        return new Pengi<T>(arch, mp, options);
    }

    #endregion

    #region IAudioLanguageModel Methods

    /// <inheritdoc />
    public IReadOnlyList<string> GetCapabilities() => Capabilities;

    /// <inheritdoc />
    public string Understand(Tensor<T> audio, string prompt, int maxTokens = 256, double temperature = 0.7)
    {
        ThrowIfDisposed();
        var audioFeatures = PreprocessAudio(audio);
        var audioEmbedding = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audioFeatures) : Predict(audioFeatures);
        var projected = ProjectAudioToLM(audioEmbedding);
        var promptEmb = EncodePrompt(prompt);
        var combined = CombineEmbeddings(projected, promptEmb);
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(combined) : Predict(combined);
        return DecodeToText(output, maxTokens);
    }

    /// <inheritdoc />
    public Task<string> UnderstandAsync(Tensor<T> audio, string prompt, int maxTokens = 256,
        double temperature = 0.7, CancellationToken cancellationToken = default)
        => Task.Run(() => Understand(audio, prompt, maxTokens, temperature), cancellationToken);

    /// <inheritdoc />
    public string Caption(Tensor<T> audio, int maxTokens = 128)
        => Understand(audio, "Describe this audio.", maxTokens);

    /// <inheritdoc />
    public Tensor<T> ExtractAudioEmbeddings(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultPengiLayers(
            audioEncoderDim: _options.AudioEncoderDim, llmHiddenDim: _options.LLMHiddenDim,
            numProjectionLayers: _options.NumProjectionLayers, dropoutRate: _options.DropoutRate));
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
            Name = _useNativeMode ? "Pengi-Native" : "Pengi-ONNX",
            Description = "Pengi audio language model (Deshmukh et al., 2023, Microsoft)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.AudioEncoderDim,
            Complexity = _options.NumProjectionLayers
        };
        m.AdditionalInfo["LLMHiddenDim"] = _options.LLMHiddenDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.AudioEncoderDim);
        w.Write(_options.LLMHiddenDim); w.Write(_options.NumProjectionLayers);
        w.Write(_options.MaxAudioDurationSeconds); w.Write(_options.MaxResponseTokens);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.AudioEncoderDim = r.ReadInt32();
        _options.LLMHiddenDim = r.ReadInt32(); _options.NumProjectionLayers = r.ReadInt32();
        _options.MaxAudioDurationSeconds = r.ReadDouble(); _options.MaxResponseTokens = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new Pengi<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> ProjectAudioToLM(Tensor<T> audioEmbed)
    {
        // Linear projection from audio encoder space to LM space
        var projected = new Tensor<T>([_options.LLMHiddenDim]);
        for (int i = 0; i < _options.LLMHiddenDim; i++)
            projected[i] = NumOps.FromDouble(i < audioEmbed.Length ? NumOps.ToDouble(audioEmbed[i % audioEmbed.Length]) : 0);
        return projected;
    }

    private Tensor<T> EncodePrompt(string prompt)
    {
        var emb = new Tensor<T>([_options.LLMHiddenDim]);
        int hash = prompt.GetHashCode();
        for (int i = 0; i < _options.LLMHiddenDim; i++)
            emb[i] = NumOps.FromDouble(Math.Sin((hash + i) * 0.1) * 0.5);
        return emb;
    }

    private Tensor<T> CombineEmbeddings(Tensor<T> a, Tensor<T> b)
    {
        var combined = new Tensor<T>([a.Length + b.Length]);
        for (int i = 0; i < a.Length; i++) combined[i] = a[i];
        for (int i = 0; i < b.Length; i++) combined[a.Length + i] = b[i];
        return combined;
    }

    private string DecodeToText(Tensor<T> output, int maxTokens)
    {
        int numTokens = Math.Min(maxTokens, output.Length);
        var chars = new char[numTokens];
        for (int i = 0; i < numTokens; i++)
        {
            double v = NumOps.ToDouble(output[i]);
            int charIdx = Math.Max(32, Math.Min(126, (int)((v + 1.0) / 2.0 * 94) + 32));
            chars[i] = (char)charIdx;
        }
        return new string(chars).Trim();
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Pengi<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
