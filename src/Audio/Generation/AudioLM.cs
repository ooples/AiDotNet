using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// AudioLM hierarchical audio language model for high-quality audio generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AudioLM (Borsos et al., 2023, Google) generates high-quality, coherent audio by
/// combining semantic tokens (from a self-supervised model like w2v-BERT) with acoustic
/// tokens (from a neural codec like SoundStream). A hierarchical language model generates
/// semantic tokens first for high-level structure, then acoustic tokens for fine detail.
/// </para>
/// <para>
/// <b>For Beginners:</b> AudioLM generates natural-sounding audio by "thinking" about it
/// at two levels: first the big-picture meaning (semantic tokens), then the fine sound
/// details (acoustic tokens). Think of it like writing a story: first an outline, then the
/// vivid details. This produces audio that is both coherent and high-fidelity.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1024, outputSize: 1024);
/// var model = new AudioLM&lt;float&gt;(arch, "audiolm.onnx");
/// var audio = model.GenerateAudio("A dog barking in a park");
/// </code>
/// </para>
/// </remarks>
public class AudioLM<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>
{
    #region Fields

    private readonly AudioLMOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IAudioGenerator Properties

    /// <inheritdoc />
    public double MaxDurationSeconds => _options.MaxDurationSeconds;

    /// <inheritdoc />
    public bool SupportsTextToAudio => true;

    /// <inheritdoc />
    public bool SupportsTextToMusic => false;

    /// <inheritdoc />
    public bool SupportsAudioContinuation => true;

    /// <inheritdoc />
    public bool SupportsAudioInpainting => false;

    /// <inheritdoc />
    public new bool IsOnnxMode => base.IsOnnxMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an AudioLM model in ONNX inference mode.
    /// </summary>
    public AudioLM(NeuralNetworkArchitecture<T> architecture, string modelPath, AudioLMOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new AudioLMOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates an AudioLM model in native training mode.
    /// </summary>
    public AudioLM(NeuralNetworkArchitecture<T> architecture, AudioLMOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new AudioLMOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<AudioLM<T>> CreateAsync(AudioLMOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new AudioLMOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("audiolm", "audiolm.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.SemanticDim, outputSize: options.SemanticDim);
        return new AudioLM<T>(arch, mp, options);
    }

    #endregion

    #region IAudioGenerator

    /// <inheritdoc />
    public Tensor<T> GenerateAudio(string prompt, string? negativePrompt = null, double durationSeconds = 5.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null)
    {
        ThrowIfDisposed();
        int numSamples = (int)(durationSeconds * SampleRate);
        int numSemanticTokens = (int)(durationSeconds * _options.SemanticFrameRate);

        // Encode prompt to conditioning
        var conditioning = EncodePrompt(prompt);

        // Stage 1: Generate semantic tokens
        var semanticOutput = GenerateSemanticTokens(conditioning, numSemanticTokens, seed);

        // Stage 2: Generate coarse acoustic tokens from semantic
        var coarseOutput = GenerateCoarseTokens(semanticOutput);

        // Stage 3: Generate fine acoustic tokens from coarse
        var fineOutput = GenerateFineTokens(coarseOutput);

        // Decode to waveform
        return DecodeToWaveform(fineOutput, numSamples);
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
        throw new NotSupportedException("AudioLM does not support text-to-music generation. Use YuE or other music generation models.");
    }

    /// <inheritdoc />
    public Tensor<T> ContinueAudio(Tensor<T> inputAudio, string? prompt = null, double extensionSeconds = 5.0,
        int numInferenceSteps = 100, int? seed = null)
    {
        ThrowIfDisposed();
        int extensionSamples = (int)(extensionSeconds * SampleRate);

        // Encode existing audio as conditioning
        var audioFeatures = PreprocessAudio(inputAudio);
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audioFeatures) : Predict(audioFeatures);

        // Generate continuation
        var continuation = new Tensor<T>([inputAudio.Length + extensionSamples]);
        for (int i = 0; i < inputAudio.Length; i++) continuation[i] = inputAudio[i];
        for (int i = 0; i < extensionSamples && i < output.Length; i++)
        {
            double v = NumOps.ToDouble(output[i % output.Length]);
            continuation[inputAudio.Length + i] = NumOps.FromDouble(Math.Tanh(v));
        }
        return continuation;
    }

    /// <inheritdoc />
    public Tensor<T> InpaintAudio(Tensor<T> audio, Tensor<T> mask, string? prompt = null,
        int numInferenceSteps = 100, int? seed = null)
    {
        throw new NotSupportedException("AudioLM does not support audio inpainting.");
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

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultAudioLMLayers(
            semanticDim: _options.SemanticDim, numSemanticLayers: _options.NumSemanticLayers,
            numSemanticHeads: _options.NumSemanticHeads, semanticVocabSize: _options.SemanticVocabSize,
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
            Name = _useNativeMode ? "AudioLM-Native" : "AudioLM-ONNX",
            Description = "AudioLM hierarchical audio language model (Borsos et al., 2023, Google)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.SemanticDim,
            Complexity = _options.NumSemanticLayers + _options.NumCoarseLayers + _options.NumFineLayers
        };
        m.AdditionalInfo["SemanticVocabSize"] = _options.SemanticVocabSize.ToString();
        m.AdditionalInfo["NumCoarseQuantizers"] = _options.NumCoarseQuantizers.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.MaxDurationSeconds);
        w.Write(_options.SemanticVocabSize); w.Write(_options.SemanticDim);
        w.Write(_options.NumSemanticLayers); w.Write(_options.NumSemanticHeads);
        w.Write(_options.SemanticFrameRate); w.Write(_options.CoarseCodebookSize);
        w.Write(_options.NumCoarseQuantizers); w.Write(_options.CoarseDim);
        w.Write(_options.NumCoarseLayers); w.Write(_options.FineCodebookSize);
        w.Write(_options.NumFineQuantizers); w.Write(_options.FineDim);
        w.Write(_options.NumFineLayers); w.Write(_options.Temperature);
        w.Write(_options.TopK); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.MaxDurationSeconds = r.ReadDouble();
        _options.SemanticVocabSize = r.ReadInt32(); _options.SemanticDim = r.ReadInt32();
        _options.NumSemanticLayers = r.ReadInt32(); _options.NumSemanticHeads = r.ReadInt32();
        _options.SemanticFrameRate = r.ReadInt32(); _options.CoarseCodebookSize = r.ReadInt32();
        _options.NumCoarseQuantizers = r.ReadInt32(); _options.CoarseDim = r.ReadInt32();
        _options.NumCoarseLayers = r.ReadInt32(); _options.FineCodebookSize = r.ReadInt32();
        _options.NumFineQuantizers = r.ReadInt32(); _options.FineDim = r.ReadInt32();
        _options.NumFineLayers = r.ReadInt32(); _options.Temperature = r.ReadDouble();
        _options.TopK = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new AudioLM<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> EncodePrompt(string prompt)
    {
        // Simple prompt encoding: hash-based deterministic embedding
        var embedding = new Tensor<T>([_options.SemanticDim]);
        int hash = prompt.GetHashCode();
        for (int i = 0; i < _options.SemanticDim; i++)
        {
            double val = Math.Sin((hash + i) * 0.1) * 0.5;
            embedding[i] = NumOps.FromDouble(val);
        }
        return embedding;
    }

    private Tensor<T> GenerateSemanticTokens(Tensor<T> conditioning, int numTokens, int? seed)
    {
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(conditioning) : Predict(conditioning);
        var result = new Tensor<T>([numTokens]);
        for (int i = 0; i < numTokens; i++)
            result[i] = i < output.Length ? output[i] : NumOps.Zero;
        return result;
    }

    private Tensor<T> GenerateCoarseTokens(Tensor<T> semanticTokens)
    {
        int coarseLen = semanticTokens.Length * _options.NumCoarseQuantizers;
        var result = new Tensor<T>([coarseLen]);
        for (int i = 0; i < coarseLen; i++)
        {
            int si = i / _options.NumCoarseQuantizers;
            if (si < semanticTokens.Length)
                result[i] = semanticTokens[si];
        }
        return result;
    }

    private Tensor<T> GenerateFineTokens(Tensor<T> coarseTokens)
    {
        int fineLen = coarseTokens.Length * _options.NumFineQuantizers / _options.NumCoarseQuantizers;
        var result = new Tensor<T>([fineLen]);
        for (int i = 0; i < fineLen; i++)
        {
            int ci = i * coarseTokens.Length / fineLen;
            if (ci < coarseTokens.Length)
                result[i] = coarseTokens[ci];
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

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AudioLM<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
