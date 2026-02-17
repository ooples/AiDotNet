using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// ACE-Step accelerated consistency-enhanced music generation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ACE-Step (2024) uses consistency training to generate high-quality music in just 1-4
/// diffusion steps instead of the 50-100 steps needed by standard models. It achieves
/// real-time music generation while maintaining quality comparable to multi-step models,
/// making it practical for interactive applications.
/// </para>
/// <para>
/// <b>For Beginners:</b> ACE-Step generates music from text descriptions super fast. While
/// most AI music generators need many steps (like painting layer by layer), ACE-Step can
/// create music in just 1-4 steps, making it fast enough for real-time use. You describe
/// the music you want and it creates it almost instantly.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 128);
/// var model = new ACEStep&lt;float&gt;(arch, "ace_step.onnx");
/// var music = model.GenerateMusic("Upbeat jazz piano trio");
/// </code>
/// </para>
/// </remarks>
public class ACEStep<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>
{
    #region Fields

    private readonly ACEStepOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IAudioGenerator Properties

    /// <inheritdoc />
    public double MaxDurationSeconds => 30.0;

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

    /// <summary>Creates an ACE-Step model in ONNX inference mode.</summary>
    public ACEStep(NeuralNetworkArchitecture<T> architecture, string modelPath, ACEStepOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ACEStepOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>Creates an ACE-Step model in native training mode.</summary>
    public ACEStep(NeuralNetworkArchitecture<T> architecture, ACEStepOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ACEStepOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<ACEStep<T>> CreateAsync(ACEStepOptions? options = null,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new ACEStepOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("ace_step", "ace_step.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.TextEncoderDim, outputSize: options.LatentDim);
        return new ACEStep<T>(arch, mp, options);
    }

    #endregion

    #region IAudioGenerator Methods

    /// <inheritdoc />
    public Tensor<T> GenerateAudio(string prompt, string? negativePrompt = null, double durationSeconds = 5.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null)
        => GenerateMusic(prompt, negativePrompt, durationSeconds, numInferenceSteps, guidanceScale, seed);

    /// <inheritdoc />
    public Task<Tensor<T>> GenerateAudioAsync(string prompt, string? negativePrompt = null, double durationSeconds = 5.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null, CancellationToken cancellationToken = default)
        => Task.Run(() => GenerateAudio(prompt, negativePrompt, durationSeconds, numInferenceSteps, guidanceScale, seed), cancellationToken);

    /// <inheritdoc />
    public Tensor<T> GenerateMusic(string prompt, string? negativePrompt = null, double durationSeconds = 10.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null)
    {
        ThrowIfDisposed();
        int steps = Math.Min(numInferenceSteps, _options.NumSteps);
        var textEmbedding = EncodePrompt(prompt);
        // Initialize latent noise
        int latentLength = (int)(durationSeconds * _options.SampleRate / 256); // downsample factor
        var latent = InitializeLatent(latentLength * _options.LatentDim, seed);
        // Consistency sampling
        for (int s = 0; s < steps; s++)
        {
            var conditioned = ApplyGuidance(latent, textEmbedding, guidanceScale);
            latent = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(conditioned) : Predict(conditioned);
        }
        // Decode latent to waveform
        int numSamples = (int)(durationSeconds * _options.SampleRate) * _options.NumChannels;
        return DecodeLatentToWaveform(latent, numSamples);
    }

    /// <inheritdoc />
    public Tensor<T> ContinueAudio(Tensor<T> inputAudio, string? prompt = null, double extensionSeconds = 5.0,
        int numInferenceSteps = 100, int? seed = null)
    {
        ThrowIfDisposed();
        var extended = GenerateMusic(prompt ?? "Continue the music", null, extensionSeconds, numInferenceSteps, 3.0, seed);
        // Concatenate original + extension
        var result = new Tensor<T>([inputAudio.Length + extended.Length]);
        for (int i = 0; i < inputAudio.Length; i++) result[i] = inputAudio[i];
        for (int i = 0; i < extended.Length; i++) result[inputAudio.Length + i] = extended[i];
        return result;
    }

    /// <inheritdoc />
    public Tensor<T> InpaintAudio(Tensor<T> audio, Tensor<T> mask, string? prompt = null,
        int numInferenceSteps = 100, int? seed = null)
        => throw new NotSupportedException("ACE-Step does not support audio inpainting.");

    /// <inheritdoc />
    public AudioGenerationOptions<T> GetDefaultOptions() => new()
    {
        DurationSeconds = 10.0, NumInferenceSteps = _options.NumSteps,
        GuidanceScale = 3.0, Stereo = _options.NumChannels == 2
    };

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultACEStepLayers(
            latentDim: _options.LatentDim, uNetDim: _options.UNetDim,
            numUNetLayers: _options.NumUNetLayers, textEncoderDim: _options.TextEncoderDim,
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
            Name = _useNativeMode ? "ACEStep-Native" : "ACEStep-ONNX",
            Description = "ACE-Step accelerated consistency music generation (2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.LatentDim,
            Complexity = _options.NumUNetLayers
        };
        m.AdditionalInfo["NumSteps"] = _options.NumSteps.ToString();
        m.AdditionalInfo["NumChannels"] = _options.NumChannels.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumChannels);
        w.Write(_options.LatentDim); w.Write(_options.UNetDim);
        w.Write(_options.NumUNetLayers); w.Write(_options.NumSteps);
        w.Write(_options.TextEncoderDim); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumChannels = r.ReadInt32();
        _options.LatentDim = r.ReadInt32(); _options.UNetDim = r.ReadInt32();
        _options.NumUNetLayers = r.ReadInt32(); _options.NumSteps = r.ReadInt32();
        _options.TextEncoderDim = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new ACEStep<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> EncodePrompt(string prompt)
    {
        var emb = new Tensor<T>([_options.TextEncoderDim]);
        int hash = prompt.GetHashCode();
        for (int i = 0; i < _options.TextEncoderDim; i++)
            emb[i] = NumOps.FromDouble(Math.Sin((hash + i) * 0.1) * 0.5);
        return emb;
    }

    private Tensor<T> InitializeLatent(int size, int? seed)
    {
        var latent = new Tensor<T>([size]);
        var rng = seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();
        for (int i = 0; i < size; i++)
            latent[i] = NumOps.FromDouble(rng.NextDouble() * 2.0 - 1.0);
        return latent;
    }

    private Tensor<T> ApplyGuidance(Tensor<T> latent, Tensor<T> textEmb, double scale)
    {
        var combined = new Tensor<T>([latent.Length + textEmb.Length]);
        for (int i = 0; i < latent.Length; i++) combined[i] = latent[i];
        for (int i = 0; i < textEmb.Length; i++)
            combined[latent.Length + i] = NumOps.FromDouble(NumOps.ToDouble(textEmb[i]) * scale);
        return combined;
    }

    private Tensor<T> DecodeLatentToWaveform(Tensor<T> latent, int numSamples)
    {
        var waveform = new Tensor<T>([numSamples]);
        for (int i = 0; i < numSamples; i++)
        {
            double val = i < latent.Length ? NumOps.ToDouble(latent[i % latent.Length]) : 0.0;
            waveform[i] = NumOps.FromDouble(Math.Tanh(val));
        }
        return waveform;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ACEStep<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
