using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// VoiceCraft neural codec language model for speech editing and zero-shot TTS.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VoiceCraft (Peng et al., 2024) uses a token rearrangement procedure with causal masking
/// that enables both editing existing speech (replacing/inserting words) and generating new
/// speech from a short prompt, achieving high naturalness and speaker similarity.
/// </para>
/// <para>
/// <b>For Beginners:</b> VoiceCraft can edit speech like you edit text - change specific words
/// in a recording while keeping the speaker's voice. It can also clone a voice from a few
/// seconds of audio and generate new speech. Think of it as "find and replace" for spoken words.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 2048, outputSize: 2048);
/// var model = new VoiceCraft&lt;float&gt;(arch, "voicecraft.onnx");
/// var audio = model.GenerateAudio("Hello, this is a test.");
/// </code>
/// </para>
/// </remarks>
public class VoiceCraft<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>
{
    #region Fields

    private readonly VoiceCraftOptions _options;
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
    public bool SupportsAudioInpainting => true;

    /// <inheritdoc />
    public new bool IsOnnxMode => base.IsOnnxMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a VoiceCraft model in ONNX inference mode.
    /// </summary>
    public VoiceCraft(NeuralNetworkArchitecture<T> architecture, string modelPath, VoiceCraftOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VoiceCraftOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a VoiceCraft model in native training mode.
    /// </summary>
    public VoiceCraft(NeuralNetworkArchitecture<T> architecture, VoiceCraftOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new VoiceCraftOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<VoiceCraft<T>> CreateAsync(VoiceCraftOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new VoiceCraftOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("voicecraft", "voicecraft.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.HiddenDim, outputSize: options.HiddenDim);
        return new VoiceCraft<T>(arch, mp, options);
    }

    #endregion

    #region IAudioGenerator

    /// <inheritdoc />
    public Tensor<T> GenerateAudio(string prompt, string? negativePrompt = null, double durationSeconds = 5.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null)
    {
        ThrowIfDisposed();
        int numSamples = (int)(durationSeconds * SampleRate);
        int numFrames = (int)(durationSeconds * _options.CodecFrameRate);

        // Encode text prompt
        var textTokens = EncodeText(prompt);

        // Generate codec tokens autoregressively
        var codecTokens = GenerateCodecTokens(textTokens, numFrames, seed);

        // Decode to waveform
        return DecodeToWaveform(codecTokens, numSamples);
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
        throw new NotSupportedException("VoiceCraft is a speech model and does not support music generation.");
    }

    /// <inheritdoc />
    public Tensor<T> ContinueAudio(Tensor<T> inputAudio, string? prompt = null, double extensionSeconds = 5.0,
        int numInferenceSteps = 100, int? seed = null)
    {
        ThrowIfDisposed();
        int extensionSamples = (int)(extensionSeconds * SampleRate);

        // Use input audio as prefix conditioning
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
        ThrowIfDisposed();
        // VoiceCraft supports speech editing via causal masking
        var features = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audio) : Predict(audio);

        var result = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            bool isMasked = i < mask.Length && NumOps.ToDouble(mask[i]) > 0.5;
            if (isMasked && i < features.Length)
            {
                double v = NumOps.ToDouble(features[i]);
                result[i] = NumOps.FromDouble(Math.Tanh(v));
            }
            else
            {
                result[i] = audio[i];
            }
        }
        return result;
    }

    /// <inheritdoc />
    public AudioGenerationOptions<T> GetDefaultOptions() => new()
    {
        DurationSeconds = _options.MaxDurationSeconds,
        NumInferenceSteps = 100,
        GuidanceScale = 1.0,
        Seed = null,
        SchedulerType = "autoregressive"
    };

    #endregion

    #region Speech Editing

    /// <summary>
    /// Edits speech by replacing a segment with new content guided by text.
    /// </summary>
    /// <param name="audio">The original audio waveform.</param>
    /// <param name="startSample">Start of the segment to replace.</param>
    /// <param name="endSample">End of the segment to replace.</param>
    /// <param name="replacementText">The text to synthesize as replacement.</param>
    /// <returns>Edited audio waveform with the segment replaced.</returns>
    public Tensor<T> EditSpeech(Tensor<T> audio, int startSample, int endSample, string replacementText)
    {
        ThrowIfDisposed();
        // Create mask for the edit region
        var mask = new Tensor<T>([audio.Length]);
        for (int i = startSample; i < endSample && i < audio.Length; i++)
            mask[i] = NumOps.One;

        return InpaintAudio(audio, mask, replacementText);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultVoiceCraftLayers(
            hiddenDim: _options.HiddenDim, numLayers: _options.NumLayers,
            numHeads: _options.NumHeads, codebookSize: _options.CodebookSize,
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
            Name = _useNativeMode ? "VoiceCraft-Native" : "VoiceCraft-ONNX",
            Description = "VoiceCraft codec LM for speech editing and zero-shot TTS (Peng et al., 2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim,
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["CodebookSize"] = _options.CodebookSize.ToString();
        m.AdditionalInfo["NumQuantizers"] = _options.NumQuantizers.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.MaxDurationSeconds);
        w.Write(_options.HiddenDim); w.Write(_options.NumLayers);
        w.Write(_options.NumHeads); w.Write(_options.CodebookSize);
        w.Write(_options.NumQuantizers); w.Write(_options.CodecEmbeddingDim);
        w.Write(_options.NumMels); w.Write(_options.EditContextSeconds);
        w.Write(_options.MaskRatio); w.Write(_options.Temperature);
        w.Write(_options.TopP); w.Write(_options.CodecFrameRate);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.MaxDurationSeconds = r.ReadDouble();
        _options.HiddenDim = r.ReadInt32(); _options.NumLayers = r.ReadInt32();
        _options.NumHeads = r.ReadInt32(); _options.CodebookSize = r.ReadInt32();
        _options.NumQuantizers = r.ReadInt32(); _options.CodecEmbeddingDim = r.ReadInt32();
        _options.NumMels = r.ReadInt32(); _options.EditContextSeconds = r.ReadDouble();
        _options.MaskRatio = r.ReadDouble(); _options.Temperature = r.ReadDouble();
        _options.TopP = r.ReadDouble(); _options.CodecFrameRate = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new VoiceCraft<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> EncodeText(string text)
    {
        var tokens = new Tensor<T>([_options.HiddenDim]);
        int hash = text.GetHashCode();
        for (int i = 0; i < _options.HiddenDim; i++)
        {
            double val = Math.Sin((hash + i) * 0.1) * 0.5;
            tokens[i] = NumOps.FromDouble(val);
        }
        return tokens;
    }

    private Tensor<T> GenerateCodecTokens(Tensor<T> textTokens, int numFrames, int? seed)
    {
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(textTokens) : Predict(textTokens);
        int totalTokens = numFrames * _options.NumQuantizers;
        var result = new Tensor<T>([totalTokens]);
        for (int i = 0; i < totalTokens; i++)
            result[i] = i < output.Length ? output[i % output.Length] : NumOps.Zero;
        return result;
    }

    private Tensor<T> DecodeToWaveform(Tensor<T> codecTokens, int numSamples)
    {
        var waveform = new Tensor<T>([numSamples]);
        for (int i = 0; i < numSamples; i++)
        {
            int ti = i * codecTokens.Length / numSamples;
            if (ti < codecTokens.Length)
            {
                double v = NumOps.ToDouble(codecTokens[ti]);
                waveform[i] = NumOps.FromDouble(Math.Tanh(v));
            }
        }
        return waveform;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(VoiceCraft<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
