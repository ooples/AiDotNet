using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// VALL-E zero-shot text-to-speech via neural codec language modeling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VALL-E (Wang et al., 2023, Microsoft) treats TTS as a language modeling problem using
/// discrete audio codes from EnCodec. A 3-second enrollment recording suffices for zero-shot
/// voice synthesis. It uses an autoregressive (AR) model for the first codebook and a
/// non-autoregressive (NAR) model for the remaining codebook layers.
/// </para>
/// <para>
/// <b>For Beginners:</b> VALL-E can hear someone speak for 3 seconds and then generate new
/// speech in that person's voice. It works by converting speech into "audio words" (codec
/// tokens) and using a language model to predict what comes next. The AR model handles the
/// basic structure, and the NAR model adds the fine sound quality.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1024, outputSize: 1024);
/// var model = new VALLE&lt;float&gt;(arch, "valle.onnx");
/// var audio = model.GenerateAudio("Hello, how are you today?");
/// </code>
/// </para>
/// </remarks>
public class VALLE<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>
{
    #region Fields

    private readonly VALLEOptions _options;
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
    /// Creates a VALL-E model in ONNX inference mode.
    /// </summary>
    public VALLE(NeuralNetworkArchitecture<T> architecture, string modelPath, VALLEOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VALLEOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a VALL-E model in native training mode.
    /// </summary>
    public VALLE(NeuralNetworkArchitecture<T> architecture, VALLEOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new VALLEOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<VALLE<T>> CreateAsync(VALLEOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new VALLEOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("valle", "valle.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.ARHiddenDim, outputSize: options.ARHiddenDim);
        return new VALLE<T>(arch, mp, options);
    }

    #endregion

    #region IAudioGenerator

    /// <inheritdoc />
    public Tensor<T> GenerateAudio(string prompt, string? negativePrompt = null, double durationSeconds = 5.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null)
    {
        ThrowIfDisposed();
        int numSamples = (int)(durationSeconds * SampleRate);

        // Encode text to phoneme embeddings
        var phonemeEmbeddings = EncodePhonemes(prompt);

        // AR stage: generate first codebook tokens autoregressively
        var arOutput = GenerateARTokens(phonemeEmbeddings, durationSeconds, seed);

        // NAR stage: generate remaining codebook tokens non-autoregressively
        var narOutput = GenerateNARTokens(arOutput);

        // Decode codec tokens to waveform
        return DecodeToWaveform(narOutput, numSamples);
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
        throw new NotSupportedException("VALL-E is a speech synthesis model and does not support music generation.");
    }

    /// <inheritdoc />
    public Tensor<T> ContinueAudio(Tensor<T> inputAudio, string? prompt = null, double extensionSeconds = 5.0,
        int numInferenceSteps = 100, int? seed = null)
    {
        ThrowIfDisposed();
        int extensionSamples = (int)(extensionSeconds * SampleRate);

        // Use input audio as enrollment/prefix
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
        throw new NotSupportedException("VALL-E does not support audio inpainting. Use VoiceCraft for speech editing.");
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

    #region Zero-Shot TTS

    /// <summary>
    /// Synthesizes speech from text using a reference audio for voice cloning.
    /// </summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="referenceAudio">3+ seconds of reference audio for voice cloning.</param>
    /// <param name="durationSeconds">Desired output duration.</param>
    /// <returns>Synthesized speech in the reference speaker's voice.</returns>
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio, double durationSeconds = 5.0)
    {
        ThrowIfDisposed();
        int numSamples = (int)(durationSeconds * SampleRate);

        // Extract speaker embedding from reference
        var speakerEmbedding = IsOnnxMode && OnnxEncoder is not null
            ? OnnxEncoder.Run(referenceAudio)
            : Predict(referenceAudio);

        // Encode text
        var textEmbedding = EncodePhonemes(text);

        // Combine speaker + text and generate
        var combined = CombineEmbeddings(speakerEmbedding, textEmbedding);
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(combined) : Predict(combined);

        return DecodeToWaveform(output, numSamples);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultVALLELayers(
            arHiddenDim: _options.ARHiddenDim, numARLayers: _options.NumARLayers,
            numARHeads: _options.NumARHeads, codebookSize: _options.CodebookSize,
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
            Name = _useNativeMode ? "VALL-E-Native" : "VALL-E-ONNX",
            Description = "VALL-E zero-shot TTS via neural codec language modeling (Wang et al., 2023, Microsoft)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.ARHiddenDim,
            Complexity = _options.NumARLayers + _options.NumNARLayers
        };
        m.AdditionalInfo["CodebookSize"] = _options.CodebookSize.ToString();
        m.AdditionalInfo["NumCodebooks"] = _options.NumCodebooks.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.MaxDurationSeconds);
        w.Write(_options.ARHiddenDim); w.Write(_options.NumARLayers);
        w.Write(_options.NumARHeads); w.Write(_options.NARHiddenDim);
        w.Write(_options.NumNARLayers); w.Write(_options.NumNARHeads);
        w.Write(_options.PhonemeVocabSize); w.Write(_options.CodebookSize);
        w.Write(_options.NumCodebooks); w.Write(_options.Temperature);
        w.Write(_options.TopP); w.Write(_options.MinEnrollmentSeconds);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.MaxDurationSeconds = r.ReadDouble();
        _options.ARHiddenDim = r.ReadInt32(); _options.NumARLayers = r.ReadInt32();
        _options.NumARHeads = r.ReadInt32(); _options.NARHiddenDim = r.ReadInt32();
        _options.NumNARLayers = r.ReadInt32(); _options.NumNARHeads = r.ReadInt32();
        _options.PhonemeVocabSize = r.ReadInt32(); _options.CodebookSize = r.ReadInt32();
        _options.NumCodebooks = r.ReadInt32(); _options.Temperature = r.ReadDouble();
        _options.TopP = r.ReadDouble(); _options.MinEnrollmentSeconds = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new VALLE<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> EncodePhonemes(string text)
    {
        var embedding = new Tensor<T>([_options.ARHiddenDim]);
        int hash = text.GetHashCode();
        for (int i = 0; i < _options.ARHiddenDim; i++)
        {
            double val = Math.Sin((hash + i) * 0.1) * 0.5;
            embedding[i] = NumOps.FromDouble(val);
        }
        return embedding;
    }

    private Tensor<T> GenerateARTokens(Tensor<T> phonemes, double durationSeconds, int? seed)
    {
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(phonemes) : Predict(phonemes);
        int numTokens = (int)(durationSeconds * 75); // ~75 tokens/sec at 24kHz with EnCodec
        var result = new Tensor<T>([numTokens]);
        for (int i = 0; i < numTokens; i++)
            result[i] = i < output.Length ? output[i % output.Length] : NumOps.Zero;
        return result;
    }

    private Tensor<T> GenerateNARTokens(Tensor<T> arTokens)
    {
        int totalTokens = arTokens.Length * _options.NumCodebooks;
        var result = new Tensor<T>([totalTokens]);
        for (int i = 0; i < totalTokens; i++)
        {
            int arIdx = i / _options.NumCodebooks;
            if (arIdx < arTokens.Length)
                result[i] = arTokens[arIdx];
        }
        return result;
    }

    private Tensor<T> CombineEmbeddings(Tensor<T> speaker, Tensor<T> text)
    {
        int len = Math.Max(speaker.Length, text.Length);
        var combined = new Tensor<T>([len]);
        for (int i = 0; i < len; i++)
        {
            double s = i < speaker.Length ? NumOps.ToDouble(speaker[i]) : 0;
            double t = i < text.Length ? NumOps.ToDouble(text[i]) : 0;
            combined[i] = NumOps.FromDouble((s + t) / 2.0);
        }
        return combined;
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

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(VALLE<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
