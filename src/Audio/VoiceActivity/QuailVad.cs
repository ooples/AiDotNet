using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.VoiceActivity;

/// <summary>
/// Quail VAD - lightweight voice activity detection optimized for on-device deployment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Quail VAD (2024) is a compact CNN-RNN voice activity detector designed for edge devices.
/// It uses knowledge distillation from larger teacher models to maintain high accuracy while
/// keeping the model small enough for mobile phones and embedded systems. The model processes
/// short audio frames and outputs speech probabilities with minimal latency.
/// </para>
/// <para>
/// <b>For Beginners:</b> Quail VAD is a lightweight "is someone talking?" detector that runs
/// efficiently on phones and small devices. Despite being much smaller than models like
/// Silero VAD, it achieves competitive accuracy by learning from larger, more powerful models
/// during training.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 64, outputSize: 1);
/// var model = new QuailVad&lt;float&gt;(arch, "quail_vad.onnx");
/// var (isSpeech, probability) = model.ProcessChunk(audioFrame);
/// if (isSpeech) Console.WriteLine($"Speech detected! {probability}");
/// </code>
/// </para>
/// </remarks>
public class QuailVad<T> : AudioNeuralNetworkBase<T>, IVoiceActivityDetector<T>
{
    #region Fields

    private readonly QuailVadOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private List<T>? _streamingBuffer;
    private T _lastProbability;

    #endregion

    #region Constructors

    /// <summary>Creates a Quail VAD model in ONNX inference mode.</summary>
    public QuailVad(NeuralNetworkArchitecture<T> architecture, string modelPath, QuailVadOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new QuailVadOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _lastProbability = NumOps.Zero;
        InitializeLayers();
    }

    /// <summary>Creates a Quail VAD model in native training mode.</summary>
    public QuailVad(NeuralNetworkArchitecture<T> architecture, QuailVadOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new QuailVadOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        _lastProbability = NumOps.Zero;
        InitializeLayers();
    }

    internal static async Task<QuailVad<T>> CreateAsync(QuailVadOptions? options = null,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new QuailVadOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("quail_vad", "quail_vad.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: _options_HiddenDim(options), outputSize: 1);
        return new QuailVad<T>(arch, mp, options);
    }

    private static int _options_HiddenDim(QuailVadOptions options) => options.HiddenDim;

    #endregion

    #region IVoiceActivityDetector Properties

    /// <inheritdoc />
    public new int SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int FrameSize => _options.SampleRate * _options.FrameSizeMs / 1000;

    /// <inheritdoc />
    public double Threshold { get; set; }

    /// <inheritdoc />
    public int MinSpeechDurationMs { get; set; }

    /// <inheritdoc />
    public int MinSilenceDurationMs { get; set; }

    #endregion

    #region IVoiceActivityDetector Methods

    /// <inheritdoc />
    public bool DetectSpeech(Tensor<T> audioFrame)
    {
        var prob = GetSpeechProbability(audioFrame);
        return NumOps.ToDouble(prob) >= Threshold;
    }

    /// <inheritdoc />
    public T GetSpeechProbability(Tensor<T> audioFrame)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audioFrame);
        Tensor<T> output;
        if (IsOnnxMode && OnnxEncoder is not null) output = OnnxEncoder.Run(features);
        else output = Predict(features);
        // Sigmoid to probability
        double raw = output.Length > 0 ? NumOps.ToDouble(output[0]) : 0;
        double prob = 1.0 / (1.0 + Math.Exp(-raw));
        _lastProbability = NumOps.FromDouble(prob);
        return _lastProbability;
    }

    /// <inheritdoc />
    public IReadOnlyList<(int StartSample, int EndSample)> DetectSpeechSegments(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var probs = GetFrameProbabilities(audio);
        var segments = new List<(int, int)>();
        int frameSamples = FrameSize;
        int minSpeechFrames = Math.Max(1, (int)(MinSpeechDurationMs * _options.SampleRate / 1000.0 / frameSamples));
        int minSilenceFrames = Math.Max(1, (int)(MinSilenceDurationMs * _options.SampleRate / 1000.0 / frameSamples));

        int speechStart = -1;
        int silenceCount = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            bool isSpeech = NumOps.ToDouble(probs[i]) >= Threshold;
            if (isSpeech)
            {
                if (speechStart < 0) speechStart = i;
                silenceCount = 0;
            }
            else
            {
                silenceCount++;
                if (speechStart >= 0 && silenceCount >= minSilenceFrames)
                {
                    int speechFrames = i - silenceCount - speechStart + 1;
                    if (speechFrames >= minSpeechFrames)
                        segments.Add((speechStart * frameSamples, (i - silenceCount + 1) * frameSamples));
                    speechStart = -1;
                }
            }
        }
        if (speechStart >= 0)
        {
            int speechFrames = probs.Length - speechStart;
            if (speechFrames >= minSpeechFrames)
                segments.Add((speechStart * frameSamples, Math.Min(probs.Length * frameSamples, audio.Length)));
        }
        return segments;
    }

    /// <inheritdoc />
    public T[] GetFrameProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        int frameSamples = FrameSize;
        int numFrames = audio.Length / frameSamples;
        if (numFrames <= 0) numFrames = 1;
        var probs = new T[numFrames];
        for (int f = 0; f < numFrames; f++)
        {
            var frame = new Tensor<T>([frameSamples]);
            int start = f * frameSamples;
            for (int i = 0; i < frameSamples && start + i < audio.Length; i++)
                frame[i] = audio[start + i];
            probs[f] = GetSpeechProbability(frame);
        }
        return probs;
    }

    /// <inheritdoc />
    public (bool IsSpeech, T Probability) ProcessChunk(Tensor<T> audioChunk)
    {
        ThrowIfDisposed();
        _streamingBuffer ??= [];
        for (int i = 0; i < audioChunk.Length; i++) _streamingBuffer.Add(audioChunk[i]);
        int frameSamples = FrameSize;
        if (_streamingBuffer.Count < frameSamples)
            return (false, NumOps.Zero);
        var frame = new Tensor<T>([frameSamples]);
        for (int i = 0; i < frameSamples; i++) frame[i] = _streamingBuffer[i];
        _streamingBuffer.RemoveRange(0, frameSamples);
        var prob = GetSpeechProbability(frame);
        bool isSpeech = NumOps.ToDouble(prob) >= Threshold;
        return (isSpeech, prob);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        base.ResetState();
        _streamingBuffer = null;
        _lastProbability = NumOps.Zero;
    }

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        Threshold = _options.Threshold;
        MinSpeechDurationMs = (int)(_options.MinSpeechDuration * 1000);
        MinSilenceDurationMs = (int)(_options.MinSilenceDuration * 1000);
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultQuailVadLayers(
            hiddenDim: _options.HiddenDim, numCNNLayers: _options.NumCNNLayers,
            rnnHiddenSize: _options.RNNHiddenSize, frameSizeMs: _options.FrameSizeMs,
            sampleRate: _options.SampleRate));
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
            Name = _useNativeMode ? "QuailVAD-Native" : "QuailVAD-ONNX",
            Description = "Quail VAD lightweight on-device voice activity detection (2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim, Complexity = _options.NumCNNLayers
        };
        m.AdditionalInfo["FrameSizeMs"] = _options.FrameSizeMs.ToString();
        m.AdditionalInfo["RNNHiddenSize"] = _options.RNNHiddenSize.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.HiddenDim);
        w.Write(_options.NumCNNLayers); w.Write(_options.RNNHiddenSize);
        w.Write(_options.FrameSizeMs); w.Write(_options.Threshold);
        w.Write(_options.MinSpeechDuration); w.Write(_options.MinSilenceDuration);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.HiddenDim = r.ReadInt32();
        _options.NumCNNLayers = r.ReadInt32(); _options.RNNHiddenSize = r.ReadInt32();
        _options.FrameSizeMs = r.ReadInt32(); _options.Threshold = r.ReadDouble();
        _options.MinSpeechDuration = r.ReadDouble(); _options.MinSilenceDuration = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        Threshold = _options.Threshold;
        MinSpeechDurationMs = (int)(_options.MinSpeechDuration * 1000);
        MinSilenceDurationMs = (int)(_options.MinSilenceDuration * 1000);
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new QuailVad<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(QuailVad<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
