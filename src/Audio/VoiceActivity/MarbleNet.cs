using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.VoiceActivity;

/// <summary>
/// MarbleNet lightweight 1D separable convolutional VAD model (NVIDIA NeMo).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MarbleNet (Jia et al., 2021, NVIDIA NeMo) is a lightweight 1D time-channel separable
/// convolutional model for voice activity detection. It uses depth-wise separable convolutions
/// with sub-word modeling to achieve state-of-the-art accuracy while being fast enough for
/// real-time streaming on edge devices.
/// </para>
/// <para>
/// <b>For Beginners:</b> MarbleNet is NVIDIA's efficient voice activity detector. It uses a
/// special neural network layer (separable convolutions) that makes it very fast while still
/// being accurate. Think of it as a "speech or not?" classifier that can run in real-time
/// even on a phone or small device.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 64, outputSize: 1);
/// var model = new MarbleNet&lt;float&gt;(arch, "marblenet.onnx");
/// bool isSpeech = model.DetectSpeech(audioFrame);
/// </code>
/// </para>
/// </remarks>
public class MarbleNet<T> : AudioNeuralNetworkBase<T>, IVoiceActivityDetector<T>
{
    #region Fields

    private readonly MarbleNetOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _speechFrameCount;
    private int _silenceFrameCount;

    #endregion

    #region IVoiceActivityDetector Properties

    /// <inheritdoc />
    public int FrameSize => _options.SampleRate * _options.FrameDurationMs / 1000;

    /// <inheritdoc />
    public double Threshold { get; set; }

    /// <inheritdoc />
    public int MinSpeechDurationMs { get; set; }

    /// <inheritdoc />
    public int MinSilenceDurationMs { get; set; }

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a MarbleNet model in ONNX inference mode.
    /// </summary>
    public MarbleNet(NeuralNetworkArchitecture<T> architecture, string modelPath, MarbleNetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MarbleNetOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        Threshold = _options.Threshold;
        MinSpeechDurationMs = _options.MinSpeechDurationMs;
        MinSilenceDurationMs = _options.MinSilenceDurationMs;
        InitializeLayers();
    }

    /// <summary>
    /// Creates a MarbleNet model in native training mode.
    /// </summary>
    public MarbleNet(NeuralNetworkArchitecture<T> architecture, MarbleNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MarbleNetOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        Threshold = _options.Threshold;
        MinSpeechDurationMs = _options.MinSpeechDurationMs;
        MinSilenceDurationMs = _options.MinSilenceDurationMs;
        InitializeLayers();
    }

    internal static async Task<MarbleNet<T>> CreateAsync(MarbleNetOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new MarbleNetOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("marblenet", "marblenet.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: 1);
        return new MarbleNet<T>(arch, mp, options);
    }

    #endregion

    #region IVoiceActivityDetector

    /// <inheritdoc />
    public bool DetectSpeech(Tensor<T> audioFrame)
    {
        ThrowIfDisposed();
        var prob = GetSpeechProbability(audioFrame);
        return NumOps.ToDouble(prob) >= Threshold;
    }

    /// <inheritdoc />
    public T GetSpeechProbability(Tensor<T> audioFrame)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audioFrame);
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        double v = output.Length > 0 ? NumOps.ToDouble(output[0]) : 0;
        double prob = 1.0 / (1.0 + Math.Exp(-v));
        return NumOps.FromDouble(prob);
    }

    /// <inheritdoc />
    public IReadOnlyList<(int StartSample, int EndSample)> DetectSpeechSegments(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var segments = new List<(int, int)>();
        int minSpeechFrames = Math.Max(1, MinSpeechDurationMs * SampleRate / (1000 * FrameSize));
        int minSilenceFrames = Math.Max(1, MinSilenceDurationMs * SampleRate / (1000 * FrameSize));

        int? segmentStart = null;
        int speechCount = 0, silenceCount = 0;
        bool inSpeech = false;

        ResetVadState();
        for (int i = 0; i + FrameSize <= audio.Length; i += FrameSize)
        {
            var frame = new Tensor<T>([FrameSize]);
            for (int j = 0; j < FrameSize; j++) frame[j] = audio[i + j];
            bool isSpeech = DetectSpeech(frame);

            if (isSpeech)
            {
                speechCount++; silenceCount = 0;
                if (!inSpeech && speechCount >= minSpeechFrames)
                {
                    inSpeech = true;
                    segmentStart = i - (speechCount - 1) * FrameSize;
                }
            }
            else
            {
                silenceCount++; speechCount = 0;
                if (inSpeech && silenceCount >= minSilenceFrames)
                {
                    inSpeech = false;
                    if (segmentStart.HasValue) { segments.Add((segmentStart.Value, i - (silenceCount - 1) * FrameSize)); segmentStart = null; }
                }
            }
        }
        if (inSpeech && segmentStart.HasValue) segments.Add((segmentStart.Value, audio.Length));
        return segments;
    }

    /// <inheritdoc />
    public T[] GetFrameProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        int numFrames = audio.Length / FrameSize;
        var probs = new T[numFrames];
        ResetVadState();
        for (int i = 0; i < numFrames; i++)
        {
            var frame = new Tensor<T>([FrameSize]);
            for (int j = 0; j < FrameSize; j++) frame[j] = audio[i * FrameSize + j];
            probs[i] = GetSpeechProbability(frame);
        }
        return probs;
    }

    /// <inheritdoc />
    public (bool IsSpeech, T Probability) ProcessChunk(Tensor<T> audioChunk)
    {
        ThrowIfDisposed();
        var prob = GetSpeechProbability(audioChunk);
        bool isSpeech = NumOps.ToDouble(prob) >= Threshold;
        if (isSpeech) { _speechFrameCount++; _silenceFrameCount = 0; }
        else { _silenceFrameCount++; _speechFrameCount = 0; }
        return (isSpeech, prob);
    }

    /// <inheritdoc />
    void IVoiceActivityDetector<T>.ResetState() => ResetVadState();

    private void ResetVadState()
    {
        _speechFrameCount = 0;
        _silenceFrameCount = 0;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultMarbleNetLayers(
            numMels: _options.NumMels, initialFilters: _options.InitialFilters,
            numBlocks: _options.NumBlocks, subBlocksPerBlock: _options.SubBlocksPerBlock,
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
            Name = _useNativeMode ? "MarbleNet-Native" : "MarbleNet-ONNX",
            Description = "MarbleNet separable convolutional VAD (Jia et al., 2021, NVIDIA NeMo)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumBlocks * _options.SubBlocksPerBlock
        };
        m.AdditionalInfo["NumBlocks"] = _options.NumBlocks.ToString();
        m.AdditionalInfo["InitialFilters"] = _options.InitialFilters.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.FftSize);
        w.Write(_options.HopLength); w.Write(_options.FrameDurationMs);
        w.Write(_options.InitialFilters); w.Write(_options.NumBlocks);
        w.Write(_options.SubBlocksPerBlock); w.Write(_options.KernelSize);
        w.Write(Threshold); w.Write(MinSpeechDurationMs); w.Write(MinSilenceDurationMs);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.FftSize = r.ReadInt32();
        _options.HopLength = r.ReadInt32(); _options.FrameDurationMs = r.ReadInt32();
        _options.InitialFilters = r.ReadInt32(); _options.NumBlocks = r.ReadInt32();
        _options.SubBlocksPerBlock = r.ReadInt32(); _options.KernelSize = r.ReadInt32();
        Threshold = r.ReadDouble(); MinSpeechDurationMs = r.ReadInt32(); MinSilenceDurationMs = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new MarbleNet<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MarbleNet<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
