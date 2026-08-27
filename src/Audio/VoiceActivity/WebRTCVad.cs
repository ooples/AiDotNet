using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.VoiceActivity;

/// <summary>
/// Neural WebRTC VAD model for low-latency voice activity detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// WebRTC VAD is a lightweight voice activity detection model inspired by the GMM-based
/// detector in the WebRTC framework but reimplemented as a neural network for improved accuracy.
/// It operates at very low latency (10-30ms frames) and is designed for real-time communication.
/// </para>
/// <para>
/// <b>For Beginners:</b> WebRTC VAD is a very fast "is someone talking?" detector used in
/// video calls and voice chat. It processes tiny chunks of audio (10-30 milliseconds) and
/// instantly decides if speech is present.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 480, outputSize: 1);
/// var model = new WebRTCVad&lt;float&gt;(arch, "webrtc_vad.onnx");
/// bool isSpeech = model.DetectSpeech(audioFrame);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("WebRTC Voice Activity Detector", "https://webrtc.googlesource.com/src/+/refs/heads/main/common_audio/vad/")]
public partial class WebRTCVad<T> : AudioNeuralNetworkBase<T>, IVoiceActivityDetector<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Measured from the output construction: <c>PredictCore</c> folds the whole <c>Layers</c> chain and
    /// <c>PostprocessOutput</c> is the identity, so the width is the final layer's output dimension.
    /// <c>LayerHelper.CreateDefaultWebRTCVadLayers</c> ends with
    /// <c>new FullyConnectedLayer&lt;T&gt;(1, identity)</c>, commented "Binary classification output".
    /// A structural 1 - <c>FrameSize</c> is the input width and <c>HiddenDim</c> an interior one.
    /// </remarks>
    protected override int OutputFeatureWidth => 1;

    #region Fields

    private readonly WebRTCVadOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
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
    /// Creates a WebRTC VAD model in ONNX inference mode.
    /// </summary>
    public WebRTCVad(NeuralNetworkArchitecture<T> architecture, string modelPath, WebRTCVadOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new WebRTCVadOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        Threshold = _options.Threshold;
        MinSpeechDurationMs = _options.MinSpeechDurationMs;
        MinSilenceDurationMs = _options.MinSilenceDurationMs;
        InitializeLayers();
    }

    /// <summary>
    /// Creates a WebRTC VAD model in native training mode.
    /// </summary>
    public WebRTCVad(NeuralNetworkArchitecture<T> architecture, WebRTCVadOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new WebRTCVadOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        Threshold = _options.Threshold;
        MinSpeechDurationMs = _options.MinSpeechDurationMs;
        MinSilenceDurationMs = _options.MinSilenceDurationMs;
        InitializeLayers();
    }

    internal static async Task<WebRTCVad<T>> CreateAsync(WebRTCVadOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new WebRTCVadOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("webrtc_vad", "webrtc_vad.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        int frameSize = options.SampleRate * options.FrameDurationMs / 1000;
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: frameSize, outputSize: 1);
        return new WebRTCVad<T>(arch, mp, options);
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
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audioFrame) : Predict(audioFrame);
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
        else Layers.AddRange(LayerHelper<T>.CreateDefaultWebRTCVadLayers(
            frameSize: FrameSize, hiddenDim: _options.HiddenDim,
            numLayers: _options.NumLayers, dropoutRate: _options.DropoutRate));
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "WebRTCVad-Native" : "WebRTCVad-ONNX",
            Description = "Neural WebRTC VAD for low-latency speech detection",
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["FrameDurationMs"] = _options.FrameDurationMs.ToString();
        m.AdditionalInfo["AggressivenessMode"] = _options.AggressivenessMode.ToString();
        return m;
    }





    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(WebRTCVad<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
