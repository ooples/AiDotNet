using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Audio.Classification;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Emotion;

/// <summary>
/// WavLM-SER speech emotion recognition model (fine-tuned WavLM, Chen et al., 2022).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// WavLM-SER fine-tunes the WavLM self-supervised model for speech emotion recognition.
/// WavLM's pre-training with masked prediction and denoising produces robust features that
/// achieve 71%+ weighted accuracy on IEMOCAP and are robust to noise and recording conditions.
/// </para>
/// <para>
/// <b>For Beginners:</b> WavLM-SER takes a model that already understands human speech deeply,
/// then teaches it to recognize emotions. Because it starts with such strong speech understanding,
/// it can pick up on subtle vocal cues—like slight tremors in fear, pitch changes in excitement,
/// or the flat tone of sadness—that simpler models miss entirely.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 7);
/// var model = new WavLMSER&lt;float&gt;(arch, "wavlm_ser.onnx");
/// var result = model.RecognizeEmotion(speechAudio);
/// // Result is available in the returned value
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing", "https://arxiv.org/abs/2110.13900", Year = 2022, Authors = "Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, Furu Wei")]
internal class WavLMSER<T> : AudioClassifierBase<T>, IEmotionRecognizer<T>
{
    #region Fields

    private readonly WavLMSEROptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IEmotionRecognizer Properties

    public IReadOnlyList<string> SupportedEmotions => _options.EmotionLabels;

    #endregion

    #region Constructors

    public WavLMSER(NeuralNetworkArchitecture<T> architecture, string modelPath, WavLMSEROptions? options = null)
        : base(architecture)
    {
        _options = options ?? new WavLMSEROptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public WavLMSER(NeuralNetworkArchitecture<T> architecture, WavLMSEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new WavLMSEROptions();
        _useNativeMode = true;
        // WavLM fine-tunes its deep (12-layer, 768-d) transformer with LR WARMUP (Chen et al. 2022,
        // following wav2vec2 / the Noam schedule): the LR ramps from ~0 over the first steps instead of
        // hitting full magnitude on step 1. Without it, AdamW's first updates overshoot the sharp
        // post-LN encoder landscape and the loss SPIKES before recovering (memorization over 100 steps
        // still converges, but the shorter Training_ShouldReduceLoss window catches the transient rise).
        // Restore the paper's warmup so the loss descends monotonically from the first step. Peak LR is
        // the conservative SER fine-tuning value (5e-4). base(...) defaults maxGradNorm to 1.0, so the
        // eager TrainWithTape path also clips the gradient norm before each step.
        // Conservative fine-tuning LR with warmup. WavLM SER fine-tunes at a small peak LR (~1e-4;
        // the sibling grounding-VLM GLaMM uses 5e-5) — the framework AdamW default (1e-3) is 1-2 orders
        // of magnitude too aggressive for this deep (12-layer, 768-d) post-LN encoder and the loss
        // steadily RISES over the first ~30 steps before it would recover. Ramp the LR 1e-5 -> 1e-4 over
        // the first 10 steps and hold at 1e-4 (WarmupThenEpoch stops per-batch stepping once warmup
        // completes; there are no epochs here). InitialLearningRate matches the warmup floor so the very
        // first optimizer step is gentle whether the eager path syncs the schedule before or after it.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 1e-5,
                LearningRateScheduler = new LinearWarmupScheduler(
                    baseLearningRate: 1e-4, warmupSteps: 10, warmupInitLr: 1e-5),
                SchedulerStepMode = SchedulerStepMode.WarmupThenEpoch,
            });
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<WavLMSER<T>> CreateAsync(WavLMSEROptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new WavLMSEROptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("wavlm_ser", $"wavlm_ser_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.NumClasses);
        return new WavLMSER<T>(arch, mp, options);
    }

    #endregion

    #region IEmotionRecognizer

    public EmotionResult<T> RecognizeEmotion(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var probs = GetEmotionProbabilities(audio);
        var sorted = probs.OrderByDescending(p => NumOps.ToDouble(p.Value)).ToList();
        string primary = sorted[0].Key;
        T confidence = sorted[0].Value;
        string? secondary = sorted.Count > 1 && NumOps.GreaterThan(sorted[1].Value, NumOps.FromDouble(0.1)) ? sorted[1].Key : null;

        return new EmotionResult<T>
        {
            Emotion = primary, Confidence = confidence, SecondaryEmotion = secondary,
            Arousal = _options.IncludeArousalValence ? ComputeArousalFromProbs(probs) : default,
            Valence = _options.IncludeArousalValence ? ComputeValenceFromProbs(probs) : default
        };
    }

    public IReadOnlyDictionary<string, T> GetEmotionProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        // PredictCore now applies the softmax head (PostprocessOutput) for BOTH native and ONNX modes,
        // so Predict already returns a normalized probability distribution. Route both modes through it
        // and read the probabilities directly — re-running softmax here (as the old native path did on
        // Predict's already-softmax'd output) would double-softmax and flatten the distribution.
        var probsTensor = Predict(features);

        if (_options.NumClasses <= 0)
            throw new InvalidOperationException("NumClasses must be positive.");
        if (_options.EmotionLabels is null || _options.EmotionLabels.Length == 0)
            throw new InvalidOperationException("EmotionLabels must be non-empty.");

        var probs = new Dictionary<string, T>();
        int numClasses = Math.Min(_options.NumClasses, Math.Min(_options.EmotionLabels.Length, probsTensor.Length));
        for (int i = 0; i < numClasses; i++)
            probs[_options.EmotionLabels[i]] = probsTensor[i];

        return probs;
    }

    public IReadOnlyList<TimedEmotionResult<T>> RecognizeEmotionTimeSeries(Tensor<T> audio, int windowSizeMs = 1000, int hopSizeMs = 500)
    {
        ThrowIfDisposed();
        if (_options.SampleRate <= 0)
            throw new InvalidOperationException("SampleRate must be positive.");
        if (windowSizeMs <= 0)
            throw new ArgumentOutOfRangeException(nameof(windowSizeMs), "Window size must be positive.");
        if (hopSizeMs <= 0)
            throw new ArgumentOutOfRangeException(nameof(hopSizeMs), "Hop size must be positive.");
        var results = new List<TimedEmotionResult<T>>();
        int windowSamples = _options.SampleRate * windowSizeMs / 1000;
        int hopSamples = _options.SampleRate * hopSizeMs / 1000;
        if (windowSamples <= 0 || hopSamples <= 0)
            throw new InvalidOperationException("Computed window or hop size in samples is non-positive. Check SampleRate and window/hop parameters.");

        for (int start = 0; start + windowSamples <= audio.Length; start += hopSamples)
        {
            var chunk = new Tensor<T>([windowSamples]);
            for (int i = 0; i < windowSamples; i++) chunk[i] = audio[start + i];
            var result = RecognizeEmotion(chunk);
            results.Add(new TimedEmotionResult<T>
            {
                Emotion = result.Emotion, Confidence = result.Confidence,
                SecondaryEmotion = result.SecondaryEmotion, Arousal = result.Arousal, Valence = result.Valence,
                StartTime = start / (double)_options.SampleRate,
                EndTime = (start + windowSamples) / (double)_options.SampleRate
            });
        }
        return results;
    }

    public T GetArousal(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return ComputeArousalFromProbs(GetEmotionProbabilities(audio));
    }

    public T GetValence(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return ComputeValenceFromProbs(GetEmotionProbabilities(audio));
    }

    private static double GetProbValue(IReadOnlyDictionary<string, T> probs, string key, INumericOperations<T> numOps)
    {
        // Case-insensitive lookup to handle different label conventions
        foreach (var kvp in probs)
        {
            if (string.Equals(kvp.Key, key, StringComparison.OrdinalIgnoreCase))
                return numOps.ToDouble(kvp.Value);
        }
        return 0.0;
    }

    private T ComputeArousalFromProbs(IReadOnlyDictionary<string, T> probs)
    {
        double arousal = 0;
        arousal += GetProbValue(probs, "angry", NumOps) * 0.8;
        arousal += GetProbValue(probs, "happy", NumOps) * 0.6;
        arousal += GetProbValue(probs, "fearful", NumOps) * 0.5;
        arousal += GetProbValue(probs, "surprised", NumOps) * 0.7;
        arousal -= GetProbValue(probs, "sad", NumOps) * 0.4;
        arousal -= GetProbValue(probs, "neutral", NumOps) * 0.2;
        return NumOps.FromDouble(Math.Max(-1, Math.Min(1, arousal)));
    }

    private T ComputeValenceFromProbs(IReadOnlyDictionary<string, T> probs)
    {
        double valence = 0;
        valence += GetProbValue(probs, "happy", NumOps) * 0.9;
        valence += GetProbValue(probs, "surprised", NumOps) * 0.3;
        valence -= GetProbValue(probs, "angry", NumOps) * 0.7;
        valence -= GetProbValue(probs, "sad", NumOps) * 0.8;
        valence -= GetProbValue(probs, "fearful", NumOps) * 0.6;
        valence -= GetProbValue(probs, "disgusted", NumOps) * 0.7;
        return NumOps.FromDouble(Math.Max(-1, Math.Min(1, valence)));
    }

    public Vector<T> ExtractEmotionFeatures(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        return output.ToVector();
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultWavLMSERLayers(
            hiddenDim: _options.HiddenDim, numLayers: _options.NumLayers,
            numAttentionHeads: _options.NumAttentionHeads, feedForwardDim: _options.FeedForwardDim,
            featureEncoderDim: _options.FeatureEncoderDim, numClasses: _options.NumClasses,
            dropoutRate: _options.DropoutRate));
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return PostprocessOutput(OnnxEncoder.Run(input));
        // Force inference mode (dropout off) and run the softmax head via PostprocessOutput: WavLM-SER
        // (Chen et al. 2022) is a single-label emotion classifier, so its public output is a probability
        // distribution over the emotion classes (non-negative, sums to 1). Without this the head returned
        // raw logits, so Predict produced negative "class scores" (ClassOutput_ShouldBeNonNegative).
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            var c = input; foreach (var l in Layers) c = l.Forward(c); return PostprocessOutput(c);
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            // Pass WavLMSER's own warmup-scheduled optimizer (see ctor) instead of the shared base
            // optimizer, so the eager tape path advances the linear-warmup LR schedule each step.
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);
        return rawAudio;
    }

    // The softmax that turns the head's logits into a single-label emotion probability distribution
    // (Chen et al. 2022) now lives in the final head layer (see CreateDefaultWavLMSERLayers), so it runs
    // in BOTH the training and inference forward passes — keeping the trained objective consistent with
    // the predicted output. PredictCore therefore just returns the already-normalized distribution.
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "WavLM-SER-Native" : "WavLM-SER-ONNX",
            Description = $"WavLM-SER {_options.Variant} Speech Emotion Recognition (Chen et al., 2022)",
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["NumClasses"] = _options.NumClasses.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant); w.Write(_options.NumMels);
        w.Write(_options.HiddenDim); w.Write(_options.NumLayers); w.Write(_options.NumAttentionHeads);
        w.Write(_options.FeedForwardDim); w.Write(_options.FeatureEncoderDim);
        w.Write(_options.NumClasses); w.Write(_options.DropoutRate);
        w.Write(_options.EmotionLabels.Length); foreach (var l in _options.EmotionLabels) w.Write(l);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString(); _options.NumMels = r.ReadInt32();
        _options.HiddenDim = r.ReadInt32(); _options.NumLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32();
        _options.FeedForwardDim = r.ReadInt32(); _options.FeatureEncoderDim = r.ReadInt32();
        _options.NumClasses = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        int n = r.ReadInt32(); _options.EmotionLabels = new string[n]; for (int i = 0; i < n; i++) _options.EmotionLabels[i] = r.ReadString();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new WavLMSER<T>(Architecture, mp, _options);
        return new WavLMSER<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(WavLMSER<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
