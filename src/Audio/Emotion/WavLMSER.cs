using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Audio.Classification;
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
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
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
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new WavLMSEROptions();
        _useNativeMode = true;
        ValidateOptions(_options);
        // Train raw logits with the classifier objective and honor the public optimization options.
        // A caller-supplied optimizer remains the full-customization path for alternate schedules.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
            });
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    private static void ValidateOptions(WavLMSEROptions options)
    {
        if (options.HiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(options.HiddenDim));
        if (options.NumLayers <= 0) throw new ArgumentOutOfRangeException(nameof(options.NumLayers));
        if (options.NumAttentionHeads <= 0) throw new ArgumentOutOfRangeException(nameof(options.NumAttentionHeads));
        if (options.HiddenDim % options.NumAttentionHeads != 0)
            throw new ArgumentException("HiddenDim must be divisible by NumAttentionHeads.", nameof(options));
        if (options.FeedForwardDim <= 0) throw new ArgumentOutOfRangeException(nameof(options.FeedForwardDim));
        if (options.FeatureEncoderDim <= 0) throw new ArgumentOutOfRangeException(nameof(options.FeatureEncoderDim));
        if (options.NumClasses <= 0) throw new ArgumentOutOfRangeException(nameof(options.NumClasses));
        if (options.EmotionLabels is null || options.EmotionLabels.Length != options.NumClasses)
            throw new ArgumentException("EmotionLabels must contain exactly NumClasses labels.", nameof(options));
        if (options.LearningRate <= 0 || double.IsNaN(options.LearningRate) || double.IsInfinity(options.LearningRate))
            throw new ArgumentOutOfRangeException(nameof(options.LearningRate));
        if (options.WeightDecay < 0 || double.IsNaN(options.WeightDecay) || double.IsInfinity(options.WeightDecay))
            throw new ArgumentOutOfRangeException(nameof(options.WeightDecay));
        if (options.DropoutRate < 0 || options.DropoutRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(options.DropoutRate));
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
        // Route BOTH modes through Predict so the returned vector has one consistent contract: the
        // emotion-class distribution PredictCore/PostprocessOutput produce (native runs the softmax head;
        // the ONNX branch applies PostprocessOutput to OnnxEncoder.Run). The old code special-cased ONNX
        // to return OnnxEncoder.Run directly — bypassing PostprocessOutput — so this method silently
        // returned probabilities for native models but raw ONNX logits for ONNX models. Mirrors the
        // sibling GetEmotionProbabilities, which likewise reads Predict(features) for both modes.
        Tensor<T> output = Predict(features);
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
        // Public inference returns a probability distribution; training uses RunLayersRaw directly so
        // CrossEntropyWithLogitsLoss applies its stable softmax exactly once.
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            return PostprocessOutput(RunLayersRaw(input));
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    /// <summary>Runs the WavLM encoder and emotion head without inference softmax.</summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => RunLayersRaw(input);

    private Tensor<T> RunLayersRaw(Tensor<T> input)
    {
        var output = input;
        foreach (var layer in Layers) output = layer.Forward(output);

        // WavLM downstream classification pools frame representations into one utterance-level
        // prediction. Reduce every leading axis while preserving the final class axis.
        if (output.Shape.Length >= 2)
        {
            int classAxis = output.Shape.Length - 1;
            int[] poolAxes = Enumerable.Range(0, classAxis).ToArray();
            output = Engine.ReduceMean(output, poolAxes, keepDims: false);
        }

        return output;
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

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => Engine.Softmax(o, axis: -1);

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
        var copiedOptions = new WavLMSEROptions(_options);
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new WavLMSER<T>(Architecture, mp, copiedOptions);
        return new WavLMSER<T>(Architecture, copiedOptions);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(WavLMSER<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
