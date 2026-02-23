using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Audio.Classification;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Emotion;

/// <summary>
/// HuBERT-SER (HuBERT for Speech Emotion Recognition) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// HuBERT-SER fine-tunes the HuBERT (Hsu et al., 2021) self-supervised speech model for
/// emotion recognition. HuBERT learns speech representations through masked prediction
/// of discrete speech units, achieving 69.7% weighted accuracy on IEMOCAP when fine-tuned.
/// </para>
/// <para>
/// <b>For Beginners:</b> HuBERT-SER uses a model that first learned to understand speech
/// patterns from millions of hours of audio (HuBERT), then was specialized to detect
/// emotions in voice. It combines deep speech understanding with emotion classification.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 7);
/// var model = new HuBERTSER&lt;float&gt;(arch, "hubert_ser.onnx");
/// var result = model.RecognizeEmotion(speechAudio);
/// Console.WriteLine($"Emotion: {result.Emotion}, Confidence: {result.Confidence}");
/// </code>
/// </para>
/// </remarks>
public class HuBERTSER<T> : AudioClassifierBase<T>, IEmotionRecognizer<T>
{
    #region Fields

    private readonly HuBERTSEROptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IEmotionRecognizer Properties

    public IReadOnlyList<string> SupportedEmotions => _options.EmotionLabels;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a HuBERT-SER model in ONNX inference mode.
    /// </summary>
    public HuBERTSER(NeuralNetworkArchitecture<T> architecture, string modelPath, HuBERTSEROptions? options = null)
        : base(architecture)
    {
        _options = options ?? new HuBERTSEROptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a HuBERT-SER model in native training mode.
    /// </summary>
    public HuBERTSER(NeuralNetworkArchitecture<T> architecture, HuBERTSEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new HuBERTSEROptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<HuBERTSER<T>> CreateAsync(HuBERTSEROptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new HuBERTSEROptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("hubert_ser", "hubert_ser.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.NumClasses);
        return new HuBERTSER<T>(arch, mp, options);
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
        string? secondary = sorted.Count > 1 && NumOps.ToDouble(sorted[1].Value) > 0.1 ? sorted[1].Key : null;

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
        Tensor<T> logits = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        var probs = new Dictionary<string, T>();
        double sumExp = 0;
        var expValues = new double[_options.NumClasses];
        for (int i = 0; i < _options.NumClasses && i < logits.Length; i++)
        {
            expValues[i] = Math.Exp(NumOps.ToDouble(logits[i]));
            sumExp += expValues[i];
        }
        for (int i = 0; i < _options.NumClasses && i < _options.EmotionLabels.Length; i++)
            probs[_options.EmotionLabels[i]] = NumOps.FromDouble(sumExp > 0 ? expValues[i] / sumExp : 1.0 / _options.NumClasses);

        return probs;
    }

    public IReadOnlyList<TimedEmotionResult<T>> RecognizeEmotionTimeSeries(Tensor<T> audio, int windowSizeMs = 1000, int hopSizeMs = 500)
    {
        ThrowIfDisposed();
        var results = new List<TimedEmotionResult<T>>();
        int windowSamples = _options.SampleRate * windowSizeMs / 1000;
        int hopSamples = _options.SampleRate * hopSizeMs / 1000;

        for (int start = 0; start + windowSamples <= audio.Length; start += hopSamples)
        {
            var chunk = new Tensor<T>([windowSamples]);
            for (int i = 0; i < windowSamples; i++) chunk[i] = audio[start + i];
            var result = RecognizeEmotion(chunk);
            results.Add(new TimedEmotionResult<T>
            {
                Emotion = result.Emotion, Confidence = result.Confidence,
                SecondaryEmotion = result.SecondaryEmotion,
                Arousal = result.Arousal, Valence = result.Valence,
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

    private T ComputeArousalFromProbs(IReadOnlyDictionary<string, T> probs)
    {
        double arousal = 0;
        if (probs.TryGetValue("angry", out var angry)) arousal += NumOps.ToDouble(angry) * 0.8;
        if (probs.TryGetValue("happy", out var happy)) arousal += NumOps.ToDouble(happy) * 0.6;
        if (probs.TryGetValue("fearful", out var fear)) arousal += NumOps.ToDouble(fear) * 0.5;
        if (probs.TryGetValue("surprised", out var surprised)) arousal += NumOps.ToDouble(surprised) * 0.7;
        if (probs.TryGetValue("sad", out var sad)) arousal -= NumOps.ToDouble(sad) * 0.4;
        if (probs.TryGetValue("neutral", out var neutral)) arousal -= NumOps.ToDouble(neutral) * 0.2;
        return NumOps.FromDouble(Math.Max(-1, Math.Min(1, arousal)));
    }

    private T ComputeValenceFromProbs(IReadOnlyDictionary<string, T> probs)
    {
        double valence = 0;
        if (probs.TryGetValue("happy", out var happy)) valence += NumOps.ToDouble(happy) * 0.9;
        if (probs.TryGetValue("surprised", out var surprised)) valence += NumOps.ToDouble(surprised) * 0.3;
        if (probs.TryGetValue("angry", out var angry)) valence -= NumOps.ToDouble(angry) * 0.7;
        if (probs.TryGetValue("sad", out var sad)) valence -= NumOps.ToDouble(sad) * 0.8;
        if (probs.TryGetValue("fearful", out var fear)) valence -= NumOps.ToDouble(fear) * 0.6;
        if (probs.TryGetValue("disgusted", out var disgusted)) valence -= NumOps.ToDouble(disgusted) * 0.7;
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
        else Layers.AddRange(LayerHelper<T>.CreateDefaultHuBERTSERLayers(
            numMels: _options.NumMels, transformerDim: _options.TransformerDim,
            numTransformerLayers: _options.NumTransformerLayers, numAttentionHeads: _options.NumAttentionHeads,
            feedForwardDim: _options.FeedForwardDim, classifierHiddenDim: _options.ClassifierHiddenDim,
            numClasses: _options.NumClasses, dropoutRate: _options.DropoutRate));
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
            Name = _useNativeMode ? "HuBERT-SER-Native" : "HuBERT-SER-ONNX",
            Description = $"HuBERT-{_options.Variant} Speech Emotion Recognition (Hsu et al., 2021)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumTransformerLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["NumClasses"] = _options.NumClasses.ToString();
        m.AdditionalInfo["ClassifierHiddenDim"] = _options.ClassifierHiddenDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.TransformerDim);
        w.Write(_options.NumTransformerLayers); w.Write(_options.NumAttentionHeads);
        w.Write(_options.FeedForwardDim); w.Write(_options.ClassifierHiddenDim); w.Write(_options.NumClasses); w.Write(_options.DropoutRate);
        w.Write(_options.Variant);
        w.Write(_options.EmotionLabels.Length); foreach (var l in _options.EmotionLabels) w.Write(l);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.TransformerDim = r.ReadInt32();
        _options.NumTransformerLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32();
        _options.FeedForwardDim = r.ReadInt32(); _options.ClassifierHiddenDim = r.ReadInt32(); _options.NumClasses = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        _options.Variant = r.ReadString();
        int n = r.ReadInt32(); _options.EmotionLabels = new string[n]; for (int i = 0; i < n; i++) _options.EmotionLabels[i] = r.ReadString();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new HuBERTSER<T>(Architecture, mp, _options);
        return new HuBERTSER<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(HuBERTSER<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
