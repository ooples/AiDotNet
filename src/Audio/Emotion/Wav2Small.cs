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
/// Wav2Small lightweight speech emotion recognition model (Gomez-Alanis et al., 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Wav2Small distills knowledge from large wav2vec 2.0 models into a compact architecture
/// suitable for edge deployment. It achieves competitive accuracy on IEMOCAP and RAVDESS
/// benchmarks while requiring significantly fewer parameters and computation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Wav2Small is a small, fast emotion detector that learned its skills
/// from a much larger model. It can tell if someone sounds happy, sad, angry, or neutral—even
/// on a phone or embedded device that can't run large AI models.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 7);
/// var model = new Wav2Small&lt;float&gt;(arch, "wav2small.onnx");
/// var result = model.RecognizeEmotion(speechAudio);
/// Console.WriteLine($"Emotion: {result.Emotion}, Confidence: {result.Confidence}");
/// </code>
/// </para>
/// </remarks>
public class Wav2Small<T> : AudioClassifierBase<T>, IEmotionRecognizer<T>
{
    #region Fields

    private readonly Wav2SmallOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IEmotionRecognizer Properties

    public IReadOnlyList<string> SupportedEmotions => _options.EmotionLabels;

    #endregion

    #region Constructors

    public Wav2Small(NeuralNetworkArchitecture<T> architecture, string modelPath, Wav2SmallOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new Wav2SmallOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    public Wav2Small(NeuralNetworkArchitecture<T> architecture, Wav2SmallOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new Wav2SmallOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<Wav2Small<T>> CreateAsync(Wav2SmallOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new Wav2SmallOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("wav2small", "wav2small.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.NumClasses);
        return new Wav2Small<T>(arch, mp, options);
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
            Arousal = _options.IncludeArousalValence ? GetArousal(audio) : default,
            Valence = _options.IncludeArousalValence ? GetValence(audio) : default
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
            double val = NumOps.ToDouble(logits[i]);
            expValues[i] = Math.Exp(val);
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
        var probs = GetEmotionProbabilities(audio);
        double arousal = 0;
        if (probs.TryGetValue("angry", out var angry)) arousal += NumOps.ToDouble(angry) * 0.8;
        if (probs.TryGetValue("happy", out var happy)) arousal += NumOps.ToDouble(happy) * 0.6;
        if (probs.TryGetValue("fearful", out var fear)) arousal += NumOps.ToDouble(fear) * 0.5;
        if (probs.TryGetValue("surprised", out var surprised)) arousal += NumOps.ToDouble(surprised) * 0.7;
        if (probs.TryGetValue("sad", out var sad)) arousal -= NumOps.ToDouble(sad) * 0.4;
        if (probs.TryGetValue("neutral", out var neutral)) arousal -= NumOps.ToDouble(neutral) * 0.2;
        return NumOps.FromDouble(Math.Max(-1, Math.Min(1, arousal)));
    }

    public T GetValence(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var probs = GetEmotionProbabilities(audio);
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
        else Layers.AddRange(LayerHelper<T>.CreateDefaultWav2SmallLayers(
            numMels: _options.NumMels, hiddenDim: _options.HiddenDim,
            numLayers: _options.NumLayers, numAttentionHeads: _options.NumAttentionHeads,
            feedForwardDim: _options.FeedForwardDim, featureEncoderDim: _options.FeatureEncoderDim,
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
        _optimizer.UpdateParameters(Layers); SetTrainingMode(false);
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
            Name = _useNativeMode ? "Wav2Small-Native" : "Wav2Small-ONNX",
            Description = "Wav2Small Lightweight SER (Gomez-Alanis et al., 2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.NumLayers
        };
        m.AdditionalInfo["NumClasses"] = _options.NumClasses.ToString();
        m.AdditionalInfo["HiddenDim"] = _options.HiddenDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.HiddenDim);
        w.Write(_options.NumLayers); w.Write(_options.NumAttentionHeads);
        w.Write(_options.NumClasses); w.Write(_options.DropoutRate);
        w.Write(_options.EmotionLabels.Length); foreach (var l in _options.EmotionLabels) w.Write(l);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.HiddenDim = r.ReadInt32();
        _options.NumLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32();
        _options.NumClasses = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        int n = r.ReadInt32(); _options.EmotionLabels = new string[n]; for (int i = 0; i < n; i++) _options.EmotionLabels[i] = r.ReadString();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new Wav2Small<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Wav2Small<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
