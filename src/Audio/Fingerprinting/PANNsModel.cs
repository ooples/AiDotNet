using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// PANNs (Pretrained Audio Neural Networks) audio classifier — a CNN14-style
/// convolutional backbone over log-mel spectrograms, trained for AudioSet
/// tagging (Kong et al. 2020).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PANNs CNN14 is the de-facto baseline transfer-learning model for audio
/// classification: four conv stages (64 → 128 → 256 → 512 channels), each
/// stage = Conv(3×3) + BN + ReLU + Conv(3×3) + BN + ReLU + AvgPool(2×2),
/// followed by a global average pool, a 2048-d embedding head, and a 527-
/// class linear classifier with sigmoid activation for multi-label output.
/// Trained on AudioSet (2 M weakly-labelled clips) at 32 kHz / 64-mel.
/// </para>
/// <para>
/// <b>Reference:</b> Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., &amp;
/// Plumbley, M. D. (2020), "PANNs: Large-Scale Pretrained Audio Neural
/// Networks for Audio Pattern Recognition", IEEE/ACM TASLP.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition",
    "https://arxiv.org/abs/1912.10211",
    Year = 2020,
    Authors = "Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley")]
public class PANNsModel<T> : AudioNeuralNetworkBase<T>, IAudioFingerprinter<T>
{
    private readonly PANNsModelOptions _options;
    private readonly bool _useNativeMode;

    /// <summary>
    /// Cached Hann window for the STFT preprocessing step. Built once on the
    /// first <see cref="PreprocessAudio"/> call and reused across batches.
    /// </summary>
    private Tensor<T>? _hannWindow;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <inheritdoc/>
    public string Name => _useNativeMode ? "PANNs-Native" : "PANNs-ONNX";

    /// <inheritdoc/>
    public int FingerprintLength => _options.EmbeddingDim;

    #region Constructors

    /// <summary>Initializes PANNs in ONNX inference mode.</summary>
    public PANNsModel(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        PANNsModelOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new PANNsModelOptions();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path is required for ONNX mode.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);

        SampleRate = _options.SampleRate;
        NumMels = _options.NumMelBands;
        _useNativeMode = false;
        _modelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Path to the loaded ONNX model file when constructed in ONNX mode;
    /// <c>null</c> in native mode. Captured so <see cref="CreateNewInstance"/>
    /// can reconstruct an ONNX clone without losing its execution-mode
    /// configuration.
    /// </summary>
    private readonly string? _modelPath;

    /// <summary>Initializes PANNs in native training / inference mode.</summary>
    public PANNsModel(
        NeuralNetworkArchitecture<T> architecture,
        PANNsModelOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new PANNsModelOptions();
        SampleRate = _options.SampleRate;
        NumMels = _options.NumMelBands;
        _useNativeMode = true;
        InitializeLayers();
    }

    #endregion

    #region Layer Construction (Golden-Standard Pattern)

    /// <summary>
    /// Initializes the CNN14 layer stack following the codebase's golden
    /// single-stream pattern: prefer user-supplied <c>Architecture.Layers</c>;
    /// otherwise fall back to <see cref="LayerHelper{T}.CreateDefaultPANNsLayers"/>.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultPANNsLayers(
                numClasses: _options.NumClasses,
                embeddingDim: _options.EmbeddingDim,
                dropoutRate: _options.DropoutRate));
        }
    }

    #endregion

    #region Preprocessing — Mel Spectrogram

    /// <summary>
    /// Converts raw audio samples into a log-mel spectrogram via the engine's
    /// fused <see cref="IEngine.MelSpectrogram{T}"/> kernel — the PANNs §3 64-
    /// mel × 320-hop pipeline routed through a single BLAS / GPU-eligible op.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (rawAudio.Shape.Length == 1)
            rawAudio = Engine.Reshape(rawAudio, [1, rawAudio.Shape[0]]);

        int batchSize = rawAudio.Shape[0];

        _hannWindow ??= BuildHannWindow(_options.StftWindowSize);

        var mel = Engine.MelSpectrogram(
            input: rawAudio,
            sampleRate: _options.SampleRate,
            nFft: _options.StftWindowSize,
            hopLength: _options.HopLength,
            nMels: _options.NumMelBands,
            fMin: NumOps.Zero,
            fMax: NumOps.FromDouble(_options.SampleRate / 2.0),
            window: _hannWindow,
            powerToDb: true);

        int numFrames = mel.Length / (batchSize * _options.NumMelBands);
        return Engine.Reshape(mel, [batchSize, 1, numFrames, _options.NumMelBands]);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => modelOutput;

    /// <summary>
    /// Builds a periodic Hann window of length <paramref name="windowSize"/>
    /// as a <see cref="Tensor{T}"/>: <c>w[n] = 0.5·(1 − cos(2πn/(N−1)))</c>.
    /// Generic in <typeparamref name="T"/> via the inherited NumOps.
    /// </summary>
    private Tensor<T> BuildHannWindow(int windowSize)
    {
        var window = new Tensor<T>([windowSize]);
        for (int n = 0; n < windowSize; n++)
        {
            double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * n / (windowSize - 1)));
            window[n] = NumOps.FromDouble(w);
        }
        return window;
    }

    #endregion

    #region Public API

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        var mel = PreprocessAudio(input);
        if (!_useNativeMode && OnnxEncoder is not null)
        {
            return PostprocessOutput(OnnxEncoder.Run(mel));
        }
        var hidden = mel;
        foreach (var layer in Layers) hidden = layer.Forward(hidden);
        return hidden;
    }

    /// <summary>
    /// Classifies audio into AudioSet categories. Returns labels with
    /// probability ≥ <paramref name="threshold"/>, sorted descending.
    /// </summary>
    public Dictionary<string, double> Classify(Tensor<T> audio, double threshold = 0.5)
    {
        var probs = Predict(audio);
        int classCount = probs.Shape[^1];
        var results = new Dictionary<string, double>();
        for (int i = 0; i < classCount; i++)
        {
            double p = Convert.ToDouble(probs[0, i]);
            if (p >= threshold)
                results[$"class_{i}"] = p;
        }
        return results.OrderByDescending(kv => kv.Value)
                      .ToDictionary(kv => kv.Key, kv => kv.Value);
    }

    /// <summary>Returns the top-K class predictions sorted descending.</summary>
    public List<(string Label, double Probability)> GetTopK(Tensor<T> audio, int k = 5)
    {
        var probs = Predict(audio);
        int classCount = probs.Shape[^1];
        var indexed = new (double prob, int idx)[classCount];
        for (int i = 0; i < classCount; i++)
            indexed[i] = (Convert.ToDouble(probs[0, i]), i);
        Array.Sort(indexed, (a, b) => b.prob.CompareTo(a.prob));

        var result = new List<(string, double)>(k);
        for (int i = 0; i < Math.Min(k, classCount); i++)
            result.Add(($"class_{indexed[i].idx}", indexed[i].prob));
        return result;
    }

    /// <inheritdoc/>
    public AudioFingerprint<T> Fingerprint(Tensor<T> audio)
    {
        // Layers is empty in ONNX mode (the ONNX graph runs via
        // OnnxEncoder), so the layer-walk below would exit immediately
        // and return the preprocessed mel spectrogram as the fingerprint
        // — silently wrong. Require native mode here until an
        // ONNX-embedding output is wired explicitly.
        if (!_useNativeMode)
        {
            throw new NotSupportedException(
                "PANNsModel.Fingerprint() in ONNX mode requires an explicit " +
                "embedding-producing output to be wired. Use the native " +
                "constructor or invoke the ONNX encoder directly via the " +
                "OnnxEncoder property.");
        }

        // For fingerprinting we want the pooled embedding (output of the
        // ReLU-activated embedding DenseLayer, just before the classifier).
        // Walk the layer stack and stop after the embedding-dim DenseLayer
        // (which has outputSize == _options.EmbeddingDim).
        var mel = PreprocessAudio(audio);
        var hidden = mel;
        foreach (var layer in Layers)
        {
            hidden = layer.Forward(hidden);
            if (layer is DenseLayer<T> dense && dense.GetOutputShape().Length == 1 &&
                dense.GetOutputShape()[0] == _options.EmbeddingDim)
            {
                break;
            }
        }

        var flat = hidden.ToVector();
        var data = new T[flat.Length];
        for (int i = 0; i < flat.Length; i++) data[i] = flat[i];
        return new AudioFingerprint<T>
        {
            Data = data,
            SampleRate = SampleRate,
            // Use the last-axis sample count so batched audio reports the
            // per-clip duration rather than batch * samples / SampleRate.
            Duration = audio.Shape[audio.Shape.Length - 1] / (double)SampleRate
        };
    }

    /// <inheritdoc/>
    public AudioFingerprint<T> Fingerprint(Vector<T> audio) =>
        Fingerprint(Tensor<T>.FromVector(audio));

    /// <inheritdoc/>
    public double ComputeSimilarity(AudioFingerprint<T> fp1, AudioFingerprint<T> fp2)
    {
        if (fp1.Data.Length != fp2.Data.Length)
            throw new ArgumentException("Fingerprint dimensions do not match.");
        double dot = 0.0, n1 = 0.0, n2 = 0.0;
        for (int i = 0; i < fp1.Data.Length; i++)
        {
            double a = Convert.ToDouble(fp1.Data[i]);
            double b = Convert.ToDouble(fp2.Data[i]);
            dot += a * b; n1 += a * a; n2 += b * b;
        }
        return dot / (Math.Sqrt(n1) * Math.Sqrt(n2) + 1e-12);
    }

    /// <inheritdoc/>
    public IReadOnlyList<FingerprintMatch> FindMatches(
        AudioFingerprint<T> query, AudioFingerprint<T> reference, int minMatchLength = 10)
    {
        // PANNs produces a single global embedding per clip rather than a
        // sequence of frame-local fingerprints, so segment-length filtering
        // doesn't apply — there are no shorter sub-segments to reject.
        // The parameter is preserved for IAudioFingerprinter<T> interface
        // compatibility; discard it explicitly to silence the warning.
        _ = minMatchLength;
        // PANNs produces global embeddings; whole-clip similarity only.
        double similarity = ComputeSimilarity(query, reference);
        if (similarity < 0.5) return Array.Empty<FingerprintMatch>();
        return new[]
        {
            new FingerprintMatch
            {
                QueryStartTime = 0.0,
                ReferenceStartTime = 0.0,
                Duration = Math.Max(query.Duration, reference.Duration),
                Confidence = similarity,
                MatchCount = Math.Min(query.Data.Length, reference.Data.Length)
            }
        };
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot train in ONNX inference mode.");
        SetTrainingMode(true);
        try { TrainWithTape(input, expected); }
        finally { SetTrainingMode(false); }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = (int)layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    #endregion

    #region Helpers

    private bool _disposed;
    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing) _disposed = true;
        base.Dispose(disposing);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        _useNativeMode
            ? new PANNsModel<T>(Architecture, _options)
            // ONNX-backed instance: reuse the loaded model path so the
            // clone preserves its execution mode. Without this, Clone() of
            // an ONNX-mode PANNs silently downgrades to native and
            // changes inference behaviour.
            : new PANNsModel<T>(Architecture, _modelPath!, _options);

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() =>
        new ModelMetadata<T>
        {
            Name = Name,
            Description = "PANNs CNN14 — Pretrained Audio Neural Networks (Kong et al. 2020).",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["EmbeddingDim"] = _options.EmbeddingDim,
                ["NumClasses"] = _options.NumClasses,
                ["NumMelBands"] = _options.NumMelBands,
                ["SampleRate"] = _options.SampleRate,
                ["StftWindowSize"] = _options.StftWindowSize,
                ["HopLength"] = _options.HopLength,
            }
        };

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.SampleRate);
        writer.Write(_options.StftWindowSize);
        writer.Write(_options.HopLength);
        writer.Write(_options.NumMelBands);
        writer.Write(_options.NumClasses);
        writer.Write(_options.EmbeddingDim);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        bool useNativeMode = reader.ReadBoolean();
        if (useNativeMode != _useNativeMode)
            throw new InvalidOperationException(
                $"Persisted PANNs mode (native={useNativeMode}) does not match this " +
                $"instance's mode (native={_useNativeMode}). Reconstruct PANNsModel " +
                $"with the matching constructor before loading this checkpoint.");
        VerifyEqual(reader.ReadInt32(),  _options.SampleRate,     nameof(_options.SampleRate));
        VerifyEqual(reader.ReadInt32(),  _options.StftWindowSize, nameof(_options.StftWindowSize));
        VerifyEqual(reader.ReadInt32(),  _options.HopLength,      nameof(_options.HopLength));
        VerifyEqual(reader.ReadInt32(),  _options.NumMelBands,    nameof(_options.NumMelBands));
        VerifyEqual(reader.ReadInt32(),  _options.NumClasses,     nameof(_options.NumClasses));
        VerifyEqual(reader.ReadInt32(),  _options.EmbeddingDim,   nameof(_options.EmbeddingDim));
        VerifyEqual(reader.ReadDouble(), _options.DropoutRate,    nameof(_options.DropoutRate));
    }

    private static void VerifyEqual<TValue>(TValue persisted, TValue current, string name)
        where TValue : IEquatable<TValue>
    {
        if (!persisted.Equals(current))
            throw new InvalidOperationException(
                $"Persisted PANNsModelOptions.{name} = {persisted} does not match constructor option {current}. " +
                "Reconstruct PANNsModel with matching PANNsModelOptions before loading this checkpoint.");
    }

    #endregion
}

/// <summary>
/// PANNs architecture variants (legacy enum — the new code uses
/// <see cref="PANNsModelOptions"/> with explicit dim / depth fields).
/// </summary>
public enum PANNsArchitecture
{
    /// <summary>CNN6: 6-layer CNN (smaller, faster).</summary>
    Cnn6,
    /// <summary>CNN10: 10-layer CNN (balanced).</summary>
    Cnn10,
    /// <summary>CNN14: 14-layer CNN (larger, more accurate).</summary>
    Cnn14,
    /// <summary>ResNet-22: residual variant.</summary>
    ResNet22,
    /// <summary>ResNet-38: deeper residual variant.</summary>
    ResNet38,
    /// <summary>MobileNetV2: lightweight mobile variant.</summary>
    MobileNetV2
}
