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
/// AST (Audio Spectrogram Transformer) — a single-stream Vision-Transformer
/// applied to log-mel spectrograms, trained for audio event classification
/// and fingerprinting (Gong et al. 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AST applies the standard ViT recipe (Dosovitskiy et al. 2021) directly to
/// audio: convert the waveform to a log-mel spectrogram, treat it as a 2-D
/// image, slice it into 16×16 patches, embed each patch linearly, add
/// positional encodings, run N=12 transformer encoder layers, mean-pool, and
/// classify. AST-Base (768-dim, 12-layer, 12-head) initialised from a
/// DeiT/ViT ImageNet checkpoint achieves SOTA on AudioSet.
/// </para>
/// <para>
/// <b>Reference:</b> Gong, Y., Chung, Y.-A. &amp; Glass, J. (2021),
/// "AST: Audio Spectrogram Transformer", Interspeech.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "AST: Audio Spectrogram Transformer",
    "https://arxiv.org/abs/2104.01778",
    Year = 2021,
    Authors = "Yuan Gong, Yu-An Chung, James Glass")]
public class ASTModel<T> : AudioNeuralNetworkBase<T>, IAudioFingerprinter<T>
{
    private readonly ASTModelOptions _options;
    private readonly bool _useNativeMode;

    /// <summary>
    /// Cached Hann window for the STFT preprocessing step. Built once on the
    /// first <see cref="PreprocessAudio"/> call and reused.
    /// </summary>
    private Tensor<T>? _hannWindow;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <inheritdoc/>
    public string Name => _useNativeMode ? "AST-Native" : "AST-ONNX";

    /// <inheritdoc/>
    public int FingerprintLength => _options.EmbeddingDim;

    #region Constructors

    /// <summary>Initializes AST in ONNX inference mode.</summary>
    public ASTModel(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ASTModelOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ASTModelOptions();
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

    /// <summary>Initializes AST in native training / inference mode.</summary>
    public ASTModel(
        NeuralNetworkArchitecture<T> architecture,
        ASTModelOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ASTModelOptions();
        SampleRate = _options.SampleRate;
        NumMels = _options.NumMelBands;
        _useNativeMode = true;
        InitializeLayers();
    }

    #endregion

    #region Layer Construction (Golden-Standard Pattern)

    /// <summary>
    /// Initializes the AST layer stack following the codebase's golden
    /// single-stream pattern: prefer user-supplied <c>Architecture.Layers</c>;
    /// otherwise fall back to the paper-faithful <see cref="LayerHelper{T}.CreateDefaultASTLayers"/>.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultASTLayers(
                patchSize: _options.PatchSize,
                embeddingDim: _options.EmbeddingDim,
                numLayers: _options.NumLayers,
                numHeads: _options.NumHeads,
                feedForwardDim: _options.FeedForwardDim,
                numClasses: _options.NumClasses,
                maxSequenceLength: _options.TargetLength));
        }
    }

    #endregion

    #region Preprocessing — Mel Spectrogram

    /// <summary>
    /// Converts raw audio samples into a log-mel spectrogram via the engine's
    /// fused <see cref="IEngine.MelSpectrogram{T}"/> kernel — the AST §2.1
    /// 128-mel × 10 ms-hop pipeline routed through a single BLAS / GPU-eligible
    /// engine op.
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

    /// <summary>Returns the top-K class predictions with softmax-normalised probabilities.</summary>
    public List<(string Label, double Probability)> Classify(Tensor<T> audio, int topK = 5)
    {
        var logits = Predict(audio);
        var probs = SoftmaxLastAxis(logits);

        // Take the first batch row and pick top-K.
        int classCount = probs.Shape[^1];
        var indexed = new (double prob, int idx)[classCount];
        for (int i = 0; i < classCount; i++)
            indexed[i] = (Convert.ToDouble(probs[0, i]), i);
        Array.Sort(indexed, (a, b) => b.prob.CompareTo(a.prob));

        var result = new List<(string, double)>(topK);
        for (int k = 0; k < Math.Min(topK, classCount); k++)
            result.Add(($"class_{indexed[k].idx}", indexed[k].prob));
        return result;
    }

    /// <inheritdoc/>
    public AudioFingerprint<T> Fingerprint(Tensor<T> audio)
    {
        // InitializeLayers leaves Layers empty in ONNX mode (the ONNX
        // graph is executed via OnnxEncoder instead), so the layer-walk
        // below would exit immediately and serialize the preprocessed
        // mel spectrogram as the "fingerprint" — silently wrong. Require
        // native mode here until an ONNX-embedding output is wired
        // explicitly.
        if (!_useNativeMode)
        {
            throw new NotSupportedException(
                "ASTModel.Fingerprint() in ONNX mode requires an explicit " +
                "embedding-producing output to be wired. Use the native " +
                "constructor or invoke the ONNX encoder directly via the " +
                "OnnxEncoder property.");
        }

        // For fingerprinting we want the pooled embedding (one layer before the
        // classification head). Run the layer stack and stop at the
        // GlobalPoolingLayer output.
        var mel = PreprocessAudio(audio);
        var hidden = mel;
        foreach (var layer in Layers)
        {
            hidden = layer.Forward(hidden);
            if (layer is GlobalPoolingLayer<T>) break;
        }

        var flat = hidden.ToVector();
        var data = new T[flat.Length];
        for (int i = 0; i < flat.Length; i++) data[i] = flat[i];
        return new AudioFingerprint<T>
        {
            Data = data,
            SampleRate = SampleRate,
            Duration = audio.Length / (double)SampleRate
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
        // AST produces global embeddings; whole-clip similarity only.
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

    /// <summary>Numerically-stable softmax along the last axis. Engine-routed.</summary>
    private Tensor<T> SoftmaxLastAxis(Tensor<T> logits) =>
        Engine.TensorSoftmax(logits, axis: logits.Shape.Length - 1);

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
            ? new ASTModel<T>(Architecture, _options)
            // ONNX-backed instance: preserve the loaded model path so the
            // clone keeps its execution mode. Without this, Clone() of an
            // ONNX-mode AST silently downgrades to native (no weights,
            // empty Layers) and changes inference behaviour.
            : new ASTModel<T>(Architecture, _modelPath!, _options);

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() =>
        new ModelMetadata<T>
        {
            Name = Name,
            Description = "AST — Audio Spectrogram Transformer (Gong et al. 2021).",
            Complexity = _options.NumLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["EmbeddingDim"] = _options.EmbeddingDim,
                ["NumLayers"] = _options.NumLayers,
                ["NumHeads"] = _options.NumHeads,
                ["PatchSize"] = _options.PatchSize,
                ["NumMelBands"] = _options.NumMelBands,
                ["TargetLength"] = _options.TargetLength,
                ["NumClasses"] = _options.NumClasses,
                ["SampleRate"] = _options.SampleRate,
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
        writer.Write(_options.TargetLength);
        writer.Write(_options.PatchSize);
        writer.Write(_options.NumClasses);
        writer.Write(_options.EmbeddingDim);
        writer.Write(_options.NumLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.FeedForwardDim);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        bool useNativeMode = reader.ReadBoolean();
        if (useNativeMode != _useNativeMode)
            throw new InvalidOperationException(
                $"Persisted AST mode (native={useNativeMode}) does not match this " +
                $"instance's mode (native={_useNativeMode}). Reconstruct ASTModel " +
                $"with the matching constructor before loading this checkpoint.");
        VerifyEqual(reader.ReadInt32(),  _options.SampleRate,     nameof(_options.SampleRate));
        VerifyEqual(reader.ReadInt32(),  _options.StftWindowSize, nameof(_options.StftWindowSize));
        VerifyEqual(reader.ReadInt32(),  _options.HopLength,      nameof(_options.HopLength));
        VerifyEqual(reader.ReadInt32(),  _options.NumMelBands,    nameof(_options.NumMelBands));
        VerifyEqual(reader.ReadInt32(),  _options.TargetLength,   nameof(_options.TargetLength));
        VerifyEqual(reader.ReadInt32(),  _options.PatchSize,      nameof(_options.PatchSize));
        VerifyEqual(reader.ReadInt32(),  _options.NumClasses,     nameof(_options.NumClasses));
        VerifyEqual(reader.ReadInt32(),  _options.EmbeddingDim,   nameof(_options.EmbeddingDim));
        VerifyEqual(reader.ReadInt32(),  _options.NumLayers,      nameof(_options.NumLayers));
        VerifyEqual(reader.ReadInt32(),  _options.NumHeads,       nameof(_options.NumHeads));
        VerifyEqual(reader.ReadInt32(),  _options.FeedForwardDim, nameof(_options.FeedForwardDim));
        VerifyEqual(reader.ReadDouble(), _options.DropoutRate,    nameof(_options.DropoutRate));
    }

    private static void VerifyEqual<TValue>(TValue persisted, TValue current, string name)
        where TValue : IEquatable<TValue>
    {
        if (!persisted.Equals(current))
            throw new InvalidOperationException(
                $"Persisted ASTModelOptions.{name} = {persisted} does not match constructor option {current}. " +
                "Reconstruct ASTModel with matching ASTModelOptions before loading this checkpoint.");
    }

    #endregion
}
