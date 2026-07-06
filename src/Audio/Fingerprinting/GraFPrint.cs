using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// GraFPrint graph neural network-based audio fingerprinting model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GraFPrint uses graph neural networks to model spectro-temporal relationships in audio for
/// robust fingerprinting. It constructs a graph from spectrogram features where nodes represent
/// time-frequency points and edges capture local relationships, then applies GNN layers to
/// produce compact fingerprint embeddings.
/// </para>
/// <para>
/// <b>For Beginners:</b> GraFPrint treats a song's spectrogram as a network (graph) of connected
/// sound points, then uses a graph neural network to turn that network into a fingerprint.
/// This captures how different parts of the sound relate to each other, making it robust to
/// distortions like noise or tempo changes.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 256, outputSize: 128);
/// var model = new GraFPrint&lt;float&gt;(arch, "grafprint.onnx");
/// var fp = model.Fingerprint(audioClip);
/// double similarity = model.ComputeSimilarity(fp, referenceFp);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelCategory(ModelCategory.EmbeddingModel)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("GraFPrint: A GNN-Based Approach for Audio Identification", "https://arxiv.org/abs/2410.10994", Year = 2025, Authors = "Aditya Bhattacharjee, Shubhr Singh, Emmanouil Benetos")]
internal class GraFPrint<T> : AudioNeuralNetworkBase<T>, IAudioFingerprinter<T>
{
    #region Fields

    private readonly GraFPrintOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private MelSpectrogram<T>? _melSpectrogram;
    private bool _useNativeMode;
    private bool _disposed;

    private int EffectiveEmbeddingDim => Architecture.OutputSize > 0
        ? Architecture.OutputSize
        : _options.EmbeddingDim;

    /// <inheritdoc />
    public override double MaxGradNormValue => _options?.MaxGradNorm ?? 0.0;

    /// <summary>
    /// Return our paper-faithful AdamW + cosine-annealing optimizer instead
    /// of the default Adam that the base class falls back to. Without this
    /// override the constructor-wired <see cref="_optimizer"/> sits unused —
    /// <see cref="NeuralNetworkBase{T}.TrainWithTape(Tensor{T}, Tensor{T})"/>
    /// resolves the optimizer via <see cref="NeuralNetworkBase{T}.GetOrCreateBaseOptimizer"/>
    /// which creates a fresh default AdamOptimizer (no scheduler, no
    /// paper-tuned LR) and our cosine schedule + LR=1e-4 silently never
    /// engage during training.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        return base.GetOrCreateBaseOptimizer();
    }

    #endregion

    #region IAudioFingerprinter Properties

    /// <inheritdoc />
    public string Name => "GraFPrint";

    /// <inheritdoc />
    public int FingerprintLength => _options.EmbeddingDim;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a GraFPrint model in ONNX inference mode.
    /// </summary>
    public GraFPrint(NeuralNetworkArchitecture<T> architecture, string modelPath, GraFPrintOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options ?? new GraFPrintOptions();
        NormalizeEmbeddingDimFromArchitecture();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels,
            _options.FftSize, _options.HopLength);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a GraFPrint model in native training mode.
    /// </summary>
    public GraFPrint(NeuralNetworkArchitecture<T> architecture, GraFPrintOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new GraFPrintOptions();
        NormalizeEmbeddingDimFromArchitecture();
        _useNativeMode = true;
        // Pass an explicit AdamW options bundle when the caller doesn't
        // supply one — the package-default LR (1e-3) is known to be unstable
        // for this architecture's small-batch BN regime; GraFPrintOptions
        // sources the paper-faithful 1e-4 from the network's own options
        // (Bhattacharjee 2023, §4.1).
        // AdamW with cosine annealing LR scheduler — paper-faithful per
        // Bhattacharjee 2023 §4.1. The scheduler is honored end-to-end:
        // the fused training kernel evaluates the per-step LR inline (no
        // perf penalty vs constant LR) so compile-mode users get the same
        // schedule the paper specifies. Without the schedule, AdamW on a
        // small-batch repeated-sample training scenario tends to overshoot
        // after ~5-10 iters as accumulated momentum pushes weights past
        // the local optimum; cosine decay smoothly reduces step size and
        // prevents that oscillation.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.CosineAnnealingLRScheduler(
                    baseLearningRate: _options.LearningRate,
                    tMax: _options.LRSchedulerTMax,
                    etaMin: _options.LearningRate * 0.01),
            });
        base.SampleRate = _options.SampleRate;
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels,
            _options.FftSize, _options.HopLength);

        // Targeted opt-out from the fused-Adam optimizer step ONLY.
        //
        // Tensors PR #352 has fixed several layers of the original #350 issue:
        //   - Engine routing through scope/tape (BindEngineIfUnset for
        //     BatchNorm + TensorAdd/Subtract/Multiply/ReduceSum)
        //   - BatchNorm specialized backward (BatchNormBackwardInto)
        //   - LazyNode auto-rematerialize through wrong engine (clear
        //     LazySource + IsRealized at compile time)
        //
        // After those fixes, BatchNormGradSlotResidualTests pass 18/18 (was
        // 3 failing). But Training_ShouldReduceLoss on the 53-layer GraFPrint
        // BN pyramid still diverges with fused-Adam: loss explodes from
        // ~75 to >300_000 over the test's 30 iterations. A minimal custom
        // testconsole harness (testconsole/GraFPrintLossTrace.cs) running
        // the SAME architecture + SAME seed + SAME data + SAME 30 iter
        // sequence shows loss DECREASING normally — so the divergence is
        // sensitive to something in the xunit test execution context
        // (static state, threading, allocator pool warm-up order) that
        // hasn't been pinned down yet.
        //
        // Caller-driven opt-out — default false (production path). Tests
        // that hit the unresolved 30-iter divergence on the 53-layer GraFPrint
        // BN pyramid can flip GraFPrintOptions.DisableFusedOptimizerStep
        // to true. ConvBnFusion / dataflow fusion / algebraic backward /
        // forward CSE / BLAS batch / pointwise fusion all stay engaged —
        // only the optimizer step itself runs through eager Adam.
        _fusedTrainingDisabled = _options.DisableFusedOptimizerStep;

        InitializeLayers();
    }

    internal static async Task<GraFPrint<T>> CreateAsync(GraFPrintOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new GraFPrintOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("grafprint", "grafprint.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.EmbeddingDim);
        return new GraFPrint<T>(arch, mp, options);
    }

    #endregion

    #region IAudioFingerprinter

    /// <inheritdoc />
    public AudioFingerprint<T> Fingerprint(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> embedding = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        double norm = 0;
        for (int i = 0; i < embedding.Length; i++) { double v = NumOps.ToDouble(embedding[i]); norm += v * v; }
        norm = Math.Sqrt(norm);

        var data = new T[embedding.Length];
        for (int i = 0; i < embedding.Length; i++)
            data[i] = norm > 0 ? NumOps.FromDouble(NumOps.ToDouble(embedding[i]) / norm) : embedding[i];

        return new AudioFingerprint<T>
        {
            Data = data,
            Duration = audio.Length / (double)_options.SampleRate,
            SampleRate = _options.SampleRate,
            Algorithm = "GraFPrint",
            FrameCount = Math.Max(1, data.Length / Math.Max(1, _options.EmbeddingDim))
        };
    }

    /// <inheritdoc />
    public AudioFingerprint<T> Fingerprint(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++) tensor[i] = audio[i];
        return Fingerprint(tensor);
    }

    /// <inheritdoc />
    public double ComputeSimilarity(AudioFingerprint<T> fp1, AudioFingerprint<T> fp2)
    {
        ThrowIfDisposed();
        double dot = NumOps.ToDouble(VectorHelper.DotProduct(new Vector<T>(fp1.Data), new Vector<T>(fp2.Data)));
        return Math.Max(0, Math.Min(1, (dot + 1.0) / 2.0));
    }

    /// <inheritdoc />
    public IReadOnlyList<FingerprintMatch> FindMatches(AudioFingerprint<T> query, AudioFingerprint<T> reference, int minMatchLength = 10)
    {
        ThrowIfDisposed();
        if (minMatchLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(minMatchLength), "Minimum match length must be positive.");
        var matches = new List<FingerprintMatch>();

        // Sliding window matching: compare sub-fingerprint segments
        int embDim = _options.EmbeddingDim;
        int queryFrames = query.Data.Length / Math.Max(1, embDim);
        int refFrames = reference.Data.Length / Math.Max(1, embDim);

        if (queryFrames <= 0 || refFrames <= 0)
            return matches;

        double threshold = _options.MatchThreshold;

        // For each position in reference, compute similarity with query
        for (int rStart = 0; rStart <= refFrames - Math.Min(queryFrames, minMatchLength); rStart++)
        {
            int matchLen = Math.Min(queryFrames, refFrames - rStart);
            double sim = 0, normQ = 0, normR = 0;
            for (int f = 0; f < matchLen; f++)
            {
                for (int d = 0; d < embDim && (f * embDim + d) < query.Data.Length && ((rStart + f) * embDim + d) < reference.Data.Length; d++)
                {
                    double q = NumOps.ToDouble(query.Data[f * embDim + d]);
                    double r = NumOps.ToDouble(reference.Data[(rStart + f) * embDim + d]);
                    sim += q * r;
                    normQ += q * q;
                    normR += r * r;
                }
            }
            double denom = Math.Sqrt(normQ) * Math.Sqrt(normR);
            double cosSim = denom > 1e-8 ? sim / denom : 0;

            if (cosSim >= threshold && matchLen >= minMatchLength)
            {
                double timePerFrame = query.Duration / Math.Max(1, queryFrames);
                matches.Add(new FingerprintMatch
                {
                    QueryStartTime = 0,
                    ReferenceStartTime = rStart * timePerFrame,
                    Duration = matchLen * timePerFrame,
                    Confidence = cosSim,
                    MatchCount = matchLen
                });
                rStart += matchLen - 1; // Skip past matched region
            }
        }
        return matches;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultGraFPrintLayers(
                numMels: _options.NumMels, gnnHiddenDim: _options.GnnHiddenDim,
                numGnnLayers: _options.NumGnnLayers, numAttentionHeads: _options.NumAttentionHeads,
                embeddingDim: _options.EmbeddingDim, dropoutRate: _options.DropoutRate));
        }

        // Per-layer deterministic seeding when the architecture pins a seed.
        // Each layer gets a distinct derived seed (base ⊕ index) so weights
        // differ across layers while staying reproducible across processes —
        // mirrors LayerHelper.Wire's seedRng.Next() pattern but inline here
        // since GraFPrint constructs its layers directly. Without this,
        // layer init falls back to RandomHelper.ThreadSafeRandom's
        // process-global counter, which advances based on construction
        // order across tests and produces different init weights every run.
        if (Architecture.RandomSeed.HasValue)
        {
            var seedRng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed.Value);
            for (int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i] is AiDotNet.NeuralNetworks.Layers.LayerBase<T> lb)
                    lb.RandomSeed = seedRng.Next();
            }
        }
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);

        // Auto-flip to eval mode for the forward, matching the base
        // NeuralNetworkBase.Predict contract that mirrors PyTorch's
        // model.eval(). The base class's auto-flip is bypassed by this
        // override, so without an explicit flip here BatchNorm consumes
        // single-sample batch statistics (degenerate, ~0 variance) and
        // DropoutLayer samples a random mask per Predict call — both
        // make output non-deterministic across two Predict calls on the
        // same input, which is exactly what the generated
        // SimilarInputs_ProduceSimilarEmbeddings invariant catches
        // (cosine ~0.31 between embeddings of inputs differing by 1e-6).
        // Restore the prior training mode in finally so a Predict-inside-
        // a-training-loop call doesn't permanently flip the network.
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {

        // The paper-faithful chain uses BatchNormalizationLayer between every
        // conv. BN's inference broadcast in this codebase
        // (BatchNormalizationLayer.ApplyInferenceAnyRank) assumes the
        // canonical NCHW layout — dim 0 = batch, dim 1 = channels. For a
        // rank-3 unbatched [C, H, W] image the broadcast picks the wrong
        // axis and the shape mismatches at the first BN. Force a leading
        // batch dim so the entire chain stays in [B, C, H, W] form,
        // matching what NeuralNetworkBase.NormalizeBatchDim would emit for
        // Train's tape forward — inference and training shape-consistent.
        var batched = input.Rank == 3
            ? Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]])
            : input;

        var c = batched;
        foreach (var l in Layers) c = l.Forward(c);

        // Paper's graph_encoder.py finishes with
        //   x = self.proj(x)            # [B, C, N, 1]
        //   x = torch.mean(x, dim=2)    # [B, C, 1]
        //   x = x.squeeze(-1).squeeze(-1)  # [B, C]
        // i.e. the public output is rank-2 [B, embeddingDim]. GlobalPooling
        // with keepDims=true yields rank-4 [B, embeddingDim, 1, 1]; squeeze
        // trailing singleton dims so the base test class's warm-up infers
        // a rank-2 EffectiveOutputShape that downstream
        // CreateRandomTargetTensor calls can use without a rank mismatch
        // against the loss target.
        while (c.Rank > 2 && c.Shape[c.Rank - 1] == 1)
        {
            var newShape = new int[c.Rank - 1];
            for (int i = 0; i < c.Rank - 1; i++) newShape[i] = c.Shape[i];
            c = Engine.Reshape(c, newShape);
        }
        return c;

        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");

        // Same rank-3 → rank-4 promotion Predict applies, so the training-
        // time ForwardForTraining walk sees the same shape sequence as
        // inference. Without this, BN's any-rank inference path picks the
        // wrong channel axis on rank-3 input and the broadcast mismatches.
        var batchedInput = input.Rank == 3
            ? Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]])
            : input;

        // ForwardForTraining walks Layers in order without the post-chain
        // squeeze Predict applies. The chain ends at GlobalPoolingLayer
        // with keepDims=true → rank-4 [B, embeddingDim, 1, 1]. Reshape the
        // expected target to match so MSE / cross-entropy see same-rank
        // tensors. Element count is preserved.
        Tensor<T> alignedTarget = expected;
        if (expected.Rank == 1)
            alignedTarget = Engine.Reshape(expected, [1, expected.Shape[0], 1, 1]);
        else if (expected.Rank == 2)
            alignedTarget = Engine.Reshape(expected, [expected.Shape[0], expected.Shape[1], 1, 1]);

        // Note: the per-model _fusedTrainingDisabled flag is set in the
        // constructor (see ctor for the BN-pyramid divergence rationale).
        // That bypasses ONLY the fused-Adam optimizer-step path while
        // leaving every other compile-mode optimization engaged
        // (ConvBnFusion, dataflow fusion, algebraic backward, etc.).

        SetTrainingMode(true);
        try
        {
            TrainWithTape(batchedInput, alignedTarget);
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
        if (_melSpectrogram is not null) return _melSpectrogram.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "GraFPrint-Native" : "GraFPrint-ONNX",
            Description = "GraFPrint graph neural network audio fingerprinting",
            Complexity = _options.NumGnnLayers
        };
        m.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        m.AdditionalInfo["GnnHiddenDim"] = _options.GnnHiddenDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.FftSize);
        w.Write(_options.HopLength); w.Write(_options.SegmentDurationSec);
        w.Write(_options.EmbeddingDim); w.Write(_options.GnnHiddenDim);
        w.Write(_options.NumGnnLayers); w.Write(_options.NumAttentionHeads);
        w.Write(_options.KNeighbors); w.Write(_options.Temperature); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.FftSize = r.ReadInt32();
        _options.HopLength = r.ReadInt32(); _options.SegmentDurationSec = r.ReadDouble();
        _options.EmbeddingDim = r.ReadInt32(); _options.GnnHiddenDim = r.ReadInt32();
        _options.NumGnnLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32();
        _options.KNeighbors = r.ReadInt32(); _options.Temperature = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        NormalizeEmbeddingDimFromArchitecture();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels, _options.FftSize, _options.HopLength);
    }

    private void NormalizeEmbeddingDimFromArchitecture()
    {
        int embeddingDim = EffectiveEmbeddingDim;
        if (embeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(_options.EmbeddingDim), embeddingDim, "EmbeddingDim must be greater than 0.");

        _options.EmbeddingDim = embeddingDim;
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new GraFPrint<T>(Architecture, mp, _options);
        return new GraFPrint<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GraFPrint<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
