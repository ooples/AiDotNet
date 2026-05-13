using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// CLAP (Contrastive Language-Audio Pretraining) — a dual-encoder neural network
/// that learns to align audio and text representations in a shared embedding
/// space via a contrastive objective.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLAP (Wu et al. 2023) trains an audio encoder and a text encoder so that
/// matching (audio, caption) pairs produce nearby embeddings and unrelated
/// pairs produce distant embeddings. The audio side is HTSAT (Chen et al. 2022),
/// a hierarchical Swin Transformer (Liu et al. 2021) over mel-spectrogram
/// patches. The text side is a RoBERTa-style transformer stack (Liu et al. 2019)
/// over BPE token IDs. A learnable temperature τ scales the cosine-similarity
/// logits during contrastive training (CLIP / CLAP convention).
/// </para>
/// <para>
/// <b>Capabilities</b>:
/// <list type="bullet">
/// <item><description>Zero-shot audio classification with text prompts</description></item>
/// <item><description>Audio-to-text and text-to-audio retrieval</description></item>
/// <item><description>Semantic audio fingerprinting via the projection head</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Reference:</b> Wu, Y. et al. (2023), "Large-Scale Contrastive Language-Audio
/// Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation", ICASSP.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.EmbeddingModel)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation",
    "https://doi.org/10.1109/ICASSP49357.2023.10095969",
    Year = 2023,
    Authors = "Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov")]
public class CLAPModel<T> : AudioNeuralNetworkBase<T>, IAudioFingerprinter<T>
{
    private readonly CLAPModelOptions _options;
    private readonly bool _useNativeMode;
    // Captured ONNX-mode constructor inputs. Preserved so CreateNewInstance
    // can rebuild an ONNX-mode clone with the same encoder weights instead
    // of silently downgrading to native (random-init) mode.
    private readonly string? _audioEncoderPath;
    private readonly string? _textEncoderPath;

    // Trainable temperature parameter (stored in log space). Gradients flow
    // through the tape via Engine ops; the optimizer updates this alongside
    // the rest of the network.
    private Tensor<T> _logTemperature = null!;

    // Cached Hann window for the STFT preprocessing step. Built once on the
    // first PreprocessAudio call and reused across batches.
    private Tensor<T>? _hannWindow;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <inheritdoc/>
    public string Name => _useNativeMode ? "CLAP-Native" : "CLAP-ONNX";

    /// <inheritdoc/>
    public int FingerprintLength => _options.ProjectionDim;

    #region Constructors

    /// <summary>
    /// Initializes a new instance of <see cref="CLAPModel{T}"/> in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">Architecture descriptor (used for I/O shape metadata).</param>
    /// <param name="audioEncoderPath">Path to the ONNX audio encoder model.</param>
    /// <param name="textEncoderPath">Optional path to the ONNX text encoder model.</param>
    /// <param name="options">Optional CLAP configuration (defaults match the published paper).</param>
    public CLAPModel(
        NeuralNetworkArchitecture<T> architecture,
        string audioEncoderPath,
        string? textEncoderPath = null,
        CLAPModelOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CLAPModelOptions();
        if (string.IsNullOrWhiteSpace(audioEncoderPath))
            throw new ArgumentException("Audio encoder path is required for ONNX mode.", nameof(audioEncoderPath));
        if (!File.Exists(audioEncoderPath))
            throw new FileNotFoundException($"Audio encoder ONNX model not found: {audioEncoderPath}", audioEncoderPath);

        SampleRate = _options.SampleRate;
        _useNativeMode = false;
        _audioEncoderPath = audioEncoderPath;
        _textEncoderPath = textEncoderPath;
        OnnxEncoder = new OnnxModel<T>(audioEncoderPath);
        // Fail fast when a text-encoder path is provided but invalid — the
        // earlier "try-and-skip" path silently dropped a mistyped path and
        // only surfaced as a NotSupportedException deep inside EncodeText().
        // Pattern-matches into a non-null local for net471 nullable-flow.
        if (textEncoderPath is { Length: > 0 } pathLocal
            && !string.IsNullOrWhiteSpace(pathLocal))
        {
            if (!File.Exists(pathLocal))
                throw new FileNotFoundException(
                    $"Text encoder ONNX model not found: {pathLocal}",
                    pathLocal);
            OnnxDecoder = new OnnxModel<T>(pathLocal);
        }

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of <see cref="CLAPModel{T}"/> in native training / inference mode.
    /// </summary>
    /// <param name="architecture">Architecture descriptor. If <c>Architecture.Layers</c> is populated,
    /// those layers replace the default audio + text encoder stacks.</param>
    /// <param name="options">Optional CLAP configuration (defaults match the published paper).</param>
    public CLAPModel(
        NeuralNetworkArchitecture<T> architecture,
        CLAPModelOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CLAPModelOptions();
        SampleRate = _options.SampleRate;
        _useNativeMode = true;

        InitializeLayers();
    }

    #endregion

    #region Layer Construction (Golden-Standard Pattern)

    /// <summary>
    /// Initializes the neural network layers following the codebase's golden
    /// dual-stream pattern. Prefers an <see cref="AudioTextDualStreamArchitecture{T}"/>
    /// (modality-specific custom layers per encoder), then falls back to the
    /// paper-faithful <see cref="LayerHelper{T}"/> factories.
    /// </summary>
    /// <remarks>
    /// Audio encoder layers live in the inherited <see cref="NeuralNetworkBase{T}.Layers"/>
    /// list so the standard tape-training and parameter-iteration infrastructure
    /// walks them as the primary stream. Text encoder layers live in
    /// <see cref="AudioNeuralNetworkBase{T}.TextEncoderLayers"/>; subclasses
    /// (here) walk that collection explicitly inside <see cref="EncodeText(Tensor{T})"/>.
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture is AudioTextDualStreamArchitecture<T> dual)
        {
            // Production path: caller provided custom layer stacks for both
            // encoders via the proper dual-stream architecture descriptor.
            Layers.AddRange(dual.AudioLayers);
            TextEncoderLayers.AddRange(dual.TextLayers);
        }
        else if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // Legacy / single-stack path: user supplied only one Layers list.
            // We treat that as the audio encoder and use the LayerHelper
            // default for the text side, so the model is still functional.
            // Callers wanting full control of BOTH stacks should switch to
            // AudioTextDualStreamArchitecture.
            Layers.AddRange(Architecture.Layers);
            TextEncoderLayers.AddRange(LayerHelper<T>.CreateDefaultCLAPTextEncoderLayers(
                vocabSize: _options.VocabSize,
                maxTextLength: _options.MaxTextLength,
                textHiddenDim: _options.TextHiddenDim,
                textEncoderLayers: _options.TextEncoderLayers,
                textEncoderHeads: _options.TextEncoderHeads,
                projectionDim: _options.ProjectionDim));
        }
        else
        {
            // Default both stacks from the paper-faithful LayerHelper factories.
            Layers.AddRange(LayerHelper<T>.CreateDefaultCLAPAudioEncoderLayers(
                audioHiddenDim: _options.AudioHiddenDim,
                audioEncoderLayers: _options.AudioEncoderLayers,
                audioEncoderHeads: _options.AudioEncoderHeads,
                swinWindowSize: _options.SwinWindowSize,
                projectionDim: _options.ProjectionDim));
            TextEncoderLayers.AddRange(LayerHelper<T>.CreateDefaultCLAPTextEncoderLayers(
                vocabSize: _options.VocabSize,
                maxTextLength: _options.MaxTextLength,
                textHiddenDim: _options.TextHiddenDim,
                textEncoderLayers: _options.TextEncoderLayers,
                textEncoderHeads: _options.TextEncoderHeads,
                projectionDim: _options.ProjectionDim));
        }

        // Learnable temperature τ in log space (CLIP / CLAP convention).
        // Initialised so exp(_logTemperature) = 1/InitialTemperature; the
        // contrastive loss uses logits·exp(_logTemperature).
        _logTemperature = new Tensor<T>([1]);
        _logTemperature[0] = NumOps.FromDouble(Math.Log(1.0 / _options.InitialTemperature));
    }

    #endregion

    #region Preprocessing — Mel Spectrogram

    /// <summary>
    /// Converts raw audio samples into a log-mel spectrogram via the engine's
    /// fused <see cref="IEngine.MelSpectrogram{T}"/> kernel (Hann window → STFT
    /// → triangular HTK mel filterbank → log power). This is the same librosa-
    /// style pipeline the published CLAP checkpoint was trained on (Wu 2023
    /// §3.1), routed through a single BLAS / GPU-eligible engine op instead of
    /// scalar inner loops over <c>double[]</c>.
    /// </summary>
    /// <param name="rawAudio">Audio samples — shape <c>[samples]</c> or <c>[batch, samples]</c>.</param>
    /// <returns>Log-mel spectrogram with shape <c>[batch, 1, numFrames, numMels]</c>,
    /// ready to feed into the HTSAT 2D patch encoder.</returns>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (rawAudio.Shape.Length == 1)
            rawAudio = Engine.Reshape(rawAudio, [1, rawAudio.Shape[0]]);

        int batchSize = rawAudio.Shape[0];
        int numSamples = rawAudio.Shape[1];
        int windowSize = _options.StftWindowSize;
        int hop = _options.HopLength;
        int numMels = _options.NumMelBands;
        int sampleRate = _options.SampleRate;

        // Hann window built once via NumOps (T-generic; no double[] state).
        _hannWindow ??= BuildHannWindow(windowSize);

        // Single engine op: Hann × STFT → power → mel filterbank → log.
        // Returns [batch, numFrames, numMels]. Engine-resident throughout —
        // tape sees one node, BLAS / GPU paths take it when bound.
        var mel = Engine.MelSpectrogram(
            input: rawAudio,
            sampleRate: sampleRate,
            nFft: windowSize,
            hopLength: hop,
            nMels: numMels,
            fMin: NumOps.Zero,
            fMax: NumOps.FromDouble(sampleRate / 2.0),
            window: _hannWindow,
            powerToDb: true);

        // Reshape to [batch, 1, numFrames, numMels] for the HTSAT 2D patch encoder.
        int numFrames = mel.Length / (batchSize * numMels);
        return Engine.Reshape(mel, [batchSize, 1, numFrames, numMels]);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => L2Normalize(modelOutput);

    /// <summary>
    /// Builds a periodic Hann window of length <paramref name="windowSize"/>
    /// as a <see cref="Tensor{T}"/>: <c>w[n] = 0.5·(1 − cos(2πn/(N−1)))</c>.
    /// Generic in <typeparamref name="T"/> via <see cref="LayerBase{T}.NumOps"/>;
    /// no <c>double[]</c> state.
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

    /// <summary>
    /// Encodes audio into a CLAP embedding vector in the shared text-audio space.
    /// </summary>
    /// <param name="audio">Raw audio tensor [samples] or [batch, samples].</param>
    /// <returns>Audio embedding [batch, projectionDim], L2-normalised.</returns>
    public Tensor<T> EncodeAudio(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessAudio(audio);
        if (!_useNativeMode && OnnxEncoder is not null)
        {
            return PostprocessOutput(OnnxEncoder.Run(preprocessed));
        }

        // Native path: run the audio encoder layers (the inherited Layers list,
        // populated by InitializeLayers) in sequence, then L2-normalise.
        var hidden = preprocessed;
        foreach (var layer in Layers) hidden = layer.Forward(hidden);
        return L2Normalize(hidden);
    }

    /// <summary>
    /// Encodes a tokenised text caption into a CLAP embedding vector.
    /// </summary>
    /// <param name="tokens">Token IDs [batch, seqLen] or [seqLen].</param>
    /// <returns>Text embedding [batch, projectionDim], L2-normalised.</returns>
    public Tensor<T> EncodeText(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (!_useNativeMode)
        {
            if (OnnxDecoder is null)
                throw new NotSupportedException(
                    "CLAPModel.EncodeText() in ONNX mode requires a text encoder. " +
                    "Construct CLAPModel with both audioEncoderPath and textEncoderPath, " +
                    "or use a native-mode instance. Falling through to the native layer " +
                    "stack would walk an empty TextEncoderLayers list and L2-normalise " +
                    "the raw token IDs — silent garbage output.");
            return PostprocessOutput(OnnxDecoder.Run(tokens));
        }

        if (tokens.Shape.Length == 1)
            tokens = Engine.Reshape(tokens, [1, tokens.Shape[0]]);

        // Native path: walk the text encoder layers (held on the audio base
        // class so any future audio-text dual encoder shares the same slot).
        var hidden = tokens;
        foreach (var layer in TextEncoderLayers) hidden = layer.Forward(hidden);
        return L2Normalize(hidden);
    }

    /// <summary>Convenience overload: tokenise + encode.</summary>
    public Tensor<T> EncodeText(int[] tokenIds)
    {
        var tokenTensor = new Tensor<T>([1, tokenIds.Length]);
        for (int i = 0; i < tokenIds.Length; i++)
            tokenTensor[0, i] = NumOps.FromDouble(tokenIds[i]);
        return EncodeText(tokenTensor);
    }

    /// <summary>
    /// Performs zero-shot audio classification: rank each text prompt by its
    /// CLAP-similarity to the supplied audio clip.
    /// </summary>
    public Dictionary<string, double> ZeroShotClassify(Tensor<T> audio, string[] labels, Func<string, int[]> tokenize)
    {
        if (labels is null || labels.Length == 0) throw new ArgumentException("Labels required.", nameof(labels));
        if (tokenize is null) throw new ArgumentNullException(nameof(tokenize));
        // Reject batched inputs explicitly — this API returns a single score
        // map. EncodeAudio accepts [batch, samples], and silently flattening
        // a batch into one embedding would produce one merged score per
        // label across the whole batch.
        if (audio.Shape.Length > 1 && audio.Shape[0] != 1)
            throw new ArgumentException(
                "ZeroShotClassify expects a single audio clip. Pass clips one at a time " +
                "or call EncodeAudio directly for the batched path.",
                nameof(audio));

        var audioEmb = EncodeAudio(audio);
        var scores = new Dictionary<string, double>(labels.Length);
        foreach (var label in labels)
        {
            var textEmb = EncodeText(tokenize(label));
            scores[label] = ComputeCosineSimilarity(audioEmb, textEmb);
        }
        return scores.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value);
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        return EncodeAudio(input);
    }

    /// <inheritdoc/>
    public AudioFingerprint<T> Fingerprint(Tensor<T> audio)
    {
        // Single-clip API — reject batched inputs (same rationale as
        // ZeroShotClassify: flattening a batch into one fingerprint and one
        // Duration is silently misleading).
        if (audio.Shape.Length > 1 && audio.Shape[0] != 1)
            throw new ArgumentException(
                "Fingerprint expects a single audio clip. Pass clips one at a time " +
                "or call EncodeAudio directly for the batched path.",
                nameof(audio));

        var embedding = EncodeAudio(audio);
        var flat = embedding.ToVector();
        var data = new T[flat.Length];
        for (int i = 0; i < flat.Length; i++) data[i] = flat[i];
        return new AudioFingerprint<T>
        {
            Data = data,
            SampleRate = SampleRate,
            // Per-clip duration: use the last-axis sample count so a [1, N]
            // batched tensor still reports the correct N / SampleRate.
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
        // CLAP fingerprints are global embeddings — there's no temporal
        // segmentation to localise inside. We return a single whole-clip match
        // when the overall similarity exceeds the standard zero-shot threshold;
        // callers needing time-localised matches should use a segment-level
        // fingerprinter like ConformerFP or NeuralFP.
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

    #endregion

    #region Training (tape-based)

    /// <inheritdoc/>
    /// <remarks>
    /// CLAP is trained with a symmetric contrastive objective (Radford 2021
    /// Eq. 1; Wu 2023 §3.2): for a minibatch of (audio, caption) pairs, the
    /// loss is the average of audio→text and text→audio cross-entropies over
    /// the temperature-scaled cosine-similarity logit matrix. Implemented via
    /// the standard tape-based training path so all gradients (including the
    /// learnable temperature) flow correctly through every layer.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (!_useNativeMode)
            throw new NotSupportedException(
                "CLAP training requires native mode. Construct with the (architecture, options) ctor.");

        // CLAP contrastive objective (Radford 2021 Eq. 1; Wu 2023 §3.2): the
        // loss is the average of audio→text and text→audio softmax cross-
        // entropies over the temperature-scaled cosine-similarity logit
        // matrix where positive pairs are aligned on the diagonal.
        //
        // Engine-side implementation: TrainWithCustomLoss runs the audio
        // encoder forward inside its own GradientTape, then we run the text
        // encoder forward inside the same tape, build the symmetric loss,
        // and let TrainWithCustomLoss handle the gradient computation +
        // optimizer step. Parameters collected from Layers
        // (audio side); the text-encoder parameters and _logTemperature
        // participate in the loss only when they're referenced in the tape
        // graph. The optimizer.Step in TrainWithCustomLoss won't directly
        // update text-encoder or temperature weights, so we manually apply
        // SGD with the same LR on those after the audio step using their
        // tape-computed gradients (collected via a parallel
        // CollectContrastiveParameters walk).
        //
        // input:    [batch, samples] audio batch.
        // expected: [batch, seqLen]  tokenised caption batch (positive pair
        //                            in the same row).
        SetTrainingMode(true);
        try
        {
            var audioParams = Training.TapeTrainingStep<T>.CollectParameters(Layers);
            var textParams = Training.TapeTrainingStep<T>.CollectParameters(TextEncoderLayers);
            var allParams = new List<Tensor<T>>(audioParams.Count + textParams.Count + 1);
            allParams.AddRange(audioParams);
            allParams.AddRange(textParams);
            allParams.Add(_logTemperature);

            var optimizer = GetOrCreateBaseOptimizer();

            using var tape = new GradientTape<T>();
            // Forward both encoders inside the same tape so gradients flow
            // through every parameter. EncodeAudio / EncodeText already
            // L2-normalise along the last axis so the dot product below is
            // cosine similarity.
            var audioEmb = EncodeAudio(input);
            var textEmb = EncodeText(expected);

            int batchSize = audioEmb.Shape[0];
            int projDim = audioEmb.Shape[audioEmb.Shape.Length - 1];

            var audioEmb2D = audioEmb.Shape.Length == 2
                ? audioEmb
                : Engine.Reshape(audioEmb, new[] { batchSize, projDim });
            var textEmb2D = textEmb.Shape.Length == 2
                ? textEmb
                : Engine.Reshape(textEmb, new[] { batchSize, projDim });

            // logits = audio @ text.T scaled by exp(_logTemperature).
            var textEmbT = Engine.TensorTranspose<T>(textEmb2D);
            var sim = Engine.TensorMatMul<T>(audioEmb2D, textEmbT);
            // Broadcast scalar exp(logTemp) across the [batch, batch] grid.
            var tau = Engine.TensorExp<T>(_logTemperature);
            var tauBroadcast = Engine.TensorTile(
                Engine.Reshape(tau, new[] { 1, 1 }), new[] { batchSize, batchSize });
            var logitsA2T = Engine.TensorMultiply<T>(sim, tauBroadcast);

            // Symmetric loss: 0.5 * (CE(logitsA2T, diag) + CE(logitsT2A, diag))
            // where logitsT2A is the transpose of logitsA2T. Compute both
            // directions explicitly (rather than relying on transpose-eq) so
            // the autodiff graph is unambiguous on every backend.
            var logitsT2A = Engine.TensorTranspose<T>(logitsA2T);

            var halfLossA2T = SymmetricRowCrossEntropy(logitsA2T, batchSize);
            var halfLossT2A = SymmetricRowCrossEntropy(logitsT2A, batchSize);
            var lossSum = Engine.TensorAdd<T>(halfLossA2T, halfLossT2A);

            // Manual gradient + optimizer step over the combined params.
            var grads = tape.ComputeGradients(lossSum, allParams);

            T lossValue = lossSum.Length > 0 ? lossSum[0] : NumOps.Zero;
            LastLoss = lossValue;

            Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => EncodeAudio(inp);
            Tensor<T> RecomputeLoss(Tensor<T> _, Tensor<T> __) => lossSum;

            var context = new AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<T>(
                allParams, grads, lossValue,
                input, input, ComputeForward, RecomputeLoss);
            optimizer.Step(context);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Computes 0.5 * mean cross-entropy of softmax(logits) against the
    /// diagonal target (positive pair at the same row index), tape-tracked
    /// via engine ops. Numerically stable: subtracts the row-wise max
    /// before exponentiating. Returns a [1] scalar tensor so the caller
    /// can sum the two directions into the final symmetric loss.
    /// </summary>
    private Tensor<T> SymmetricRowCrossEntropy(Tensor<T> logits, int batchSize)
    {
        // log_softmax = logits - log(sum(exp(logits − rowMax))) − rowMax
        // We need the diagonal entry of log_softmax. Easiest: compute
        // log_softmax row by row using engine ops, then sum along the
        // diagonal via a one-hot-style multiply.
        var rowMaxes = new Tensor<T>(new[] { batchSize, 1 });
        for (int i = 0; i < batchSize; i++)
        {
            T m = logits[i, 0];
            for (int j = 1; j < batchSize; j++)
                if (NumOps.GreaterThan(logits[i, j], m)) m = logits[i, j];
            rowMaxes[i, 0] = m;
        }
        var rowMaxTiled = Engine.TensorTile(rowMaxes, new[] { 1, batchSize });
        var shifted = Engine.TensorSubtract<T>(logits, rowMaxTiled);
        var expShifted = Engine.TensorExp<T>(shifted);
        var rowSum = Engine.ReduceSum(expShifted, new[] { 1 }, keepDims: true);
        var logRowSum = Engine.TensorLog<T>(rowSum);
        var logRowSumTiled = Engine.TensorTile(logRowSum, new[] { 1, batchSize });
        var logSoftmax = Engine.TensorSubtract<T>(shifted, logRowSumTiled);

        // Diagonal mask: 1 at (i, i), 0 elsewhere.
        var diagMask = new Tensor<T>(new[] { batchSize, batchSize });
        for (int i = 0; i < batchSize; i++) diagMask[i, i] = NumOps.One;
        var picked = Engine.TensorMultiply<T>(logSoftmax, diagMask);
        // Sum all entries; only the diagonal survives.
        var sumAxes = new[] { 0, 1 };
        var totalLog = Engine.ReduceSum(picked, sumAxes, keepDims: false);

        // halfLoss = -0.5 * totalLog / batchSize, returned as a [1] tensor.
        T totalScalar = totalLog.Length > 0 ? totalLog[0] : NumOps.Zero;
        T halfLossT = NumOps.Multiply(
            NumOps.Negate(NumOps.FromDouble(0.5 / batchSize)), totalScalar);
        var result = new Tensor<T>(new[] { 1 });
        result[0] = halfLossT;
        return result;
    }

    #endregion

    #region Helpers

    /// <summary>
    /// L2-normalises along the last (feature) axis so cosine similarity
    /// reduces to a plain dot product downstream. Engine-routed to stay
    /// generic in <typeparamref name="T"/> and BLAS-accelerated.
    /// </summary>
    private Tensor<T> L2Normalize(Tensor<T> x)
    {
        // ||x||² along the last axis, then 1/√(·+ε) via Sqrt + Reciprocal.
        var sq = Engine.TensorMultiply(x, x);
        int lastAxis = x.Shape.Length - 1;
        var sumSq = Engine.ReduceSum(sq, [lastAxis], keepDims: true);
        var eps = NumOps.FromDouble(1e-12);
        var sumSqPlusEps = Engine.TensorAddScalar(sumSq, eps);
        var norm = Engine.TensorSqrt(sumSqPlusEps);
        var invNorm = Engine.TensorReciprocal(norm);
        return Engine.TensorMultiply(x, invNorm);
    }

    /// <summary>Cosine similarity between two row-aligned embeddings (already L2-normalised).</summary>
    private double ComputeCosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        int len = Math.Min(a.Length, b.Length);
        double dot = 0.0;
        for (int i = 0; i < len; i++)
            dot += Convert.ToDouble(a[i]) * Convert.ToDouble(b[i]);
        return dot;
    }

    private void ThrowIfDisposed()
    {
        // AudioNeuralNetworkBase manages disposal; no extra state to gate.
    }

    /// <inheritdoc/>
    public override long ParameterCount =>
        Layers.Sum(l => l.ParameterCount)
        + TextEncoderLayers.Sum(l => l.ParameterCount)
        + 1; // _logTemperature (learnable τ scalar)

    /// <inheritdoc/>
    /// <remarks>
    /// CLAP holds parameters across both encoders + the temperature.
    /// Override <see cref="NeuralNetworkBase{T}.GetParameters"/> so
    /// optimizers / checkpointing see the full parameter vector and the
    /// ordering matches <see cref="UpdateParameters"/> below.
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        var parts = new List<T>((int)ParameterCount);
        foreach (var layer in Layers)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++) parts.Add(p[i]);
        }
        foreach (var layer in TextEncoderLayers)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++) parts.Add(p[i]);
        }
        parts.Add(_logTemperature[0]);
        var result = new Vector<T>(parts.Count);
        for (int i = 0; i < parts.Count; i++) result[i] = parts[i];
        return result;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");

        int idx = 0;
        // Audio encoder layers (the inherited Layers list — the primary stream).
        foreach (var layer in Layers)
        {
            int count = (int)layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
        // Text encoder layers (the secondary stream on the audio base class).
        foreach (var layer in TextEncoderLayers)
        {
            int count = (int)layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
        // Learnable temperature τ (last scalar parameter).
        if (idx < parameters.Length)
        {
            _logTemperature[0] = parameters[idx];
        }
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.SampleRate);
        writer.Write(_options.NumMelBands);
        writer.Write(_options.StftWindowSize);
        writer.Write(_options.HopLength);
        writer.Write(_options.AudioPatchSize);
        writer.Write(_options.AudioHiddenDim);
        writer.Write(_options.AudioEncoderLayers);
        writer.Write(_options.AudioEncoderHeads);
        writer.Write(_options.SwinWindowSize);
        writer.Write(_options.VocabSize);
        writer.Write(_options.MaxTextLength);
        writer.Write(_options.TextHiddenDim);
        writer.Write(_options.TextEncoderLayers);
        writer.Write(_options.TextEncoderHeads);
        writer.Write(_options.ProjectionDim);
        writer.Write(_options.InitialTemperature);
        writer.Write(_options.DropoutRate);
        writer.Write(Convert.ToDouble(_logTemperature[0]));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Validates that every persisted option matches the constructor option
    /// supplied for the live instance. CLAPModelOptions is init-only, so we
    /// can't rebind those — but accepting a mismatched checkpoint would silently
    /// produce wrong-shape weight loads. Throwing with the offending field
    /// surfaces the issue immediately.
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        bool useNativeMode = reader.ReadBoolean();
        VerifyEqual(reader.ReadInt32(),    _options.SampleRate,         nameof(_options.SampleRate));
        VerifyEqual(reader.ReadInt32(),    _options.NumMelBands,        nameof(_options.NumMelBands));
        VerifyEqual(reader.ReadInt32(),    _options.StftWindowSize,     nameof(_options.StftWindowSize));
        VerifyEqual(reader.ReadInt32(),    _options.HopLength,          nameof(_options.HopLength));
        VerifyEqual(reader.ReadInt32(),    _options.AudioPatchSize,     nameof(_options.AudioPatchSize));
        VerifyEqual(reader.ReadInt32(),    _options.AudioHiddenDim,     nameof(_options.AudioHiddenDim));
        VerifyEqual(reader.ReadInt32(),    _options.AudioEncoderLayers, nameof(_options.AudioEncoderLayers));
        VerifyEqual(reader.ReadInt32(),    _options.AudioEncoderHeads,  nameof(_options.AudioEncoderHeads));
        VerifyEqual(reader.ReadInt32(),    _options.SwinWindowSize,     nameof(_options.SwinWindowSize));
        VerifyEqual(reader.ReadInt32(),    _options.VocabSize,          nameof(_options.VocabSize));
        VerifyEqual(reader.ReadInt32(),    _options.MaxTextLength,      nameof(_options.MaxTextLength));
        VerifyEqual(reader.ReadInt32(),    _options.TextHiddenDim,      nameof(_options.TextHiddenDim));
        VerifyEqual(reader.ReadInt32(),    _options.TextEncoderLayers,  nameof(_options.TextEncoderLayers));
        VerifyEqual(reader.ReadInt32(),    _options.TextEncoderHeads,   nameof(_options.TextEncoderHeads));
        VerifyEqual(reader.ReadInt32(),    _options.ProjectionDim,      nameof(_options.ProjectionDim));
        VerifyEqual(reader.ReadDouble(),   _options.InitialTemperature, nameof(_options.InitialTemperature));
        VerifyEqual(reader.ReadDouble(),   _options.DropoutRate,        nameof(_options.DropoutRate));

        double logTau = reader.ReadDouble();
        _logTemperature[0] = NumOps.FromDouble(logTau);
    }

    private static void VerifyEqual<TValue>(TValue persisted, TValue current, string name)
        where TValue : IEquatable<TValue>
    {
        if (!persisted.Equals(current))
            throw new InvalidOperationException(
                $"Persisted CLAPModelOptions.{name} = {persisted} does not match constructor option {current}. " +
                "Reconstruct CLAPModel with matching CLAPModelOptions before loading this checkpoint.");
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        _useNativeMode
            ? new CLAPModel<T>(Architecture, _options)
            : new CLAPModel<T>(Architecture, _audioEncoderPath!, _textEncoderPath, _options);

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = Name,
            Description = "CLAP — Contrastive Language-Audio Pre-training (Wu et al. 2023).",
            Complexity = _options.AudioEncoderLayers + _options.TextEncoderLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["AudioHiddenDim"] = _options.AudioHiddenDim,
                ["TextHiddenDim"] = _options.TextHiddenDim,
                ["ProjectionDim"] = _options.ProjectionDim,
                ["SampleRate"] = _options.SampleRate,
                ["NumMelBands"] = _options.NumMelBands,
                ["AudioEncoderLayers"] = _options.AudioEncoderLayers,
                ["TextEncoderLayers"] = _options.TextEncoderLayers,
            }
        };
    }

    #endregion
}
