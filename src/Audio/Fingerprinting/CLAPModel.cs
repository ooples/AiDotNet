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

    // Native-mode layer stacks. Both inherit from LayerBase<T> via LayerHelper.
    // ONNX mode leaves these empty and routes Predict through OnnxEncoder.
    private readonly List<ILayer<T>> _audioEncoder = new();
    private readonly List<ILayer<T>> _textEncoder = new();

    /// <summary>
    /// Trainable temperature parameter (stored in log space). Gradients flow
    /// through the tape via <see cref="LayerBase{T}.Engine"/> ops; the optimizer
    /// updates this alongside the rest of the network.
    /// </summary>
    private Tensor<T> _logTemperature = null!;

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
        OnnxEncoder = new OnnxModel<T>(audioEncoderPath);
        if (!string.IsNullOrWhiteSpace(textEncoderPath) && File.Exists(textEncoderPath))
            OnnxDecoder = new OnnxModel<T>(textEncoderPath);

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
    /// pattern: prefer user-provided <c>Architecture.Layers</c>; otherwise fall
    /// back to the paper-faithful <see cref="LayerHelper{T}"/> factories.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // User-supplied layer stack — first half audio, second half text by
            // convention. Callers can mix-and-match with LayerHelper helpers.
            Layers.AddRange(Architecture.Layers);
            ValidateLayerConfiguration(Layers);
            int half = Layers.Count / 2;
            _audioEncoder.Clear();
            _audioEncoder.AddRange(Layers.Take(half));
            _textEncoder.Clear();
            _textEncoder.AddRange(Layers.Skip(half));
        }
        else
        {
            // Default audio encoder: HTSAT (Swin) → mean-pool → proj.
            _audioEncoder.Clear();
            _audioEncoder.AddRange(LayerHelper<T>.CreateDefaultCLAPAudioEncoderLayers(
                audioHiddenDim: _options.AudioHiddenDim,
                audioEncoderLayers: _options.AudioEncoderLayers,
                audioEncoderHeads: _options.AudioEncoderHeads,
                swinWindowSize: _options.SwinWindowSize,
                projectionDim: _options.ProjectionDim));

            // Default text encoder: RoBERTa-style transformer → mean-pool → proj.
            _textEncoder.Clear();
            _textEncoder.AddRange(LayerHelper<T>.CreateDefaultCLAPTextEncoderLayers(
                vocabSize: _options.VocabSize,
                maxTextLength: _options.MaxTextLength,
                textHiddenDim: _options.TextHiddenDim,
                textEncoderLayers: _options.TextEncoderLayers,
                textEncoderHeads: _options.TextEncoderHeads,
                projectionDim: _options.ProjectionDim));

            // Register everything with the base class's Layers list so the
            // standard parameter / training infrastructure can walk it.
            Layers.AddRange(_audioEncoder);
            Layers.AddRange(_textEncoder);
        }

        // Learnable temperature τ in log space (CLIP / CLAP convention).
        // Initialised so exp(_logTemperature) = 1/InitialTemperature; the
        // contrastive loss uses logits·exp(_logTemperature).
        _logTemperature = new Tensor<T>([1]);
        _logTemperature[0] = NumOps.FromDouble(Math.Log(1.0 / _options.InitialTemperature));
    }

    /// <summary>
    /// Validates that user-supplied layers form a balanced audio + text stack.
    /// </summary>
    private static void ValidateLayerConfiguration(List<ILayer<T>> layers)
    {
        if (layers.Count < 4 || layers.Count % 2 != 0)
        {
            throw new ArgumentException(
                "CLAP custom layer stacks must contain an even number of layers " +
                "(>= 4) split equally between the audio encoder (first half) and " +
                "the text encoder (second half). Use LayerHelper.CreateDefault" +
                "CLAPAudioEncoderLayers + CreateDefaultCLAPTextEncoderLayers as a reference.",
                nameof(layers));
        }
    }

    #endregion

    #region Preprocessing — Mel Spectrogram

    /// <summary>
    /// Converts raw audio samples into a log-mel spectrogram. The pipeline
    /// (Hann window → real FFT → triangular mel filterbank → log) matches
    /// the librosa-style mel extraction CLAP §3.1 was trained on.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        int windowSize = _options.StftWindowSize;
        int hop = _options.HopLength;
        int numMels = _options.NumMelBands;
        int sampleRate = _options.SampleRate;

        if (rawAudio.Shape.Length == 1)
            rawAudio = Engine.Reshape(rawAudio, [1, rawAudio.Shape[0]]);

        int batchSize = rawAudio.Shape[0];
        int numSamples = rawAudio.Shape[1];
        int numFrames = Math.Max(1, (numSamples - windowSize) / hop + 1);

        var filterbank = ComputeMelFilterbank(windowSize, numMels, sampleRate);
        var melSpec = new Tensor<T>([batchSize, 1, numFrames, numMels]);
        var frameData = new double[windowSize];

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < numFrames; f++)
            {
                int start = f * hop;
                Array.Clear(frameData, 0, frameData.Length);
                for (int i = 0; i < windowSize; i++)
                {
                    int idx = start + i;
                    if (idx >= numSamples) break;
                    double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (windowSize - 1)));
                    frameData[i] = Convert.ToDouble(rawAudio[b, idx]) * w;
                }
                var spectrum = FftSharp.FFT.Forward(frameData);
                int specLen = windowSize / 2 + 1;
                for (int m = 0; m < numMels; m++)
                {
                    double energy = 0.0;
                    for (int k = 0; k < specLen; k++)
                        energy += spectrum[k].Magnitude * spectrum[k].Magnitude * filterbank[m, k];
                    melSpec[b, 0, f, m] = NumOps.FromDouble(Math.Log(Math.Max(energy, 1e-10)));
                }
            }
        }
        return melSpec;
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => L2Normalize(modelOutput);

    /// <summary>
    /// Triangular mel filterbank in the HTK-style mel scale (matches librosa
    /// defaults used by the published CLAP checkpoint).
    /// </summary>
    private static double[,] ComputeMelFilterbank(int windowSize, int numMelBands, int sampleRate)
    {
        int numBins = windowSize / 2 + 1;
        var filterbank = new double[numMelBands, numBins];
        double melMin = HzToMel(0);
        double melMax = HzToMel(sampleRate / 2.0);

        var melPoints = new double[numMelBands + 2];
        for (int i = 0; i < melPoints.Length; i++)
            melPoints[i] = melMin + (melMax - melMin) * i / (numMelBands + 1);

        var binPoints = new int[numMelBands + 2];
        for (int i = 0; i < melPoints.Length; i++)
            binPoints[i] = (int)Math.Floor((windowSize + 1) * MelToHz(melPoints[i]) / sampleRate);

        for (int m = 0; m < numMelBands; m++)
        {
            int s = binPoints[m], c = binPoints[m + 1], e = binPoints[m + 2];
            for (int k = s; k < c && k < numBins; k++)
                if (c != s) filterbank[m, k] = (double)(k - s) / (c - s);
            for (int k = c; k < e && k < numBins; k++)
                if (e != c) filterbank[m, k] = (double)(e - k) / (e - c);
        }
        return filterbank;
    }

    private static double HzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);
    private static double MelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

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

        // Native path: run the audio encoder layers in sequence, then L2 normalise.
        var hidden = preprocessed;
        foreach (var layer in _audioEncoder) hidden = layer.Forward(hidden);
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
        if (!_useNativeMode && OnnxDecoder is not null)
        {
            return PostprocessOutput(OnnxDecoder.Run(tokens));
        }

        if (tokens.Shape.Length == 1)
            tokens = Engine.Reshape(tokens, [1, tokens.Shape[0]]);

        var hidden = tokens;
        foreach (var layer in _textEncoder) hidden = layer.Forward(hidden);
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
        var embedding = EncodeAudio(audio);
        var flat = embedding.ToVector();
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

        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected);
        }
        finally
        {
            SetTrainingMode(false);
        }
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
        // Update the learnable temperature too (last parameter in the model).
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
        new CLAPModel<T>(Architecture, _options);

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
