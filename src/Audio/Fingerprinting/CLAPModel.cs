using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// CLAP (Contrastive Language-Audio Pretraining) - A neural network model that learns to align
/// audio and text representations in a shared embedding space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLAP is a multimodal model trained using contrastive learning to create embeddings where
/// similar audio-text pairs are close together and dissimilar pairs are far apart. This enables:
/// <list type="bullet">
/// <item><description>Zero-shot audio classification using text prompts</description></item>
/// <item><description>Audio-to-text retrieval (find descriptions matching audio)</description></item>
/// <item><description>Text-to-audio retrieval (find audio matching descriptions)</description></item>
/// <item><description>Semantic audio fingerprinting</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CLAP understands both audio and text! It can:
///
/// - Tell you what's in an audio clip without pre-defined categories
/// - Find audio that matches a text description ("a dog barking in the rain")
/// - Create embeddings for audio search and recommendation
///
/// Unlike traditional fingerprinting that matches exact audio, CLAP understands
/// audio semantics - it knows a dog barking and a recording of barking are related!
///
/// Example use cases:
/// - "Is this audio of a happy or sad scene?" (sentiment analysis)
/// - "Find all audio clips with birds singing" (content-based search)
/// - "Classify this sound into one of these categories: ..." (zero-shot classification)
/// - Audio content moderation (detect specific sounds)
/// </para>
/// <para>
/// Reference: Wu, Y., et al. (2023). Large-scale Contrastive Language-Audio Pretraining
/// with Feature Fusion and Keyword-to-Caption Augmentation.
/// </para>
/// </remarks>
public class CLAPModel<T> : AudioNeuralNetworkBase<T>, IAudioFingerprinter<T>
{
    private readonly INumericOperations<T> _numOps;

    // Model configuration
    private readonly int _embeddingDim;
    private readonly int _projectionDim;
    private readonly int _numMelBands;
    private readonly int _windowSize;
    private readonly int _hopSize;

    // Audio encoder
    private readonly int _audioEncoderLayers;
    private readonly int _audioEncoderHeads;
    private readonly int _audioHiddenDim;

    // Audio encoder weights
    private T[] _audioPatchEmbedWeight;
    private T[] _audioPatchEmbedBias;
    private T[] _audioPositionalEncoding;
    private T[] _audioClsToken;
    private readonly List<TransformerLayer> _audioTransformerLayers;
    private T[] _audioProjectionWeight;
    private T[] _audioProjectionBias;

    // Text encoder (simplified - typically uses BERT/RoBERTa)
    private T[] _textEmbeddingWeight;
    private T[] _textPositionalEncoding;
    private readonly List<TransformerLayer> _textTransformerLayers;
    private T[] _textProjectionWeight;
    private T[] _textProjectionBias;

    // Temperature parameter for contrastive learning
    private T _temperature;

    // Vocabulary size for text tokenization
    private readonly int _vocabSize;
    private readonly int _maxTextLength;

    // Optimizer for training
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <inheritdoc/>
    public string Name => "CLAP";

    /// <inheritdoc/>
    public int FingerprintLength => _projectionDim;

    /// <summary>
    /// Gets the embedding dimension used internally.
    /// </summary>
    public int EmbeddingDimension => _embeddingDim;

    /// <summary>
    /// Gets the projection dimension (final embedding size).
    /// </summary>
    public int ProjectionDimension => _projectionDim;

    /// <summary>
    /// Initializes a new instance of the <see cref="CLAPModel{T}"/> class for ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="audioEncoderPath">Path to the ONNX audio encoder model.</param>
    /// <param name="textEncoderPath">Optional path to the ONNX text encoder model.</param>
    /// <param name="sampleRate">Sample rate of input audio (default: 48000 Hz).</param>
    /// <param name="embeddingDim">Embedding dimension (default: 768).</param>
    /// <param name="projectionDim">Projection dimension for output embeddings (default: 512).</param>
    /// <param name="onnxOptions">Optional ONNX model options.</param>
    /// <exception cref="FileNotFoundException">Thrown when the ONNX model file is not found.</exception>
    public CLAPModel(
        NeuralNetworkArchitecture<T> architecture,
        string audioEncoderPath,
        string? textEncoderPath = null,
        int sampleRate = 48000,
        int embeddingDim = 768,
        int projectionDim = 512,
        OnnxModelOptions? onnxOptions = null)
        : base(architecture)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        if (string.IsNullOrWhiteSpace(audioEncoderPath))
        {
            throw new ArgumentException("Audio encoder path cannot be null or empty.", nameof(audioEncoderPath));
        }

        if (!File.Exists(audioEncoderPath))
        {
            throw new FileNotFoundException($"Audio encoder ONNX model not found: {audioEncoderPath}", audioEncoderPath);
        }

        SampleRate = sampleRate;
        _embeddingDim = embeddingDim;
        _projectionDim = projectionDim;
        _numMelBands = 64;
        _windowSize = 1024;
        _hopSize = 480;

        // Load ONNX models
        OnnxEncoder = new OnnxModel<T>(audioEncoderPath, onnxOptions);
        if (textEncoderPath is not null && File.Exists(textEncoderPath))
        {
            OnnxDecoder = new OnnxModel<T>(textEncoderPath, onnxOptions);
        }

        // Initialize empty arrays (not used in ONNX mode)
        _audioPatchEmbedWeight = Array.Empty<T>();
        _audioPatchEmbedBias = Array.Empty<T>();
        _audioPositionalEncoding = Array.Empty<T>();
        _audioClsToken = Array.Empty<T>();
        _audioProjectionWeight = Array.Empty<T>();
        _audioProjectionBias = Array.Empty<T>();
        _textEmbeddingWeight = Array.Empty<T>();
        _textPositionalEncoding = Array.Empty<T>();
        _textProjectionWeight = Array.Empty<T>();
        _textProjectionBias = Array.Empty<T>();
        _audioTransformerLayers = new List<TransformerLayer>();
        _textTransformerLayers = new List<TransformerLayer>();
        _temperature = _numOps.FromDouble(0.07);
        _vocabSize = 49408;
        _maxTextLength = 77;
        _audioEncoderLayers = 12;
        _audioEncoderHeads = 12;
        _audioHiddenDim = embeddingDim;

        // Initialize optimizer (not used in ONNX mode but required for readonly field)
        _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CLAPModel{T}"/> class for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="sampleRate">Sample rate of input audio (default: 48000 Hz).</param>
    /// <param name="embeddingDim">Internal embedding dimension (default: 768).</param>
    /// <param name="projectionDim">Projection dimension for output embeddings (default: 512).</param>
    /// <param name="numMelBands">Number of mel spectrogram bands (default: 64).</param>
    /// <param name="audioEncoderLayers">Number of transformer layers in audio encoder (default: 12).</param>
    /// <param name="audioEncoderHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size for text encoding (default: 49408).</param>
    /// <param name="maxTextLength">Maximum text sequence length (default: 77).</param>
    /// <param name="windowSize">STFT window size (default: 1024).</param>
    /// <param name="hopSize">STFT hop size (default: 480).</param>
    /// <param name="temperature">Temperature for contrastive loss (default: 0.07).</param>
    /// <param name="optimizer">Optimizer for training. If null, a default Adam optimizer is used.</param>
    /// <param name="lossFunction">Loss function. If null, contrastive loss is used.</param>
    public CLAPModel(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 48000,
        int embeddingDim = 768,
        int projectionDim = 512,
        int numMelBands = 64,
        int audioEncoderLayers = 12,
        int audioEncoderHeads = 12,
        int vocabSize = 49408,
        int maxTextLength = 77,
        int windowSize = 1024,
        int hopSize = 480,
        double temperature = 0.07,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        SampleRate = sampleRate;
        _embeddingDim = embeddingDim;
        _projectionDim = projectionDim;
        _numMelBands = numMelBands;
        _audioEncoderLayers = audioEncoderLayers;
        _audioEncoderHeads = audioEncoderHeads;
        _audioHiddenDim = embeddingDim;
        _vocabSize = vocabSize;
        _maxTextLength = maxTextLength;
        _windowSize = windowSize;
        _hopSize = hopSize;
        _temperature = _numOps.FromDouble(temperature);

        // Initialize audio encoder components
        int patchSize = 16;
        int numPatches = (numMelBands / patchSize) * 50; // Approximate for 10s audio

        _audioPatchEmbedWeight = InitializeWeights(embeddingDim * patchSize * patchSize);
        _audioPatchEmbedBias = InitializeWeights(embeddingDim, 0.0);
        _audioPositionalEncoding = InitializeWeights((numPatches + 1) * embeddingDim);
        _audioClsToken = InitializeWeights(embeddingDim);

        _audioTransformerLayers = new List<TransformerLayer>();
        for (int i = 0; i < audioEncoderLayers; i++)
        {
            _audioTransformerLayers.Add(new TransformerLayer(_numOps, embeddingDim, audioEncoderHeads));
        }

        _audioProjectionWeight = InitializeWeights(projectionDim * embeddingDim);
        _audioProjectionBias = InitializeWeights(projectionDim, 0.0);

        // Initialize text encoder components
        _textEmbeddingWeight = InitializeWeights(vocabSize * embeddingDim);
        _textPositionalEncoding = InitializeWeights(maxTextLength * embeddingDim);

        _textTransformerLayers = new List<TransformerLayer>();
        for (int i = 0; i < audioEncoderLayers; i++)
        {
            _textTransformerLayers.Add(new TransformerLayer(_numOps, embeddingDim, audioEncoderHeads));
        }

        _textProjectionWeight = InitializeWeights(projectionDim * embeddingDim);
        _textProjectionBias = InitializeWeights(projectionDim, 0.0);

        // Initialize optimizer (Adam by default)
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        // Layers are handled manually for CLAP's dual-encoder architecture
    }

    /// <summary>
    /// Preprocesses raw audio waveform for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Convert to mel spectrogram
        int numSamples = rawAudio.Shape[^1];
        int numFrames = (numSamples - _windowSize) / _hopSize + 1;

        var melSpec = new T[numFrames * _numMelBands];

        // Simplified mel spectrogram computation
        for (int f = 0; f < numFrames; f++)
        {
            int start = f * _hopSize;
            for (int m = 0; m < _numMelBands; m++)
            {
                double sum = 0;
                for (int i = 0; i < Math.Min(_windowSize, numSamples - start); i++)
                {
                    int idx = start + i;
                    if (idx < rawAudio.Length)
                    {
                        double sample = _numOps.ToDouble(rawAudio.Data.Span[idx]);
                        // Simplified mel bin calculation
                        double melWeight = Math.Exp(-Math.Pow(m - (double)i / _windowSize * _numMelBands, 2) / 10);
                        sum += Math.Abs(sample) * melWeight;
                    }
                }
                melSpec[f * _numMelBands + m] = _numOps.FromDouble(Math.Log(sum + 1e-7));
            }
        }

        return new Tensor<T>(melSpec, new[] { 1, numFrames, _numMelBands });
    }

    /// <summary>
    /// Postprocesses model output.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // L2 normalize the embeddings
        return NormalizeEmbeddings(modelOutput);
    }

    /// <summary>
    /// Encodes audio into an embedding vector.
    /// </summary>
    /// <param name="audio">Audio tensor [samples] or [batch, samples].</param>
    /// <returns>Audio embedding [batch, projectionDim].</returns>
    public Tensor<T> EncodeAudio(Tensor<T> audio)
    {
        var preprocessed = PreprocessAudio(audio);

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            var output = OnnxEncoder.Run(preprocessed);
            return PostprocessOutput(output);
        }

        return EncodeAudioNative(preprocessed);
    }

    /// <summary>
    /// Encodes text into an embedding vector.
    /// </summary>
    /// <param name="tokens">Text token IDs [batch, seqLen].</param>
    /// <returns>Text embedding [batch, projectionDim].</returns>
    public Tensor<T> EncodeText(int[] tokens)
    {
        if (IsOnnxMode && OnnxDecoder is not null)
        {
            var tokenTensor = new Tensor<T>(
                tokens.Select(t => _numOps.FromDouble(t)).ToArray(),
                new[] { 1, tokens.Length });
            var output = OnnxDecoder.Run(tokenTensor);
            return PostprocessOutput(output);
        }

        return EncodeTextNative(tokens);
    }

    /// <summary>
    /// Native audio encoding implementation.
    /// </summary>
    private Tensor<T> EncodeAudioNative(Tensor<T> melSpec)
    {
        int batchSize = melSpec.Shape[0];
        int numFrames = melSpec.Shape[1];
        int numMels = melSpec.Shape[2];

        // Patch embedding (simplified)
        int patchSize = 16;
        int numPatchesH = numFrames / patchSize;
        int numPatchesW = numMels / patchSize;
        int numPatches = Math.Max(1, numPatchesH * numPatchesW);

        var patches = new T[batchSize * (numPatches + 1) * _embeddingDim]; // +1 for CLS token

        // Add CLS token
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < _embeddingDim; d++)
            {
                int idx = b * (numPatches + 1) * _embeddingDim + d;
                patches[idx] = d < _audioClsToken.Length ? _audioClsToken[d] : _numOps.Zero;
            }
        }

        // Embed patches
        for (int b = 0; b < batchSize; b++)
        {
            int patchIdx = 0;
            for (int ph = 0; ph < numPatchesH && patchIdx < numPatches; ph++)
            {
                for (int pw = 0; pw < numPatchesW && patchIdx < numPatches; pw++)
                {
                    for (int d = 0; d < _embeddingDim; d++)
                    {
                        T sum = d < _audioPatchEmbedBias.Length ? _audioPatchEmbedBias[d] : _numOps.Zero;
                        for (int i = 0; i < patchSize && ph * patchSize + i < numFrames; i++)
                        {
                            for (int j = 0; j < patchSize && pw * patchSize + j < numMels; j++)
                            {
                                int specIdx = b * numFrames * numMels +
                                             (ph * patchSize + i) * numMels +
                                             (pw * patchSize + j);
                                int wIdx = d * patchSize * patchSize + i * patchSize + j;
                                if (specIdx < melSpec.Length && wIdx < _audioPatchEmbedWeight.Length)
                                {
                                    sum = _numOps.Add(sum, _numOps.Multiply(
                                        melSpec.Data.Span[specIdx],
                                        _audioPatchEmbedWeight[wIdx]));
                                }
                            }
                        }
                        int outIdx = b * (numPatches + 1) * _embeddingDim +
                                     (patchIdx + 1) * _embeddingDim + d;
                        patches[outIdx] = sum;
                    }
                    patchIdx++;
                }
            }
        }

        // Add positional encoding
        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < numPatches + 1; p++)
            {
                for (int d = 0; d < _embeddingDim; d++)
                {
                    int idx = b * (numPatches + 1) * _embeddingDim + p * _embeddingDim + d;
                    int posIdx = p * _embeddingDim + d;
                    if (posIdx < _audioPositionalEncoding.Length)
                    {
                        patches[idx] = _numOps.Add(patches[idx], _audioPositionalEncoding[posIdx]);
                    }
                }
            }
        }

        var hidden = new Tensor<T>(patches, new[] { batchSize, numPatches + 1, _embeddingDim });

        // Transformer layers
        foreach (var layer in _audioTransformerLayers)
        {
            hidden = layer.Forward(hidden);
        }

        // Extract CLS token and project
        var clsTokens = new T[batchSize * _embeddingDim];
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < _embeddingDim; d++)
            {
                int idx = b * (numPatches + 1) * _embeddingDim + d;
                clsTokens[b * _embeddingDim + d] = hidden.Data.Span[idx];
            }
        }

        // Project to embedding space
        var embeddings = new T[batchSize * _projectionDim];
        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < _projectionDim; p++)
            {
                T sum = p < _audioProjectionBias.Length ? _audioProjectionBias[p] : _numOps.Zero;
                for (int d = 0; d < _embeddingDim; d++)
                {
                    int wIdx = p * _embeddingDim + d;
                    if (wIdx < _audioProjectionWeight.Length)
                    {
                        sum = _numOps.Add(sum, _numOps.Multiply(
                            clsTokens[b * _embeddingDim + d],
                            _audioProjectionWeight[wIdx]));
                    }
                }
                embeddings[b * _projectionDim + p] = sum;
            }
        }

        var result = new Tensor<T>(embeddings, new[] { batchSize, _projectionDim });
        return NormalizeEmbeddings(result);
    }

    /// <summary>
    /// Native text encoding implementation.
    /// </summary>
    private Tensor<T> EncodeTextNative(int[] tokens)
    {
        int seqLen = Math.Min(tokens.Length, _maxTextLength);

        // Token embedding
        var embedded = new T[seqLen * _embeddingDim];
        for (int i = 0; i < seqLen; i++)
        {
            int tokenId = tokens[i];
            for (int d = 0; d < _embeddingDim; d++)
            {
                int embIdx = tokenId * _embeddingDim + d;
                int posIdx = i * _embeddingDim + d;
                T tokenEmbed = embIdx < _textEmbeddingWeight.Length
                    ? _textEmbeddingWeight[embIdx]
                    : _numOps.Zero;
                T posEnc = posIdx < _textPositionalEncoding.Length
                    ? _textPositionalEncoding[posIdx]
                    : _numOps.Zero;
                embedded[i * _embeddingDim + d] = _numOps.Add(tokenEmbed, posEnc);
            }
        }

        var hidden = new Tensor<T>(embedded, new[] { 1, seqLen, _embeddingDim });

        // Transformer layers
        foreach (var layer in _textTransformerLayers)
        {
            hidden = layer.Forward(hidden);
        }

        // Use last token (EOS) for pooling
        var lastToken = new T[_embeddingDim];
        for (int d = 0; d < _embeddingDim; d++)
        {
            int idx = (seqLen - 1) * _embeddingDim + d;
            lastToken[d] = hidden.Data.Span[idx];
        }

        // Project to embedding space
        var embedding = new T[_projectionDim];
        for (int p = 0; p < _projectionDim; p++)
        {
            T sum = p < _textProjectionBias.Length ? _textProjectionBias[p] : _numOps.Zero;
            for (int d = 0; d < _embeddingDim; d++)
            {
                int wIdx = p * _embeddingDim + d;
                if (wIdx < _textProjectionWeight.Length)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(lastToken[d], _textProjectionWeight[wIdx]));
                }
            }
            embedding[p] = sum;
        }

        var result = new Tensor<T>(embedding, new[] { 1, _projectionDim });
        return NormalizeEmbeddings(result);
    }

    /// <summary>
    /// Performs zero-shot classification using text prompts.
    /// </summary>
    /// <param name="audio">Audio tensor to classify.</param>
    /// <param name="classLabels">Array of text labels to classify against.</param>
    /// <param name="tokenizer">Function to tokenize text labels.</param>
    /// <returns>Classification probabilities for each label.</returns>
    public Dictionary<string, double> ZeroShotClassify(
        Tensor<T> audio,
        string[] classLabels,
        Func<string, int[]> tokenizer)
    {
        // Encode audio
        var audioEmbedding = EncodeAudio(audio);

        // Encode all text labels
        var textEmbeddings = new List<Tensor<T>>();
        foreach (var label in classLabels)
        {
            var tokens = tokenizer(label);
            textEmbeddings.Add(EncodeText(tokens));
        }

        // Compute cosine similarities
        var similarities = new double[classLabels.Length];
        for (int i = 0; i < classLabels.Length; i++)
        {
            similarities[i] = ComputeCosineSimilarity(audioEmbedding, textEmbeddings[i]);
        }

        // Apply softmax with temperature
        double temp = _numOps.ToDouble(_temperature);
        double maxSim = similarities.Max();
        double sumExp = 0;
        for (int i = 0; i < similarities.Length; i++)
        {
            sumExp += Math.Exp((similarities[i] - maxSim) / temp);
        }

        var results = new Dictionary<string, double>();
        for (int i = 0; i < classLabels.Length; i++)
        {
            results[classLabels[i]] = Math.Exp((similarities[i] - maxSim) / temp) / sumExp;
        }

        return results;
    }

    /// <summary>
    /// Computes cosine similarity between two embeddings.
    /// </summary>
    private double ComputeCosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        double dot = 0;
        double normA = 0;
        double normB = 0;

        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            double valA = _numOps.ToDouble(a.Data.Span[i]);
            double valB = _numOps.ToDouble(b.Data.Span[i]);
            dot += valA * valB;
            normA += valA * valA;
            normB += valB * valB;
        }

        if (normA < 1e-10 || normB < 1e-10) return 0;
        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    /// <summary>
    /// L2 normalizes embeddings.
    /// </summary>
    private Tensor<T> NormalizeEmbeddings(Tensor<T> embeddings)
    {
        int batchSize = embeddings.Shape.Length > 1 ? embeddings.Shape[0] : 1;
        int dim = embeddings.Shape[^1];

        var normalized = new T[embeddings.Length];

        for (int b = 0; b < batchSize; b++)
        {
            double norm = 0;
            for (int d = 0; d < dim; d++)
            {
                int idx = b * dim + d;
                double val = _numOps.ToDouble(embeddings.Data.Span[idx]);
                norm += val * val;
            }
            norm = Math.Sqrt(norm + 1e-10);

            for (int d = 0; d < dim; d++)
            {
                int idx = b * dim + d;
                normalized[idx] = _numOps.FromDouble(_numOps.ToDouble(embeddings.Data.Span[idx]) / norm);
            }
        }

        return new Tensor<T>(normalized, embeddings.Shape);
    }

    /// <summary>
    /// Predicts audio embedding.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return EncodeAudio(input);
    }

    #region IAudioFingerprinter Implementation

    /// <inheritdoc/>
    public AudioFingerprint<T> Fingerprint(Tensor<T> audio)
    {
        var embedding = EncodeAudio(audio);
        double duration = (double)audio.Shape[^1] / SampleRate;

        return new AudioFingerprint<T>
        {
            Data = embedding.Data.ToArray(),
            Duration = duration,
            SampleRate = SampleRate,
            Algorithm = Name,
            FrameCount = 1, // CLAP produces a single embedding
            Metadata = new Dictionary<string, object>
            {
                { "EmbeddingDim", _projectionDim },
                { "ModelType", "CLAP" }
            }
        };
    }

    /// <inheritdoc/>
    public AudioFingerprint<T> Fingerprint(Vector<T> audio)
    {
        var tensor = new Tensor<T>(audio.ToArray(), new[] { audio.Length });
        return Fingerprint(tensor);
    }

    /// <inheritdoc/>
    public double ComputeSimilarity(AudioFingerprint<T> fp1, AudioFingerprint<T> fp2)
    {
        var tensor1 = new Tensor<T>(fp1.Data.ToArray(), new[] { fp1.Data.Length });
        var tensor2 = new Tensor<T>(fp2.Data.ToArray(), new[] { fp2.Data.Length });
        return ComputeCosineSimilarity(tensor1, tensor2);
    }

    /// <inheritdoc/>
    public IReadOnlyList<FingerprintMatch> FindMatches(
        AudioFingerprint<T> query,
        AudioFingerprint<T> reference,
        int minMatchLength = 10)
    {
        // CLAP produces single embeddings, so we can only do whole-audio matching
        double similarity = ComputeSimilarity(query, reference);

        if (similarity > 0.5) // Threshold for considering a match
        {
            return new List<FingerprintMatch>
            {
                new FingerprintMatch
                {
                    QueryStartTime = 0,
                    ReferenceStartTime = 0,
                    Duration = Math.Min(query.Duration, reference.Duration),
                    Confidence = similarity,
                    MatchCount = 1
                }
            };
        }

        return new List<FingerprintMatch>();
    }

    #endregion

    #region Training

    /// <summary>
    /// Trains the model on audio-text pairs using contrastive loss.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
        {
            throw new InvalidOperationException("Cannot train in ONNX inference mode.");
        }

        // For CLAP, expected should contain text embeddings
        // This is a simplified training loop
        SetTrainingMode(true);

        var audioEmbeddings = EncodeAudioNative(PreprocessAudio(input));

        // Compute contrastive loss
        // In practice, this would need matching text embeddings
        var loss = ComputeContrastiveLoss(audioEmbeddings, expected);

        SetTrainingMode(false);
    }

    private T ComputeContrastiveLoss(Tensor<T> audioEmb, Tensor<T> textEmb)
    {
        // InfoNCE loss
        int batchSize = audioEmb.Shape[0];
        double totalLoss = 0;
        double temp = _numOps.ToDouble(_temperature);

        for (int i = 0; i < batchSize; i++)
        {
            double positiveScore = 0;
            double negativeSum = 0;

            for (int d = 0; d < _projectionDim; d++)
            {
                int aIdx = i * _projectionDim + d;
                int tIdx = i * _projectionDim + d; // Positive pair
                if (aIdx < audioEmb.Length && tIdx < textEmb.Length)
                {
                    positiveScore += _numOps.ToDouble(audioEmb.Data.Span[aIdx]) *
                                    _numOps.ToDouble(textEmb.Data.Span[tIdx]);
                }
            }

            for (int j = 0; j < batchSize; j++)
            {
                double score = 0;
                for (int d = 0; d < _projectionDim; d++)
                {
                    int aIdx = i * _projectionDim + d;
                    int tIdx = j * _projectionDim + d;
                    if (aIdx < audioEmb.Length && tIdx < textEmb.Length)
                    {
                        score += _numOps.ToDouble(audioEmb.Data.Span[aIdx]) *
                                _numOps.ToDouble(textEmb.Data.Span[tIdx]);
                    }
                }
                negativeSum += Math.Exp(score / temp);
            }

            totalLoss -= positiveScore / temp - Math.Log(negativeSum + 1e-10);
        }

        return _numOps.FromDouble(totalLoss / batchSize);
    }

    #endregion

    #region Serialization

    public override byte[] Serialize()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        writer.Write(SampleRate);
        writer.Write(_embeddingDim);
        writer.Write(_projectionDim);
        writer.Write(_numMelBands);
        writer.Write(_audioEncoderLayers);
        writer.Write(_vocabSize);

        WriteArray(writer, _audioPatchEmbedWeight);
        WriteArray(writer, _audioProjectionWeight);
        WriteArray(writer, _textEmbeddingWeight);
        WriteArray(writer, _textProjectionWeight);

        return stream.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        SampleRate = reader.ReadInt32();
        int embeddingDim = reader.ReadInt32();
        int projectionDim = reader.ReadInt32();
        int numMelBands = reader.ReadInt32();
        int audioEncoderLayers = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();

        _audioPatchEmbedWeight = ReadArray(reader);
        _audioProjectionWeight = ReadArray(reader);
        _textEmbeddingWeight = ReadArray(reader);
        _textProjectionWeight = ReadArray(reader);
    }

    private void WriteArray(BinaryWriter writer, T[] array)
    {
        writer.Write(array.Length);
        foreach (var val in array)
        {
            writer.Write(_numOps.ToDouble(val));
        }
    }

    private T[] ReadArray(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        var array = new T[len];
        for (int i = 0; i < len; i++)
        {
            array[i] = _numOps.FromDouble(reader.ReadDouble());
        }
        return array;
    }

    #endregion

    #region Helper Methods

    private T[] InitializeWeights(int size, double initValue = double.NaN)
    {
        var weights = new T[size];
        if (double.IsNaN(initValue))
        {
            double scale = Math.Sqrt(2.0 / size);
            var rand = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
            for (int i = 0; i < size; i++)
            {
                weights[i] = _numOps.FromDouble(rand.NextGaussian() * scale);
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                weights[i] = _numOps.FromDouble(initValue);
            }
        }
        return weights;
    }

    #endregion

    #region Abstract Method Implementations

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
        }

        // Get current parameters and apply gradient descent
        var currentParams = GetParameters();
        T learningRate = _numOps.FromDouble(0.0001);
        for (int i = 0; i < Math.Min(currentParams.Length, gradients.Length); i++)
        {
            currentParams[i] = _numOps.Subtract(currentParams[i], _numOps.Multiply(learningRate, gradients[i]));
        }
        SetParameters(currentParams);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "CLAP",
            Description = $"Contrastive Language-Audio Pretraining ({_embeddingDim}D embedding, {_projectionDim}D projection)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = SampleRate,
            Complexity = _audioEncoderLayers
        };
        metadata.AdditionalInfo["EmbeddingDim"] = _embeddingDim.ToString();
        metadata.AdditionalInfo["ProjectionDim"] = _projectionDim.ToString();
        metadata.AdditionalInfo["AudioEncoderLayers"] = _audioEncoderLayers.ToString();
        metadata.AdditionalInfo["Mode"] = IsOnnxMode ? "ONNX" : "Native";
        return metadata;
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(IsOnnxMode);
        writer.Write(SampleRate);
        writer.Write(_embeddingDim);
        writer.Write(_projectionDim);
        writer.Write(_numMelBands);
        writer.Write(_windowSize);
        writer.Write(_hopSize);
        writer.Write(_audioEncoderLayers);
        writer.Write(_audioEncoderHeads);
        writer.Write(_vocabSize);
        writer.Write(_maxTextLength);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadBoolean(); // IsOnnxMode
        _ = reader.ReadInt32();   // SampleRate
        _ = reader.ReadInt32();   // _embeddingDim
        _ = reader.ReadInt32();   // _projectionDim
        _ = reader.ReadInt32();   // _numMelBands
        _ = reader.ReadInt32();   // _windowSize
        _ = reader.ReadInt32();   // _hopSize
        _ = reader.ReadInt32();   // _audioEncoderLayers
        _ = reader.ReadInt32();   // _audioEncoderHeads
        _ = reader.ReadInt32();   // _vocabSize
        _ = reader.ReadInt32();   // _maxTextLength
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CLAPModel<T>(
            Architecture,
            sampleRate: SampleRate,
            embeddingDim: _embeddingDim,
            projectionDim: _projectionDim,
            numMelBands: _numMelBands,
            audioEncoderLayers: _audioEncoderLayers,
            audioEncoderHeads: _audioEncoderHeads,
            vocabSize: _vocabSize,
            maxTextLength: _maxTextLength,
            windowSize: _windowSize,
            hopSize: _hopSize);
    }

    #endregion

    #region Nested Types

    private class TransformerLayer
    {
        private readonly INumericOperations<T> _ops;
        private readonly int _dim;
        private readonly int _numHeads;
        private readonly int _headDim;

        private T[] _qWeight, _kWeight, _vWeight, _oWeight;
        private T[] _ffnW1, _ffnW2;
        private T[] _norm1Gamma, _norm1Beta, _norm2Gamma, _norm2Beta;

        public TransformerLayer(INumericOperations<T> ops, int dim, int numHeads)
        {
            _ops = ops;
            _dim = dim;
            _numHeads = numHeads;
            _headDim = dim / numHeads;

            var rand = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
            double scale = Math.Sqrt(2.0 / dim);

            _qWeight = InitWeights(dim * dim, rand, scale);
            _kWeight = InitWeights(dim * dim, rand, scale);
            _vWeight = InitWeights(dim * dim, rand, scale);
            _oWeight = InitWeights(dim * dim, rand, scale);

            int ffnDim = dim * 4;
            _ffnW1 = InitWeights(ffnDim * dim, rand, scale);
            _ffnW2 = InitWeights(dim * ffnDim, rand, scale);

            _norm1Gamma = Enumerable.Range(0, dim).Select(_ => _ops.FromDouble(1.0)).ToArray();
            _norm1Beta = new T[dim];
            _norm2Gamma = Enumerable.Range(0, dim).Select(_ => _ops.FromDouble(1.0)).ToArray();
            _norm2Beta = new T[dim];
        }

        private T[] InitWeights(int size, Random rand, double scale)
        {
            return Enumerable.Range(0, size).Select(_ => _ops.FromDouble(rand.NextGaussian() * scale)).ToArray();
        }

        public Tensor<T> Forward(Tensor<T> input)
        {
            // Simplified transformer layer
            int batchSize = input.Shape[0];
            int seqLen = input.Shape[1];
            int dim = input.Shape[2];

            // Self-attention (simplified)
            var attended = new T[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                attended[i] = input.Data.Span[i]; // Placeholder for actual attention
            }

            // Residual + LayerNorm
            var normed = new T[input.Length];
            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    double mean = 0, variance = 0;
                    for (int d = 0; d < dim; d++)
                    {
                        int idx = b * seqLen * dim + s * dim + d;
                        mean += _ops.ToDouble(attended[idx]);
                    }
                    mean /= dim;

                    for (int d = 0; d < dim; d++)
                    {
                        int idx = b * seqLen * dim + s * dim + d;
                        double diff = _ops.ToDouble(attended[idx]) - mean;
                        variance += diff * diff;
                    }
                    variance = Math.Sqrt(variance / dim + 1e-5);

                    for (int d = 0; d < dim; d++)
                    {
                        int idx = b * seqLen * dim + s * dim + d;
                        double val = (_ops.ToDouble(attended[idx]) - mean) / variance;
                        val = val * _ops.ToDouble(_norm1Gamma[d]) + _ops.ToDouble(_norm1Beta[d]);
                        normed[idx] = _ops.Add(input.Data.Span[idx], _ops.FromDouble(val));
                    }
                }
            }

            return new Tensor<T>(normed, input.Shape);
        }
    }

    #endregion
}
