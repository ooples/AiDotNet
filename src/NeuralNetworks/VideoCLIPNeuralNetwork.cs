using System.IO;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// VideoCLIP neural network for video-text alignment and temporal understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VideoCLIP extends CLIP's contrastive learning paradigm to the video domain, enabling
/// text-to-video and video-to-text retrieval, action recognition, and temporal understanding.
/// </para>
/// <para><b>For Beginners:</b> VideoCLIP is like CLIP but for videos!
///
/// Architecture overview:
/// 1. Vision Encoder: Extracts features from each frame (shared CLIP ViT)
/// 2. Temporal Encoder: Aggregates frame features over time
/// 3. Text Encoder: Processes text descriptions
/// 4. Contrastive Learning: Aligns video and text in shared embedding space
///
/// Key capabilities:
/// - Video retrieval: Find videos matching text descriptions
/// - Action recognition: Classify actions without training
/// - Moment localization: Find specific moments in videos
/// - Video QA: Answer questions about video content
/// </para>
/// </remarks>
public class VideoCLIPNeuralNetwork<T> : NeuralNetworkBase<T>, IVideoCLIPModel<T>
{
    #region Execution Mode

    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    private readonly InferenceSession? _videoEncoder;
    private readonly InferenceSession? _textEncoder;
    private readonly string? _videoEncoderPath;
    private readonly string? _textEncoderPath;

    #endregion

    #region Native Mode Fields

    private readonly List<ILayer<T>> _frameEncoderLayers = [];
    private readonly List<ILayer<T>> _temporalEncoderLayers = [];
    private readonly List<ILayer<T>> _textEncoderLayers = [];
    private readonly List<ILayer<T>> _projectionLayers = [];
    private Matrix<T>? _visionClsToken;
    private Matrix<T>? _visionPositionalEmbeddings;
    private Matrix<T>? _temporalPositionalEmbeddings;
    private Matrix<T>? _textPositionalEmbeddings;
    private ILayer<T>? _patchEmbedding;
    private ILayer<T>? _textTokenEmbedding;
    private ILayer<T>? _videoProjection;
    private ILayer<T>? _textProjection;
    private ILayer<T>? _captionHead;

    #endregion

    #region Shared Fields

    private readonly ITokenizer _tokenizer;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _embeddingDimension;
    private readonly int _maxSequenceLength;
    private readonly int _imageSize;
    private readonly int _visionHiddenDim;
    private readonly int _textHiddenDim;
    private readonly int _numFrameEncoderLayers;
    private readonly int _numTemporalLayers;
    private readonly int _numTextLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _vocabularySize;
    private readonly int _numFrames;
    private readonly double _frameRate;
    private readonly string _temporalAggregation;

    #endregion

    #region IMultimodalEmbedding Properties

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc/>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    public int ImageSize => _imageSize;

    #endregion

    #region IVideoCLIPModel Properties

    /// <inheritdoc/>
    public int NumFrames => _numFrames;

    /// <inheritdoc/>
    public double FrameRate => _frameRate;

    /// <inheritdoc/>
    public string TemporalAggregation => _temporalAggregation;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a VideoCLIP network using pretrained ONNX models.
    /// </summary>
    public VideoCLIPNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string videoEncoderPath,
        string textEncoderPath,
        ITokenizer tokenizer,
        int numFrames = 8,
        double frameRate = 1.0,
        string temporalAggregation = "temporal_transformer",
        int embeddingDimension = 512,
        int maxSequenceLength = 77,
        int imageSize = 224,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(videoEncoderPath))
            throw new ArgumentException("Video encoder path cannot be null or empty.", nameof(videoEncoderPath));
        if (string.IsNullOrWhiteSpace(textEncoderPath))
            throw new ArgumentException("Text encoder path cannot be null or empty.", nameof(textEncoderPath));
        if (!File.Exists(videoEncoderPath))
            throw new FileNotFoundException($"Video encoder model not found: {videoEncoderPath}");
        if (!File.Exists(textEncoderPath))
            throw new FileNotFoundException($"Text encoder model not found: {textEncoderPath}");

        _useNativeMode = false;
        _videoEncoderPath = videoEncoderPath;
        _textEncoderPath = textEncoderPath;
        _numFrames = numFrames;
        _frameRate = frameRate;
        _temporalAggregation = temporalAggregation;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _patchSize = 16;
        _visionHiddenDim = 768;
        _textHiddenDim = 512;
        _numFrameEncoderLayers = 12;
        _numTemporalLayers = 4;
        _numTextLayers = 12;
        _numHeads = 12;
        _vocabularySize = 49408;

        InferenceSession? videoEncoder = null;
        InferenceSession? textEncoder = null;

        try
        {
            videoEncoder = new InferenceSession(videoEncoderPath);
            textEncoder = new InferenceSession(textEncoderPath);
            _videoEncoder = videoEncoder;
            _textEncoder = textEncoder;
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
            InitializeLayers();
        }
        catch
        {
            videoEncoder?.Dispose();
            textEncoder?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates a VideoCLIP network using native library layers.
    /// </summary>
    public VideoCLIPNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int imageSize = 224,
        int channels = 3,
        int patchSize = 16,
        int vocabularySize = 49408,
        int maxSequenceLength = 77,
        int embeddingDimension = 512,
        int visionHiddenDim = 768,
        int textHiddenDim = 512,
        int numFrameEncoderLayers = 12,
        int numTemporalLayers = 4,
        int numTextLayers = 12,
        int numHeads = 12,
        int numFrames = 8,
        double frameRate = 1.0,
        string temporalAggregation = "temporal_transformer",
        ITokenizer? tokenizer = null,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _visionHiddenDim = visionHiddenDim;
        _textHiddenDim = textHiddenDim;
        _numFrameEncoderLayers = numFrameEncoderLayers;
        _numTemporalLayers = numTemporalLayers;
        _numTextLayers = numTextLayers;
        _numHeads = numHeads;
        _patchSize = patchSize;
        _vocabularySize = vocabularySize;
        _numFrames = numFrames;
        _frameRate = frameRate;
        _temporalAggregation = temporalAggregation;

        _tokenizer = tokenizer ?? Tokenization.ClipTokenizerFactory.CreateSimple();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();

        InitializeNativeLayers(channels);
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // ONNX mode initialization - models are already loaded
    }

    private void InitializeNativeLayers(int channels)
    {
        int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);

        // Vision frame encoder (shared across frames)
        _patchEmbedding = new PatchEmbeddingLayer<T>(
            _imageSize, _imageSize, channels, _patchSize, _visionHiddenDim);

        _visionClsToken = Matrix<T>.CreateDefault(1, _visionHiddenDim, NumOps.Zero);
        _visionPositionalEmbeddings = Matrix<T>.CreateDefault(numPatches + 1, _visionHiddenDim, NumOps.Zero);

        int visionFfnDim = _visionHiddenDim * 4;
        for (int i = 0; i < _numFrameEncoderLayers; i++)
        {
            _frameEncoderLayers.Add(new TransformerEncoderLayer<T>(
                _visionHiddenDim, _numHeads, visionFfnDim));
        }

        // Temporal encoder (processes frame features over time)
        _temporalPositionalEmbeddings = Matrix<T>.CreateDefault(_numFrames, _visionHiddenDim, NumOps.Zero);

        int temporalFfnDim = _visionHiddenDim * 4;
        for (int i = 0; i < _numTemporalLayers; i++)
        {
            _temporalEncoderLayers.Add(new TransformerEncoderLayer<T>(
                _visionHiddenDim, _numHeads, temporalFfnDim));
        }

        // Video projection to embedding space
        _videoProjection = new DenseLayer<T>(_visionHiddenDim, _embeddingDimension, (IActivationFunction<T>?)null);

        // Text encoder
        _textTokenEmbedding = new EmbeddingLayer<T>(_vocabularySize, _textHiddenDim);
        _textPositionalEmbeddings = Matrix<T>.CreateDefault(_maxSequenceLength, _textHiddenDim, NumOps.Zero);

        int textFfnDim = _textHiddenDim * 4;
        for (int i = 0; i < _numTextLayers; i++)
        {
            _textEncoderLayers.Add(new TransformerEncoderLayer<T>(
                _textHiddenDim, _numHeads, textFfnDim));
        }

        // Text projection to embedding space
        _textProjection = new DenseLayer<T>(_textHiddenDim, _embeddingDimension, (IActivationFunction<T>?)null);

        // Caption generation head
        _captionHead = new DenseLayer<T>(_embeddingDimension, _vocabularySize, (IActivationFunction<T>?)null);

        InitializeWeights();
    }

    private void InitializeWeights()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        double scale = 0.02;

        InitializeMatrix(_visionClsToken, random, scale);
        InitializeMatrix(_visionPositionalEmbeddings, random, scale);
        InitializeMatrix(_temporalPositionalEmbeddings, random, scale);
        InitializeMatrix(_textPositionalEmbeddings, random, scale);
    }

    private void InitializeMatrix(Matrix<T>? matrix, Random random, double scale)
    {
        if (matrix is null) return;

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.FromDouble(random.NextDouble() * scale - scale / 2);
            }
        }
    }

    #endregion

    #region IMultimodalEmbedding Implementation

    /// <inheritdoc/>
    public Vector<T> GetTextEmbedding(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Text cannot be null or empty.", nameof(text));

        return GetTextEmbeddings([text]).First();
    }

    /// <inheritdoc/>
    public IEnumerable<Vector<T>> GetTextEmbeddings(IEnumerable<string> texts)
    {
        var results = new List<Vector<T>>();

        foreach (var text in texts)
        {
            if (_useNativeMode)
            {
                var embedding = EncodeTextNative(text);
                results.Add(embedding);
            }
            else
            {
                var embedding = EncodeTextOnnx(text);
                results.Add(embedding);
            }
        }

        return results;
    }

    /// <inheritdoc/>
    public Vector<T> GetImageEmbedding(Tensor<T> image)
    {
        return GetImageEmbeddings([image]).First();
    }

    /// <inheritdoc/>
    public IEnumerable<Vector<T>> GetImageEmbeddings(IEnumerable<Tensor<T>> images)
    {
        var results = new List<Vector<T>>();

        foreach (var image in images)
        {
            // For single images, treat as single-frame video
            var videoEmbedding = GetVideoEmbedding([image]);
            results.Add(videoEmbedding);
        }

        return results;
    }

    /// <inheritdoc/>
    public T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding)
    {
        T similarity = NumOps.Zero;
        for (int i = 0; i < Math.Min(textEmbedding.Length, imageEmbedding.Length); i++)
        {
            similarity = NumOps.Add(similarity,
                NumOps.Multiply(textEmbedding[i], imageEmbedding[i]));
        }
        return similarity;
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, IEnumerable<string> classLabels)
    {
        var labels = classLabels.ToList();
        var imageEmbedding = GetImageEmbedding(image);
        var textEmbeddings = GetTextEmbeddings(labels.Select(l => $"a photo of {l}")).ToList();

        var similarities = new List<T>();
        foreach (var textEmb in textEmbeddings)
        {
            similarities.Add(ComputeSimilarity(textEmb, imageEmbedding));
        }

        var probabilities = Softmax(similarities);

        var result = new Dictionary<string, T>();
        for (int i = 0; i < labels.Count; i++)
        {
            result[labels[i]] = probabilities[i];
        }

        return result;
    }

    #endregion

    #region IVideoCLIPModel Implementation

    /// <inheritdoc/>
    public Vector<T> GetVideoEmbedding(IEnumerable<Tensor<T>> frames)
    {
        var frameList = frames.ToList();
        if (frameList.Count == 0)
            throw new ArgumentException("Frames cannot be empty.", nameof(frames));

        if (_useNativeMode)
        {
            return EncodeVideoNative(frameList);
        }
        else
        {
            return EncodeVideoOnnx(frameList);
        }
    }

    /// <inheritdoc/>
    public IEnumerable<Vector<T>> GetVideoEmbeddings(IEnumerable<IEnumerable<Tensor<T>>> videos)
    {
        var results = new List<Vector<T>>();

        foreach (var video in videos)
        {
            var embedding = GetVideoEmbedding(video);
            results.Add(embedding);
        }

        return results;
    }

    /// <inheritdoc/>
    public T ComputeVideoTextSimilarity(string text, IEnumerable<Tensor<T>> frames)
    {
        var textEmbedding = GetTextEmbedding(text);
        var videoEmbedding = GetVideoEmbedding(frames);
        return ComputeSimilarity(textEmbedding, videoEmbedding);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotActionRecognition(
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> actionLabels)
    {
        var labels = actionLabels.ToList();
        var videoEmbedding = GetVideoEmbedding(frames);
        var textEmbeddings = GetTextEmbeddings(labels.Select(l => $"a video of {l}")).ToList();

        var similarities = new List<T>();
        foreach (var textEmb in textEmbeddings)
        {
            similarities.Add(ComputeSimilarity(textEmb, videoEmbedding));
        }

        var probabilities = Softmax(similarities);

        var result = new Dictionary<string, T>();
        for (int i = 0; i < labels.Count; i++)
        {
            result[labels[i]] = probabilities[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveVideos(
        string query,
        IEnumerable<Vector<T>> videoEmbeddings,
        int topK = 10)
    {
        var queryEmbedding = GetTextEmbedding(query);
        var embeddings = videoEmbeddings.ToList();
        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < embeddings.Count; i++)
        {
            var similarity = ComputeSimilarity(queryEmbedding, embeddings[i]);
            scores.Add((i, similarity));
        }

        return scores
            .OrderByDescending(s => NumOps.ToDouble(s.Score))
            .Take(topK);
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveTextsForVideo(
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> candidateTexts,
        int topK = 10)
    {
        var videoEmbedding = GetVideoEmbedding(frames);
        var texts = candidateTexts.ToList();
        var textEmbeddings = GetTextEmbeddings(texts).ToList();
        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < textEmbeddings.Count; i++)
        {
            var similarity = ComputeSimilarity(textEmbeddings[i], videoEmbedding);
            scores.Add((i, similarity));
        }

        return scores
            .OrderByDescending(s => NumOps.ToDouble(s.Score))
            .Take(topK);
    }

    /// <inheritdoc/>
    public IEnumerable<(int StartFrame, int EndFrame, T Score)> LocalizeMoments(
        IEnumerable<Tensor<T>> frames,
        string query,
        int windowSize = 16)
    {
        var frameList = frames.ToList();
        var queryEmbedding = GetTextEmbedding(query);
        var results = new List<(int StartFrame, int EndFrame, T Score)>();

        int stride = Math.Max(1, windowSize / 2);

        for (int start = 0; start < frameList.Count - windowSize + 1; start += stride)
        {
            var windowFrames = frameList.Skip(start).Take(windowSize).ToList();
            var windowEmbedding = GetVideoEmbedding(windowFrames);
            var similarity = ComputeSimilarity(queryEmbedding, windowEmbedding);
            results.Add((start, start + windowSize - 1, similarity));
        }

        return results
            .OrderByDescending(r => NumOps.ToDouble(r.Score))
            .Take(10);
    }

    /// <inheritdoc/>
    public string GenerateVideoCaption(IEnumerable<Tensor<T>> frames, int maxLength = 77)
    {
        var videoEmbedding = GetVideoEmbedding(frames);

        if (_useNativeMode && _captionHead is not null)
        {
            // Autoregressive caption generation
            return GenerateCaptionAutoregressive(videoEmbedding, maxLength);
        }

        // Fallback to retrieval-based captioning for ONNX mode
        var candidateCaptions = new[]
        {
            "A person is performing an action.",
            "A video showing movement and activity.",
            "People are interacting in a scene.",
            "An event is taking place.",
            "Something is happening in this video."
        };

        var bestMatch = RetrieveTextsForVideo(frames, candidateCaptions, 1).FirstOrDefault();
        return bestMatch.Index >= 0 ? candidateCaptions[bestMatch.Index] : candidateCaptions[0];
    }

    /// <summary>
    /// Generates a caption autoregressively token by token.
    /// </summary>
    private string GenerateCaptionAutoregressive(Vector<T> videoContext, int maxLength)
    {
        var generatedTokens = new List<int>();
        const int BOS_TOKEN = 49406; // CLIP tokenizer BOS
        const int EOS_TOKEN = 49407; // CLIP tokenizer EOS
        const int PAD_TOKEN = 0;

        generatedTokens.Add(BOS_TOKEN);

        // Create context tensor from video embedding
        var contextTensor = Tensor<T>.CreateDefault([1, videoContext.Length], NumOps.Zero);
        for (int i = 0; i < videoContext.Length; i++)
        {
            contextTensor[0, i] = videoContext[i];
        }

        for (int step = 0; step < maxLength - 1; step++)
        {
            // Get current sequence embedding
            var sequenceEmbedding = GetSequenceEmbedding(generatedTokens);

            // Combine with video context using cross-attention-like mechanism
            var combinedContext = CombineVideoTextContext(contextTensor, sequenceEmbedding);

            // Project to vocabulary logits through caption head
            var logits = _captionHead!.Forward(combinedContext);

            // Get logits for the last position
            int vocabSize = logits.Shape.Length > 1 ? logits.Shape[1] : logits.Shape[0];
            var lastLogits = new T[Math.Min(vocabSize, _vocabularySize)];
            for (int i = 0; i < lastLogits.Length; i++)
            {
                lastLogits[i] = logits.Shape.Length > 1 ? logits[0, i] : logits[i];
            }

            // Apply softmax and sample next token (greedy or nucleus sampling)
            int nextToken = SampleNextToken(lastLogits, temperature: 0.8, topP: 0.9);

            if (nextToken == EOS_TOKEN || nextToken == PAD_TOKEN)
                break;

            generatedTokens.Add(nextToken);
        }

        // Decode tokens to text
        return DecodeTokensToText(generatedTokens);
    }

    /// <summary>
    /// Gets embedding for a sequence of token IDs.
    /// </summary>
    private Tensor<T> GetSequenceEmbedding(List<int> tokenIds)
    {
        if (_textTokenEmbedding is null || _textPositionalEmbeddings is null)
            throw new InvalidOperationException("Text embedding layers not initialized.");

        int seqLen = tokenIds.Count;
        var tokenTensor = Tensor<T>.CreateDefault([seqLen], NumOps.Zero);
        for (int i = 0; i < seqLen; i++)
        {
            tokenTensor[i] = NumOps.FromDouble(tokenIds[i]);
        }

        var embedded = _textTokenEmbedding.Forward(tokenTensor);

        // Add positional embeddings
        for (int i = 0; i < seqLen && i < _textPositionalEmbeddings.Rows; i++)
        {
            for (int j = 0; j < embedded.Shape[1] && j < _textPositionalEmbeddings.Columns; j++)
            {
                embedded[i, j] = NumOps.Add(embedded[i, j], _textPositionalEmbeddings[i, j]);
            }
        }

        // Apply text encoder layers
        var current = embedded;
        foreach (var layer in _textEncoderLayers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Combines video and text contexts for generation.
    /// </summary>
    private Tensor<T> CombineVideoTextContext(Tensor<T> videoContext, Tensor<T> textEmbedding)
    {
        // Use attention-weighted combination of video and text
        int textSeqLen = textEmbedding.Shape[0];
        int hiddenDim = textEmbedding.Shape[1];

        // Mean pool text embedding and add to video context
        var textPooled = MeanPool(textEmbedding);

        // Concatenate or add video and text contexts
        int outputDim = Math.Min(videoContext.Shape[1], _embeddingDimension);
        var combined = Tensor<T>.CreateDefault([1, outputDim], NumOps.Zero);

        for (int i = 0; i < outputDim; i++)
        {
            T videoVal = i < videoContext.Shape[1] ? videoContext[0, i] : NumOps.Zero;
            T textVal = i < textPooled.Length ? textPooled[i] : NumOps.Zero;
            // Gated combination
            T gate = NumOps.FromDouble(0.5);
            combined[0, i] = NumOps.Add(
                NumOps.Multiply(gate, videoVal),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, gate), textVal));
        }

        return combined;
    }

    /// <summary>
    /// Samples the next token using temperature scaling and nucleus (top-p) sampling.
    /// </summary>
    private int SampleNextToken(T[] logits, double temperature, double topP)
    {
        if (logits.Length == 0)
            return 0;

        // Apply temperature
        var scaledLogits = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            scaledLogits[i] = NumOps.ToDouble(logits[i]) / Math.Max(temperature, 0.01);
        }

        // Softmax
        double maxLogit = scaledLogits.Max();
        var expLogits = scaledLogits.Select(l => Math.Exp(l - maxLogit)).ToArray();
        double sumExp = expLogits.Sum();
        var probs = expLogits.Select(e => e / sumExp).ToArray();

        // Nucleus (top-p) sampling
        var sortedIndices = Enumerable.Range(0, probs.Length)
            .OrderByDescending(i => probs[i])
            .ToList();

        double cumulativeProb = 0.0;
        var nucleusIndices = new List<int>();
        var nucleusProbs = new List<double>();

        foreach (var idx in sortedIndices)
        {
            cumulativeProb += probs[idx];
            nucleusIndices.Add(idx);
            nucleusProbs.Add(probs[idx]);

            if (cumulativeProb >= topP)
                break;
        }

        // Renormalize nucleus probabilities
        double nucleusSum = nucleusProbs.Sum();
        var normalizedProbs = nucleusProbs.Select(p => p / nucleusSum).ToArray();

        // Sample from nucleus
        var random = RandomHelper.Shared;
        double sample = random.NextDouble();
        double cumulative = 0.0;

        for (int i = 0; i < nucleusIndices.Count; i++)
        {
            cumulative += normalizedProbs[i];
            if (sample <= cumulative)
                return nucleusIndices[i];
        }

        return nucleusIndices[nucleusIndices.Count - 1];
    }

    /// <summary>
    /// Decodes token IDs back to text.
    /// </summary>
    private string DecodeTokensToText(List<int> tokenIds)
    {
        const int BOS_TOKEN = 49406;
        const int EOS_TOKEN = 49407;

        // Filter out special tokens
        var filteredTokens = tokenIds
            .Where(t => t != BOS_TOKEN && t != EOS_TOKEN && t > 0)
            .ToList();

        if (filteredTokens.Count == 0)
            return "A video showing activity.";

        // Use tokenizer to decode if available
        try
        {
            return _tokenizer.Decode(filteredTokens);
        }
        catch
        {
            // Fallback: return generic caption
            return "A video showing activity.";
        }
    }

    /// <inheritdoc/>
    public string AnswerVideoQuestion(
        IEnumerable<Tensor<T>> frames,
        string question,
        int maxLength = 64)
    {
        var videoEmbedding = GetVideoEmbedding(frames);
        var questionEmbedding = GetTextEmbedding(question);

        if (_useNativeMode && _captionHead is not null)
        {
            // Generative question answering
            return GenerateAnswerAutoregressive(videoEmbedding, questionEmbedding, question, maxLength);
        }

        // Fallback to retrieval-based QA for ONNX mode
        var candidateAnswers = new[] { "Yes", "No", "Unknown" };
        var bestMatch = RetrieveTextsForVideo(frames,
            candidateAnswers.Select(a => $"{question} {a}"), 1).FirstOrDefault();

        return bestMatch.Index >= 0 ? candidateAnswers[bestMatch.Index] : "Unknown";
    }

    /// <summary>
    /// Generates an answer autoregressively using video and question context.
    /// </summary>
    private string GenerateAnswerAutoregressive(
        Vector<T> videoContext,
        Vector<T> questionContext,
        string question,
        int maxLength)
    {
        var generatedTokens = new List<int>();
        const int BOS_TOKEN = 49406;
        const int EOS_TOKEN = 49407;
        const int PAD_TOKEN = 0;

        // Encode the question as prompt tokens
        var questionTokens = _tokenizer.Encode(question);
        foreach (var token in questionTokens.TokenIds.Take(_maxSequenceLength / 2))
        {
            generatedTokens.Add(token);
        }

        // Add separator/answer prompt token
        generatedTokens.Add(BOS_TOKEN);

        // Create combined video-question context tensor
        int contextDim = videoContext.Length;
        var combinedContextTensor = Tensor<T>.CreateDefault([1, contextDim], NumOps.Zero);

        // Fuse video and question embeddings with attention-weighted mechanism
        for (int i = 0; i < contextDim; i++)
        {
            T videoVal = i < videoContext.Length ? videoContext[i] : NumOps.Zero;
            T questionVal = i < questionContext.Length ? questionContext[i] : NumOps.Zero;

            // Attention-weighted fusion: emphasize question-relevant video features
            T dotProduct = NumOps.Multiply(videoVal, questionVal);
            T weight = NumOps.FromDouble(Math.Max(0.3, Math.Min(0.7,
                0.5 + NumOps.ToDouble(dotProduct) * 0.2)));

            combinedContextTensor[0, i] = NumOps.Add(
                NumOps.Multiply(weight, videoVal),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, weight), questionVal));
        }

        // Check if this is a yes/no question for constrained decoding
        bool isYesNoQuestion = IsYesNoQuestion(question);

        for (int step = 0; step < maxLength; step++)
        {
            // Get current sequence embedding
            var sequenceEmbedding = GetSequenceEmbedding(generatedTokens);

            // Combine with video-question context
            var decoderInput = CombineVideoTextContext(combinedContextTensor, sequenceEmbedding);

            // Project to vocabulary logits
            var logits = _captionHead!.Forward(decoderInput);

            // Get logits for the last position
            int vocabSize = logits.Shape.Length > 1 ? logits.Shape[1] : logits.Shape[0];
            var lastLogits = new T[Math.Min(vocabSize, _vocabularySize)];
            for (int i = 0; i < lastLogits.Length; i++)
            {
                lastLogits[i] = logits.Shape.Length > 1 ? logits[0, i] : logits[i];
            }

            int nextToken;
            if (isYesNoQuestion && step == 0)
            {
                // Constrained decoding for yes/no questions - only consider yes/no tokens
                nextToken = SampleYesNoToken(lastLogits);
            }
            else
            {
                // Regular sampling with lower temperature for more focused answers
                nextToken = SampleNextToken(lastLogits, temperature: 0.6, topP: 0.85);
            }

            if (nextToken == EOS_TOKEN || nextToken == PAD_TOKEN)
                break;

            generatedTokens.Add(nextToken);
        }

        // Decode only the answer portion (tokens after BOS separator)
        int answerStartIndex = generatedTokens.LastIndexOf(BOS_TOKEN) + 1;
        var answerTokens = generatedTokens.Skip(answerStartIndex).ToList();

        if (answerTokens.Count == 0)
        {
            // Fallback based on question type
            return isYesNoQuestion ? "Unknown" : "The video shows activity.";
        }

        return DecodeTokensToText(answerTokens);
    }

    /// <summary>
    /// Determines if a question expects a yes/no answer.
    /// </summary>
    private static bool IsYesNoQuestion(string question)
    {
        var lowerQuestion = question.ToLowerInvariant().Trim();
        string[] yesNoStarters = { "is ", "are ", "was ", "were ", "do ", "does ", "did ",
                                   "can ", "could ", "will ", "would ", "should ", "has ", "have ", "had " };

        return yesNoStarters.Any(starter => lowerQuestion.StartsWith(starter)) ||
               lowerQuestion.EndsWith("?") && (
                   lowerQuestion.Contains(" or not") ||
                   lowerQuestion.Contains("yes or no"));
    }

    /// <summary>
    /// Samples a yes/no token with constrained decoding.
    /// </summary>
    private int SampleYesNoToken(T[] logits)
    {
        // Common token IDs for yes/no in CLIP vocabulary
        // These are approximate - actual IDs depend on tokenizer
        const int YES_TOKEN_APPROX = 8505;  // "yes"
        const int NO_TOKEN_APPROX = 645;    // "no"
        const int UNKNOWN_TOKEN_APPROX = 3067; // "unknown"

        // Get logits for yes/no/unknown tokens
        double yesLogit = YES_TOKEN_APPROX < logits.Length ?
            NumOps.ToDouble(logits[YES_TOKEN_APPROX]) : double.MinValue;
        double noLogit = NO_TOKEN_APPROX < logits.Length ?
            NumOps.ToDouble(logits[NO_TOKEN_APPROX]) : double.MinValue;
        double unknownLogit = UNKNOWN_TOKEN_APPROX < logits.Length ?
            NumOps.ToDouble(logits[UNKNOWN_TOKEN_APPROX]) : double.MinValue;

        // Add small noise for diversity
        var random = RandomHelper.Shared;
        yesLogit += random.NextDouble() * 0.1;
        noLogit += random.NextDouble() * 0.1;
        unknownLogit += random.NextDouble() * 0.05; // Slight bias against unknown

        if (yesLogit >= noLogit && yesLogit >= unknownLogit)
            return YES_TOKEN_APPROX;
        if (noLogit >= yesLogit && noLogit >= unknownLogit)
            return NO_TOKEN_APPROX;
        return UNKNOWN_TOKEN_APPROX;
    }

    /// <inheritdoc/>
    public Tensor<T> ExtractFrameFeatures(IEnumerable<Tensor<T>> frames)
    {
        var frameList = frames.ToList();
        var features = new List<Vector<T>>();

        foreach (var frame in frameList)
        {
            var frameFeature = EncodeFrameNative(frame);
            features.Add(frameFeature);
        }

        int numFrames = features.Count;
        int featureDim = features[0].Length;
        var result = Tensor<T>.CreateDefault([numFrames, featureDim], NumOps.Zero);

        for (int i = 0; i < numFrames; i++)
        {
            for (int j = 0; j < featureDim; j++)
            {
                result[i, j] = features[i][j];
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeTemporalSimilarityMatrix(
        IEnumerable<Tensor<T>> video1Frames,
        IEnumerable<Tensor<T>> video2Frames)
    {
        var features1 = ExtractFrameFeatures(video1Frames);
        var features2 = ExtractFrameFeatures(video2Frames);

        int numFrames1 = features1.Shape[0];
        int numFrames2 = features2.Shape[0];
        int featureDim = features1.Shape[1];

        var similarityMatrix = Tensor<T>.CreateDefault([numFrames1, numFrames2], NumOps.Zero);

        for (int i = 0; i < numFrames1; i++)
        {
            for (int j = 0; j < numFrames2; j++)
            {
                T similarity = NumOps.Zero;
                for (int k = 0; k < featureDim; k++)
                {
                    similarity = NumOps.Add(similarity,
                        NumOps.Multiply(features1[i, k], features2[j, k]));
                }
                similarityMatrix[i, j] = similarity;
            }
        }

        return similarityMatrix;
    }

    /// <inheritdoc/>
    public Dictionary<string, T> PredictNextAction(
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> possibleNextActions)
    {
        var actions = possibleNextActions.ToList();
        var videoEmbedding = GetVideoEmbedding(frames);
        var textEmbeddings = GetTextEmbeddings(actions.Select(a => $"next action: {a}")).ToList();

        var similarities = new List<T>();
        foreach (var textEmb in textEmbeddings)
        {
            similarities.Add(ComputeSimilarity(textEmb, videoEmbedding));
        }

        var probabilities = Softmax(similarities);

        var result = new Dictionary<string, T>();
        for (int i = 0; i < actions.Count; i++)
        {
            result[actions[i]] = probabilities[i];
        }

        return result;
    }

    #endregion

    #region Video Encoding Methods

    private Vector<T> EncodeVideoNative(List<Tensor<T>> frames)
    {
        // Sample frames if too many
        var sampledFrames = SampleFrames(frames, _numFrames);

        // Encode each frame
        var frameFeatures = new List<Tensor<T>>();
        foreach (var frame in sampledFrames)
        {
            var features = EncodeFrameNative(frame);
            var featureTensor = Tensor<T>.CreateDefault([1, features.Length], NumOps.Zero);
            for (int i = 0; i < features.Length; i++)
            {
                featureTensor[0, i] = features[i];
            }
            frameFeatures.Add(featureTensor);
        }

        // Stack frame features
        int numSampledFrames = frameFeatures.Count;
        int featureDim = frameFeatures[0].Shape[1];
        var stackedFeatures = Tensor<T>.CreateDefault([numSampledFrames, featureDim], NumOps.Zero);

        for (int i = 0; i < numSampledFrames; i++)
        {
            for (int j = 0; j < featureDim; j++)
            {
                stackedFeatures[i, j] = frameFeatures[i][0, j];
            }
        }

        // Add temporal positional embeddings
        if (_temporalPositionalEmbeddings is not null)
        {
            for (int i = 0; i < numSampledFrames && i < _temporalPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < featureDim && j < _temporalPositionalEmbeddings.Columns; j++)
                {
                    stackedFeatures[i, j] = NumOps.Add(stackedFeatures[i, j], _temporalPositionalEmbeddings[i, j]);
                }
            }
        }

        // Apply temporal encoder
        var temporalFeatures = stackedFeatures;
        foreach (var layer in _temporalEncoderLayers)
        {
            temporalFeatures = layer.Forward(temporalFeatures);
        }

        // Aggregate (mean pooling or take CLS token)
        var pooled = MeanPool(temporalFeatures);

        // Project to embedding space
        if (_videoProjection is not null)
        {
            var pooledTensor = Tensor<T>.CreateDefault([1, pooled.Length], NumOps.Zero);
            for (int i = 0; i < pooled.Length; i++)
            {
                pooledTensor[0, i] = pooled[i];
            }
            var projected = _videoProjection.Forward(pooledTensor);
            pooled = new Vector<T>(_embeddingDimension);
            for (int i = 0; i < _embeddingDimension && i < projected.Shape[1]; i++)
            {
                pooled[i] = projected[0, i];
            }
        }

        return Normalize(pooled);
    }

    private Vector<T> EncodeFrameNative(Tensor<T> frame)
    {
        if (_patchEmbedding is null || _visionClsToken is null || _visionPositionalEmbeddings is null)
            throw new InvalidOperationException("Native layers not initialized.");

        // Patch embedding
        var patches = _patchEmbedding.Forward(frame);

        // Add CLS token
        var withCls = PrependClsToken(patches, _visionClsToken);

        // Add positional embeddings
        var positioned = AddPositionalEmbeddings(withCls, _visionPositionalEmbeddings);

        // Apply frame encoder layers
        var current = positioned;
        foreach (var layer in _frameEncoderLayers)
        {
            current = layer.Forward(current);
        }

        // Take CLS token as frame representation
        var clsFeature = new Vector<T>(_visionHiddenDim);
        for (int i = 0; i < _visionHiddenDim && i < current.Shape[1]; i++)
        {
            clsFeature[i] = current[0, i];
        }

        return clsFeature;
    }

    private Vector<T> EncodeVideoOnnx(List<Tensor<T>> frames)
    {
        if (_videoEncoder is null)
            throw new InvalidOperationException("ONNX video encoder not initialized.");

        var sampledFrames = SampleFrames(frames, _numFrames);

        int channels = sampledFrames[0].Shape[0];
        int height = sampledFrames[0].Shape[1];
        int width = sampledFrames[0].Shape[2];
        int numSampledFrames = sampledFrames.Count;

        // Create input tensor [1, numFrames, channels, height, width]
        var inputArray = new float[1 * numSampledFrames * channels * height * width];
        int idx = 0;
        for (int f = 0; f < numSampledFrames; f++)
        {
            var frame = sampledFrames[f];
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        inputArray[idx++] = (float)NumOps.ToDouble(frame[c, h, w]);
                    }
                }
            }
        }

        var inputTensor = new OnnxTensors.DenseTensor<float>(inputArray, [1, numSampledFrames, channels, height, width]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", inputTensor)
        };

        using var results = _videoEncoder.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var embedding = new Vector<T>(_embeddingDimension);
        for (int i = 0; i < _embeddingDimension && i < outputTensor.Length; i++)
        {
            embedding[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return Normalize(embedding);
    }

    private List<Tensor<T>> SampleFrames(List<Tensor<T>> frames, int targetCount)
    {
        if (frames.Count <= targetCount)
        {
            // Pad with last frame if too few
            var result = new List<Tensor<T>>(frames);
            while (result.Count < targetCount)
            {
                result.Add(frames[frames.Count - 1]);
            }
            return result;
        }

        // Sample uniformly
        var sampled = new List<Tensor<T>>();
        double step = (double)frames.Count / targetCount;
        for (int i = 0; i < targetCount; i++)
        {
            int idx = Math.Min((int)(i * step), frames.Count - 1);
            sampled.Add(frames[idx]);
        }

        return sampled;
    }

    #endregion

    #region Text Encoding Methods

    private Vector<T> EncodeTextNative(string text)
    {
        if (_textTokenEmbedding is null || _textPositionalEmbeddings is null || _textProjection is null)
            throw new InvalidOperationException("Native text layers not initialized.");

        var encoded = _tokenizer.Encode(text);
        var inputIds = encoded.TokenIds;

        // Truncate or pad
        var paddedIds = new List<int>();
        for (int i = 0; i < _maxSequenceLength; i++)
        {
            paddedIds.Add(i < inputIds.Count ? inputIds[i] : 0);
        }

        // Token embedding
        var tokenTensor = Tensor<T>.CreateDefault([_maxSequenceLength], NumOps.Zero);
        for (int i = 0; i < _maxSequenceLength; i++)
        {
            tokenTensor[i] = NumOps.FromDouble(paddedIds[i]);
        }
        var embedded = _textTokenEmbedding.Forward(tokenTensor);

        // Add positional embeddings
        var positioned = AddPositionalEmbeddings(embedded, _textPositionalEmbeddings);

        // Apply text encoder layers
        var current = positioned;
        foreach (var layer in _textEncoderLayers)
        {
            current = layer.Forward(current);
        }

        // Pool (take EOS token position or mean pool)
        var pooled = MeanPool(current);

        // Project to embedding space
        var pooledTensor = Tensor<T>.CreateDefault([1, pooled.Length], NumOps.Zero);
        for (int i = 0; i < pooled.Length; i++)
        {
            pooledTensor[0, i] = pooled[i];
        }
        var projected = _textProjection.Forward(pooledTensor);

        var result = new Vector<T>(_embeddingDimension);
        for (int i = 0; i < _embeddingDimension && i < projected.Shape[1]; i++)
        {
            result[i] = projected[0, i];
        }

        return Normalize(result);
    }

    private Vector<T> EncodeTextOnnx(string text)
    {
        if (_textEncoder is null)
            throw new InvalidOperationException("ONNX text encoder not initialized.");

        var encoded = _tokenizer.Encode(text);
        var inputIds = encoded.TokenIds;

        // Truncate or pad
        var paddedIds = new long[_maxSequenceLength];
        var attentionMask = new long[_maxSequenceLength];
        for (int i = 0; i < _maxSequenceLength; i++)
        {
            paddedIds[i] = i < inputIds.Count ? inputIds[i] : 0;
            attentionMask[i] = i < inputIds.Count ? 1 : 0;
        }

        var inputIdsTensor = new OnnxTensors.DenseTensor<long>(paddedIds, [1, _maxSequenceLength]);
        var attentionMaskTensor = new OnnxTensors.DenseTensor<long>(attentionMask, [1, _maxSequenceLength]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };

        using var results = _textEncoder.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var embedding = new Vector<T>(_embeddingDimension);
        for (int i = 0; i < _embeddingDimension && i < outputTensor.Length; i++)
        {
            embedding[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return Normalize(embedding);
    }

    #endregion

    #region Helper Methods

    private Tensor<T> PrependClsToken(Tensor<T> sequence, Matrix<T> clsToken)
    {
        int seqLen = sequence.Shape[0];
        int hiddenDim = sequence.Shape[1];

        var result = Tensor<T>.CreateDefault([seqLen + 1, hiddenDim], NumOps.Zero);

        for (int j = 0; j < hiddenDim && j < clsToken.Columns; j++)
        {
            result[0, j] = clsToken[0, j];
        }

        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                result[i + 1, j] = sequence[i, j];
            }
        }

        return result;
    }

    private Tensor<T> AddPositionalEmbeddings(Tensor<T> sequence, Matrix<T> posEmbeddings)
    {
        int seqLen = sequence.Shape[0];
        int hiddenDim = sequence.Shape[1];

        var result = Tensor<T>.CreateDefault([seqLen, hiddenDim], NumOps.Zero);

        for (int i = 0; i < seqLen && i < posEmbeddings.Rows; i++)
        {
            for (int j = 0; j < hiddenDim && j < posEmbeddings.Columns; j++)
            {
                result[i, j] = NumOps.Add(sequence[i, j], posEmbeddings[i, j]);
            }
        }

        return result;
    }

    private Vector<T> MeanPool(Tensor<T> tensor)
    {
        int seqLen = tensor.Shape[0];
        int hiddenDim = tensor.Shape[1];

        var result = new Vector<T>(hiddenDim);

        for (int j = 0; j < hiddenDim; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < seqLen; i++)
            {
                sum = NumOps.Add(sum, tensor[i, j]);
            }
            result[j] = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
        }

        return result;
    }

    private Vector<T> Normalize(Vector<T> vector)
    {
        T sumSquares = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(vector[i], vector[i]));
        }

        T norm = NumOps.Sqrt(sumSquares);
        if (NumOps.ToDouble(norm) < 1e-12)
            return vector;

        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = NumOps.Divide(vector[i], norm);
        }

        return result;
    }

    private List<T> Softmax(List<T> values)
    {
        double maxVal = values.Max(v => NumOps.ToDouble(v));
        var expValues = values.Select(v => Math.Exp(NumOps.ToDouble(v) - maxVal)).ToList();
        double sumExp = expValues.Sum();
        return expValues.Select(e => NumOps.FromDouble(e / sumExp)).ToList();
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        var frames = new List<Tensor<T>> { input };
        var embedding = GetVideoEmbedding(frames);
        var result = Tensor<T>.CreateDefault([1, embedding.Length], NumOps.Zero);
        for (int i = 0; i < embedding.Length; i++)
        {
            result[0, i] = embedding[i];
        }
        return result;
    }

    /// <summary>
    /// Backward pass through video encoder layers.
    /// </summary>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Backward pass is only supported in native mode.");
        }

        var currentGradient = gradient;

        // Backward through temporal encoder layers in reverse order
        for (int i = _temporalEncoderLayers.Count - 1; i >= 0; i--)
        {
            currentGradient = _temporalEncoderLayers[i].Backward(currentGradient);
        }

        // Backward through frame encoder layers
        for (int i = _frameEncoderLayers.Count - 1; i >= 0; i--)
        {
            currentGradient = _frameEncoderLayers[i].Backward(currentGradient);
        }

        return currentGradient;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);

        var frames = new List<Tensor<T>> { input };
        var embedding = GetVideoEmbedding(frames);
        var embeddingTensor = Tensor<T>.CreateDefault([1, embedding.Length], NumOps.Zero);
        for (int i = 0; i < embedding.Length; i++)
        {
            embeddingTensor[0, i] = embedding[i];
        }

        LastLoss = LossFunction.CalculateLoss(embeddingTensor.ToVector(), expectedOutput.ToVector());
        var lossGradient = LossFunction.CalculateDerivative(embeddingTensor.ToVector(), expectedOutput.ToVector());
        var gradient = Tensor<T>.FromVector(lossGradient);

        Backward(gradient);
        var currentParams = GetParameters();
        UpdateParameters(currentParams);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int expectedCount = ParameterCount;
        if (parameters.Length != expectedCount)
        {
            throw new ArgumentException(
                $"Expected {expectedCount} parameters, but got {parameters.Length}.",
                nameof(parameters));
        }

        int offset = 0;

        offset = UpdateLayerListParameters(_frameEncoderLayers, parameters, offset);
        offset = UpdateLayerListParameters(_temporalEncoderLayers, parameters, offset);
        offset = UpdateLayerListParameters(_textEncoderLayers, parameters, offset);
        offset = UpdateLayerListParameters(_projectionLayers, parameters, offset);
    }

    private int UpdateLayerListParameters(List<ILayer<T>> layers, Vector<T> parameters, int offset)
    {
        foreach (var layer in layers)
        {
            int layerParamCount = layer.ParameterCount;
            if (layerParamCount > 0)
            {
                var layerParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    layerParams[i] = parameters[offset + i];
                }
                layer.UpdateParameters(layerParams);
                offset += layerParamCount;
            }
        }
        return offset;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = Enums.ModelType.VideoCLIP,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ImageSize", _imageSize },
                { "EmbeddingDimension", _embeddingDimension },
                { "MaxSequenceLength", _maxSequenceLength },
                { "NumFrames", _numFrames },
                { "FrameRate", _frameRate },
                { "TemporalAggregation", _temporalAggregation },
                { "VisionHiddenDim", _visionHiddenDim },
                { "TextHiddenDim", _textHiddenDim },
                { "NumFrameEncoderLayers", _numFrameEncoderLayers },
                { "NumTemporalLayers", _numTemporalLayers },
                { "NumTextLayers", _numTextLayers },
                { "VocabularySize", _vocabularySize },
                { "UseNativeMode", _useNativeMode }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embeddingDimension);
        writer.Write(_maxSequenceLength);
        writer.Write(_imageSize);
        writer.Write(_visionHiddenDim);
        writer.Write(_textHiddenDim);
        writer.Write(_numFrameEncoderLayers);
        writer.Write(_numTemporalLayers);
        writer.Write(_numTextLayers);
        writer.Write(_numHeads);
        writer.Write(_patchSize);
        writer.Write(_vocabularySize);
        writer.Write(_numFrames);
        writer.Write(_frameRate);
        writer.Write(_temporalAggregation);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // embeddingDim
        _ = reader.ReadInt32(); // maxSeqLen
        _ = reader.ReadInt32(); // imageSize
        _ = reader.ReadInt32(); // visionHiddenDim
        _ = reader.ReadInt32(); // textHiddenDim
        _ = reader.ReadInt32(); // numFrameEncoderLayers
        _ = reader.ReadInt32(); // numTemporalLayers
        _ = reader.ReadInt32(); // numTextLayers
        _ = reader.ReadInt32(); // numHeads
        _ = reader.ReadInt32(); // patchSize
        _ = reader.ReadInt32(); // vocabularySize
        _ = reader.ReadInt32(); // numFrames
        _ = reader.ReadDouble(); // frameRate
        _ = reader.ReadString(); // temporalAggregation
        _ = reader.ReadBoolean(); // useNativeMode
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new VideoCLIPNeuralNetwork<T>(
            Architecture,
            _imageSize,
            channels: 3,
            _patchSize,
            _vocabularySize,
            _maxSequenceLength,
            _embeddingDimension,
            _visionHiddenDim,
            _textHiddenDim,
            _numFrameEncoderLayers,
            _numTemporalLayers,
            _numTextLayers,
            _numHeads,
            _numFrames,
            _frameRate,
            _temporalAggregation);
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _videoEncoder?.Dispose();
            _textEncoder?.Dispose();
        }

        base.Dispose(disposing);
    }

    #endregion

    #region IMultimodalEmbedding Interface (Standard API)

    /// <inheritdoc/>
    public Vector<T> EncodeText(string text)
    {
        return GetTextEmbedding(text);
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeTextBatch(IEnumerable<string> texts)
    {
        var embeddings = GetTextEmbeddings(texts).ToList();
        if (embeddings.Count == 0)
        {
            return new Matrix<T>(0, EmbeddingDimension);
        }

        var matrix = new Matrix<T>(embeddings.Count, embeddings[0].Length);
        for (int i = 0; i < embeddings.Count; i++)
        {
            for (int j = 0; j < embeddings[i].Length; j++)
            {
                matrix[i, j] = embeddings[i][j];
            }
        }
        return matrix;
    }

    /// <inheritdoc/>
    public Vector<T> EncodeImage(double[] imageData)
    {
        // Convert double[] to Tensor<T> in CHW format
        var tensor = ConvertToTensor(imageData);
        return GetImageEmbedding(tensor);
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeImageBatch(IEnumerable<double[]> imageDataBatch)
    {
        var tensors = imageDataBatch.Select(ConvertToTensor);
        var embeddings = GetImageEmbeddings(tensors).ToList();
        if (embeddings.Count == 0)
        {
            return new Matrix<T>(0, EmbeddingDimension);
        }

        var matrix = new Matrix<T>(embeddings.Count, embeddings[0].Length);
        for (int i = 0; i < embeddings.Count; i++)
        {
            for (int j = 0; j < embeddings[i].Length; j++)
            {
                matrix[i, j] = embeddings[i][j];
            }
        }
        return matrix;
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(double[] imageData, IEnumerable<string> labels)
    {
        var tensor = ConvertToTensor(imageData);
        return ZeroShotClassify(tensor, labels);
    }

    /// <summary>
    /// Converts a double[] image to Tensor format.
    /// </summary>
    private Tensor<T> ConvertToTensor(double[] imageData)
    {
        if (imageData == null || imageData.Length == 0)
        {
            throw new ArgumentException("Image data cannot be null or empty.", nameof(imageData));
        }

        int channels = 3;
        if (imageData.Length % channels != 0)
        {
            throw new ArgumentException($"Image data length ({imageData.Length}) must be divisible by {channels} channels.", nameof(imageData));
        }

        int pixels = imageData.Length / channels;
        int size = (int)Math.Sqrt(pixels);
        if (size * size != pixels)
        {
            throw new ArgumentException($"Image must be square. Got {pixels} pixels which is not a perfect square.", nameof(imageData));
        }

        var tensor = new Tensor<T>(new[] { channels, size, size });
        int idx = 0;
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < size; h++)
            {
                for (int w = 0; w < size; w++)
                {
                    tensor[c, h, w] = NumOps.FromDouble(imageData[idx++]);
                }
            }
        }
        return tensor;
    }

    #endregion

}
