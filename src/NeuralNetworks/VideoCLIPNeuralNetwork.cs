using AiDotNet.Optimizers;
using AiDotNet.LearningRateSchedulers;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;
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
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 8, outputSize: 4);
/// var video = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 3, 8, 224, 224 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new VideoCLIPNeuralNetwork&lt;float&gt;(architecture))
///     .Build(trainX, trainY);
/// var embedding = result.Predict(video);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.EmbeddingModel)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding", "https://arxiv.org/abs/2109.14084", Year = 2021, Authors = "Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Arber Zela, Florian Metze, Luke Zettlemoyer, Christoph Feichtenhofer")]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, LearningRate = 5e-5,
                WarmupSteps = 1000, Schedule = LearningRateSchedulerType.Polynomial,
                MinLearningRate = 0, MaxGradientNorm = 2.0,
                Source = "Xu et al. 2021, Training Details: Adam with betas (0.9, 0.98), initial learning rate 5e-5, 1000 warm-up steps, a polynomial decay schedule, and gradients clipped at 2.0. The decay power is not stated, so the library default of 1 applies.")]
public partial class VideoCLIPNeuralNetwork<T> : MultimodalModelLayoutBase<T>, IVideoCLIPModel<T>
{
    private readonly VideoCLIPOptions _options;

    #region Execution Mode

    private bool _useNativeMode;

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
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _visionClsToken;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _visionPositionalEmbeddings;
    [Scratch]
    private Tensor<T>? _temporalPositionalEmbeddings;
    private Tensor<T>? _textPositionalEmbeddings;
    private ILayer<T>? _patchEmbedding;
    private ILayer<T>? _textTokenEmbedding;
    private ILayer<T>? _videoProjection;
    private ILayer<T>? _textProjection;
    private ILayer<T>? _captionHead;

    // Gradient checkpointing: cached frames from the last forward pass for backward recomputation
    [Scratch]
    private List<Tensor<T>>? _cachedTrainingFrames;


    #endregion

    #region Shared Fields

    private readonly ITokenizer _tokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
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
    private TemporalAggregationType _temporalAggregation;

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
    public TemporalAggregationType TemporalAggregation => _temporalAggregation;

    #endregion

    #region Constructors

    private string _videoOutputName = string.Empty;
    private string _textOutputName = string.Empty;
    private int _onnxChannels = 3;
    private const OnnxEmbeddingLayouts EmbeddingLayouts = OnnxEmbeddingLayouts.Vector
        | OnnxEmbeddingLayouts.BatchedVector | OnnxEmbeddingLayouts.FirstToken;

    /// <summary>
    /// Creates a VideoCLIP network using pretrained ONNX models.
    /// </summary>
    /// <remarks>Frame count, image/channel dimensions, text context and embedding width must
    /// match the supplied graphs. Native-only tower overrides are rejected. FrameRate remains
    /// nominal metadata, not a resampling operation. Loaded signatures are exposed separately
    /// in <see cref="MultimodalModelLayoutBase{T}.OnnxConfiguration"/>.</remarks>
    public VideoCLIPNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string videoEncoderPath,
        string textEncoderPath,
        ITokenizer tokenizer,
        VideoCLIPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CosineSimilarityLoss<T>(), 1.0)
    {
        _options = options ?? new VideoCLIPOptions();
        _options.ValidateOnnx();
        Options = _options;

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
        _numFrames = _options.NumFrames;
        _frameRate = _options.FrameRate;
        _temporalAggregation = _options.TemporalAggregation;
        _embeddingDimension = _options.EmbeddingDimension;
        _maxSequenceLength = _options.MaxSequenceLength;
        _imageSize = _options.ImageSize;
        _onnxChannels = _options.Channels;
        _patchSize = _options.PatchSize;
        _visionHiddenDim = _options.VisionDim;
        _textHiddenDim = _options.TextHiddenDim;
        _numFrameEncoderLayers = _options.NumFrameEncoderLayers;
        _numTemporalLayers = _options.NumTemporalLayers;
        _numTextLayers = _options.NumTextLayers;
        _numHeads = _options.NumHeads;
        _vocabularySize = _options.VocabSize;

        InferenceSession? videoEncoder = null;
        InferenceSession? textEncoder = null;

        try
        {
            videoEncoder = new InferenceSession(videoEncoderPath);
            textEncoder = new InferenceSession(textEncoderPath);
            var videoGraph = new OnnxGraphSignature(OnnxModelRole.VideoEncoder, videoEncoder);
            var textGraph = new OnnxGraphSignature(OnnxModelRole.TextEncoder, textEncoder);
            videoGraph.RequireInputSet("pixel_values");
            videoGraph.RequireInput("pixel_values", OnnxTensors.TensorElementType.Float,
                1, _numFrames, _onnxChannels, _imageSize, _imageSize);
            textGraph.RequireInputSet("input_ids", "attention_mask");
            textGraph.RequireInput("input_ids", OnnxTensors.TensorElementType.Int64, 1, _maxSequenceLength);
            textGraph.RequireInput("attention_mask", OnnxTensors.TensorElementType.Int64, 1, _maxSequenceLength);
            string videoOutputName = videoGraph.RequireEmbeddingOutput(_embeddingDimension, EmbeddingLayouts);
            string textOutputName = textGraph.RequireEmbeddingOutput(_embeddingDimension, EmbeddingLayouts);
            Guard.NotNull(tokenizer);
            _tokenizer = tokenizer;
            _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _lossFunction = lossFunction ?? new CosineSimilarityLoss<T>();
            var configuration = new OnnxMultimodalConfiguration(_embeddingDimension, _maxSequenceLength,
                _imageSize, _tokenizer.VocabularySize, _numFrames, _onnxChannels, videoGraph, textGraph);
            InitializeLayers();
            _videoEncoder = videoEncoder;
            _textEncoder = textEncoder;
            _videoOutputName = videoOutputName;
            _textOutputName = textOutputName;
            OnnxConfiguration = configuration;
        }
        catch (Exception constructionFailure)
        {
            // Release each session independently so one failed Dispose neither skips the other nor replaces the
            // construction failure the caller needs to see.
            ReleaseAfterFailedConstruction(textEncoder, constructionFailure);
            ReleaseAfterFailedConstruction(videoEncoder, constructionFailure);
            throw;
        }
    }

    /// <summary>Disposes a session created by a constructor that is failing, without masking that failure.</summary>
    private static void ReleaseAfterFailedConstruction(InferenceSession? session, Exception constructionFailure)
    {
        if (session is null) return;
        try
        {
            session.Dispose();
        }
        catch (Exception cleanupFailure)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"VideoCLIP: disposing an ONNX session after construction failed ({constructionFailure.GetType().Name}) " +
                $"also failed: {cleanupFailure}");
        }
    }

    /// <summary>
    /// Creates a VideoCLIP network using native library layers.
    /// </summary>
    public VideoCLIPNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        VideoCLIPOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CosineSimilarityLoss<T>(), 1.0)
    {
        _options = options ?? new VideoCLIPOptions();
        _options.Validate();
        _options.Validate();
        Options = _options;

        _useNativeMode = true;
        _embeddingDimension = _options.EmbeddingDimension;
        _maxSequenceLength = _options.MaxSequenceLength;
        _imageSize = _options.ImageSize;
        _visionHiddenDim = _options.VisionDim;
        _textHiddenDim = _options.TextHiddenDim;
        _numFrameEncoderLayers = _options.NumFrameEncoderLayers;
        _numTemporalLayers = _options.NumTemporalLayers;
        _numTextLayers = _options.NumTextLayers;
        _numHeads = _options.NumHeads;
        _patchSize = _options.PatchSize;
        _vocabularySize = _options.VocabSize;
        _numFrames = _options.NumFrames;
        _frameRate = _options.FrameRate;
        _temporalAggregation = _options.TemporalAggregation;

        _tokenizer = tokenizer ?? Tokenization.ClipTokenizerFactory.CreateSimple();
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new CosineSimilarityLoss<T>();

        InitializeNativeLayers(_options.Channels);
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

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateVideoCLIPLayers(
                _imageSize, channels, _patchSize, _visionHiddenDim, _textHiddenDim,
                _embeddingDimension, _numFrameEncoderLayers, _numTemporalLayers,
                _numTextLayers, _numHeads, _vocabularySize, _maxSequenceLength));
        }

        // Distribute layers to internal sub-lists
        int idx = 0;
        _patchEmbedding = Layers[idx++];

        for (int i = 0; i < _numFrameEncoderLayers; i++)
            _frameEncoderLayers.Add(Layers[idx++]);

        for (int i = 0; i < _numTemporalLayers; i++)
            _temporalEncoderLayers.Add(Layers[idx++]);

        _videoProjection = Layers[idx++];
        _textTokenEmbedding = Layers[idx++];

        for (int i = 0; i < _numTextLayers; i++)
            _textEncoderLayers.Add(Layers[idx++]);

        _textProjection = Layers[idx++];
        _captionHead = Layers[idx++];

        // Initialize positional embeddings
        _visionClsToken = new Tensor<T>([1, _visionHiddenDim]);
        _visionPositionalEmbeddings = new Tensor<T>([numPatches + 1, _visionHiddenDim]);
        _temporalPositionalEmbeddings = new Tensor<T>([_numFrames, _visionHiddenDim]);
        _textPositionalEmbeddings = new Tensor<T>([_maxSequenceLength, _textHiddenDim]);

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

    private void InitializeMatrix(Tensor<T>? matrix, Random random, double scale)
    {
        if (matrix is null) return;

        for (int i = 0; i < matrix.Shape[0]; i++)
        {
            for (int j = 0; j < matrix.Shape[1]; j++)
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
        // Dot product over the overlapping prefix
        int length = Math.Min(textEmbedding.Length, imageEmbedding.Length);
        if (length == 0)
        {
            return NumOps.Zero;
        }

        var text = textEmbedding.Length == length ? textEmbedding : textEmbedding.GetSubVector(0, length);
        var image = imageEmbedding.Length == length ? imageEmbedding : imageEmbedding.GetSubVector(0, length);
        return Engine.DotProduct(text, image);
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
        var scores = new T[embeddings.Count];

        for (int i = 0; i < embeddings.Count; i++)
        {
            scores[i] = ComputeSimilarity(queryEmbedding, embeddings[i]);
        }

        return TopScoring(scores, topK);
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
        var scores = new T[textEmbeddings.Count];

        for (int i = 0; i < textEmbeddings.Count; i++)
        {
            scores[i] = ComputeSimilarity(textEmbeddings[i], videoEmbedding);
        }

        return TopScoring(scores, topK);
    }

    /// <summary>
    /// The <paramref name="topK"/> best (index, score) pairs, highest score first, ties in index order and
    /// NaN last: the order <c>OrderByDescending(score).Take(topK)</c> gives.
    /// </summary>
    private static List<(int Index, T Score)> TopScoring(T[] scores, int topK)
    {
        int take = Math.Max(0, Math.Min(topK, scores.Length));
        var picked = new List<(int Index, T Score)>(take);
        if (take == 0)
        {
            return picked;
        }

        var order = TieBreakArgMax.StableDescendingOrder1D(new Tensor<T>(scores, new[] { scores.Length })).ToArray();
        for (int rank = 0; rank < take; rank++)
        {
            picked.Add((order[rank], scores[order[rank]]));
        }

        return picked;
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

        var scores = results.Select(r => r.Score).ToArray();
        return TopScoring(scores, 10).Select(top => results[top.Index]);
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
        var captionHead = _captionHead ?? throw new InvalidOperationException("Caption head has not been initialized.");
        var generatedTokens = new List<int>();
        const int BOS_TOKEN = 49406; // CLIP tokenizer BOS
        const int EOS_TOKEN = 49407; // CLIP tokenizer EOS
        const int PAD_TOKEN = 0;

        generatedTokens.Add(BOS_TOKEN);

        // Create context tensor from video embedding
        var contextTensor = Tensor<T>.FromVector(videoContext).Reshape(1, videoContext.Length);

        for (int step = 0; step < maxLength - 1; step++)
        {
            // Get current sequence embedding
            var sequenceEmbedding = GetSequenceEmbedding(generatedTokens);

            // Combine with video context using cross-attention-like mechanism
            var combinedContext = CombineVideoTextContext(contextTensor, sequenceEmbedding);

            // Project to vocabulary logits through caption head
            var logits = captionHead.Forward(combinedContext);

            // Get logits for the last position
            var lastLogits = LastPositionLogits(logits);

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
        var tokenTensor = new Tensor<int>(tokenIds.ToArray(), new[] { seqLen }).Cast<T>();

        var embedded = _textTokenEmbedding.Forward(tokenTensor);

        // Add positional embeddings to the overlapping [rows, cols] corner
        int rows = Math.Min(seqLen, _textPositionalEmbeddings.Shape[0]);
        int cols = Math.Min(embedded.Shape[1], _textPositionalEmbeddings.Shape[1]);
        if (rows > 0 && cols > 0)
        {
            var corner = Engine.TensorNarrow(Engine.TensorNarrow(_textPositionalEmbeddings, 0, 0, rows), 1, 0, cols);
            if (cols < embedded.Shape[1])
            {
                corner = Engine.TensorConcatenate(
                    new[] { corner, new Tensor<T>(new[] { rows, embedded.Shape[1] - cols }) }, 1);
            }

            if (rows < embedded.Shape[0])
            {
                corner = Engine.TensorConcatenate(
                    new[] { corner, new Tensor<T>(new[] { embedded.Shape[0] - rows, embedded.Shape[1] }) }, 0);
            }

            embedded = Engine.TensorAdd(embedded, corner);
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
        var videoVal = Engine.TensorNarrow(Engine.TensorNarrow(videoContext, 0, 0, 1), 1, 0, outputDim).Reshape(outputDim);
        var textVal = PadOrTruncate(textPooled, outputDim);

        // Gated combination
        T gate = NumOps.FromDouble(0.5);
        var combined = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(videoVal, gate),
            Engine.TensorMultiplyScalar(textVal, NumOps.Subtract(NumOps.One, gate)));

        return combined.Reshape(1, outputDim);
    }

    /// <summary>
    /// The first <paramref name="length"/> entries of <paramref name="values"/>, zero-padded when it is shorter.
    /// </summary>
    private Tensor<T> PadOrTruncate(Vector<T> values, int length)
    {
        int kept = Math.Min(values.Length, length);
        var head = Tensor<T>.FromVector(kept == values.Length ? values : values.GetSubVector(0, kept));
        return kept == length
            ? head
            : Engine.TensorConcatenate(new[] { head, new Tensor<T>(new[] { length - kept }) }, 0);
    }

    /// <summary>
    /// Logits of the first row (or the vector itself), limited to the vocabulary size.
    /// </summary>
    private T[] LastPositionLogits(Tensor<T> logits)
    {
        bool rows = logits.Shape.Length > 1;
        int vocabSize = rows ? logits.Shape[1] : logits.Shape[0];
        int count = Math.Min(vocabSize, _vocabularySize);
        var first = rows
            ? Engine.TensorNarrow(Engine.TensorNarrow(logits, 0, 0, 1), 1, 0, count)
            : Engine.TensorNarrow(logits, 0, 0, count);
        return first.Contiguous().ToArray();
    }

    private static Tensor<T> Full(int length, T value) => Tensor<T>.CreateDefault(new[] { length }, value);

    /// <summary>
    /// Samples the next token using temperature scaling and nucleus (top-p) sampling.
    /// </summary>
    private int SampleNextToken(T[] logits, double temperature, double topP)
    {
        if (logits.Length == 0)
            return 0;

        int n = logits.Length;

        // Apply temperature, then softmax
        var raw = new Tensor<T>(logits, new[] { n });
        var scaled = Engine.TensorDivide(raw, Full(n, NumOps.FromDouble(Math.Max(temperature, 0.01))));
        var probs = Engine.Softmax(scaled, -1);

        // Nucleus (top-p) sampling: most probable first (stable, NaN last), up to and including the token
        // that brings the running total to topP
        var order = TieBreakArgMax.StableDescendingOrder1D(probs);
        var sortedProbs = Engine.TensorTake(probs, order);
        var cumulativeProb = Engine.TensorCumSum(sortedProbs, 0);
        var reached = Engine.TensorGreaterEqual(cumulativeProb, Full(n, NumOps.FromDouble(topP)));
        int nucleusSize = Engine.TensorCountNonzero(reached) == 0 ? n : TieBreakArgMax.FirstMaxIndex(reached) + 1;

        // Renormalize nucleus probabilities (their sum is the running total at the last nucleus token)
        var nucleusProbs = Engine.TensorNarrow(sortedProbs, 0, 0, nucleusSize);
        T nucleusSum = cumulativeProb.GetFlat(nucleusSize - 1);
        var normalizedProbs = Engine.TensorDivide(nucleusProbs, Full(nucleusSize, nucleusSum));

        // Sample from nucleus: the first token whose running total reaches the draw
        var random = RandomHelper.Shared;
        double sample = random.NextDouble();
        var cumulative = Engine.TensorCumSum(normalizedProbs, 0);
        var hit = Engine.TensorLessEqual(Full(nucleusSize, NumOps.FromDouble(sample)), cumulative);
        int chosen = Engine.TensorCountNonzero(hit) == 0 ? nucleusSize - 1 : TieBreakArgMax.FirstMaxIndex(hit);
        return order.GetFlat(chosen);
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
        var captionHead = _captionHead ?? throw new InvalidOperationException("Caption head has not been initialized.");
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
        var videoVal = Tensor<T>.FromVector(videoContext);
        var questionVal = PadOrTruncate(questionContext, contextDim);

        // Fuse video and question embeddings with attention-weighted mechanism:
        // weight = clamp(0.5 + 0.2 * video * question, 0.3, 0.7), NaN passed through
        var dotProduct = Engine.TensorMultiply(videoVal, questionVal);
        var rawWeight = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(dotProduct, NumOps.FromDouble(0.2)), NumOps.FromDouble(0.5));
        var weight = Engine.TensorWhere(
            Engine.TensorIsNan(rawWeight),
            rawWeight,
            Engine.TensorClamp(rawWeight, NumOps.FromDouble(0.3), NumOps.FromDouble(0.7)));
        var combinedContextTensor = Engine.TensorAdd(
                Engine.TensorMultiply(weight, videoVal),
                Engine.TensorMultiply(Engine.ScalarMinusTensor(NumOps.One, weight), questionVal))
            .Reshape(1, contextDim);

        // Check if this is a yes/no question for constrained decoding
        bool isYesNoQuestion = IsYesNoQuestion(question);

        for (int step = 0; step < maxLength; step++)
        {
            // Get current sequence embedding
            var sequenceEmbedding = GetSequenceEmbedding(generatedTokens);

            // Combine with video-question context
            var decoderInput = CombineVideoTextContext(combinedContextTensor, sequenceEmbedding);

            // Project to vocabulary logits
            var logits = captionHead.Forward(decoderInput);

            // Get logits for the last position
            var lastLogits = LastPositionLogits(logits);

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
        var frameTensors = new Tensor<T>[frameList.Count];

        for (int f = 0; f < frameList.Count; f++)
        {
            frameTensors[f] = EncodeFrameNativeTensor(frameList[f]); // [hiddenDim]
        }

        // Stack into [numFrames, hiddenDim] via Engine
        return Engine.TensorStack(frameTensors, axis: 0);
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeTemporalSimilarityMatrix(
        IEnumerable<Tensor<T>> video1Frames,
        IEnumerable<Tensor<T>> video2Frames)
    {
        var features1 = ExtractFrameFeatures(video1Frames); // [numFrames1, featureDim]
        var features2 = ExtractFrameFeatures(video2Frames); // [numFrames2, featureDim]

        // Similarity = features1 @ features2^T via Engine matmul
        var features2T = Engine.TensorTranspose(features2); // [featureDim, numFrames2]
        return Engine.TensorMatMul(features1, features2T); // [numFrames1, numFrames2]
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

    /// <summary>
    /// Encodes a video as a normalized embedding tensor using tape-tracked Engine operations.
    /// Returns a 1D tensor [embeddingDim].
    /// </summary>
    private Tensor<T> EncodeVideoNativeTensor(List<Tensor<T>> frames)
    {
        // Sample frames if too many
        var sampledFrames = SampleFrames(frames, _numFrames);

        // Cache frames for gradient checkpointing during backward
        if (IsTrainingMode)
        {
            _cachedTrainingFrames = sampledFrames;
        }

        // Encode each frame into 1D tensors and stack via Engine
        var frameTensors = new Tensor<T>[sampledFrames.Count];
        for (int f = 0; f < sampledFrames.Count; f++)
        {
            frameTensors[f] = EncodeFrameNativeTensor(sampledFrames[f]); // [hiddenDim]
        }

        // Stack frame features: [numFrames, hiddenDim]
        var stackedFeatures = Engine.TensorStack(frameTensors, axis: 0);

        // Add temporal positional embeddings via Engine
        if (_temporalPositionalEmbeddings is not null)
        {
            var tempPosEmb = _temporalPositionalEmbeddings;
            int numSampledFrames = stackedFeatures.Shape[0];
            int featureDim = stackedFeatures.Shape[1];
            int sliceRows = Math.Min(numSampledFrames, tempPosEmb.Shape[0]);
            int sliceCols = Math.Min(featureDim, tempPosEmb.Shape[1]);
            var posSlice = Engine.TensorSlice(tempPosEmb, [0, 0], [sliceRows, sliceCols]);
            stackedFeatures = Engine.TensorAdd(stackedFeatures, posSlice);
        }

        // Apply temporal encoder layers
        var temporalFeatures = stackedFeatures;
        foreach (var layer in _temporalEncoderLayers)
        {
            temporalFeatures = layer.Forward(temporalFeatures);
        }

        // Mean pool along sequence dimension: [hiddenDim]
        var pooled = MeanPoolTensor(temporalFeatures);

        // Project to embedding space
        if (_videoProjection is not null)
        {
            // Reshape [hiddenDim] -> [1, hiddenDim] for projection layer
            var pooled2d = Engine.Reshape(pooled, [1, pooled.Shape[0]]);
            var projected = _videoProjection.Forward(pooled2d);
            // Extract [embeddingDim] from [1, embeddingDim]
            pooled = Engine.Reshape(projected, [projected.Shape[^1]]);
        }

        // Paper-faithful contrastive training (Xu et al. 2021, §3.2): VideoCLIP average-pools the
        // token sequence and contrasts the RAW pooled features with InfoNCE at temperature 1.0 — it
        // does NOT L2-normalize the embeddings. We only normalize on the INFERENCE path (so the public
        // similarity API returns cosine similarities). Normalizing on the TRAINING path additionally
        // routes every gradient through the L2-norm Jacobian (I - yyᵀ)/‖x‖, whose 1/‖x‖ term produces a
        // NaN gradient that drives the parameters to NaN on the first optimizer step (the batch's pooled
        // embedding can sit near the origin for the random-init/random-target smoke fixture). Returning
        // the un-normalized pooled features in training matches the paper and removes that NaN source;
        // the encoder's LayerNorms keep the pooled features O(1) so the contrastive softmax stays stable.
        if (IsTrainingMode)
        {
            return pooled;
        }

        // L2 normalize via Engine ops (inference / similarity path only).
        return NormalizeTensor(pooled);
    }

    /// <summary>
    /// Legacy video encoder returning Vector for public API and non-training paths.
    /// </summary>
    private Vector<T> EncodeVideoNative(List<Tensor<T>> frames)
    {
        var tensor = EncodeVideoNativeTensor(frames);
        int dim = tensor.Shape[0];
        return PrefixVector(tensor, dim, dim);
    }

    /// <summary>
    /// Encodes a single frame using Engine operations for tape-tracked gradient flow.
    /// Returns a 1D tensor [visionHiddenDim] representing the CLS token feature.
    /// </summary>
    private Tensor<T> EncodeFrameNativeTensor(Tensor<T> frame)
    {
        if (_patchEmbedding is null || _visionClsToken is null || _visionPositionalEmbeddings is null)
            throw new InvalidOperationException("Native layers not initialized.");

        // Patch embedding
        var patches = _patchEmbedding.Forward(frame);

        // Add CLS token via Engine concat
        var withCls = PrependClsToken(patches, _visionClsToken);

        // Add positional embeddings via Engine add
        var positioned = AddPositionalEmbeddings(withCls, _visionPositionalEmbeddings);

        // Apply frame encoder layers
        var current = positioned;
        foreach (var layer in _frameEncoderLayers)
        {
            current = layer.Forward(current);
        }

        // Take CLS token (row 0) as the frame representation. Use TAPE-TRACKED Engine ops
        // (TensorNarrow + Reshape), NOT a raw Data.Span copy into a fresh Tensor: the raw copy
        // produced a tensor with no GradFn, severing the ViT frame encoder from the gradient tape so
        // its patch-embedding / attention / FFN weights received NO gradient and never trained — the
        // opposite of the paper, where the frame encoder is learned end-to-end under the contrastive
        // objective (Xu et al. 2021). Slicing row 0 through the Engine keeps the encoder on the tape.
        int hiddenDim = current.Shape[^1];
        var clsRow = Engine.TensorNarrow(current, 0, 0, 1); // [seqLen, hiddenDim] -> [1, hiddenDim]
        return Engine.Reshape(clsRow, new[] { hiddenDim });
    }

    /// <summary>
    /// Legacy frame encoder returning Vector for non-training paths.
    /// </summary>
    private Vector<T> EncodeFrameNative(Tensor<T> frame)
    {
        var tensor = EncodeFrameNativeTensor(frame);
        int dim = Math.Min(_visionHiddenDim, tensor.Shape[0]);
        return PrefixVector(tensor, dim, _visionHiddenDim);
    }

    private Vector<T> EncodeVideoOnnx(List<Tensor<T>> frames)
    {
        if (_videoEncoder is null)
            throw new InvalidOperationException("ONNX video encoder not initialized.");

        var sampledFrames = SampleFrames(frames, _numFrames);

        foreach (var frame in sampledFrames)
            if (frame.Rank != 3 || frame.Shape[0] != _onnxChannels
                || frame.Shape[1] != _imageSize || frame.Shape[2] != _imageSize)
                throw new ArgumentException($"ONNX video frames must have shape [{_onnxChannels}, {_imageSize}, {_imageSize}].", nameof(frames));

        int channels = sampledFrames[0].Shape[0];
        int height = sampledFrames[0].Shape[1];
        int width = sampledFrames[0].Shape[2];
        int numSampledFrames = sampledFrames.Count;

        // Create input tensor [1, numFrames, channels, height, width]
        var stacked = Engine.TensorConcatenate(
            sampledFrames.Select(frame => frame.Reshape(1, channels, height, width)).ToArray(), 0);
        var inputArray = stacked.Contiguous().Cast<float>().ToArray();

        var inputTensor = new OnnxTensors.DenseTensor<float>(inputArray, [1, numSampledFrames, channels, height, width]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", inputTensor)
        };

        using var results = _videoEncoder.Run(inputs);
        var outputTensor = results.First(result => result.Name == _videoOutputName).AsTensor<float>();
        return Normalize(OnnxEmbeddingContract.Read<T>(outputTensor, _embeddingDimension, EmbeddingLayouts, OnnxModelRole.VideoEncoder));
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

    /// <summary>
    /// Encodes text as a normalized embedding tensor using tape-tracked Engine operations.
    /// Returns a 1D tensor [embeddingDim].
    /// </summary>
    private Tensor<T> EncodeTextNativeTensor(string text)
    {
        if (_textTokenEmbedding is null || _textPositionalEmbeddings is null || _textProjection is null)
            throw new InvalidOperationException("Native text layers not initialized.");

        var encoded = _tokenizer.Encode(text);
        var inputIds = encoded.TokenIds;

        // Truncate or pad
        int kept = Math.Min(inputIds.Count, _maxSequenceLength);
        var paddedIds = inputIds.Take(kept).Concat(Enumerable.Repeat(0, _maxSequenceLength - kept)).ToArray();
        var tokenTensor = new Tensor<int>(paddedIds, new[] { _maxSequenceLength }).Cast<T>();
        var embedded = _textTokenEmbedding.Forward(tokenTensor);

        // Add positional embeddings via Engine
        var positioned = AddPositionalEmbeddings(embedded, _textPositionalEmbeddings);

        // Apply text encoder layers
        var current = positioned;
        foreach (var layer in _textEncoderLayers)
        {
            current = layer.Forward(current);
        }

        // Mean pool along sequence dimension: [hiddenDim]
        var pooled = MeanPoolTensor(current);

        // Project to embedding space: reshape [hiddenDim] -> [1, hiddenDim]
        var pooled2d = Engine.Reshape(pooled, [1, pooled.Shape[0]]);
        var projected = _textProjection.Forward(pooled2d);

        // Extract [embeddingDim] from [1, embeddingDim]
        var result = Engine.Reshape(projected, [projected.Shape[^1]]);

        // L2 normalize via Engine ops
        return NormalizeTensor(result);
    }

    /// <summary>
    /// Legacy text encoder returning Vector for public API and non-training paths.
    /// </summary>
    private Vector<T> EncodeTextNative(string text)
    {
        var tensor = EncodeTextNativeTensor(text);
        int dim = Math.Min(_embeddingDimension, tensor.Shape[0]);
        return PrefixVector(tensor, dim, _embeddingDimension);
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
        var outputTensor = results.First(result => result.Name == _textOutputName).AsTensor<float>();
        return Normalize(OnnxEmbeddingContract.Read<T>(outputTensor, _embeddingDimension, EmbeddingLayouts, OnnxModelRole.TextEncoder));
    }

    #endregion

    #region Helper Methods

    private Tensor<T> PrependClsToken(Tensor<T> sequence, Tensor<T> clsToken)
    {
        // Convert CLS token matrix [1, hiddenDim] to tensor and expand for concat
        var clsTensor = clsToken; // [1, hiddenDim]

        // Concatenate CLS token with sequence along axis 0: [1+seqLen, hiddenDim]
        return Engine.TensorConcatenate([clsTensor, sequence], axis: 0);
    }

    private Tensor<T> AddPositionalEmbeddings(Tensor<T> sequence, Tensor<T> posEmbeddings)
    {
        int seqLen = sequence.Shape[0];
        int hiddenDim = sequence.Shape[1];

        // Convert positional embeddings matrix to tensor
        var posEmbTensor = posEmbeddings;

        // Slice to match sequence length if needed
        int sliceLen = Math.Min(seqLen, posEmbTensor.Shape[0]);
        int sliceDim = Math.Min(hiddenDim, posEmbTensor.Shape[1]);
        var posSlice = Engine.TensorSlice(posEmbTensor, [0, 0], [sliceLen, sliceDim]);

        // If sequence is longer than positional embeddings, slice sequence for the add
        // then concatenate the remainder. In practice they should match.
        if (sliceLen < seqLen)
        {
            var seqSlice = Engine.TensorSlice(sequence, [0, 0], [sliceLen, hiddenDim]);
            var added = Engine.TensorAdd(seqSlice, posSlice);
            var remainder = Engine.TensorSlice(sequence, [sliceLen, 0], [seqLen - sliceLen, hiddenDim]);
            return Engine.TensorConcatenate([added, remainder], axis: 0);
        }

        return Engine.TensorAdd(sequence, posSlice);
    }

    /// <summary>
    /// Mean-pools a 2D tensor [seqLen, hiddenDim] along axis 0, returning a 1D tensor [hiddenDim].
    /// Uses Engine.ReduceMean for tape-tracked gradient flow.
    /// </summary>
    private Tensor<T> MeanPoolTensor(Tensor<T> tensor)
    {
        // ReduceMean along axis 0 (sequence dimension), producing [hiddenDim]
        return Engine.ReduceMean(tensor, [0], keepDims: false);
    }

    /// <summary>
    /// Legacy mean-pool returning Vector for non-training paths (ONNX, caption generation).
    /// </summary>
    private Vector<T> MeanPool(Tensor<T> tensor)
    {
        var pooled = MeanPoolTensor(tensor);
        int dim = pooled.Shape[0];
        return PrefixVector(pooled, dim, dim);
    }

    /// <summary>
    /// A zero vector of <paramref name="length"/> whose first <paramref name="copied"/> entries are the
    /// tensor's first flat entries.
    /// </summary>
    private static Vector<T> PrefixVector(Tensor<T> tensor, int copied, int length)
    {
        var result = new Vector<T>(length);
        tensor.Contiguous().AsSpan().Slice(0, copied).CopyTo(result.AsWritableSpan());
        return result;
    }

    /// <summary>
    /// L2-normalizes a 1D tensor using Engine operations for tape-tracked gradient flow.
    /// Computes x / (||x||_2 + epsilon).
    /// </summary>
    private Tensor<T> NormalizeTensor(Tensor<T> tensor)
    {
        // Compute squared values
        var squared = Engine.TensorMultiply(tensor, tensor);
        // Sum all elements to get ||x||^2
        var allAxes = Enumerable.Range(0, squared.Shape.Length).ToArray();
        var sumSquared = Engine.ReduceSum(squared, allAxes, keepDims: true);
        // Add epsilon for numerical stability, then sqrt to get ||x||. The epsilon is 1e-6, NOT a
        // token 1e-12: ||x||^2 is a sum of squares that is mathematically >= 0, but a parallel-BLAS
        // reduction can return a slightly NEGATIVE value from summation-order rounding (~float32
        // machine-eps x num_terms). With a 1e-12 floor that negative survives into sqrt -> NaN, which
        // then poisons the whole embedding (VideoCLIP DifferentInputs_AfterTraining saw L2 distance =
        // NaN on the parallel CI runner while passing locally). A 1e-6 floor keeps the radicand
        // positive under that drift and bounds the 1/||x|| normalize backward as the single-pair
        // contrastive objective drives the embedding norm toward zero, while staying negligible for a
        // well-formed unit embedding (||x||^2 ~ dim).
        var norm = Engine.TensorSqrt(Engine.TensorAddScalar(sumSquared, NumOps.FromDouble(1e-6)));
        // Divide element-wise: x / ||x||
        return Engine.TensorDivide(tensor, norm);
    }

    /// <summary>
    /// Legacy normalize returning Vector for non-training paths.
    /// </summary>
    private Vector<T> Normalize(Vector<T> vector)
    {
        return VectorHelper.Normalize(vector);
    }

    private List<T> Softmax(List<T> values)
    {
        if (values.Count == 0)
        {
            throw new InvalidOperationException("Sequence contains no elements");
        }

        var probabilities = Engine.Softmax(new Tensor<T>(values.ToArray(), new[] { values.Count }), -1);
        return new List<T>(probabilities.ToArray());
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {

        SetTrainingMode(false);

        return Accelerate(input, () =>
        {
            var frames = ParseInputToFrames(input);
            var embedding = GetVideoEmbedding(frames);
            return Tensor<T>.FromVector(embedding).Reshape(1, embedding.Length);
        });
    }

    /// <summary>
    /// Parses a batched input tensor into a list of individual frame tensors.
    /// Handles rank-4 [numFrames, C, H, W], rank-3 [C, H, W], and other shapes.
    /// </summary>
    private List<Tensor<T>> ParseInputToFrames(Tensor<T> input)
    {
        var frames = new List<Tensor<T>>();

        if (input.Shape.Length == 4)
        {
            // Input is [numFrames, channels, height, width] - split into individual 3D frames
            int numFrames = input.Shape[0];
            int c = input.Shape[1];
            int h = input.Shape[2];
            int w = input.Shape[3];
            int frameSize = c * h * w;
            var inputSpan = input.Data.Span;

            for (int f = 0; f < numFrames; f++)
            {
                var frameData = new T[frameSize];
                inputSpan.Slice(f * frameSize, frameSize).CopyTo(frameData);
                frames.Add(new Tensor<T>(new[] { c, h, w }, new Vector<T>(frameData)));
            }
        }
        else if (input.Shape.Length == 3)
        {
            // Input is [channels, height, width] - treat as single frame
            frames.Add(input);
        }
        else
        {
            // Fallback for other shapes
            frames.Add(input);
        }

        return frames;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// VideoCLIP overrides ForwardForTraining because its forward pass extracts
    /// video embeddings through a custom pipeline (frame encoding → temporal
    /// encoding → projection → normalization) that isn't captured in the Layers list.
    /// Unlike Predict, this does NOT reset training mode so layers stay in training
    /// mode for tape-based gradient computation.
    /// All operations use Engine methods for tape-tracked gradient flow.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        // Parse input into frames
        var frames = ParseInputToFrames(input);

        if (_useNativeMode)
        {
            // Use tape-tracked tensor encoding pipeline
            var embedding = EncodeVideoNativeTensor(frames);
            // Reshape [embeddingDim] -> [1, embeddingDim] via Engine
            return Engine.Reshape(embedding, [1, embedding.Shape[0]]);
        }
        else
        {
            // ONNX mode: not tape-tracked, fallback to legacy path
            var embedding = EncodeVideoOnnx(frames);
            return Tensor<T>.FromVector(embedding).Reshape(1, embedding.Length);
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        if (!_useNativeMode)
        {
            var configuration = OnnxConfiguration
                ?? throw new InvalidOperationException("ONNX configuration has not been initialized.");
            return new ModelMetadata<T>
            {
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "ImageSize", _imageSize }, { "EmbeddingDimension", _embeddingDimension },
                    { "MaxSequenceLength", _maxSequenceLength }, { "NumFrames", _numFrames },
                    { "FrameRate", _frameRate }, { "UseNativeMode", false },
                    { nameof(OnnxConfiguration), configuration }
                },
                ModelDataProvider = () => SerializeForMetadata()
            };
        }
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ImageSize", _imageSize },
                { "EmbeddingDimension", _embeddingDimension },
                { "MaxSequenceLength", _maxSequenceLength },
                { "NumFrames", _numFrames },
                { "FrameRate", _frameRate },
                { "TemporalAggregation", _temporalAggregation.ToString() },
                { "VisionHiddenDim", _visionHiddenDim },
                { "TextHiddenDim", _textHiddenDim },
                { "NumFrameEncoderLayers", _numFrameEncoderLayers },
                { "NumTemporalLayers", _numTemporalLayers },
                { "NumTextLayers", _numTextLayers },
                { "VocabularySize", _vocabularySize },
                { "UseNativeMode", _useNativeMode }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    private static void SerializeMatrix(BinaryWriter writer, Tensor<T>? matrix)
    {
        if (matrix is null)
        {
            writer.Write(0);
            writer.Write(0);
            return;
        }
        writer.Write(matrix.Shape[0]);
        writer.Write(matrix.Shape[1]);

        // Row-major [rows, cols] in the double wire format
        var engine = AiDotNetEngine.Current;
        var rowsByCols = engine.TensorNarrow(engine.TensorNarrow(matrix, 0, 0, matrix.Shape[0]), 1, 0, matrix.Shape[1]);
        foreach (double value in rowsByCols.Contiguous().Cast<double>().ToArray())
            writer.Write(value);
    }

    private static Tensor<T>? DeserializeMatrix(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        if (rows == 0 && cols == 0)
            return null;
        var stored = new double[rows * cols];
        for (int i = 0; i < stored.Length; i++)
            stored[i] = reader.ReadDouble();
        return new Tensor<double>(stored, new[] { rows, cols }).Cast<T>();
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

        return StackRows(embeddings);
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

        return StackRows(embeddings);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(double[] imageData, IEnumerable<string> labels)
    {
        var tensor = ConvertToTensor(imageData);
        return ZeroShotClassify(tensor, labels);
    }

    /// <summary>
    /// Stacks embeddings as the rows of a matrix as wide as the first one.
    /// </summary>
    private static Matrix<T> StackRows(List<Vector<T>> embeddings)
    {
        int cols = embeddings[0].Length;
        var matrix = new Matrix<T>(embeddings.Count, cols);
        var rows = matrix.AsWritableSpan();
        for (int i = 0; i < embeddings.Count; i++)
        {
            embeddings[i].AsSpan().CopyTo(rows.Slice(i * cols, cols));
        }

        return matrix;
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

        // Channel-major [C, H, W], converted from the caller's doubles
        return new Tensor<double>(imageData, new[] { channels, size, size }).Cast<T>();
    }

    #endregion

}





