using AiDotNet.LearningRateSchedulers;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Understanding;

/// <summary>
/// VideoCLIP model for video-text understanding and retrieval.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> VideoCLIP learns to understand both videos and text descriptions
/// in a shared "embedding space" where similar concepts are close together.
///
/// Key capabilities:
/// - Video-to-Text Search: Find text descriptions that match a video
/// - Text-to-Video Search: Find videos that match a text query
/// - Zero-Shot Classification: Classify videos into categories without training
/// - Video Captioning: Generate descriptions for videos
/// - Video Question Answering: Answer questions about video content
///
/// The model creates embeddings (numerical representations) for both videos and text
/// that can be compared using similarity measures. Videos and their corresponding
/// descriptions will have similar embeddings.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Contrastive learning on video-text pairs
/// - Temporal transformer for video understanding
/// - Text transformer for language understanding
/// - Joint embedding space with cosine similarity
/// - Pre-trained on large-scale video-text datasets
/// </para>
/// <para>
/// <b>Reference:</b> Xu et al., "VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding"
/// EMNLP 2021.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a VideoCLIP model for video-text understanding
/// var videoCLIP = new VideoCLIP&lt;double&gt;();
///
/// // Or configure with custom embedding dimensions
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 400);
/// var model = new VideoCLIP&lt;double&gt;(architecture, numFrames: 32, embeddingDim: 512);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding",
    "https://arxiv.org/abs/2109.14084",
    Year = 2021,
    Authors = "Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, Christoph Feichtenhofer")]
public class VideoCLIP<T> : NeuralNetworkBase<T>
{
    private readonly VideoCLIPVideoOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numFrames;
    private readonly int _embeddingDim;
    private readonly int _textMaxLength;
    private readonly int _vocabSize;
    private readonly double _temperature;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    // Video encoder components.
    //
    // These are VIEWS into Layers, not a second ownership graph: the constructor builds the layers,
    // adds them to Layers, and then binds these fields by index. Deserialization REPLACES every
    // entry in Layers with a restored instance, so the single-layer views below cannot be readonly —
    // BindLayerViewsFromLayers() re-points them at the restored instances. Leaving them bound to the
    // constructor's now-orphaned layers made the explicit forward run on fresh random weights while
    // Layers held the trained ones, so a trained clone predicted differently from its original with
    // provably identical parameters (measured: 48173/48173 parameters and both embedding tables
    // equal to the last bit, outputs differing by 1.7e+00, and gradients on Layers pinned at zero
    // because the backward flowed through the orphans instead).
    private readonly List<ConvolutionalLayer<T>> _videoEncoder;
    private readonly List<ConvolutionalLayer<T>> _temporalTransformer;
    private ConvolutionalLayer<T> _videoProjection;

    // Text encoder components
    // Proper CLIP-style token embedding: embedding lookup table [vocab_size, hidden_dim]
    private readonly Tensor<T> _tokenEmbeddingTable;          // Embedding lookup table
    private readonly Tensor<T> _positionalEmbeddingTable;     // Learned positional embeddings
    private readonly List<ConvolutionalLayer<T>> _textTransformerQKV;      // QKV projections
    private readonly List<ConvolutionalLayer<T>> _textTransformerAttnProj; // Attention output
    private readonly List<ConvolutionalLayer<T>> _textTransformerFFN1;     // FFN expand
    private readonly List<ConvolutionalLayer<T>> _textTransformerFFN2;     // FFN contract
    private ConvolutionalLayer<T> _textProjection;
    private readonly int _textHiddenDim;

    /// <summary>
    /// Width of the vision and text trunks. CLIP ViT-B/32's 768 stays the default; it was a hardcoded
    /// local before, which left the model at ~171M parameters no matter how small a clip it was given.
    /// </summary>
    private readonly int _hiddenDim;

    // Shared components
    private ConvolutionalLayer<T> _logitScale;

    // Tokenizer for text encoding
    private readonly BpeTokenizer? _tokenizer;
    private readonly EncodingOptions _encodingOptions;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether training is supported.
    /// </summary>
    public override bool SupportsTraining => true;


    /// <summary>
    /// Gets the video frame height.
    /// </summary>
    internal int InputHeight => _height;

    /// <summary>
    /// Gets the video frame width.
    /// </summary>
    internal int InputWidth => _width;

    /// <summary>
    /// Gets the number of frames processed.
    /// </summary>
    internal int NumFrames => _numFrames;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    internal int EmbeddingDimension => _embeddingDim;

    /// <summary>
    /// Gets or sets the temperature parameter for softmax.
    /// </summary>
    internal double Temperature { get; set; }

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public VideoCLIP()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 224, inputWidth: 224, inputDepth: 3,
            outputSize: 400))
    {
    }

    /// <summary>
    /// Initializes a new instance of the VideoCLIP class.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFrames">Number of video frames to process.</param>
    /// <param name="embeddingDim">Dimension of the shared embedding space.</param>
    /// <param name="textMaxLength">Maximum text sequence length.</param>
    /// <param name="vocabSize">Vocabulary size for text encoding.</param>
    /// <param name="temperature">Temperature for softmax scaling.</param>
    /// <param name="vocabPath">Optional path to CLIP vocabulary JSON file for production tokenization.</param>
    /// <param name="mergesPath">Optional path to CLIP BPE merges file for production tokenization.</param>
    /// <param name="options">Video and text encoder configuration.</param>
    /// <param name="optimizer">Optional optimizer. Defaults to AdamW configured by <paramref name="options"/>.</param>
    /// <param name="lossFunction">Optional objective for generic embedding training. Defaults to mean squared error.</param>
    /// <remarks>
    /// <para>
    /// <b>For Production Use:</b> Provide vocabPath and mergesPath to use proper CLIP tokenization.
    /// Download these files from HuggingFace's openai/clip-vit-base-patch32 repository:
    /// - vocab.json: Token vocabulary mapping
    /// - merges.txt: BPE merge rules
    ///
    /// <b>For Testing:</b> Omit vocabPath and mergesPath to use a simple test tokenizer.
    /// </para>
    /// </remarks>
    public VideoCLIP(
        NeuralNetworkArchitecture<T> architecture,
        int numFrames = 32,
        int embeddingDim = 512,
        int textMaxLength = 77,
        int vocabSize = 49408,
        double temperature = 0.07,
        int hiddenDim = 768,
        string? vocabPath = null,
        string? mergesPath = null,
        VideoCLIPVideoOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new VideoCLIPVideoOptions();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 224;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFrames = numFrames;
        _embeddingDim = embeddingDim;
        _textMaxLength = textMaxLength;
        _vocabSize = vocabSize;
        _temperature = temperature;
        Temperature = temperature;

        // Initialize tokenizer
        if (vocabPath is not null && mergesPath is not null &&
            !string.IsNullOrEmpty(vocabPath) && !string.IsNullOrEmpty(mergesPath))
        {
            // Use proper CLIP tokenization from pretrained files
            _tokenizer = ClipTokenizerFactory.FromPretrained(vocabPath, mergesPath);
        }
        else
        {
            // Use simple tokenizer for testing (will warn in logs)
            _tokenizer = ClipTokenizerFactory.CreateSimple();
        }
        _encodingOptions = ClipTokenizerFactory.GetDefaultEncodingOptions(_textMaxLength);

        _videoEncoder = [];
        _temporalTransformer = [];
        _textTransformerQKV = [];
        _textTransformerAttnProj = [];
        _textTransformerFFN1 = [];
        _textTransformerFFN2 = [];

        Guard.Positive(_options.NumSpatialBlocks, nameof(_options.NumSpatialBlocks));
        Guard.Positive(_options.NumTemporalBlocks, nameof(_options.NumTemporalBlocks));
        Guard.Positive(_options.NumTextBlocks, nameof(_options.NumTextBlocks));
        Guard.Positive(_options.LearningRate, nameof(_options.LearningRate));

        int effectiveHiddenDim = options is null ? hiddenDim : _options.HiddenDimension;
        Guard.Positive(effectiveHiddenDim, nameof(hiddenDim));
        _hiddenDim = effectiveHiddenDim;
        _textHiddenDim = effectiveHiddenDim;
        int numSpatialBlocks = _options.NumSpatialBlocks;
        int numTemporalBlocks = _options.NumTemporalBlocks;
        int numTextBlocks = _options.NumTextBlocks;

        // Check for user-provided custom layers
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            var layers = LayerHelper<T>.CreateVideoCLIPLayers(
                channels: _channels, height: _height, width: _width,
                hiddenDim: effectiveHiddenDim, embeddingDim: _embeddingDim,
                numSpatialBlocks: numSpatialBlocks, numTemporalBlocks: numTemporalBlocks,
                numTextBlocks: numTextBlocks, numFrames: _numFrames, textMaxLength: _textMaxLength).ToList();
            Layers.AddRange(layers);
        }

        // Distribute layers to sub-lists for forward pass
        BindLayerViewsFromLayers();

        // Initialize embedding tables (not part of layer list)
        _tokenEmbeddingTable = new Tensor<T>([_vocabSize, hiddenDim]);
        InitializeEmbeddingTable(_tokenEmbeddingTable, _vocabSize, hiddenDim);
        _positionalEmbeddingTable = new Tensor<T>([_textMaxLength, hiddenDim]);
        InitializeEmbeddingTable(_positionalEmbeddingTable, _textMaxLength, hiddenDim);

        // Paper-faithful training configuration (Xu et al. 2021, arXiv:2109.14084, Training Details):
        // "Adam ... with betas of (0.9, 0.98), an initial learning rate of 5e-5, 1000 steps of
        // warm-up, and a polynomial decay learning rate schedule ... Gradients are clipped at 2.0."
        //
        // What was here instead: AdamW at 1e-4 with clipping at 1.0 and default betas (0.9, 0.999).
        // Four departures from the paper, and the learning rate was twice what VideoCLIP specifies,
        // which is what the more-data invariant caught -- an extra step on identical data raised the
        // loss from 0.6702 to 0.7258.
        //
        // Every value stays overridable: pass `optimizer` for a different optimizer entirely, or set
        // VideoCLIPVideoOptions.LearningRate / Beta1 / Beta2 / MaxGradientNorm / WarmupSteps /
        // TotalTrainingSteps / DecayPower.
        //
        // The schedule is warm-up followed by polynomial decay, stepped per batch because the paper
        // counts warm-up in optimizer STEPS, not epochs.
        int warmupSteps = Math.Max(0, _options.WarmupSteps);
        int decaySteps = Math.Max(1, _options.TotalTrainingSteps - warmupSteps);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                Beta1 = _options.Beta1,
                Beta2 = _options.Beta2,
                EnableGradientClipping = true,
                MaxGradientNorm = _options.MaxGradientNorm,
                // "1000 steps of warm-up, and a polynomial decay learning rate schedule."
                // Stepped per batch because the paper counts warm-up in optimizer STEPS, not epochs.
                LearningRateScheduler = warmupSteps > 0
                    ? new SequentialLRScheduler(
                        new ILearningRateScheduler[]
                        {
                            new LinearWarmupScheduler(_options.LearningRate, warmupSteps),
                            new PolynomialLRScheduler(_options.LearningRate, decaySteps, _options.DecayPower)
                        },
                        new[] { warmupSteps })
                    : new PolynomialLRScheduler(_options.LearningRate, decaySteps, _options.DecayPower),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch
            });
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Encodes a video into an embedding vector.
    /// </summary>
    /// <param name="videoFrames">Input video [T, C, H, W] or [B, T, C, H, W].</param>
    /// <returns>Video embedding [EmbeddingDim] or [B, EmbeddingDim].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts a video into a numerical vector (embedding).
    /// Videos with similar content will have similar embeddings.
    /// </para>
    /// </remarks>
    public Tensor<T> EncodeVideo(Tensor<T> videoFrames)
    {
        bool hasBatch = videoFrames.Rank == 5;
        if (!hasBatch)
        {
            videoFrames = AddBatchDimension5D(videoFrames);
        }

        // Process each frame through spatial encoder
        var frameFeatures = ProcessFrames(videoFrames);

        // Apply temporal transformer
        var temporalFeatures = ApplyTemporalAttention(frameFeatures);

        // Global average pooling
        var pooled = GlobalAveragePool(temporalFeatures);

        // Project to embedding space
        var embedding = _videoProjection.Forward(pooled);
        int embeddingBatchSize = embedding.Shape[0];
        embedding = Engine.Reshape(
            embedding,
            [embeddingBatchSize, embedding.Length / embeddingBatchSize]);
        embedding = L2Normalize(embedding);

        if (!hasBatch)
        {
            embedding = RemoveBatchDimension(embedding);
        }

        return embedding;
    }

    /// <summary>
    /// Encodes text into an embedding vector.
    /// </summary>
    /// <param name="tokenIds">Token IDs [SeqLen] or [B, SeqLen].</param>
    /// <returns>Text embedding [EmbeddingDim] or [B, EmbeddingDim].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts text (as token IDs) into a numerical vector.
    /// Text with similar meaning will have similar embeddings.
    /// </para>
    /// </remarks>
    public Tensor<T> EncodeText(Tensor<T> tokenIds)
    {
        bool hasBatch = tokenIds.Rank == 2;
        if (!hasBatch)
        {
            tokenIds = AddBatchDimension2D(tokenIds);
        }

        int batchSize = tokenIds.Shape[0];
        int seqLen = Math.Min(tokenIds.Shape[1], _textMaxLength);

        // Pad to max length if needed
        var paddedTokens = new Tensor<T>([batchSize, _textMaxLength]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < _textMaxLength; i++)
            {
                if (i < seqLen)
                {
                    paddedTokens[b, i] = tokenIds[b, i];
                }
                else
                {
                    paddedTokens[b, i] = NumOps.Zero; // Padding token
                }
            }
        }

        // Token embedding lookup with positional embeddings (proper CLIP style)
        var features = LookupTokenEmbeddings(paddedTokens);

        // Text transformer blocks with Pre-LN architecture (following CLIP)
        int numLayers = _textTransformerQKV.Count;
        for (int layer = 0; layer < numLayers; layer++)
        {
            // Pre-LayerNorm
            var normed = TextLayerNorm(features);

            // Multi-head self-attention with causal mask
            var attnOut = TextMultiHeadAttention(normed, layer);

            // Residual connection
            features = AddTensors(features, attnOut);

            // FFN with Pre-LN
            var ffnNormed = TextLayerNorm(features);
            var ffnOut = TextFFN(ffnNormed, layer);

            // Residual connection
            features = AddTensors(features, ffnOut);
        }

        // Final layer norm
        features = TextLayerNorm(features);

        // Take [EOS] token embedding (last position before padding, following CLIP)
        var eosFeature = ExtractEOSFeature(features);

        // Project to embedding space
        var embedding = _textProjection.Forward(eosFeature);
        embedding = L2Normalize(embedding);

        if (!hasBatch)
        {
            embedding = RemoveBatchDimension(embedding);
        }

        return embedding;
    }

    /// <summary>
    /// Computes similarity between video and text embeddings.
    /// </summary>
    /// <param name="videoEmbedding">Video embedding.</param>
    /// <param name="textEmbedding">Text embedding.</param>
    /// <returns>Similarity score (higher = more similar).</returns>
    public double ComputeSimilarity(Tensor<T> videoEmbedding, Tensor<T> textEmbedding)
    {
        return CosineSimilarity(videoEmbedding, textEmbedding);
    }

    /// <summary>
    /// Performs zero-shot video classification.
    /// </summary>
    /// <param name="videoFrames">Input video frames.</param>
    /// <param name="classTexts">List of class descriptions (e.g., "a video of cooking").</param>
    /// <returns>Probability distribution over classes.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This classifies videos without any training on those specific categories.
    /// Simply provide text descriptions of each class (like "a video of someone running"),
    /// and the model will determine which description best matches the video.
    /// </para>
    /// </remarks>
    public List<(string ClassName, double Probability)> ZeroShotClassify(
        Tensor<T> videoFrames,
        List<string> classTexts)
    {
        // Encode video
        var videoEmbed = EncodeVideo(videoFrames);

        // Encode all class texts
        var textEmbeds = new List<Tensor<T>>();
        foreach (var text in classTexts)
        {
            var tokenIds = Tokenize(text);
            var textEmbed = EncodeText(tokenIds);
            textEmbeds.Add(textEmbed);
        }

        // Compute similarities
        var similarities = new List<double>();
        foreach (var textEmbed in textEmbeds)
        {
            double sim = ComputeSimilarity(videoEmbed, textEmbed);
            similarities.Add(sim);
        }

        // Apply softmax with temperature
        var probabilities = Softmax(similarities, Temperature);

        // Create result pairs
        var results = new List<(string, double)>();
        for (int i = 0; i < classTexts.Count; i++)
        {
            results.Add((classTexts[i], probabilities[i]));
        }

        return results.OrderByDescending(x => x.Item2).ToList();
    }

    /// <summary>
    /// Retrieves the most similar videos to a text query.
    /// </summary>
    /// <param name="query">Text query describing the desired video.</param>
    /// <param name="videoEmbeddings">Pre-computed video embeddings.</param>
    /// <param name="topK">Number of results to return.</param>
    /// <returns>List of (videoIndex, similarity) pairs, sorted by similarity.</returns>
    public List<(int VideoIndex, double Similarity)> TextToVideoRetrieval(
        string query,
        List<Tensor<T>> videoEmbeddings,
        int topK = 10)
    {
        var tokenIds = Tokenize(query);
        var queryEmbed = EncodeText(tokenIds);

        var results = new List<(int, double)>();
        for (int i = 0; i < videoEmbeddings.Count; i++)
        {
            double sim = ComputeSimilarity(videoEmbeddings[i], queryEmbed);
            results.Add((i, sim));
        }

        return results.OrderByDescending(x => x.Item2).Take(topK).ToList();
    }

    /// <summary>
    /// Retrieves the most similar text descriptions for a video.
    /// </summary>
    /// <param name="videoFrames">Input video frames.</param>
    /// <param name="candidateTexts">List of candidate text descriptions.</param>
    /// <param name="topK">Number of results to return.</param>
    /// <returns>List of (text, similarity) pairs, sorted by similarity.</returns>
    public List<(string Text, double Similarity)> VideoToTextRetrieval(
        Tensor<T> videoFrames,
        List<string> candidateTexts,
        int topK = 10)
    {
        var videoEmbed = EncodeVideo(videoFrames);

        var results = new List<(string, double)>();
        foreach (var text in candidateTexts)
        {
            var tokenIds = Tokenize(text);
            var textEmbed = EncodeText(tokenIds);
            double sim = ComputeSimilarity(videoEmbed, textEmbed);
            results.Add((text, sim));
        }

        return results.OrderByDescending(x => x.Item2).Take(topK).ToList();
    }

    /// <summary>
    /// Computes video-text similarity matrix for a batch.
    /// </summary>
    /// <param name="videoFramesBatch">Batch of videos [B, T, C, H, W].</param>
    /// <param name="textsBatch">Batch of text token IDs [B, SeqLen].</param>
    /// <returns>Similarity matrix [B, B] where (i,j) is similarity between video i and text j.</returns>
    public Tensor<T> ComputeSimilarityMatrix(
        List<Tensor<T>> videoFramesBatch,
        List<Tensor<T>> textsBatch)
    {
        int batchSize = videoFramesBatch.Count;

        // Encode all videos
        var videoEmbeds = new List<Tensor<T>>();
        foreach (var video in videoFramesBatch)
        {
            videoEmbeds.Add(EncodeVideo(video));
        }

        // Encode all texts
        var textEmbeds = new List<Tensor<T>>();
        foreach (var text in textsBatch)
        {
            textEmbeds.Add(EncodeText(text));
        }

        // Compute similarity matrix
        var simMatrix = new Tensor<T>([batchSize, batchSize]);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < batchSize; j++)
            {
                double sim = ComputeSimilarity(videoEmbeds[i], textEmbeds[j]);
                simMatrix[i, j] = NumOps.FromDouble(sim / Temperature);
            }
        }

        return simMatrix;
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // Default: encode video
        return EncodeVideo(input);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        // VideoCLIP's published Layers collection contains the components of
        // several branches, not one flat sequential graph. Training must use
        // the same video path as inference so the loss sees the projected
        // embedding rather than an intermediate convolutional feature map.
        return EncodeVideo(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    #endregion

    #region Private Methods

    private Tensor<T> ProcessFrames(Tensor<T> videoFrames)
    {
        int batchSize = videoFrames.Shape[0];
        int numFrames = videoFrames.Shape[1];
        int channels = videoFrames.Shape[2];
        int height = videoFrames.Shape[3];
        int width = videoFrames.Shape[4];


        // Process each frame
        var allFrameFeatures = new List<Tensor<T>>();

        for (int t = 0; t < numFrames; t++)
        {
            // Extract frame
            var frame = new Tensor<T>([batchSize, channels, height, width]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            frame[b, c, h, w] = videoFrames[b, t, c, h, w];
                        }
                    }
                }
            }

            // Apply spatial encoder
            var features = frame;
            foreach (var layer in _videoEncoder)
            {
                features = layer.Forward(features);
                features = ApplyGELU(features);
            }

            allFrameFeatures.Add(features);
        }

        if (allFrameFeatures.Count == 0)
            throw new ArgumentException("A video must contain at least one frame.", nameof(videoFrames));

        // Flatten each encoder patch grid while preserving its real channel
        // count. In particular, smoke-sized/custom encoders need not use the
        // paper-scale width of 768. Keeping the reshape and concatenation on
        // the engine also preserves the training tape through this layout
        // transformation.
        var firstFeatures = allFrameFeatures[0];
        if (firstFeatures.Rank != 4 || firstFeatures.Shape[0] != batchSize)
        {
            throw new InvalidOperationException(
                "The VideoCLIP spatial encoder must produce [batch, channels, height, width] features.");
        }

        int hiddenDim = firstFeatures.Shape[1];
        int spatialDim = firstFeatures.Length / (batchSize * hiddenDim);
        var temporalSlices = new Tensor<T>[numFrames];

        for (int t = 0; t < numFrames; t++)
        {
            var features = allFrameFeatures[t];
            if (features.Rank != 4 || features.Shape[0] != batchSize ||
                features.Shape[1] != hiddenDim || features.Length != firstFeatures.Length)
            {
                throw new InvalidOperationException(
                    "All VideoCLIP frames must produce the same spatial feature shape.");
            }

            temporalSlices[t] = Engine.Reshape(
                features, [batchSize, hiddenDim, 1, spatialDim]);
        }

        return Engine.TensorConcatenate(temporalSlices, axis: 2);
    }

    private Tensor<T> ApplyTemporalAttention(Tensor<T> features)
    {
        int batchSize = features.Shape[0];
        int channels = features.Shape[1];
        int numFrames = features.Shape[2];
        int spatialDim = features.Shape[3];

        // Fold batch and spatial positions together so each patch has a
        // temporal sequence [T], while retaining the computation graph.
        var reshaped = Engine.Reshape(
            Engine.TensorPermute(features, [0, 3, 1, 2]),
            [batchSize * spatialDim, channels, 1, numFrames]);

        // Apply temporal transformer
        var attended = reshaped;
        foreach (var layer in _temporalTransformer)
        {
            attended = layer.Forward(attended);
            attended = ApplyGELU(attended);
        }

        // The temporal encoder treats each spatial location as an independent
        // sequence by folding B and S together. Restore the original layout
        // before pooling; otherwise S is mistaken for the batch dimension and
        // an unbatched clip produces S embeddings instead of one embedding.
        return Engine.TensorPermute(
            Engine.Reshape(attended, [batchSize, spatialDim, channels, numFrames]),
            [0, 2, 3, 1]);
    }

    private Tensor<T> ExtractEOSFeature(Tensor<T> features)
    {
        int batchSize = features.Shape[0];
        int channels = features.Shape[1];
        int seqLen = features.Shape[3];

        // Extract feature at last position
        var eosFeature = new Tensor<T>([batchSize, channels, 1, 1]);
        int lastPos = seqLen - 1;

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                eosFeature[b, c, 0, 0] = features[b, c, 0, lastPos];
            }
        }

        return eosFeature;
    }

    private Tensor<T> GlobalAveragePool(Tensor<T> input)
    {
        return Engine.ReduceMean(input, [2, 3], keepDims: true);
    }

    private Tensor<T> L2Normalize(Tensor<T> embedding)
    {
        int featureAxis = embedding.Rank - 1;
        var squared = Engine.TensorMultiply(embedding, embedding);
        var sumSquared = Engine.ReduceSum(squared, [featureAxis], keepDims: true);
        var norm = Engine.TensorSqrt(
            Engine.TensorAddScalar(sumSquared, NumOps.FromDouble(1e-6)));
        return Engine.TensorBroadcastDivide(embedding, norm);
    }

    private double CosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        return VectorHelper.CosineSimilarity(a.ToVector(), b.ToVector());
    }

    private List<double> Softmax(List<double> values, double temperature)
    {
        var scaled = values.Select(v => v / temperature).ToList();
        double maxVal = scaled.Max();

        var exps = scaled.Select(v => Math.Exp(v - maxVal)).ToList();
        double sum = exps.Sum();

        return exps.Select(e => e / sum).ToList();
    }

    /// <summary>
    /// Tokenizes text using the CLIP BPE tokenizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses proper BPE tokenization with the CLIP vocabulary.
    /// The tokenizer handles:
    /// - Subword segmentation using BPE merges
    /// - Special token insertion (BOS/EOS)
    /// - Padding to max length
    /// - Truncation for long texts
    /// </para>
    /// </remarks>
    private Tensor<T> Tokenize(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer is not initialized.");

        // Encode text using the BPE tokenizer
        var encoded = _tokenizer.Encode(text, _encodingOptions);
        var tokenIds = encoded.TokenIds;

        // Convert to tensor with padding
        var tokens = new Tensor<T>([_textMaxLength]);
        for (int i = 0; i < _textMaxLength; i++)
        {
            if (i < tokenIds.Count)
            {
                tokens[i] = NumOps.FromDouble(tokenIds[i]);
            }
            else
            {
                // Pad with padding token (typically 0 or the pad_token_id)
                tokens[i] = NumOps.Zero;
            }
        }

        return tokens;
    }

    private Tensor<T> ApplyGELU(Tensor<T> input)
    {
        return Engine.GELU(input);
    }

    /// <summary>
    /// Initializes an embedding table with Xavier/Glorot uniform initialization.
    /// </summary>
    private void InitializeEmbeddingTable(Tensor<T> table, int numEmbeddings, int embeddingDim)
    {
        var random = RandomHelper.CreateSecureRandom();

        // Xavier/Glorot uniform: range = sqrt(6 / (fan_in + fan_out))
        // For embeddings: fan_in = 1, fan_out = embeddingDim
        double limit = Math.Sqrt(6.0 / (1 + embeddingDim));

        for (int i = 0; i < numEmbeddings; i++)
        {
            for (int j = 0; j < embeddingDim; j++)
            {
                double val = (random.NextDouble() * 2 - 1) * limit;
                table[i, j] = NumOps.FromDouble(val);
            }
        }
    }

    /// <summary>
    /// Performs embedding lookup from the token embedding table.
    /// </summary>
    /// <param name="tokenIds">Input token IDs [batch, seq_len].</param>
    /// <returns>Embedded tokens [batch, hidden_dim, 1, seq_len].</returns>
    private Tensor<T> LookupTokenEmbeddings(Tensor<T> tokenIds)
    {
        int batchSize = tokenIds.Shape[0];
        int seqLen = tokenIds.Shape.Length > 1 ? tokenIds.Shape[1] : _textMaxLength;

        var output = new Tensor<T>([batchSize, _textHiddenDim, 1, seqLen]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int pos = 0; pos < seqLen; pos++)
            {
                // Get token ID and clamp to valid range
                int tokenId = Math.Min(Math.Max(0, (int)NumOps.ToDouble(tokenIds[b, pos])), _vocabSize - 1);

                // Lookup embedding from table and add positional embedding
                for (int d = 0; d < _textHiddenDim; d++)
                {
                    double tokenEmbed = NumOps.ToDouble(_tokenEmbeddingTable[tokenId, d]);
                    double posEmbed = NumOps.ToDouble(_positionalEmbeddingTable[pos, d]);
                    output[b, d, 0, pos] = NumOps.FromDouble(tokenEmbed + posEmbed);
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Text transformer multi-head self-attention following CLIP architecture.
    /// </summary>
    private Tensor<T> TextMultiHeadAttention(Tensor<T> input, int layerIdx)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int seqLen = input.Shape[3];
        int numHeads = 12;
        int headDim = channels / numHeads;
        double scale = 1.0 / Math.Sqrt(headDim);

        // Compute QKV projections
        var qkv = _textTransformerQKV[layerIdx].Forward(input);

        // Split into Q, K, V
        var query = new Tensor<T>([batchSize, channels, 1, seqLen]);
        var key = new Tensor<T>([batchSize, channels, 1, seqLen]);
        var value = new Tensor<T>([batchSize, channels, 1, seqLen]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int pos = 0; pos < seqLen; pos++)
            {
                for (int c = 0; c < channels; c++)
                {
                    query[b, c, 0, pos] = qkv[b, c, 0, pos];
                    key[b, c, 0, pos] = qkv[b, channels + c, 0, pos];
                    value[b, c, 0, pos] = qkv[b, 2 * channels + c, 0, pos];
                }
            }
        }

        // Multi-head attention with causal mask (for autoregressive text)
        var output = new Tensor<T>([batchSize, channels, 1, seqLen]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int head = 0; head < numHeads; head++)
            {
                int headStart = head * headDim;

                // Compute attention scores for this head
                var attnScores = new double[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j <= i; j++) // Causal mask: can only attend to past positions
                    {
                        double score = 0;
                        for (int d = 0; d < headDim; d++)
                        {
                            double q = NumOps.ToDouble(query[b, headStart + d, 0, i]);
                            double k = NumOps.ToDouble(key[b, headStart + d, 0, j]);
                            score += q * k;
                        }
                        attnScores[i, j] = score * scale;
                    }
                    // Masked positions get very negative value
                    for (int j = i + 1; j < seqLen; j++)
                    {
                        attnScores[i, j] = -1e9;
                    }
                }

                // Softmax over each row
                for (int i = 0; i < seqLen; i++)
                {
                    double maxScore = double.NegativeInfinity;
                    for (int j = 0; j < seqLen; j++)
                    {
                        maxScore = Math.Max(maxScore, attnScores[i, j]);
                    }

                    double sumExp = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        attnScores[i, j] = Math.Exp(attnScores[i, j] - maxScore);
                        sumExp += attnScores[i, j];
                    }
                    for (int j = 0; j < seqLen; j++)
                    {
                        attnScores[i, j] /= sumExp;
                    }
                }

                // Weighted sum of values
                for (int i = 0; i < seqLen; i++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        double sum = 0;
                        for (int j = 0; j < seqLen; j++)
                        {
                            sum += attnScores[i, j] * NumOps.ToDouble(value[b, headStart + d, 0, j]);
                        }
                        output[b, headStart + d, 0, i] = NumOps.FromDouble(sum);
                    }
                }
            }
        }

        // Output projection
        return _textTransformerAttnProj[layerIdx].Forward(output);
    }

    /// <summary>
    /// Text transformer feed-forward network with quick GELU activation.
    /// </summary>
    private Tensor<T> TextFFN(Tensor<T> input, int layerIdx)
    {
        // Expand: hidden_dim -> 4 * hidden_dim
        var expanded = _textTransformerFFN1[layerIdx].Forward(input);
        // Quick GELU (following CLIP implementation)
        expanded = ApplyQuickGELU(expanded);
        // Contract: 4 * hidden_dim -> hidden_dim
        return _textTransformerFFN2[layerIdx].Forward(expanded);
    }

    /// <summary>
    /// Quick GELU approximation as used in OpenAI CLIP.
    /// </summary>
    private Tensor<T> ApplyQuickGELU(Tensor<T> input)
    {
        return input.Transform((v, _) =>
        {
            double x = NumOps.ToDouble(v);
            double quickGelu = x * (1.0 / (1.0 + Math.Exp(-1.702 * x)));
            return NumOps.FromDouble(quickGelu);
        });
    }

    /// <summary>
    /// Layer normalization for text transformer.
    /// </summary>
    private Tensor<T> TextLayerNorm(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int seqLen = input.Shape[3];
        var output = new Tensor<T>(input._shape);
        double eps = 1e-5;

        for (int b = 0; b < batchSize; b++)
        {
            for (int pos = 0; pos < seqLen; pos++)
            {
                // Compute mean and variance across channels
                double sum = 0.0;
                for (int c = 0; c < channels; c++)
                {
                    sum += NumOps.ToDouble(input[b, c, 0, pos]);
                }
                double mean = sum / channels;

                double varSum = 0.0;
                for (int c = 0; c < channels; c++)
                {
                    double diff = NumOps.ToDouble(input[b, c, 0, pos]) - mean;
                    varSum += diff * diff;
                }
                double variance = varSum / channels;
                double invStd = 1.0 / Math.Sqrt(variance + eps);

                // Normalize
                for (int c = 0; c < channels; c++)
                {
                    double val = NumOps.ToDouble(input[b, c, 0, pos]);
                    double normalized = (val - mean) * invStd;
                    output[b, c, 0, pos] = NumOps.FromDouble(normalized);
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Element-wise tensor addition for residual connections.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    private Tensor<T> AddBatchDimension5D(Tensor<T> tensor)
    {
        int t = tensor.Shape[0];
        int c = tensor.Shape[1];
        int h = tensor.Shape[2];
        int w = tensor.Shape[3];

        var result = new Tensor<T>([1, t, c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private Tensor<T> AddBatchDimension2D(Tensor<T> tensor)
    {
        int len = tensor.Shape[0];

        var result = new Tensor<T>([1, len]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        if (tensor.Shape[0] != 1)
        {
            throw new InvalidOperationException(
                $"Cannot remove a non-singleton batch dimension of size {tensor.Shape[0]}.");
        }

        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++)
        {
            newShape[i] = tensor.Shape[i + 1];
        }

        return Engine.Reshape(tensor, newShape);
    }

    #endregion

    #region Abstract Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        ClearLayers();

        foreach (var layer in _videoEncoder) Layers.Add(layer);
        foreach (var layer in _temporalTransformer) Layers.Add(layer);
        Layers.Add(_videoProjection);
        foreach (var layer in _textTransformerQKV) Layers.Add(layer);
        foreach (var layer in _textTransformerAttnProj) Layers.Add(layer);
        foreach (var layer in _textTransformerFFN1) Layers.Add(layer);
        foreach (var layer in _textTransformerFFN2) Layers.Add(layer);
        Layers.Add(_textProjection);
        Layers.Add(_logitScale);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int paramCount = layerParams.Length;
            if (paramCount > 0 && offset + paramCount <= parameters.Length)
            {
                var slice = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++)
                {
                    slice[i] = parameters[offset + i];
                }
                layer.SetParameters(slice);
                offset += paramCount;
            }
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "VideoCLIP" },
            { "Description", "Video-Text Understanding and Retrieval Model" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "NumFrames", _numFrames },
            { "EmbeddingDim", _embeddingDim },
            { "TextMaxLength", _textMaxLength },
            { "Temperature", _temperature },
            { "NumLayers", Layers.Count }
        };

        return new ModelMetadata<T>
        {
            AdditionalInfo = additionalInfo,
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numFrames);
        writer.Write(_embeddingDim);
        writer.Write(_textMaxLength);
        writer.Write(_vocabSize);
        writer.Write(_temperature);

        // The learned embedding tables. They are trainable (see GetExtraTrainableTensors) and live
        // outside Layers, so the layer-by-layer weight sections of the stream do not carry them and
        // a reload rebuilt them from InitializeEmbeddingTable's RNG instead — dropping trained text
        // -tower weights on every save/load. Same element-by-element idiom VisionTransformer uses
        // for its CLS and positional tokens.
        for (int i = 0; i < _tokenEmbeddingTable.Length; i++)
            writer.Write(Convert.ToDouble(_tokenEmbeddingTable[i]));
        for (int i = 0; i < _positionalEmbeddingTable.Length; i++)
            writer.Write(Convert.ToDouble(_positionalEmbeddingTable[i]));
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadDouble();

        // Restore the learned embedding tables written above. The geometry fields are discarded
        // because they are readonly and the constructor has already rebuilt this instance at the
        // right sizes; these tensors, by contrast, carry trained values that only the stream has.
        for (int i = 0; i < _tokenEmbeddingTable.Length; i++)
            _tokenEmbeddingTable[i] = NumOps.FromDouble(reader.ReadDouble());
        for (int i = 0; i < _positionalEmbeddingTable.Length; i++)
            _positionalEmbeddingTable[i] = NumOps.FromDouble(reader.ReadDouble());

        // The base deserializer has just replaced Layers with the restored instances. Rebind the
        // per-stage views so the explicit forward and the tape both consume those restored weights
        // rather than the constructor-fresh layers they were bound to.
        BindLayerViewsFromLayers();
    }

    /// <inheritdoc/>
    /// <summary>
    /// Surfaces the token and positional embedding tables, which are learned parameters the model
    /// owns OUTSIDE <c>Layers</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In CLIP (Radford et al. 2021 §2.4) the text encoder's token embedding and positional
    /// embedding are both learned — <c>nn.Embedding(vocab_size, width)</c> and an
    /// <c>nn.Parameter</c> respectively — so they appear in <c>state_dict()</c>, receive gradients,
    /// and survive a module copy. Here they are plain tensors built in the constructor, so without
    /// this hook the <c>Layers</c>-only parameter walk never saw them and they were frozen at their
    /// random initialization for the model's entire lifetime, never trained and never persisted.
    /// </para>
    /// <para>
    /// The clone consequence was the sharper one. A copy re-runs the constructor, which
    /// re-initializes both tables to FRESH random values, and nothing afterwards overwrote them:
    /// the clone's text tower therefore computed a different function from the original's while
    /// every tensor in <c>Layers</c> matched bit-for-bit (measured: 22/22 chunks and 48173/48173
    /// parameters identical, parameter L2 equal to 17 digits, yet the outputs differed by 1.6e+00
    /// on identical input, and MoreData_ShouldNotDegrade failed on the clone).
    /// </para>
    /// <para>
    /// Yielding them here opts into the three base paths that already handle model-owned tensors:
    /// the tape optimizer's step, the serialization round-trip, and the copy-on-write clone. Same
    /// mechanism <see cref="AiDotNet.NeuralNetworks.VisionTransformer{T}"/> uses for its CLS and
    /// positional tokens.
    /// </para>
    /// </remarks>
    protected override IEnumerable<Tensor<T>> GetExtraTrainableTensors()
    {
        yield return _tokenEmbeddingTable;
        yield return _positionalEmbeddingTable;
    }

    /// <summary>
    /// (Re)binds the per-stage layer views to the current contents of <c>Layers</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>Layers</c> is the single ownership graph; <c>_videoEncoder</c>, <c>_temporalTransformer</c>,
    /// <c>_videoProjection</c>, the four text-transformer lists, <c>_textProjection</c> and
    /// <c>_logitScale</c> are ordered views into it that the explicit forward walks. The constructor
    /// calls this after populating <c>Layers</c>; <see cref="DeserializeNetworkSpecificData"/> calls
    /// it again because the base deserializer REPLACES every entry in <c>Layers</c> with a restored
    /// instance, which orphans any view still bound to the constructor's layers.
    /// </para>
    /// <para>
    /// Idempotent, so the copy-on-write clone path — which calls
    /// <c>DeserializeNetworkSpecificData</c> without replacing <c>Layers</c> — simply re-binds each
    /// view to the object it already referenced.
    /// </para>
    /// <para>
    /// Same contract <c>MusicSourceSeparator.TryBindDemucsTopologyFromLayers</c> maintains for its
    /// Demucs encoder/decoder views, and the reason it exists there too.
    /// </para>
    /// </remarks>
    [System.Diagnostics.CodeAnalysis.MemberNotNull(
        nameof(_videoProjection), nameof(_textProjection), nameof(_logitScale))]
    private void BindLayerViewsFromLayers()
    {
        _videoEncoder.Clear();
        _temporalTransformer.Clear();
        _textTransformerQKV.Clear();
        _textTransformerAttnProj.Clear();
        _textTransformerFFN1.Clear();
        _textTransformerFFN2.Clear();

        int idx = 0;
        // Video encoder: 1 patch embed + NumSpatialBlocks * 2
        int videoEncoderCount = 1 + _options.NumSpatialBlocks * 2;
        for (int i = 0; i < videoEncoderCount; i++)
            _videoEncoder.Add((ConvolutionalLayer<T>)Layers[idx++]);

        // Temporal transformer
        for (int i = 0; i < _options.NumTemporalBlocks; i++)
            _temporalTransformer.Add((ConvolutionalLayer<T>)Layers[idx++]);

        // Video projection
        _videoProjection = (ConvolutionalLayer<T>)Layers[idx++];

        // Text transformer: 4 layers per block (QKV, AttnProj, FFN1, FFN2)
        for (int i = 0; i < _options.NumTextBlocks; i++)
        {
            _textTransformerQKV.Add((ConvolutionalLayer<T>)Layers[idx++]);
            _textTransformerAttnProj.Add((ConvolutionalLayer<T>)Layers[idx++]);
            _textTransformerFFN1.Add((ConvolutionalLayer<T>)Layers[idx++]);
            _textTransformerFFN2.Add((ConvolutionalLayer<T>)Layers[idx++]);
        }

        // Text projection
        _textProjection = (ConvolutionalLayer<T>)Layers[idx++];

        // Logit scale
        _logitScale = (ConvolutionalLayer<T>)Layers[idx++];
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new VideoCLIP<T>(
            Architecture, _numFrames, _embeddingDim, _textMaxLength, _vocabSize, _temperature,
            // EVERY option that affects training must be carried, not just the topology ones. This
            // copied five fields and dropped Beta1, Beta2, MaxGradientNorm, WarmupSteps,
            // TotalTrainingSteps and DecayPower, so a clone silently rebuilt its optimizer from the
            // DEFAULTS — including the paper's 1000-step warm-up, which the caller may deliberately have
            // turned off. A clone that warms up when the original does not is not the same model.
            //
            // Measured: MoreData_ShouldNotDegrade clones the network and trains the clone, and the
            // clone's loss came back byte-identical (0.7257835234621279) at 2, 4 and 12 iterations with
            // its parameter L2 unchanged to 16 digits — the LR sat at ~5e-8 on the first rung of a ramp
            // the original had disabled, so no step could move anything. The invariant was reporting a
            // real defect, not task-to-task variance.
            options: new VideoCLIPVideoOptions
            {
                HiddenDimension = _options.HiddenDimension,
                NumSpatialBlocks = _options.NumSpatialBlocks,
                NumTemporalBlocks = _options.NumTemporalBlocks,
                NumTextBlocks = _options.NumTextBlocks,
                LearningRate = _options.LearningRate,
                Beta1 = _options.Beta1,
                Beta2 = _options.Beta2,
                MaxGradientNorm = _options.MaxGradientNorm,
                WarmupSteps = _options.WarmupSteps,
                TotalTrainingSteps = _options.TotalTrainingSteps,
                DecayPower = _options.DecayPower
            });
    }

    #endregion
}
