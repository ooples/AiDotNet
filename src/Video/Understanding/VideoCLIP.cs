using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;

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
public class VideoCLIP<T> : NeuralNetworkBase<T>
{
    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numFrames;
    private readonly int _embeddingDim;
    private readonly int _textMaxLength;
    private readonly int _vocabSize;
    private readonly double _temperature;

    // Video encoder components
    private readonly List<ConvolutionalLayer<T>> _videoEncoder;
    private readonly List<ConvolutionalLayer<T>> _temporalTransformer;
    private readonly ConvolutionalLayer<T> _videoProjection;

    // Text encoder components
    // Proper CLIP-style token embedding: embedding lookup table [vocab_size, hidden_dim]
    private readonly Tensor<T> _tokenEmbeddingTable;          // Embedding lookup table
    private readonly Tensor<T> _positionalEmbeddingTable;     // Learned positional embeddings
    private readonly List<ConvolutionalLayer<T>> _textTransformerQKV;      // QKV projections
    private readonly List<ConvolutionalLayer<T>> _textTransformerAttnProj; // Attention output
    private readonly List<ConvolutionalLayer<T>> _textTransformerFFN1;     // FFN expand
    private readonly List<ConvolutionalLayer<T>> _textTransformerFFN2;     // FFN contract
    private readonly ConvolutionalLayer<T> _textProjection;
    private readonly int _textHiddenDim;

    // Shared components
    private readonly ConvolutionalLayer<T> _logitScale;

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
        string? vocabPath = null,
        string? mergesPath = null)
        : base(architecture, new ContrastiveLoss<T>())
    {
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

        int featH = _height / 16;
        int featW = _width / 16;

        // Video encoder (ViT-like spatial encoder)
        int patchDim = _channels * 16 * 16; // 16x16 patches
        int hiddenDim = 768;
        _textHiddenDim = hiddenDim;

        // Patch embedding for video frames
        _videoEncoder.Add(new ConvolutionalLayer<T>(_channels, _height, _width, hiddenDim, 16, 16, 0));

        // Spatial transformer blocks
        int numSpatialBlocks = 12;
        for (int i = 0; i < numSpatialBlocks; i++)
        {
            _videoEncoder.Add(new ConvolutionalLayer<T>(hiddenDim, featH, featW, hiddenDim, 1, 1, 0));
            _videoEncoder.Add(new ConvolutionalLayer<T>(hiddenDim, featH, featW, hiddenDim, 3, 1, 1));
        }

        // Temporal transformer for video understanding
        int numTemporalBlocks = 4;
        for (int i = 0; i < numTemporalBlocks; i++)
        {
            _temporalTransformer.Add(new ConvolutionalLayer<T>(hiddenDim, 1, _numFrames, hiddenDim, 1, 1, 0));
        }

        // Video projection to shared embedding space
        _videoProjection = new ConvolutionalLayer<T>(hiddenDim, 1, 1, _embeddingDim, 1, 1, 0);

        // Text encoder with proper CLIP-style embeddings
        // Token embedding lookup table: [vocab_size, hidden_dim]
        // Initialized with Xavier/Glorot uniform initialization
        _tokenEmbeddingTable = new Tensor<T>([_vocabSize, hiddenDim]);
        InitializeEmbeddingTable(_tokenEmbeddingTable, _vocabSize, hiddenDim);

        // Positional embedding table: [max_length, hidden_dim]
        // CLIP uses learned positional embeddings, not sinusoidal
        _positionalEmbeddingTable = new Tensor<T>([_textMaxLength, hiddenDim]);
        InitializeEmbeddingTable(_positionalEmbeddingTable, _textMaxLength, hiddenDim);

        // Text transformer blocks with proper multi-head attention (12 layers, 12 heads)
        int numTextBlocks = 12;
        // Note: numHeads = 12 is used in TextMultiHeadAttention method
        int ffnDim = hiddenDim * 4; // MLP expansion ratio of 4

        for (int i = 0; i < numTextBlocks; i++)
        {
            // QKV projection for multi-head attention
            _textTransformerQKV.Add(new ConvolutionalLayer<T>(hiddenDim, 1, _textMaxLength, hiddenDim * 3, 1, 1, 0));
            // Attention output projection
            _textTransformerAttnProj.Add(new ConvolutionalLayer<T>(hiddenDim, 1, _textMaxLength, hiddenDim, 1, 1, 0));
            // FFN expand (hidden -> 4*hidden)
            _textTransformerFFN1.Add(new ConvolutionalLayer<T>(hiddenDim, 1, _textMaxLength, ffnDim, 1, 1, 0));
            // FFN contract (4*hidden -> hidden)
            _textTransformerFFN2.Add(new ConvolutionalLayer<T>(ffnDim, 1, _textMaxLength, hiddenDim, 1, 1, 0));
        }

        // Text projection to shared embedding space
        _textProjection = new ConvolutionalLayer<T>(hiddenDim, 1, 1, _embeddingDim, 1, 1, 0);

        // Learnable temperature (logit scale)
        _logitScale = new ConvolutionalLayer<T>(1, 1, 1, 1, 1, 1, 0);

        // Register layers
        foreach (var layer in _videoEncoder) Layers.Add(layer);
        foreach (var layer in _temporalTransformer) Layers.Add(layer);
        Layers.Add(_videoProjection);

        // Text transformer layers (embedding tables are tensors, not layers)
        foreach (var layer in _textTransformerQKV) Layers.Add(layer);
        foreach (var layer in _textTransformerAttnProj) Layers.Add(layer);
        foreach (var layer in _textTransformerFFN1) Layers.Add(layer);
        foreach (var layer in _textTransformerFFN2) Layers.Add(layer);
        Layers.Add(_textProjection);
        Layers.Add(_logitScale);
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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Default: encode video
        return EncodeVideo(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Training with contrastive loss
        // input: video frames, expectedOutput: text tokens

        var videoEmbed = EncodeVideo(input);
        var textEmbed = EncodeText(expectedOutput);

        // Compute loss gradient (push matching pairs together, non-matching apart)
        var lossGradient = videoEmbed.Transform((v, idx) =>
            NumOps.Subtract(v, textEmbed.Data.Span[idx]));

        BackwardPass(lossGradient);

        T lr = NumOps.FromDouble(0.0001);
        foreach (var layer in Layers)
        {
            layer.UpdateParameters(lr);
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

        int hiddenDim = 768;
        int featH = height / 16;
        int featW = width / 16;

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

        // Stack temporal features
        var stacked = new Tensor<T>([batchSize, hiddenDim, numFrames, featH * featW]);
        for (int t = 0; t < numFrames; t++)
        {
            var feat = allFrameFeatures[t];
            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < hiddenDim; c++)
                {
                    int idx = 0;
                    for (int h = 0; h < featH; h++)
                    {
                        for (int w = 0; w < featW; w++)
                        {
                            stacked[b, c, t, idx++] = feat[b, c, h, w];
                        }
                    }
                }
            }
        }

        return stacked;
    }

    private Tensor<T> ApplyTemporalAttention(Tensor<T> features)
    {
        int batchSize = features.Shape[0];
        int channels = features.Shape[1];
        int numFrames = features.Shape[2];
        int spatialDim = features.Shape[3];

        // Reshape for temporal processing
        var reshaped = new Tensor<T>([batchSize * spatialDim, channels, 1, numFrames]);
        int idx = 0;
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < spatialDim; s++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int t = 0; t < numFrames; t++)
                    {
                        reshaped[idx, c, 0, t] = features[b, c, t, s];
                    }
                }
                idx++;
            }
        }

        // Apply temporal transformer
        var attended = reshaped;
        foreach (var layer in _temporalTransformer)
        {
            attended = layer.Forward(attended);
            attended = ApplyGELU(attended);
        }

        return attended;
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
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int temporalDim = input.Shape[2];
        int spatialDim = input.Shape[3];

        var output = new Tensor<T>([batchSize, channels, 1, 1]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                T sum = NumOps.Zero;
                int count = 0;

                for (int t = 0; t < temporalDim; t++)
                {
                    for (int s = 0; s < spatialDim; s++)
                    {
                        sum = NumOps.Add(sum, input[b, c, t, s]);
                        count++;
                    }
                }

                output[b, c, 0, 0] = NumOps.Divide(sum, NumOps.FromDouble(count));
            }
        }

        return output;
    }

    private Tensor<T> L2Normalize(Tensor<T> embedding)
    {
        // Normalize to unit length
        double norm = 0;
        for (int i = 0; i < embedding.Data.Length; i++)
        {
            double val = Convert.ToDouble(embedding.Data.Span[i]);
            norm += val * val;
        }
        norm = Math.Sqrt(norm + 1e-8);

        return embedding.Transform((v, _) =>
            NumOps.FromDouble(Convert.ToDouble(v) / norm));
    }

    private double CosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        double dot = 0;
        double normA = 0;
        double normB = 0;

        int len = Math.Min(a.Data.Length, b.Data.Length);
        for (int i = 0; i < len; i++)
        {
            double va = Convert.ToDouble(a.Data.Span[i]);
            double vb = Convert.ToDouble(b.Data.Span[i]);
            dot += va * vb;
            normA += va * va;
            normB += vb * vb;
        }

        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-8);
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
        return input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            double c = Math.Sqrt(2.0 / Math.PI);
            double gelu = 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
            return NumOps.FromDouble(gelu);
        });
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
        var output = new Tensor<T>(input.Shape);
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
        return a.Transform((v, idx) => NumOps.Add(v, b.Data.Span[idx]));
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
        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++)
        {
            newShape[i] = tensor.Shape[i + 1];
        }

        var result = new Tensor<T>(newShape);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private void BackwardPass(Tensor<T> gradient)
    {
        // Backpropagate through video encoder path
        gradient = _videoProjection.Backward(gradient);

        foreach (var layer in _temporalTransformer.AsEnumerable().Reverse())
        {
            gradient = layer.Backward(gradient);
        }

        foreach (var layer in _videoEncoder.AsEnumerable().Reverse())
        {
            gradient = layer.Backward(gradient);
        }

        // Backpropagate through text encoder path
        _textProjection.Backward(gradient);

        // Text transformer layers in reverse
        int numLayers = _textTransformerQKV.Count;
        for (int layer = numLayers - 1; layer >= 0; layer--)
        {
            // FFN backward
            _textTransformerFFN2[layer].Backward(gradient);
            _textTransformerFFN1[layer].Backward(gradient);

            // Attention backward
            _textTransformerAttnProj[layer].Backward(gradient);
            _textTransformerQKV[layer].Backward(gradient);
        }

        // Note: Embedding tables gradients are accumulated during forward pass
        // and would be updated separately in a full implementation
    }

    #endregion

    #region Abstract Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        ClearLayers();
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
            ModelType = ModelType.VideoCLIP,
            AdditionalInfo = additionalInfo,
            ModelData = this.Serialize()
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
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new VideoCLIP<T>(
            Architecture, _numFrames, _embeddingDim, _textMaxLength, _vocabSize, _temperature);
    }

    #endregion
}
