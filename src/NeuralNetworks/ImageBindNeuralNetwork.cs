using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// ImageBind neural network for binding multiple modalities (6+) into a shared embedding space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ImageBind learns a joint embedding space across multiple modalities: images, text, audio, depth,
/// thermal, and IMU data. It uses images as a binding modality - since web data contains
/// many (image, text) pairs, (image, audio) pairs from videos, etc., the model can learn
/// cross-modal relationships even without direct pairs between all modalities.
/// </para>
/// <para><b>For Beginners:</b> ImageBind connects ALL types of data together!
///
/// Architecture overview:
/// 1. Modality-Specific Encoders: Each modality has its own encoder (ViT for images, Transformer for text, etc.)
/// 2. Projection Heads: Map each modality's features to the shared embedding space
/// 3. Contrastive Learning: Align modalities using image as the bridge modality
///
/// Key capabilities:
/// - Cross-modal retrieval: Find images matching audio, text matching video, etc.
/// - Zero-shot classification: Classify any modality using text labels
/// - Emergent alignment: Compare modalities never directly paired during training
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new ImageBindOptions { ImageSize = 224, EmbeddingDim = 1024 };
/// var model = new ImageBindNeuralNetwork&lt;float&gt;(options);
/// var image = Tensor&lt;float&gt;.Random(new[] { 1, 3, 224, 224 });
/// var embedding = model.Predict(image);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.EmbeddingModel)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ImageBind: One Embedding Space To Bind Them All", "https://arxiv.org/abs/2305.05665", Year = 2023, Authors = "Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra")]
public partial class ImageBindNeuralNetwork<T> : MultimodalModelLayoutBase<T>, IImageBindModel<T>
{
    private readonly ImageBindOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Execution Mode

    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    private readonly InferenceSession? _imageEncoder;
    private readonly InferenceSession? _textEncoder;
    private readonly InferenceSession? _audioEncoder;
    private readonly string? _imageEncoderPath;
    private readonly string? _textEncoderPath;
    private readonly string? _audioEncoderPath;

    #endregion

    #region Native Mode Fields

    // Image encoder layers
    private readonly List<ILayer<T>> _imageEncoderLayers = [];
    private Tensor<T>? _imageClsToken;
    private Tensor<T>? _imagePositionalEmbeddings;
    private ILayer<T>? _imagePatchEmbedding;
    private ILayer<T>? _imageProjection;

    // Text encoder layers
    private readonly List<ILayer<T>> _textEncoderLayers = [];
    private Tensor<T>? _textPositionalEmbeddings;
    private ILayer<T>? _textTokenEmbedding;
    private ILayer<T>? _textProjection;

    // Audio encoder layers (uses spectrogram input)
    private readonly List<ILayer<T>> _audioEncoderLayers = [];
    private Tensor<T>? _audioPositionalEmbeddings;
    private ILayer<T>? _audioConv;
    private ILayer<T>? _audioProjection;

    // Thermal encoder (similar to image encoder)
    private readonly List<ILayer<T>> _thermalEncoderLayers = [];
    private Tensor<T>? _thermalClsToken;
    private Tensor<T>? _thermalPositionalEmbeddings;
    private ILayer<T>? _thermalPatchEmbedding;
    private ILayer<T>? _thermalProjection;

    // Depth encoder
    private readonly List<ILayer<T>> _depthEncoderLayers = [];
    private Tensor<T>? _depthClsToken;
    private Tensor<T>? _depthPositionalEmbeddings;
    private ILayer<T>? _depthPatchEmbedding;
    private ILayer<T>? _depthProjection;

    // IMU encoder
    private readonly List<ILayer<T>> _imuEncoderLayers = [];
    private Tensor<T>? _imuPositionalEmbeddings;
    private ILayer<T>? _imuEmbedding;
    private ILayer<T>? _imuProjection;

    // Video encoder (temporal aggregation over frames)
    private readonly List<ILayer<T>> _videoTemporalLayers = [];
    private Tensor<T>? _videoTemporalPositionalEmbeddings;
    private ILayer<T>? _videoProjection;

    #endregion

    #region Shared Fields

    private readonly ITokenizer _tokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _embeddingDimension;
    private readonly int _maxSequenceLength;
    private readonly int _imageSize;
    private readonly int _hiddenDim;
    private readonly int _numEncoderLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _vocabularySize;
    private readonly int _audioSampleRate;
    private readonly int _audioMaxDuration;
    private readonly int _imuTimesteps;
    private readonly int _numVideoFrames;
    private readonly IReadOnlyList<ModalityType> _supportedModalities;

    #endregion

    #region IImageBindModel Properties

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc/>
    public IReadOnlyList<ModalityType> SupportedModalities => _supportedModalities;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an ImageBind network using pretrained ONNX models.
    /// </summary>
    public ImageBindNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string imageEncoderPath,
        string textEncoderPath,
        string audioEncoderPath,
        ITokenizer tokenizer,
        int embeddingDimension = 1024,
        int maxSequenceLength = 77,
        int imageSize = 224,
        int audioSampleRate = 16000,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        ImageBindOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new ImageBindOptions();
        Options = _options;
        if (string.IsNullOrWhiteSpace(imageEncoderPath))
            throw new ArgumentException("Image encoder path cannot be null or empty.", nameof(imageEncoderPath));
        if (string.IsNullOrWhiteSpace(textEncoderPath))
            throw new ArgumentException("Text encoder path cannot be null or empty.", nameof(textEncoderPath));
        if (string.IsNullOrWhiteSpace(audioEncoderPath))
            throw new ArgumentException("Audio encoder path cannot be null or empty.", nameof(audioEncoderPath));
        if (!File.Exists(imageEncoderPath))
            throw new FileNotFoundException($"Image encoder model not found: {imageEncoderPath}");
        if (!File.Exists(textEncoderPath))
            throw new FileNotFoundException($"Text encoder model not found: {textEncoderPath}");
        if (!File.Exists(audioEncoderPath))
            throw new FileNotFoundException($"Audio encoder model not found: {audioEncoderPath}");

        _useNativeMode = false;
        _imageEncoderPath = imageEncoderPath;
        _textEncoderPath = textEncoderPath;
        _audioEncoderPath = audioEncoderPath;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _audioSampleRate = audioSampleRate;
        _audioMaxDuration = 10; // 10 seconds max
        _patchSize = 14;
        _hiddenDim = 1280;
        _numEncoderLayers = 32;
        _numHeads = 16;
        _vocabularySize = 49408;
        _imuTimesteps = 2000;
        _numVideoFrames = 2;

        _supportedModalities = new List<ModalityType>
        {
            ModalityType.Image, ModalityType.Text, ModalityType.Audio,
            ModalityType.Video, ModalityType.Thermal, ModalityType.Depth, ModalityType.IMU
        }.AsReadOnly();

        InferenceSession? imageEncoder = null;
        InferenceSession? textEncoder = null;
        InferenceSession? audioEncoder = null;

        try
        {
            imageEncoder = new InferenceSession(imageEncoderPath);
            textEncoder = new InferenceSession(textEncoderPath);
            audioEncoder = new InferenceSession(audioEncoderPath);
            _imageEncoder = imageEncoder;
            _textEncoder = textEncoder;
            _audioEncoder = audioEncoder;
            Guard.NotNull(tokenizer);
            _tokenizer = tokenizer;
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _lossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();
            InitializeLayers();
        }
        catch
        {
            imageEncoder?.Dispose();
            textEncoder?.Dispose();
            audioEncoder?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates an ImageBind network using native library layers.
    /// </summary>
    public ImageBindNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int imageSize = 224,
        int channels = 3,
        int patchSize = 14,
        int vocabularySize = 49408,
        int maxSequenceLength = 77,
        int embeddingDimension = 1024,
        int hiddenDim = 1280,
        int numEncoderLayers = 32,
        int numHeads = 16,
        int audioSampleRate = 16000,
        int audioMaxDuration = 10,
        int imuTimesteps = 2000,
        int numVideoFrames = 2,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        ImageBindOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new ImageBindOptions();
        Options = _options;
        _useNativeMode = true;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _hiddenDim = hiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numHeads = numHeads;
        _patchSize = patchSize;
        _vocabularySize = vocabularySize;
        _audioSampleRate = audioSampleRate;
        _audioMaxDuration = audioMaxDuration;
        _imuTimesteps = imuTimesteps;
        _numVideoFrames = numVideoFrames;

        _supportedModalities = new List<ModalityType>
        {
            ModalityType.Image, ModalityType.Text, ModalityType.Audio,
            ModalityType.Video, ModalityType.Thermal, ModalityType.Depth, ModalityType.IMU
        }.AsReadOnly();

        _tokenizer = tokenizer ?? Tokenization.ClipTokenizerFactory.CreateSimple();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();

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

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateImageBindLayers(
                _imageSize, channels, _patchSize, _hiddenDim, _embeddingDimension,
                _numEncoderLayers, _numHeads, _vocabularySize, _maxSequenceLength,
                _audioSampleRate, _audioMaxDuration, _imuTimesteps, _numVideoFrames));
        }

        BindNativeLayers();

        // Initialize positional embeddings
        int audioPatchSize = 16;
        int audioSeqLen = (_audioSampleRate * _audioMaxDuration) / 160;
        audioSeqLen = (audioSeqLen / audioPatchSize) * audioPatchSize;
        if (audioSeqLen < audioPatchSize) audioSeqLen = audioPatchSize;

        _imageClsToken = new Tensor<T>([1, _hiddenDim]);
        _imagePositionalEmbeddings = new Tensor<T>([numPatches + 1, _hiddenDim]);
        _textPositionalEmbeddings = new Tensor<T>([_maxSequenceLength, _hiddenDim]);
        _audioPositionalEmbeddings = new Tensor<T>([audioSeqLen / audioPatchSize + 1, _hiddenDim]);
        _thermalClsToken = new Tensor<T>([1, _hiddenDim]);
        _thermalPositionalEmbeddings = new Tensor<T>([numPatches + 1, _hiddenDim]);
        _depthClsToken = new Tensor<T>([1, _hiddenDim]);
        _depthPositionalEmbeddings = new Tensor<T>([numPatches + 1, _hiddenDim]);
        _imuPositionalEmbeddings = new Tensor<T>([_imuTimesteps, _hiddenDim]);
        _videoTemporalPositionalEmbeddings = new Tensor<T>([_numVideoFrames, _hiddenDim]);

        InitializeWeights();
    }

    /// <summary>
    /// Rebinds the modality-specific views to the authoritative base layer graph.
    /// </summary>
    /// <remarks>
    /// The base deserializer replaces <see cref="NeuralNetworkBase{T}.Layers"/> with restored
    /// layer instances. Keeping the constructor-created references here would make prediction
    /// and training continue through fresh random layers while parameter enumeration and COW
    /// cloning operate on the restored graph.
    /// </remarks>
    private void BindNativeLayers()
    {
        _imageEncoderLayers.Clear();
        _textEncoderLayers.Clear();
        _audioEncoderLayers.Clear();
        _thermalEncoderLayers.Clear();
        _depthEncoderLayers.Clear();
        _imuEncoderLayers.Clear();
        _videoTemporalLayers.Clear();

        int imuLayerCount = Math.Min(6, _numEncoderLayers);

        // Distribute layers to internal sub-lists
        int idx = 0;

        // Image encoder: PatchEmbed + numEncoderLayers + projection
        _imagePatchEmbedding = Layers[idx++];
        for (int i = 0; i < _numEncoderLayers; i++)
            _imageEncoderLayers.Add(Layers[idx++]);
        _imageProjection = Layers[idx++];

        // Text encoder: EmbeddingLayer + numEncoderLayers + projection
        _textTokenEmbedding = Layers[idx++];
        for (int i = 0; i < _numEncoderLayers; i++)
            _textEncoderLayers.Add(Layers[idx++]);
        _textProjection = Layers[idx++];

        // Audio encoder: PatchEmbed + numEncoderLayers + projection
        _audioConv = Layers[idx++];
        for (int i = 0; i < _numEncoderLayers; i++)
            _audioEncoderLayers.Add(Layers[idx++]);
        _audioProjection = Layers[idx++];

        // Thermal encoder: PatchEmbed + numEncoderLayers + projection
        _thermalPatchEmbedding = Layers[idx++];
        for (int i = 0; i < _numEncoderLayers; i++)
            _thermalEncoderLayers.Add(Layers[idx++]);
        _thermalProjection = Layers[idx++];

        // Depth encoder: PatchEmbed + numEncoderLayers + projection
        _depthPatchEmbedding = Layers[idx++];
        for (int i = 0; i < _numEncoderLayers; i++)
            _depthEncoderLayers.Add(Layers[idx++]);
        _depthProjection = Layers[idx++];

        // IMU encoder: DenseLayer + imuLayerCount + projection
        _imuEmbedding = Layers[idx++];
        for (int i = 0; i < imuLayerCount; i++)
            _imuEncoderLayers.Add(Layers[idx++]);
        _imuProjection = Layers[idx++];

        // Video temporal: 4 layers + projection
        for (int i = 0; i < 4; i++)
            _videoTemporalLayers.Add(Layers[idx++]);
        _videoProjection = Layers[idx++];
    }

    private void InitializeWeights()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        double scale = 0.02;

        InitializeMatrix(_imageClsToken, random, scale);
        InitializeMatrix(_imagePositionalEmbeddings, random, scale);
        InitializeMatrix(_textPositionalEmbeddings, random, scale);
        InitializeMatrix(_audioPositionalEmbeddings, random, scale);
        InitializeMatrix(_thermalClsToken, random, scale);
        InitializeMatrix(_thermalPositionalEmbeddings, random, scale);
        InitializeMatrix(_depthClsToken, random, scale);
        InitializeMatrix(_depthPositionalEmbeddings, random, scale);
        InitializeMatrix(_imuPositionalEmbeddings, random, scale);
        InitializeMatrix(_videoTemporalPositionalEmbeddings, random, scale);
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

    #region IImageBindModel Implementation

    /// <inheritdoc/>
    public Vector<T> GetImageEmbedding(Tensor<T> image)
    {
        if (_useNativeMode)
        {
            return EncodeImageNative(image);
        }
        else
        {
            return EncodeImageOnnx(image);
        }
    }

    /// <inheritdoc/>
    public Vector<T> GetTextEmbedding(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Text cannot be null or empty.", nameof(text));

        if (_useNativeMode)
        {
            return EncodeTextNative(text);
        }
        else
        {
            return EncodeTextOnnx(text);
        }
    }

    /// <inheritdoc/>
    public Vector<T> GetAudioEmbedding(Tensor<T> audioWaveform, int sampleRate = 16000)
    {
        if (_useNativeMode)
        {
            return EncodeAudioNative(audioWaveform, sampleRate);
        }
        else
        {
            return EncodeAudioOnnx(audioWaveform, sampleRate);
        }
    }

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
            // Fall back to image encoding for first frame in ONNX mode
            return EncodeImageOnnx(frameList[0]);
        }
    }

    /// <inheritdoc/>
    public Vector<T> GetThermalEmbedding(Tensor<T> thermalImage)
    {
        if (_useNativeMode)
        {
            return EncodeThermalNative(thermalImage);
        }
        else
        {
            // Fall back to image encoding in ONNX mode
            return EncodeImageOnnx(thermalImage);
        }
    }

    /// <inheritdoc/>
    public Vector<T> GetDepthEmbedding(Tensor<T> depthMap)
    {
        if (_useNativeMode)
        {
            return EncodeDepthNative(depthMap);
        }
        else
        {
            // Fall back to image encoding in ONNX mode
            var depthAs3D = ExpandToThreeChannels(depthMap);
            return EncodeImageOnnx(depthAs3D);
        }
    }

    /// <inheritdoc/>
    public Vector<T> GetIMUEmbedding(Tensor<T> imuData)
    {
        if (_useNativeMode)
        {
            return EncodeIMUNative(imuData);
        }
        else
        {
            throw new NotSupportedException("IMU encoding is only supported in native mode.");
        }
    }

    /// <inheritdoc/>
    public Vector<T> GetEmbedding(ModalityType modality, object data)
    {
        return modality switch
        {
            ModalityType.Image => GetImageEmbedding((Tensor<T>)data),
            ModalityType.Text => GetTextEmbedding((string)data),
            ModalityType.Audio => GetAudioEmbedding((Tensor<T>)data),
            ModalityType.Video => GetVideoEmbedding((IEnumerable<Tensor<T>>)data),
            ModalityType.Thermal => GetThermalEmbedding((Tensor<T>)data),
            ModalityType.Depth => GetDepthEmbedding((Tensor<T>)data),
            ModalityType.IMU => GetIMUEmbedding((Tensor<T>)data),
            _ => throw new ArgumentException($"Unsupported modality: {modality}", nameof(modality))
        };
    }

    /// <inheritdoc/>
    public T ComputeCrossModalSimilarity(Vector<T> embedding1, Vector<T> embedding2)
    {
        T similarity = NumOps.Zero;
        int length = Math.Min(embedding1.Length, embedding2.Length);

        for (int i = 0; i < length; i++)
        {
            similarity = NumOps.Add(similarity,
                NumOps.Multiply(embedding1[i], embedding2[i]));
        }

        return similarity;
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> CrossModalRetrieval(
        Vector<T> queryEmbedding,
        IEnumerable<Vector<T>> targetEmbeddings,
        int topK = 10)
    {
        var embeddings = targetEmbeddings.ToList();
        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < embeddings.Count; i++)
        {
            var similarity = ComputeCrossModalSimilarity(queryEmbedding, embeddings[i]);
            scores.Add((i, similarity));
        }

        return scores
            .OrderByDescending(s => NumOps.ToDouble(s.Score))
            .Take(topK);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(
        ModalityType modality,
        object data,
        IEnumerable<string> classLabels)
    {
        var labels = classLabels.ToList();

        // Handle empty label set
        if (labels.Count == 0)
        {
            return new Dictionary<string, T>();
        }

        var dataEmbedding = GetEmbedding(modality, data);
        var textEmbeddings = labels.Select(l => GetTextEmbedding($"a photo of {l}")).ToList();

        var similarities = new List<T>();
        foreach (var textEmb in textEmbeddings)
        {
            similarities.Add(ComputeCrossModalSimilarity(dataEmbedding, textEmb));
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
    public (ModalityType Modality, object Data, T Score) FindBestMatch(
        ModalityType queryModality,
        object queryData,
        IEnumerable<(ModalityType Modality, object Data)> candidates)
    {
        var candidateList = candidates.ToList();

        // Handle empty candidates set
        if (candidateList.Count == 0)
        {
            throw new ArgumentException("Candidates collection cannot be empty.", nameof(candidates));
        }

        var queryEmbedding = GetEmbedding(queryModality, queryData);

        T bestScore = NumOps.MinValue;
        int bestIndex = 0;

        for (int i = 0; i < candidateList.Count; i++)
        {
            var candidateEmbedding = GetEmbedding(candidateList[i].Modality, candidateList[i].Data);
            var score = ComputeCrossModalSimilarity(queryEmbedding, candidateEmbedding);

            if (NumOps.GreaterThan(score, bestScore))
            {
                bestScore = score;
                bestIndex = i;
            }
        }

        return (candidateList[bestIndex].Modality, candidateList[bestIndex].Data, bestScore);
    }

    /// <inheritdoc/>
    public T ComputeEmergentAudioTextSimilarity(Tensor<T> audio, string text)
    {
        var audioEmbedding = GetAudioEmbedding(audio);
        var textEmbedding = GetTextEmbedding(text);
        return ComputeCrossModalSimilarity(audioEmbedding, textEmbedding);
    }

    /// <inheritdoc/>
    public IEnumerable<(string Description, T Score)> GenerateDescriptions(
        ModalityType modality,
        object data,
        IEnumerable<string> candidateDescriptions,
        int topK = 5)
    {
        var dataEmbedding = GetEmbedding(modality, data);
        var descriptions = candidateDescriptions.ToList();
        var scores = new List<(string Description, T Score)>();

        foreach (var description in descriptions)
        {
            var textEmbedding = GetTextEmbedding(description);
            var similarity = ComputeCrossModalSimilarity(dataEmbedding, textEmbedding);
            scores.Add((description, similarity));
        }

        return scores
            .OrderByDescending(s => NumOps.ToDouble(s.Score))
            .Take(topK);
    }

    /// <inheritdoc/>
    public (T AlignmentScore, Dictionary<string, object> Details) ComputeAlignment(
        ModalityType modality1,
        object data1,
        ModalityType modality2,
        object data2)
    {
        var embedding1 = GetEmbedding(modality1, data1);
        var embedding2 = GetEmbedding(modality2, data2);
        var alignmentScore = ComputeCrossModalSimilarity(embedding1, embedding2);

        // Compute additional metrics
        T norm1 = VectorHelper.L2Norm(embedding1);
        T norm2 = VectorHelper.L2Norm(embedding2);

        var details = new Dictionary<string, object>
        {
            { "Modality1", modality1.ToString() },
            { "Modality2", modality2.ToString() },
            { "Norm1", NumOps.ToDouble(norm1) },
            { "Norm2", NumOps.ToDouble(norm2) },
            { "EmbeddingDimension", _embeddingDimension }
        };

        return (alignmentScore, details);
    }

    /// <inheritdoc/>
    public Vector<T> FuseModalities(
        Dictionary<ModalityType, Vector<T>> modalityEmbeddings,
        string fusionMethod = "mean")
    {
        var embeddings = modalityEmbeddings.Values.ToList();

        if (embeddings.Count == 0)
            throw new ArgumentException("No embeddings provided for fusion.", nameof(modalityEmbeddings));

        return fusionMethod.ToLowerInvariant() switch
        {
            "mean" => MeanFusion(embeddings),
            "concat" => ConcatFusion(embeddings),
            "attention" => AttentionFusion(embeddings),
            _ => MeanFusion(embeddings)
        };
    }

    #endregion

    #region Encoding Methods

    private Vector<T> EncodeImageNative(Tensor<T> image)
    {
        var encoded = EncodeImageNativeTensor(image);
        int embeddingDim = encoded.Shape[^1];

        // EncodeImageNativeTensor returns [batch, embeddingDim], and the loop below reads the first
        // embeddingDim flat elements -- row 0. This method returns one Vector<T>, and so does its
        // caller GetImageEmbedding, so there is no shape here that can carry a batch: every row past
        // the first would be dropped with nothing to tell the caller it happened. Refuse instead.
        long embeddingCount = encoded.Length / embeddingDim;
        if (embeddingCount != 1)
        {
            throw new ArgumentException(
                $"The image encoder produced {embeddingCount} embeddings, but this returns a single " +
                "vector. Pass one image rather than a batch, and call once per image to encode several.",
                nameof(image));
        }

        var result = new Vector<T>(embeddingDim);
        for (int i = 0; i < embeddingDim; i++)
        {
            result[i] = encoded[i];
        }

        return result;
    }

    /// <summary>
    /// Runs the native image branch entirely through tape-aware engine operations.
    /// ImageBind's <see cref="Layers"/> collection contains the image, text, audio,
    /// thermal, depth, IMU, and video towers consecutively, so the base class's
    /// sequential training forward is not a legal ImageBind graph.
    /// </summary>
    private Tensor<T> EncodeImageNativeTensor(
        Tensor<T> image,
        Dictionary<string, Tensor<T>>? activations = null)
    {
        if (_imagePatchEmbedding is null || _imageClsToken is null ||
            _imagePositionalEmbeddings is null || _imageProjection is null)
            throw new InvalidOperationException("Native image encoder not initialized.");

        var patches = _imagePatchEmbedding.Forward(image);
        if (activations is not null) activations["Image/PatchEmbedding"] = patches.Clone();
        if (patches.Rank is not (2 or 3))
            throw new InvalidOperationException($"Image patch encoder returned unsupported rank {patches.Rank}; expected [N,D] or [B,N,D].");

        int batch = patches.Rank == 3 ? patches.Shape[0] : 1;
        int tokens = patches.Rank == 3 ? patches.Shape[1] : patches.Shape[0];
        int hiddenDim = patches.Shape[^1];
        int sequenceLength = tokens + 1;
        if (_imageClsToken.Shape[1] < hiddenDim ||
            _imagePositionalEmbeddings.Shape[0] < sequenceLength ||
            _imagePositionalEmbeddings.Shape[1] < hiddenDim)
        {
            throw new InvalidOperationException("ImageBind positional or CLS embedding dimensions do not match the image encoder output.");
        }

        // Keep the model-owned CLS and positional tables on the active tape.
        // Copying their Data into fresh tensors makes the forward numerically
        // correct but leaves those registered trainable tensors unreachable.
        var clsRow = Engine.TensorSlice(
            _imageClsToken,
            [0, 0],
            [1, hiddenDim]);
        var cls = patches.Rank == 3
            ? Engine.TensorTile(Engine.Reshape(clsRow, [1, 1, hiddenDim]), [batch, 1, 1])
            : clsRow;

        var withCls = Engine.TensorConcatenate([cls, patches], axis: patches.Rank == 3 ? 1 : 0);
        var positionalRows = Engine.TensorSlice(
            _imagePositionalEmbeddings,
            [0, 0],
            [sequenceLength, hiddenDim]);
        var positional = patches.Rank == 3
            ? Engine.TensorTile(Engine.Reshape(positionalRows, [1, sequenceLength, hiddenDim]), [batch, 1, 1])
            : positionalRows;
        var positioned = Engine.TensorAdd(withCls, positional);
        if (activations is not null) activations["Image/PositionedTokens"] = positioned.Clone();

        var current = positioned;
        for (int i = 0; i < _imageEncoderLayers.Count; i++)
        {
            current = _imageEncoderLayers[i].Forward(current);
            if (activations is not null) activations[$"Image/Encoder_{i}"] = current.Clone();
        }

        Tensor<T> clsFeature;
        if (current.Rank == 3)
        {
            var clsSlice = Engine.TensorSlice(current, [0, 0, 0], [batch, 1, hiddenDim]);
            clsFeature = Engine.Reshape(clsSlice, [batch, hiddenDim]);
        }
        else
        {
            clsFeature = Engine.TensorSlice(current, [0, 0], [1, hiddenDim]);
        }

        var projected = _imageProjection.Forward(clsFeature);
        if (activations is not null) activations["Image/Projection"] = projected.Clone();
        var sumSquared = Engine.ReduceSum(
            Engine.TensorMultiply(projected, projected),
            [projected.Rank - 1],
            keepDims: true);
        var norm = Engine.TensorSqrt(Engine.TensorAddScalar(sumSquared, NumOps.FromDouble(1e-12)));
        var normalized = Engine.TensorBroadcastDivide(projected, norm);
        if (activations is not null) activations["Image/NormalizedEmbedding"] = normalized.Clone();
        return normalized;
    }

    private Vector<T> EncodeTextNative(string text)
    {
        if (_textTokenEmbedding is null || _textPositionalEmbeddings is null || _textProjection is null)
            throw new InvalidOperationException("Native text encoder not initialized.");

        var encoded = _tokenizer.Encode(text);
        var inputIds = encoded.TokenIds;

        var paddedIds = new List<int>();
        for (int i = 0; i < _maxSequenceLength; i++)
        {
            paddedIds.Add(i < inputIds.Count ? inputIds[i] : 0);
        }

        var tokenTensor = Tensor<T>.CreateDefault([_maxSequenceLength], NumOps.Zero);
        for (int i = 0; i < _maxSequenceLength; i++)
        {
            tokenTensor[i] = NumOps.FromDouble(paddedIds[i]);
        }

        var embedded = _textTokenEmbedding.Forward(tokenTensor);
        var positioned = AddPositionalEmbeddings(embedded, _textPositionalEmbeddings);

        var current = positioned;
        foreach (var layer in _textEncoderLayers)
        {
            current = layer.Forward(current);
        }

        var pooled = MeanPool(current);
        var projected = ProjectFeatures(pooled, _textProjection);
        return Normalize(projected);
    }

    private Vector<T> EncodeAudioNative(Tensor<T> audioWaveform, int sampleRate)
    {
        if (_audioConv is null || _audioPositionalEmbeddings is null || _audioProjection is null)
            throw new InvalidOperationException("Native audio encoder not initialized.");

        // Convert waveform to mel spectrogram (simplified)
        var melSpec = ComputeMelSpectrogram(audioWaveform, sampleRate);

        var patches = _audioConv.Forward(melSpec);
        var positioned = AddPositionalEmbeddings(patches, _audioPositionalEmbeddings);

        var current = positioned;
        foreach (var layer in _audioEncoderLayers)
        {
            current = layer.Forward(current);
        }

        var pooled = MeanPool(current);
        var projected = ProjectFeatures(pooled, _audioProjection);
        return Normalize(projected);
    }

    private Vector<T> EncodeVideoNative(List<Tensor<T>> frames)
    {
        // Sample frames if needed
        var sampledFrames = SampleFrames(frames, _numVideoFrames);

        // Encode each frame using image encoder
        var frameFeatures = new List<Vector<T>>();
        foreach (var frame in sampledFrames)
        {
            var features = EncodeImageNative(frame);
            frameFeatures.Add(features);
        }

        // Stack frame features and add temporal positional embeddings
        var stackedFeatures = Tensor<T>.CreateDefault([frameFeatures.Count, _hiddenDim], NumOps.Zero);
        for (int i = 0; i < frameFeatures.Count; i++)
        {
            for (int j = 0; j < Math.Min(frameFeatures[i].Length, _hiddenDim); j++)
            {
                stackedFeatures[i, j] = frameFeatures[i][j];
            }
        }

        if (_videoTemporalPositionalEmbeddings is not null)
        {
            for (int i = 0; i < stackedFeatures.Shape[0] && i < _videoTemporalPositionalEmbeddings.Shape[0]; i++)
            {
                for (int j = 0; j < _hiddenDim && j < _videoTemporalPositionalEmbeddings.Shape[1]; j++)
                {
                    stackedFeatures[i, j] = NumOps.Add(stackedFeatures[i, j], _videoTemporalPositionalEmbeddings[i, j]);
                }
            }
        }

        // Apply temporal transformer
        var current = stackedFeatures;
        foreach (var layer in _videoTemporalLayers)
        {
            current = layer.Forward(current);
        }

        var pooled = MeanPool(current);
        var projected = ProjectFeatures(pooled, _videoProjection);
        return Normalize(projected);
    }

    private Vector<T> EncodeThermalNative(Tensor<T> thermalImage)
    {
        if (_thermalPatchEmbedding is null || _thermalClsToken is null ||
            _thermalPositionalEmbeddings is null || _thermalProjection is null)
            throw new InvalidOperationException("Native thermal encoder not initialized.");

        // Ensure single channel input
        var input = EnsureSingleChannel(thermalImage);

        var patches = _thermalPatchEmbedding.Forward(input);
        var withCls = PrependClsToken(patches, _thermalClsToken);
        var positioned = AddPositionalEmbeddings(withCls, _thermalPositionalEmbeddings);

        var current = positioned;
        foreach (var layer in _thermalEncoderLayers)
        {
            current = layer.Forward(current);
        }

        var clsFeature = ExtractClsToken(current);
        var projected = ProjectFeatures(clsFeature, _thermalProjection);
        return Normalize(projected);
    }

    private Vector<T> EncodeDepthNative(Tensor<T> depthMap)
    {
        if (_depthPatchEmbedding is null || _depthClsToken is null ||
            _depthPositionalEmbeddings is null || _depthProjection is null)
            throw new InvalidOperationException("Native depth encoder not initialized.");

        // Ensure single channel input
        var input = EnsureSingleChannel(depthMap);

        var patches = _depthPatchEmbedding.Forward(input);
        var withCls = PrependClsToken(patches, _depthClsToken);
        var positioned = AddPositionalEmbeddings(withCls, _depthPositionalEmbeddings);

        var current = positioned;
        foreach (var layer in _depthEncoderLayers)
        {
            current = layer.Forward(current);
        }

        var clsFeature = ExtractClsToken(current);
        var projected = ProjectFeatures(clsFeature, _depthProjection);
        return Normalize(projected);
    }

    private Vector<T> EncodeIMUNative(Tensor<T> imuData)
    {
        if (_imuEmbedding is null || _imuPositionalEmbeddings is null || _imuProjection is null)
            throw new InvalidOperationException("Native IMU encoder not initialized.");

        // IMU data shape: [timesteps, 6] (3 accel + 3 gyro)
        int timesteps = imuData.Shape[0];
        int features = imuData.Shape[1];

        // Project each timestep
        var embedded = Tensor<T>.CreateDefault([timesteps, _hiddenDim], NumOps.Zero);
        for (int t = 0; t < timesteps; t++)
        {
            var timestepData = Tensor<T>.CreateDefault([1, features], NumOps.Zero);
            for (int f = 0; f < features; f++)
            {
                timestepData[0, f] = imuData[t, f];
            }
            var embeddedTimestep = _imuEmbedding.Forward(timestepData);
            for (int j = 0; j < _hiddenDim && j < embeddedTimestep.Shape[1]; j++)
            {
                embedded[t, j] = embeddedTimestep[0, j];
            }
        }

        // Add positional embeddings
        if (_imuPositionalEmbeddings is not null)
        {
            for (int i = 0; i < timesteps && i < _imuPositionalEmbeddings.Shape[0]; i++)
            {
                for (int j = 0; j < _hiddenDim && j < _imuPositionalEmbeddings.Shape[1]; j++)
                {
                    embedded[i, j] = NumOps.Add(embedded[i, j], _imuPositionalEmbeddings[i, j]);
                }
            }
        }

        var current = embedded;
        foreach (var layer in _imuEncoderLayers)
        {
            current = layer.Forward(current);
        }

        var pooled = MeanPool(current);
        var projected = ProjectFeatures(pooled, _imuProjection);
        return Normalize(projected);
    }

    #endregion

    #region ONNX Encoding Methods

    private Vector<T> EncodeImageOnnx(Tensor<T> image)
    {
        if (_imageEncoder is null)
            throw new InvalidOperationException("ONNX image encoder not initialized.");

        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        var inputArray = new float[1 * channels * height * width];
        int idx = 0;
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    inputArray[idx++] = (float)NumOps.ToDouble(image[c, h, w]);
                }
            }
        }

        var inputTensor = new OnnxTensors.DenseTensor<float>(inputArray, [1, channels, height, width]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", inputTensor)
        };

        using var results = _imageEncoder.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var embedding = new Vector<T>(_embeddingDimension);
        for (int i = 0; i < _embeddingDimension && i < outputTensor.Length; i++)
        {
            embedding[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return Normalize(embedding);
    }

    private Vector<T> EncodeTextOnnx(string text)
    {
        if (_textEncoder is null)
            throw new InvalidOperationException("ONNX text encoder not initialized.");

        var encoded = _tokenizer.Encode(text);
        var inputIds = encoded.TokenIds;

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

    private Vector<T> EncodeAudioOnnx(Tensor<T> audioWaveform, int sampleRate)
    {
        if (_audioEncoder is null)
            throw new InvalidOperationException("ONNX audio encoder not initialized.");

        // Compute mel spectrogram
        var melSpec = ComputeMelSpectrogram(audioWaveform, sampleRate);

        int melBins = melSpec.Shape[0];
        int timeSteps = melSpec.Shape[1];

        var inputArray = new float[1 * 1 * melBins * timeSteps];
        int idx = 0;
        for (int m = 0; m < melBins; m++)
        {
            for (int t = 0; t < timeSteps; t++)
            {
                inputArray[idx++] = (float)NumOps.ToDouble(melSpec[m, t]);
            }
        }

        var inputTensor = new OnnxTensors.DenseTensor<float>(inputArray, [1, 1, melBins, timeSteps]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_values", inputTensor)
        };

        using var results = _audioEncoder.Run(inputs);
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

    private Tensor<T> ComputeMelSpectrogram(Tensor<T> waveform, int sampleRate)
    {
        // Simplified mel spectrogram computation
        int numSamples = waveform.Shape.Length == 1 ? waveform.Shape[0] : waveform.Shape[1];
        int hopLength = 160;
        int numMelBins = 128;
        int numFrames = Math.Max(1, numSamples / hopLength);

        var melSpec = Tensor<T>.CreateDefault([numMelBins, numFrames], NumOps.Zero);

        // Simple energy-based features as placeholder
        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * hopLength;
            double energy = 0;
            int count = 0;

            for (int i = start; i < Math.Min(start + hopLength * 2, numSamples); i++)
            {
                double val = waveform.Shape.Length == 1
                    ? NumOps.ToDouble(waveform[i])
                    : NumOps.ToDouble(waveform[0, i]);
                energy += val * val;
                count++;
            }

            if (count > 0) energy /= count;
            double logEnergy = Math.Log(energy + 1e-10);

            for (int mel = 0; mel < numMelBins; mel++)
            {
                // Simple frequency-dependent scaling
                double scale = 1.0 - (double)mel / numMelBins * 0.5;
                melSpec[mel, frame] = NumOps.FromDouble(logEnergy * scale);
            }
        }

        return melSpec;
    }

    private Tensor<T> EnsureSingleChannel(Tensor<T> input)
    {
        if (input.Shape.Length == 2)
        {
            // [H, W] -> [1, H, W]
            var result = Tensor<T>.CreateDefault([1, input.Shape[0], input.Shape[1]], NumOps.Zero);
            for (int h = 0; h < input.Shape[0]; h++)
            {
                for (int w = 0; w < input.Shape[1]; w++)
                {
                    result[0, h, w] = input[h, w];
                }
            }
            return result;
        }
        else if (input.Shape.Length == 3 && input.Shape[0] > 1)
        {
            // [C, H, W] -> [1, H, W] (take mean)
            var result = Tensor<T>.CreateDefault([1, input.Shape[1], input.Shape[2]], NumOps.Zero);
            for (int h = 0; h < input.Shape[1]; h++)
            {
                for (int w = 0; w < input.Shape[2]; w++)
                {
                    T sum = NumOps.Zero;
                    for (int c = 0; c < input.Shape[0]; c++)
                    {
                        sum = NumOps.Add(sum, input[c, h, w]);
                    }
                    result[0, h, w] = NumOps.Divide(sum, NumOps.FromDouble(input.Shape[0]));
                }
            }
            return result;
        }

        return input;
    }

    private Tensor<T> ExpandToThreeChannels(Tensor<T> input)
    {
        var singleChannel = EnsureSingleChannel(input);
        var result = Tensor<T>.CreateDefault([3, singleChannel.Shape[1], singleChannel.Shape[2]], NumOps.Zero);

        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < singleChannel.Shape[1]; h++)
            {
                for (int w = 0; w < singleChannel.Shape[2]; w++)
                {
                    result[c, h, w] = singleChannel[0, h, w];
                }
            }
        }

        return result;
    }

    private List<Tensor<T>> SampleFrames(List<Tensor<T>> frames, int targetCount)
    {
        if (frames.Count <= targetCount)
        {
            var result = new List<Tensor<T>>(frames);
            while (result.Count < targetCount)
            {
                result.Add(frames[frames.Count - 1]);
            }
            return result;
        }

        var sampled = new List<Tensor<T>>();
        double step = (double)frames.Count / targetCount;
        for (int i = 0; i < targetCount; i++)
        {
            int idx = Math.Min((int)(i * step), frames.Count - 1);
            sampled.Add(frames[idx]);
        }

        return sampled;
    }

    private Tensor<T> PrependClsToken(Tensor<T> sequence, Tensor<T> clsToken)
    {
        int seqLen = sequence.Shape[0];
        int hiddenDim = sequence.Shape[1];

        var result = Tensor<T>.CreateDefault([seqLen + 1, hiddenDim], NumOps.Zero);

        for (int j = 0; j < hiddenDim && j < clsToken.Shape[1]; j++)
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

    private Tensor<T> AddPositionalEmbeddings(Tensor<T> sequence, Tensor<T> posEmbeddings)
    {
        int seqLen = sequence.Shape[0];
        int hiddenDim = sequence.Shape[1];

        var result = Tensor<T>.CreateDefault([seqLen, hiddenDim], NumOps.Zero);

        for (int i = 0; i < seqLen && i < posEmbeddings.Shape[0]; i++)
        {
            for (int j = 0; j < hiddenDim && j < posEmbeddings.Shape[1]; j++)
            {
                result[i, j] = NumOps.Add(sequence[i, j], posEmbeddings[i, j]);
            }
        }

        return result;
    }

    private Vector<T> ExtractClsToken(Tensor<T> sequence)
    {
        int hiddenDim = sequence.Shape[1];
        var result = new Vector<T>(hiddenDim);

        for (int j = 0; j < hiddenDim; j++)
        {
            result[j] = sequence[0, j];
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

    private Vector<T> ProjectFeatures(Vector<T> features, ILayer<T>? projection)
    {
        if (projection is null)
            return features;

        var tensor = Tensor<T>.CreateDefault([1, features.Length], NumOps.Zero);
        for (int i = 0; i < features.Length; i++)
        {
            tensor[0, i] = features[i];
        }

        var projected = projection.Forward(tensor);

        var result = new Vector<T>(_embeddingDimension);
        for (int i = 0; i < _embeddingDimension && i < projected.Shape[1]; i++)
        {
            result[i] = projected[0, i];
        }

        return result;
    }

    private Vector<T> Normalize(Vector<T> vector)
    {
        return VectorHelper.Normalize(vector);
    }

    private List<T> Softmax(List<T> values)
    {
        double maxVal = values.Max(v => NumOps.ToDouble(v));
        var expValues = values.Select(v => Math.Exp(NumOps.ToDouble(v) - maxVal)).ToList();
        double sumExp = expValues.Sum();
        return expValues.Select(e => NumOps.FromDouble(e / sumExp)).ToList();
    }

    private Vector<T> MeanFusion(List<Vector<T>> embeddings)
    {
        int dim = embeddings[0].Length;
        var result = new Vector<T>(dim);

        for (int j = 0; j < dim; j++)
        {
            T sum = NumOps.Zero;
            foreach (var emb in embeddings)
            {
                if (j < emb.Length)
                    sum = NumOps.Add(sum, emb[j]);
            }
            result[j] = NumOps.Divide(sum, NumOps.FromDouble(embeddings.Count));
        }

        return Normalize(result);
    }

    private Vector<T> ConcatFusion(List<Vector<T>> embeddings)
    {
        int totalDim = embeddings.Sum(e => e.Length);
        var result = new Vector<T>(totalDim);

        int offset = 0;
        foreach (var emb in embeddings)
        {
            for (int i = 0; i < emb.Length; i++)
            {
                result[offset + i] = emb[i];
            }
            offset += emb.Length;
        }

        return Normalize(result);
    }

    private Vector<T> AttentionFusion(List<Vector<T>> embeddings)
    {
        // Simple attention: compute attention weights based on embedding norms
        var weights = new List<T>();
        foreach (var emb in embeddings)
        {
            weights.Add(VectorHelper.L2Norm(emb));
        }

        var softmaxWeights = Softmax(weights);

        int dim = embeddings[0].Length;
        var result = new Vector<T>(dim);

        for (int j = 0; j < dim; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < embeddings.Count; i++)
            {
                if (j < embeddings[i].Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(softmaxWeights[i], embeddings[i][j]));
                }
            }
            result[j] = sum;
        }

        return Normalize(result);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Declares the CLS tokens and positional embedding tables for all six modalities, which live
    /// outside <see cref="NeuralNetworkBase{T}.Layers"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Declared in the order the deleted GetParameters concatenated them, so existing checkpoints
    /// still restore: image CLS, image positional, text, audio, thermal CLS, thermal positional,
    /// depth CLS, depth positional, IMU, video-temporal.
    /// </para>
    /// <para>
    /// This replaces 350 lines -- ParameterCount, GetParameters, SetParameters, UpdateParameters
    /// and six private helpers (AppendLayerListParameters, AppendSingleLayerParameters,
    /// AppendMatrixParameters and their Update counterparts) -- each walking the same seven encoder
    /// towers, thirteen projections and ten tables with its own running offset. Ten modalities'
    /// worth of layout repeated four times, where a single missed line in any one of them silently
    /// misaligns a checkpoint.
    /// </para>
    /// <para>
    /// The towers need no declaration and must not get one: every per-modality list is filled FROM
    /// <c>Layers</c> (<c>Layers[idx++]</c>), so they are typed views of layers the base walk already
    /// reaches and declaring them would double-count. The tables became <c>Tensor&lt;T&gt;</c>
    /// because a <c>Matrix&lt;T&gt;</c> is invisible to the trainable-parameter walk, which is the
    /// reason these surfaces had to exist at all.
    /// </para>
    /// </remarks>
    protected override IEnumerable<Tensor<T>> GetExtraTrainableTensors()
    {
        if (!_useNativeMode)
        {
            yield break;
        }

        Tensor<T>?[] tables =
        [
            _imageClsToken,
            _imagePositionalEmbeddings,
            _textPositionalEmbeddings,
            _audioPositionalEmbeddings,
            _thermalClsToken,
            _thermalPositionalEmbeddings,
            _depthClsToken,
            _depthPositionalEmbeddings,
            _imuPositionalEmbeddings,
            _videoTemporalPositionalEmbeddings,
        ];

        foreach (var table in tables)
        {
            if (table is not null)
            {
                yield return table;
            }
        }
    }

    // ---- behavioural overrides, restored ----
    // These four were deleted as collateral when this file's parameter surfaces were removed:
    // the deletion took a LINE RANGE that happened to contain them, rather than the members it
    // had identified. Without PredictCore, ImageBind falls through to the base, which chains the
    // flat Layers list -- and Layers is all seven modality towers concatenated, so image input
    // reaches the TEXT tower's embedding and EmbeddingLayer throws "in Indices mode". 25 tests
    // went from passing to failing. The parameter work itself was correct and is unchanged.
    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        SetTrainingMode(false);
        if (_useNativeMode)
            return Accelerate(input, () => EncodeImageNativeTensor(input));

        return Accelerate(input, () =>
        {
            var embedding = EncodeImageOnnx(input);
            var result = Tensor<T>.CreateDefault([1, embedding.Length], NumOps.Zero);
            for (int i = 0; i < embedding.Length; i++)
            {
                result[0, i] = embedding[i];
            }
            return result;
        });
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training is not supported for ONNX ImageBind models.");

        return EncodeImageNativeTensor(input);
    }

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        SetTrainingMode(false);

        if (_useNativeMode)
        {
            EncodeImageNativeTensor(input, activations);
        }
        else
        {
            activations["Image/OnnxEmbedding"] = Predict(input).Clone();
        }

        return activations;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainWithTape(input, expectedOutput, _optimizer);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ImageSize", _imageSize },
                { "EmbeddingDimension", _embeddingDimension },
                { "MaxSequenceLength", _maxSequenceLength },
                { "HiddenDim", _hiddenDim },
                { "NumEncoderLayers", _numEncoderLayers },
                { "NumHeads", _numHeads },
                { "VocabularySize", _vocabularySize },
                { "AudioSampleRate", _audioSampleRate },
                { "AudioMaxDuration", _audioMaxDuration },
                { "IMUTimesteps", _imuTimesteps },
                { "NumVideoFrames", _numVideoFrames },
                { "SupportedModalities", _supportedModalities.Select(m => m.ToString()).ToList() },
                { "UseNativeMode", _useNativeMode }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _imageEncoder?.Dispose();
            _textEncoder?.Dispose();
            _audioEncoder?.Dispose();
        }

        base.Dispose(disposing);
    }

    #endregion
}






