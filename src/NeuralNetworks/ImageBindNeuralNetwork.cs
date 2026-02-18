using System.IO;
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
public class ImageBindNeuralNetwork<T> : NeuralNetworkBase<T>, IImageBindModel<T>
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
    private Matrix<T>? _imageClsToken;
    private Matrix<T>? _imagePositionalEmbeddings;
    private ILayer<T>? _imagePatchEmbedding;
    private ILayer<T>? _imageProjection;

    // Text encoder layers
    private readonly List<ILayer<T>> _textEncoderLayers = [];
    private Matrix<T>? _textPositionalEmbeddings;
    private ILayer<T>? _textTokenEmbedding;
    private ILayer<T>? _textProjection;

    // Audio encoder layers (uses spectrogram input)
    private readonly List<ILayer<T>> _audioEncoderLayers = [];
    private Matrix<T>? _audioPositionalEmbeddings;
    private ILayer<T>? _audioConv;
    private ILayer<T>? _audioProjection;

    // Thermal encoder (similar to image encoder)
    private readonly List<ILayer<T>> _thermalEncoderLayers = [];
    private Matrix<T>? _thermalClsToken;
    private Matrix<T>? _thermalPositionalEmbeddings;
    private ILayer<T>? _thermalPatchEmbedding;
    private ILayer<T>? _thermalProjection;

    // Depth encoder
    private readonly List<ILayer<T>> _depthEncoderLayers = [];
    private Matrix<T>? _depthClsToken;
    private Matrix<T>? _depthPositionalEmbeddings;
    private ILayer<T>? _depthPatchEmbedding;
    private ILayer<T>? _depthProjection;

    // IMU encoder
    private readonly List<ILayer<T>> _imuEncoderLayers = [];
    private Matrix<T>? _imuPositionalEmbeddings;
    private ILayer<T>? _imuEmbedding;
    private ILayer<T>? _imuProjection;

    // Video encoder (temporal aggregation over frames)
    private readonly List<ILayer<T>> _videoTemporalLayers = [];
    private Matrix<T>? _videoTemporalPositionalEmbeddings;
    private ILayer<T>? _videoProjection;

    #endregion

    #region Shared Fields

    private readonly ITokenizer _tokenizer;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
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
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        ImageBindOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
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
            _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
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
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        ImageBindOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
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
        int imuLayerCount = Math.Min(6, _numEncoderLayers);

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

        // Initialize positional embeddings
        int audioPatchSize = 16;
        int audioSeqLen = (_audioSampleRate * _audioMaxDuration) / 160;
        audioSeqLen = (audioSeqLen / audioPatchSize) * audioPatchSize;
        if (audioSeqLen < audioPatchSize) audioSeqLen = audioPatchSize;

        _imageClsToken = Matrix<T>.CreateDefault(1, _hiddenDim, NumOps.Zero);
        _imagePositionalEmbeddings = Matrix<T>.CreateDefault(numPatches + 1, _hiddenDim, NumOps.Zero);
        _textPositionalEmbeddings = Matrix<T>.CreateDefault(_maxSequenceLength, _hiddenDim, NumOps.Zero);
        _audioPositionalEmbeddings = Matrix<T>.CreateDefault(audioSeqLen / audioPatchSize + 1, _hiddenDim, NumOps.Zero);
        _thermalClsToken = Matrix<T>.CreateDefault(1, _hiddenDim, NumOps.Zero);
        _thermalPositionalEmbeddings = Matrix<T>.CreateDefault(numPatches + 1, _hiddenDim, NumOps.Zero);
        _depthClsToken = Matrix<T>.CreateDefault(1, _hiddenDim, NumOps.Zero);
        _depthPositionalEmbeddings = Matrix<T>.CreateDefault(numPatches + 1, _hiddenDim, NumOps.Zero);
        _imuPositionalEmbeddings = Matrix<T>.CreateDefault(_imuTimesteps, _hiddenDim, NumOps.Zero);
        _videoTemporalPositionalEmbeddings = Matrix<T>.CreateDefault(_numVideoFrames, _hiddenDim, NumOps.Zero);

        InitializeWeights();
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

        T bestScore = NumOps.FromDouble(double.MinValue);
        int bestIndex = 0;

        for (int i = 0; i < candidateList.Count; i++)
        {
            var candidateEmbedding = GetEmbedding(candidateList[i].Modality, candidateList[i].Data);
            var score = ComputeCrossModalSimilarity(queryEmbedding, candidateEmbedding);

            if (NumOps.ToDouble(score) > NumOps.ToDouble(bestScore))
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
        T norm1 = ComputeNorm(embedding1);
        T norm2 = ComputeNorm(embedding2);

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
        if (_imagePatchEmbedding is null || _imageClsToken is null ||
            _imagePositionalEmbeddings is null || _imageProjection is null)
            throw new InvalidOperationException("Native image encoder not initialized.");

        var patches = _imagePatchEmbedding.Forward(image);
        var withCls = PrependClsToken(patches, _imageClsToken);
        var positioned = AddPositionalEmbeddings(withCls, _imagePositionalEmbeddings);

        var current = positioned;
        foreach (var layer in _imageEncoderLayers)
        {
            current = layer.Forward(current);
        }

        var clsFeature = ExtractClsToken(current);
        var projected = ProjectFeatures(clsFeature, _imageProjection);
        return Normalize(projected);
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
            for (int i = 0; i < stackedFeatures.Shape[0] && i < _videoTemporalPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < _hiddenDim && j < _videoTemporalPositionalEmbeddings.Columns; j++)
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
            for (int i = 0; i < timesteps && i < _imuPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < _hiddenDim && j < _imuPositionalEmbeddings.Columns; j++)
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

    private T ComputeNorm(Vector<T> vector)
    {
        T sumSquares = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(vector[i], vector[i]));
        }
        return NumOps.Sqrt(sumSquares);
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
            weights.Add(ComputeNorm(emb));
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

    /// <inheritdoc/>
    public override int ParameterCount
    {
        get
        {
            if (!_useNativeMode)
            {
                return 0;
            }

            int count = 0;

            // Image encoder layers
            foreach (var layer in _imageEncoderLayers)
            {
                count += layer.ParameterCount;
            }

            // Text encoder layers
            foreach (var layer in _textEncoderLayers)
            {
                count += layer.ParameterCount;
            }

            // Audio encoder layers
            foreach (var layer in _audioEncoderLayers)
            {
                count += layer.ParameterCount;
            }

            // Thermal encoder layers
            foreach (var layer in _thermalEncoderLayers)
            {
                count += layer.ParameterCount;
            }

            // Depth encoder layers
            foreach (var layer in _depthEncoderLayers)
            {
                count += layer.ParameterCount;
            }

            // IMU encoder layers
            foreach (var layer in _imuEncoderLayers)
            {
                count += layer.ParameterCount;
            }

            // Video temporal layers
            foreach (var layer in _videoTemporalLayers)
            {
                count += layer.ParameterCount;
            }

            // Single layers
            if (_imagePatchEmbedding is not null) count += _imagePatchEmbedding.ParameterCount;
            if (_imageProjection is not null) count += _imageProjection.ParameterCount;
            if (_textTokenEmbedding is not null) count += _textTokenEmbedding.ParameterCount;
            if (_textProjection is not null) count += _textProjection.ParameterCount;
            if (_audioConv is not null) count += _audioConv.ParameterCount;
            if (_audioProjection is not null) count += _audioProjection.ParameterCount;
            if (_thermalPatchEmbedding is not null) count += _thermalPatchEmbedding.ParameterCount;
            if (_thermalProjection is not null) count += _thermalProjection.ParameterCount;
            if (_depthPatchEmbedding is not null) count += _depthPatchEmbedding.ParameterCount;
            if (_depthProjection is not null) count += _depthProjection.ParameterCount;
            if (_imuEmbedding is not null) count += _imuEmbedding.ParameterCount;
            if (_imuProjection is not null) count += _imuProjection.ParameterCount;
            if (_videoProjection is not null) count += _videoProjection.ParameterCount;

            // Positional embeddings and CLS tokens
            if (_imageClsToken is not null) count += _imageClsToken.Rows * _imageClsToken.Columns;
            if (_imagePositionalEmbeddings is not null) count += _imagePositionalEmbeddings.Rows * _imagePositionalEmbeddings.Columns;
            if (_textPositionalEmbeddings is not null) count += _textPositionalEmbeddings.Rows * _textPositionalEmbeddings.Columns;
            if (_audioPositionalEmbeddings is not null) count += _audioPositionalEmbeddings.Rows * _audioPositionalEmbeddings.Columns;
            if (_thermalClsToken is not null) count += _thermalClsToken.Rows * _thermalClsToken.Columns;
            if (_thermalPositionalEmbeddings is not null) count += _thermalPositionalEmbeddings.Rows * _thermalPositionalEmbeddings.Columns;
            if (_depthClsToken is not null) count += _depthClsToken.Rows * _depthClsToken.Columns;
            if (_depthPositionalEmbeddings is not null) count += _depthPositionalEmbeddings.Rows * _depthPositionalEmbeddings.Columns;
            if (_imuPositionalEmbeddings is not null) count += _imuPositionalEmbeddings.Rows * _imuPositionalEmbeddings.Columns;
            if (_videoTemporalPositionalEmbeddings is not null) count += _videoTemporalPositionalEmbeddings.Rows * _videoTemporalPositionalEmbeddings.Columns;

            return count;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(ParameterCount);
        if (!_useNativeMode)
        {
            return parameters;
        }

        int offset = 0;
        offset = AppendLayerListParameters(_imageEncoderLayers, parameters, offset);
        offset = AppendLayerListParameters(_textEncoderLayers, parameters, offset);
        offset = AppendLayerListParameters(_audioEncoderLayers, parameters, offset);
        offset = AppendLayerListParameters(_thermalEncoderLayers, parameters, offset);
        offset = AppendLayerListParameters(_depthEncoderLayers, parameters, offset);
        offset = AppendLayerListParameters(_imuEncoderLayers, parameters, offset);
        offset = AppendLayerListParameters(_videoTemporalLayers, parameters, offset);

        offset = AppendSingleLayerParameters(_imagePatchEmbedding, parameters, offset);
        offset = AppendSingleLayerParameters(_imageProjection, parameters, offset);
        offset = AppendSingleLayerParameters(_textTokenEmbedding, parameters, offset);
        offset = AppendSingleLayerParameters(_textProjection, parameters, offset);
        offset = AppendSingleLayerParameters(_audioConv, parameters, offset);
        offset = AppendSingleLayerParameters(_audioProjection, parameters, offset);
        offset = AppendSingleLayerParameters(_thermalPatchEmbedding, parameters, offset);
        offset = AppendSingleLayerParameters(_thermalProjection, parameters, offset);
        offset = AppendSingleLayerParameters(_depthPatchEmbedding, parameters, offset);
        offset = AppendSingleLayerParameters(_depthProjection, parameters, offset);
        offset = AppendSingleLayerParameters(_imuEmbedding, parameters, offset);
        offset = AppendSingleLayerParameters(_imuProjection, parameters, offset);
        offset = AppendSingleLayerParameters(_videoProjection, parameters, offset);

        offset = AppendMatrixParameters(_imageClsToken, parameters, offset);
        offset = AppendMatrixParameters(_imagePositionalEmbeddings, parameters, offset);
        offset = AppendMatrixParameters(_textPositionalEmbeddings, parameters, offset);
        offset = AppendMatrixParameters(_audioPositionalEmbeddings, parameters, offset);
        offset = AppendMatrixParameters(_thermalClsToken, parameters, offset);
        offset = AppendMatrixParameters(_thermalPositionalEmbeddings, parameters, offset);
        offset = AppendMatrixParameters(_depthClsToken, parameters, offset);
        offset = AppendMatrixParameters(_depthPositionalEmbeddings, parameters, offset);
        offset = AppendMatrixParameters(_imuPositionalEmbeddings, parameters, offset);
        offset = AppendMatrixParameters(_videoTemporalPositionalEmbeddings, parameters, offset);

        return parameters;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        UpdateParameters(parameters);
    }
    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        SetTrainingMode(false);
        var embedding = GetImageEmbedding(input);
        var result = Tensor<T>.CreateDefault([1, embedding.Length], NumOps.Zero);
        for (int i = 0; i < embedding.Length; i++)
        {
            result[0, i] = embedding[i];
        }
        return result;
    }

    /// <summary>
    /// Backward pass through encoder layers.
    /// </summary>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Backward pass is only supported in native mode.");
        }

        var currentGradient = gradient;

        for (int i = _imageEncoderLayers.Count - 1; i >= 0; i--)
        {
            currentGradient = _imageEncoderLayers[i].Backward(currentGradient);
        }

        return currentGradient;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);

        var embedding = GetImageEmbedding(input);
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

        if (!_useNativeMode)
        {
            return;
        }

        int offset = 0;
        offset = UpdateLayerListParameters(_imageEncoderLayers, parameters, offset);
        offset = UpdateLayerListParameters(_textEncoderLayers, parameters, offset);
        offset = UpdateLayerListParameters(_audioEncoderLayers, parameters, offset);
        offset = UpdateLayerListParameters(_thermalEncoderLayers, parameters, offset);
        offset = UpdateLayerListParameters(_depthEncoderLayers, parameters, offset);
        offset = UpdateLayerListParameters(_imuEncoderLayers, parameters, offset);
        offset = UpdateLayerListParameters(_videoTemporalLayers, parameters, offset);

        offset = UpdateSingleLayerParameters(_imagePatchEmbedding, parameters, offset);
        offset = UpdateSingleLayerParameters(_imageProjection, parameters, offset);
        offset = UpdateSingleLayerParameters(_textTokenEmbedding, parameters, offset);
        offset = UpdateSingleLayerParameters(_textProjection, parameters, offset);
        offset = UpdateSingleLayerParameters(_audioConv, parameters, offset);
        offset = UpdateSingleLayerParameters(_audioProjection, parameters, offset);
        offset = UpdateSingleLayerParameters(_thermalPatchEmbedding, parameters, offset);
        offset = UpdateSingleLayerParameters(_thermalProjection, parameters, offset);
        offset = UpdateSingleLayerParameters(_depthPatchEmbedding, parameters, offset);
        offset = UpdateSingleLayerParameters(_depthProjection, parameters, offset);
        offset = UpdateSingleLayerParameters(_imuEmbedding, parameters, offset);
        offset = UpdateSingleLayerParameters(_imuProjection, parameters, offset);
        offset = UpdateSingleLayerParameters(_videoProjection, parameters, offset);

        offset = UpdateMatrixParameters(_imageClsToken, parameters, offset);
        offset = UpdateMatrixParameters(_imagePositionalEmbeddings, parameters, offset);
        offset = UpdateMatrixParameters(_textPositionalEmbeddings, parameters, offset);
        offset = UpdateMatrixParameters(_audioPositionalEmbeddings, parameters, offset);
        offset = UpdateMatrixParameters(_thermalClsToken, parameters, offset);
        offset = UpdateMatrixParameters(_thermalPositionalEmbeddings, parameters, offset);
        offset = UpdateMatrixParameters(_depthClsToken, parameters, offset);
        offset = UpdateMatrixParameters(_depthPositionalEmbeddings, parameters, offset);
        offset = UpdateMatrixParameters(_imuPositionalEmbeddings, parameters, offset);
        offset = UpdateMatrixParameters(_videoTemporalPositionalEmbeddings, parameters, offset);
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

    private int AppendLayerListParameters(List<ILayer<T>> layers, Vector<T> parameters, int offset)
    {
        foreach (var layer in layers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                parameters[offset + i] = layerParams[i];
            }
            offset += layerParams.Length;
        }
        return offset;
    }

    private int AppendSingleLayerParameters(ILayer<T>? layer, Vector<T> parameters, int offset)
    {
        if (layer is null)
        {
            return offset;
        }

        var layerParams = layer.GetParameters();
        for (int i = 0; i < layerParams.Length; i++)
        {
            parameters[offset + i] = layerParams[i];
        }

        return offset + layerParams.Length;
    }

    private int AppendMatrixParameters(Matrix<T>? matrix, Vector<T> parameters, int offset)
    {
        if (matrix is null)
        {
            return offset;
        }

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                parameters[offset++] = matrix[i, j];
            }
        }

        return offset;
    }

    private int UpdateSingleLayerParameters(ILayer<T>? layer, Vector<T> parameters, int offset)
    {
        if (layer is null)
        {
            return offset;
        }

        int layerParamCount = layer.ParameterCount;
        if (layerParamCount > 0)
        {
            var layerParams = new Vector<T>(layerParamCount);
            for (int i = 0; i < layerParamCount; i++)
            {
                layerParams[i] = parameters[offset + i];
            }
            layer.UpdateParameters(layerParams);
        }

        return offset + layerParamCount;
    }

    private int UpdateMatrixParameters(Matrix<T>? matrix, Vector<T> parameters, int offset)
    {
        if (matrix is null)
        {
            return offset;
        }

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = parameters[offset++];
            }
        }

        return offset;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = Enums.ModelType.ImageBind,
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
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embeddingDimension);
        writer.Write(_maxSequenceLength);
        writer.Write(_imageSize);
        writer.Write(_hiddenDim);
        writer.Write(_numEncoderLayers);
        writer.Write(_numHeads);
        writer.Write(_patchSize);
        writer.Write(_vocabularySize);
        writer.Write(_audioSampleRate);
        writer.Write(_audioMaxDuration);
        writer.Write(_imuTimesteps);
        writer.Write(_numVideoFrames);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // embeddingDim
        _ = reader.ReadInt32(); // maxSeqLen
        _ = reader.ReadInt32(); // imageSize
        _ = reader.ReadInt32(); // hiddenDim
        _ = reader.ReadInt32(); // numEncoderLayers
        _ = reader.ReadInt32(); // numHeads
        _ = reader.ReadInt32(); // patchSize
        _ = reader.ReadInt32(); // vocabularySize
        _ = reader.ReadInt32(); // audioSampleRate
        _ = reader.ReadInt32(); // audioMaxDuration
        _ = reader.ReadInt32(); // imuTimesteps
        _ = reader.ReadInt32(); // numVideoFrames
        _ = reader.ReadBoolean(); // useNativeMode
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ImageBindNeuralNetwork<T>(
            Architecture,
            _imageSize,
            channels: 3,
            _patchSize,
            _vocabularySize,
            _maxSequenceLength,
            _embeddingDimension,
            _hiddenDim,
            _numEncoderLayers,
            _numHeads,
            _audioSampleRate,
            _audioMaxDuration,
            _imuTimesteps,
            _numVideoFrames);
    }

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






