using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Flamingo neural network for in-context visual learning and few-shot tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Flamingo is a visual language model that excels at few-shot learning. It uses a Perceiver
/// Resampler to compress visual features and gated cross-attention layers to integrate
/// visual information into a frozen language model.
/// </para>
/// </remarks>
public class FlamingoNeuralNetwork<T> : NeuralNetworkBase<T>, IFlamingoModel<T>
{
    #region Execution Mode

    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    private readonly InferenceSession? _visionEncoder;
    private readonly InferenceSession? _languageModel;
    private readonly string? _visionEncoderPath;
    private readonly string? _languageModelPath;

    #endregion

    #region Native Mode Fields

    private readonly List<ILayer<T>> _visionEncoderLayers = [];
    private readonly List<ILayer<T>> _perceiverLayers = [];
    private readonly List<ILayer<T>> _gatedCrossAttentionLayers = [];
    private readonly List<ILayer<T>> _languageModelLayers = [];
    private Matrix<T>? _perceiverQueries;
    private Matrix<T>? _visionPositionalEmbeddings;
    private ILayer<T>? _patchEmbedding;
    private ILayer<T>? _textTokenEmbedding;
    private Matrix<T>? _textPositionalEmbeddings;
    private ILayer<T>? _outputProjection;

    #endregion

    #region Shared Fields

    private readonly ITokenizer _tokenizer;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _embeddingDimension;
    private readonly int _maxSequenceLength;
    private readonly int _imageSize;
    private readonly int _visionHiddenDim;
    private readonly int _lmHiddenDim;
    private readonly int _numVisionLayers;
    private readonly int _numLmLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _vocabularySize;
    private readonly string _languageModelType;
    private readonly int _numPerceiverTokens;
    private readonly int _maxImagesInContext;
    private readonly int _numPerceiverLayers;

    #endregion

    #region IMultimodalEmbedding Properties

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc/>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    public int ImageSize => _imageSize;

    #endregion

    #region IFlamingoModel Properties

    /// <inheritdoc/>
    public int NumPerceiverTokens => _numPerceiverTokens;

    /// <inheritdoc/>
    public int MaxImagesInContext => _maxImagesInContext;

    /// <inheritdoc/>
    public string LanguageModelType => _languageModelType;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance using ONNX models.
    /// </summary>
    public FlamingoNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string visionEncoderPath,
        string languageModelPath,
        ITokenizer tokenizer,
        int embeddingDimension = 768,
        int maxSequenceLength = 2048,
        int imageSize = 224,
        int numPerceiverTokens = 64,
        int maxImagesInContext = 5,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(visionEncoderPath))
            throw new ArgumentException("Vision encoder path cannot be null or empty.", nameof(visionEncoderPath));
        if (string.IsNullOrWhiteSpace(languageModelPath))
            throw new ArgumentException("Language model path cannot be null or empty.", nameof(languageModelPath));
        if (!File.Exists(visionEncoderPath))
            throw new FileNotFoundException($"Vision encoder model not found: {visionEncoderPath}");
        if (!File.Exists(languageModelPath))
            throw new FileNotFoundException($"Language model not found: {languageModelPath}");

        _useNativeMode = false;
        _visionEncoderPath = visionEncoderPath;
        _languageModelPath = languageModelPath;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _numPerceiverTokens = numPerceiverTokens;
        _maxImagesInContext = maxImagesInContext;
        _visionHiddenDim = 1024;
        _lmHiddenDim = 2048;
        _numVisionLayers = 24;
        _numLmLayers = 32;
        _numHeads = 16;
        _patchSize = 14;
        _vocabularySize = 32000;
        _languageModelType = "chinchilla";
        _numPerceiverLayers = 6;

        InferenceSession? visionEncoder = null;
        InferenceSession? languageModel = null;

        try
        {
            visionEncoder = new InferenceSession(visionEncoderPath);
            languageModel = new InferenceSession(languageModelPath);
            _visionEncoder = visionEncoder;
            _languageModel = languageModel;
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
            InitializeLayers();
        }
        catch
        {
            visionEncoder?.Dispose();
            languageModel?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Initializes a new instance using native layers.
    /// </summary>
    public FlamingoNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int embeddingDimension = 768,
        int maxSequenceLength = 2048,
        int imageSize = 224,
        int channels = 3,
        int numPerceiverTokens = 64,
        int maxImagesInContext = 5,
        int visionHiddenDim = 1024,
        int lmHiddenDim = 2048,
        int numVisionLayers = 24,
        int numLmLayers = 32,
        int numHeads = 16,
        int vocabularySize = 32000,
        string languageModelType = "chinchilla",
        int numPerceiverLayers = 6,
        ITokenizer? tokenizer = null,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _numPerceiverTokens = numPerceiverTokens;
        _maxImagesInContext = maxImagesInContext;
        _visionHiddenDim = visionHiddenDim;
        _lmHiddenDim = lmHiddenDim;
        _numVisionLayers = numVisionLayers;
        _numLmLayers = numLmLayers;
        _numHeads = numHeads;
        _patchSize = 14;
        _vocabularySize = vocabularySize;
        _languageModelType = languageModelType;
        _numPerceiverLayers = numPerceiverLayers;

        _tokenizer = tokenizer ?? ClipTokenizerFactory.CreateSimple();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();

        InitializeNativeLayers(channels);
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // ONNX mode initialization
    }

    private void InitializeNativeLayers(int channels)
    {
        int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);

        // Initialize patch embedding for vision encoder
        _patchEmbedding = new PatchEmbeddingLayer<T>(
            _imageSize, _imageSize, channels, _patchSize, _visionHiddenDim);

        // Initialize vision positional embeddings
        _visionPositionalEmbeddings = Matrix<T>.CreateDefault(numPatches + 1, _visionHiddenDim, NumOps.Zero);
        InitializePositionalEmbeddings(_visionPositionalEmbeddings);

        // Vision encoder transformer layers
        int visionFfnDim = _visionHiddenDim * 4;
        for (int i = 0; i < _numVisionLayers; i++)
        {
            _visionEncoderLayers.Add(new TransformerEncoderLayer<T>(
                _visionHiddenDim, _numHeads, visionFfnDim));
        }

        // Initialize perceiver queries
        _perceiverQueries = Matrix<T>.CreateDefault(_numPerceiverTokens, _lmHiddenDim, NumOps.Zero);
        InitializePerceiverQueries(_perceiverQueries);

        // Perceiver Resampler layers
        for (int i = 0; i < _numPerceiverLayers; i++)
        {
            _perceiverLayers.Add(new CrossAttentionLayer<T>(_lmHiddenDim, _visionHiddenDim, _numHeads));
            _perceiverLayers.Add(new DenseLayer<T>(
                _lmHiddenDim, _lmHiddenDim * 4,
                (IActivationFunction<T>)new GELUActivation<T>()));
            _perceiverLayers.Add(new DenseLayer<T>(
                _lmHiddenDim * 4, _lmHiddenDim,
                (IActivationFunction<T>?)null));
        }

        // Gated cross-attention layers
        int gatedCrossAttnCount = _numLmLayers / 4;
        for (int i = 0; i < gatedCrossAttnCount; i++)
        {
            _gatedCrossAttentionLayers.Add(new CrossAttentionLayer<T>(_lmHiddenDim, _lmHiddenDim, _numHeads));
        }

        // Text token embedding
        _textTokenEmbedding = new EmbeddingLayer<T>(_vocabularySize, _lmHiddenDim);

        // Text positional embeddings
        _textPositionalEmbeddings = Matrix<T>.CreateDefault(_maxSequenceLength, _lmHiddenDim, NumOps.Zero);
        InitializePositionalEmbeddings(_textPositionalEmbeddings);

        // Language model transformer layers
        int lmFfnDim = _lmHiddenDim * 4;
        for (int i = 0; i < _numLmLayers; i++)
        {
            _languageModelLayers.Add(new TransformerEncoderLayer<T>(
                _lmHiddenDim, _numHeads, lmFfnDim));
        }

        // Output projection to vocabulary
        _outputProjection = new DenseLayer<T>(_lmHiddenDim, _vocabularySize, (IActivationFunction<T>?)null);
    }

    private void InitializePositionalEmbeddings(Matrix<T> embeddings)
    {
        for (int i = 0; i < embeddings.Rows; i++)
        {
            for (int j = 0; j < embeddings.Columns; j++)
            {
                double angle = i / Math.Pow(10000, 2.0 * (j / 2) / embeddings.Columns);
                double value = j % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle);
                embeddings[i, j] = NumOps.FromDouble(value);
            }
        }
    }

    private void InitializePerceiverQueries(Matrix<T> queries)
    {
        var rand = RandomHelper.CreateSeededRandom(42);
        double scale = 1.0 / Math.Sqrt(queries.Columns);
        for (int i = 0; i < queries.Rows; i++)
        {
            for (int j = 0; j < queries.Columns; j++)
            {
                double value = (rand.NextDouble() * 2 - 1) * scale;
                queries[i, j] = NumOps.FromDouble(value);
            }
        }
    }

    #endregion

    #region IMultimodalEmbedding Implementation

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
            var features = ExtractPerceiverFeatures(image);
            var embedding = MeanPool(features);
            var normalized = Normalize(embedding);
            results.Add(normalized);
        }
        return results;
    }

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
            var encoded = _tokenizer.Encode(text);
            var inputIds = encoded.TokenIds;

            var paddedIds = new List<int>();
            for (int i = 0; i < _maxSequenceLength; i++)
            {
                paddedIds.Add(i < inputIds.Count ? inputIds[i] : 0);
            }

            var embedded = EmbedTextTokens(paddedIds);
            var embedding = MeanPool(embedded);
            var normalized = Normalize(embedding);
            results.Add(normalized);
        }

        return results;
    }

    private Tensor<T> EmbedTextTokens(IReadOnlyList<int> tokenIds)
    {
        int seqLen = tokenIds.Count;
        var embeddings = Tensor<T>.CreateDefault([seqLen, _lmHiddenDim], NumOps.Zero);

        if (_textTokenEmbedding is null || _textPositionalEmbeddings is null)
        {
            return embeddings;
        }

        for (int i = 0; i < seqLen; i++)
        {
            var tokenInput = Tensor<T>.CreateDefault([1], NumOps.FromDouble(tokenIds[i]));
            var tokenEmb = _textTokenEmbedding.Forward(tokenInput);

            for (int j = 0; j < _lmHiddenDim; j++)
            {
                T posEmb = i < _textPositionalEmbeddings.Rows ? _textPositionalEmbeddings[i, j] : NumOps.Zero;
                embeddings[i, j] = NumOps.Add(tokenEmb[0, j], posEmb);
            }
        }

        return embeddings;
    }

    private Vector<T> MeanPool(Tensor<T> features)
    {
        int seqLen = features.Shape[0];
        int hiddenDim = features.Shape[1];
        var result = new Vector<T>(hiddenDim);

        for (int j = 0; j < hiddenDim; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < seqLen; i++)
            {
                sum = NumOps.Add(sum, features[i, j]);
            }
            result[j] = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
        }

        return result;
    }

    private Vector<T> Normalize(Vector<T> vec)
    {
        T norm = NumOps.Zero;
        for (int i = 0; i < vec.Length; i++)
        {
            norm = NumOps.Add(norm, NumOps.Multiply(vec[i], vec[i]));
        }
        norm = NumOps.Sqrt(norm);

        if (NumOps.ToDouble(norm) < 1e-10)
            return vec;

        var result = new Vector<T>(vec.Length);
        for (int i = 0; i < vec.Length; i++)
        {
            result[i] = NumOps.Divide(vec[i], norm);
        }
        return result;
    }

    /// <inheritdoc/>
    public T ComputeSimilarity(Vector<T> embedding1, Vector<T> embedding2)
    {
        T dotProduct = NumOps.Zero;
        T norm1 = NumOps.Zero;
        T norm2 = NumOps.Zero;

        for (int i = 0; i < embedding1.Length && i < embedding2.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(embedding1[i], embedding2[i]));
            norm1 = NumOps.Add(norm1, NumOps.Multiply(embedding1[i], embedding1[i]));
            norm2 = NumOps.Add(norm2, NumOps.Multiply(embedding2[i], embedding2[i]));
        }

        T normProduct = NumOps.Multiply(NumOps.Sqrt(norm1), NumOps.Sqrt(norm2));
        if (NumOps.ToDouble(normProduct) < 1e-10)
            return NumOps.Zero;

        return NumOps.Divide(dotProduct, normProduct);
    }

    /// <inheritdoc/>
    public T ComputeImageTextSimilarity(Tensor<T> image, string text)
    {
        var imageEmbedding = GetImageEmbedding(image);
        var textEmbedding = GetTextEmbedding(text);
        return ComputeSimilarity(imageEmbedding, textEmbedding);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, IEnumerable<string> classLabels)
    {
        var imageEmbedding = GetImageEmbedding(image);
        var result = new Dictionary<string, T>();
        var scores = new List<(string label, T score)>();

        foreach (var label in classLabels)
        {
            var textEmbedding = GetTextEmbedding(label);
            var similarity = ComputeSimilarity(imageEmbedding, textEmbedding);
            scores.Add((label, similarity));
        }

        var expScores = scores.Select(s => (s.label, exp: Math.Exp(NumOps.ToDouble(s.score)))).ToList();
        double sumExp = expScores.Sum(s => s.exp);

        foreach (var (label, exp) in expScores)
        {
            result[label] = NumOps.FromDouble(exp / sumExp);
        }

        return result;
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveImages(
        string query,
        IEnumerable<Vector<T>> imageEmbeddings,
        int topK = 10)
    {
        var queryEmbedding = GetTextEmbedding(query);
        var embeddingsList = imageEmbeddings.ToList();
        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < embeddingsList.Count; i++)
        {
            var similarity = ComputeSimilarity(queryEmbedding, embeddingsList[i]);
            scores.Add((i, similarity));
        }

        return scores.OrderByDescending(s => NumOps.ToDouble(s.Score)).Take(topK);
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveTexts(
        Tensor<T> image,
        IEnumerable<string> texts,
        int topK = 10)
    {
        var imageEmbedding = GetImageEmbedding(image);
        var textsList = texts.ToList();
        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < textsList.Count; i++)
        {
            var textEmbedding = GetTextEmbedding(textsList[i]);
            var similarity = ComputeSimilarity(imageEmbedding, textEmbedding);
            scores.Add((i, similarity));
        }

        return scores.OrderByDescending(s => NumOps.ToDouble(s.Score)).Take(topK);
    }

    /// <inheritdoc/>
    public string GenerateCaption(Tensor<T> image, int maxLength = 77)
    {
        return FewShotGenerate([], image, "Describe this image:", maxLength);
    }

    /// <inheritdoc/>
    public string AnswerQuestion(Tensor<T> image, string question, int maxLength = 64)
    {
        return FewShotVQA([], image, question);
    }

    #endregion

    #region IFlamingoModel Implementation

    /// <inheritdoc/>
    public string FewShotGenerate(
        IEnumerable<(Tensor<T> Image, string Text)> examples,
        Tensor<T> queryImage,
        string? queryPrompt = null,
        int maxLength = 256)
    {
        var examplesList = examples.ToList();
        if (examplesList.Count > _maxImagesInContext)
        {
            throw new ArgumentException(
                $"Too many examples. Maximum allowed: {_maxImagesInContext}",
                nameof(examples));
        }

        var allImageFeatures = new List<Tensor<T>>();
        foreach (var (image, _) in examplesList)
        {
            allImageFeatures.Add(ExtractPerceiverFeatures(image));
        }
        allImageFeatures.Add(ExtractPerceiverFeatures(queryImage));

        var contextBuilder = new List<string>();
        for (int i = 0; i < examplesList.Count; i++)
        {
            contextBuilder.Add($"<image>{examplesList[i].Text}");
        }
        contextBuilder.Add($"<image>{queryPrompt ?? ""}");

        string context = string.Join(" ", contextBuilder);
        var encoded = _tokenizer.Encode(context);
        var inputIds = encoded.TokenIds;

        return GenerateWithVisualContext(inputIds, allImageFeatures, maxLength);
    }

    /// <inheritdoc/>
    public string GenerateWithMultipleImages(
        IEnumerable<Tensor<T>> images,
        string prompt,
        int maxLength = 512)
    {
        var imagesList = images.ToList();
        if (imagesList.Count > _maxImagesInContext)
        {
            throw new ArgumentException(
                $"Too many images. Maximum allowed: {_maxImagesInContext}",
                nameof(images));
        }

        var allImageFeatures = imagesList.Select(img => ExtractPerceiverFeatures(img)).ToList();
        var encoded = _tokenizer.Encode(prompt);
        var inputIds = encoded.TokenIds;

        return GenerateWithVisualContext(inputIds, allImageFeatures, maxLength);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> InContextClassify(
        IEnumerable<(Tensor<T> Image, string Label)> labeledExamples,
        Tensor<T> queryImage)
    {
        var examplesList = labeledExamples.ToList();
        var labels = examplesList.Select(e => e.Label).Distinct().ToList();

        var examples = examplesList.Select(e => (e.Image, $"This is: {e.Label}")).ToList();
        var generated = FewShotGenerate(examples, queryImage, "This is:", maxLength: 50);

        var result = new Dictionary<string, T>();
        double totalScore = 0;

        foreach (var label in labels)
        {
            double score = generated.ToLowerInvariant().Contains(label.ToLowerInvariant()) ? 1.0 : 0.1;
            totalScore += score;
            result[label] = NumOps.FromDouble(score);
        }

        foreach (var label in labels)
        {
            result[label] = NumOps.Divide(result[label], NumOps.FromDouble(totalScore));
        }

        return result;
    }

    /// <inheritdoc/>
    public string FewShotVQA(
        IEnumerable<(Tensor<T> Image, string Question, string Answer)> examples,
        Tensor<T> queryImage,
        string question)
    {
        var examplesList = examples.ToList();
        var fewShotExamples = examplesList.Select(e =>
            (e.Image, $"Question: {e.Question}\nAnswer: {e.Answer}")).ToList();

        return FewShotGenerate(
            fewShotExamples,
            queryImage,
            $"Question: {question}\nAnswer:",
            maxLength: 128);
    }

    /// <inheritdoc/>
    public Tensor<T> ExtractPerceiverFeatures(Tensor<T> image)
    {
        var visionFeatures = ExtractVisionFeatures(image);

        if (_useNativeMode)
        {
            return ExtractPerceiverFeaturesNative(visionFeatures);
        }
        else
        {
            return ExtractPerceiverFeaturesOnnx(visionFeatures);
        }
    }

    private Tensor<T> ExtractVisionFeatures(Tensor<T> image)
    {
        if (_useNativeMode)
        {
            return ExtractVisionFeaturesNative(image);
        }
        else
        {
            return ExtractVisionFeaturesOnnx(image);
        }
    }

    private Tensor<T> ExtractVisionFeaturesNative(Tensor<T> image)
    {
        if (_patchEmbedding is null || _visionPositionalEmbeddings is null)
        {
            throw new InvalidOperationException("Vision layers not initialized.");
        }

        var patchFeatures = _patchEmbedding.Forward(image);
        int numPatches = patchFeatures.Shape[0];
        int hiddenDim = patchFeatures.Shape.Length > 1 ? patchFeatures.Shape[1] : _visionHiddenDim;

        var features = Tensor<T>.CreateDefault([numPatches, hiddenDim], NumOps.Zero);
        for (int i = 0; i < numPatches && i < _visionPositionalEmbeddings.Rows; i++)
        {
            for (int j = 0; j < hiddenDim && j < _visionPositionalEmbeddings.Columns; j++)
            {
                features[i, j] = NumOps.Add(patchFeatures[i, j], _visionPositionalEmbeddings[i, j]);
            }
        }

        var current = features;
        foreach (var layer in _visionEncoderLayers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    private Tensor<T> ExtractVisionFeaturesOnnx(Tensor<T> image)
    {
        if (_visionEncoder is null)
        {
            throw new InvalidOperationException("Vision encoder not initialized.");
        }

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

        using var results = _visionEncoder.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);
        var features = Tensor<T>.CreateDefault([numPatches, _visionHiddenDim], NumOps.Zero);

        for (int i = 0; i < numPatches && i * _visionHiddenDim < output.Length; i++)
        {
            for (int j = 0; j < _visionHiddenDim; j++)
            {
                int outputIdx = i * _visionHiddenDim + j;
                if (outputIdx < output.Length)
                {
                    features[i, j] = NumOps.FromDouble(output[outputIdx]);
                }
            }
        }

        return features;
    }

    private Tensor<T> ExtractPerceiverFeaturesNative(Tensor<T> visionFeatures)
    {
        if (_perceiverQueries is null)
        {
            throw new InvalidOperationException("Perceiver queries not initialized.");
        }

        var current = Tensor<T>.CreateDefault([_numPerceiverTokens, _lmHiddenDim], NumOps.Zero);
        for (int i = 0; i < _numPerceiverTokens; i++)
        {
            for (int j = 0; j < _lmHiddenDim; j++)
            {
                current[i, j] = _perceiverQueries[i, j];
            }
        }

        for (int i = 0; i < _perceiverLayers.Count; i += 3)
        {
            if (_perceiverLayers[i] is CrossAttentionLayer<T> crossAttn)
            {
                var attnOut = crossAttn.Forward(current, visionFeatures);
                current = AddTensors(current, attnOut);
            }

            if (i + 1 < _perceiverLayers.Count && i + 2 < _perceiverLayers.Count)
            {
                var ffn1 = _perceiverLayers[i + 1].Forward(current);
                var ffn2 = _perceiverLayers[i + 2].Forward(ffn1);
                current = AddTensors(current, ffn2);
            }
        }

        return current;
    }

    private Tensor<T> ExtractPerceiverFeaturesOnnx(Tensor<T> visionFeatures)
    {
        return ExtractPerceiverFeaturesNative(visionFeatures);
    }

    /// <inheritdoc/>
    public string DescribeVideo(
        IEnumerable<Tensor<T>> frames,
        string? prompt = null,
        int maxLength = 256)
    {
        var framesList = frames.ToList();
        var sampledFrames = new List<Tensor<T>>();
        int step = Math.Max(1, framesList.Count / _maxImagesInContext);
        for (int i = 0; i < framesList.Count && sampledFrames.Count < _maxImagesInContext; i += step)
        {
            sampledFrames.Add(framesList[i]);
        }

        string videoPrompt = prompt ?? "Describe what is happening in this video:";
        return GenerateWithMultipleImages(sampledFrames, videoPrompt, maxLength);
    }

    /// <inheritdoc/>
    public T ScoreImageText(Tensor<T> image, string text)
    {
        var imageEmbedding = GetImageEmbedding(image);
        var textEmbedding = GetTextEmbedding(text);
        var similarity = ComputeSimilarity(imageEmbedding, textEmbedding);

        double logProb = Math.Log(Math.Max((NumOps.ToDouble(similarity) + 1) / 2, 1e-10));
        var encoded = _tokenizer.Encode(text);
        return NumOps.FromDouble(logProb * encoded.TokenIds.Count);
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> FewShotImageRetrieval(
        IEnumerable<Tensor<T>> queryExamples,
        string? queryDescription,
        IEnumerable<Tensor<T>> candidateImages,
        int topK = 10)
    {
        var queryList = queryExamples.ToList();
        var candidateList = candidateImages.ToList();

        var queryEmbeddings = queryList.Select(img => GetImageEmbedding(img)).ToList();
        var avgQueryEmbedding = new Vector<T>(_embeddingDimension);

        for (int j = 0; j < _embeddingDimension; j++)
        {
            T sum = NumOps.Zero;
            foreach (var emb in queryEmbeddings)
            {
                if (j < emb.Length)
                    sum = NumOps.Add(sum, emb[j]);
            }
            avgQueryEmbedding[j] = NumOps.Divide(sum, NumOps.FromDouble(queryEmbeddings.Count));
        }

        if (queryDescription is not null && queryDescription.Length > 0)
        {
            var textEmbedding = GetTextEmbedding(queryDescription);
            for (int j = 0; j < _embeddingDimension && j < textEmbedding.Length; j++)
            {
                avgQueryEmbedding[j] = NumOps.Divide(
                    NumOps.Add(avgQueryEmbedding[j], textEmbedding[j]),
                    NumOps.FromDouble(2.0));
            }
        }

        avgQueryEmbedding = Normalize(avgQueryEmbedding);

        var scores = new List<(int Index, T Score)>();
        for (int i = 0; i < candidateList.Count; i++)
        {
            var candidateEmbedding = GetImageEmbedding(candidateList[i]);
            var similarity = ComputeSimilarity(avgQueryEmbedding, candidateEmbedding);
            scores.Add((i, similarity));
        }

        return scores.OrderByDescending(s => NumOps.ToDouble(s.Score)).Take(topK);
    }

    #endregion

    #region Helper Methods

    private string GenerateWithVisualContext(
        IReadOnlyList<int> inputIds,
        List<Tensor<T>> imageFeatures,
        int maxLength)
    {
        var generatedIds = new List<int>(inputIds);
        var specialTokens = _tokenizer.SpecialTokens;
        var eosTokenStr = specialTokens?.EosToken ?? "[SEP]";
        var eosEncoded = _tokenizer.Encode(eosTokenStr);
        int eosTokenId = eosEncoded.TokenIds.Count > 0 ? eosEncoded.TokenIds[0] : 0;

        var combinedImageFeatures = CombineImageFeatures(imageFeatures);

        for (int step = 0; step < maxLength; step++)
        {
            int seqLen = Math.Min(generatedIds.Count, _maxSequenceLength);
            var embeddings = EmbedTextTokens(generatedIds.Take(seqLen).ToList());

            var current = embeddings;
            int gatedAttnIdx = 0;

            for (int layer = 0; layer < _numLmLayers && layer < _languageModelLayers.Count; layer++)
            {
                if (layer % 4 == 0 && gatedAttnIdx < _gatedCrossAttentionLayers.Count)
                {
                    if (_gatedCrossAttentionLayers[gatedAttnIdx] is CrossAttentionLayer<T> crossAttn)
                    {
                        var attnOut = crossAttn.Forward(current, combinedImageFeatures);
                        current = AddTensors(current, attnOut);
                    }
                    gatedAttnIdx++;
                }

                current = _languageModelLayers[layer].Forward(current);
            }

            if (_outputProjection is null)
            {
                break;
            }

            var lastPosition = Tensor<T>.CreateDefault([1, _lmHiddenDim], NumOps.Zero);
            for (int j = 0; j < _lmHiddenDim; j++)
            {
                lastPosition[0, j] = current[seqLen - 1, j];
            }

            var logits = _outputProjection.Forward(lastPosition);
            int nextToken = SampleFromLogits(logits);
            generatedIds.Add(nextToken);

            if (nextToken == eosTokenId)
                break;
        }

        return _tokenizer.Decode(generatedIds.Skip(inputIds.Count).ToList());
    }

    private Tensor<T> CombineImageFeatures(List<Tensor<T>> imageFeatures)
    {
        if (imageFeatures.Count == 0)
        {
            return Tensor<T>.CreateDefault([1, _lmHiddenDim], NumOps.Zero);
        }

        int totalTokens = imageFeatures.Sum(f => f.Shape[0]);
        int hiddenDim = imageFeatures[0].Shape[1];

        var combined = Tensor<T>.CreateDefault([totalTokens, hiddenDim], NumOps.Zero);
        int offset = 0;

        foreach (var features in imageFeatures)
        {
            for (int i = 0; i < features.Shape[0]; i++)
            {
                for (int j = 0; j < hiddenDim; j++)
                {
                    combined[offset + i, j] = features[i, j];
                }
            }
            offset += features.Shape[0];
        }

        return combined;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = Tensor<T>.CreateDefault(a.Shape, NumOps.Zero);
        for (int i = 0; i < a.Shape[0]; i++)
        {
            for (int j = 0; j < a.Shape[1]; j++)
            {
                result[i, j] = NumOps.Add(a[i, j], b[i, j]);
            }
        }
        return result;
    }

    private int SampleFromLogits(Tensor<T> logits)
    {
        int vocabSize = logits.Shape[1];
        int maxIdx = 0;
        T maxVal = logits[0, 0];

        for (int i = 1; i < vocabSize; i++)
        {
            if (NumOps.GreaterThan(logits[0, i], maxVal))
            {
                maxVal = logits[0, i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override int ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var layer in _visionEncoderLayers)
                count += layer.ParameterCount;
            foreach (var layer in _perceiverLayers)
                count += layer.ParameterCount;
            foreach (var layer in _gatedCrossAttentionLayers)
                count += layer.ParameterCount;
            foreach (var layer in _languageModelLayers)
                count += layer.ParameterCount;
            if (_patchEmbedding is not null)
                count += _patchEmbedding.ParameterCount;
            if (_textTokenEmbedding is not null)
                count += _textTokenEmbedding.ParameterCount;
            if (_outputProjection is not null)
                count += _outputProjection.ParameterCount;
            return count;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        foreach (var layer in _perceiverLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                allParams.Add(layerParams[i]);
            }
        }

        foreach (var layer in _gatedCrossAttentionLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                allParams.Add(layerParams[i]);
            }
        }

        return new Vector<T>([.. allParams]);
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        var features = ExtractPerceiverFeatures(input);
        return features;
    }

    /// <summary>
    /// Backward pass through perceiver and gated cross-attention layers.
    /// </summary>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Backward pass is only supported in native mode.");
        }

        var currentGradient = gradient;

        for (int i = _gatedCrossAttentionLayers.Count - 1; i >= 0; i--)
        {
            currentGradient = _gatedCrossAttentionLayers[i].Backward(currentGradient);
        }

        for (int i = _perceiverLayers.Count - 1; i >= 0; i--)
        {
            currentGradient = _perceiverLayers[i].Backward(currentGradient);
        }

        return currentGradient;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);

        var perceiverFeatures = ExtractPerceiverFeatures(input);

        LastLoss = LossFunction.CalculateLoss(perceiverFeatures.ToVector(), expectedOutput.ToVector());
        var lossGradient = LossFunction.CalculateDerivative(perceiverFeatures.ToVector(), expectedOutput.ToVector());
        var gradient = Tensor<T>.FromVector(lossGradient);

        Backward(gradient);
        var currentParams = GetParameters();
        UpdateParameters(currentParams);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;

        foreach (var layer in _perceiverLayers)
        {
            int layerParamCount = layer.ParameterCount;
            if (layerParamCount > 0)
            {
                var layerParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount && offset + i < parameters.Length; i++)
                {
                    layerParams[i] = parameters[offset + i];
                }
                layer.UpdateParameters(layerParams);
                offset += layerParamCount;
            }
        }

        foreach (var layer in _gatedCrossAttentionLayers)
        {
            int layerParamCount = layer.ParameterCount;
            if (layerParamCount > 0)
            {
                var layerParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount && offset + i < parameters.Length; i++)
                {
                    layerParams[i] = parameters[offset + i];
                }
                layer.UpdateParameters(layerParams);
                offset += layerParamCount;
            }
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = Enums.ModelType.Flamingo,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ImageSize", _imageSize },
                { "EmbeddingDimension", _embeddingDimension },
                { "MaxSequenceLength", _maxSequenceLength },
                { "NumPerceiverTokens", _numPerceiverTokens },
                { "MaxImagesInContext", _maxImagesInContext },
                { "VisionHiddenDim", _visionHiddenDim },
                { "LmHiddenDim", _lmHiddenDim },
                { "NumVisionLayers", _numVisionLayers },
                { "NumLmLayers", _numLmLayers },
                { "NumPerceiverLayers", _numPerceiverLayers },
                { "VocabularySize", _vocabularySize },
                { "LanguageModelType", _languageModelType },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", ParameterCount },
                { "TaskType", Architecture.TaskType.ToString() }
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
        writer.Write(_numPerceiverTokens);
        writer.Write(_maxImagesInContext);
        writer.Write(_visionHiddenDim);
        writer.Write(_lmHiddenDim);
        writer.Write(_numVisionLayers);
        writer.Write(_numLmLayers);
        writer.Write(_numHeads);
        writer.Write(_patchSize);
        writer.Write(_vocabularySize);
        writer.Write(_languageModelType);
        writer.Write(_numPerceiverLayers);
        writer.Write(_useNativeMode);
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
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadString();
        _ = reader.ReadInt32();
        _ = reader.ReadBoolean();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Create a fresh optimizer instance to avoid state sharing between models
        var freshOptimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        if (_useNativeMode)
        {
            return new FlamingoNeuralNetwork<T>(
                Architecture,
                _embeddingDimension,
                _maxSequenceLength,
                _imageSize,
                3,
                _numPerceiverTokens,
                _maxImagesInContext,
                _visionHiddenDim,
                _lmHiddenDim,
                _numVisionLayers,
                _numLmLayers,
                _numHeads,
                _vocabularySize,
                _languageModelType,
                _numPerceiverLayers,
                _tokenizer,
                freshOptimizer,
                _lossFunction);
        }
        else
        {
            // ONNX mode - use the stored paths
            string visionPath = _visionEncoderPath ?? string.Empty;
            string languagePath = _languageModelPath ?? string.Empty;

            if (visionPath.Length == 0 || languagePath.Length == 0)
            {
                throw new InvalidOperationException("Cannot clone ONNX mode instance without valid model paths.");
            }

            return new FlamingoNeuralNetwork<T>(
                Architecture,
                visionPath,
                languagePath,
                _tokenizer,
                _embeddingDimension,
                _maxSequenceLength,
                _imageSize,
                _numPerceiverTokens,
                _maxImagesInContext,
                freshOptimizer,
                _lossFunction);
        }
    }

    #endregion
}
