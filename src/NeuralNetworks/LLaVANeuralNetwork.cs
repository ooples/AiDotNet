using System.IO;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// LLaVA (Large Language and Vision Assistant) neural network for visual instruction following.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LLaVA connects a vision encoder (CLIP ViT) with a large language model (LLaMA/Vicuna)
/// through a simple projection layer, enabling visual conversations and instruction following.
/// </para>
/// <para><b>For Beginners:</b> LLaVA is like giving eyes to ChatGPT!
///
/// Architecture overview:
/// 1. Vision Encoder (CLIP ViT-L/14): Extracts image patch features
/// 2. Projection Layer (MLP): Maps visual features to LLM's embedding space
/// 3. Large Language Model (LLaMA/Vicuna): Generates text responses
///
/// Key capabilities:
/// - Visual conversations: "What's in this image?" followed by "What color is the car?"
/// - Visual reasoning: Understanding relationships, counting, spatial awareness
/// - Instruction following: "Describe this image as if you were a poet"
/// - Multi-turn dialogue: Context-aware conversations about images
/// </para>
/// </remarks>
public class LLaVANeuralNetwork<T> : NeuralNetworkBase<T>, ILLaVAModel<T>
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
    private readonly List<ILayer<T>> _projectionLayers = [];
    private readonly List<ILayer<T>> _languageModelLayers = [];
    private Matrix<T>? _visionClsToken;
    private Matrix<T>? _visionPositionalEmbeddings;
    private ILayer<T>? _patchEmbedding;
    private ILayer<T>? _textTokenEmbedding;
    private Matrix<T>? _textPositionalEmbeddings;
    private ILayer<T>? _outputProjection;
    private ILayer<T>? _groundingHead;

    #endregion

    #region Shared Fields

    private readonly ITokenizer _tokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
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
    private readonly string _visionEncoderType;
    private readonly int _numVisualTokens;

    #endregion

    #region IMultimodalEmbedding Properties

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc/>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    public int ImageSize => _imageSize;

    #endregion

    #region ILLaVAModel Properties

    /// <inheritdoc/>
    public string LanguageModelType => _languageModelType;

    /// <inheritdoc/>
    public string VisionEncoderType => _visionEncoderType;

    /// <inheritdoc/>
    public int NumVisualTokens => _numVisualTokens;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a LLaVA network using pretrained ONNX models.
    /// </summary>
    public LLaVANeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string visionEncoderPath,
        string languageModelPath,
        ITokenizer tokenizer,
        string languageModelType = "llama",
        string visionEncoderType = "clip-vit-l",
        int embeddingDimension = 4096,
        int maxSequenceLength = 2048,
        int imageSize = 336,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
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
        _languageModelType = languageModelType.ToLowerInvariant();
        _visionEncoderType = visionEncoderType.ToLowerInvariant();
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _patchSize = 14;
        _numVisualTokens = (imageSize / _patchSize) * (imageSize / _patchSize);
        _visionHiddenDim = 1024;
        _lmHiddenDim = embeddingDimension;
        _numVisionLayers = 24;
        _numLmLayers = 32;
        _numHeads = 16;
        _vocabularySize = 32000;

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
    /// Creates a LLaVA network using native library layers.
    /// </summary>
    public LLaVANeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int imageSize = 336,
        int channels = 3,
        int patchSize = 14,
        int vocabularySize = 32000,
        int maxSequenceLength = 2048,
        int embeddingDimension = 4096,
        int visionHiddenDim = 1024,
        int numVisionLayers = 24,
        int numLmLayers = 32,
        int numHeads = 16,
        string languageModelType = "llama",
        string visionEncoderType = "clip-vit-l",
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _visionHiddenDim = visionHiddenDim;
        _lmHiddenDim = embeddingDimension;
        _numVisionLayers = numVisionLayers;
        _numLmLayers = numLmLayers;
        _numHeads = numHeads;
        _patchSize = patchSize;
        _vocabularySize = vocabularySize;
        _languageModelType = languageModelType.ToLowerInvariant();
        _visionEncoderType = visionEncoderType.ToLowerInvariant();
        _numVisualTokens = (imageSize / patchSize) * (imageSize / patchSize);

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
        // ONNX mode initialization
    }

    private void InitializeNativeLayers(int channels)
    {
        int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);

        _patchEmbedding = new PatchEmbeddingLayer<T>(
            _imageSize, _imageSize, channels, _patchSize, _visionHiddenDim);

        _visionClsToken = Matrix<T>.CreateDefault(1, _visionHiddenDim, NumOps.Zero);
        _visionPositionalEmbeddings = Matrix<T>.CreateDefault(numPatches + 1, _visionHiddenDim, NumOps.Zero);

        int visionFfnDim = _visionHiddenDim * 4;
        for (int i = 0; i < _numVisionLayers; i++)
        {
            _visionEncoderLayers.Add(new TransformerEncoderLayer<T>(
                _visionHiddenDim, _numHeads, visionFfnDim));
        }

        // Projection layers (2-layer MLP) - explicit cast to avoid ambiguity
        _projectionLayers.Add(new DenseLayer<T>(_visionHiddenDim, _lmHiddenDim,
            (IActivationFunction<T>)new GELUActivation<T>()));
        _projectionLayers.Add(new DenseLayer<T>(_lmHiddenDim, _lmHiddenDim,
            (IActivationFunction<T>?)null));

        _textTokenEmbedding = new EmbeddingLayer<T>(_vocabularySize, _lmHiddenDim);
        _textPositionalEmbeddings = Matrix<T>.CreateDefault(_maxSequenceLength, _lmHiddenDim, NumOps.Zero);

        int lmFfnDim = _lmHiddenDim * 4;
        for (int i = 0; i < _numLmLayers; i++)
        {
            _languageModelLayers.Add(new TransformerEncoderLayer<T>(
                _lmHiddenDim, _numHeads, lmFfnDim));
        }

        _outputProjection = new DenseLayer<T>(_lmHiddenDim, _vocabularySize, (IActivationFunction<T>?)null);
        _groundingHead = new DenseLayer<T>(_lmHiddenDim, 4, (IActivationFunction<T>)new SigmoidActivation<T>());

        InitializeWeights();
    }

    private void InitializeWeights()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        double scale = 0.02;

        if (_visionClsToken is not null)
        {
            for (int j = 0; j < _visionClsToken.Columns; j++)
            {
                _visionClsToken[0, j] = NumOps.FromDouble(random.NextDouble() * scale - scale / 2);
            }
        }

        if (_visionPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _visionPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < _visionPositionalEmbeddings.Columns; j++)
                {
                    _visionPositionalEmbeddings[i, j] = NumOps.FromDouble(random.NextDouble() * scale - scale / 2);
                }
            }
        }

        if (_textPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _textPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < _textPositionalEmbeddings.Columns; j++)
                {
                    _textPositionalEmbeddings[i, j] = NumOps.FromDouble(random.NextDouble() * scale - scale / 2);
                }
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
            var encoded = _tokenizer.Encode(text);
            var inputIds = encoded.TokenIds;

            // Truncate or pad
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
            var features = ExtractVisualFeatures(image);
            var embedding = MeanPool(features);
            var normalized = Normalize(embedding);
            results.Add(normalized);
        }

        return results;
    }

    /// <inheritdoc/>
    public T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding)
    {
        // Use the minimum length to avoid out-of-range access when embeddings have different dimensions
        int length = Math.Min(textEmbedding.Length, imageEmbedding.Length);

        T similarity = NumOps.Zero;
        for (int i = 0; i < length; i++)
        {
            similarity = NumOps.Add(similarity,
                NumOps.Multiply(textEmbedding[i], imageEmbedding[i]));
        }
        return similarity;
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, IEnumerable<string> classLabels)
    {
        if (classLabels is null)
        {
            throw new ArgumentNullException(nameof(classLabels));
        }

        var labels = classLabels.ToList();
        if (labels.Count == 0)
        {
            throw new ArgumentException("At least one class label must be provided.", nameof(classLabels));
        }

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

    #region ILLaVAModel Implementation

    /// <inheritdoc/>
    public string Generate(
        Tensor<T> image,
        string prompt,
        int maxLength = 512,
        double temperature = 0.7,
        double topP = 0.9)
    {
        var visualFeatures = ExtractVisualFeatures(image);
        var projectedFeatures = ProjectToLanguageSpace(visualFeatures);

        var encoded = _tokenizer.Encode(prompt);
        var promptTokens = encoded.TokenIds;
        var promptEmbeddings = EmbedTextTokens(promptTokens);

        var combinedInput = ConcatenateSequences(projectedFeatures, promptEmbeddings);

        var generatedTokens = new List<int>();
        var currentInput = combinedInput;

        for (int step = 0; step < maxLength; step++)
        {
            var output = ForwardLLM(currentInput);
            var logits = GetNextTokenLogits(output, temperature);
            int nextToken = SampleToken(logits, topP);

            var specialTokens = _tokenizer.SpecialTokens;
            var eosTokenStr = specialTokens?.EosToken ?? "[SEP]";
            var eosEncoded = _tokenizer.Encode(eosTokenStr);
            if (eosEncoded.TokenIds.Count > 0 && nextToken == eosEncoded.TokenIds[0])
                break;

            generatedTokens.Add(nextToken);
            currentInput = AppendToken(currentInput, nextToken);
        }

        return _tokenizer.Decode(generatedTokens);
    }

    /// <inheritdoc/>
    public string Chat(
        Tensor<T> image,
        IEnumerable<(string Role, string Content)> conversationHistory,
        string userMessage,
        int maxLength = 512,
        double temperature = 0.7)
    {
        var conversationPrompt = BuildConversationPrompt(conversationHistory, userMessage);
        return Generate(image, conversationPrompt, maxLength, temperature);
    }

    /// <inheritdoc/>
    public IEnumerable<(string Response, T Score)> GenerateMultiple(
        Tensor<T> image,
        string prompt,
        int numResponses = 5,
        double temperature = 0.9)
    {
        var results = new List<(string Response, T Score)>();

        // Cap maximum temperature to prevent degenerate outputs
        const double maxTemperature = 1.2;
        const double tempIncrement = 0.05;

        for (int i = 0; i < numResponses; i++)
        {
            // Calculate temperature with upper bound to prevent sampling issues
            double adjustedTemp = Math.Min(temperature + (i * tempIncrement), maxTemperature);
            var response = Generate(image, prompt, 256, adjustedTemp);
            var score = NumOps.FromDouble(Math.Min(response.Length / 100.0, 1.0));
            results.Add((response, score));
        }

        return results.OrderByDescending(r => NumOps.ToDouble(r.Score));
    }

    /// <inheritdoc/>
    public Tensor<T> ExtractVisualFeatures(Tensor<T> image)
    {
        if (_useNativeMode)
        {
            return ExtractVisualFeaturesNative(image);
        }
        else
        {
            return ExtractVisualFeaturesOnnx(image);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> ProjectToLanguageSpace(Tensor<T> visualFeatures)
    {
        if (_useNativeMode)
        {
            Tensor<T> output = visualFeatures;
            foreach (var layer in _projectionLayers)
            {
                output = layer.Forward(output);
            }
            return output;
        }
        else
        {
            return visualFeatures;
        }
    }

    /// <inheritdoc/>
    public Vector<T> GroundObject(Tensor<T> image, string description)
    {
        var visualFeatures = ExtractVisualFeatures(image);
        var projectedFeatures = ProjectToLanguageSpace(visualFeatures);

        var encoded = _tokenizer.Encode(description);
        var descEmbeddings = EmbedTextTokens(encoded.TokenIds);

        var combinedInput = ConcatenateSequences(projectedFeatures, descEmbeddings);
        var output = ForwardLLM(combinedInput);

        if (_groundingHead is not null)
        {
            int seqLen = output.Shape[0];
            int hiddenDim = output.Shape[1];

            var lastHidden = Tensor<T>.CreateDefault([1, hiddenDim], NumOps.Zero);
            for (int i = 0; i < hiddenDim; i++)
            {
                lastHidden[0, i] = output[seqLen - 1, i];
            }

            var bbox = _groundingHead.Forward(lastHidden);

            var bboxVector = new Vector<T>(4);
            bboxVector[0] = bbox[0, 0];
            bboxVector[1] = bbox[0, 1];
            bboxVector[2] = bbox[0, 2];
            bboxVector[3] = bbox[0, 3];
            return bboxVector;
        }

        return new Vector<T>([NumOps.Zero, NumOps.Zero, NumOps.One, NumOps.One]);
    }

    /// <inheritdoc/>
    public IEnumerable<string> DescribeRegions(Tensor<T> image, IEnumerable<Vector<T>> regions)
    {
        var descriptions = new List<string>();

        foreach (var region in regions)
        {
            var croppedImage = CropImageToRegion(image, region);
            var description = Generate(croppedImage, "Describe what is in this region of the image:", 128, 0.7);
            descriptions.Add(description);
        }

        return descriptions;
    }

    /// <inheritdoc/>
    public string CompareImages(
        Tensor<T> image1,
        Tensor<T> image2,
        IEnumerable<string>? aspectsToCompare = null)
    {
        var prompt = "Compare these two images";
        if (aspectsToCompare is not null && aspectsToCompare.Any())
        {
            prompt += $" focusing on: {string.Join(", ", aspectsToCompare)}";
        }
        prompt += ". Describe the similarities and differences.";

        var features1 = ExtractVisualFeatures(image1);
        var features2 = ExtractVisualFeatures(image2);
        var projected1 = ProjectToLanguageSpace(features1);
        var projected2 = ProjectToLanguageSpace(features2);

        var combinedVisual = ConcatenateSequences(projected1, projected2);

        var encoded = _tokenizer.Encode(prompt);
        var promptEmbeddings = EmbedTextTokens(encoded.TokenIds);

        var combinedInput = ConcatenateSequences(combinedVisual, promptEmbeddings);

        var generatedTokens = new List<int>();
        var currentInput = combinedInput;

        for (int step = 0; step < 256; step++)
        {
            var output = ForwardLLM(currentInput);
            var logits = GetNextTokenLogits(output, 0.7);
            int nextToken = SampleToken(logits, 0.9);

            var specialTokens = _tokenizer.SpecialTokens;
            var eosTokenStr = specialTokens?.EosToken ?? "[SEP]";
            var eosEncoded = _tokenizer.Encode(eosTokenStr);
            if (eosEncoded.TokenIds.Count > 0 && nextToken == eosEncoded.TokenIds[0])
                break;

            generatedTokens.Add(nextToken);
            currentInput = AppendToken(currentInput, nextToken);
        }

        return _tokenizer.Decode(generatedTokens);
    }

    #endregion

    #region Helper Methods

    private Tensor<T> ExtractVisualFeaturesNative(Tensor<T> image)
    {
        if (_patchEmbedding is null)
            throw new InvalidOperationException("Patch embedding layer not initialized.");

        var output = _patchEmbedding.Forward(image);

        if (_visionClsToken is not null)
        {
            output = PrependClsToken(output, _visionClsToken);
        }

        if (_visionPositionalEmbeddings is not null)
        {
            output = AddPositionalEmbeddings(output, _visionPositionalEmbeddings);
        }

        foreach (var layer in _visionEncoderLayers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    private Tensor<T> ExtractVisualFeaturesOnnx(Tensor<T> image)
    {
        if (_visionEncoder is null)
            throw new InvalidOperationException("Vision encoder not initialized.");

        var onnxTensor = PrepareImageForOnnx(image);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", onnxTensor)
        };

        using var results = _visionEncoder.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var output = Tensor<T>.CreateDefault(outputShape, NumOps.Zero);
        var flatOutput = outputTensor.ToArray();
        for (int i = 0; i < flatOutput.Length; i++)
        {
            output[i] = NumOps.FromDouble(flatOutput[i]);
        }

        return output;
    }

    private OnnxTensors.DenseTensor<float> PrepareImageForOnnx(Tensor<T> image)
    {
        bool is3D = image.Shape.Length == 3;
        int channels = is3D ? image.Shape[0] : image.Shape[1];
        int height = is3D ? image.Shape[1] : image.Shape[2];
        int width = is3D ? image.Shape[2] : image.Shape[3];

        var onnxTensor = new OnnxTensors.DenseTensor<float>([1, channels, height, width]);

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    T value = is3D ? image[c, h, w] : image[0, c, h, w];
                    onnxTensor[0, c, h, w] = (float)NumOps.ToDouble(value);
                }
            }
        }

        return onnxTensor;
    }

    private Tensor<T> EmbedTextTokens(IList<int> tokenIds)
    {
        if (_textTokenEmbedding is null)
            throw new InvalidOperationException("Text embedding layer not initialized.");

        var inputTensor = Tensor<T>.CreateDefault([tokenIds.Count], NumOps.Zero);
        for (int i = 0; i < tokenIds.Count; i++)
        {
            inputTensor[i] = NumOps.FromDouble(tokenIds[i]);
        }

        var embedded = _textTokenEmbedding.Forward(inputTensor);

        if (_textPositionalEmbeddings is not null && tokenIds.Count <= _textPositionalEmbeddings.Rows)
        {
            for (int i = 0; i < tokenIds.Count && i < _textPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < _lmHiddenDim && j < _textPositionalEmbeddings.Columns; j++)
                {
                    embedded[i, j] = NumOps.Add(embedded[i, j], _textPositionalEmbeddings[i, j]);
                }
            }
        }

        return embedded;
    }

    private Tensor<T> ForwardLLM(Tensor<T> input)
    {
        if (_useNativeMode)
        {
            var output = input;
            foreach (var layer in _languageModelLayers)
            {
                output = layer.Forward(output);
            }
            return output;
        }
        else
        {
            return ForwardLLMOnnx(input);
        }
    }

    private Tensor<T> ForwardLLMOnnx(Tensor<T> input)
    {
        if (_languageModel is null)
            throw new InvalidOperationException("Language model not initialized.");

        int seqLen = input.Shape[0];
        int hiddenDim = input.Shape[1];

        var onnxTensor = new OnnxTensors.DenseTensor<float>([1, seqLen, hiddenDim]);
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                onnxTensor[0, i, j] = (float)NumOps.ToDouble(input[i, j]);
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs_embeds", onnxTensor)
        };

        using var results = _languageModel.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var output = Tensor<T>.CreateDefault([outputShape[1], outputShape[2]], NumOps.Zero);
        for (int i = 0; i < outputShape[1]; i++)
        {
            for (int j = 0; j < outputShape[2]; j++)
            {
                output[i, j] = NumOps.FromDouble(outputTensor[0, i, j]);
            }
        }

        return output;
    }

    private Tensor<T> ConcatenateSequences(Tensor<T> seq1, Tensor<T> seq2)
    {
        int seq1Len = seq1.Shape[0];
        int seq2Len = seq2.Shape[0];
        int hiddenDim = seq1.Shape[1];

        var result = Tensor<T>.CreateDefault([seq1Len + seq2Len, hiddenDim], NumOps.Zero);

        for (int i = 0; i < seq1Len; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                result[i, j] = seq1[i, j];
            }
        }

        for (int i = 0; i < seq2Len; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                result[seq1Len + i, j] = seq2[i, j];
            }
        }

        return result;
    }

    private Vector<T> GetNextTokenLogits(Tensor<T> output, double temperature)
    {
        int seqLen = output.Shape[0];
        int hiddenDim = output.Shape[1];

        var lastHidden = new Vector<T>(hiddenDim);
        for (int i = 0; i < hiddenDim; i++)
        {
            lastHidden[i] = output[seqLen - 1, i];
        }

        if (_outputProjection is not null)
        {
            var inputTensor = Tensor<T>.CreateDefault([1, hiddenDim], NumOps.Zero);
            for (int i = 0; i < hiddenDim; i++)
            {
                inputTensor[0, i] = lastHidden[i];
            }

            var logitsTensor = _outputProjection.Forward(inputTensor);

            var logits = new Vector<T>(_vocabularySize);
            for (int i = 0; i < _vocabularySize; i++)
            {
                var logit = logitsTensor[0, i];
                logits[i] = NumOps.Divide(logit, NumOps.FromDouble(temperature));
            }
            return logits;
        }

        return new Vector<T>(_vocabularySize);
    }

    private int SampleToken(Vector<T> logits, double topP)
    {
        var probabilities = Softmax(logits.ToList());

        var indexed = probabilities.Select((p, i) => (Prob: p, Index: i))
            .OrderByDescending(x => NumOps.ToDouble(x.Prob))
            .ToList();

        double cumProb = 0;
        var nucleus = new List<(T Prob, int Index)>();
        foreach (var item in indexed)
        {
            nucleus.Add(item);
            cumProb += NumOps.ToDouble(item.Prob);
            if (cumProb >= topP)
                break;
        }

        double totalProb = nucleus.Sum(x => NumOps.ToDouble(x.Prob));
        // Use thread-safe random instead of creating new Random() per call
        // (new Random() produces identical sequences when called rapidly)
        double rand = Tensors.Helpers.RandomHelper.ThreadSafeRandom.NextDouble() * totalProb;

        double runningSum = 0;
        foreach (var item in nucleus)
        {
            runningSum += NumOps.ToDouble(item.Prob);
            if (runningSum >= rand)
                return item.Index;
        }

        return nucleus.Last().Index;
    }

    private Tensor<T> AppendToken(Tensor<T> input, int tokenId)
    {
        if (_textTokenEmbedding is null)
            return input;

        var tokenTensor = Tensor<T>.CreateDefault([1], NumOps.Zero);
        tokenTensor[0] = NumOps.FromDouble(tokenId);
        var tokenEmbedding = _textTokenEmbedding.Forward(tokenTensor);

        return ConcatenateSequences(input, tokenEmbedding);
    }

    private string BuildConversationPrompt(
        IEnumerable<(string Role, string Content)> history,
        string userMessage)
    {
        var prompt = new System.Text.StringBuilder();

        foreach (var (role, content) in history)
        {
            prompt.AppendLine($"{role}: {content}");
        }

        prompt.AppendLine($"User: {userMessage}");
        prompt.Append("Assistant: ");

        return prompt.ToString();
    }

    private Tensor<T> CropImageToRegion(Tensor<T> image, Vector<T> region)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        int x1 = (int)(NumOps.ToDouble(region[0]) * width);
        int y1 = (int)(NumOps.ToDouble(region[1]) * height);
        int x2 = (int)(NumOps.ToDouble(region[2]) * width);
        int y2 = (int)(NumOps.ToDouble(region[3]) * height);

        x1 = Math.Max(0, Math.Min(x1, width - 1));
        y1 = Math.Max(0, Math.Min(y1, height - 1));
        x2 = Math.Max(x1 + 1, Math.Min(x2, width));
        y2 = Math.Max(y1 + 1, Math.Min(y2, height));

        int cropWidth = x2 - x1;
        int cropHeight = y2 - y1;

        var cropped = Tensor<T>.CreateDefault([channels, cropHeight, cropWidth], NumOps.Zero);

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < cropHeight; y++)
            {
                for (int x = 0; x < cropWidth; x++)
                {
                    cropped[c, y, x] = image[c, y1 + y, x1 + x];
                }
            }
        }

        return ResizeImage(cropped, _imageSize, _imageSize);
    }

    private Tensor<T> ResizeImage(Tensor<T> image, int targetHeight, int targetWidth)
    {
        int channels = image.Shape[0];
        int srcHeight = image.Shape[1];
        int srcWidth = image.Shape[2];

        var result = Tensor<T>.CreateDefault([channels, targetHeight, targetWidth], NumOps.Zero);

        double scaleY = (double)srcHeight / targetHeight;
        double scaleX = (double)srcWidth / targetWidth;

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    int srcY = Math.Min((int)(y * scaleY), srcHeight - 1);
                    int srcX = Math.Min((int)(x * scaleX), srcWidth - 1);
                    result[c, y, x] = image[c, srcY, srcX];
                }
            }
        }

        return result;
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
        var features = ExtractVisualFeatures(input);
        return ProjectToLanguageSpace(features);
    }

    /// <summary>
    /// Backward pass through projection layers (vision encoder is frozen).
    /// </summary>
    /// <param name="gradient">The gradient tensor from the loss function.</param>
    /// <returns>The gradient after backward propagation.</returns>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        // Backward pass through projection layers
        // Vision encoder is frozen; only projection layers are trainable
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Backward pass is only supported in native mode.");
        }

        var currentGradient = gradient;

        // Backward through projection layers in reverse order
        for (int i = _projectionLayers.Count - 1; i >= 0; i--)
        {
            currentGradient = _projectionLayers[i].Backward(currentGradient);
        }

        return currentGradient;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            var visualFeatures = ExtractVisualFeatures(input);
            var projectedFeatures = ProjectToLanguageSpace(visualFeatures);

            LastLoss = LossFunction.CalculateLoss(projectedFeatures.ToVector(), expectedOutput.ToVector());
            var lossGradient = LossFunction.CalculateDerivative(projectedFeatures.ToVector(), expectedOutput.ToVector());
            var gradient = Tensor<T>.FromVector(lossGradient);

            // Propagate gradients through the network
            Backward(gradient);

            // Use optimizer to update all layer parameters based on their gradients
            // (not just setting current params back unchanged)
            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
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

        foreach (var layer in _visionEncoderLayers)
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

        foreach (var layer in _projectionLayers)
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

        foreach (var layer in _languageModelLayers)
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
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = Enums.ModelType.LLaVA,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ImageSize", _imageSize },
                { "EmbeddingDimension", _embeddingDimension },
                { "MaxSequenceLength", _maxSequenceLength },
                { "LanguageModelType", _languageModelType },
                { "VisionEncoderType", _visionEncoderType },
                { "NumVisualTokens", _numVisualTokens },
                { "VisionHiddenDim", _visionHiddenDim },
                { "LMHiddenDim", _lmHiddenDim },
                { "NumVisionLayers", _numVisionLayers },
                { "NumLMLayers", _numLmLayers },
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
        writer.Write(_lmHiddenDim);
        writer.Write(_numVisionLayers);
        writer.Write(_numLmLayers);
        writer.Write(_numHeads);
        writer.Write(_patchSize);
        writer.Write(_vocabularySize);
        writer.Write(_numVisualTokens);
        writer.Write(_languageModelType);
        writer.Write(_visionEncoderType);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // embeddingDim
        _ = reader.ReadInt32(); // maxSeqLen
        _ = reader.ReadInt32(); // imgSize
        _ = reader.ReadInt32(); // visionHiddenDim
        _ = reader.ReadInt32(); // lmHiddenDim
        _ = reader.ReadInt32(); // numVisionLayers
        _ = reader.ReadInt32(); // numLmLayers
        _ = reader.ReadInt32(); // numHeads
        _ = reader.ReadInt32(); // patchSize
        _ = reader.ReadInt32(); // vocabularySize
        _ = reader.ReadInt32(); // numVisualTokens
        _ = reader.ReadString(); // languageModelType
        _ = reader.ReadString(); // visionEncoderType
        _ = reader.ReadBoolean(); // useNativeMode
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new LLaVANeuralNetwork<T>(
            Architecture,
            _imageSize,
            channels: 3,
            _patchSize,
            _vocabularySize,
            _maxSequenceLength,
            _embeddingDimension,
            _visionHiddenDim,
            _numVisionLayers,
            _numLmLayers,
            _numHeads,
            _languageModelType,
            _visionEncoderType);
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _visionEncoder?.Dispose();
            _languageModel?.Dispose();
        }

        base.Dispose(disposing);
    }

    #endregion
}
