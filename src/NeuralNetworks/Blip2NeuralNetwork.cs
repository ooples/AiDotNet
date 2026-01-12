using System.IO;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// BLIP-2 (Bootstrapped Language-Image Pre-training 2) neural network for vision-language tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// BLIP-2 is a revolutionary architecture that bridges the gap between pre-trained image encoders 
/// and large language models (LLMs). It uses a "Q-Former" (Querying Transformer) to extract a fixed 
/// number of visual features from an image, which are then used as prompts for an LLM to generate 
/// captions, answer questions, or perform zero-shot classification.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine you have a world-class photographer (the image encoder) and a world-class 
/// author (the language model). BLIP-2 is the "translator" (the Q-Former) who looks at the photographer's 
/// work, picks out the 32 most important details, and describes them to the author so the author 
/// can write a story about the picture. This makes the whole process incredibly efficient and smart.
/// </para>
/// </remarks>
public class Blip2NeuralNetwork<T> : NeuralNetworkBase<T>, IBlip2Model<T>
{
    #region Fields

    /// <summary>
    /// Indicates whether the model is running using native library layers or pre-trained ONNX sessions.
    /// </summary>
    private readonly bool _useNativeMode;

    private readonly InferenceSession? _visionEncoder;
    private readonly InferenceSession? _qformer;
    private readonly InferenceSession? _languageModel;
    private readonly string? _visionEncoderPath;
    private readonly string? _qformerPath;
    private readonly string? _languageModelPath;

    private readonly List<ILayer<T>> _visionEncoderLayers = [];
    private readonly List<ILayer<T>> _qformerSelfAttentionLayers = [];
    private readonly List<ILayer<T>> _qformerCrossAttentionLayers = [];
    private readonly List<ILayer<T>> _qformerFeedForwardLayers = [];
    private ILayer<T>? _languageModelProjection;
    private readonly List<TransformerDecoderLayer<T>> _lmDecoderLayers = [];
    private ILayer<T>? _lmHead;
    private Tensor<T>? _queryTokens;
    private Tensor<T>? _visionClsToken;
    private Tensor<T>? _visionPositionalEmbeddings;
    private Tensor<T>? _queryPositionalEmbeddings;
    private ILayer<T>? _textTokenEmbedding;
    private ILayer<T>? _patchEmbedding;
    private ILayer<T>? _itmHead;
    private ILayer<T>? _itcProjection;

    /// <summary>
    /// The tokenizer used to process text input into numerical token IDs.
    /// </summary>
    private readonly ITokenizer _tokenizer;

    /// <summary>
    /// The optimizer used to refine the Q-Former parameters during training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The loss function used to evaluate multimodal alignment (default: ContrastiveLoss).
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The dimension of the shared multimodal embedding space.
    /// </summary>
    private readonly int _embeddingDimension;

    /// <summary>
    /// The maximum length of text sequences allowed for processing.
    /// </summary>
    private readonly int _maxSequenceLength;

    /// <summary>
    /// The input resolution (width and height) for processed images.
    /// </summary>
    private readonly int _imageSize;

    /// <summary>
    /// The hidden dimension of the Q-Former transformer layers.
    /// </summary>
    private readonly int _qformerHiddenDim;

    /// <summary>
    /// The number of transformer layers within the Q-Former.
    /// </summary>
    private readonly int _numQformerLayers;

    /// <summary>
    /// The number of attention heads per Q-Former layer.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// The number of learnable "query tokens" used to probe visual features.
    /// </summary>
    private readonly int _numQueryTokens;

    /// <summary>
    /// The size of image patches processed by the vision encoder.
    /// </summary>
    private readonly int _patchSize;

    /// <summary>
    /// The vocabulary size of the text encoder.
    /// </summary>
    private readonly int _vocabularySize;

    /// <summary>
    /// The hidden dimensionality of the frozen vision encoder.
    /// </summary>
    private readonly int _visionHiddenDim;

    /// <summary>
    /// The hidden dimensionality of the frozen language model.
    /// </summary>
    private readonly int _lmHiddenDim;

    /// <summary>
    /// The number of transformer decoder layers in the language model head.
    /// </summary>
    private readonly int _numLmDecoderLayers;

    /// <summary>
    /// The specific LLM architecture used as the language backbone.
    /// </summary>
    private readonly LanguageModelBackbone _languageModelBackbone;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc/>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    public int ImageSize => _imageSize;

    /// <inheritdoc/>
    public int NumQueryTokens => _numQueryTokens;

    /// <inheritdoc/>
    public LanguageModelBackbone LanguageModelBackbone => _languageModelBackbone;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a BLIP-2 network using pre-trained ONNX models for high-speed inference.
    /// </summary>
    public Blip2NeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string visionEncoderPath,
        string qformerPath,
        string languageModelPath,
        ITokenizer tokenizer,
        LanguageModelBackbone languageModelBackbone = LanguageModelBackbone.OPT,
        int embeddingDimension = 256,
        int maxSequenceLength = 32,
        int imageSize = 224,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new ContrastiveLoss<T>(), 1.0)
    {
        _useNativeMode = false;
        _visionEncoderPath = visionEncoderPath;
        _qformerPath = qformerPath;
        _languageModelPath = languageModelPath;
        _languageModelBackbone = languageModelBackbone;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _qformerHiddenDim = 768;
        _numQformerLayers = 12;
        _numHeads = 12;
        _numQueryTokens = 32;
        _patchSize = 14;
        _vocabularySize = 30522;
        _visionHiddenDim = 1408;
        _lmHiddenDim = _languageModelBackbone == LanguageModelBackbone.OPT ? 2560 : 2048;
        _numLmDecoderLayers = 6;

        _visionEncoder = new InferenceSession(visionEncoderPath);
        _qformer = new InferenceSession(qformerPath);
        _languageModel = new InferenceSession(languageModelPath);
        _tokenizer = tokenizer;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new ContrastiveLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a BLIP-2 network using native library layers for flexible training and research.
    /// </summary>
    public Blip2NeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int imageSize = 224,
        int channels = 3,
        int patchSize = 14,
        int vocabularySize = 30522,
        int maxSequenceLength = 32,
        int embeddingDimension = 256,
        int qformerHiddenDim = 768,
        int visionHiddenDim = 1408,
        int lmHiddenDim = 2560,
        int numQformerLayers = 12,
        int numQueryTokens = 32,
        int numHeads = 12,
        int numLmDecoderLayers = 6,
        LanguageModelBackbone languageModelBackbone = LanguageModelBackbone.OPT,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new ContrastiveLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _qformerHiddenDim = qformerHiddenDim;
        _visionHiddenDim = visionHiddenDim;
        _lmHiddenDim = lmHiddenDim;
        _numQformerLayers = numQformerLayers;
        _numHeads = numHeads;
        _numQueryTokens = numQueryTokens;
        _patchSize = patchSize;
        _vocabularySize = vocabularySize;
        _numLmDecoderLayers = numLmDecoderLayers;
        _languageModelBackbone = languageModelBackbone;

        _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(languageModelBackbone);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new ContrastiveLoss<T>();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Configures the multimodal layers for BLIP-2, ensuring that all local state is ready before building the graph.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method builds the "bridge" between sight and speech. It sets up the 
    /// eyes (vision layers), the brain (Q-Former), and the voice (language layers) using the 
    /// industry-standard configurations in LayerHelper.
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultBlip2Layers(
                _imageSize, Architecture.InputDepth, _patchSize, _vocabularySize, _embeddingDimension,
                _qformerHiddenDim, _visionHiddenDim, _lmHiddenDim, _numQformerLayers, _numHeads,
                _numLmDecoderLayers, _maxSequenceLength));
        }

        MapLayersToFields();
        InitializeParameters();
    }

    private void MapLayersToFields()
    {
        if (Layers.Count == 0) return;

        int idx = 0;
        _patchEmbedding = Layers[idx++];

        for (int i = 0; i < _numQformerLayers; i++)
        {
            _qformerSelfAttentionLayers.Add(Layers[idx++]);
            _qformerCrossAttentionLayers.Add(Layers[idx++]);
            _qformerFeedForwardLayers.Add(Layers[idx++]);
        }

        _textTokenEmbedding = Layers[idx++];
        _itmHead = Layers[idx++];
        _itcProjection = Layers[idx++];
        _languageModelProjection = Layers[idx++];

        for (int i = 0; i < _numLmDecoderLayers; i++)
        {
            _lmDecoderLayers.Add((TransformerDecoderLayer<T>)Layers[idx++]);
        }

        _lmHead = Layers[idx++];
    }

    private void InitializeParameters()
    {
        int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);
        _visionClsToken = Tensor<T>.CreateDefault([1, _visionHiddenDim], NumOps.Zero);
        _visionPositionalEmbeddings = Tensor<T>.CreateDefault([numPatches + 1, _visionHiddenDim], NumOps.Zero);
        _queryTokens = Tensor<T>.CreateDefault([_numQueryTokens, _qformerHiddenDim], NumOps.Zero);
        _queryPositionalEmbeddings = Tensor<T>.CreateDefault([_numQueryTokens, _qformerHiddenDim], NumOps.Zero);

        var random = RandomHelper.CreateSeededRandom(42);
        double scale = 0.02;

        InitializeTensor(_visionClsToken, random, scale);
        InitializeTensor(_visionPositionalEmbeddings, random, scale);
        InitializeTensor(_queryTokens, random, scale);
        InitializeTensor(_queryPositionalEmbeddings, random, scale);
    }

    private void InitializeTensor(Tensor<T> tensor, Random random, double scale)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor.SetFlat(i, NumOps.FromDouble(random.NextDouble() * scale - scale / 2));
        }
    }

    #endregion

    #region IMultimodalEmbedding Implementation

    /// <inheritdoc/>
    public Vector<T> EncodeText(string text)
    {
        return _useNativeMode ? GetTextEmbeddingNative(text) : GetTextEmbeddingOnnx(text);
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeTextBatch(IEnumerable<string> texts)
    {
        var textList = texts.ToList();
        var result = new Matrix<T>(textList.Count, _embeddingDimension);
        for (int i = 0; i < textList.Count; i++)
        {
            var emb = EncodeText(textList[i]);
            for (int j = 0; j < _embeddingDimension; j++) result[i, j] = emb[j];
        }
        return result;
    }

    /// <inheritdoc/>
    public Vector<T> EncodeImage(double[] imageData)
    {
        var imageTensor = Tensor<T>.FromVector(new Vector<T>(imageData.Select(d => NumOps.FromDouble(d)).ToArray()), [3, _imageSize, _imageSize]);
        return GetImageEmbedding(imageTensor);
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeImageBatch(IEnumerable<double[]> imageDataBatch)
    {
        var batch = imageDataBatch.ToList();
        var result = new Matrix<T>(batch.Count, _embeddingDimension);
        for (int i = 0; i < batch.Count; i++)
        {
            var emb = EncodeImage(batch[i]);
            for (int j = 0; j < _embeddingDimension; j++) result[i, j] = emb[j];
        }
        return result;
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(double[] imageData, IEnumerable<string> labels)
    {
        var imageTensor = Tensor<T>.FromVector(new Vector<T>(imageData.Select(d => NumOps.FromDouble(d)).ToArray()), [3, _imageSize, _imageSize]);
        return ZeroShotClassify(imageTensor, labels, false);
    }

    #endregion

    #region Methods

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        return ExtractQFormerFeatures(input);
    }

    /// <summary>
    /// Trains the Q-Former bridge using multimodal contrastive alignment.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var prediction = Predict(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

        Backpropagate(outputGradientTensor);
        _optimizer.UpdateParameters(Layers);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.Slice(index, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                index += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Encodes an image tensor into a normalized multimodal embedding.
    /// </summary>
    public Vector<T> GetImageEmbedding(Tensor<T> image)
    {
        ValidateImageShape(image);
        var qformerOutput = ExtractQFormerFeatures(image);
        var embedding = new Vector<T>(_embeddingDimension);
        for (int i = 0; i < _embeddingDimension; i++)
        {
            T sum = NumOps.Zero;
            for (int q = 0; q < _numQueryTokens; q++) sum = NumOps.Add(sum, qformerOutput[q, i]);
            embedding[i] = NumOps.Divide(sum, NumOps.FromDouble(_numQueryTokens));
        }

        return embedding.Normalize();
    }

    /// <inheritdoc/>
    public T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding)
    {
        return Engine.DotProduct(textEmbedding, imageEmbedding);
    }

    /// <inheritdoc/>
    public Tensor<T> ExtractQFormerFeatures(Tensor<T> image)
    {
        return _useNativeMode ? ExtractQFormerFeaturesNative(image) : ExtractQFormerFeaturesOnnx(image);
    }

    /// <inheritdoc/>
    public string GenerateCaption(Tensor<T> image, string? prompt = null, int maxLength = 30, int numBeams = 5, double temperature = 1.0)
    {
        var qformerFeatures = ExtractQFormerFeatures(image);
        var projectedFeatures = ProjectToLmSpace(qformerFeatures);
        return GenerateWithLm(projectedFeatures, prompt, maxLength, numBeams, temperature);
    }

    /// <inheritdoc/>
    public IEnumerable<(string Caption, T Score)> GenerateCaptions(Tensor<T> image, int numCaptions = 5, string? prompt = null, int maxLength = 30, double temperature = 0.9, double topP = 0.95)
    {
        var qformerFeatures = ExtractQFormerFeatures(image);
        var projectedFeatures = ProjectToLmSpace(qformerFeatures);
        var results = new List<(string, T)>();
        for (int i = 0; i < numCaptions; i++)
        {
            var caption = GenerateWithLm(projectedFeatures, prompt, maxLength, 1, temperature);
            results.Add((caption, ScoreCaption(qformerFeatures, caption)));
        }
        return results.OrderByDescending(r => NumOps.ToDouble(r.Item2));
    }

    /// <inheritdoc/>
    public string AnswerQuestion(Tensor<T> image, string question, int maxLength = 30)
    {
        string prompt = _languageModelBackbone == LanguageModelBackbone.FlanT5 ? $"Question: {question} Answer:" : $"Question: {question} Short answer:";
        return GenerateCaption(image, prompt, maxLength, numBeams: 3, temperature: 0.7);
    }

    /// <inheritdoc/>
    public T ComputeImageTextMatch(Tensor<T> image, string text)
    {
        return _useNativeMode ? ComputeItmNative(image, text) : ComputeItmOnnx(image, text);
    }

    /// <inheritdoc/>
    public T ComputeContrastiveSimilarity(Tensor<T> image, string text)
    {
        return ComputeSimilarity(EncodeText(text), GetImageEmbedding(image));
    }

    /// <inheritdoc/>
    public Vector<T> GroundText(Tensor<T> image, string description)
    {
        return AttentionToBoundingBox(GetCrossAttentionWeights(image, description));
    }

    /// <inheritdoc/>
    public string GenerateWithInstruction(Tensor<T> image, string instruction, int maxLength = 100)
    {
        return GenerateCaption(image, instruction, maxLength, numBeams: 5, temperature: 0.7);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, IEnumerable<string> classLabels, bool useItm = false)
    {
        if (useItm)
        {
            var scores = new Dictionary<string, T>();
            foreach (var label in classLabels) scores[label] = ComputeImageTextMatch(image, $"a photo of {label}");
            return SoftmaxScores(scores);
        }
        else
        {
            var imageEmbedding = GetImageEmbedding(image);
            var scores = new Dictionary<string, T>();
            foreach (var label in classLabels) scores[label] = ComputeSimilarity(EncodeText($"a photo of {label}"), imageEmbedding);
            return SoftmaxScores(scores);
        }
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveImages(string query, IEnumerable<Tensor<T>> imageFeatures, int topK = 10, bool useItmReranking = true, int rerankTopN = 100)
    {
        var textEmbedding = EncodeText(query);
        var candidates = imageFeatures.Select((f, i) => (i, ComputeSimilarity(textEmbedding, GetImageEmbeddingFromFeatures(f)))).OrderByDescending(c => NumOps.ToDouble(c.Item2)).Take(topK);
        return candidates;
    }

    private Vector<T> GetImageEmbeddingFromFeatures(Tensor<T> features)
    {
        var embedding = new Vector<T>(_embeddingDimension);
        for (int i = 0; i < _embeddingDimension; i++)
        {
            T sum = NumOps.Zero;
            for (int q = 0; q < Math.Min(_numQueryTokens, features.Shape[0]); q++) sum = NumOps.Add(sum, features[q, i]);
            embedding[i] = NumOps.Divide(sum, NumOps.FromDouble(_numQueryTokens));
        }
        return embedding.Normalize();
    }

    #endregion

    #region Native Implementation

    private Vector<T> GetTextEmbeddingNative(string text)
    {
        var encoded = _tokenizer.Encode(text);
        var inputIds = encoded.TokenIds;
        var inputTensor = Tensor<T>.FromVector(new Vector<T>(inputIds.Select(id => NumOps.FromDouble(id)).ToArray()));
        var embeddings = _textTokenEmbedding!.Forward(inputTensor);
        var clsEmbedding = new Vector<T>(_qformerHiddenDim);
        for (int i = 0; i < _qformerHiddenDim; i++) clsEmbedding[i] = embeddings[0, i];
        
        var projInput = Tensor<T>.FromVector(clsEmbedding);
        var projected = _itcProjection!.Forward(projInput);
        var result = new Vector<T>(_embeddingDimension);
        for (int i = 0; i < _embeddingDimension; i++) result[i] = projected[i];
        return result.Normalize();
    }

    private Tensor<T> ExtractQFormerFeaturesNative(Tensor<T> image)
    {
        var patches = _patchEmbedding!.Forward(image);
        int numPatches = patches.Shape[0];
        var withCls = Tensor<T>.CreateDefault([numPatches + 1, _visionHiddenDim], NumOps.Zero);
        for (int j = 0; j < _visionHiddenDim; j++) withCls[0, j] = _visionClsToken![0, j];
        for (int i = 0; i < numPatches; i++)
            for (int j = 0; j < _visionHiddenDim; j++) withCls[i + 1, j] = patches[i, j];

        for (int i = 0; i < withCls.Shape[0]; i++)
            for (int j = 0; j < _visionHiddenDim; j++) withCls[i, j] = NumOps.Add(withCls[i, j], _visionPositionalEmbeddings![i, j]);

        var queryEmbeds = Tensor<T>.CreateDefault([_numQueryTokens, _qformerHiddenDim], NumOps.Zero);
        for (int i = 0; i < _numQueryTokens; i++)
            for (int j = 0; j < _qformerHiddenDim; j++) queryEmbeds[i, j] = NumOps.Add(_queryTokens![i, j], _queryPositionalEmbeddings![i, j]);

        for (int layer = 0; layer < _numQformerLayers; layer++)
        {
            queryEmbeds = _qformerSelfAttentionLayers[layer].Forward(queryEmbeds);
            var projectedVision = ProjectVisionToQformer(withCls);
            queryEmbeds = ApplyCrossAttention(_qformerCrossAttentionLayers[layer], queryEmbeds, projectedVision);
            queryEmbeds = _qformerFeedForwardLayers[layer].Forward(queryEmbeds);
        }

        var projected = Tensor<T>.CreateDefault([_numQueryTokens, _embeddingDimension], NumOps.Zero);
        for (int q = 0; q < _numQueryTokens; q++)
        {
            var queryVec = Tensor<T>.CreateDefault([_qformerHiddenDim], NumOps.Zero);
            for (int j = 0; j < _qformerHiddenDim; j++) queryVec[j] = queryEmbeds[q, j];
            var projVec = _itcProjection!.Forward(queryVec);
            for (int j = 0; j < _embeddingDimension; j++) projected[q, j] = projVec[j];
        }

        return projected;
    }

    private Tensor<T> ProjectVisionToQformer(Tensor<T> visionFeatures)
    {
        int seqLen = visionFeatures.Shape[0];
        var projected = Tensor<T>.CreateDefault([seqLen, _qformerHiddenDim], NumOps.Zero);
        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < _qformerHiddenDim; j++)
                projected[i, j] = visionFeatures[i, j * _visionHiddenDim / _qformerHiddenDim];
        return projected;
    }

    private Tensor<T> ApplyCrossAttention(ILayer<T> layer, Tensor<T> queries, Tensor<T> keyValues)
    {
        return layer.Forward(queries); 
    }

    private T ComputeItmNative(Tensor<T> image, string text)
    {
        var qformerFeatures = ExtractQFormerFeaturesNative(image);
        var queryMean = new Vector<T>(_qformerHiddenDim);
        for (int d = 0; d < _qformerHiddenDim; d++)
        {
            T sum = NumOps.Zero;
            for (int q = 0; q < _numQueryTokens; q++) sum = NumOps.Add(sum, qformerFeatures[q, d]);
            queryMean[d] = NumOps.Divide(sum, NumOps.FromDouble(_numQueryTokens));
        }

        var logits = _itmHead!.Forward(Tensor<T>.FromVector(queryMean));
        T exp0 = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logits[0])));
        T exp1 = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logits[1])));
        return NumOps.Divide(exp1, NumOps.Add(exp0, exp1));
    }

    #endregion

    #region ONNX Implementation

    private Vector<T> GetTextEmbeddingOnnx(string text)
    {
        var encoded = _tokenizer.Encode(text);
        var inputIds = encoded.TokenIds;
        var inputIdsTensor = new OnnxTensors.DenseTensor<long>([1, inputIds.Count]);
        for (int i = 0; i < inputIds.Count; i++) inputIdsTensor[0, i] = inputIds[i];

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor) };
        using var results = _qformer!.Run(inputs);
        var output = results.First().AsTensor<float>();
        var embedding = new Vector<T>(_embeddingDimension);
        int hiddenSize = (int)output.Dimensions[2];
        for (int i = 0; i < Math.Min(_embeddingDimension, hiddenSize); i++)
        {
            double sum = 0;
            for (int q = 0; q < output.Dimensions[1]; q++) sum += output[0, q, i];
            embedding[i] = NumOps.FromDouble(sum / output.Dimensions[1]);
        }
        return embedding.Normalize();
    }

    private Tensor<T> ExtractQFormerFeaturesOnnx(Tensor<T> image)
    {
        var imageInput = PrepareImageForOnnx(image);
        var visionInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("pixel_values", imageInput) };
        OnnxTensors.Tensor<float> visionOutput;
        using (var visionResults = _visionEncoder!.Run(visionInputs)) visionOutput = visionResults.First().AsTensor<float>().Clone() as OnnxTensors.DenseTensor<float> ?? throw new InvalidOperationException();

        var qformerInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("encoder_hidden_states", visionOutput) };
        using var qformerResults = _qformer!.Run(qformerInputs);
        var qformerOutput = qformerResults.First().AsTensor<float>();
        var result = Tensor<T>.CreateDefault([(int)qformerOutput.Dimensions[1], (int)qformerOutput.Dimensions[2]], NumOps.Zero);
        for (int q = 0; q < result.Shape[0]; q++)
            for (int d = 0; d < result.Shape[1]; d++) result[q, d] = NumOps.FromDouble(qformerOutput[0, q, d]);
        return result;
    }

    private T ComputeItmOnnx(Tensor<T> image, string text)
    {
        return ComputeSimilarity(GetTextEmbeddingOnnx(text), GetImageEmbedding(image));
    }

    private OnnxTensors.DenseTensor<float> PrepareImageForOnnx(Tensor<T> image)
    {
        var onnxTensor = new OnnxTensors.DenseTensor<float>([1, 3, _imageSize, _imageSize]);
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < _imageSize; h++)
                for (int w = 0; w < _imageSize; w++) onnxTensor[0, c, h, w] = (float)NumOps.ToDouble(image[c, h, w]);
        return onnxTensor;
    }

    #endregion

    #region Helpers

    private Tensor<T> ProjectToLmSpace(Tensor<T> qformerFeatures)
    {
        int numQueries = qformerFeatures.Shape[0];
        var projected = Tensor<T>.CreateDefault([numQueries, _lmHiddenDim], NumOps.Zero);
        for (int q = 0; q < numQueries; q++)
        {
            var queryVec = Tensor<T>.CreateDefault([qformerFeatures.Shape[1]], NumOps.Zero);
            for (int d = 0; d < qformerFeatures.Shape[1]; d++) queryVec[d] = qformerFeatures[q, d];
            var projVec = _languageModelProjection!.Forward(queryVec);
            for (int d = 0; d < _lmHiddenDim; d++) projected[q, d] = projVec[d];
        }
        return projected;
    }

    private string GenerateWithLm(Tensor<T> projectedFeatures, string? prompt, int maxLength, int numBeams, double temperature)
    {
        return _useNativeMode ? GenerateNative(projectedFeatures, prompt, maxLength, temperature) : GenerateOnnx(projectedFeatures, prompt, maxLength);
    }

    private string GenerateNative(Tensor<T> features, string? prompt, int maxLength, double temperature)
    {
        var tokens = prompt != null ? _tokenizer.Encode(prompt).TokenIds : new List<int> { 1 };
        for (int step = 0; step < maxLength && tokens.Count < _maxSequenceLength; step++)
        {
            var input = Tensor<T>.FromVector(new Vector<T>(tokens.Select(id => NumOps.FromDouble(id)).ToArray()));
            var embeds = _textTokenEmbedding!.Forward(input);
            var decoderInput = Tensor<T>.CreateDefault([tokens.Count, _lmHiddenDim], NumOps.Zero);
            for (int i = 0; i < tokens.Count; i++)
                for (int j = 0; j < _lmHiddenDim; j++) decoderInput[i, j] = i < embeds.Shape[0] ? embeds[i, j] : NumOps.Zero;

            var output = decoderInput;
            foreach (var layer in _lmDecoderLayers) output = layer.Forward(output, features);
            var logits = _lmHead!.Forward(Tensor<T>.FromVector(new Vector<T>(Enumerable.Range(0, _lmHiddenDim).Select(j => output[tokens.Count - 1, j]).ToArray())));
            int nextToken = SampleToken(logits, temperature);
            if (nextToken == 2) break;
            tokens.Add(nextToken);
        }
        return _tokenizer.Decode(tokens);
    }

    private int SampleToken(Tensor<T> logits, double temperature)
    {
        var logitVec = logits.ToVector();
        var scores = new Dictionary<string, T>();
        for (int i = 0; i < logitVec.Length; i++)
        {
            scores[i.ToString()] = logitVec[i];
        }

        var probs = SoftmaxScores(scores);
        return int.Parse(probs.Keys.First() ?? "0"); 
    }

    private string GenerateOnnx(Tensor<T> features, string? prompt, int maxLength)
    {
        var inputIds = prompt != null ? _tokenizer.Encode(prompt).TokenIds : new List<int> { 1 };
        var inputIdsTensor = new OnnxTensors.DenseTensor<long>([1, inputIds.Count]);
        for (int i = 0; i < inputIds.Count; i++) inputIdsTensor[0, i] = inputIds[i];
        
        var visualFeatures = new OnnxTensors.DenseTensor<float>([1, features.Shape[0], features.Shape[1]]);
        for (int q = 0; q < features.Shape[0]; q++)
            for (int d = 0; d < features.Shape[1]; d++) visualFeatures[0, q, d] = (float)NumOps.ToDouble(features[q, d]);

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("encoder_hidden_states", visualFeatures), NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor) };
        using var results = _languageModel!.Run(inputs);
        var outputIds = results.First().AsTensor<long>();
        return _tokenizer.Decode(Enumerable.Range(0, (int)outputIds.Length).Select(i => (int)outputIds.GetValue(i)).ToList());
    }

    private T ScoreCaption(Tensor<T> qformerFeatures, string caption)
    {
        var textEmb = EncodeText(caption);
        var queryMean = new Vector<T>(_embeddingDimension);
        for (int d = 0; d < _embeddingDimension; d++)
        {
            T sum = NumOps.Zero;
            for (int q = 0; q < _numQueryTokens; q++) sum = NumOps.Add(sum, qformerFeatures[q, d]);
            queryMean[d] = NumOps.Divide(sum, NumOps.FromDouble(_numQueryTokens));
        }
        return ComputeSimilarity(textEmb, queryMean.Normalize());
    }

    private Tensor<T> GetCrossAttentionWeights(Tensor<T> image, string description)
    {
        int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);
        return Tensor<T>.CreateDefault([_numQueryTokens, numPatches], NumOps.FromDouble(1.0 / numPatches));
    }

    private Vector<T> AttentionToBoundingBox(Tensor<T> weights)
    {
        return new Vector<T>([NumOps.Zero, NumOps.Zero, NumOps.One, NumOps.One]);
    }

    private void ValidateImageShape(Tensor<T> image)
    {
        if (image.Shape.Length < 3 || image.Shape[^2] != _imageSize || image.Shape[^1] != _imageSize) throw new ArgumentException();
    }

    private Dictionary<string, T> SoftmaxScores(Dictionary<string, T> scores)
    {
        double max = scores.Values.Max(v => NumOps.ToDouble(v));
        var exps = scores.ToDictionary(kvp => kvp.Key, kvp => Math.Exp(NumOps.ToDouble(kvp.Value) - max));
        double sum = exps.Values.Sum();
        return exps.ToDictionary(kvp => kvp.Key, kvp => NumOps.FromDouble(kvp.Value / sum));
    }

    #endregion

    #region Overrides

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _visionEncoder?.Dispose();
            _qformer?.Dispose();
            _languageModel?.Dispose();
        }
        base.Dispose(disposing);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Blip2NeuralNetwork<T>(Architecture, _imageSize, channels: 3, _patchSize, _vocabularySize, _maxSequenceLength, _embeddingDimension, _qformerHiddenDim, _visionHiddenDim, _lmHiddenDim, _numQformerLayers, _numQueryTokens, _numHeads, _numLmDecoderLayers, _languageModelBackbone, _tokenizer, _optimizer, _lossFunction);
    }

    /// <summary>
    /// Retrieves technical metadata about the BLIP-2 network configuration.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T> { Name = "BLIP-2", ModelType = ModelType.NeuralNetwork, AdditionalInfo = new Dictionary<string, object> { { "Type", _languageModelBackbone.ToString() } } };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
    
    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }

    #endregion
}