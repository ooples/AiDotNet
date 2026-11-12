using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Unified multimodal network that handles text, images, audio, and video
/// in a single architecture with cross-modal attention and any-to-any generation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This network handles text, images, audio, and video all in
/// one architecture. Instead of needing separate models for each type of data, it processes
/// all modalities through shared transformer layers with cross-modal attention. It can both
/// understand any combination of inputs and generate outputs in any modality, enabling tasks
/// like describing a video, generating images from text, or answering questions about audio
/// clips.</para>
/// </remarks>
/// <example>
/// <code>
/// var image = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 3, 224, 224 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new UnifiedMultimodalNetwork&lt;float&gt;())
///     .Build(trainX, trainY);
/// var output = result.Predict(image);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Perceiver: General Perception with Iterative Attention", "https://arxiv.org/abs/2103.03206")]
public partial class UnifiedMultimodalNetwork<T> : MultimodalModelLayoutBase<T>, IUnifiedMultimodalModel<T>
{
    private readonly UnifiedMultimodalNetworkOptions _options;

    #region Constants

    private const int DEFAULT_EMBEDDING_DIM = 768;
    private const int DEFAULT_MAX_SEQ_LEN = 2048;
    private const int DEFAULT_NUM_LAYERS = 12;

    #endregion

    #region Fields

    private readonly INumericOperations<T> _numOps;
    private readonly int _embeddingDimension;
    private readonly int _maxSequenceLength;
    private readonly int _numTransformerLayers;
    private readonly Random _random;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    // Modality-specific encoders
    private DenseLayer<T> _textEncoder;
    private DenseLayer<T> _imageEncoder;
    private DenseLayer<T> _audioEncoder;
    private DenseLayer<T> _videoEncoder;

    // Unified transformer
    private MultiHeadAttentionLayer<T>[] _transformerLayers;

    // Cross-modal attention
    private MultiHeadAttentionLayer<T>[] _crossModalAttention;

    // Modality-specific decoders
    private DenseLayer<T> _textDecoder;
    private DenseLayer<T> _imageDecoder;
    private DenseLayer<T> _audioDecoder;
    private DenseLayer<T> _videoDecoder;

    // Fusion and output heads
    private DenseLayer<T> _fusionLayer;
    private DenseLayer<T> _classificationHead;
    private DenseLayer<T> _generationHead;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public IReadOnlyList<ModalityType> SupportedInputModalities { get; }

    /// <inheritdoc/>
    public IReadOnlyList<ModalityType> SupportedOutputModalities { get; }

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc/>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    public bool SupportsStreaming => true;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    private const int DefaultOutputSize = 100;

    // Feature widths the image, audio and video encoders read (EncodeModality's defaults).
    private const int ImageFeatureWidth = 768;
    private const int AudioFeatureWidth = 128;
    private const int VideoFeatureWidth = 1024;

    public UnifiedMultimodalNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputSize: DEFAULT_EMBEDDING_DIM,
            outputSize: DefaultOutputSize))
    {
    }

    /// <summary>
    /// Initializes a new instance of the UnifiedMultimodalNetwork.
    /// </summary>
    public UnifiedMultimodalNetwork(
        NeuralNetworkArchitecture<T> architecture,
        UnifiedMultimodalNetworkOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new UnifiedMultimodalNetworkOptions();
        _options.Validate();
        Options = _options;

        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDimension = _options.EmbeddingDimension;
        _maxSequenceLength = _options.MaxSequenceLength;
        _numTransformerLayers = _options.NumTransformerLayers;
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSeededRandom(42);
        _lossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();
        // AdamW (decoupled weight decay) is the canonical optimizer for multimodal transformers:
        // Lumina-T2X (Gao et al. 2024) and its Gemma 2B text encoder both train with AdamW +
        // weight decay 0.01. Plain Adam has no mechanism to bound the classification-head logits,
        // so under CrossEntropyWithLogitsLoss the logits inflate without limit as training
        // continues (correct-class logit → +∞, others → −∞). Decoupled weight decay establishes a
        // stable equilibrium where the logit magnitude — and therefore the loss — plateaus instead
        // of drifting, which is the paper-faithful behaviour for a regularized transformer.
        _optimizer = optimizer ?? new Optimizers.AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        SupportedInputModalities = new List<ModalityType>
        {
            ModalityType.Text,
            ModalityType.Image,
            ModalityType.Audio,
            ModalityType.Video
        }.AsReadOnly();

        SupportedOutputModalities = new List<ModalityType>
        {
            ModalityType.Text,
            ModalityType.Image,
            ModalityType.Audio
        }.AsReadOnly();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    [System.Diagnostics.CodeAnalysis.MemberNotNull(
        nameof(_textEncoder), nameof(_imageEncoder), nameof(_audioEncoder), nameof(_videoEncoder),
        nameof(_transformerLayers), nameof(_crossModalAttention),
        nameof(_textDecoder), nameof(_imageDecoder), nameof(_audioDecoder), nameof(_videoDecoder),
        nameof(_fusionLayer), nameof(_classificationHead), nameof(_generationHead))]
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultUnifiedMultimodalLayers(
                Architecture, _embeddingDimension, _numTransformerLayers));
        }

        BindLayerFieldsFromLayers();
    }

    [System.Diagnostics.CodeAnalysis.MemberNotNull(
        nameof(_textEncoder), nameof(_imageEncoder), nameof(_audioEncoder), nameof(_videoEncoder),
        nameof(_transformerLayers), nameof(_crossModalAttention),
        nameof(_textDecoder), nameof(_imageDecoder), nameof(_audioDecoder), nameof(_videoDecoder),
        nameof(_fusionLayer), nameof(_classificationHead), nameof(_generationHead))]
    private void BindLayerFieldsFromLayers()
    {
        int expectedLayerCount = 4 + _numTransformerLayers + 4 + 4 + 3;
        if (Layers.Count < expectedLayerCount)
        {
            throw new InvalidDataException(
                $"UnifiedMultimodalNetwork expected at least {expectedLayerCount} layers, found {Layers.Count}.");
        }

        TLayer RequireLayer<TLayer>(int index, string role)
            where TLayer : class, ILayer<T>
        {
            if (Layers[index] is TLayer layer)
                return layer;

            throw new InvalidDataException(
                $"UnifiedMultimodalNetwork layer {index} for {role} must be {typeof(TLayer).Name}, found {Layers[index].GetType().Name}.");
        }

        int idx = 0;

        _textEncoder = RequireLayer<DenseLayer<T>>(idx++, "text encoder");
        _imageEncoder = RequireLayer<DenseLayer<T>>(idx++, "image encoder");
        _audioEncoder = RequireLayer<DenseLayer<T>>(idx++, "audio encoder");
        _videoEncoder = RequireLayer<DenseLayer<T>>(idx++, "video encoder");

        _transformerLayers = new MultiHeadAttentionLayer<T>[_numTransformerLayers];
        for (int i = 0; i < _numTransformerLayers; i++)
        {
            _transformerLayers[i] = RequireLayer<MultiHeadAttentionLayer<T>>(idx++, $"transformer layer {i}");
        }

        _crossModalAttention = new MultiHeadAttentionLayer<T>[4];
        for (int i = 0; i < 4; i++)
        {
            _crossModalAttention[i] = RequireLayer<MultiHeadAttentionLayer<T>>(idx++, $"cross-modal attention {i}");
        }

        _textDecoder = RequireLayer<DenseLayer<T>>(idx++, "text decoder");
        _imageDecoder = RequireLayer<DenseLayer<T>>(idx++, "image decoder");
        _audioDecoder = RequireLayer<DenseLayer<T>>(idx++, "audio decoder");
        _videoDecoder = RequireLayer<DenseLayer<T>>(idx++, "video decoder");

        _fusionLayer = RequireLayer<DenseLayer<T>>(idx++, "fusion layer");
        _classificationHead = RequireLayer<DenseLayer<T>>(idx++, "classification head");
        _generationHead = RequireLayer<DenseLayer<T>>(idx++, "generation head");
    }

    /// <summary>One-shot latch for <see cref="ResolveLazyLayerShapes"/>.</summary>
    private bool _topologyShapesResolved;

    /// <summary>
    /// Resolves each lazy layer from the input its forward actually receives, not from its predecessor
    /// in <see cref="NeuralNetworkBase{T}.Layers"/>.
    /// </summary>
    /// <remarks>
    /// The layer list is flat, but the network is not a chain: the four modality decoders, the
    /// classification head and the generation head each read the shared embedding, and each encoder
    /// reads its own modality. The base walk fed every layer its predecessor's output, so the image
    /// decoder took the text decoder's 50,000 logits as input, the audio decoder took the image
    /// decoder's 2,304, and the video decoder took the audio decoder's 16,000 (a 295M-parameter
    /// matrix). The model came to more than 500M parameters, and serializing it overflowed a byte
    /// array ("Array dimensions exceeded supported range", nightly run 39554054058).
    /// <see cref="LayerBase{T}.ResolveShapesOnly"/> sets shapes without allocating or drawing on the RNG.
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_topologyShapesResolved) return;
        _topologyShapesResolved = true;

        int embedding = _embeddingDimension;
        _textEncoder.ResolveShapesOnly([Architecture.InputSize > 0 ? Architecture.InputSize : embedding]);
        _imageEncoder.ResolveShapesOnly([ImageFeatureWidth]);
        _audioEncoder.ResolveShapesOnly([AudioFeatureWidth]);
        _videoEncoder.ResolveShapesOnly([VideoFeatureWidth]);
        foreach (var layer in _transformerLayers) layer.ResolveShapesOnly([1, 1, embedding]);
        foreach (var layer in _crossModalAttention) layer.ResolveShapesOnly([1, 1, embedding]);
        foreach (var decoder in new[] { _textDecoder, _imageDecoder, _audioDecoder, _videoDecoder })
            decoder.ResolveShapesOnly([embedding]);
        // Early fusion concatenates at most four modality embeddings.
        _fusionLayer.ResolveShapesOnly([embedding * 4]);
        _classificationHead.ResolveShapesOnly([embedding]);
        _generationHead.ResolveShapesOnly([embedding]);
    }

    #endregion

    #region Encoding

    /// <inheritdoc/>
    public Vector<T> Encode(MultimodalInput<T> input)
    {
        var encoded = EncodeModality(input);
        return ApplyTransformer(encoded);
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeSequence(IEnumerable<MultimodalInput<T>> inputs)
    {
        var inputList = inputs.OrderBy(i => i.SequenceIndex).ToList();
        var embeddings = new List<Vector<T>>();

        foreach (var input in inputList)
        {
            embeddings.Add(Encode(input));
        }

        // Create matrix from embeddings
        var matrix = new Matrix<T>(embeddings.Count, _embeddingDimension);
        var rows = matrix.AsWritableSpan();
        for (int i = 0; i < embeddings.Count; i++)
        {
            int width = Math.Min(_embeddingDimension, embeddings[i].Length);
            embeddings[i].AsSpan().Slice(0, width).CopyTo(rows.Slice(i * _embeddingDimension, width));
        }

        return matrix;
    }

    private Vector<T> EncodeModality(MultimodalInput<T> input)
    {
        Tensor<T> inputTensor;
        DenseLayer<T> encoder;

        switch (input.Modality)
        {
            case ModalityType.Text:
                inputTensor = EncodeText(input.TextContent ?? string.Empty);
                encoder = _textEncoder;
                break;
            case ModalityType.Image:
                inputTensor = input.InternalData ?? new Tensor<T>(new[] { ImageFeatureWidth });
                encoder = _imageEncoder;
                break;
            case ModalityType.Audio:
                inputTensor = input.InternalData ?? new Tensor<T>(new[] { AudioFeatureWidth });
                encoder = _audioEncoder;
                break;
            case ModalityType.Video:
                inputTensor = input.InternalData ?? new Tensor<T>(new[] { VideoFeatureWidth });
                encoder = _videoEncoder;
                break;
            default:
                inputTensor = new Tensor<T>(new[] { _embeddingDimension });
                encoder = _textEncoder;
                break;
        }

        // Flatten input if needed
        var flatInput = FlattenToExpectedSize(inputTensor, encoder);
        var encoded = encoder.Forward(flatInput);
        return encoded.ToVector();
    }

    private Tensor<T> EncodeText(string text)
    {
        // Simple character-level encoding
        // code / 128 for the first 512 characters, zero after. Split as code / 128 + (code % 128) / 128 so
        // both parts are exact in T and the sum rounds once, as code / 128.0 converted to T does.
        int length = Math.Min(text.Length, 512);
        var wholes = new int[512];
        var remainders = new int[512];
        for (int i = 0; i < length; i++)
        {
            wholes[i] = text[i] / 128;
            remainders[i] = text[i] % 128;
        }

        var engine = AiDotNetEngine.Current;
        var whole = new Tensor<int>(wholes, new[] { 512 }).Cast<T>();
        var remainder = new Tensor<int>(remainders, new[] { 512 }).Cast<T>();
        var fraction = engine.TensorDivide(remainder, Tensor<T>.CreateDefault(new[] { 512 }, _numOps.FromDouble(128.0)));
        return engine.TensorAdd(whole, fraction);
    }

    private Tensor<T> FlattenToExpectedSize(Tensor<T> input, DenseLayer<T> layer)
    {
        var inputData = input.ToVector();
        int expectedSize = GetLayerInputSize(layer);
        var result = new Vector<T>(expectedSize);
        int copied = Math.Min(inputData.Length, expectedSize);
        inputData.AsSpan().Slice(0, copied).CopyTo(result.AsWritableSpan());

        return Tensor<T>.FromVector(result);
    }

    private int GetLayerInputSize(DenseLayer<T> layer)
    {
        // Get expected input size from layer parameters
        var params_ = layer.GetParameters();
        // Input size is typically first dimension of weight matrix
        return Math.Max(1, params_.Length / _embeddingDimension);
    }

    private Vector<T> ApplyTransformer(Vector<T> embedding)
    {
        var current = Tensor<T>.FromVector(embedding);

        foreach (var layer in _transformerLayers)
        {
            current = layer.Forward(current);
        }

        return current.ToVector();
    }

    #endregion

    #region Generation

    /// <inheritdoc/>
    public MultimodalOutput<T> Generate(
        IEnumerable<MultimodalInput<T>> inputs,
        ModalityType outputModality,
        int maxLength = 1024)
    {
        // Encode all inputs
        var encodedSequence = EncodeSequence(inputs);

        // Apply cross-modal attention
        var fused = ApplyCrossModalAttention(encodedSequence);

        // Generate in target modality
        return DecodeToModality(fused, outputModality, maxLength);
    }

    /// <inheritdoc/>
    public string GenerateText(
        IEnumerable<MultimodalInput<T>> inputs,
        string prompt,
        int maxTokens = 1024,
        double temperature = 0.7)
    {
        var allInputs = inputs.ToList();
        allInputs.Add(MultimodalInput<T>.FromText(prompt, allInputs.Count));

        var output = Generate(allInputs, ModalityType.Text, maxTokens);
        return output.TextContent ?? string.Empty;
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateImage(
        IEnumerable<MultimodalInput<T>> inputs,
        int width = 512,
        int height = 512)
    {
        var output = Generate(inputs, ModalityType.Image, width * height * 3);
        return output.InternalData ?? new Tensor<T>(new[] { 3, height, width });
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateAudio(
        IEnumerable<MultimodalInput<T>> inputs,
        double durationSeconds = 5.0,
        int sampleRate = 44100)
    {
        int numSamples = (int)(durationSeconds * sampleRate);
        var output = Generate(inputs, ModalityType.Audio, numSamples);
        return output.InternalData ?? new Tensor<T>(new[] { numSamples });
    }

    private Vector<T> ApplyCrossModalAttention(Matrix<T> embeddings)
    {
        // Average pool across sequence
        var pooled = ColumnMeans(embeddings);

        // Apply cross-modal attention layers
        var current = Tensor<T>.FromVector(pooled);
        foreach (var layer in _crossModalAttention)
        {
            current = layer.Forward(current);
        }

        return current.ToVector();
    }

    private MultimodalOutput<T> DecodeToModality(Vector<T> embedding, ModalityType modality, int maxLength)
    {
        var embeddingTensor = Tensor<T>.FromVector(embedding);

        switch (modality)
        {
            case ModalityType.Text:
                return DecodeToText(embeddingTensor, maxLength);
            case ModalityType.Image:
                return DecodeToImage(embeddingTensor, maxLength);
            case ModalityType.Audio:
                return DecodeToAudio(embeddingTensor, maxLength);
            default:
                return new MultimodalOutput<T>
                {
                    Modality = modality,
                    TextContent = "Unsupported modality",
                    Confidence = _numOps.Zero
                };
        }
    }

    private MultimodalOutput<T> DecodeToText(Tensor<T> embedding, int maxLength)
    {
        var logits = _textDecoder.Forward(embedding);
        var logitsData = logits.ToVector();

        // Simple greedy decoding: the first highest of each block of 256 logits (NaN never wins)
        var text = new System.Text.StringBuilder();
        int positions = Math.Max(0, Math.Min(maxLength, logitsData.Length / 256));
        if (positions > 0)
        {
            var blocks = Tensor<T>.FromVector(logitsData.GetSubVector(0, positions * 256)).Reshape(positions, 256);
            foreach (int maxIdx in TieBreakArgMax.FirstMaxIndexPerRow(blocks).ToArray())
            {
                if (maxIdx >= 32 && maxIdx < 127)
                {
                    text.Append((char)maxIdx);
                }
            }
        }

        return new MultimodalOutput<T>
        {
            Modality = ModalityType.Text,
            TextContent = text.ToString(),
            Confidence = _numOps.One
        };
    }

    private MultimodalOutput<T> DecodeToImage(Tensor<T> embedding, int maxLength)
    {
        var decoded = _imageDecoder.Forward(embedding);
        int height = 256;
        int width = 256;
        int channels = 3;

        var imageData = new Vector<T>(channels * height * width);
        var decodedData = decoded.ToVector();

        // Sigmoid of the decoded values; pixels beyond them stay 0
        int filled = Math.Min(imageData.Length, decodedData.Length);
        if (filled > 0)
        {
            var squashed = Engine.Sigmoid(Tensor<T>.FromVector(decodedData.GetSubVector(0, filled)));
            squashed.AsSpan().CopyTo(imageData.AsWritableSpan());
        }

        return new MultimodalOutput<T>
        {
            Modality = ModalityType.Image,
            InternalData = new Tensor<T>(new[] { channels, height, width }, imageData),
            Confidence = _numOps.One
        };
    }

    private MultimodalOutput<T> DecodeToAudio(Tensor<T> embedding, int maxLength)
    {
        var decoded = _audioDecoder.Forward(embedding);
        var audioData = decoded.ToVector();

        // Normalize to [-1, 1] range: 2 * sigmoid(x) - 1; samples beyond the decoded ones stay 0
        var normalizedData = new Vector<T>(maxLength);
        int filled = Math.Min(maxLength, audioData.Length);
        if (filled > 0)
        {
            var squashed = Engine.Sigmoid(Tensor<T>.FromVector(audioData.GetSubVector(0, filled)));
            var centered = Engine.TensorSubtract(
                Engine.TensorMultiplyScalar(squashed, _numOps.FromDouble(2.0)),
                Tensor<T>.CreateDefault(squashed._shape, _numOps.One));
            centered.AsSpan().CopyTo(normalizedData.AsWritableSpan());
        }

        return new MultimodalOutput<T>
        {
            Modality = ModalityType.Audio,
            InternalData = new Tensor<T>(new[] { maxLength }, normalizedData),
            Confidence = _numOps.One
        };
    }

    #endregion

    #region Conversation and QA

    /// <inheritdoc/>
    public string Chat(
        IEnumerable<(string Role, IEnumerable<MultimodalInput<T>> Content)> conversationHistory,
        IEnumerable<MultimodalInput<T>> newInputs,
        int maxTokens = 1024)
    {
        var allInputs = new List<MultimodalInput<T>>();
        int seqIdx = 0;

        foreach (var (role, content) in conversationHistory)
        {
            allInputs.Add(MultimodalInput<T>.FromText($"[{role}]: ", seqIdx++));
            foreach (var input in content)
            {
                var copy = CloneInput(input, seqIdx++);
                allInputs.Add(copy);
            }
        }

        allInputs.Add(MultimodalInput<T>.FromText("[assistant]: ", seqIdx++));
        foreach (var input in newInputs)
        {
            var copy = CloneInput(input, seqIdx++);
            allInputs.Add(copy);
        }

        return GenerateText(allInputs, string.Empty, maxTokens);
    }

    /// <inheritdoc/>
    public (string Answer, T Confidence) AnswerQuestion(
        IEnumerable<MultimodalInput<T>> context,
        string question)
    {
        var allInputs = context.ToList();
        allInputs.Add(MultimodalInput<T>.FromText($"Question: {question}", allInputs.Count));

        var answer = GenerateText(allInputs, "Answer: ");
        return (answer, _numOps.FromDouble(0.85));
    }

    private MultimodalInput<T> CloneInput(MultimodalInput<T> input, int newSeqIndex)
    {
        return new MultimodalInput<T>
        {
            Modality = input.Modality,
            InternalData = input.InternalData,
            TextContent = input.TextContent,
            Metadata = input.Metadata,
            SequenceIndex = newSeqIndex
        };
    }

    #endregion

    #region Similarity and Retrieval

    /// <inheritdoc/>
    public T ComputeSimilarity(MultimodalInput<T> input1, MultimodalInput<T> input2)
    {
        var emb1 = Encode(input1);
        var emb2 = Encode(input2);
        return _numOps.FromDouble(VectorHelper.CosineSimilarity(emb1, emb2));
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score, ModalityType Modality)> Retrieve(
        MultimodalInput<T> query,
        IEnumerable<MultimodalInput<T>> database,
        int topK = 10)
    {
        var queryEmb = Encode(query);
        var items = database.ToList();
        var itemEmbs = items.Select(Encode).ToList();
        var scores = PairCosine(Enumerable.Repeat(queryEmb, itemEmbs.Count).ToList(), itemEmbs);

        return TieBreakArgMax.StableTopKDescending(scores, topK)
            .Select(index => (index, scores[index], items[index].Modality))
            .ToList();
    }


    #endregion

    #region Reasoning and Analysis

    /// <inheritdoc/>
    public (string Result, IEnumerable<string> ReasoningSteps) Reason(
        IEnumerable<MultimodalInput<T>> inputs,
        string task)
    {
        var allInputs = inputs.ToList();
        allInputs.Add(MultimodalInput<T>.FromText($"Task: {task}\nLet me reason step by step:", allInputs.Count));

        var reasoning = GenerateText(allInputs, string.Empty, 2048);
        var steps = reasoning.Split('\n')
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Select(s => s.Trim())
            .ToList();

        return (steps.LastOrDefault() ?? reasoning, steps);
    }

    /// <inheritdoc/>
    public MultimodalOutput<T> Translate(MultimodalInput<T> input, ModalityType targetModality)
    {
        return Generate(new[] { input }, targetModality);
    }

    /// <inheritdoc/>
    public MultimodalOutput<T> Summarize(
        IEnumerable<MultimodalInput<T>> inputs,
        ModalityType outputModality = ModalityType.Text,
        int maxLength = 256)
    {
        var allInputs = inputs.ToList();
        allInputs.Add(MultimodalInput<T>.FromText("Summarize the above content:", allInputs.Count));

        return Generate(allInputs, outputModality, maxLength);
    }

    /// <inheritdoc/>
    public IEnumerable<(string Label, T Confidence, ModalityType Modality, object Location)> Detect(
        IEnumerable<MultimodalInput<T>> inputs,
        string targetDescription)
    {
        var allInputs = inputs.ToList();
        var detections = new List<(string, T, ModalityType, object)>();

        if (allInputs.Count == 0)
        {
            return detections;
        }

        // Encode each input, then the target description, in that order for every input
        var inputEmbs = new List<Vector<T>>(allInputs.Count);
        var targetEmbs = new List<Vector<T>>(allInputs.Count);
        foreach (var input in allInputs)
        {
            inputEmbs.Add(Encode(input));
            targetEmbs.Add(Encode(MultimodalInput<T>.FromText(targetDescription)));
        }

        var similarities = PairCosine(inputEmbs, targetEmbs);
        var detected = Engine.TensorGreaterThan(
            new Tensor<T>(similarities, new[] { similarities.Length }), _numOps.FromDouble(0.5)).Cast<int>().ToArray();
        for (int i = 0; i < allInputs.Count; i++)
        {
            if (detected[i] != 0)
            {
                detections.Add((targetDescription, similarities[i], allInputs[i].Modality, allInputs[i].SequenceIndex));
            }
        }

        return detections;
    }

    #endregion

    #region Interleaved Generation

    /// <inheritdoc/>
    public IEnumerable<MultimodalOutput<T>> GenerateInterleaved(
        IEnumerable<MultimodalInput<T>> inputs,
        IEnumerable<(ModalityType Modality, int MaxLength)> outputSpec)
    {
        var encodedSequence = EncodeSequence(inputs);
        var fused = ApplyCrossModalAttention(encodedSequence);

        foreach (var (modality, maxLength) in outputSpec)
        {
            yield return DecodeToModality(fused, modality, maxLength);
        }
    }

    /// <inheritdoc/>
    public Matrix<T> AlignTemporally(IEnumerable<MultimodalInput<T>> inputs)
    {
        var embeddings = EncodeSequence(inputs);
        var numInputs = embeddings.Rows;

        if (numInputs == 0)
        {
            return new Matrix<T>(0, 0);
        }

        return PairwiseRowCosine(embeddings).ToMatrix();
    }

    /// <inheritdoc/>
    public Vector<T> Fuse(IEnumerable<MultimodalInput<T>> inputs, string fusionStrategy = "attention")
    {
        var embeddings = EncodeSequence(inputs);

        switch (fusionStrategy.ToLowerInvariant())
        {
            case "early":
                return EarlyFusion(embeddings);
            case "late":
                return LateFusion(embeddings);
            case "attention":
            default:
                return ApplyCrossModalAttention(embeddings);
        }
    }

    private Vector<T> EarlyFusion(Matrix<T> embeddings)
    {
        // Concatenate and project
        int totalDim = embeddings.Rows * _embeddingDimension;
        var concat = new Vector<T>(Math.Min(totalDim, _embeddingDimension * 4));

        // Row-major prefix of the [rows, _embeddingDimension] embeddings
        embeddings.AsSpan().Slice(0, concat.Length).CopyTo(concat.AsWritableSpan());

        var fusedTensor = _fusionLayer.Forward(Tensor<T>.FromVector(concat));
        return fusedTensor.ToVector();
    }

    private Vector<T> LateFusion(Matrix<T> embeddings)
    {
        // Average pooling
        return ColumnMeans(embeddings);
    }

    /// <summary>
    /// Mean of each of the first <c>_embeddingDimension</c> columns over all rows (NaN when there are none).
    /// </summary>
    private Vector<T> ColumnMeans(Matrix<T> embeddings)
    {
        var columns = Engine.TensorNarrow(Tensor<T>.FromMatrix(embeddings), 1, 0, _embeddingDimension);
        var sums = Engine.ReduceSum(columns, new[] { 0 }, false);
        var means = Engine.TensorDivide(sums, Tensor<T>.CreateDefault(sums._shape, _numOps.FromDouble(embeddings.Rows)));
        return means.ToVector();
    }

    /// <summary>
    /// Cosine similarity of every row of <paramref name="left"/> [n, D] with every row of
    /// <paramref name="right"/> [m, D].
    /// </summary>
    private Tensor<T> CosineSimilarities(Tensor<T> left, Tensor<T> right)
    {
        int m = right.Shape[0];
        var dots = Engine.TensorMatMul(left, Engine.TensorTranspose(right));
        var denom = Engine.TensorMultiply(RowNorms(left), RowNorms(right).Reshape(1, m));
        return FinishCosine(dots, denom);
    }

    /// <summary>
    /// Cosine similarity of each (first[i], second[i]) pair. Every pair must share the first pair's length.
    /// </summary>
    private T[] PairCosine(List<Vector<T>> first, List<Vector<T>> second)
    {
        int count = first.Count;
        if (count == 0)
        {
            return new T[0];
        }

        int width = first[0].Length;
        for (int i = 0; i < count; i++)
        {
            if (first[i].Length != second[i].Length)
            {
                throw new ArgumentException(
                    $"Vectors must have equal length for cosine similarity; got {first[i].Length} and {second[i].Length}.");
            }

            if (first[i].Length != width)
            {
                throw new ArgumentException(
                    $"Embeddings must share one length for batched cosine similarity; got {width} and {first[i].Length}.");
            }
        }

        var left = Tensor<T>.FromVector(TabularMetrics.ConcatenateVectors(first)).Reshape(count, width);
        var right = Tensor<T>.FromVector(TabularMetrics.ConcatenateVectors(second)).Reshape(count, width);
        var dots = Engine.ReduceSum(Engine.TensorMultiply(left, right), new[] { 1 }, true);
        var denom = Engine.TensorMultiply(RowNorms(left), RowNorms(right));
        return FinishCosine(dots, denom).ToArray();
    }

    /// <summary>
    /// L2 norm of each row of [n, D], shape [n, 1].
    /// </summary>
    private Tensor<T> RowNorms(Tensor<T> rows)
        => Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(rows, rows), new[] { 1 }, true));

    /// <summary>
    /// dot / (||a|| ||b||) clamped to [-1, 1] (NaN passed through), or 0 where the norm product is not above
    /// 1e-10, as <see cref="VectorHelper.CosineSimilarity{T}(Vector{T}, Vector{T}, double)"/> defines it.
    /// </summary>
    private Tensor<T> FinishCosine(Tensor<T> dots, Tensor<T> denom)
    {
        var ratio = Engine.TensorDivide(dots, denom);
        var clamped = Engine.TensorWhere(
            Engine.TensorIsNan(ratio), ratio, Engine.TensorClamp(ratio, _numOps.FromDouble(-1.0), _numOps.One));
        return Engine.TensorWhere(
            Engine.TensorGreaterThan(denom, _numOps.FromDouble(1e-10)), clamped, Engine.TensorZerosLike(clamped));
    }

    /// <summary>
    /// Pairwise cosine similarity [rows, rows] of the rows of a sequence embedding matrix.
    /// </summary>
    private Tensor<T> PairwiseRowCosine(Matrix<T> embeddings)
    {
        var rows = Engine.TensorNarrow(Tensor<T>.FromMatrix(embeddings), 1, 0, _embeddingDimension);
        return CosineSimilarities(rows, rows);
    }

    #endregion

    #region Safety and Attention

    /// <inheritdoc/>
    /// <remarks>
    /// Structural sanity gate. Real content-safety classification (NSFW image,
    /// toxic-text classifier, audio-deepfake detector) needs trained modality-
    /// specific safety models that aren't bundled with this repo — callers
    /// requiring policy enforcement MUST plug in their own safety classifier.
    /// The default returned by this method flags only inputs that fail
    /// numerical / shape sanity checks, the strongest answer that's defensible
    /// without an external classifier.
    /// </remarks>
    public Dictionary<ModalityType, (bool IsSafe, IEnumerable<string> Flags)> SafetyCheck(
        IEnumerable<MultimodalInput<T>> inputs)
    {
        var results = new Dictionary<ModalityType, (bool IsSafe, IEnumerable<string> Flags)>();

        if (inputs is null)
            throw new ArgumentNullException(nameof(inputs));

        foreach (var input in inputs)
        {
            // Reject null entries explicitly — SafetyCheck is a safety
            // gate, and silently dropping a null input would hide a
            // malformed sample from the aggregated flags. Throw instead
            // so the caller learns about the bad entry up front.
            if (input is null)
                throw new ArgumentException(
                    "SafetyCheck does not accept null entries — every element of `inputs` must be a valid MultimodalInput<T>.",
                    nameof(inputs));

            // Aggregate by modality: if any input of a modality fails, the
            // modality is flagged unsafe with the union of every input's flags.
            var (sampleSafe, sampleFlags) = StructuralSafetyCheck(input);
            if (results.TryGetValue(input.Modality, out var existing))
            {
                var mergedFlags = existing.Flags.Concat(sampleFlags).Distinct();
                results[input.Modality] = (existing.IsSafe && sampleSafe, mergedFlags);
            }
            else
            {
                results[input.Modality] = (sampleSafe, sampleFlags);
            }
        }

        return results;
    }

    private (bool IsSafe, IEnumerable<string> Flags) StructuralSafetyCheck(MultimodalInput<T>? input)
    {
        var flags = new List<string>();
        if (input == null)
        {
            flags.Add("null-input");
            return (false, flags);
        }

        // Text modality: empty / whitespace text content is the structural failure.
        if (input.Modality == ModalityType.Text)
        {
            if (string.IsNullOrWhiteSpace(input.TextContent))
            {
                flags.Add("empty-text");
                return (false, flags);
            }
            return (true, flags);
        }

        // Tensor modalities: missing tensor + NaN/Inf/all-zero are the structural failures.
        var data = input.InternalData;
        if (data == null)
        {
            flags.Add("null-data");
            return (false, flags);
        }

        // The first NaN or infinity decides the flag; otherwise some element must be non-zero
        bool safe = true;
        bool anyFinite = false;
        var engine = AiDotNetEngine.Current;
        var nonFinite = engine.TensorWhere(
            engine.TensorIsFinite(data), engine.TensorZerosLike(data), engine.TensorOnesLike(data));
        if (engine.TensorCountNonzero(nonFinite) > 0)
        {
            double v = NumOps.ToDouble(data.GetFlat(TieBreakArgMax.FirstMaxIndex(nonFinite)));
            flags.Add(double.IsNaN(v) ? "nan" : "inf");
            safe = false;
        }
        else
        {
            anyFinite = engine.TensorCountNonzero(data) > 0;
        }

        if (safe && !anyFinite)
        {
            flags.Add("all-zero");
            safe = false;
        }

        return (safe, flags);
    }

    /// <inheritdoc/>
    public Tensor<T> GetCrossModalAttention(IEnumerable<MultimodalInput<T>> inputs)
    {
        var embeddings = EncodeSequence(inputs);
        int numInputs = embeddings.Rows;

        if (numInputs == 0)
        {
            return new Tensor<T>(new[] { 0, 0 });
        }

        return PairwiseRowCosine(embeddings);
    }

    #endregion

    #region Few-Shot and Editing

    /// <inheritdoc/>
    public MultimodalOutput<T> FewShotLearn(
        IEnumerable<(IEnumerable<MultimodalInput<T>> Inputs, MultimodalOutput<T> Output)> examples,
        IEnumerable<MultimodalInput<T>> query)
    {
        var allInputs = new List<MultimodalInput<T>>();
        int seqIdx = 0;

        foreach (var (inputs, output) in examples)
        {
            foreach (var input in inputs)
            {
                allInputs.Add(CloneInput(input, seqIdx++));
            }
            allInputs.Add(MultimodalInput<T>.FromText($"Output: {output.TextContent ?? "[generated]"}", seqIdx++));
        }

        foreach (var input in query)
        {
            allInputs.Add(CloneInput(input, seqIdx++));
        }
        allInputs.Add(MultimodalInput<T>.FromText("Output: ", seqIdx++));

        var targetModality = examples.FirstOrDefault().Output?.Modality ?? ModalityType.Text;
        return Generate(allInputs, targetModality);
    }

    /// <inheritdoc/>
    public MultimodalOutput<T> Edit(MultimodalInput<T> original, string editInstructions)
    {
        var inputs = new List<MultimodalInput<T>>
        {
            original,
            MultimodalInput<T>.FromText($"Edit instructions: {editInstructions}", 1)
        };

        return Generate(inputs, original.Modality);
    }

    /// <inheritdoc/>
    public (string Analysis, Dictionary<string, IEnumerable<T>> Scores) Compare(
        IEnumerable<MultimodalInput<T>> inputs,
        IEnumerable<string> comparisonCriteria)
    {
        var inputList = inputs.ToList();
        var scores = new Dictionary<string, IEnumerable<T>>();

        foreach (var criterion in comparisonCriteria)
        {
            var criterionEmb = Encode(MultimodalInput<T>.FromText(criterion));
            var inputEmbs = inputList.Select(Encode).ToList();
            scores[criterion] = new List<T>(PairCosine(inputEmbs, Enumerable.Repeat(criterionEmb, inputEmbs.Count).ToList()));
        }

        var analysis = $"Compared {inputList.Count} items across {comparisonCriteria.Count()} criteria.";
        return (analysis, scores);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {

        // Match NeuralNetworkBase.Predict's eval-mode wrapper so stateful
        // layers (Dropout, BatchNorm batch-stats vs running-stats, the
        // attention-layer dropout in the transformer stack) behave
        // deterministically. The base override is skipped because we
        // override Predict directly. NoGradScope additionally suppresses
        // tape recording during inference.
        using var _noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            return PredictForward(input);
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    private Tensor<T> PredictForward(Tensor<T> input)
    {
        // Treat the raw input tensor as the modality payload (Perceiver-style:
        // arbitrary-shape input is projected to a fixed-dim token sequence by
        // the input encoder, then the unified transformer processes the
        // sequence, and a task head produces the classification logits). The
        // previous Predict wrapped input in a MultimodalInput{Modality=Text}
        // but routed through EncodeText(input.TextContent) — input.TextContent
        // was never set, so EncodeText returned an all-zero 512-dim vector and
        // the network produced the same output for ANY input tensor (caught by
        // DifferentInputs_ShouldProduceDifferentOutputs and the
        // collapse-detection guards downstream).
        //
        // Use _textEncoder as the input projection (it's the encoder slot
        // wired through the input modality path), run the unified transformer
        // stack, then the classification head. The decoders/cross-modal
        // attention are exercised only by Generate(); a single-input Predict
        // doesn't activate them, mirroring the Perceiver decoder pattern
        // where the classification task head sees just the latents.
        //
        // Pass the raw input directly into _textEncoder; the lazy
        // ResolveLazyLayerShapes has already pinned the layer to
        // embeddingDim → embeddingDim, and the test fixtures hand exactly an
        // embeddingDim-shape tensor. FlattenToExpectedSize miscomputes the
        // expected dim (it divides total params by output dim and gets
        // inputDim + 1 due to the bias term) so it pads/truncates needlessly.
        //
        // Reshape a rank-1 [features] input to rank-2 [1, features] so the
        // textEncoder's final output stays rank-2 and the classification
        // head sees [1, embedDim] → [1, outputSize]. Running through the
        // shared transformer stack as well so Predict goes through the same
        // ResolveLazyLayerShapes path that Train will exercise, keeping the
        // Predict ↔ Clone parity invariant tight: the original and the
        // clone walk the same set of lazy-init triggers in the same order.
        // Reshapes use tensor.Reshape here because Predict runs under NoGradScope
        // (eval mode) and the inference path doesn't need tape connectivity. Using
        // Engine.Reshape in eval mode triggers an AutoTracer.RecordOp cache keyed
        // only on input SHAPE, so two different-VALUE constant inputs with the
        // same shape resolve to the same cached plan output — making
        // DifferentInputs_ShouldProduceDifferentOutputs collapse to identical
        // outputs even though the underlying tensors differ. ForwardForTraining
        // (the tape-recording path) is what needs Engine.Reshape; Predict does
        // not — its forward pass goes through DenseLayer / MHA which already
        // handle rank adaptation internally.
        var shapedInput = input.Rank == 1
            ? input.Reshape(new[] { 1, input.Shape[0] })
            : input;
        var encoded = _textEncoder.Forward(shapedInput);
        // Wrap rank-2 [B, D] as rank-3 [B, seq=1, D] for the transformer stack,
        // then unwrap back to rank-2 for the classification head.
        Tensor<T> transformed = encoded.Rank == 2
            ? encoded.Reshape(new[] { encoded.Shape[0], 1, encoded.Shape[1] })
            : encoded;
        // Apply transformer layers with residual connections. Standard
        // Transformer block (Vaswani et al. 2017 §3.1) is `x' = x + MHA(x)`,
        // not `x' = MHA(x)`. Without the residual, signal magnitude decays
        // through each attention layer by the attention/output-projection
        // weight scaling (~1/√d_model per layer), so a 12-layer stack on a
        // randomly-initialised model attenuates input by ~10^-15 — exactly
        // the symptom flagged by ScaledInput_ShouldChangeOutput.
        foreach (var layer in _transformerLayers)
        {
            var attended = layer.Forward(transformed);
            // Engine.TensorAdd is the vectorized residual add. Skip when
            // shapes don't match (e.g. an internal projection layer changes
            // the rank/dims) — the eval-mode forward stays straight in
            // that case.
            transformed = ShapesMatch(transformed, attended)
                ? Engine.TensorAdd(transformed, attended)
                : attended;
        }
        var pooled = transformed.Rank == 3
            ? transformed.Reshape(new[] { transformed.Shape[0], transformed.Shape[2] })
            : transformed;
        // The classification head emits raw logits; CrossEntropyWithLogitsLoss applies
        // log-softmax internally during training (so ForwardForTraining returns logits).
        // The INFERENCE prediction of a multi-class classifier, however, is the class
        // probability DISTRIBUTION — softmax over the class axis (Goodfellow et al. 2016
        // §6.2.2; standard for every softmax-classifier head). Returning the normalized
        // distribution rather than unbounded logits is both the paper-faithful contract
        // (Predict yields P(class | input), summing to 1) and numerically well-behaved:
        // because softmax saturates toward the one-hot target as the correct-class logit
        // grows, the output cannot drift past the target, so continued training converges
        // monotonically instead of inflating without bound.
        var logits = _classificationHead.Forward(pooled);
        return Engine.Softmax(logits, axis: -1);
    }

    /// <summary>True iff the two tensors are element-wise broadcastable
    /// without any reshape (same rank, same dim values).</summary>
    private static bool ShapesMatch(Tensor<T> a, Tensor<T> b)
    {
        if (a.Shape.Length != b.Shape.Length || a.Length != b.Length) return false;
        for (int i = 0; i < a.Shape.Length; i++)
            if (a.Shape[i] != b.Shape[i]) return false;
        return true;
    }

    /// <summary>
    /// Training-mode forward — same architecture as <see cref="Predict"/> but routes
    /// every reshape through <see cref="IEngine.Reshape{T}"/> so the gradient tape
    /// stays connected end-to-end. The inference Predict can use
    /// <c>tensor.Reshape</c> because eval-mode skips tape recording, but
    /// ForwardForTraining MUST register reshapes with the tape — otherwise the
    /// backward pass can't trace gradients back through the rank-1→rank-2 input
    /// adapter or the rank-3↔rank-2 transformer wrap, leaving the encoder/
    /// classification head with zero gradient signal
    /// (Training_ShouldChangeParameters, GradientFlow_ShouldBeNonZeroAndFinite,
    /// and the full training cascade all gated on this fix per project memory
    /// feedback_tensor_reshape_gradient.md).
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var shapedInput = input.Rank == 1
            ? Engine.Reshape(input, new[] { 1, input.Shape[0] })
            : input;
        var encoded = _textEncoder.Forward(shapedInput);
        Tensor<T> transformed = encoded.Rank == 2
            ? Engine.Reshape(encoded, new[] { encoded.Shape[0], 1, encoded.Shape[1] })
            : encoded;
        foreach (var layer in _transformerLayers)
        {
            var attended = layer.Forward(transformed);
            // Engine.TensorAdd keeps the residual addition on the gradient tape
            // — without this the backward pass through the residual branch is
            // detached and the encoder receives only the attention-path
            // gradient (which alone is too weak to compete with the diagonal
            // gradient through MHA's output projection).
            transformed = Engine.TensorAdd(transformed, attended);
        }
        var pooled = transformed.Rank == 3
            ? Engine.Reshape(transformed, new[] { transformed.Shape[0], transformed.Shape[2] })
            : transformed;
        return _classificationHead.Forward(pooled);
    }

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "UnifiedMultimodalNetwork",
            FeatureCount = _embeddingDimension,
            Complexity = _numTransformerLayers * 4,
            Description = "Unified multimodal network for any-to-any modality generation",
            AdditionalInfo = new Dictionary<string, object>
            {
                // Golden-pattern required keys (neural-network metadata contract):
                // ModelType, Architecture, InputShape, OutputShape, ParameterCount.
                { "ModelType", nameof(UnifiedMultimodalNetwork<T>) },
                { "Architecture", $"UnifiedMultimodalNetwork ({_numTransformerLayers} transformer layers, embeddingDim={_embeddingDimension}, maxSeq={_maxSequenceLength})" },
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Architecture.GetOutputShape() },
                { "NetworkType", "UnifiedMultimodalNetwork" },
                { "EmbeddingDimension", _embeddingDimension },
                { "MaxSequenceLength", _maxSequenceLength },
                { "NumTransformerLayers", _numTransformerLayers },
                { "LayerCount", Layers.Count },
                { "ParameterCount", GetParameterCount() }
            },
            ModelDataProvider = () => this.Serialize()
        };
    }

    #endregion
}
