using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Unified multimodal network that handles text, images, audio, and video
/// in a single architecture with cross-modal attention and any-to-any generation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class UnifiedMultimodalNetwork<T> : NeuralNetworkBase<T>, IUnifiedMultimodalModel<T>
{
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
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
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
    /// Initializes a new instance of the UnifiedMultimodalNetwork.
    /// </summary>
    public UnifiedMultimodalNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int embeddingDimension = DEFAULT_EMBEDDING_DIM,
        int maxSequenceLength = DEFAULT_MAX_SEQ_LEN,
        int numTransformerLayers = DEFAULT_NUM_LAYERS,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _numTransformerLayers = numTransformerLayers;
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSeededRandom(42);
        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
        _optimizer = optimizer ?? new Optimizers.AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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

        // Initialize null references
        _textEncoder = null!;
        _imageEncoder = null!;
        _audioEncoder = null!;
        _videoEncoder = null!;
        _transformerLayers = null!;
        _crossModalAttention = null!;
        _textDecoder = null!;
        _imageDecoder = null!;
        _audioDecoder = null!;
        _videoDecoder = null!;
        _fusionLayer = null!;
        _classificationHead = null!;
        _generationHead = null!;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        IActivationFunction<T>? nullActivation = null;
        var geluActivation = new GELUActivation<T>() as IActivationFunction<T>;

        // Modality encoders - project to unified embedding space
        _textEncoder = new DenseLayer<T>(512, _embeddingDimension, geluActivation);
        _imageEncoder = new DenseLayer<T>(768, _embeddingDimension, geluActivation);
        _audioEncoder = new DenseLayer<T>(128, _embeddingDimension, geluActivation);
        _videoEncoder = new DenseLayer<T>(1024, _embeddingDimension, geluActivation);

        // Unified transformer layers
        _transformerLayers = new MultiHeadAttentionLayer<T>[_numTransformerLayers];
        for (int i = 0; i < _numTransformerLayers; i++)
        {
            _transformerLayers[i] = new MultiHeadAttentionLayer<T>(
                _embeddingDimension, 12, _embeddingDimension / 12, geluActivation);
        }

        // Cross-modal attention
        _crossModalAttention = new MultiHeadAttentionLayer<T>[4];
        for (int i = 0; i < 4; i++)
        {
            _crossModalAttention[i] = new MultiHeadAttentionLayer<T>(
                _embeddingDimension, 8, _embeddingDimension / 8, geluActivation);
        }

        // Modality decoders
        _textDecoder = new DenseLayer<T>(_embeddingDimension, 50000, nullActivation); // vocab size
        _imageDecoder = new DenseLayer<T>(_embeddingDimension, 768 * 3, nullActivation); // patches * channels
        _audioDecoder = new DenseLayer<T>(_embeddingDimension, 16000, nullActivation); // sample rate
        _videoDecoder = new DenseLayer<T>(_embeddingDimension, 768 * 3 * 8, nullActivation); // frames

        // Fusion and output
        _fusionLayer = new DenseLayer<T>(_embeddingDimension * 4, _embeddingDimension, geluActivation);
        _classificationHead = new DenseLayer<T>(_embeddingDimension, 1000, nullActivation);
        _generationHead = new DenseLayer<T>(_embeddingDimension, _embeddingDimension, geluActivation);
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
        for (int i = 0; i < embeddings.Count; i++)
        {
            for (int j = 0; j < _embeddingDimension && j < embeddings[i].Length; j++)
            {
                matrix[i, j] = embeddings[i][j];
            }
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
                inputTensor = input.InternalData ?? new Tensor<T>(new[] { 768 });
                encoder = _imageEncoder;
                break;
            case ModalityType.Audio:
                inputTensor = input.InternalData ?? new Tensor<T>(new[] { 128 });
                encoder = _audioEncoder;
                break;
            case ModalityType.Video:
                inputTensor = input.InternalData ?? new Tensor<T>(new[] { 1024 });
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
        var result = new Vector<T>(512);
        var chars = text.ToCharArray();

        for (int i = 0; i < Math.Min(chars.Length, 512); i++)
        {
            result[i] = _numOps.FromDouble(chars[i] / 128.0);
        }

        return Tensor<T>.FromVector(result);
    }

    private Tensor<T> FlattenToExpectedSize(Tensor<T> input, DenseLayer<T> layer)
    {
        var inputData = input.ToVector();
        int expectedSize = GetLayerInputSize(layer);
        var result = new Vector<T>(expectedSize);

        for (int i = 0; i < Math.Min(inputData.Length, expectedSize); i++)
        {
            result[i] = inputData[i];
        }

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
        var pooled = new Vector<T>(_embeddingDimension);
        for (int j = 0; j < _embeddingDimension; j++)
        {
            T sum = _numOps.Zero;
            for (int i = 0; i < embeddings.Rows; i++)
            {
                sum = _numOps.Add(sum, embeddings[i, j]);
            }
            pooled[j] = _numOps.Divide(sum, _numOps.FromDouble(embeddings.Rows));
        }

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

        // Simple greedy decoding
        var text = new System.Text.StringBuilder();
        for (int i = 0; i < Math.Min(maxLength, logitsData.Length / 256); i++)
        {
            int maxIdx = 0;
            T maxVal = logitsData[i * 256];
            for (int j = 1; j < 256 && (i * 256 + j) < logitsData.Length; j++)
            {
                if (_numOps.Compare(logitsData[i * 256 + j], maxVal) > 0)
                {
                    maxVal = logitsData[i * 256 + j];
                    maxIdx = j;
                }
            }
            if (maxIdx >= 32 && maxIdx < 127)
            {
                text.Append((char)maxIdx);
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

        for (int i = 0; i < imageData.Length && i < decodedData.Length; i++)
        {
            imageData[i] = MathHelper.Sigmoid(decodedData[i]);
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

        // Normalize to [-1, 1] range
        var normalizedData = new Vector<T>(maxLength);
        for (int i = 0; i < maxLength && i < audioData.Length; i++)
        {
            normalizedData[i] = _numOps.Multiply(
                _numOps.Subtract(_numOps.Multiply(MathHelper.Sigmoid(audioData[i]), _numOps.FromDouble(2.0)),
                _numOps.One), _numOps.One);
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
        return CosineSimilarity(emb1, emb2);
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score, ModalityType Modality)> Retrieve(
        MultimodalInput<T> query,
        IEnumerable<MultimodalInput<T>> database,
        int topK = 10)
    {
        var queryEmb = Encode(query);
        var results = new List<(int Index, T Score, ModalityType Modality)>();
        int idx = 0;

        foreach (var item in database)
        {
            var itemEmb = Encode(item);
            var score = CosineSimilarity(queryEmb, itemEmb);
            results.Add((idx, score, item.Modality));
            idx++;
        }

        return results.OrderByDescending(r => _numOps.ToDouble(r.Score)).Take(topK);
    }

    private T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        int minLen = Math.Min(a.Length, b.Length);
        T dot = _numOps.Zero;
        T normA = _numOps.Zero;
        T normB = _numOps.Zero;

        for (int i = 0; i < minLen; i++)
        {
            dot = _numOps.Add(dot, _numOps.Multiply(a[i], b[i]));
            normA = _numOps.Add(normA, _numOps.Multiply(a[i], a[i]));
            normB = _numOps.Add(normB, _numOps.Multiply(b[i], b[i]));
        }

        var denom = _numOps.Multiply(_numOps.Sqrt(normA), _numOps.Sqrt(normB));
        if (_numOps.ToDouble(denom) < 1e-8) return _numOps.Zero;

        return _numOps.Divide(dot, denom);
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

        foreach (var input in allInputs)
        {
            var embedding = Encode(input);
            var targetEmb = Encode(MultimodalInput<T>.FromText(targetDescription));
            var similarity = CosineSimilarity(embedding, targetEmb);

            if (_numOps.ToDouble(similarity) > 0.5)
            {
                detections.Add((targetDescription, similarity, input.Modality, input.SequenceIndex));
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

        var alignment = new Matrix<T>(numInputs, numInputs);

        for (int i = 0; i < numInputs; i++)
        {
            for (int j = 0; j < numInputs; j++)
            {
                var vecI = new Vector<T>(_embeddingDimension);
                var vecJ = new Vector<T>(_embeddingDimension);

                for (int k = 0; k < _embeddingDimension; k++)
                {
                    vecI[k] = embeddings[i, k];
                    vecJ[k] = embeddings[j, k];
                }

                alignment[i, j] = CosineSimilarity(vecI, vecJ);
            }
        }

        return alignment;
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

        int idx = 0;
        for (int i = 0; i < embeddings.Rows && idx < concat.Length; i++)
        {
            for (int j = 0; j < _embeddingDimension && idx < concat.Length; j++)
            {
                concat[idx++] = embeddings[i, j];
            }
        }

        var fusedTensor = _fusionLayer.Forward(Tensor<T>.FromVector(concat));
        return fusedTensor.ToVector();
    }

    private Vector<T> LateFusion(Matrix<T> embeddings)
    {
        // Average pooling
        var result = new Vector<T>(_embeddingDimension);

        for (int j = 0; j < _embeddingDimension; j++)
        {
            T sum = _numOps.Zero;
            for (int i = 0; i < embeddings.Rows; i++)
            {
                sum = _numOps.Add(sum, embeddings[i, j]);
            }
            result[j] = _numOps.Divide(sum, _numOps.FromDouble(embeddings.Rows));
        }

        return result;
    }

    #endregion

    #region Safety and Attention

    /// <inheritdoc/>
    public Dictionary<ModalityType, (bool IsSafe, IEnumerable<string> Flags)> SafetyCheck(
        IEnumerable<MultimodalInput<T>> inputs)
    {
        var results = new Dictionary<ModalityType, (bool IsSafe, IEnumerable<string> Flags)>();

        foreach (var input in inputs)
        {
            if (!results.ContainsKey(input.Modality))
            {
                // Placeholder safety check - all inputs considered safe
                results[input.Modality] = (true, Enumerable.Empty<string>());
            }
        }

        return results;
    }

    /// <inheritdoc/>
    public Tensor<T> GetCrossModalAttention(IEnumerable<MultimodalInput<T>> inputs)
    {
        var embeddings = EncodeSequence(inputs);
        int numInputs = embeddings.Rows;

        var attentionData = new Vector<T>(numInputs * numInputs);
        int idx = 0;

        for (int i = 0; i < numInputs; i++)
        {
            for (int j = 0; j < numInputs; j++)
            {
                var vecI = new Vector<T>(_embeddingDimension);
                var vecJ = new Vector<T>(_embeddingDimension);

                for (int k = 0; k < _embeddingDimension; k++)
                {
                    vecI[k] = embeddings[i, k];
                    vecJ[k] = embeddings[j, k];
                }

                attentionData[idx++] = CosineSimilarity(vecI, vecJ);
            }
        }

        return new Tensor<T>(new[] { numInputs, numInputs }, attentionData);
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
            var criterionScores = new List<T>();
            var criterionEmb = Encode(MultimodalInput<T>.FromText(criterion));

            foreach (var input in inputList)
            {
                var inputEmb = Encode(input);
                criterionScores.Add(CosineSimilarity(inputEmb, criterionEmb));
            }

            scores[criterion] = criterionScores;
        }

        var analysis = $"Compared {inputList.Count} items across {comparisonCriteria.Count()} criteria.";
        return (analysis, scores);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var inputVec = input.ToVector();
        var multimodalInput = new MultimodalInput<T>
        {
            Modality = ModalityType.Text,
            InternalData = input
        };

        var embedding = Encode(multimodalInput);
        return Tensor<T>.FromVector(embedding);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        var prediction = Predict(input);
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        SetParameters(gradients);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "UnifiedMultimodalNetwork",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _embeddingDimension,
            Complexity = _numTransformerLayers * 4,
            Description = "Unified multimodal network for any-to-any modality generation"
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embeddingDimension);
        writer.Write(_maxSequenceLength);
        writer.Write(_numTransformerLayers);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new UnifiedMultimodalNetwork<T>(
            Architecture,
            _embeddingDimension,
            _maxSequenceLength,
            _numTransformerLayers);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        void AddLayerParams(ILayer<T> layer)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++)
            {
                allParams.Add(p[i]);
            }
        }

        AddLayerParams(_textEncoder);
        AddLayerParams(_imageEncoder);
        AddLayerParams(_audioEncoder);
        AddLayerParams(_videoEncoder);

        foreach (var layer in _transformerLayers) AddLayerParams(layer);
        foreach (var layer in _crossModalAttention) AddLayerParams(layer);

        AddLayerParams(_textDecoder);
        AddLayerParams(_imageDecoder);
        AddLayerParams(_audioDecoder);
        AddLayerParams(_videoDecoder);

        AddLayerParams(_fusionLayer);
        AddLayerParams(_classificationHead);
        AddLayerParams(_generationHead);

        var result = new Vector<T>(allParams.Count);
        for (int i = 0; i < allParams.Count; i++)
        {
            result[i] = allParams[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var offset = 0;

        void SetLayerParams(ILayer<T> layer)
        {
            var count = layer.ParameterCount;
            var p = new Vector<T>(count);
            for (int i = 0; i < count; i++)
            {
                if (offset + i < parameters.Length)
                {
                    p[i] = parameters[offset + i];
                }
            }
            layer.SetParameters(p);
            offset += count;
        }

        SetLayerParams(_textEncoder);
        SetLayerParams(_imageEncoder);
        SetLayerParams(_audioEncoder);
        SetLayerParams(_videoEncoder);

        foreach (var layer in _transformerLayers) SetLayerParams(layer);
        foreach (var layer in _crossModalAttention) SetLayerParams(layer);

        SetLayerParams(_textDecoder);
        SetLayerParams(_imageDecoder);
        SetLayerParams(_audioDecoder);
        SetLayerParams(_videoDecoder);

        SetLayerParams(_fusionLayer);
        SetLayerParams(_classificationHead);
        SetLayerParams(_generationHead);
    }

    /// <inheritdoc/>
    public override int ParameterCount
    {
        get
        {
            var count = 0;

            count += _textEncoder.ParameterCount;
            count += _imageEncoder.ParameterCount;
            count += _audioEncoder.ParameterCount;
            count += _videoEncoder.ParameterCount;

            foreach (var layer in _transformerLayers) count += layer.ParameterCount;
            foreach (var layer in _crossModalAttention) count += layer.ParameterCount;

            count += _textDecoder.ParameterCount;
            count += _imageDecoder.ParameterCount;
            count += _audioDecoder.ParameterCount;
            count += _videoDecoder.ParameterCount;

            count += _fusionLayer.ParameterCount;
            count += _classificationHead.ParameterCount;
            count += _generationHead.ParameterCount;

            return count;
        }
    }

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var copy = new UnifiedMultimodalNetwork<T>(
            Architecture,
            _embeddingDimension,
            _maxSequenceLength,
            _numTransformerLayers,
            _optimizer,
            _lossFunction);

        copy.SetParameters(GetParameters());
        return copy;
    }

    #endregion
}
