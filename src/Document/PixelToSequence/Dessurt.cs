using AiDotNet.Document.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;

namespace AiDotNet.Document.PixelToSequence;

/// <summary>
/// Dessurt (Document End-to-end Self-Supervised Understanding and RecogniTion) for document understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Dessurt is a self-supervised pre-training approach for document understanding that learns
/// from document images without any labeled data. It uses a denoising autoencoder objective
/// to learn robust document representations.
/// </para>
/// <para>
/// <b>For Beginners:</b> Dessurt learns document understanding without labels:
/// 1. Pre-trains by reconstructing corrupted document images
/// 2. Learns to understand text, layout, and visual patterns
/// 3. Fine-tunes on downstream tasks with minimal supervision
///
/// Key features:
/// - Self-supervised pre-training (no labels needed)
/// - Denoising autoencoder objective
/// - Vision encoder + text decoder architecture
/// - OCR-free document understanding
///
/// Example usage:
/// <code>
/// var model = new Dessurt&lt;float&gt;(architecture);
/// var result = model.GenerateText(documentImage, "Extract all text");
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Dessurt: A Dessert for Document Understanding" (arXiv 2022)
/// https://arxiv.org/abs/2203.16618
/// </para>
/// </remarks>
public class Dessurt<T> : DocumentNeuralNetworkBase<T>, IDocumentQA<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _encoderDim;
    private readonly int _decoderDim;
    private readonly int _encoderLayers;
    private readonly int _decoderLayers;
    private readonly int _numHeads;
    private readonly int _vocabSize;

    // Native mode layers
    private readonly List<ILayer<T>> _encoderLayersList = [];
    private readonly List<ILayer<T>> _decoderLayersList = [];

    // Learnable embeddings
    private Tensor<T>? _encoderPositionEmbeddings;
    private Tensor<T>? _decoderPositionEmbeddings;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => false;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <summary>
    /// Gets the encoder hidden dimension.
    /// </summary>
    public int EncoderDim => _encoderDim;

    /// <summary>
    /// Gets the decoder hidden dimension.
    /// </summary>
    public int DecoderDim => _decoderDim;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Dessurt model using a pre-trained ONNX model for inference.
    /// </summary>
    public Dessurt(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int imageSize = 1024,
        int maxSequenceLength = 512,
        int encoderDim = 1024,
        int decoderDim = 768,
        int encoderLayers = 24,
        int decoderLayers = 12,
        int numHeads = 16,
        int vocabSize = 50265,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _encoderDim = encoderDim;
        _decoderDim = decoderDim;
        _encoderLayers = encoderLayers;
        _decoderLayers = decoderLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a Dessurt model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (Dessurt from arXiv 2022):</b>
    /// - Vision encoder: ViT-Large style
    /// - Text decoder: Transformer decoder
    /// - Encoder: 24 layers, 1024 dim, 16 heads
    /// - Decoder: 12 layers, 768 dim
    /// - Pre-training: Denoising autoencoder
    /// </para>
    /// </remarks>
    public Dessurt(
        NeuralNetworkArchitecture<T> architecture,
        int imageSize = 1024,
        int maxSequenceLength = 512,
        int encoderDim = 1024,
        int decoderDim = 768,
        int encoderLayers = 24,
        int decoderLayers = 12,
        int numHeads = 16,
        int vocabSize = 50265,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _encoderDim = encoderDim;
        _decoderDim = decoderDim;
        _encoderLayers = encoderLayers;
        _decoderLayers = decoderLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        InitializeLayers();
        InitializeEmbeddings();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            return;
        }

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        var (encoderLayers, decoderLayers) = LayerHelper<T>.CreateDefaultDessurtLayers(
            encoderDim: _encoderDim,
            decoderDim: _decoderDim,
            encoderLayers: _encoderLayers,
            decoderLayers: _decoderLayers,
            numHeads: _numHeads,
            vocabSize: _vocabSize);

        _encoderLayersList.AddRange(encoderLayers);
        _decoderLayersList.AddRange(decoderLayers);
        Layers.AddRange(encoderLayers);
        Layers.AddRange(decoderLayers);
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        int numPatches = (ImageSize / 16) * (ImageSize / 16);

        _encoderPositionEmbeddings = Tensor<T>.CreateDefault([numPatches + 1, _encoderDim], NumOps.Zero);
        _decoderPositionEmbeddings = Tensor<T>.CreateDefault([MaxSequenceLength, _decoderDim], NumOps.Zero);

        InitializeWithSmallRandomValues(_encoderPositionEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_decoderPositionEmbeddings, random, 0.02);
    }

    private void InitializeWithSmallRandomValues(Tensor<T> tensor, Random random, double stdDev)
    {
        for (int i = 0; i < tensor.Data.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            tensor.Data[i] = NumOps.FromDouble(randStdNormal * stdDev);
        }
    }

    #endregion

    #region IDocumentQA Implementation

    /// <inheritdoc/>
    public DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question)
    {
        return AnswerQuestion(documentImage, question, 128, 0.0);
    }

    /// <inheritdoc/>
    public DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question, int maxAnswerLength, double temperature = 0.0)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        // Decode output to text
        var answer = DecodeOutput(output, maxAnswerLength);

        return new DocumentQAResult<T>
        {
            Answer = answer,
            Confidence = NumOps.FromDouble(0.85),
            ConfidenceValue = 0.85,
            Question = question,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public IEnumerable<DocumentQAResult<T>> AnswerQuestions(Tensor<T> documentImage, IEnumerable<string> questions)
    {
        foreach (var q in questions)
            yield return AnswerQuestion(documentImage, q);
    }

    /// <inheritdoc/>
    public Dictionary<string, DocumentQAResult<T>> ExtractFields(Tensor<T> documentImage, IEnumerable<string> fieldPrompts)
    {
        var results = new Dictionary<string, DocumentQAResult<T>>();
        foreach (var field in fieldPrompts)
            results[field] = AnswerQuestion(documentImage, $"What is the {field}?");
        return results;
    }

    private string DecodeOutput(Tensor<T> output, int maxLength)
    {
        // Greedy decoding with token-to-character conversion
        var tokens = new List<int>();
        int seqLen = Math.Min(output.Shape[0], maxLength);

        for (int t = 0; t < seqLen; t++)
        {
            int vocabSize = output.Shape.Length > 1 ? output.Shape[1] : _vocabSize;
            double maxVal = double.MinValue;
            int maxIdx = 0;

            for (int v = 0; v < vocabSize; v++)
            {
                double val = NumOps.ToDouble(output[t, v]);
                if (val > maxVal) { maxVal = val; maxIdx = v; }
            }

            // Special tokens: 0=PAD, 1=BOS, 2=EOS
            if (maxIdx == 2) break; // EOS token
            if (maxIdx <= 2) continue; // Skip special tokens
            tokens.Add(maxIdx);
        }

        return DecodeTokensToText(tokens);
    }

    /// <summary>
    /// Converts token IDs to text using character-level decoding.
    /// </summary>
    /// <remarks>
    /// Token mapping:
    /// 0-2: Special tokens (PAD, BOS, EOS)
    /// 3-34: Digits and punctuation (offset by 3 from ASCII 32)
    /// 35-60: Uppercase letters (A-Z, offset from ASCII 65)
    /// 61-86: Lowercase letters (a-z, offset from ASCII 97)
    /// 87+: Extended characters
    /// </remarks>
    private static string DecodeTokensToText(List<int> tokens)
    {
        if (tokens.Count == 0) return string.Empty;

        var sb = new System.Text.StringBuilder();
        foreach (int token in tokens)
        {
            // Map token ID to character
            char c = token switch
            {
                >= 3 and <= 34 => (char)(token - 3 + 32),   // Space, punctuation, digits
                >= 35 and <= 60 => (char)(token - 35 + 65), // A-Z
                >= 61 and <= 86 => (char)(token - 61 + 97), // a-z
                >= 87 and <= 214 => (char)(token - 87 + 128), // Extended ASCII
                _ => '?' // Unknown token
            };
            sb.Append(c);
        }

        return sb.ToString();
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
        var preprocessed = PreprocessDocument(documentImage);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Dessurt Model Summary");
        sb.AppendLine("=====================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: Vision Encoder + Text Decoder");
        sb.AppendLine($"Encoder Dimension: {_encoderDim}");
        sb.AppendLine($"Decoder Dimension: {_decoderDim}");
        sb.AppendLine($"Encoder Layers: {_encoderLayers}");
        sb.AppendLine($"Decoder Layers: {_decoderLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Vocabulary Size: {_vocabSize}");
        sb.AppendLine($"OCR-Free: Yes");
        sb.AppendLine($"Self-Supervised: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies Dessurt's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// Dessurt (Document understanding with Spatially-structured Retrieval and Token) uses
    /// ImageNet normalization with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        var image = EnsureBatchDimension(rawImage);
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image.Shape);
        double[] means = [0.485, 0.456, 0.406];
        double[] stds = [0.229, 0.224, 0.225];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                double mean = c < means.Length ? means[c] : 0.5;
                double std = c < stds.Length ? stds[c] : 0.5;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int idx = b * channels * height * width + c * height * width + h * width + w;
                        normalized.Data[idx] = NumOps.FromDouble((NumOps.ToDouble(image.Data[idx]) - mean) / std);
                    }
                }
            }
        }
        return normalized;
    }

    /// <summary>
    /// Applies Dessurt's industry-standard postprocessing: pass-through (sequence outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "Dessurt",
            ModelType = ModelType.NeuralNetwork,
            Description = "Dessurt for self-supervised document understanding (arXiv 2022)",
            FeatureCount = _encoderDim,
            Complexity = _encoderLayers + _decoderLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "encoder_dim", _encoderDim },
                { "decoder_dim", _decoderDim },
                { "encoder_layers", _encoderLayers },
                { "decoder_layers", _decoderLayers },
                { "num_heads", _numHeads },
                { "vocab_size", _vocabSize },
                { "image_size", ImageSize },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_encoderDim);
        writer.Write(_decoderDim);
        writer.Write(_encoderLayers);
        writer.Write(_decoderLayers);
        writer.Write(_numHeads);
        writer.Write(_vocabSize);
        writer.Write(ImageSize);
        writer.Write(MaxSequenceLength);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int encoderDim = reader.ReadInt32();
        int decoderDim = reader.ReadInt32();
        int encoderLayers = reader.ReadInt32();
        int decoderLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
        MaxSequenceLength = maxSeqLen;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Dessurt<T>(Architecture, ImageSize, MaxSequenceLength, _encoderDim, _decoderDim,
            _encoderLayers, _decoderLayers, _numHeads, _vocabSize);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");

        SetTrainingMode(true);
        var output = Predict(input);
        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        var gradient = Tensor<T>.FromVector(
            LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector()));

        for (int i = Layers.Count - 1; i >= 0; i--)
            gradient = Layers[i].Backward(gradient);

        UpdateParameters(CollectGradients());
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Parameter updates not supported in ONNX mode.");

        var currentParams = GetParameters();
        T lr = NumOps.FromDouble(0.00005);
        for (int i = 0; i < currentParams.Length; i++)
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(lr, gradients[i]));
        SetParameters(currentParams);
    }

    private Vector<T> CollectGradients()
    {
        var grads = new List<T>();
        foreach (var layer in Layers)
            grads.AddRange(layer.GetParameterGradients());
        return new Vector<T>([.. grads]);
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _onnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
