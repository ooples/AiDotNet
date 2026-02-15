using System.Text.RegularExpressions;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Postprocessing.Document;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;

namespace AiDotNet.Document.PixelToSequence;

/// <summary>
/// Nougat neural network for academic document understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Nougat (Neural Optical Understanding for Academic Documents) is an OCR-free
/// transformer model specifically designed to parse academic papers from PDF images
/// into structured Markdown format with mathematical notation support.
/// </para>
/// <para>
/// <b>For Beginners:</b> Nougat excels at:
/// 1. Converting PDF pages to Markdown
/// 2. Preserving mathematical equations (LaTeX)
/// 3. Understanding document structure (sections, tables, figures)
/// 4. Processing scientific papers without OCR
///
/// Example usage:
/// <code>
/// var model = new Nougat&lt;float&gt;(architecture);
/// var markdown = model.ParseAcademicDocument(pdfPageImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Nougat: Neural Optical Understanding for Academic Documents" (arXiv 2023)
/// https://arxiv.org/abs/2308.13418
/// </para>
/// </remarks>
public class Nougat<T> : DocumentNeuralNetworkBase<T>, IDocumentQA<T>
{
    private readonly NougatOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly ITokenizer _tokenizer;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private int _hiddenDim;
    private int _numEncoderLayers;
    private int _numDecoderLayers;
    private int _numHeads;
    private int _vocabSize;
    private int _patchSize;

    // Native mode layers
    private readonly List<ILayer<T>> _encoderLayers = [];
    private readonly List<ILayer<T>> _decoderLayers = [];

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => false; // OCR-free model

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <summary>
    /// Gets whether this model supports LaTeX equation output.
    /// </summary>
    public bool SupportsLatex => true;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Nougat model using a pre-trained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="tokenizer">Tokenizer for text processing.</param>
    /// <param name="imageSize">Input image size (default: 896).</param>
    /// <param name="patchSize">Patch size for vision encoder (default: 16).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 4096).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 1024).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 12).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 10).</param>
    /// <param name="numHeads">Number of attention heads (default: 16).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50000).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    public Nougat(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        int imageSize = 896,
        int patchSize = 16,
        int maxSequenceLength = 4096,
        int hiddenDim = 1024,
        int numEncoderLayers = 12,
        int numDecoderLayers = 10,
        int numHeads = 16,
        int vocabSize = 50000,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        NougatOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new NougatOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        Guard.NotNull(tokenizer);
        _tokenizer = tokenizer;
        _useNativeMode = false;
        _hiddenDim = hiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _patchSize = patchSize;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a Nougat model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="tokenizer">Tokenizer for text processing (optional).</param>
    /// <param name="imageSize">Input image size (default: 896).</param>
    /// <param name="patchSize">Patch size for vision encoder (default: 16).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 4096).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 1024).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 12).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 10).</param>
    /// <param name="numHeads">Number of attention heads (default: 16).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50000).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (Nougat-Base from arXiv 2023):</b>
    /// - Swin Transformer encoder
    /// - mBART-style decoder
    /// - Image size: 896x896
    /// - Patch size: 16x16
    /// - Hidden dimension: 1024
    /// - Encoder layers: 12, Decoder layers: 10
    /// - Supports LaTeX equations and Markdown output
    /// </para>
    /// </remarks>
    public Nougat(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer? tokenizer = null,
        int imageSize = 896,
        int patchSize = 16,
        int maxSequenceLength = 4096,
        int hiddenDim = 1024,
        int numEncoderLayers = 12,
        int numDecoderLayers = 10,
        int numHeads = 16,
        int vocabSize = 50000,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        NougatOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new NougatOptions();
        Options = _options;

        _useNativeMode = true;
        _hiddenDim = hiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _patchSize = patchSize;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);

        InitializeLayers();
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

        var (encoderLayers, decoderLayers) = LayerHelper<T>.CreateDefaultNougatLayers(
            hiddenDim: _hiddenDim,
            numEncoderLayers: _numEncoderLayers,
            numDecoderLayers: _numDecoderLayers,
            numHeads: _numHeads,
            vocabSize: _vocabSize,
            imageSize: ImageSize,
            patchSize: _patchSize,
            maxSequenceLength: MaxSequenceLength);

        _encoderLayers.AddRange(encoderLayers);
        _decoderLayers.AddRange(decoderLayers);

        Layers.AddRange(_encoderLayers);
        Layers.AddRange(_decoderLayers);
    }

    #endregion

    #region IDocumentQA Implementation

    /// <inheritdoc/>
    public DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question)
    {
        return AnswerQuestion(documentImage, question, MaxSequenceLength, 0.0);
    }

    /// <inheritdoc/>
    public DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question, int maxAnswerLength, double temperature = 0.0)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        if (!IsMarkdownPrompt(question))
        {
            throw new NotSupportedException(
                "Nougat currently supports Markdown conversion prompts only. Use ParseAcademicDocument or a Markdown prompt.");
        }

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var (markdown, confidence) = DecodeToMarkdown(output, maxAnswerLength, temperature);

        return new DocumentQAResult<T>
        {
            Answer = markdown,
            Confidence = NumOps.FromDouble(confidence),
            ConfidenceValue = confidence,
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
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var markdown = ParseAcademicDocument(documentImage);
        using var parser = new StructuredOutputParser<T>();
        var kvPairs = parser.ParseKeyValuePairs(markdown);

        var results = new Dictionary<string, DocumentQAResult<T>>();
        foreach (var field in fieldPrompts)
        {
            kvPairs.TryGetValue(field, out var value);
            double confidence = string.IsNullOrWhiteSpace(value) ? 0.0 : 0.5;
            results[field] = new DocumentQAResult<T>
            {
                Answer = value ?? string.Empty,
                Confidence = NumOps.FromDouble(confidence),
                ConfidenceValue = confidence,
                Question = field,
                ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
            };
        }

        return results;
    }

    private (string markdown, double confidence) DecodeToMarkdown(Tensor<T> output, int maxLength, double temperature)
    {
        if (output.Shape.Length == 0 || maxLength <= 0)
            return (string.Empty, 0.0);

        int cappedMaxLength = Math.Min(maxLength, MaxSequenceLength);
        if (cappedMaxLength <= 0)
            return (string.Empty, 0.0);

        string eosToken = _tokenizer.SpecialTokens.EosToken;
        bool hasEosToken = !string.IsNullOrWhiteSpace(eosToken);

        var tokens = new List<int>();
        double totalConfidence = 0.0;
        int confidenceSteps = 0;

        if (output.Shape.Length == 1)
        {
            int length = Math.Min(cappedMaxLength, output.Shape[0]);
            for (int i = 0; i < length; i++)
            {
                int tokenId = Convert.ToInt32(NumOps.ToDouble(output[i]));
                if (hasEosToken && _tokenizer.Vocabulary.GetToken(tokenId) == eosToken)
                    break;

                tokens.Add(tokenId);
            }

            string text = tokens.Count == 0 ? string.Empty : _tokenizer.Decode(tokens);
            return (text, 0.0);
        }

        double clampedTemperature = Math.Max(0.0, Math.Min(temperature, 2.0));
        bool useSampling = clampedTemperature > 0.0;
        if (useSampling)
            clampedTemperature = Math.Max(clampedTemperature, 0.01);

        int rank = output.Shape.Length;
        int seqDim = rank - 2;
        int vocabDim = rank - 1;
        int seqLength = output.Shape[seqDim];
        int vocabSize = output.Shape[vocabDim];
        int maxSteps = Math.Min(cappedMaxLength, seqLength);
        if (maxSteps <= 0 || vocabSize <= 0)
            return (string.Empty, 0.0);

        int[] indices = new int[rank];
        var random = RandomHelper.Shared;

        for (int i = 0; i < maxSteps; i++)
        {
            indices[seqDim] = i;
            int tokenId;
            double tokenProb;

            if (useSampling)
            {
                tokenId = SampleToken(output, indices, vocabDim, vocabSize, clampedTemperature, random, out tokenProb);
            }
            else
            {
                tokenId = SelectGreedyToken(output, indices, vocabDim, vocabSize, out tokenProb);
            }

            if (hasEosToken && _tokenizer.Vocabulary.GetToken(tokenId) == eosToken)
                break;

            tokens.Add(tokenId);
            if (tokenProb > 0)
            {
                totalConfidence += tokenProb;
                confidenceSteps++;
            }
        }

        string decoded = tokens.Count == 0 ? string.Empty : _tokenizer.Decode(tokens);
        double confidence = confidenceSteps > 0 ? totalConfidence / confidenceSteps : 0.0;
        return (decoded, confidence);
    }

    private static bool IsMarkdownPrompt(string question)
    {
        if (string.IsNullOrWhiteSpace(question))
            return true;

        return question.Contains("markdown", StringComparison.OrdinalIgnoreCase)
            || question.Contains("convert", StringComparison.OrdinalIgnoreCase)
            || question.Contains("parse", StringComparison.OrdinalIgnoreCase);
    }

    private int SelectGreedyToken(Tensor<T> output, int[] indices, int vocabDim, int vocabSize, out double probability)
    {
        double maxLogit = double.MinValue;
        int maxIdx = 0;
        for (int v = 0; v < vocabSize; v++)
        {
            indices[vocabDim] = v;
            double logit = NumOps.ToDouble(output[indices]);
            if (logit > maxLogit)
            {
                maxLogit = logit;
                maxIdx = v;
            }
        }

        double sumExp = 0;
        for (int v = 0; v < vocabSize; v++)
        {
            indices[vocabDim] = v;
            double logit = NumOps.ToDouble(output[indices]);
            sumExp += Math.Exp(logit - maxLogit);
        }

        probability = sumExp > 0 ? 1.0 / sumExp : 0.0;
        return maxIdx;
    }

    private int SampleToken(Tensor<T> output, int[] indices, int vocabDim, int vocabSize, double temperature, Random random, out double probability)
    {
        double maxLogit = double.MinValue;
        for (int v = 0; v < vocabSize; v++)
        {
            indices[vocabDim] = v;
            double logit = NumOps.ToDouble(output[indices]) / temperature;
            if (logit > maxLogit)
            {
                maxLogit = logit;
            }
        }

        double sumExp = 0;
        for (int v = 0; v < vocabSize; v++)
        {
            indices[vocabDim] = v;
            double logit = NumOps.ToDouble(output[indices]) / temperature;
            sumExp += Math.Exp(logit - maxLogit);
        }

        if (sumExp <= 0)
        {
            probability = 0.0;
            return 0;
        }

        double sample = random.NextDouble() * sumExp;
        double cumulative = 0;
        int selected = vocabSize - 1;
        double selectedScore = 0;

        for (int v = 0; v < vocabSize; v++)
        {
            indices[vocabDim] = v;
            double logit = NumOps.ToDouble(output[indices]) / temperature;
            double score = Math.Exp(logit - maxLogit);
            cumulative += score;
            if (sample <= cumulative)
            {
                selected = v;
                selectedScore = score;
                break;
            }
        }

        probability = selectedScore / sumExp;
        return selected;
    }

    /// <summary>
    /// Parses an academic document page and generates Markdown output.
    /// </summary>
    /// <param name="pageImage">The document page image tensor.</param>
    /// <returns>Markdown-formatted text with LaTeX equations.</returns>
    public string ParseAcademicDocument(Tensor<T> pageImage)
    {
        var result = AnswerQuestion(pageImage, "Convert to Markdown");
        return result.Answer ?? string.Empty;
    }

    /// <summary>
    /// Parses a PDF page and extracts equations in LaTeX format.
    /// </summary>
    /// <param name="pageImage">The PDF page image tensor.</param>
    /// <returns>List of extracted LaTeX equations.</returns>
    public IEnumerable<string> ExtractEquations(Tensor<T> pageImage)
    {
        var markdown = ParseAcademicDocument(pageImage);
        if (string.IsNullOrWhiteSpace(markdown))
            yield break;

        foreach (Match match in RegexHelper.Matches(markdown, @"(?<!\\)(\$\$|\$)(.+?)(?<!\\)\1", RegexOptions.Singleline))
        {
            string equation = match.Value.Trim();
            if (equation.Length > 0)
                yield return equation;
        }
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
        sb.AppendLine("Nougat Model Summary");
        sb.AppendLine("====================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: Swin Transformer encoder + mBART decoder");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Encoder Layers: {_numEncoderLayers}");
        sb.AppendLine($"Decoder Layers: {_numDecoderLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Patch Size: {_patchSize}x{_patchSize}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"OCR-Free: Yes");
        sb.AppendLine($"LaTeX Support: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies Nougat's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// Nougat (Neural Optical Understanding for Academic documents) uses ImageNet normalization
    /// with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] (Meta paper).
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        var image = EnsureBatchDimension(rawImage);
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        // Normalize with ImageNet stats
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
                        normalized.Data.Span[idx] = NumOps.FromDouble((NumOps.ToDouble(image.Data.Span[idx]) - mean) / std);
                    }
                }
            }
        }
        return normalized;
    }

    /// <summary>
    /// Applies Nougat's industry-standard postprocessing: pass-through (Markdown outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "Nougat",
            ModelType = ModelType.NeuralNetwork,
            Description = "Nougat for academic document understanding (arXiv 2023)",
            FeatureCount = _hiddenDim,
            Complexity = _numEncoderLayers + _numDecoderLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_encoder_layers", _numEncoderLayers },
                { "num_decoder_layers", _numDecoderLayers },
                { "num_heads", _numHeads },
                { "patch_size", _patchSize },
                { "image_size", ImageSize },
                { "vocab_size", _vocabSize },
                { "supports_latex", SupportsLatex },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numEncoderLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_numHeads);
        writer.Write(_vocabSize);
        writer.Write(ImageSize);
        writer.Write(MaxSequenceLength);
        writer.Write(_patchSize);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numEncoderLayers = reader.ReadInt32();
        int numDecoderLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        int patchSize = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        _hiddenDim = hiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _patchSize = patchSize;
        _useNativeMode = useNativeMode;

        ImageSize = imageSize;
        MaxSequenceLength = maxSeqLen;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Nougat<T>(Architecture, _tokenizer, ImageSize, _patchSize, MaxSequenceLength,
            _hiddenDim, _numEncoderLayers, _numDecoderLayers, _numHeads, _vocabSize);
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
        T lr = NumOps.FromDouble(0.0001);
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



