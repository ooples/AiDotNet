using AiDotNet.LearningRateSchedulers;
using System.Text.RegularExpressions;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.Options;
using AiDotNet.Attributes;
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
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Nougat: Neural Optical Understanding for Academic Documents", "https://doi.org/10.48550/arXiv.2308.13418", Year = 2023, Authors = "Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, ReferenceBatchSize = 192,
                MinLearningRate = 7.5e-6, DecayRate = 0.9996, StepSize = 15,
                Schedule = LearningRateSchedulerType.Exponential,
                Source = "Blecher et al. 2023, Sec. 4: an AdamW optimizer trained for 3 epochs at an "
                        + "effective batch size of 192, with an initial learning rate of 5e-5 reduced by "
                        + "a factor of 0.9996 every 15 updates until it reaches 7.5e-6. This model "
                        + "defaults to Adam; the paper's AdamW is the stronger of the two, so the "
                        + "declaration is applied rather than only verified.")]
public partial class Nougat<T> : DocumentNeuralNetworkBase<T>, IDocumentQA<T>
{
    private readonly NougatOptions _options;

    #region Fields

    private bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly ITokenizer _tokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private int _hiddenDim;
    private int _numEncoderLayers;
    private int _numDecoderLayers;
    private int _numHeads;
    private int _vocabSize;
    private int _patchSize;

    private bool _nativeLayersInitialized;

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
        NougatOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
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
        // Validated after the path checks so a missing model file reports itself
        // as FileNotFoundException rather than being pre-empted by the options.
        _options.Validate();

        _hiddenDim = _options.HiddenDim;
        _numEncoderLayers = _options.NumEncoderLayers;
        _numDecoderLayers = _options.NumDecoderLayers;
        _numHeads = _options.NumHeads;
        _vocabSize = _options.VocabSize;
        _patchSize = _options.PatchSize;
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = _options.ImageSize;
        MaxSequenceLength = _options.MaxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

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
        NougatOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new NougatOptions();
        _options.Validate();
        Options = _options;

        _useNativeMode = true;
        _hiddenDim = _options.HiddenDim;
        _numEncoderLayers = _options.NumEncoderLayers;
        _numDecoderLayers = _options.NumDecoderLayers;
        _numHeads = _options.NumHeads;
        _vocabSize = _options.VocabSize;
        _patchSize = _options.PatchSize;
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = _options.ImageSize;
        MaxSequenceLength = _options.MaxSequenceLength;

        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);

        // Native layers are materialized on first use to avoid constructor-time
        // allocation for metadata and construction probes.
        if (Architecture.Layers is { Count: > 0 })
        {
            EnsureNativeInitialized();
        }
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

        Layers.AddRange(LayerHelper<T>.CreateDefaultNougatLayers(
            hiddenDim: _hiddenDim,
            numEncoderLayers: _numEncoderLayers,
            numDecoderLayers: _numDecoderLayers,
            numHeads: _numHeads,
            vocabSize: _vocabSize,
            imageSize: ImageSize,
            patchSize: _patchSize,
            maxSequenceLength: MaxSequenceLength));
    }

    private void EnsureNativeInitialized()
    {
        if (!_useNativeMode || _nativeLayersInitialized)
        {
            return;
        }

        BuildFromConstructionSequence(InitializeLayers);
        _nativeLayersInitialized = true;
        InvalidateParameterCountCache();
    }

    // Native layers are materialized on first use to keep metadata-only construction cheap. The
    // materialized graph contains one branch-aware encoder-decoder composite, so parameter
    // introspection, prediction, and tape training all walk the same executable graph.

    /// <inheritdoc />
    /// <remarks>
    /// Nougat builds its encoder and decoder lists inside EnsureNativeInitialized, so there is nothing for the base to walk until that has run.
    /// <para>
    /// This replaces the ParameterCount and GetParameters overrides that each ran the
    /// initializer and then called base -- and SetParameters, which had NEITHER, so a restore
    /// could run against an unbuilt model. The base calls this hook from all of them, so the count,
    /// the vector, the restore and the chunks cannot observe different stages of construction.
    /// </para>
    /// </remarks>
    protected override void EnsureParametersReady()
    {
        EnsureNativeInitialized();
    }

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        EnsureNativeInitialized();
        return base.GetNamedLayerActivations(input);
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
        var output = _useNativeMode
            ? ForwardAfterNativeInitialization(preprocessed)
            : RunOnnxInference(preprocessed);

        var (markdown, confidence) = DecodeToMarkdown(output, maxAnswerLength, temperature);

        return new DocumentQAResult<T>
        {
            Answer = markdown,
            Confidence = confidence,
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

        // A field found in the page's Markdown carries that decode's confidence (the mean token probability);
        // a missing field has 0. It used to report a constant 0.5.
        var page = AnswerQuestion(documentImage, "Convert to Markdown");
        var markdown = page.Answer ?? string.Empty;
        using var parser = new StructuredOutputParser<T>();
        var kvPairs = parser.ParseKeyValuePairs(markdown);

        var results = new Dictionary<string, DocumentQAResult<T>>();
        T found = page.Confidence;
        T missing = NumOps.Zero;
        foreach (var field in fieldPrompts)
        {
            kvPairs.TryGetValue(field, out var value);
            results[field] = new DocumentQAResult<T>
            {
                Answer = value ?? string.Empty,
                Confidence = string.IsNullOrWhiteSpace(value) ? missing : found,
                Question = field,
                ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
            };
        }

        return results;
    }

    private (string markdown, T confidence) DecodeToMarkdown(Tensor<T> output, int maxLength, double temperature)
    {
        if (output.Shape.Length == 0 || maxLength <= 0)
            return (string.Empty, NumOps.Zero);

        int cappedMaxLength = Math.Min(maxLength, MaxSequenceLength);
        if (cappedMaxLength <= 0)
            return (string.Empty, NumOps.Zero);

        string eosToken = _tokenizer.SpecialTokens.EosToken;
        bool hasEosToken = !string.IsNullOrWhiteSpace(eosToken);

        var tokens = new List<int>();

        if (output.Shape.Length == 1)
        {
            int length = Math.Min(cappedMaxLength, output.Shape[0]);
            int[] ids = length == 0
                ? Array.Empty<int>()
                : Engine.TensorRound(Engine.TensorSlice(output, new[] { 0 }, new[] { length })).Cast<int>().ToArray();
            foreach (int tokenId in ids)
            {
                if (hasEosToken && _tokenizer.Vocabulary.GetToken(tokenId) == eosToken)
                    break;

                tokens.Add(tokenId);
            }

            string text = tokens.Count == 0 ? string.Empty : _tokenizer.Decode(tokens);
            return (text, NumOps.Zero);
        }

        double clampedTemperature = Math.Max(0.0, Math.Min(temperature, 2.0));
        bool useSampling = clampedTemperature > 0.0;
        if (useSampling)
            clampedTemperature = Math.Max(clampedTemperature, 0.01);

        int rank = output.Shape.Length;
        int seqLength = output.Shape[rank - 2];
        int vocabSize = output.Shape[rank - 1];
        int maxSteps = Math.Min(cappedMaxLength, seqLength);
        if (maxSteps <= 0 || vocabSize <= 0)
            return (string.Empty, NumOps.Zero);

        // The logits of the first sequence (every leading index 0), one row per decoding step.
        int leading = output.Length / (seqLength * vocabSize);
        var steps = Engine.Reshape(
            Engine.TensorSlice(Engine.Reshape(output, new[] { leading, seqLength, vocabSize }),
                new[] { 0, 0, 0 }, new[] { 1, maxSteps, vocabSize }),
            new[] { maxSteps, vocabSize });
        var (stepTokens, probabilities) = useSampling
            ? DocumentTensorOps.SampleWithProbabilities(Engine, steps, NumOps.FromDouble(clampedTemperature), RandomHelper.Shared.Next())
            : DocumentTensorOps.GreedyWithProbabilities(Engine, steps);

        foreach (int tokenId in stepTokens)
        {
            if (hasEosToken && _tokenizer.Vocabulary.GetToken(tokenId) == eosToken)
                break;

            tokens.Add(tokenId);
        }

        string decoded = tokens.Count == 0 ? string.Empty : _tokenizer.Decode(tokens);
        return (decoded, DocumentTensorOps.MeanOfPositivePrefix(Engine, probabilities, tokens.Count));
    }

    private static bool IsMarkdownPrompt(string question)
    {
        if (string.IsNullOrWhiteSpace(question))
            return true;

        return question.Contains("markdown", StringComparison.OrdinalIgnoreCase)
            || question.Contains("convert", StringComparison.OrdinalIgnoreCase)
            || question.Contains("parse", StringComparison.OrdinalIgnoreCase);
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
        return _useNativeMode
            ? ForwardAfterNativeInitialization(preprocessed)
            : RunOnnxInference(preprocessed);
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
        => DocumentTensorOps.NormalizeChannels(Engine, EnsureBatchDimension(rawImage),
            DocumentTensorOps.ImageNetMean, DocumentTensorOps.ImageNetStd);

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
            ModelDataProvider = () => SafeSerializeMaterializedModel()
        };
    }

    private byte[] SafeSerializeMaterializedModel()
    {
        return _useNativeMode && !_nativeLayersInitialized
            ? Array.Empty<byte>()
            : SafeSerialize();
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode
            ? ForwardAfterNativeInitialization(preprocessed)
            : RunOnnxInference(preprocessed);
    }

    private Tensor<T> ForwardAfterNativeInitialization(Tensor<T> input)
    {
        EnsureNativeInitialized();
        return Forward(input);
    }

    // UpdateParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    private Vector<T> CollectGradients()
    {
        var grads = new List<T>();
        EnsureNativeInitialized();
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



