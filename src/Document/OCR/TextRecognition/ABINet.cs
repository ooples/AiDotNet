using AiDotNet.Attributes;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;

namespace AiDotNet.Document.OCR.TextRecognition;

/// <summary>
/// ABINet (Autonomous, Bidirectional, Iterative Network) for text recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ABINet uses a novel architecture with autonomous vision, bidirectional language modeling,
/// and iterative correction to achieve robust text recognition.
/// </para>
/// <para>
/// <b>For Beginners:</b> ABINet has three key innovations:
/// 1. Autonomous vision model (works without external language model)
/// 2. Bidirectional language model (looks at context from both directions)
/// 3. Iterative correction (refines predictions multiple times)
///
/// Key features:
/// - Self-contained (no external LM needed)
/// - Built-in spell correction via language model
/// - Iterative refinement for accuracy
/// - Strong on noisy/occluded text
///
/// Example usage:
/// <code>
/// var model = new ABINet&lt;float&gt;(architecture);
/// var result = model.RecognizeText(textImage);
/// // Result is available in the returned value
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling" (CVPR 2021)
/// https://arxiv.org/abs/2103.06495
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition", "https://doi.org/10.48550/arXiv.2103.06495", Year = 2021, Authors = "Shancheng Fang, Hongtao Xie, Yuxin Wang, Zhendong Mao, Yongdong Zhang")]
public partial class ABINet<T> : DocumentNeuralNetworkBase<T>, ITextRecognizer<T>
{
    private readonly ABINetOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _visionDim;
    private readonly int _languageDim;
    private readonly int _visionLayers;
    private readonly int _languageLayers;
    private readonly int _numIterations;
    private readonly int _imageHeight;
    private readonly string _charset;

    // Native mode layers. ABINet is a three-branch model (Fang et al., CVPR 2021), not a single
    // chain: the vision model, the language model and the fusion each emit character
    // probabilities and each carries its own loss term. These lists hold the branches so the
    // training forward can supervise all three; every layer in them is also in Layers, so
    // parameter enumeration, serialization and device transfer are unaffected.
    private readonly List<ILayer<T>> _visionModelLayers = [];
    private readonly List<ILayer<T>> _languageModelLayers = [];
    private readonly List<ILayer<T>> _fusionLayers = [];

    /// <summary>Character head for the vision branch, supervised by L_v.</summary>
    private readonly List<ILayer<T>> _visionHead = [];

    /// <summary>Character head for the language branch, supervised by L_l.</summary>
    private readonly List<ILayer<T>> _languageHead = [];

    private int[]? _branchCounts;

    /// <summary>
    /// True when this instance built the paper's three-branch stack. False when the caller
    /// supplied a flat <c>Architecture.Layers</c> chain, which has no branch structure to
    /// supervise separately.
    /// </summary>
    private bool _branched;

    // Learnable embeddings

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => false;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    public string SupportedCharacters => _charset;

    // NO CONTRACT YET, and the sweep is why. The class count IS charset+1 - that half was right - but
    // the STEP count is not MaxSequenceLength: the contract said [1,26,96] and Predict returned
    // [1,256,96]. The [MaxSequenceLength, charset+1] tensor this class builds is a FALLBACK path; the
    // real forward decodes 256 steps from the vision backbone, and 26 is just the configured maximum.
    // Stating the family law here would assert a step count the model does not use, so this declines
    // until the 256 is traced to whatever produces it. CRNN, whose CTC head really does emit
    // MaxSequenceLength steps, agrees with the family law and keeps its contract.

    /// <inheritdoc/>
    public new int MaxSequenceLength => base.MaxSequenceLength;

    /// <inheritdoc/>
    public bool SupportsAttentionVisualization => true;

    /// <summary>
    /// Gets the number of iterative refinement steps.
    /// </summary>
    public int NumIterations => _numIterations;

    /// <summary>
    /// Gets the input image height.
    /// </summary>
    public int ImageHeight => _imageHeight;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an ABINet model using a pre-trained ONNX model for inference.
    /// </summary>
    public ABINet(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int imageWidth = 128,
        int imageHeight = 32,
        int maxSequenceLength = 26,
        int visionDim = 512,
        int languageDim = 512,
        int visionLayers = 3,
        int languageLayers = 4,
        int numIterations = 3,
        string? charset = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        ABINetOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new ABINetOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _visionDim = visionDim;
        _languageDim = languageDim;
        _visionLayers = visionLayers;
        _languageLayers = languageLayers;
        _numIterations = numIterations;
        _imageHeight = imageHeight;
        _charset = charset ?? GetDefaultCharset();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                // "decayed to 1e-5 after 6 epochs" -- a single 10x step at epoch 6.
                LearningRateScheduler = new MultiStepLRScheduler(
                    _options.LearningRate, milestones: new[] { 6 }, gamma: 0.1),
                SchedulerStepMode = SchedulerStepMode.StepPerEpoch
            });

        ImageSize = imageWidth;
        base.MaxSequenceLength = maxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates an ABINet model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (ABINet from CVPR 2021):</b>
    /// - Vision Model: ResNet + Transformer
    /// - Language Model: Bidirectional Transformer
    /// - Fusion: Iterative refinement
    /// - 3 correction iterations by default
    /// </para>
    /// </remarks>
    public ABINet(
        NeuralNetworkArchitecture<T> architecture,
        int imageWidth = 128,
        int imageHeight = 32,
        int maxSequenceLength = 26,
        int visionDim = 512,
        int languageDim = 512,
        int visionLayers = 3,
        int languageLayers = 4,
        int numIterations = 3,
        string? charset = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        ABINetOptions? options = null,
        double? visionLossWeight = null,
        double? languageLossWeight = null)
        : base(architecture, BuildMultiTaskObjective(lossFunction, options, visionLossWeight, languageLossWeight), 1.0)
    {
        _options = options ?? new ABINetOptions();
        Options = _options;

        // Record the resolved weights so GetOptions() reports what training actually used.
        if (visionLossWeight.HasValue) _options.VisionLossWeight = visionLossWeight.Value;
        if (languageLossWeight.HasValue) _options.LanguageLossWeight = languageLossWeight.Value;

        _useNativeMode = true;
        _visionDim = visionDim;
        _languageDim = languageDim;
        _visionLayers = visionLayers;
        _languageLayers = languageLayers;
        _numIterations = numIterations;
        _imageHeight = imageHeight;
        _charset = charset ?? GetDefaultCharset();

        // Adam at the paper's initial learning rate (Fang et al., CVPR 2021 §4.2: 1e-4, decayed
        // to 1e-5). Constructing AdamOptimizer with no options left it at the optimizer's own
        // 1e-3 default, 10x the paper's rate.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });

        ImageSize = imageWidth;
        base.MaxSequenceLength = maxSequenceLength;

        InitializeLayers();
        InitializeEmbeddings();
    }

    /// <summary>
    /// Builds ABINet's training objective: the paper's weighted sum of the vision, language and
    /// fusion character losses (Fang et al., CVPR 2021, Eq. 5).
    /// </summary>
    /// <remarks>
    /// A caller-supplied loss becomes the per-branch character loss rather than replacing the
    /// multi-task structure, so overriding the loss still trains all three branches. A loss that
    /// is not a <see cref="LossFunctionBase{T}"/> cannot expose the tape entry point the sum
    /// needs, so it is used as-is and only the fused output is graded.
    /// </remarks>
    private static ILossFunction<T> BuildMultiTaskObjective(
        ILossFunction<T>? lossFunction,
        ABINetOptions? options,
        double? visionLossWeight,
        double? languageLossWeight)
    {
        var characterLoss = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();
        if (characterLoss is not LossFunctionBase<T> tapeCapable)
            return characterLoss;

        var resolved = options ?? new ABINetOptions();
        return new ABINetMultiTaskLoss<T>(
            tapeCapable,
            visionLossWeight ?? resolved.VisionLossWeight,
            languageLossWeight ?? resolved.LanguageLossWeight);
    }

    private static string GetDefaultCharset()
    {
        return "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";
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

        // Build the paper's three branches separately so each can be supervised by its own loss
        // term. Chained in this order for inference, they reproduce exactly the flat stack this
        // used to create.
        int charsetSize = _charset.Length + 1;

        _visionModelLayers.AddRange(LayerHelper<T>.CreateDefaultABINetVisionLayers(
            imageWidth: ImageSize,
            imageHeight: _imageHeight,
            visionDim: _visionDim));

        _languageModelLayers.AddRange(LayerHelper<T>.CreateDefaultABINetLanguageLayers(
            charsetSize: charsetSize,
            visionDim: _visionDim,
            languageDim: _languageDim));

        _fusionLayers.AddRange(LayerHelper<T>.CreateDefaultABINetFusionLayers(
            visionDim: _visionDim,
            numIterations: _numIterations,
            charsetSize: charsetSize));

        _visionHead.Add(LayerHelper<T>.CreateDefaultABINetBranchHead(charsetSize));
        _languageHead.Add(LayerHelper<T>.CreateDefaultABINetBranchHead(charsetSize));

        ResolveBranchShapes();

        // Everything goes into Layers so parameter enumeration, serialization and device
        // transfer see every weight. The two branch heads are appended last and are NOT part of
        // the inference chain; Forward walks the three branches explicitly.
        Layers.AddRange(_visionModelLayers);
        Layers.AddRange(_languageModelLayers);
        Layers.AddRange(_fusionLayers);
        Layers.AddRange(_visionHead);
        Layers.AddRange(_languageHead);

        // Deserialization refills Layers with new objects; record the branch extents so
        // RebindBranchLayers can re-point these lists at the restored ones.
        _branchCounts = new[]
        {
            _visionModelLayers.Count, _languageModelLayers.Count, _fusionLayers.Count,
            _visionHead.Count, _languageHead.Count
        };

        _branched = true;
    }

    /// <summary>
    /// Resolves each branch's lazy layers, carrying the shape across the two forks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The character heads hang off the vision and language trunks rather than sitting in the
    /// inference chain, so nothing else would ever size them: a deserialized model never runs
    /// them, they would report a <c>ParameterCount</c> of 0, and <c>SetParameters</c> would then
    /// hand every following layer the wrong slice of the flat parameter vector.
    /// </para>
    /// <para>
    /// <see cref="LayerHelper{T}.ResolveChain"/> derives every layer's input from the previous
    /// layer's actual <c>GetOutputShape()</c> and returns the shape leaving the branch, so the
    /// fork points get real shapes instead of hand-written ones. Sizing the heads from a
    /// hand-written <c>[1, 1, visionDim]</c> instead produced weights the forward never matched.
    /// Each layer is skipped once resolved, so this is a no-op on an already-run model.
    /// </para>
    /// </remarks>
    private void ResolveBranchShapes()
    {
        var rootShape = Architecture.GetInputShape();
        if (rootShape is null || rootShape.Length == 0) return;

        // KNOWN LIMITATION: resolution currently stops inside the vision trunk, at the
        // ReshapeLayer between the convolutions and the transformer. The convolution layers
        // report GetOutputShape() WITHOUT a batch axis ([512, 8, 32]), while
        // ReshapeLayer.ResolveFromShape treats the leading axis as batch — so it reads that as
        // 512 samples of 256 elements against its 131072-element target and rejects it. Chain
        // resolution stops at the first such failure by design, leaving the rest of the stack
        // lazy. Adding a batch axis to the root does not help: the convolutions report
        // per-sample shapes regardless of what they were resolved from.
        //
        // Consequence: a freshly built ABINet reports 1,718,624 parameters where one that has
        // run a forward reports 4,281,376, so restoring a trained parameter vector into a fresh
        // clone misaligns (Clone_AfterTraining_ShouldPreserveLearnedWeights). The layers that DO
        // resolve here still benefit, and everything else resolves on first forward as before.
        //
        // The real fix is to make the two conventions agree — either the convolutions report a
        // batched output shape or ReshapeLayer accepts a per-sample one — which is a framework
        // change affecting every model that chains a convolution into a reshape, not something
        // to settle inside ABINet.

        // Vision trunk -> vision head (character logits) -> language model -> language head.
        // The language model is rooted at the HEAD's output, not the trunk's, because it
        // consumes character probabilities.
        var visionOut = LayerHelper<T>.ResolveChain(_visionModelLayers, rootShape);
        var visionLogitsShape = LayerHelper<T>.ResolveChain(_visionHead, visionOut);

        var languageOut = LayerHelper<T>.ResolveChain(_languageModelLayers, visionLogitsShape);
        LayerHelper<T>.ResolveChain(_languageHead, languageOut);

        // The fusion gate is rooted at [F_v, F_l] concatenated on the feature axis, so its
        // input is the vision width doubled.
        var fusionRoot = (int[])visionOut.Clone();
        fusionRoot[fusionRoot.Length - 1] = visionOut[visionOut.Length - 1] + languageOut[languageOut.Length - 1];
        LayerHelper<T>.ResolveChain(_fusionLayers, fusionRoot);
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);
    }

    private void InitializeWithSmallRandomValues(Tensor<T> tensor, Random random, double stdDev)
    {
        for (int i = 0; i < tensor.Data.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            tensor.Data.Span[i] = NumOps.FromDouble(randStdNormal * stdDev);
        }
    }

    #endregion

    #region ITextRecognizer Implementation

    /// <inheritdoc/>
    public TextRecognitionResult<T> RecognizeText(Tensor<T> croppedImage)
    {
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessTextImage(croppedImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var (text, confidence) = Decode(output);

        return new TextRecognitionResult<T>
        {
            Text = text,
            Confidence = NumOps.FromDouble(confidence),
            ConfidenceValue = confidence,
            Characters = GetCharacterConfidences(output, text),
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public IEnumerable<TextRecognitionResult<T>> RecognizeTextBatch(IEnumerable<Tensor<T>> croppedImages)
    {
        foreach (var image in croppedImages)
            yield return RecognizeText(image);
    }

    /// <inheritdoc/>
    public Tensor<T> GetCharacterProbabilities()
    {
        return Tensor<T>.CreateDefault([MaxSequenceLength, _charset.Length + 1], NumOps.Zero);
    }

    /// <inheritdoc/>
    public Tensor<T>? GetAttentionWeights()
    {
        return Tensor<T>.CreateDefault([MaxSequenceLength, MaxSequenceLength], NumOps.Zero);
    }

    private (string text, double confidence) Decode(Tensor<T> output)
    {
        var chars = new List<char>();
        double totalConf = 0;
        int validSteps = 0;

        int seqLen = Math.Min(output.Shape[0], MaxSequenceLength);
        int vocabSize = output.Shape.Length > 1 ? output.Shape[1] : _charset.Length + 1;

        for (int t = 0; t < seqLen; t++)
        {
            double maxVal = double.MinValue;
            int maxIdx = 0;
            for (int c = 0; c < vocabSize; c++)
            {
                double val = NumOps.ToDouble(output[t, c]);
                if (val > maxVal) { maxVal = val; maxIdx = c; }
            }

            if (maxIdx == 0) break; // EOS
            if (maxIdx - 1 < _charset.Length)
            {
                chars.Add(_charset[maxIdx - 1]);
                totalConf += maxVal;
                validSteps++;
            }
        }

        string text = new string([.. chars]);
        double avgConf = validSteps > 0 ? totalConf / validSteps : 0;

        return (text, avgConf);
    }

    private List<CharacterRecognition<T>> GetCharacterConfidences(Tensor<T> output, string text)
    {
        var result = new List<CharacterRecognition<T>>();
        for (int i = 0; i < text.Length; i++)
        {
            result.Add(new CharacterRecognition<T>
            {
                Character = text[i],
                Confidence = NumOps.FromDouble(0.92),
                ConfidenceValue = 0.92,
                Position = i
            });
        }
        return result;
    }

    private Tensor<T> PreprocessTextImage(Tensor<T> image)
    {
        var processed = EnsureBatchDimension(image);
        if (processed.Shape[2] != _imageHeight || processed.Shape[3] != ImageSize)
        {
            processed = Engine.Interpolate(
                processed,
                [_imageHeight, ImageSize],
                InterpolateMode.Bilinear,
                alignCorners: false);
        }

        var normalized = new Tensor<T>(processed._shape);

        for (int i = 0; i < processed.Data.Length; i++)
        {
            double val = NumOps.ToDouble(processed.Data.Span[i]);
            normalized.Data.Span[i] = NumOps.FromDouble((val / 255.0 - 0.5) / 0.5);
        }

        return normalized;
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        var preprocessed = PreprocessTextImage(documentImage);
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
        sb.AppendLine("ABINet Model Summary");
        sb.AppendLine("====================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: Vision + Language + Iterative Fusion");
        sb.AppendLine($"Vision Dimension: {_visionDim}");
        sb.AppendLine($"Language Dimension: {_languageDim}");
        sb.AppendLine($"Vision Layers: {_visionLayers}");
        sb.AppendLine($"Language Layers: {_languageLayers}");
        sb.AppendLine($"Iterations: {_numIterations}");
        sb.AppendLine($"Image Size: {ImageSize}x{_imageHeight}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Charset Size: {_charset.Length}");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies ABINet's industry-standard preprocessing: text image preprocessing.
    /// </summary>
    /// <remarks>
    /// ABINet (Attention-Based Implicit Network) uses text-specific preprocessing
    /// with grayscale conversion and height normalization.
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage) => PreprocessTextImage(rawImage);

    /// <summary>
    /// Applies ABINet's industry-standard postprocessing: pass-through (language model outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "ABINet",
            Description = "ABINet for robust text recognition (CVPR 2021)",
            FeatureCount = _visionDim,
            Complexity = _visionLayers + _languageLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "vision_dim", _visionDim },
                { "language_dim", _languageDim },
                { "vision_layers", _visionLayers },
                { "language_layers", _languageLayers },
                { "num_iterations", _numIterations },
                { "image_height", _imageHeight },
                { "image_width", ImageSize },
                { "charset_size", _charset.Length },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = SafeSerialize()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>
    /// <summary>
    /// Re-points the branch lists at the layers deserialization just rebuilt.
    /// </summary>
    /// <remarks>
    /// Every branch is appended to <c>Layers</c>, so the weights round-trip correctly, but the
    /// forward pass reads these private lists and deserialization never re-points them — they
    /// still referenced the objects this instance built in its own constructor. The restored
    /// weights landed in layers the model never evaluated, so a clone predicted from its
    /// initialisation values while reporting success.
    /// </remarks>
    private void RebindBranchLayers()
    {
        if (_branchCounts is not { Length: 5 }) return;

        int total = 0;
        foreach (var count in _branchCounts) total += count;
        if (total == 0 || Layers.Count < total) return;

        var targets = new[]
        {
            _visionModelLayers, _languageModelLayers, _fusionLayers, _visionHead, _languageHead
        };

        int offset = Layers.Count - total;
        for (int b = 0; b < targets.Length; b++)
        {
            targets[b].Clear();
            for (int i = 0; i < _branchCounts[b]; i++) targets[b].Add(Layers[offset + i]);
            offset += _branchCounts[b];
        }
    }



    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Runs ABINet's explicit sequential graph without the document base
    /// class's inference-only CNN-to-sequence auto-reshape. The default ABINet
    /// graph contains its own tape-compatible ReshapeLayer so inference and
    /// training follow the same shape transitions.
    /// </summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_branched)
            return ForwardBranches(input).Fusion;

        Tensor<T> output = input;
        foreach (var layer in Layers)
            output = layer.Forward(output);

        return output;
    }

    /// <summary>
    /// Runs the vision model, then the language model, then the fusion branch, returning all
    /// three character predictions.
    /// </summary>
    /// <remarks>
    /// The language branch begins with the gradient barrier, so language and fusion gradients
    /// stop there and never reach the vision encoder — ABINet's AUTONOMOUS principle. The vision
    /// encoder still learns, from its own <c>Vision</c> prediction's loss term.
    /// </remarks>
    private (Tensor<T> Vision, Tensor<T> Language, Tensor<T> Fusion) ForwardBranches(Tensor<T> input)
    {
        var visionFeatures = input;
        foreach (var layer in _visionModelLayers)
            visionFeatures = layer.Forward(visionFeatures);

        // F_v -> character logits. These ARE the language model's input: the paper's LM is a
        // spelling corrector over probability vectors, so the branch begins with the gradient
        // barrier and a softmax rather than reading visual features.
        var visionLogits = visionFeatures;
        foreach (var layer in _visionHead)
            visionLogits = layer.Forward(visionLogits);

        // ITERATIVE correction, the third of the paper's three principles (Fang et al. 2021,
        // sec. 3.3). The language model is executed M times: the first pass reads the VISION
        // model's character probabilities, and every later pass reads the FUSION model's
        // prediction from the previous iteration, so each round corrects the last round's
        // spelling using bidirectional context. The paper measures M = 3 as the sweet spot and
        // uses the final iteration's fused prediction as the output.
        //
        // Only one pass was run before this, which reduced the model to Autonomous +
        // Bidirectional and silently dropped the principle the paper is named for. The iteration
        // count was already threaded in as _numIterations and consumed only when constructing
        // the language branch; nothing ever looped.
        //
        // Nothing carries across calls: every input restarts from its own vision prediction,
        // matching the paper's "each new text instance starts fresh".
        var languageInput = visionLogits;
        Tensor<T> languageLogits = visionLogits;
        Tensor<T> fused = visionFeatures;

        int iterations = _numIterations > 0 ? _numIterations : 1;
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            var languageFeatures = languageInput;
            foreach (var layer in _languageModelLayers)
                languageFeatures = layer.Forward(languageFeatures);

            languageLogits = languageFeatures;
            foreach (var layer in _languageHead)
                languageLogits = layer.Forward(languageLogits);

            // Gated fusion consumes BOTH streams: G = sigmoid([F_v, F_l] W_f),
            // F_f = G * F_v + (1 - G) * F_l. The gate layer takes them concatenated.
            fused = Engine.TensorConcatenate(
                new[] { visionFeatures, languageFeatures },
                axis: visionFeatures.Shape.Length - 1);
            foreach (var layer in _fusionLayers)
                fused = layer.Forward(fused);

            // The next round corrects this round's fused prediction.
            languageInput = fused;
        }

        return (visionLogits, languageLogits, fused);
    }

    /// <summary>
    /// Emits all three branch predictions stacked along axis 0 so the multi-task objective can
    /// grade each of them.
    /// </summary>
    /// <remarks>
    /// Pairs with <see cref="Train"/>, which repeats the character target three times to match,
    /// and with <see cref="ABINetMultiTaskLoss{T}"/>, which splits both back into three blocks
    /// and returns lambda_v * L_v + lambda_l * L_l + L_f.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_branched)
            return base.ForwardForTraining(input);

        // Subclasses that bypass the base forward must seed stochastic layers themselves.
        EnsureLayerRandomSeedsWired();

        var (vision, language, fusion) = ForwardBranches(input);
        return Engine.TensorConcatenate(new[] { vision, language, fusion }, axis: 0);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessTextImage(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        return new Dictionary<string, Tensor<T>>
        {
            ["ABINetOutput"] = PredictCore(input)
        };
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");

        SetTrainingMode(true);
        try
        {
            // The branched forward returns the vision, language and fusion predictions stacked
            // along axis 0, so the target is repeated three times to line up block-for-block.
            // ABINetMultiTaskLoss splits both and returns lambda_v * L_v + lambda_l * L_l + L_f.
            var target = _branched
                ? Engine.TensorConcatenate(new[] { expectedOutput, expectedOutput, expectedOutput }, axis: 0)
                : expectedOutput;

            TrainWithTape(
                PreprocessTextImage(input),
                target,
                _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    // UpdateParameters applied a GRADIENT STEP, but its one-argument form is the value setter and every caller passes values -- the override corrupted the model. Removed under AIDN082.


    /// <summary>
    /// Parameters cannot be written while the model is backed by a loaded ONNX graph: the weights
    /// belong to that graph, not to this instance.
    /// </summary>
    /// <remarks>
    /// Replaces a hand-written throw that used to sit inside UpdateParameters. The base checks this
    /// on every mutating entry point rather than the one member the throw happened to guard, and
    /// reading -- ParameterCount and GetParameters -- stays available either way.
    /// </remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
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
