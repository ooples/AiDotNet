using AiDotNet.Attributes;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// MedCLIP model using decoupled semantic matching for medical image-text alignment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MedCLIP (Wang et al., 2022) addresses limited medical data by decoupling image-text inputs:
/// any image can be paired with any text sharing the same medical concepts (diagnosis, anatomy),
/// using a semantic matching loss alongside contrastive learning.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "MedCLIP: Contrastive Learning from Unpaired Medical Images and Text" (Wang et al., EMNLP 2022)</item></list></para>
/// <para><b>For Beginners:</b> MedCLIP adapts CLIP for medical imaging by solving a key
/// problem: medical datasets are small and images are not always paired with matching text.
/// It uses decoupled semantic matching — any medical image can be paired with any text that
/// shares the same medical concepts (like "pneumonia" or "chest X-ray"), allowing it to learn
/// from unpaired data. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new MedCLIP&lt;double&gt;(architecture, new MedCLIPOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "MedCLIP: Contrastive Learning from Unpaired Medical Images and Text",
    "https://arxiv.org/abs/2210.10163",
    Year = 2022,
    Authors = "Wang et al."
)]
public partial class MedCLIP<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    private const int MedClipExtrasFormatVersion = 1;
    private readonly MedCLIPOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private readonly LearnableLogitScaleLayer<T> _logitScale;

    public MedCLIP(
        NeuralNetworkArchitecture<T> architecture,
        string imageEncoderModelPath,
        MedCLIPOptions? options = null
    )
        : base(architecture)
    {
        _options = options is null ? new MedCLIPOptions() : new MedCLIPOptions(options);
        _logitScale = new LearnableLogitScaleLayer<T>(_options.Temperature);
        SyncImageSizeWithArchitecture();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.ProjectionDim;
        if (string.IsNullOrWhiteSpace(imageEncoderModelPath))
            throw new ArgumentException(
                "Image encoder model path cannot be null or empty.",
                nameof(imageEncoderModelPath)
            );
        if (!File.Exists(imageEncoderModelPath))
            throw new FileNotFoundException(
                $"ONNX model not found: {imageEncoderModelPath}",
                imageEncoderModelPath
            );
        _options.ImageEncoderModelPath = imageEncoderModelPath;
        OnnxImageEncoder = new OnnxModel<T>(imageEncoderModelPath, _options.OnnxOptions);
        if (_options.TextEncoderModelPath is { } tp && !string.IsNullOrEmpty(tp))
        {
            if (!File.Exists(tp))
                throw new FileNotFoundException($"Text ONNX not found: {tp}", tp);
            OnnxTextEncoder = new OnnxModel<T>(tp, _options.OnnxOptions);
        }
        _tokenizer = CreateTokenizer();
        InitializeLayers();
    }

    public MedCLIP(
        NeuralNetworkArchitecture<T> architecture,
        MedCLIPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options is null ? new MedCLIPOptions() : new MedCLIPOptions(options);
        _logitScale = new LearnableLogitScaleLayer<T>(_options.Temperature);
        SyncImageSizeWithArchitecture();
        _useNativeMode = true;
        // The official MedCLIP pretraining recipe uses AdamW at 2e-5 with
        // decoupled weight decay 1e-4. Route the model options into the optimizer
        // instead of silently falling back to AdamW's generic 1e-3 / 1e-2 defaults.
        _optimizer =
            optimizer
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate,
                    Beta1 = 0.9,
                    Beta2 = 0.999,
                    Epsilon = 1e-8,
                    WeightDecay = _options.WeightDecay,
                }
            );
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.ProjectionDim;
        _tokenizer = CreateTokenizer();
        InitializeLayers();
    }

    private void SyncImageSizeWithArchitecture()
    {
        int h = Architecture.InputHeight;
        int w = Architecture.InputWidth;
        if (h > 0 && w > 0 && h == w)
            _options.ImageSize = h;
    }

    public int EmbeddingDimension => _options.ProjectionDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxSequenceLength => _options.MaxSequenceLength;
    public int TextEmbeddingDimension => _options.TextEmbeddingDim;
    public int ProjectionDimension => _options.ProjectionDim;
    /// <summary>Gets the number of ResNet-50 bottleneck blocks.</summary>
    public int VisionBottleneckBlockCount => Layers.Count(layer => layer is BottleneckBlock<T>);
    /// <summary>Gets the number of ClinicalBERT transformer blocks.</summary>
    public int TextTransformerBlockCount =>
        TextEncoderLayers.Count(layer => layer is TransformerEncoderBlock<T>);
    /// <summary>Gets whether the ClinicalBERT embedding stack retains token-type embeddings.</summary>
    public bool UsesTokenTypeEmbeddings =>
        TextEncoderLayers.Any(layer => layer is LearnedTokenTypeEmbeddingLayer<T>);
    /// <summary>Gets whether both reference 512-dimensional projections are bias-free.</summary>
    public bool UsesBiasFreeReferenceProjections =>
        Layers.LastOrDefault() is BiasFreeLinearLayer<T> &&
        TextEncoderLayers.LastOrDefault() is BiasFreeLinearLayer<T>;
    /// <summary>Gets the current learned temperature (the reciprocal of the clamped logit scale).</summary>
    public T Temperature => NumOps.Divide(NumOps.One, _logitScale.Scale);

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxImageEncoder is not null)
            return L2Normalize(OnnxImageEncoder.Run(p));
        var c = p;
        foreach (var l in Layers)
            c = l.Forward(c);
        return L2Normalize(c);
    }

    public Tensor<T> EncodeText(string text)
    {
        ThrowIfDisposed();
        var t = TokenizeText(text);
        if (IsOnnxMode && OnnxTextEncoder is not null)
            return L2Normalize(OnnxTextEncoder.Run(t));
        var c = t;
        var semanticStates = new List<Tensor<T>>(3);
        int encoderBlock = 0;
        // The projection runs after the reference implementation's three-state pooling.
        for (int i = 0; i < TextEncoderLayers.Count - 1; i++)
        {
            var l = TextEncoderLayers[i];
            c = l.Forward(c);
            if (l is TransformerEncoderBlock<T>)
            {
                encoderBlock++;
                if (encoderBlock is 1 or 2 || encoderBlock == _options.NumTextLayers)
                    semanticStates.Add(c);
            }
        }

        if (semanticStates.Count == 0)
            throw new InvalidOperationException("MedCLIP clinical text encoder produced no hidden states.");

        Tensor<T>? pooled = null;
        foreach (var state in semanticStates)
        {
            int tokenAxis = state.Rank == 3 ? 1 : 0;
            var layerPooled = Engine.ReduceMean(state, [tokenAxis], keepDims: false);
            pooled = pooled is null ? layerPooled : Engine.TensorAdd(pooled, layerPooled);
        }
        pooled = Engine.TensorMultiplyScalar(
            pooled!, NumOps.FromDouble(1.0 / semanticStates.Count));
        return L2Normalize(TextEncoderLayers[^1].Forward(pooled));
    }

    public Tensor<T>[] EncodeTexts(string[] texts)
    {
        var e = new Tensor<T>[texts.Length];
        for (int i = 0; i < texts.Length; i++)
            e[i] = EncodeText(texts[i]);
        return e;
    }

    public T ComputeSimilarity(Tensor<T> image, string text) =>
        CosineSimilarity(EncodeImage(image), EncodeText(text));

    /// <summary>
    /// Computes MedCLIP's bidirectional semantic matching objective for decoupled
    /// image/report batches. Semantic scores are converted to row-wise soft targets;
    /// the same target matrix supervises image-to-text and text-to-image logits.
    /// </summary>
    public T ComputeSemanticMatchingLoss(
        Tensor<T> imageEmbeddings,
        Tensor<T> textEmbeddings,
        Tensor<T> semanticScores)
        => ComputeSemanticMatchingLossTensor(imageEmbeddings, textEmbeddings, semanticScores)[0];

    /// <summary>
    /// Computes the tape-connected MedCLIP semantic objective. Unlike the scalar convenience
    /// overload, this form preserves gradients through both encoders and the learned temperature.
    /// </summary>
    public Tensor<T> ComputeSemanticMatchingLossTensor(
        Tensor<T> imageEmbeddings,
        Tensor<T> textEmbeddings,
        Tensor<T> semanticScores)
    {
        if (imageEmbeddings.Rank != 2 || textEmbeddings.Rank != 2 || semanticScores.Rank != 2)
            throw new ArgumentException("Semantic matching expects rank-2 embedding and score matrices.");
        int imageCount = imageEmbeddings.Shape[0];
        int textCount = textEmbeddings.Shape[0];
        if (imageEmbeddings.Shape[1] != textEmbeddings.Shape[1] ||
            semanticScores.Shape[0] != imageCount || semanticScores.Shape[1] != textCount)
            throw new ArgumentException("Embedding and semantic-score matrix shapes are inconsistent.");

        imageEmbeddings = L2Normalize(imageEmbeddings);
        textEmbeddings = L2Normalize(textEmbeddings);
        var textTranspose = Engine.TensorPermute(textEmbeddings, [1, 0]);
        var logits = _logitScale.Forward(Engine.TensorMatMul(imageEmbeddings, textTranspose));
        var boundedScores = Engine.TensorClamp(
            semanticScores, NumOps.FromDouble(-1.0), NumOps.One);
        var imageToText = SoftTargetCrossEntropyTensor(logits, boundedScores);
        var textToImage = SoftTargetCrossEntropyTensor(
            Engine.TensorPermute(logits, [1, 0]),
            Engine.TensorPermute(boundedScores, [1, 0]));
        return Engine.TensorMultiplyScalar(
            Engine.TensorAdd(imageToText, textToImage),
            NumOps.FromDouble(0.5 * _options.SemanticMatchingWeight));
    }

    private Tensor<T> SoftTargetCrossEntropyTensor(Tensor<T> logits, Tensor<T> scores)
    {
        // The text-to-image branch supplies transposed tensor views.  The CPU
        // softmax kernels require contiguous storage, so materialize those
        // views here while preserving the tensor-engine operation graph.
        var targets = Engine.Softmax(scores.Contiguous(), axis: -1);
        var logProbabilities = Engine.TensorLogSoftmax(logits.Contiguous(), axis: -1);
        var perRow = Engine.ReduceSum(
            Engine.TensorMultiply(targets, logProbabilities), [1], keepDims: false);
        return Engine.TensorNegate(Engine.ReduceMean(perRow, [0], keepDims: false));
    }

    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels)
    {
        var ie = EncodeImage(image);
        var te = EncodeTexts(labels);
        var logits = new Tensor<T>([labels.Length]);
        for (int i = 0; i < labels.Length; i++)
            logits[i] = NumOps.Multiply(CosineSimilarity(ie, te[i]), _logitScale.Scale);
        var probs = Softmax(logits);
        var r = new Dictionary<string, T>();
        for (int i = 0; i < labels.Length; i++)
            r[labels[i]] = probs[i];
        return r;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            BuildReferenceResNet50VisionEncoder();
        }
        BuildReferenceClinicalBertTextEncoder();
    }

    private void BuildReferenceResNet50VisionEncoder()
    {
        // The official default is torchvision ResNet-50 with stage depths [3,4,6,3]
        // and a 2048 -> 512 projection replacing the classifier.
        Layers.Add(new ConvolutionalLayer<T>(64, kernelSize: 7, stride: 2, padding: 3,
            activationFunction: new IdentityActivation<T>()));
        Layers.Add(new BatchNormalizationLayer<T>());
        Layers.Add(new ActivationLayer<T>((IActivationFunction<T>)new ReLUActivation<T>()));
        Layers.Add(new MaxPoolingLayer<T>(poolSize: 3, stride: 2));

        int[] stageDepths = [3, 4, 6, 3];
        int[] stageWidths = [64, 128, 256, 512];
        for (int stage = 0; stage < stageDepths.Length; stage++)
        {
            for (int block = 0; block < stageDepths[stage]; block++)
            {
                int stride = stage > 0 && block == 0 ? 2 : 1;
                Layers.Add(new BottleneckBlock<T>(stageWidths[stage], stride));
            }
        }

        Layers.Add(AdaptiveAveragePoolingLayer<T>.GlobalPool());
        Layers.Add(new FlattenLayer<T>());
        Layers.Add(new BiasFreeLinearLayer<T>(2048, _options.ProjectionDim));
    }

    private void BuildReferenceClinicalBertTextEncoder()
    {
        TextEncoderLayers.Add(new EmbeddingLayer<T>(_options.VocabSize, _options.TextEmbeddingDim));
        TextEncoderLayers.Add(new LearnedPositionalEmbeddingLayer<T>(
            _options.MaxSequenceLength, _options.TextEmbeddingDim));
        TextEncoderLayers.Add(new LearnedTokenTypeEmbeddingLayer<T>(
            tokenTypeCount: 2, _options.TextEmbeddingDim));
        TextEncoderLayers.Add(new LayerNormalizationLayer<T>(_options.TextEmbeddingDim));
        TextEncoderLayers.Add(new DropoutLayer<T>(_options.DropoutRate));
        for (int i = 0; i < _options.NumTextLayers; i++)
        {
            TextEncoderLayers.Add(new TransformerEncoderBlock<T>(
                _options.TextEmbeddingDim,
                _options.NumTextHeads,
                _options.TextEmbeddingDim * 4,
                _options.DropoutRate,
                new GELUActivation<T>()));
        }
        TextEncoderLayers.Add(new BiasFreeLinearLayer<T>(
            _options.TextEmbeddingDim, _options.ProjectionDim));
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxImageEncoder is not null)
            return OnnxImageEncoder.Run(input);
        SetTrainingMode(false);
        var c = PreprocessImage(input);
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(PreprocessImage(input), expected);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    private IEnumerable<LayerBase<T>?> EnumerateMedClipExtraLayers()
    {
        foreach (var layer in EnumerateTextEncoderTrainableLayers()) yield return layer;
        yield return _logitScale;
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image)
    {
        var normalized = NormalizeImage(image, _options.ImageMean, _options.ImageStd);
        if (normalized.Rank != 3)
            return normalized;

        // The native ResNet path is batch-first. Model-family fixtures and the public single-image
        // API also accept [C,H,W]; preserve that contract by materializing an explicit batch of one
        // before global pooling. Without it, Flatten interprets C as the batch dimension and the
        // reference 2048-wide projection receives [2048,1] instead of [1,2048].
        return Engine.Reshape(normalized,
            [1, normalized.Shape[0], normalized.Shape[1], normalized.Shape[2]]);
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MedCLIP-Native" : "MedCLIP-ONNX",
            Description =
                "MedCLIP: Contrastive Learning from Unpaired Medical Images and Text (Wang et al., EMNLP 2022)",
            FeatureCount = _options.ProjectionDim,
            Complexity = _options.NumVisionLayers + _options.NumTextLayers,
        };
        m.AdditionalInfo["Architecture"] = "MedCLIP";
        m.AdditionalInfo["VisionEncoder"] = $"{_options.VisionBackbone} ({_options.VisionModelId})";
        m.AdditionalInfo["TextEncoder"] = $"ClinicalBERT ({_options.TextModelId})";
        m.AdditionalInfo["TextPooling"] =
            $"mean(hidden_state_1, hidden_state_2, hidden_state_{_options.NumTextLayers})";
        m.AdditionalInfo["VisionTopology"] =
            $"{_options.VisionBackbone} bottleneck depths [3,4,6,3]; " +
            $"{_options.VisionEmbeddingDim}->{_options.ProjectionDim} bias-free projection";
        m.AdditionalInfo["TextTopology"] =
            $"ClinicalBERT {_options.NumTextLayers} layers, {_options.NumTextHeads} heads, " +
            $"hidden size {_options.TextEmbeddingDim}; " +
            $"{_options.TextEmbeddingDim}->{_options.ProjectionDim} bias-free projection";
        m.AdditionalInfo["LogitScale"] = "learned log(1/0.07), clamped to [0,log(100)]";
        m.AdditionalInfo["ImageNormalization"] =
            $"mean [{string.Join(",", _options.ImageMean.Select(value => value.ToString("R", System.Globalization.CultureInfo.InvariantCulture)))}]; " +
            $"std [{string.Join(",", _options.ImageStd.Select(value => value.ToString("R", System.Globalization.CultureInfo.InvariantCulture)))}]";
        m.AdditionalInfo["Domain"] = _options.Domain.ToString();
        m.AdditionalInfo["SemanticMatchingWeight"] = _options.SemanticMatchingWeight.ToString();
        return m;
    }





    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        var enc = _tokenizer.Encode(text);
        int sl = Math.Min(enc.TokenIds.Count, _options.MaxSequenceLength);
        var tk = new Tensor<T>([sl]);
        for (int i = 0; i < sl; i++)
            tk[i] = NumOps.FromDouble(enc.TokenIds[i]);
        return tk;
    }

    private ITokenizer CreateTokenizer()
    {
        if (!string.IsNullOrWhiteSpace(_options.TokenizerDirectory))
            return HuggingFaceTokenizerLoader.LoadFromDirectory(_options.TokenizerDirectory!);

        // Offline/randomly initialized models use a deterministic BERT-compatible fallback.
        // A pretrained MedCLIP checkpoint must set TokenizerDirectory so token IDs match
        // emilyalsentzer/Bio_ClinicalBERT exactly.
        return WordPieceTokenizer.Train(
            [
                "chest radiograph with no acute cardiopulmonary abnormality",
                "pneumonia pleural effusion cardiomegaly atelectasis",
                "medical image and clinical report"
            ],
            vocabSize: _options.VocabSize,
            specialTokens: SpecialTokens.Bert());
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(MedCLIP<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        if (disposing)
        {
            OnnxImageEncoder?.Dispose();
            OnnxTextEncoder?.Dispose();
        }
        base.Dispose(disposing);
    }
}
