using AiDotNet.Attributes;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// RegionCLIP model extending CLIP to learn region-level (object-level) visual representations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RegionCLIP (Zhong et al., CVPR 2022) generates region-text pairs from image captions using object
/// proposals and learns to align individual image regions with text descriptions, enabling zero-shot
/// and open-vocabulary object detection.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "RegionCLIP: Region-based Language-Image Pretraining" (Zhong et al., CVPR 2022)</item></list></para>
/// <para><b>For Beginners:</b> RegionCLIP extends CLIP from whole-image understanding to
/// individual object regions within images. While CLIP matches an entire image to text,
/// RegionCLIP learns to match specific regions (bounding boxes) to text descriptions,
/// enabling zero-shot object detection — finding objects in images by describing them in
/// natural language. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new RegionCLIP&lt;double&gt;(architecture, new RegionCLIPOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "RegionCLIP: Region-based Language-Image Pretraining",
    "https://arxiv.org/abs/2112.09106",
    Year = 2022,
    Authors = "Zhong et al."
)]
public partial class RegionCLIP<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    private readonly RegionCLIPOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    public RegionCLIP(
        NeuralNetworkArchitecture<T> architecture,
        string imageEncoderModelPath,
        RegionCLIPOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new RegionCLIPOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;
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
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public RegionCLIP(
        NeuralNetworkArchitecture<T> architecture,
        RegionCLIPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new RegionCLIPOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = true;
        _optimizer = optimizer ?? new StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new StochasticGradientDescentOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                UseAdaptiveLearningRate = false,
            });
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    private void SyncImageSizeWithArchitecture()
    {
        int h = Architecture.InputHeight;
        int w = Architecture.InputWidth;
        if (h > 0 && w > 0 && h == w)
            _options.ImageSize = h;
    }

    public int EmbeddingDimension => _options.VisionEmbeddingDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxSequenceLength => _options.MaxSequenceLength;
    public int TextEmbeddingDimension => _options.TextEmbeddingDim;
    public int ProjectionDimension => _options.ProjectionDim;
    public T Temperature => NumOps.FromDouble(_options.Temperature);

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
        // Fail fast in ONNX mode when no text encoder is configured. The
        // ONNX-mode constructor never populates TextEncoderLayers, so the
        // previous fallback ran the empty native stack and L2-normalized
        // raw token IDs — ComputeSimilarity / ZeroShotClassify silently
        // returned garbage instead of surfacing the configuration error.
        if (IsOnnxMode)
        {
            if (OnnxTextEncoder is null)
                throw new InvalidOperationException(
                    "Text encoding in ONNX mode requires a configured text encoder model path. "
                        + "Set RegionCLIPOptions.TextEncoderModelPath before constructing the model."
                );
            return L2Normalize(OnnxTextEncoder.Run(t));
        }
        var c = t;
        foreach (var l in TextEncoderLayers)
            c = l.Forward(c);
        return L2Normalize(c);
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

    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels)
    {
        var ie = EncodeImage(image);
        var te = EncodeTexts(labels);
        var logits = new Tensor<T>([labels.Length]);
        double temp = _options.Temperature;
        for (int i = 0; i < labels.Length; i++)
            logits[i] = NumOps.FromDouble(NumOps.ToDouble(CosineSimilarity(ie, te[i])) / temp);
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
        if (Architecture is DualStreamArchitecture<T> dual)
        {
            Layers.AddRange(dual.VisionLayers);
            TextEncoderLayers.AddRange(dual.TextLayers);
            return;
        }

        if (_options.UseResNet50Backbone)
        {
            if (_options.ResNetStageWidths is null || _options.ResNetStageWidths.Length != 4
                || _options.ResNetStageDepths is null || _options.ResNetStageDepths.Length != 4)
                throw new ArgumentException("RegionCLIP ResNet-50 requires four stage widths and four stage depths.");
            if (_options.ResNetStageWidths.Any(value => value <= 0 || value % 4 != 0)
                || _options.ResNetStageDepths.Any(value => value <= 0))
                throw new ArgumentException("ResNet stage widths must be positive multiples of four and depths must be positive.");

            Layers.AddRange(LayerHelper<T>.CreateResNetBottleneckEncoderLayers(
                3, _options.ImageSize, _options.ImageSize,
                _options.ResNetStageWidths, _options.ResNetStageDepths));
            Layers.Add(AdaptiveAveragePoolingLayer<T>.GlobalPool());
            Layers.Add(new FlattenLayer<T>());
            Layers.Add(new DenseLayer<T>(
                _options.ProjectionDim,
                (IActivationFunction<T>)new IdentityActivation<T>()));

            // RegionCLIP freezes/reuses CLIP's text tower while training the
            // region-aware ResNet visual encoder. Reuse the same configurable
            // text stream emitted for the ViT compatibility path.
            int textSplit = 2 + _options.NumVisionLayers * (_options.DropoutRate > 0 ? 6 : 5);
            TextEncoderLayers.AddRange(
                LayerHelper<T>.CreateDefaultOpenCLIPLayers(
                    _options.VisionEmbeddingDim,
                    _options.TextEmbeddingDim,
                    _options.ProjectionDim,
                    _options.NumVisionLayers,
                    _options.NumTextLayers,
                    _options.NumVisionHeads,
                    _options.NumTextHeads,
                    _options.DropoutRate)
                .Skip(textSplit));
            return;
        }

        // PatchSize is a public architecture option for the optional ViT path.
        // Deriving it from ImageSize silently ignored customization and produced
        // a fixed 16x16 token grid, inflating attention work.
        int patchSize = _options.PatchSize;
        if (patchSize <= 0 || patchSize > _options.ImageSize || _options.ImageSize % patchSize != 0)
            throw new ArgumentOutOfRangeException(
                nameof(_options.PatchSize),
                "PatchSize must be positive, no larger than ImageSize, and evenly divide ImageSize.");
        Layers.Add(
            new PatchEmbeddingLayer<T>(
                patchSize,
                _options.VisionEmbeddingDim,
                expectedInputChannels: 3
            )
        );

        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int visionLayerCount = 2 + _options.NumVisionLayers * blockSize;
        SplitDualStreamLayers(
            LayerHelper<T>.CreateDefaultOpenCLIPLayers(
                _options.VisionEmbeddingDim,
                _options.TextEmbeddingDim,
                _options.ProjectionDim,
                _options.NumVisionLayers,
                _options.NumTextLayers,
                _options.NumVisionHeads,
                _options.NumTextHeads,
                _options.DropoutRate
            ),
            visionLayerCount
        );
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        // Normalize ONNX inputs the same way the native path does — both
        // EncodeImage and the native Predict call PreprocessImage. Without
        // this, the ONNX fast path would diverge silently from native.
        var c = PreprocessImage(input);
        if (IsOnnxMode && OnnxImageEncoder is not null)
            return OnnxImageEncoder.Run(c);
        SetTrainingMode(false);
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
            TrainWithTape(PreprocessImage(input), expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    // This forwarded to a helper the base now calls from its own
    // GetExtraTrainableLayers, so the override restated it. Removed under AIDN082.

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "RegionCLIP-Native" : "RegionCLIP-ONNX",
            Description =
                "RegionCLIP: Region-based Language-Image Pretraining (Zhong et al., CVPR 2022)",
            FeatureCount = _options.ProjectionDim,
            Complexity = _options.NumVisionLayers + _options.NumTextLayers,
        };
        m.AdditionalInfo["Architecture"] = "RegionCLIP";
        m.AdditionalInfo["MaxRegionsPerImage"] = _options.MaxRegionsPerImage.ToString();
        m.AdditionalInfo["Domain"] = _options.Domain.ToString();
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

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(RegionCLIP<T>));
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
