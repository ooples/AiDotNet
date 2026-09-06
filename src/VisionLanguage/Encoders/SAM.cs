using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Segment Anything Model (SAM) vision encoder for promptable image segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SAM (Kirillov et al., 2023) consists of a ViT image encoder producing image embeddings, a prompt
/// encoder that handles points/boxes/masks, and a lightweight mask decoder. The image encoder uses
/// windowed attention with occasional global attention blocks for efficiency at high resolution.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Segment Anything" (Kirillov et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> SAM (Segment Anything Model) from Meta is a promptable
/// segmentation model — you give it an image and a prompt (a point click, a bounding box,
/// or a rough mask) and it segments the object you indicated. Trained on over 1 billion masks,
/// its ViT encoder uses windowed attention with occasional global attention for efficient
/// high-resolution processing. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SAM model for promptable image segmentation
/// // using ViT encoder with windowed attention for high-resolution images
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 256);
///
/// // ONNX inference mode with pre-trained model
/// var model = new SAM&lt;double&gt;(architecture, "sam.onnx");
///
/// // Training mode with native layers
/// var trainModel = new SAM&lt;double&gt;(architecture, new SAMOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Segment Anything",
    "https://arxiv.org/abs/2304.02643",
    Year = 2023,
    Authors = "Kirillov et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, Beta1 = 0.9, Beta2 = 0.999, LearningRate = 8e-4,
                WeightDecay = 0.1, ReferenceBatchSize = 256, WarmupSteps = 250,
                Schedule = LearningRateSchedulerType.MultiStep, DecayRate = 0.1,
                Milestones = [60000, 86666],
                Source = "Kirillov et al. 2023, Training recipe: AdamW with beta1 0.9, beta2 0.999, linear warmup for 250 iterations, initial rate 8e-4 after warmup, decreased 10x at 60k and again at 86666 iterations over a 90k-iteration run, batch size 256, weight decay 0.1.")]
public partial class SAM<T> : VisionLanguageModelBase<T>, IVisualEncoder<T>
{
    private readonly SAMOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public SAM(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SAMOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new SAMOptions();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.EmbeddingDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public SAM(
        NeuralNetworkArchitecture<T> architecture,
        SAMOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new SAMOptions();
        if (architecture.InputType == InputType.ThreeDimensional && architecture.InputHeight > 0)
        {
            _options = new SAMOptions(_options) { ImageSize = architecture.InputHeight };
        }
        _useNativeMode = true;
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.EmbeddingDim;
        InitializeLayers();
    }

    public int EmbeddingDimension => _options.EmbeddingDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);
        var c = p;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultViTLayers(
                    _options.EmbeddingDim,
                    _options.NumLayers,
                    _options.NumHeads,
                    _options.DropoutRate,
                    patchSize: _options.PatchSize
                )
            );
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        TrainWithTape(input, expected, _optimizer);
        SetTrainingMode(false);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SAM-Native" : "SAM-ONNX",
            Description = "Segment Anything Model (Kirillov et al., 2023)",
            FeatureCount = _options.EmbeddingDim,
            Complexity = _options.NumLayers,
        };
        m.AdditionalInfo["Architecture"] = "SAM";
        m.AdditionalInfo["MaskDecoderDim"] = _options.MaskDecoderDim.ToString();
        m.AdditionalInfo["WindowSize"] = _options.WindowSize.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SAM<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
