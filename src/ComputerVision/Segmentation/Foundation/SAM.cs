using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Segment Anything Model (SAM): the first promptable foundation model for image segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM can segment any object in any image given a prompt (point click,
/// bounding box, or text). It was trained on the SA-1B dataset containing over 1 billion masks
/// across 11 million images, making it extremely versatile.
///
/// Common use cases:
/// - Interactive object selection in image editors
/// - Automatic mask generation for datasets
/// - Zero-shot transfer to new domains without fine-tuning
/// - Foundation for downstream segmentation tasks
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - ViT-H/L/B image encoder with 16x16 patch embedding
/// - Lightweight prompt encoder for points, boxes, and masks
/// - Two-way transformer mask decoder with IoU prediction head
/// - Ambiguity-aware: predicts 3 masks per prompt (whole, part, subpart)
/// - 1024x1024 input resolution; encoder runs once per image
/// </para>
/// <para>
/// <b>Reference:</b> Kirillov et al., "Segment Anything", ICCV 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SAM model for promptable image segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new SAM&lt;double&gt;(architecture);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new SAM&lt;double&gt;(architecture,
///     options: new SAMOptions { NumClasses = 1, DropRate = 0.1, ModelSize = SAMModelSize.ViTHuge });
///
/// // Or load a pre-trained ONNX model for zero-shot segmentation
/// var onnxModel = new SAM&lt;double&gt;(architecture, "sam_vit_h.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Segment Anything", "https://arxiv.org/abs/2304.02643", Year = 2023, Authors = "Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick")]
[PaperOptimizer(OptimizerKind.AdamW, Beta1 = 0.9, Beta2 = 0.999, LearningRate = 8e-4,
                WeightDecay = 0.1, ReferenceBatchSize = 256, WarmupSteps = 250,
                Schedule = LearningRateSchedulerType.MultiStep, DecayRate = 0.1,
                Milestones = [60000, 86666],
                Source = "Kirillov et al. 2023, Training recipe: AdamW with beta1 0.9, beta2 0.999, linear warmup for 250 iterations, initial rate 8e-4 after warmup, decreased 10x at 60k and again at 86666 iterations over a 90k-iteration run, batch size 256, weight decay 0.1.")]
public partial class SAM<T> : Common.PromptableSegmentationBase<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// This encoder downsamples by 16, not the family's 32. Measured, not assumed: a [1,3,64,64] input
    /// returns [1,C,4,4] where the inherited /32 law claims [1,C,2,2]. The conformance sweep surfaced
    /// the difference - which is what a virtual family law plus per-model overrides exists for.
    /// </remarks>
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => SpatialStrideContract(inputRank, 16);

    private readonly SAMOptions _options;

    #region Fields

    // Only SAM's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed, _encoderLayerEnd and the
    // promptable state (_imageEmbedding, _imageSet) all come from PromptableSegmentationBase ->
    // SegmentationModelBase.
    private readonly SAMModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    /// <summary>
    /// Builds the paper's mask objective from <paramref name="options"/>.
    /// </summary>
    /// <remarks>
    /// Kirillov et al. 2023 (§3) supervises masks with "a linear combination of focal loss and dice
    /// loss in a 20:1 ratio", focal being the RetinaNet form. Every coefficient comes from
    /// <see cref="SAMOptions"/> — <see cref="SAMOptions.MaskFocalWeight"/>,
    /// <see cref="SAMOptions.MaskDiceWeight"/>, <see cref="SAMOptions.FocalGamma"/> and
    /// <see cref="SAMOptions.FocalAlpha"/> — whose defaults ARE the paper's values, so the objective
    /// is paper-faithful out of the box and fully overridable. Static because it is invoked from the
    /// base-constructor initializer, before instance fields are assigned.
    /// </remarks>
    private static ILossFunction<T> BuildMaskLoss(SAMOptions options)
    {
        // Multi-class masks generalize to softmax CE rather than the binary focal+dice pair.
        if (options.NumClasses != 1)
        {
            return new CrossEntropyWithLogitsLoss<T>();
        }

        return new CompositeLossWithLogits<T>(
            (new FocalLoss<T>(gamma: options.FocalGamma, alpha: options.FocalAlpha), options.MaskFocalWeight),
            (new DiceLoss<T>(), options.MaskDiceWeight));
    }

    // SAM's own promptable state. _imageEmbedding and _imageSet live on PromptableSegmentationBase.
    [Scratch]
    private Tensor<T>? _imageProbabilities;

    #endregion

    #region Properties

    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal SAMModelSize ModelSize => _modelSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes SAM in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function. Default for <paramref name="numClasses"/> == 1 is the paper's objective: a logits-aware <see cref="CompositeLossWithLogits{T}"/> of <see cref="FocalLoss{T}"/> (gamma 2, alpha 0.25) and <see cref="DiceLoss{T}"/> in a 20:1 ratio (Kirillov et al. 2023, §3). Multi-class uses <see cref="CrossEntropyWithLogitsLoss{T}"/>.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable SAM model from scratch. The original SAM was
    /// trained on SA-1B (1B+ masks from 11M images). Fine-tuning on domain data typically uses
    /// ViT-B for efficiency.
    /// </para>
    /// </remarks>
    public SAM(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SAMOptions? options = null)
        // Kirillov et al. 2023 ("Segment Anything", §3 Segment Anything Model / Training) supervises
        // mask prediction with "a linear combination of focal loss and dice loss in a 20:1 ratio",
        // focal being the RetinaNet form (gamma=2, alpha=0.25) the paper cites. The previous plain
        // BinaryCrossEntropyWithLogitsLoss was NOT that objective: it weights every pixel equally,
        // where focal deliberately down-weights easy background and dice corrects the foreground/
        // background imbalance a promptable segmenter depends on. Use the paper's composite for the
        // single-mask case; multi-class keeps softmax CE, which is the correct generalisation.
        //
        // The base resolves numClasses/native-mode and defaults the optimizer, but SAM's default is
        // not the base's plain AdamW - it carries the paper's warmup schedule - so it is still built
        // here, EAGERLY, exactly as before. `_optimizer` is a settable field on SegmentationModelBase
        // precisely so a model can do this.
        : base(architecture, optimizer, lossFunction ?? BuildMaskLoss(options ??= new SAMOptions()), (options ??= new SAMOptions()).NumClasses)
    {
        _options = options;
        Options = _options;
        // SAM's own 1024x1024 input default, which differs from the base's 512x512.
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _modelSize = _options.ModelSize;
        _dropRate = _options.DropRate;
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this,
            optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate,
                    Beta1 = _options.AdamBeta1,
                    Beta2 = _options.AdamBeta2,
                    Epsilon = _options.AdamEpsilon,
                    WeightDecay = _options.WeightDecay,
                    UseAdaptiveLearningRate = false,
                    UseAdaptiveBetas = false,
                    // Kirillov et al. 2023 ("Segment Anything", §A Training algorithm) reaches its
                    // lr only AFTER a LINEAR WARMUP over the first 250 iterations, then holds it until
                    // the step decays at 60k/86.6k of 90k iterations. SAMOptions.LearningRate is that
                    // post-warmup PEAK, so applying it from step 0 was not paper-faithful. Constant
                    // decay after warmup is faithful for every horizon the suite exercises -- the
                    // paper's first step decay is at 60k iterations, far beyond any test.
                    //
                    // Measured effect: this removes the first-step overshoot that drove
                    // Training_ShouldReduceLoss and MoreData_ShouldNotDegrade (loss was RISING
                    // 0.44 -> 29.95) and takes SAM from 3 failures to 1. It does NOT fix
                    // LossStrictlyDecreasesOnMemorizationTask: with warmup the divergence is only
                    // DELAYED (0.694916 -> 7.090303 by step 300, past the 250-step warmup), and the
                    // generated fixture already runs at LearningRate = 1e-5 -- 1/80th of the paper's
                    // peak -- so the remaining divergence is NOT a learning-rate-scale problem. That
                    // is a separate open defect in this model's training path, still under
                    // investigation; the warmup here is correct on its own merits.
                    LearningRateScheduler = new LinearWarmupScheduler(
                        baseLearningRate: _options.LearningRate,
                        warmupSteps: _options.WarmupSteps,
                        totalSteps: 0,
                        // Start at ONE STEP's worth of the peak rather than exactly 0. A 0 start makes the
                        // first optimizer step a no-op, which left parameters bit-identical after a single
                        // Train call and tripped GradientFlow_ShouldBeNonZeroAndFinite ("No parameters
                        // changed after training"). Ramping from base/warmupSteps is the standard linear
                        // warmup (equivalent to PyTorch LinearLR with start_factor = 1/warmup_steps) and
                        // still reaches the paper's peak exactly at step warmupSteps.
                        warmupInitLr: _options.LearningRate / System.Math.Max(1, _options.WarmupSteps),
                        decayMode: LinearWarmupScheduler.DecayMode.Constant),
                    // Warmup is defined per ITERATION, so the schedule must advance per optimizer step.
                    SchedulerStepMode = SchedulerStepMode.StepPerBatch,
                }));

        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes SAM in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained SAM model from ONNX for fast inference.
    /// Download pre-trained weights from Meta's SAM repository.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public SAM(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SAMOptions? options = null)
        // Same paper objective as the native constructor above (focal + dice, 20:1), kept in sync so
        // the two entry points do not disagree about what SAM optimises.
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry and opens the InferenceSession - the same twenty lines this used to repeat.
        : base(architecture, onnxModelPath, (options ??= new SAMOptions()).NumClasses)
    {
        // The base's ONNX constructor installs a plain CrossEntropyWithLogitsLoss because it has no
        // lossFunction parameter. Restore SAM's paper objective so the two entry points still agree
        // about what SAM optimises, exactly as the old `: base(architecture, BuildMaskLoss(...))` did.
        LossFunction = BuildMaskLoss(options);
        _options = options;
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _modelSize = _options.ModelSize;
        _dropRate = _options.DropRate;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through SAM to produce segmentation mask logits.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
        => _useNativeMode ? Forward(input) : PredictOnnx(input);

    #endregion

    #region Private Methods

    // SAM paper: ViT-B = 12 blocks / 768 dim, ViT-L = 24 blocks / 1024 dim, ViT-H = 32 blocks / 1280 dim
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(SAMModelSize modelSize)
    {
        return modelSize switch
        {
            SAMModelSize.ViTBase => ([768, 768, 768, 768], [12, 0, 0, 0], 256),
            SAMModelSize.ViTLarge => ([1024, 1024, 1024, 1024], [24, 0, 0, 0], 256),
            SAMModelSize.ViTHuge => ([1280, 1280, 1280, 1280], [32, 0, 0, 0], 256),
            _ => ([1280, 1280, 1280, 1280], [32, 0, 0, 0], 256)
        };
    }

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var features = input;
        for (int i = 0; i < Layers.Count; i++)
            features = Layers[i].Forward(features);

        if (!hasBatch) features = RemoveBatchDimension(features);
        return features;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var onnxInput = CvTensorOps<T>.ToOnnxFloat(input);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "pixel_values";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var result = CvTensorOps<T>.FromOnnxFloat(outputTensor);
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
    }

    // AddBatchDimension and RemoveBatchDimension come from SegmentationModelBase.

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the ViT encoder and mask decoder layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = _options.EncoderLayerCount ?? Architecture.Layers.Count / 2;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateSAMEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);
            int featureH = _height / 16;
            int featureW = _width / 16;
            var decoderLayers = LayerHelper<T>.CreateSAMDecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, featureH, featureW);
            Layers.AddRange(decoderLayers);
        }
    }

    // UpdateParameters folded one enumeration the base already folds. Removed under AIDN082.
    /// <summary>
    /// Collects metadata describing this SAM model's configuration.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "SAM" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "NumClasses", _numClasses },
            { "ModelSize", _modelSize.ToString() },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count },
            { "EncoderLayerEnd", _encoderLayerEnd }
        },
        ModelDataProvider = () => SerializeForMetadata()
    };

    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and sets _disposed,
    // and SAM owns no further unmanaged resources.

    #endregion

    #region IPromptableSegmentation Implementation

    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase;
    // SupportsPointPrompts/BoxPrompts/MaskPrompts/TextPrompts come from PromptableSegmentationBase
    // with exactly these values (true, true, true, false).

    /// <summary>
    /// Encodes an image into the embedding the prompt methods score against. SAM's encoder is the
    /// full forward pass, so this is simply <see cref="NeuralNetworkBase{T}.Predict"/>.
    /// </summary>
    protected override Tensor<T> EncodeImage(Tensor<T> image) => Predict(image);

    /// <inheritdoc/>
    public override void SetImage(Tensor<T> image)
    {
        // Same two steps as before - encode once, then cache the softmax of that embedding for
        // SegmentEverything - plus the base's _imageSet flag that EnsureImageSet() reads.
        var embedding = EncodeImage(image);
        _imageEmbedding = embedding;
        _imageSet = true;
        _imageProbabilities = Common.SegmentationTensorOps.SoftmaxAlongClassDim(embedding);
    }

    /// <inheritdoc/>
    public override PromptedSegmentationResult<T> SegmentFromPoints(Tensor<T> points, Tensor<T> labels)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var scoreMap = Common.SegmentationTensorOps.PointPromptScoreMap(features, points, labels);
        return BuildPromptMaskResult(scoreMap, features.Shape[1], features.Shape[2]);
    }

    /// <inheritdoc/>
    public override PromptedSegmentationResult<T> SegmentFromBox(Tensor<T> box)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var scoreMap = Common.SegmentationTensorOps.BoxPromptScoreMap(features, box);
        return BuildPromptMaskResult(scoreMap, features.Shape[1], features.Shape[2]);
    }

    /// <inheritdoc/>
    public override PromptedSegmentationResult<T> SegmentFromMask(Tensor<T> mask)
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var scoreMap = Common.SegmentationTensorOps.MaskPromptScoreMap(features, mask);
        return BuildPromptMaskResult(scoreMap, features.Shape[1], features.Shape[2]);
    }

    /// <inheritdoc/>
    public override List<PromptedSegmentationResult<T>> SegmentEverything()
    {
        var features = _imageEmbedding ?? Predict(new Tensor<T>([_channels, _height, _width]));
        var probs = _imageProbabilities ?? Common.SegmentationTensorOps.SoftmaxAlongClassDim(features);
        return Common.SegmentationTensorOps.ClassMaskCandidates(features, probs);
    }

    private PromptedSegmentationResult<T> BuildPromptMaskResult(Tensor<T> scoreMap, int h, int w)
        => Common.SegmentationTensorOps.BuildPromptMaskResult(scoreMap, h, w);

    #endregion
}
