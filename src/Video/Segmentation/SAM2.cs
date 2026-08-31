using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Segmentation;

/// <summary>
/// Segment Anything Model 2 (SAM2) for video object segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM2 is a powerful model that can segment any object in video.
/// You can interact with it by:
/// - Clicking on an object in the first frame to select it
/// - Drawing a bounding box around objects
/// - Providing text prompts describing what to segment
///
/// Once you identify an object, SAM2 automatically tracks and segments it across
/// all frames in the video, even when the object moves, rotates, or is partially occluded.
///
/// Common use cases:
/// - Video editing (isolating subjects for effects)
/// - Object tracking and analysis
/// - Video annotation and labeling
/// - Interactive video manipulation
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Memory attention mechanism for temporal consistency
/// - Hierarchical image encoder (similar to MAE/ViT)
/// - Prompt encoder for points, boxes, and masks
/// - Mask decoder with occlusion prediction
/// - Memory bank for efficient object tracking
/// </para>
/// <para>
/// <b>Reference:</b> Ravi et al., "SAM 2: Segment Anything in Images and Videos"
/// Meta AI, 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SAM2 model for interactive video object segmentation
/// var sam2 = new SAM2&lt;double&gt;();
///
/// // Or configure with a specific model size and memory bank
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 1);
/// var model = new SAM2&lt;double&gt;(architecture, modelSize: SAM2ModelSize.Large, memoryBankSize: 7);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SAM 2: Segment Anything in Images and Videos",
    "https://arxiv.org/abs/2408.00714",
    Year = 2024,
    Authors = "Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, Christoph Feichtenhofer")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class SAM2<T> : NeuralNetworkBase<T>
{
    private readonly SAM2Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numFeatures;
    private readonly int _memoryBankSize;
    private readonly SAM2ModelSize _modelSize;
    private readonly bool _useNativeMode;
    private readonly string? _onnxModelPath;
    private InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    // Memory bank for tracking
    [Scratch]
    private readonly List<Tensor<T>> _memoryBank;
    [Scratch]
    private readonly List<Tensor<T>> _memoryObjectPointers;
    [Scratch]
    private readonly List<int> _memoryFrameIndices;

    [Scratch]
    private int[]? _lastTrainingMaskIndices;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether training is supported.
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// SAM2's public masks are sigmoid probabilities, so supervised targets must
    /// be finite values in the closed unit interval.
    /// </summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) =>
        UnitIntervalTensorDomain.Value;

    /// <summary>
    /// SAM2's decoder selects one mask through a data-dependent IoU argmax and its video path also
    /// reads mutable memory-bank state. Those branches cannot be captured once and safely replayed
    /// by a static fused-training plan. Keep the real paper topology on the eager autodiff tape so
    /// every step evaluates the current scores, selects the current mask, and updates live weights.
    /// </summary>
    protected override bool SupportsFusedCompiledTraining => false;

    /// <summary>
    /// Gets the input height.
    /// </summary>
    internal int InputHeight => _height;

    /// <summary>
    /// Gets the input width.
    /// </summary>
    internal int InputWidth => _width;

    /// <summary>
    /// Gets the model size variant.
    /// </summary>
    internal SAM2ModelSize ModelSize => _modelSize;

    /// <summary>
    /// Gets whether using native mode (trainable) or ONNX mode (inference only).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the current memory bank size.
    /// </summary>
    internal int CurrentMemorySize => _memoryBank.Count;

    /// <summary>Gets the paper-topology native pipeline when this model owns its default layers.</summary>
    internal SAM2NativePipelineLayer<T>? NativePipeline =>
        Layers.Count == 1 ? Layers[0] as SAM2NativePipelineLayer<T> : null;

    /// <summary>Gets the oracle mask assignments used by the latest native training step.</summary>
    internal IReadOnlyList<int> LastTrainingMaskIndices => _lastTrainingMaskIndices
        ?? throw new InvalidOperationException("Train SAM2 before reading mask assignments.");

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the SAM2 class in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="optimizer">Optional optimizer for training (default: null uses layer-wise learning).</param>
    /// <param name="lossFunction">Optional loss function (default: BinaryCrossEntropyLoss).</param>
    /// <param name="modelSize">The model size variant (Tiny, Small, Base, Large).</param>
    /// <param name="memoryBankSize">Maximum number of frames to keep in memory.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a trainable SAM2 model.
    /// Use this when you want to fine-tune the model on your own video data.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public SAM2()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.BinaryClassification,
            inputHeight: 256, inputWidth: 256, inputDepth: 3,
            outputSize: 1))
    {
    }

    /// <summary>
    /// Builds the paper's mask objective from <paramref name="options"/>.
    /// </summary>
    /// <remarks>
    /// Every coefficient comes from <see cref="SAM2Options"/> -- <see cref="SAM2Options.MaskFocalWeight"/>,
    /// <see cref="SAM2Options.MaskDiceWeight"/>, <see cref="SAM2Options.FocalGamma"/> and
    /// <see cref="SAM2Options.FocalAlpha"/> -- whose defaults ARE the paper's values, so the objective is
    /// paper-faithful out of the box and fully overridable. Static because it is invoked from the
    /// base-constructor initializer, before instance fields are assigned.
    /// </remarks>
    private static ILossFunction<T> BuildMaskLoss(SAM2Options? options)
    {
        var o = options ?? new SAM2Options();
        return new CompositeLoss<T>(
            (new FocalLoss<T>(gamma: o.FocalGamma, alpha: o.FocalAlpha), o.MaskFocalWeight),
            (new DiceLoss<T>(), o.MaskDiceWeight));
    }
    private static void ValidateOptions(SAM2Options options)
    {
        if (options.ModelDimension <= 0 || options.ModelDimension > 256)
            throw new ArgumentOutOfRangeException(nameof(options.ModelDimension));
        if (options.MemoryDimension <= 0 || options.ModelDimension % options.MemoryDimension != 0)
            throw new ArgumentOutOfRangeException(nameof(options.MemoryDimension));
        if (options.DecoderHeadCount <= 0 || options.ModelDimension % options.DecoderHeadCount != 0)
            throw new ArgumentOutOfRangeException(nameof(options.DecoderHeadCount));
        if (options.MemoryAttentionLayerCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(options.MemoryAttentionLayerCount));
        if (options.MaskDecoderDepth <= 0)
            throw new ArgumentOutOfRangeException(nameof(options.MaskDecoderDepth));
        if (options.MaskDecoderMlpDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options.MaskDecoderMlpDimension));
        if (double.IsNaN(options.RopeTheta) || double.IsInfinity(options.RopeTheta) || options.RopeTheta <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(options.RopeTheta));
        if (double.IsNaN(options.MemoryMaskScale) || double.IsInfinity(options.MemoryMaskScale) || options.MemoryMaskScale <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(options.MemoryMaskScale));
        if (double.IsNaN(options.MemoryMaskBias) || double.IsInfinity(options.MemoryMaskBias))
            throw new ArgumentOutOfRangeException(nameof(options.MemoryMaskBias));
        if (options.HieraEmbeddingDimension is <= 0)
            throw new ArgumentOutOfRangeException(nameof(options.HieraEmbeddingDimension));
        if (options.HieraInitialHeadCount is <= 0)
            throw new ArgumentOutOfRangeException(nameof(options.HieraInitialHeadCount));
        if (options.HieraStageDepths is { Length: not 4 }
            || options.HieraStageDepths?.Any(x => x <= 0) == true)
            throw new ArgumentException("HieraStageDepths must contain four positive values.", nameof(options));
        if (options.HieraWindowSizes is { Length: not 4 }
            || options.HieraWindowSizes?.Any(x => x <= 0) == true)
            throw new ArgumentException("HieraWindowSizes must contain four positive values.", nameof(options));
        if (options.HieraGlobalAttentionBlockIndexes?.Any(x => x < 0) == true
            || options.HieraGlobalAttentionBlockIndexes?.Distinct().Count()
                != options.HieraGlobalAttentionBlockIndexes?.Length)
            throw new ArgumentException(
                "Hiera global-attention indexes must be unique and non-negative.", nameof(options));
        if (options.MaskFocalWeight < 0.0 || options.MaskDiceWeight < 0.0
            || options.IouLossWeight < 0.0 || options.ObjectPresenceLossWeight < 0.0)
            throw new ArgumentOutOfRangeException(nameof(options), "Loss weights cannot be negative.");
    }

    internal static (int Embedding, int[] Depths, int Heads, int[] Windows, int[] Globals)
        ResolveHieraPreset(SAM2ModelSize size, SAM2Options options)
    {
        (int Embedding, int[] Depths, int Heads, int[] Windows, int[] Globals) preset = size switch
        {
            SAM2ModelSize.Tiny => (
                96, [1, 2, 7, 2], 1, [8, 4, 14, 7], [5, 7, 9]),
            SAM2ModelSize.Small => (
                96, [1, 2, 11, 2], 1, [8, 4, 14, 7], [7, 10, 13]),
            SAM2ModelSize.Base => (
                112, [2, 3, 16, 3], 2, [8, 4, 14, 7], [12, 16, 20]),
            SAM2ModelSize.Large => (
                144, [2, 6, 36, 4], 2, [8, 4, 16, 8], [23, 33, 43]),
            _ => throw new ArgumentOutOfRangeException(nameof(size))
        };

        int[] depths = options.HieraStageDepths?.ToArray() ?? preset.Depths;
        int[] globals = options.HieraGlobalAttentionBlockIndexes?.ToArray()
            ?? (depths.SequenceEqual(preset.Depths)
                ? preset.Globals
                : [depths[0] + depths[1] + depths[2] - 1]);

        return (
            options.HieraEmbeddingDimension ?? preset.Embedding,
            depths,
            options.HieraInitialHeadCount ?? preset.Heads,
            options.HieraWindowSizes?.ToArray() ?? preset.Windows,
            globals);
    }


    public SAM2(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SAM2ModelSize modelSize = SAM2ModelSize.Base,
        int memoryBankSize = 7,
        SAM2Options? options = null)
        // SAM 2 (Ravi et al. 2024, §D) inherits SAM's mask supervision: "a linear combination of focal
        // and dice loss" in a 20:1 ratio, focal at the RetinaNet gamma=2 / alpha=0.25 the original SAM
        // paper cites. Plain BinaryCrossEntropyLoss weights every pixel equally and left the
        // memorization loss sitting exactly at ln(2) = 0.693147, unchanged across 15 steps, because
        // DecodeMask emits sigmoid-activated masks and BCE at p=0.5 is stationary. Both DecodeMask's
        // sigmoid and these losses expect probabilities, so the composite is applied directly.
        : base(architecture, lossFunction ?? BuildMaskLoss(options))
    {
        _options = options ?? new SAM2Options();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _modelSize = modelSize;
        if (memoryBankSize <= 0) throw new ArgumentOutOfRangeException(nameof(memoryBankSize));
        _memoryBankSize = memoryBankSize;
        _useNativeMode = true;
        _onnxModelPath = null;
        _optimizer = optimizer;
        ValidateOptions(_options);
        _numFeatures = _options.ModelDimension;

        _memoryBank = [];
        _memoryObjectPointers = [];
        _memoryFrameIndices = [];

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the SAM2 class in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="modelSize">The model size variant for configuration.</param>
    /// <param name="memoryBankSize">Maximum number of frames to keep in memory.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor loads a pre-trained SAM2 model from ONNX format.
    /// Use this for fast inference when you don't need to train the model.
    /// Download pre-trained models from Meta's SAM2 repository.
    /// </para>
    /// </remarks>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX model fails to load.</exception>
    public SAM2(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SAM2ModelSize modelSize = SAM2ModelSize.Base,
        int memoryBankSize = 7,
        SAM2Options? options = null)
        : base(architecture, new BinaryCrossEntropyLoss<T>())
    {
        _options = options ?? new SAM2Options();
        Options = _options;
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"SAM2 ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _modelSize = modelSize;
        if (memoryBankSize <= 0) throw new ArgumentOutOfRangeException(nameof(memoryBankSize));
        _memoryBankSize = memoryBankSize;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        ValidateOptions(_options);
        _numFeatures = _options.ModelDimension;

        _memoryBank = [];
        _memoryObjectPointers = [];
        _memoryFrameIndices = [];

        // Initialize ONNX session
        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load SAM2 ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Segments objects in an image given point prompts.
    /// </summary>
    /// <param name="image">The input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <param name="points">Point coordinates [[x, y], ...] for foreground/background.</param>
    /// <param name="pointLabels">Label for each point: 1 for foreground, 0 for background.</param>
    /// <returns>Segmentation mask tensor [H, W] or [B, H, W] with values in [0, 1].</returns>
    public Tensor<T> SegmentWithPoints(Tensor<T> image, float[,] points, int[] pointLabels)
    {
        bool hasBatch = image.Rank == 4;
        if (!hasBatch)
        {
            image = AddBatchDimension(image);
        }

        // Encode image
        var imageFeatures = EncodeImage(image);

        // Encode point prompts
        var pointFeatures = EncodePoints(points, pointLabels);

        // Apply memory attention if we have previous frames
        imageFeatures = PrepareMemoryConditioning(imageFeatures);

        // Decode mask
        var masks = DecodeMask(imageFeatures, pointFeatures, null, null);

        // Select best mask
        var bestMask = SelectBestMask(masks.Masks, masks.IouScores);

        // Upsample to original resolution
        var outputMask = UpsampleMask(bestMask, _height, _width);

        if (!hasBatch)
        {
            outputMask = RemoveBatchDimension(outputMask);
        }

        return outputMask;
    }

    /// <summary>
    /// Segments objects in an image given a bounding box.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="box">Bounding box [x1, y1, x2, y2] in pixel coordinates.</param>
    /// <returns>Segmentation mask tensor.</returns>
    public Tensor<T> SegmentWithBox(Tensor<T> image, float[] box)
    {
        bool hasBatch = image.Rank == 4;
        if (!hasBatch)
        {
            image = AddBatchDimension(image);
        }

        var imageFeatures = EncodeImage(image);
        var boxFeatures = EncodeBox(box);

        imageFeatures = PrepareMemoryConditioning(imageFeatures);

        var masks = DecodeMask(imageFeatures, null, boxFeatures, null);
        var bestMask = SelectBestMask(masks.Masks, masks.IouScores);
        var outputMask = UpsampleMask(bestMask, _height, _width);

        if (!hasBatch)
        {
            outputMask = RemoveBatchDimension(outputMask);
        }

        return outputMask;
    }

    /// <summary>
    /// Segments objects using a mask prompt (for refinement).
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="maskPrompt">Low-resolution mask prompt [H/4, W/4].</param>
    /// <returns>Refined segmentation mask tensor.</returns>
    public Tensor<T> SegmentWithMask(Tensor<T> image, Tensor<T> maskPrompt)
    {
        bool hasBatch = image.Rank == 4;
        if (!hasBatch)
        {
            image = AddBatchDimension(image);
            maskPrompt = AddBatchDimension(maskPrompt);
        }

        var imageFeatures = EncodeImage(image);
        var maskFeatures = EncodeMaskPrompt(maskPrompt);

        imageFeatures = PrepareMemoryConditioning(imageFeatures);

        var masks = DecodeMask(imageFeatures, null, null, maskFeatures);
        var bestMask = SelectBestMask(masks.Masks, masks.IouScores);
        var outputMask = UpsampleMask(bestMask, _height, _width);

        if (!hasBatch)
        {
            outputMask = RemoveBatchDimension(outputMask);
        }

        return outputMask;
    }

    /// <summary>
    /// Tracks and segments an object across video frames.
    /// </summary>
    /// <param name="frames">List of video frames.</param>
    /// <param name="initialPoints">Point prompts for the first frame.</param>
    /// <param name="pointLabels">Labels for initial points.</param>
    /// <returns>List of segmentation masks for each frame.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main video tracking method.
    /// Simply provide the initial frame with point clicks to identify objects,
    /// and SAM2 will automatically track and segment those objects in all
    /// subsequent frames.
    /// </para>
    /// </remarks>
    public List<Tensor<T>> TrackObject(List<Tensor<T>> frames, float[,] initialPoints, int[] pointLabels)
    {
        ClearMemory();
        var masks = new List<Tensor<T>>();

        for (int i = 0; i < frames.Count; i++)
        {
            var frame = frames[i];
            bool hasBatch = frame.Rank == 4;
            if (!hasBatch)
            {
                frame = AddBatchDimension(frame);
            }

            // Encode current frame
            var imageFeatures = EncodeImage(frame);
            var conditionedFeatures = PrepareMemoryConditioning(imageFeatures);

            Tensor<T> mask;
            if (i == 0)
            {
                var pointFeatures = EncodePoints(initialPoints, pointLabels);
                var maskResult = DecodeMask(conditionedFeatures, pointFeatures, null, null);
                mask = SelectBestMask(maskResult.Masks, maskResult.IouScores);
            }
            else
            {
                var maskResult = DecodeMask(conditionedFeatures, null, null, null);
                mask = SelectBestMask(maskResult.Masks, maskResult.IouScores);
            }

            // Update memory bank
            UpdateMemoryBank(imageFeatures, mask, i);

            // Upsample mask to original resolution
            var outputMask = UpsampleMask(mask, _height, _width);
            if (!hasBatch)
            {
                outputMask = RemoveBatchDimension(outputMask);
            }

            masks.Add(outputMask);
        }

        return masks;
    }

    /// <summary>
    /// Performs interactive video segmentation with refinement.
    /// </summary>
    /// <param name="frames">List of video frames.</param>
    /// <param name="framePrompts">Dictionary of frame index to prompts for refinement.</param>
    /// <returns>List of refined segmentation masks.</returns>
    public List<Tensor<T>> InteractiveVideoSegmentation(
        List<Tensor<T>> frames,
        Dictionary<int, (float[,] Points, int[] Labels)> framePrompts)
    {
        ClearMemory();
        var masks = new List<Tensor<T>>();

        for (int i = 0; i < frames.Count; i++)
        {
            var frame = frames[i];
            bool hasBatch = frame.Rank == 4;
            if (!hasBatch)
            {
                frame = AddBatchDimension(frame);
            }

            var imageFeatures = EncodeImage(frame);
            var conditionedFeatures = PrepareMemoryConditioning(imageFeatures);

            Tensor<T> mask;
            if (framePrompts.TryGetValue(i, out var prompts))
            {
                var pointFeatures = EncodePoints(prompts.Points, prompts.Labels);
                var maskResult = DecodeMask(conditionedFeatures, pointFeatures, null, null);
                mask = SelectBestMask(maskResult.Masks, maskResult.IouScores);
            }
            else
            {
                var maskResult = DecodeMask(conditionedFeatures, null, null, null);
                mask = SelectBestMask(maskResult.Masks, maskResult.IouScores);
            }

            UpdateMemoryBank(imageFeatures, mask, i);

            var outputMask = UpsampleMask(mask, _height, _width);
            if (!hasBatch)
            {
                outputMask = RemoveBatchDimension(outputMask);
            }

            masks.Add(outputMask);
        }

        return masks;
    }

    /// <summary>
    /// Gets the occlusion score for the current segmentation.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="points">Point prompts.</param>
    /// <param name="pointLabels">Point labels.</param>
    /// <returns>Occlusion score in [0, 1] where 1 means fully occluded.</returns>
    public double GetOcclusionScore(Tensor<T> image, float[,] points, int[] pointLabels)
    {
        bool hasBatch = image.Rank == 4;
        if (!hasBatch)
        {
            image = AddBatchDimension(image);
        }

        var imageFeatures = EncodeImage(image);
        var pointFeatures = EncodePoints(points, pointLabels);

        imageFeatures = PrepareMemoryConditioning(imageFeatures);

        var masks = DecodeMask(imageFeatures, pointFeatures, null, null);
        return masks.OcclusionScore;
    }

    /// <summary>
    /// Clears the memory bank for starting a new video.
    /// </summary>
    public void ClearMemory()
    {
        _memoryBank.Clear();
        _memoryObjectPointers.Clear();
        _memoryFrameIndices.Clear();
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // Default: segment with automatic mode (no prompts)
        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
        {
            input = AddBatchDimension(input);
        }

        var imageFeatures = EncodeImage(input);

        imageFeatures = PrepareMemoryConditioning(imageFeatures);

        var masks = DecodeMask(imageFeatures, null, null, null);
        var bestMask = SelectBestMask(masks.Masks, masks.IouScores);
        var outputMask = UpsampleMask(bestMask, _height, _width);

        if (!hasBatch)
        {
            outputMask = RemoveBatchDimension(outputMask);
        }

        return outputMask;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// SAM2's four modules are PARALLEL BRANCHES, not one sequence: the prompt encoder consumes
    /// prompts, the memory attention consumes the memory bank, and the mask decoder consumes the
    /// fused features. The base implementation walks <c>Layers</c> in order, which feeds the
    /// memory/mask convolutions the wrong spatial extent and throws "Input spatial dims after
    /// padding (1, 256) must be >= kernelSize (4)". Run the model's REAL topology instead -- the
    /// same path <see cref="PredictCore"/> takes -- while keeping the base class's seed wiring so
    /// stochastic layers stay reproducible across runs.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();
        if (!_useNativeMode)
        {
            return activations;
        }

        // Same reason as ForwardForTraining below: the base implementation walks the flat Layers list
        // sequentially, which is invalid for SAM2's parallel branches and throws "Input spatial dims
        // after padding (1, 256) must be >= kernelSize (4)". Capture along the REAL path instead,
        // mirroring the overrides SpeechEmotionRecognizer and RWKVForecaster use.
        bool hasBatch = input.Rank == 4;
        var batched = hasBatch ? input : AddBatchDimension(input);

        var imageFeatures = EncodeImage(batched);
        activations["ImageEncoder"] = imageFeatures.Clone();

        imageFeatures = PrepareMemoryConditioning(imageFeatures);
        activations[_memoryBank.Count > 0 ? "MemoryAttention" : "NoMemoryEmbedding"] = imageFeatures.Clone();

        var decoded = DecodeMask(imageFeatures, null, null, null);
        activations["MaskDecoder"] = decoded.Masks.Clone();
        activations["IouScores"] = decoded.IouScores.Clone();

        var selected = SelectBestMask(decoded.Masks, decoded.IouScores);
        activations["SelectedMask"] = selected.Clone();
        activations["UpsampledMask"] = UpsampleMask(selected, _height, _width).Clone();

        return activations;
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        EnsureLayerRandomSeedsWired();

        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
        {
            input = AddBatchDimension(input);
        }

        var imageFeatures = EncodeImage(input);

        imageFeatures = PrepareMemoryConditioning(imageFeatures);

        var masks = DecodeMask(imageFeatures, null, null, null);
        var bestMask = SelectBestMask(masks.Masks, masks.IouScores);
        var outputMask = UpsampleMask(bestMask, _height, _width);

        if (!hasBatch)
        {
            outputMask = RemoveBatchDimension(outputMask);
        }

        return outputMask;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException(
                "Training is not supported in ONNX mode. Use native mode constructor for training.");
        }

        if (NativePipeline is null)
        {
            TrainWithTape(input, expectedOutput, _optimizer);
            return;
        }

        _ = TrainWithCustomLoss(
            input,
            output => ApplyCompositeObjective(
                ComputeNativeTrainingLoss(output, expectedOutput), input),
            _optimizer);
    }

    private Tensor<T> ComputeNativeTrainingLoss(
        Tensor<T> selectedOutput, Tensor<T> expectedOutput)
    {
        var pipeline = NativePipeline
            ?? throw new InvalidOperationException("The native SAM2 pipeline is not initialized.");
        int batchSize = pipeline.LastMaskLogits.Shape[0];
        int targetHeight = selectedOutput.Shape[^2];
        int targetWidth = selectedOutput.Shape[^1];
        var target = NormalizeTrainingMaskTarget(
            expectedOutput, batchSize, targetHeight, targetWidth);
        var maskLogits = Engine.Interpolate(
            pipeline.LastMaskLogits,
            [targetHeight, targetWidth],
            InterpolateMode.Bilinear,
            alignCorners: false);
        var candidateTargets = Engine.TensorTile(
            target, [1, pipeline.CandidateCount, 1, 1]);
        var binaryIouTargets = BuildBinaryIouTargets(maskLogits, candidateTargets);
        var objectTargets = BuildObjectPresenceTargets(target);

        Tensor<T>? maskLossSum = null;
        Tensor<T>? iouLossSum = null;
        var selectedIndices = new int[batchSize];

        // Meta's MultiStepMultiMasksAndIous supervises the candidate with the lowest
        // 20*focal + dice loss. Predicted-IoU argmax is an inference decision, not the
        // training assignment rule.
        for (int batch = 0; batch < batchSize; batch++)
        {
            Tensor<T>? bestMaskLoss = null;
            double bestMaskLossValue = double.PositiveInfinity;
            int bestCandidate = 0;

            for (int candidate = 0; candidate < pipeline.CandidateCount; candidate++)
            {
                var oneBatchLogits = Engine.TensorNarrow(maskLogits, 0, batch, 1);
                var oneMaskLogits = Engine.TensorNarrow(oneBatchLogits, 1, candidate, 1);
                var oneBatchTargets = Engine.TensorNarrow(candidateTargets, 0, batch, 1);
                var oneMaskTarget = Engine.TensorNarrow(oneBatchTargets, 1, candidate, 1);
                var candidateLoss = ComputeSamMaskTapeLoss(oneMaskLogits, oneMaskTarget);
                double candidateLossValue = Convert.ToDouble(candidateLoss[0]);
                if (double.IsNaN(candidateLossValue) || double.IsInfinity(candidateLossValue))
                {
                    throw new InvalidOperationException(
                        $"SAM2 produced a non-finite paper mask loss for batch {batch}, " +
                        $"candidate {candidate}. Refusing to fall back to another mask because " +
                        "that would hide a broken gradient/loss path.");
                }

                if (candidateLossValue < bestMaskLossValue)
                {
                    bestMaskLossValue = candidateLossValue;
                    bestMaskLoss = candidateLoss;
                    bestCandidate = candidate;
                }
            }

            selectedIndices[batch] = bestCandidate;
            T objectPresent = objectTargets[batch, 0];
            var supervisedMaskLoss = Engine.TensorMultiplyScalar(
                bestMaskLoss
                    ?? throw new InvalidOperationException("SAM2 emitted no mask candidates."),
                objectPresent);
            maskLossSum = maskLossSum is null
                ? supervisedMaskLoss
                : Engine.TensorAdd(maskLossSum, supervisedMaskLoss);

            var oneBatchIouLogits = Engine.TensorNarrow(
                pipeline.LastIouLogits, 0, batch, 1);
            var selectedIouLogit = Engine.TensorNarrow(
                oneBatchIouLogits, 1, bestCandidate, 1);
            var selectedIouTarget = new Tensor<T>([1, 1]);
            selectedIouTarget[0, 0] = binaryIouTargets[batch, bestCandidate];
            var iouError = Engine.TensorSubtract(selectedIouLogit, selectedIouTarget);
            var iouSquaredError = Engine.TensorMultiply(iouError, iouError);
            var selectedIouLoss = Engine.Reshape(
                Engine.ReduceMean(iouSquaredError, [0, 1], keepDims: false), [1]);
            selectedIouLoss = Engine.TensorMultiplyScalar(selectedIouLoss, objectPresent);
            iouLossSum = iouLossSum is null
                ? selectedIouLoss
                : Engine.TensorAdd(iouLossSum, selectedIouLoss);
        }

        _lastTrainingMaskIndices = selectedIndices;
        var maskLoss = Engine.TensorDivideScalar(
            maskLossSum
                ?? throw new InvalidOperationException("SAM2 emitted an empty training batch."),
            NumOps.FromDouble(batchSize));
        var iouLoss = Engine.TensorDivideScalar(
            iouLossSum
                ?? throw new InvalidOperationException("SAM2 emitted an empty IoU batch."),
            NumOps.FromDouble(batchSize));
        var objectLoss = new BinaryCrossEntropyLoss<T>().ComputeTapeLoss(
            Engine.Sigmoid(pipeline.LastObjectPresenceLogits), objectTargets);

        var total = maskLoss;
        if (_options.IouLossWeight != 0.0)
        {
            total = Engine.TensorAdd(
                total,
                Engine.TensorMultiplyScalar(
                    iouLoss, NumOps.FromDouble(_options.IouLossWeight)));
        }
        if (_options.ObjectPresenceLossWeight != 0.0)
        {
            total = Engine.TensorAdd(
                total,
                Engine.TensorMultiplyScalar(
                    objectLoss, NumOps.FromDouble(_options.ObjectPresenceLossWeight)));
        }
        return total;
    }

    private Tensor<T> ComputeSamMaskTapeLoss(Tensor<T> maskLogits, Tensor<T> target)
    {
        var probabilities = Engine.Sigmoid(maskLogits);
        var focal = new FocalLoss<T>(_options.FocalGamma, _options.FocalAlpha)
            .ComputeTapeLoss(probabilities, target);
        var dice = ComputeSamDiceTapeLoss(probabilities, target);
        return Engine.TensorAdd(
            Engine.TensorMultiplyScalar(
                focal, NumOps.FromDouble(_options.MaskFocalWeight)),
            Engine.TensorMultiplyScalar(
                dice, NumOps.FromDouble(_options.MaskDiceWeight)));
    }

    private Tensor<T> ComputeSamDiceTapeLoss(Tensor<T> probabilities, Tensor<T> target)
    {
        var intersection = Engine.TensorMultiply(probabilities, target);
        int[] axes = Enumerable.Range(0, intersection.Shape.Length).ToArray();
        var intersectionSum = Engine.ReduceSum(intersection, axes, keepDims: false);
        var probabilitySum = Engine.ReduceSum(probabilities, axes, keepDims: false);
        var targetSum = Engine.ReduceSum(target, axes, keepDims: false);
        var numerator = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(intersectionSum, NumOps.FromDouble(2.0)),
            NumOps.One);
        var denominator = Engine.TensorAddScalar(
            Engine.TensorAdd(probabilitySum, targetSum),
            NumOps.One);
        var dice = Engine.TensorDivide(
            Engine.Reshape(numerator, [1]),
            Engine.Reshape(denominator, [1]));
        return Engine.ScalarMinusTensor(NumOps.One, dice);
    }

    private Tensor<T> BuildBinaryIouTargets(
        Tensor<T> maskLogits, Tensor<T> candidateTargets)
    {
        var result = new Tensor<T>([maskLogits.Shape[0], maskLogits.Shape[1]]);
        for (int batch = 0; batch < maskLogits.Shape[0]; batch++)
        for (int candidate = 0; candidate < maskLogits.Shape[1]; candidate++)
        {
            int intersection = 0;
            int union = 0;
            for (int h = 0; h < maskLogits.Shape[2]; h++)
            for (int w = 0; w < maskLogits.Shape[3]; w++)
            {
                bool predicted = Convert.ToDouble(maskLogits[batch, candidate, h, w]) > 0.0;
                bool expected = Convert.ToDouble(candidateTargets[batch, candidate, h, w]) > 0.0;
                if (predicted && expected) intersection++;
                if (predicted || expected) union++;
            }
            result[batch, candidate] = NumOps.FromDouble(
                intersection / (double)Math.Max(union, 1));
        }
        return result;
    }

    private Tensor<T> BuildObjectPresenceTargets(Tensor<T> target)
    {
        var result = new Tensor<T>([target.Shape[0], 1]);
        int pixelsPerBatch = target.Length / target.Shape[0];
        for (int batch = 0; batch < target.Shape[0]; batch++)
        {
            int start = batch * pixelsPerBatch;
            bool present = false;
            for (int i = 0; i < pixelsPerBatch; i++)
            {
                if (Convert.ToDouble(target[start + i]) > 0.0)
                {
                    present = true;
                    break;
                }
            }
            result[batch, 0] = present ? NumOps.One : NumOps.Zero;
        }
        return result;
    }

    private Tensor<T> NormalizeTrainingMaskTarget(
        Tensor<T> target, int batch, int height, int width)
    {
        if (target.Rank == 2)
        {
            target = Engine.Reshape(target, [1, 1, target.Shape[0], target.Shape[1]]);
        }
        else if (target.Rank == 3)
        {
            target = Engine.Reshape(
                target, [target.Shape[0] == batch ? batch : 1, 1, target.Shape[1], target.Shape[2]]);
        }
        if (target.Rank != 4 || target.Shape[1] != 1)
        {
            throw new ArgumentException(
                "SAM2 training targets must be [H,W], [B,H,W], or [B,1,H,W].",
                nameof(target));
        }
        if (target.Shape[0] == 1 && batch > 1)
            target = Engine.TensorTile(target, [batch, 1, 1, 1]);
        else if (target.Shape[0] != batch)
            throw new ArgumentException("SAM2 target batch size does not match the input batch.", nameof(target));
        if (target.Shape[2] != height || target.Shape[3] != width)
        {
            target = Engine.Interpolate(
                target, [height, width], InterpolateMode.Bilinear, alignCorners: false);
        }
        for (int i = 0; i < target.Length; i++)
        {
            double value = Convert.ToDouble(target[i]);
            if (double.IsNaN(value) || double.IsInfinity(value) || value < 0.0 || value > 1.0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(target),
                    $"SAM2 mask target at index {i} must be finite and in [0,1], but was {value:R}.");
            }
        }
        return target;
    }

    #endregion

    #region Private Methods

    /// <summary>
    /// Reshapes a head's [batch, channels] score output to the [batch, channels, 1, 1] rank the rest
    /// of the decoder reads, leaving an already-rank-4 tensor untouched.
    /// </summary>
    /// <remarks>
    /// Uses <c>Engine.Reshape</c>, never <c>tensor.Reshape</c>: the latter is not a recorded op, so it
    /// would sever the gradient tape between the IoU / occlusion heads and the loss and those heads
    /// would silently never train.
    /// </remarks>
    private Tensor<T> NormalizeScoreRank(Tensor<T> scores, int channels)
    {
        int batch = scores.Shape[0];
        int emitted = scores.Length / batch;
        if (emitted != channels)
        {
            throw new InvalidOperationException(
                $"SAM2 score head emitted {emitted} values per batch item but {channels} were expected. " +
                "Check the head factory in LayerHelper against RecordModuleSpans.");
        }

        // Canonicalise by ELEMENT COUNT, not by rank: a rank-4 tensor is not necessarily
        // [batch, channels, 1, 1]. GlobalPoolingLayer + DenseLayer emits the channel values on a
        // trailing axis, so shape[1] was 1 and SelectBestMask's iouScores[b, m, 0, 0] threw
        // "Index 1 is out of range" for m >= 1. Reshaping unconditionally puts the candidate scores on
        // axis 1 whatever layout the head used.
        if (scores.Rank == 4 && scores.Shape[1] == channels)
        {
            return scores;
        }

        return Engine.Reshape(scores, new[] { batch, channels, 1, 1 });
    }

    /// <summary>Runs a half-open span of <see cref="NeuralNetworkBase{T}.Layers"/> in order.</summary>
    private Tensor<T> RunSpan(Tensor<T> input, int start, int end)
    {
        var x = input;
        for (int i = start; i < end; i++)
        {
            x = Layers[i].Forward(x);
        }

        return x;
    }

    /// <summary>
    /// Runs one of SAM2's global-pooling score heads, canonicalizing pooled NCHW features to the
    /// two-dimensional layout expected by the head's Dense layer.
    /// </summary>
    /// <remarks>
    /// <see cref="GlobalPoolingLayer{T}"/> preserves the channel axis and emits
    /// [batch, channels, 1, 1]. <see cref="DenseLayer{T}"/> deliberately follows linear-layer
    /// semantics and transforms only the final axis, so forwarding that rank-4 tensor directly
    /// would emit [batch, channels, 1, scores]. The reshape belongs at this branch boundary and uses
    /// <c>Engine.Reshape</c> so gradients from the IoU and object-presence losses remain on the tape.
    /// </remarks>
    private Tensor<T> RunScoreHead(Tensor<T> input, int start, int end)
    {
        if (end - start != 2)
        {
            throw new InvalidOperationException(
                "A SAM2 score head must contain global pooling followed by a Dense layer.");
        }

        var pooled = Layers[start].Forward(input);
        int batch = pooled.Shape[0];
        if (batch <= 0 || pooled.Length % batch != 0)
        {
            throw new InvalidOperationException("SAM2 score-head pooling emitted an invalid batch layout.");
        }

        var flattened = Engine.Reshape(pooled, new[] { batch, pooled.Length / batch });
        return Layers[start + 1].Forward(flattened);
    }

    #region Module spans

    /// <summary>
    /// Number of ambiguity-resolving mask candidates the decoder emits. SAM predicts 3 masks
    /// (whole / part / subpart) plus one, i.e. 4 output tokens (Kirillov et al. 2023, section 3), and
    /// SAM 2 keeps that head unchanged (Ravi et al. 2024, section 3).
    /// </summary>
    private const int MaskCandidateCount = 4;

    // Half-open [start, end) spans into Layers, one per module. SAM2's forward is a BRANCHING
    // topology, so each branch has to address its own layers -- but addressing them by literal
    // index (the previous Layers[14] / Layers[21] / Layers[24] scheme) silently desynchronised from
    // the factories: the fixed indices assumed 14 encoder / 3 prompt / 5 memory / 2 refine layers,
    // while the factories emit 13 / 6 / 5 / 2. Consequences measured on this PR: EncodeImage ran the
    // first PROMPT layer as if it were an encoder layer; DecodeMask ran the last two MEMORY
    // convolutions as its "decoder"; and the masks came out of the first REFINEMENT conv -- a
    // 256-channel ReLU -- which was then squashed by an extra sigmoid. A dead ReLU there produces
    // exactly 0.0, so every mask pixel was exactly sigmoid(0) = 0.5 with a ReLU-blocked gradient,
    // which is precisely why the memorization loss sat frozen at
    // focal(0.0433) * 20 + dice(0.5) = 1.366417 for all 15 steps. The real mask / IoU / occlusion
    // heads were never added to Layers at all, so they were neither run nor trained. Deriving the
    // spans from the actual layer counts makes that class of drift impossible.
    private int _imageEncoderStart, _imageEncoderEnd;
    private int _pointEncoderStart, _boxEncoderStart, _maskPromptEncoderStart;
    private int _memoryAttentionStart, _memoryProjectionIndex;
    private int _maskRefineStart, _maskRefineEnd;
    private int _maskHeadIndex;
    private int _iouHeadStart, _iouHeadEnd;
    private int _occlusionHeadStart, _occlusionHeadEnd;
    private int _spansForLayerCount = -1;

    /// <summary>
    /// Whether Layers currently holds SAM2's own seven-module layout, so the branch forwards may
    /// address it. Recomputed whenever the layer count changes, because Clone / Deserialize /
    /// SetLayers repopulate Layers WITHOUT going back through InitializeLayers -- a one-shot flag set
    /// only in InitializeLayers would leave a cloned model permanently on the constant-mask fallback.
    /// </summary>
    private bool ModulesWired
    {
        get
        {
            if (_spansForLayerCount != Layers.Count)
            {
                RecordModuleSpans();
            }

            return _spansForLayerCount == Layers.Count && _modulesWired;
        }
    }

    private bool _modulesWired;

    /// <summary>
    /// Derives each module's span in <see cref="NeuralNetworkBase{T}.Layers"/> from the layer counts
    /// the LayerHelper factories emit, so no branch forward ever addresses a literal index.
    /// </summary>
    private void RecordModuleSpans()
    {
        // Counts emitted by the factories used in InitializeLayers, in append order.
        const int imageEncoderCount = 13;
        const int promptEncoderCount = 6;   // point x2, box x2, mask-prompt x2
        const int memoryCount = 5;          // 4 attention convs + 1 projection
        const int maskRefineCount = 2;
        const int maskHeadCount = 1;
        const int iouHeadCount = 2;         // global-average pool + dense(sigmoid)
        const int occlusionHeadCount = 2;   // global-average pool + dense(sigmoid)

        int expected = imageEncoderCount + promptEncoderCount + memoryCount + maskRefineCount
            + maskHeadCount + iouHeadCount + occlusionHeadCount;

        // A caller-supplied Architecture.Layers list has an unknown layout; the branch forwards then
        // fall back to their prompt-free / head-free paths rather than mis-indexing someone else's
        // network.
        _spansForLayerCount = Layers.Count;
        _modulesWired = _useNativeMode && Layers.Count == expected;
        if (!_modulesWired)
        {
            return;
        }

        _imageEncoderStart = 0;
        _imageEncoderEnd = imageEncoderCount;

        _pointEncoderStart = _imageEncoderEnd;
        _boxEncoderStart = _pointEncoderStart + 2;
        _maskPromptEncoderStart = _boxEncoderStart + 2;

        _memoryAttentionStart = _imageEncoderEnd + promptEncoderCount;
        _memoryProjectionIndex = _memoryAttentionStart + memoryCount - 1;

        _maskRefineStart = _memoryAttentionStart + memoryCount;
        _maskRefineEnd = _maskRefineStart + maskRefineCount;

        _maskHeadIndex = _maskRefineEnd;

        _iouHeadStart = _maskHeadIndex + maskHeadCount;
        _iouHeadEnd = _iouHeadStart + iouHeadCount;

        _occlusionHeadStart = _iouHeadEnd;
        _occlusionHeadEnd = _occlusionHeadStart + occlusionHeadCount;
    }

    #endregion


    private Tensor<T> EncodeImage(Tensor<T> image)
    {
        if (!_useNativeMode)
        {
            return EncodeImageOnnx(image);
        }
        if (NativePipeline is { } pipeline)
            return pipeline.EncodeImage(image);


        var features = image;

        int encoderEnd = ModulesWired ? _imageEncoderEnd : Layers.Count;
        for (int i = _imageEncoderStart; i < encoderEnd; i++)
        {
            features = Layers[i].Forward(features);
        }

        return features;
    }

    /// <summary>
    /// Encodes an image using the ONNX model.
    /// </summary>
    private Tensor<T> EncodeImageOnnx(Tensor<T> image)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        // Convert input tensor to float array for ONNX
        var inputData = new float[image.Length];
        for (int i = 0; i < image.Length; i++)
        {
            inputData[i] = Convert.ToSingle(image.Data.Span[i]);
        }

        // Create ONNX input tensor
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, image._shape);
        var inputMeta = _onnxSession.InputMetadata;

        // SAM2 encoder typically has 'image' as input name
        string inputName = inputMeta.Keys.FirstOrDefault() ?? "image";

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        // Run inference
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Convert output to our tensor format
        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    private Tensor<T> EncodePoints(float[,] points, int[] pointLabels)
    {
        if (NativePipeline is { } pipeline)
            return pipeline.EncodePoints(points, pointLabels);

        int numPoints = points.GetLength(0);
        int batchSize = 1;

        // Create point embedding with positional encoding
        var pointTensor = new Tensor<T>([batchSize, 2, 1, 1]);

        // Aggregate points with their labels
        for (int i = 0; i < numPoints; i++)
        {
            T x = NumOps.FromDouble(points[i, 0] / _width);
            T y = NumOps.FromDouble(points[i, 1] / _height);
            T weight = NumOps.FromDouble(pointLabels[i] == 1 ? 1.0 : -1.0);

            pointTensor[0, 0, 0, 0] = NumOps.Add(pointTensor[0, 0, 0, 0], NumOps.Multiply(x, weight));
            pointTensor[0, 1, 0, 0] = NumOps.Add(pointTensor[0, 1, 0, 0], NumOps.Multiply(y, weight));
        }

        if (ModulesWired)
        {
            return ApplyGELU(RunSpan(pointTensor, _pointEncoderStart, _pointEncoderStart + 2));
        }

        return ApplyGELU(pointTensor);
    }

    private Tensor<T> EncodeBox(float[] box)
    {
        if (NativePipeline is { } pipeline)
            return pipeline.EncodeBox(box);

        int batchSize = 1;
        var boxTensor = new Tensor<T>([batchSize, 4, 1, 1]);

        // Normalize coordinates
        boxTensor[0, 0, 0, 0] = NumOps.FromDouble(box[0] / _width);
        boxTensor[0, 1, 0, 0] = NumOps.FromDouble(box[1] / _height);
        boxTensor[0, 2, 0, 0] = NumOps.FromDouble(box[2] / _width);
        boxTensor[0, 3, 0, 0] = NumOps.FromDouble(box[3] / _height);

        if (ModulesWired)
        {
            return ApplyGELU(RunSpan(boxTensor, _boxEncoderStart, _boxEncoderStart + 2));
        }

        return ApplyGELU(boxTensor);
    }

    private Tensor<T> EncodeMaskPrompt(Tensor<T> maskPrompt)
    {
        if (NativePipeline is { } pipeline)
            return pipeline.EncodeMaskPrompt(maskPrompt);

        if (ModulesWired)
        {
            return ApplyGELU(RunSpan(maskPrompt, _maskPromptEncoderStart, _maskPromptEncoderStart + 2));
        }

        return ApplyGELU(maskPrompt);
    }

    private Tensor<T> PrepareMemoryConditioning(Tensor<T> currentFeatures)
    {
        if (NativePipeline is { } pipeline)
        {
            return _memoryBank.Count == 0
                ? pipeline.AddNoMemoryEmbedding(currentFeatures)
                : pipeline.ApplyMemoryAttention(currentFeatures, _memoryBank, _memoryObjectPointers);
        }

        return _memoryBank.Count == 0 ? currentFeatures : ApplyMemoryAttention(currentFeatures);
    }


    private Tensor<T> ApplyMemoryAttention(Tensor<T> currentFeatures)
    {
        if (NativePipeline is { } pipeline)
            return pipeline.ApplyMemoryAttention(currentFeatures, _memoryBank, _memoryObjectPointers);

        if (_memoryBank.Count == 0)
        {
            return currentFeatures;
        }

        // Average pool memory features
        var memoryAggregate = new Tensor<T>(currentFeatures._shape);
        foreach (var memory in _memoryBank)
        {
            memoryAggregate = AddTensors(memoryAggregate, memory);
        }
        memoryAggregate = memoryAggregate.Transform((v, _) =>
            NumOps.Divide(v, NumOps.FromDouble(_memoryBank.Count)));

        // Concatenate current and memory features
        var combined = ConcatenateChannels(currentFeatures, memoryAggregate);

        if (ModulesWired)
        {
            var attended = combined;
            for (int i = _memoryAttentionStart; i < _memoryProjectionIndex; i++)
            {
                attended = Layers[i].Forward(attended);
                attended = ApplyGELU(attended);
            }

            attended = Layers[_memoryProjectionIndex].Forward(attended);
            return AddTensors(currentFeatures, attended);
        }

        return currentFeatures;
    }

    private (Tensor<T> Masks, Tensor<T> IouScores, double OcclusionScore) DecodeMask(
        Tensor<T> imageFeatures,
        Tensor<T>? pointFeatures,
        Tensor<T>? boxFeatures,
        Tensor<T>? maskFeatures)
    {
        var features = imageFeatures;
        if (NativePipeline is { } pipeline)
        {
            Tensor<T>? sparsePrompt = pointFeatures;
            if (boxFeatures is not null)
                sparsePrompt = sparsePrompt is null
                    ? boxFeatures
                    : Engine.TensorConcatenate([sparsePrompt, boxFeatures], axis: 1);
            var decoded = pipeline.Decode(imageFeatures, sparsePrompt, maskFeatures);
            double objectPresence = Convert.ToDouble(decoded.ObjectPresenceScores[0, 0]);
            var nativeIouScores = Engine.Reshape(
                decoded.IouScores, [decoded.IouScores.Shape[0], MaskCandidateCount, 1, 1]);
            return (decoded.Masks, nativeIouScores, 1.0 - objectPresence);
        }


        // Add prompt features if provided
        if (pointFeatures != null)
        {
            features = AddPromptFeatures(features, pointFeatures);
        }
        if (boxFeatures != null)
        {
            features = AddPromptFeatures(features, boxFeatures);
        }
        if (maskFeatures != null)
        {
            features = AddTensors(features, maskFeatures);
        }

        // Shared refinement, then the three heads branch off it (Ravi et al. 2024, section 3: SAM 2
        // keeps SAM's mask decoder and adds the occlusion / "is object present" head).
        Tensor<T> masks;
        Tensor<T> iouScores;
        double occlusionScore = 0.0;

        if (ModulesWired)
        {
            // The helper-created refinement convolutions already contain the ReLU activations used
            // by SAM2's retained SAM mask decoder. Running them directly also keeps their outputs on
            // the autodiff tape; the former extra raw GELU transform both changed the reference
            // activation stack and disconnected the refinement/encoder parameters from mask loss.
            features = RunSpan(features, _maskRefineStart, _maskRefineEnd);

            // The mask head is a 1x1 convolution whose activation IS a sigmoid, so it already emits
            // probabilities. Applying ApplySigmoid on top of it would squash [0,1] into
            // [0.5, 0.731] and flatten the gradient, so it deliberately is NOT applied here.
            masks = Layers[_maskHeadIndex].Forward(features);

            // The IoU and occlusion heads each begin with their own global-average pool and end in a
            // sigmoid dense layer, so they take the FULL feature map and need no extra activation.
            // Global pooling preserves the NCHW channel axis; RunScoreHead graph-safely flattens its
            // [batch, channels, 1, 1] output to [batch, channels] before the Dense layer.
            // The head ends in a dense layer, so it emits [batch, candidates]; SelectBestMask and
            // GetNamedLayerActivations read IoU as [batch, candidates, 1, 1]. Normalise the rank with
            // Engine.Reshape rather than a raw tensor.Reshape so the tape survives.
            iouScores = NormalizeScoreRank(
                RunScoreHead(features, _iouHeadStart, _iouHeadEnd), MaskCandidateCount);

            var occlusion = NormalizeScoreRank(
                RunScoreHead(features, _occlusionHeadStart, _occlusionHeadEnd), 1);
            occlusionScore = Convert.ToDouble(occlusion[0, 0, 0, 0]);
        }
        else
        {
            int batchSize = features.Shape[0];
            int h = features.Shape[2];
            int w = features.Shape[3];
            masks = ApplySigmoid(new Tensor<T>([batchSize, MaskCandidateCount, h, w]));

            var pooled = GlobalAveragePool(features);
            iouScores = new Tensor<T>([pooled.Shape[0], MaskCandidateCount, 1, 1]);
            for (int b = 0; b < iouScores.Shape[0]; b++)
            {
                for (int m = 0; m < MaskCandidateCount; m++)
                {
                    iouScores[b, m, 0, 0] = NumOps.FromDouble(0.5);
                }
            }
        }

        return (masks, iouScores, occlusionScore);
    }

    private Tensor<T> AddPromptFeatures(Tensor<T> imageFeatures, Tensor<T> promptFeatures)
    {
        int batchSize = imageFeatures.Shape[0];
        int channels = imageFeatures.Shape[1];
        int height = imageFeatures.Shape[2];
        int width = imageFeatures.Shape[3];

        // Broadcast prompt features spatially
        var broadcastedPrompt = new Tensor<T>([batchSize, channels, height, width]);

        int promptChannels = Math.Min(channels, promptFeatures.Shape[1]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < promptChannels; c++)
            {
                T promptVal = promptFeatures[b, c, 0, 0];
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        broadcastedPrompt[b, c, h, w] = promptVal;
                    }
                }
            }
        }

        return AddTensors(imageFeatures, broadcastedPrompt);
    }

    /// <summary>
    /// Selects the single-mask output for the native decoder, or the highest-IoU
    /// candidate for legacy custom layer architectures.
    /// </summary>
    /// <remarks>
    /// Meta's decoder reserves token 0 for single-mask output and only IoU-ranks
    /// tokens 1-3 when the caller explicitly requests multimask output. This API has
    /// no multimask flag, so the native path must return token 0. The legacy branch
    /// keeps its prior IoU selection, using tape-tracked slices for gradient flow.
    /// </remarks>
    private Tensor<T> SelectBestMask(Tensor<T> masks, Tensor<T> iouScores)
    {
        if (NativePipeline is not null)
            return Engine.TensorGather(
                masks, new Tensor<int>([0], [1]), axis: 1);

        int batchSize = masks.Shape[0];
        int numMasks = masks.Shape[1];

        var perBatch = new Tensor<T>[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            // Index selection only -- read off-tape, never differentiated.
            int bestIdx = 0;
            double bestScore = double.MinValue;
            for (int m = 0; m < numMasks; m++)
            {
                double score = Convert.ToDouble(iouScores[b, m, 0, 0]);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestIdx = m;
                }
            }

            // Both narrows are recorded ops, so the chosen slice keeps its gradient path.
            var batchSlice = Engine.TensorNarrow(masks, dim: 0, start: b, length: 1);
            perBatch[b] = Engine.TensorNarrow(batchSlice, dim: 1, start: bestIdx, length: 1);
        }

        return batchSize == 1 ? perBatch[0] : Engine.TensorConcatenate(perBatch, axis: 0);
    }

    /// <summary>
    /// Bilinearly resizes a mask to the target spatial size.
    /// </summary>
    /// <remarks>
    /// <c>Engine.Interpolate</c> with <see cref="InterpolateMode.Bilinear"/> is the tape-tracked
    /// equivalent of the hand-rolled bilinear loop this replaces. That loop read every corner via
    /// <c>Convert.ToDouble(mask[...])</c> and wrote scalars back through the raw indexer, so it
    /// severed the autodiff tape at the very last step of the forward pass -- no gradient could
    /// reach any layer, leaving parameters unchanged after training.
    /// </remarks>
    private Tensor<T> UpsampleMask(Tensor<T> mask, int targetH, int targetW)
    {
        if (mask.Shape[2] == targetH && mask.Shape[3] == targetW)
        {
            return mask;
        }

        return Engine.Interpolate(
            mask,
            new[] { targetH, targetW },
            InterpolateMode.Bilinear,
            alignCorners: false);
    }

    private void UpdateMemoryBank(Tensor<T> features, Tensor<T> mask, int frameIndex)
    {
        if (NativePipeline is { } pipeline)
        {
            var nativeMemoryFeatures = pipeline.EncodeMemory(features, mask).Clone();
            var objectPointer = pipeline.LastObjectPointer.Clone();
            if (_memoryBank.Count >= _memoryBankSize)
            {
                _memoryBank.RemoveAt(0);
                _memoryObjectPointers.RemoveAt(0);
                _memoryFrameIndices.RemoveAt(0);
            }
            _memoryBank.Add(nativeMemoryFeatures);
            _memoryObjectPointers.Add(objectPointer);
            _memoryFrameIndices.Add(frameIndex);
            return;
        }

        // Combine features with mask information
        var memoryFeatures = features; // In a real implementation, this would also incorporate the mask

        if (_memoryBank.Count >= _memoryBankSize)
        {
            // Remove oldest memory
            _memoryBank.RemoveAt(0);
            _memoryFrameIndices.RemoveAt(0);
        }

        _memoryBank.Add(memoryFeatures);
        _memoryFrameIndices.Add(frameIndex);
    }

    private Tensor<T> GlobalAveragePool(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var output = new Tensor<T>([batchSize, channels, 1, 1]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                T sum = NumOps.Zero;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        sum = NumOps.Add(sum, input[b, c, h, w]);
                    }
                }
                output[b, c, 0, 0] = NumOps.Divide(sum, NumOps.FromDouble(height * width));
            }
        }

        return output;
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorConcatenate([a, b], axis: 1);
    }

    private Tensor<T> ApplyGELU(Tensor<T> input)
    {
        return input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            double c = Math.Sqrt(2.0 / Math.PI);
            double gelu = 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
            return NumOps.FromDouble(gelu);
        });
    }

    private Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        return Engine.Sigmoid(input);
    }

    /// <summary>
    /// Prepends a singleton batch axis, turning <c>[C,H,W]</c> into <c>[1,C,H,W]</c>.
    /// </summary>
    /// <remarks>
    /// Adding a batch axis is a pure RESHAPE, so it goes through <c>Engine.Reshape</c> and stays on
    /// the autodiff tape. The previous implementation allocated a fresh tensor and did a raw
    /// <c>Data.Span.CopyTo</c>, which severed the tape: any unbatched input entering
    /// <see cref="ForwardForTraining"/> or <see cref="PredictCore"/> lost its gradient path at the
    /// very first step, so no layer could ever be trained on that route. Latent for the 4-D fixture
    /// (which is already batched) but a real defect for the rank-3 single-image path this model
    /// documents.
    /// </remarks>
    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        int c = tensor.Shape[0];
        int h = tensor.Shape[1];
        int w = tensor.Shape[2];

        return Engine.Reshape(tensor, new[] { 1, c, h, w });
    }

    /// <summary>
    /// Drops the leading singleton batch axis.
    /// </summary>
    /// <remarks>
    /// Same reasoning as <see cref="AddBatchDimension"/>: a reshape through <c>Engine</c> keeps the
    /// gradient path, where the previous raw <c>Data.Span.CopyTo</c> cut it — and this one sits on the
    /// OUTPUT of the forward pass, so it severed the tape for every unbatched call.
    /// </remarks>
    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++)
        {
            newShape[i] = tensor.Shape[i + 1];
        }

        return Engine.Reshape(tensor, newShape);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    #endregion

    #region Abstract Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            ClearLayers();
            return;
        }

        if (Architecture.Layers is { Count: > 0 })
        {
            Layers.AddRange(Architecture.Layers);
            RecordModuleSpans();
            return;
        }

        var hiera = ResolveHieraPreset(_modelSize, _options);
        Layers.Add(new SAM2NativePipelineLayer<T>(
            _channels,
            _height,
            _width,
            hiera.Embedding,
            hiera.Depths,
            hiera.Heads,
            hiera.Windows,
            hiera.Globals,
            _options.ModelDimension,
            _options.MemoryDimension,
            _options.DecoderHeadCount,
            _options.MemoryAttentionLayerCount,
            _options.MaskDecoderDepth,
            _options.MaskDecoderMlpDimension,
            _options.RopeTheta,
            _options.MemoryMaskScale,
            _options.MemoryMaskBias));
        RecordModuleSpans();
    }

    // UpdateParameters redistributed the vector across Layers, which the base already folds -- and
    // did it less safely: the `offset + count <= parameters.Length` guard silently left the
    // remaining layers untouched on a short vector instead of failing. Removed under AIDN082.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "SAM2" },
            { "Description", "Segment Anything Model 2 for Video Object Segmentation" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "ModelSize", _modelSize.ToString() },
            { "NumFeatures", _numFeatures },
            { "MemoryBankSize", _memoryBankSize },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count }
        };

        return new ModelMetadata<T>
        {
            AdditionalInfo = additionalInfo,
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion
}

/// <summary>
/// Model size variants for SAM2.
/// </summary>
public enum SAM2ModelSize
{
    /// <summary>
    /// Tiny model - fastest, lowest memory.
    /// </summary>
    Tiny,

    /// <summary>
    /// Small model - balanced speed/accuracy.
    /// </summary>
    Small,

    /// <summary>
    /// Base model - good accuracy.
    /// </summary>
    Base,

    /// <summary>
    /// Large model - highest accuracy.
    /// </summary>
    Large
}
