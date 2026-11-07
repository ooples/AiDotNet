using AiDotNet.Attributes;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision.Segmentation.Video;

/// <summary>
/// EfficientTAM (Efficient Track Anything): SAM 2's promptable video tracking on a lightweight plain ViT, for devices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Give EfficientTAM the masks of the objects to follow in the first frame, then feed it the next
/// frames one at a time: it returns every object's mask in each, remembering what the objects looked like.
/// </para>
/// <para>
/// EfficientTAM (Xiong et al. 2024; the official yformer/EfficientTAM code) replaces SAM 2's hierarchical Hiera encoder with
/// a plain ViT-Tiny or ViT-Small (16×16 patches, 14×14 windows with four global blocks, a single-scale neck, and so no
/// high-resolution skips in the mask decoder), and its "/2" models replace the memory cross-attention with an efficient
/// one that pools each memory frame's keys and values over 2×2 windows. Everything else is SAM 2's network and tracker
/// (<see cref="Foundation.SAM21{T}"/>, verified against the official code), and the streaming API tracks each object as
/// the official video predictor does with its prompt on the first frame. Training follows the paper (see
/// <see cref="EfficientTAMOptions"/>). Released checkpoints load with <see cref="LoadPretrainedWeights"/> (the official
/// state dict saved as safetensors).
/// </para>
/// <para>
/// <b>Reference:</b> Xiong et al., "Efficient Track Anything", arXiv 2024.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Tracking)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Efficient Track Anything", "https://arxiv.org/abs/2411.18933", Year = 2024, Authors = "Yunyang Xiong, Chong Zhou, Xiaoyu Xiang, Lemeng Wu, Chenchen Zhu, Zechun Liu, Saksham Suri, Balakrishnan Varadarajan, Ramya Akula, Forrest Iandola, Raghuraman Krishnamoorthi, Bilge Soran, Vikas Chandra")]
[PaperOptimizer(OptimizerKind.AdamW, Beta1 = 0.9, Beta2 = 0.999, WeightDecay = 0.1, ReferenceBatchSize = 256, WarmupSteps = 15000,
                Schedule = LearningRateSchedulerType.CosineAnnealing, Phase = TrainingPhase.FineTuning,
                Source = "Xiong et al. 2024, full training: AdamW (0.9, 0.999), batch 256, 6e-5 for the image encoder and 3e-4 for "
                        + "the other components with layer-wise decay 0.8, a cosine schedule after a 15k-iteration linear warmup, "
                        + "weight decay 0.1, no drop path, 300k steps. The model builds this optimizer itself (Sam2Optimizer): no "
                        + "single rate is declared because the rates differ per parameter.")]
public partial class EfficientTAM<T> : Common.VideoSegmentationBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly EfficientTAMOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _userOptimizer;
    private Sam2Runtime<T>? _runtime;

    // Streaming state: a tracker per object, the frames' size, and corrections for the next frame.
    [Scratch]
    private readonly Dictionary<int, Sam2VideoTracker<T>> _trackers = new();
    [Scratch]
    private readonly Dictionary<int, Tensor<T>> _corrections = new();
    [Scratch]
    private int _frameHeight, _frameWidth;

    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => SpatialStrideContract(inputRank, 1);

    private Sam2Runtime<T> Runtime => _runtime ?? throw new InvalidOperationException("The network is not initialized.");

    /// <summary>The network.</summary>
    internal Sam2Network<T> Network => Runtime.Network;

    internal EfficientTAMModelSize ModelSize => _options.ModelSize;

    /// <summary>Creates a trainable EfficientTAM.</summary>
    /// <param name="architecture">The frames' size (the encoder's square size is the larger side unless
    /// <see cref="EfficientTAMOptions.ImageSize"/> sets it; 1024 for the released models) and 3 channels.</param>
    /// <param name="optimizer">The optimizer (default: the paper's AdamW recipe).</param>
    /// <param name="lossFunction">Reported by the base; training minimizes the paper's losses.</param>
    /// <param name="options">Model options.</param>
    public EfficientTAM(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        EfficientTAMOptions? options = null)
        : base(architecture, optimizer, lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>(), (options ??= new EfficientTAMOptions()).NumClasses, maxTrackedObjects: 64)
    {
        _options = options;
        _options.Validate();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        if (_channels != 3) throw new ArgumentException("EfficientTAM segments RGB frames (3 channels).", nameof(architecture));
        _userOptimizer = optimizer;
        InitializeLayers();
    }

    /// <summary>The network configuration: the size's ViT with the option overrides, SAM 2's (2.0) video modules.</summary>
    internal static Sam2Configuration BuildConfiguration(EfficientTAMOptions o, int side)
    {
        bool small = o.ModelSize == EfficientTAMModelSize.Small;
        var preset = Sam2Configuration.Vit(small);
        int imageSize = o.ImageSize ?? side;
        if (imageSize <= 0 || imageSize % 16 != 0)
            throw new ArgumentException($"EfficientTAM's encoder size must be a positive multiple of 16 pixels; got {imageSize}. Set EfficientTAMOptions.ImageSize.");
        return preset with
        {
            ImageSize = imageSize,
            EmbedDims = new[] { o.VitEmbeddingDimension ?? preset.EmbedDims[0] },
            VitDepth = o.VitDepth ?? preset.VitDepth,
            VitHeads = o.VitHeadCount ?? preset.VitHeads,
            VitWindow = o.VitWindowSize ?? preset.VitWindow,
            VitGlobalBlocks = o.VitGlobalAttentionBlockIndexes ?? preset.VitGlobalBlocks,
            Dim = o.ModelDimension ?? 256,
            DecoderHeads = o.DecoderHeadCount ?? 8,
            DecoderMlp = o.MaskDecoderMlpDimension ?? 2048,
            DropPath = o.DropPath ?? 0.0,
            Memory = new Sam2MemoryConfiguration
            {
                MaskMemories = o.MemoryBankSize,
                MemoryDim = o.MemoryDimension ?? 64,
                AttentionLayers = o.MemoryAttentionLayerCount ?? 4,
                AttentionFeedForward = o.MemoryAttentionFeedForwardDimension ?? 2048,
                FuserHidden = o.MemoryFuserDimension ?? 1024,
                CrossAttention = o.MemoryCrossAttention,
            },
        };
    }

    protected override void InitializeLayers()
    {
        if (_options is null || _runtime is not null) return;
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("EfficientTAM builds the paper's network; custom layers are not supported.");
        _runtime = new Sam2Runtime<T>(Engine, _options, BuildConfiguration(_options, Math.Max(_height, _width)), sizeIndex: 0,
            new Sam2TrainingSettings(), _userOptimizer);
        Layers.AddRange(_runtime.Layers);
        _encoderLayerEnd = _runtime.Layers.Count;
    }

    /// <summary>Loads a checkpoint saved as safetensors: the official state dict (keys such as
    /// <c>image_encoder.trunk.blocks.0.attn.qkv.weight</c>) or its Hugging Face-style renaming.</summary>
    public void LoadPretrainedWeights(string path) => Network.LoadSafetensors(path);

    /// <summary>Loads a state dict in Hugging Face-style names.</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        if (_runtime is not null) _runtime.Network.Training = isTraining;
    }

    // ------------------------------------------------------------------ streaming tracking

    /// <summary>The probabilities <c>[H, W]</c> of an object's mask <c>[H, W]</c> (or <c>[1, H, W]</c>) at the frames' size,
    /// as a mask prompt.</summary>
    private Sam2Prompt<T> MaskPrompt(Tensor<T> mask) => Runtime.MaskPrompt(Engine.Reshape(mask, new[] { _frameHeight, _frameWidth }), _frameHeight, _frameWidth);

    /// <summary>Starts tracking the objects whose masks <c>[N, H, W]</c> are given on a frame <c>[3, H, W]</c> (values in 0..1):
    /// each is a mask prompt on frame 0 of its own tracker.</summary>
    protected override void InitializeTrackingInternal(Tensor<T> frame, Tensor<T> masks, int[] objectIds)
    {
        var batched = Sam2Runtime<T>.Frame(frame, nameof(frame));
        _frameHeight = batched.Shape[2];
        _frameWidth = batched.Shape[3];
        int count = objectIds.Length, plane = _frameHeight * _frameWidth;
        if (masks.Length != count * plane) throw new ArgumentException($"Give one {_frameHeight}x{_frameWidth} mask per object ({count}).", nameof(masks));
        _trackers.Clear();
        _corrections.Clear();
        using var _ = new NoGradScope<T>();
        var features = Runtime.EncodeVideoFrame(batched);
        var flat = Engine.Reshape(masks, new[] { count, plane });
        for (int i = 0; i < count; i++)
        {
            var tracker = Runtime.StreamingTracker();
            var mask = Engine.Reshape(Engine.TensorSlice(flat, new[] { i, 0 }, new[] { 1, plane }), new[] { _frameHeight, _frameWidth });
            tracker.TrackFrame(0, prompted: true, features, MaskPrompt(mask));
            _trackers[objectIds[i]] = tracker;
        }
    }

    /// <summary>Tracks every object into the next frame <c>[3, H, W]</c>; an object given a correction is prompted with that
    /// mask on this frame.</summary>
    protected override VideoSegmentationResult<T> PropagateToFrameInternal(Tensor<T> frame, int frameIndex)
    {
        var batched = Sam2Runtime<T>.Frame(frame, nameof(frame));
        if (batched.Shape[2] != _frameHeight || batched.Shape[3] != _frameWidth)
            throw new ArgumentException($"Every frame must be {_frameHeight}x{_frameWidth}.", nameof(frame));
        using var _ = new NoGradScope<T>();
        var features = Runtime.EncodeVideoFrame(batched);
        var ids = _trackers.Keys.ToArray();
        var masks = new Tensor<T>[ids.Length];
        var confidences = new double[ids.Length];
        var visible = new bool[ids.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            Sam2Prompt<T>? prompt = _corrections.TryGetValue(ids[i], out var correction) ? MaskPrompt(correction) : null;
            var output = _trackers[ids[i]].TrackFrame(frameIndex, prompt is not null, features, prompt);
            masks[i] = Runtime.VideoProbabilities(output, _frameHeight, _frameWidth);
            confidences[i] = 1.0 / (1.0 + Math.Exp(-output.ObjectScore));
            visible[i] = output.ObjectScore > 0;
        }
        _corrections.Clear();
        return new VideoSegmentationResult<T>
        {
            Masks = Engine.TensorStack(masks, 0), ObjectIds = ids, Confidences = confidences, FrameIndex = frameIndex, IsVisible = visible,
        };
    }

    /// <summary>Corrects an object with its mask <c>[H, W]</c> on the next frame, which becomes one of its prompted frames.</summary>
    public override void AddCorrection(int objectId, Tensor<T> correctionMask)
    {
        if (!_trackers.ContainsKey(objectId)) throw new ArgumentException($"Object {objectId} is not tracked.", nameof(objectId));
        _corrections[objectId] = correctionMask;
    }

    protected override void ResetTrackingInternal()
    {
        _trackers.Clear();
        _corrections.Clear();
    }

    // ------------------------------------------------------------------ the tensor contract and training

    /// <summary>The single-mask logits <c>[B, 1, H, W]</c> (or <c>[1, H, W]</c>) of frames with no prompt: the image predictor's
    /// decoder on the model's empty prompt, resized to each frame.</summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        int b = input.Shape[0], h = input.Shape[2], w = input.Shape[3];
        var outputs = new Tensor<T>[b];
        for (int i = 0; i < b; i++)
        {
            var frame = Engine.TensorSlice(input, new[] { i, 0, 0, 0 }, new[] { 1, input.Shape[1], h, w });
            var (low, _, _) = Runtime.DecodeImage(Runtime.EncodeImage(frame), null, null, null, multimask: false);
            outputs[i] = SpatialResampling.Resize(Engine, low, h, w);
        }
        var logits = b == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 0);
        return hasBatch ? logits : RemoveBatchDimension(logits);
    }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    private IReadOnlyList<(Tensor<T> Input, Tensor<T> Target)> Images(Tensor<T> input, Tensor<T> target)
    {
        if (input.Rank == 3) return new[] { (input, target) };
        if (input.Rank != 4) throw new ArgumentException("EfficientTAM trains on frames [C, H, W] or [B, C, H, W].", nameof(input));
        int b = input.Shape[0], c = input.Shape[1], h = input.Shape[2], w = input.Shape[3];
        if (target.Length != b * h * w) throw new ArgumentException($"The target is one {h}x{w} mask per frame ({b}).", nameof(target));
        var flat = Engine.Reshape(target, new[] { b, h * w });
        return Enumerable.Range(0, b).Select(i => (
            Engine.Reshape(Engine.TensorSlice(input, new[] { i, 0, 0, 0 }, new[] { 1, c, h, w }), new[] { c, h, w }),
            Engine.Reshape(Engine.TensorSlice(flat, new[] { i, 0 }, new[] { 1, h * w }), new[] { h, w }))).ToList();
    }

    /// <summary>One training step on images <c>[C, H, W]</c> or <c>[B, C, H, W]</c> and their masks (each a one-frame
    /// example); videos train through <see cref="TrainOnVideo"/>.</summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        Step(Images(input, expectedOutput));
        return true;
    }

    /// <summary>One step of the video training on one object: frames <c>[F, 3, H, W]</c> and their masks <c>[F, H, W]</c>.</summary>
    public void TrainOnVideo(Tensor<T> video, Tensor<T> masks)
    {
        if (video.Rank != 4) throw new ArgumentException("A video is [F, 3, H, W].", nameof(video));
        Step(new[] { (video, masks) });
    }

    private void Step(IReadOnlyList<(Tensor<T> Input, Tensor<T> Target)> examples)
    {
        SetTrainingMode(true);
        try
        {
            LastLoss = Runtime.Train(examples, gradients => PublishParameterGradients(gradients));
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>EfficientTAM trains on a supervised objective: its losses against an object's masks.</summary>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    /// <inheritdoc/>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <summary>The training objective estimated with common random numbers (fixed draws).</summary>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) => Runtime.Evaluate(Images(input, target));

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "EfficientTAM" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "ImageSize", Network.Configuration.ImageSize },
            { "ModelSize", _options.ModelSize.ToString() },
            { "MemoryCrossAttention", _options.MemoryCrossAttention.ToString() },
            { "NumLayers", Layers.Count },
        },
        ModelDataProvider = () => SerializeForMetadata()
    };
}
