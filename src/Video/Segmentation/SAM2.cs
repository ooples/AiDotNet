using AiDotNet.Attributes;
using AiDotNet.ComputerVision.Segmentation;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Segmentation;

/// <summary>
/// SAM 2 (Segment Anything Model 2) for promptable segmentation of images and videos.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM 2 cuts an object out of an image, or follows it through a video, from a few clicks, a box
/// or a mask. Give it clicks on the first frame and it segments the object in every frame, remembering what the object
/// looked like in earlier frames so it can find it again after it moves or is hidden for a while.
/// </para>
/// <para>
/// This is the network the paper describes and Meta released (facebookresearch/sam2): the Hiera image encoder with an FPN
/// neck, SAM's prompt encoder and a mask decoder with an object-score head, a memory encoder that stores each frame's
/// prediction, memory attention that conditions every new frame on the stored memories and the object pointers, and the
/// video predictor's tracking (prompted frames decoded on their own, the object propagated forward and then backward).
/// It matches Hugging Face's port of the official code at 1e-12, and loads the released checkpoints in Hugging Face's
/// format with <see cref="LoadPretrainedWeights"/>. Training follows the paper's recipe (Table 12b) and the official
/// <c>SAM2Train</c>: simulated mask, click and box prompts, corrective clicks, and the focal, dice, IoU and occlusion
/// losses (see <see cref="SAM2Options"/>).
/// </para>
/// <para>
/// <b>Reference:</b> Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var sam2 = new SAM2&lt;float&gt;(architecture, options: new SAM2Options { ModelSize = SAM2ModelSize.Tiny });
/// sam2.LoadPretrainedWeights("sam2-hiera-tiny/model.safetensors");
/// var masks = sam2.TrackObject(frames, new float[,] { { 410, 220 } }, new[] { 1 });
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
public partial class SAM2<T> : NeuralNetworkBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly SAM2Options _options;
    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _userOptimizer;
    private Sam2Network<T>? _network;
    private List<LayerBase<T>>? _networkLayers;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _trainingOptimizer;

    /// <summary>Draws the training prompts, corrective clicks, drop-path and dropout masks.</summary>
    [Scratch]
    private Random _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>SAM 2's masks are probabilities.</summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => UnitIntervalTensorDomain.Value;

    /// <summary>The decoder's mask selection and the memory bank are data-dependent, so training stays on the eager tape.</summary>
    protected override bool SupportsFusedCompiledTraining => false;

    /// <summary>The network.</summary>
    internal Sam2Network<T> Network => _network ?? throw new InvalidOperationException("The network is not initialized.");

    /// <summary>The square size frames are resized to.</summary>
    internal int ImageSize => Network.Configuration.ImageSize;

    /// <summary>Creates SAM 2 (Base+) for 1024-pixel frames.</summary>
    public SAM2()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputHeight: 1024, inputWidth: 1024, inputDepth: 3,
            outputSize: 1))
    {
    }

    /// <summary>Creates SAM 2.</summary>
    /// <param name="architecture">The frames' size (height, width, 3 channels); frames are resized to the encoder's square
    /// size, and masks come back at the frames' size.</param>
    /// <param name="optimizer">The optimizer (default: the paper's AdamW recipe).</param>
    /// <param name="lossFunction">The loss the base reports; training minimizes the paper's losses.</param>
    /// <param name="options">Model options.</param>
    public SAM2(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SAM2Options? options = null)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>())
    {
        _options = options ?? new SAM2Options();
        _options.Validate();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        if (_channels != 3) throw new ArgumentException("SAM 2 segments RGB frames (3 channels).", nameof(architecture));
        _userOptimizer = optimizer;
        InitializeLayers();
    }

    private int SizeIndex => _options.ModelSize switch
    {
        SAM2ModelSize.Tiny => 0,
        SAM2ModelSize.Small => 1,
        SAM2ModelSize.Base => 2,
        _ => 3,
    };

    /// <summary>The selected size's configuration with the option overrides.</summary>
    internal Sam2Configuration BuildConfiguration()
    {
        var o = _options;
        var preset = Sam2Configuration.Hiera(SizeIndex);
        int side = o.ImageSize ?? Math.Max(_height, _width);
        if (side % 32 != 0)
            throw new ArgumentException($"SAM 2's encoder size must be a multiple of 32 pixels; got {side}. Set SAM2Options.ImageSize.");
        int stem = o.HieraEmbeddingDimension ?? preset.EmbedDims[0];
        int heads = o.HieraInitialHeadCount ?? preset.HeadsPerStage[0];
        var depths = o.HieraStageDepths ?? preset.BlocksPerStage;
        var globals = o.HieraGlobalAttentionBlockIndexes
            ?? (depths.SequenceEqual(preset.BlocksPerStage) ? preset.GlobalAttentionBlocks : new[] { depths[0] + depths[1] + depths[2] - 1 });
        var memory = (o.Sam21 ? Sam2MemoryConfiguration.Sam21 : new Sam2MemoryConfiguration()) with
        {
            MaskMemories = o.MemoryBankSize,
            MemoryDim = o.MemoryDimension ?? 64,
            AttentionLayers = o.MemoryAttentionLayerCount ?? 4,
            AttentionFeedForward = o.MemoryAttentionFeedForwardDimension ?? 2048,
            FuserHidden = o.MemoryFuserDimension ?? 1024,
        };
        return preset with
        {
            ImageSize = side,
            EmbedDims = new[] { stem, 2 * stem, 4 * stem, 8 * stem },
            HeadsPerStage = new[] { heads, 2 * heads, 4 * heads, 8 * heads },
            BlocksPerStage = depths,
            WindowSizes = o.HieraWindowSizes ?? preset.WindowSizes,
            GlobalAttentionBlocks = globals,
            Dim = o.ModelDimension ?? 256,
            DecoderHeads = o.DecoderHeadCount ?? 8,
            DecoderMlp = o.MaskDecoderMlpDimension ?? 2048,
            DropPath = o.DropPath ?? preset.DropPath,
            Memory = memory,
        };
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (_options is null || _network is not null) return;
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("SAM 2 builds the paper's network; custom layers are not supported.");
        var layers = new List<LayerBase<T>>();
        _network = new Sam2Network<T>(Engine, layers, BuildConfiguration());
        int seed = _options.Seed ?? 42;
        _network.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
        _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1);
        _network.TrainingRandom = _trainingRandom;
        _networkLayers = layers;
        Layers.AddRange(layers);
    }

    /// <summary>Loads a released checkpoint in Hugging Face's format (the <c>model.safetensors</c> of
    /// <c>facebook/sam2-hiera-*</c>, or <c>facebook/sam2.1-hiera-*</c> with <see cref="SAM2Options.Sam21"/>); the model must
    /// be built with that checkpoint's size.</summary>
    public void LoadPretrainedWeights(string path) => Network.LoadSafetensors(path);

    /// <summary>Loads a Hugging Face <c>Sam2VideoModel</c> state dict.</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        if (_network is not null) _network.Training = isTraining;
    }

    // ------------------------------------------------------------------ frames and prompts

    private static Tensor<T> Frame(Tensor<T> frame, string name)
    {
        if (frame.Rank == 4 && frame.Shape[0] == 1) return frame;
        if (frame.Rank != 3) throw new ArgumentException("A frame is [C, H, W] (or [1, C, H, W]).", name);
        return frame.Reshape(new[] { 1, frame.Shape[0], frame.Shape[1], frame.Shape[2] });
    }

    /// <summary>A frame <c>[1, 3, H, W]</c> in 0..1 resized to the encoder's square and normalized.</summary>
    private Tensor<T> Preprocess(Tensor<T> frame)
    {
        int size = ImageSize;
        var x = frame.Shape[2] == size && frame.Shape[3] == size ? frame : SpatialResampling.ResizeAntialias(Engine, frame, size, size);
        return Network.Normalize(x);
    }

    /// <summary>Points in a frame of the given size, in the encoder's frame (the official coordinate normalization).</summary>
    private Tensor<T> ScalePoints(float[,] points, int height, int width)
    {
        int k = points.GetLength(0), size = ImageSize;
        if (points.GetLength(1) != 2) throw new ArgumentException("Points are [k, 2] (x, y).", nameof(points));
        var scaled = new Tensor<T>(new[] { 1, k, 2 });
        for (int i = 0; i < k; i++)
        {
            scaled[0, i, 0] = NumOps.FromDouble(points[i, 0] * (double)size / width);
            scaled[0, i, 1] = NumOps.FromDouble(points[i, 1] * (double)size / height);
        }
        return scaled;
    }

    private Tensor<T> Labels(int[] labels)
    {
        var l = new Tensor<T>(new[] { 1, labels.Length });
        for (int i = 0; i < labels.Length; i++) l[0, i] = NumOps.FromDouble(labels[i]);
        return l;
    }

    private Sam2Prompt<T> PointPrompt(float[,] points, int[] labels, int height, int width)
    {
        if (labels.Length != points.GetLength(0)) throw new ArgumentException("Each point needs a label.", nameof(labels));
        if (labels.Any(l => l is not (0 or 1))) throw new ArgumentException("Point labels are 1 (object) or 0 (background).", nameof(labels));
        return new Sam2Prompt<T>(ScalePoints(points, height, width), Labels(labels), null);
    }

    /// <summary>A box as its two corner points (labels 2 and 3), as the official predictor gives it.</summary>
    private Sam2Prompt<T> BoxPrompt(float[] box, int height, int width)
    {
        if (box.Length != 4) throw new ArgumentException("A box is [x0, y0, x1, y1].", nameof(box));
        return new Sam2Prompt<T>(ScalePoints(new float[,] { { box[0], box[1] }, { box[2], box[3] } }, height, width), Labels(new[] { 2, 3 }), null);
    }

    /// <summary>A mask prompt <c>[H, W]</c> resized to the encoder's square (bilinear, antialiased) and thresholded at 0.5,
    /// as the official <c>add_new_mask</c> does.</summary>
    private Sam2Prompt<T> MaskPrompt(Tensor<T> mask, int height, int width)
    {
        if (mask.Length != height * width) throw new ArgumentException($"The mask prompt is one {height}x{width} map.", nameof(mask));
        int size = ImageSize;
        var m = Engine.Reshape(mask, new[] { 1, 1, height, width });
        if (height != size || width != size) m = SpatialResampling.ResizeAntialias(Engine, m, size, size);
        var binary = new Tensor<T>(new[] { 1, 1, size, size });
        var values = m.ToArray();
        for (int i = 0; i < values.Length; i++) binary[i] = NumOps.ToDouble(values[i]) >= 0.5 ? NumOps.One : NumOps.Zero;
        return new Sam2Prompt<T>(null, null, binary);
    }

    /// <summary>A frame's mask probabilities <c>[H, W]</c>: the logits resized to the frame and passed through the sigmoid.</summary>
    private Tensor<T> Probabilities(Tensor<T> logits, int height, int width)
    {
        var resized = logits.Shape[2] == height && logits.Shape[3] == width ? logits : SpatialResampling.Resize(Engine, logits, height, width);
        return Engine.Reshape(Engine.Sigmoid(resized), new[] { height, width });
    }

    /// <summary>Tracks one object through frames <c>[C, H, W]</c> (values in 0..1) from its prompts per frame.</summary>
    private List<Tensor<T>> Track(List<Tensor<T>> frames, Func<int, int, Dictionary<int, Sam2Prompt<T>>> prompts)
    {
        if (frames is null || frames.Count == 0) throw new ArgumentException("No frames.", nameof(frames));
        var batched = frames.Select(f => Frame(f, nameof(frames))).ToList();
        int height = batched[0].Shape[2], width = batched[0].Shape[3];
        if (batched.Any(f => f.Shape[2] != height || f.Shape[3] != width))
            throw new ArgumentException("Every frame must have the same size.", nameof(frames));
        using var _ = new NoGradScope<T>();
        var tracker = new Sam2VideoTracker<T>(Engine, Network);
        var outputs = tracker.Propagate(frames.Count, t => Network.EncodeFeatures(Preprocess(batched[t])), prompts(height, width));
        return outputs.Select(o => Probabilities(o.HighRes, height, width)).ToList();
    }

    /// <summary>
    /// Segments an object through a video from clicks on its first frame.
    /// </summary>
    /// <param name="frames">The frames <c>[3, H, W]</c>, values in 0..1.</param>
    /// <param name="points">Clicks <c>[k, 2]</c> (x, y in pixels of the first frame).</param>
    /// <param name="pointLabels">Each click's label: 1 on the object, 0 on the background.</param>
    /// <returns>Each frame's mask probabilities <c>[H, W]</c>.</returns>
    public List<Tensor<T>> TrackObject(List<Tensor<T>> frames, float[,] points, int[] pointLabels)
        => Track(frames, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [0] = PointPrompt(points, pointLabels, h, w) });

    /// <summary>Segments an object through a video from a box <c>[x0, y0, x1, y1]</c> on its first frame.</summary>
    public List<Tensor<T>> TrackObjectWithBox(List<Tensor<T>> frames, float[] box)
        => Track(frames, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [0] = BoxPrompt(box, h, w) });

    /// <summary>Segments an object through a video from its mask <c>[H, W]</c> on the first frame.</summary>
    public List<Tensor<T>> TrackObjectWithMask(List<Tensor<T>> frames, Tensor<T> mask)
        => Track(frames, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [0] = MaskPrompt(mask, h, w) });

    /// <summary>
    /// Segments an object through a video from clicks on any frames: each prompted frame is decoded from its clicks, and
    /// the object is propagated from the first prompted frame forward to the end and back to the start.
    /// </summary>
    public List<Tensor<T>> InteractiveVideoSegmentation(List<Tensor<T>> frames, Dictionary<int, (float[,] Points, int[] Labels)> framePrompts)
        => Track(frames, (h, w) => framePrompts.ToDictionary(kv => kv.Key, kv => PointPrompt(kv.Value.Points, kv.Value.Labels, h, w)));

    // ------------------------------------------------------------------ images

    private (Tensor<T> Logits, Tensor<T> Iou, Tensor<T> ObjectScore) SegmentImage(Tensor<T> image, Sam2Prompt<T> prompt, Tensor<T>? lowResLogits,
        out int height, out int width)
    {
        var frame = Frame(image, nameof(image));
        height = frame.Shape[2];
        width = frame.Shape[3];
        using var _ = new NoGradScope<T>();
        var net = Network;
        var (embedding, s0, s1) = net.EncodeImage(Preprocess(frame));
        var (sparse, dense) = net.EncodePrompts(prompt.Points, prompt.Labels, null, lowResLogits);
        bool multimask = (prompt.Labels?.Shape[1] ?? 0) <= 1;
        var (masks, iou, score) = net.DecodeMasks(embedding, s0, s1, net.DensePositional(), sparse, dense, multimask);
        int best = 0;
        for (int k = 1; k < iou.Shape[1]; k++)
            if (NumOps.ToDouble(iou[0, k]) > NumOps.ToDouble(iou[0, best])) best = k;
        int side = masks.Shape[2];
        return (Engine.TensorSlice(masks, new[] { 0, best, 0, 0 }, new[] { 1, 1, side, side }), iou, score);
    }

    /// <summary>
    /// Segments an object in one image from clicks (the image predictor): with one click the best of the three
    /// ambiguity-aware masks by predicted IoU, otherwise the single mask.
    /// </summary>
    /// <returns>The mask probabilities <c>[H, W]</c>.</returns>
    public Tensor<T> SegmentWithPoints(Tensor<T> image, float[,] points, int[] pointLabels)
    {
        var frame = Frame(image, nameof(image));
        var (logits, _, _) = SegmentImage(frame, PointPrompt(points, pointLabels, frame.Shape[2], frame.Shape[3]), null, out int h, out int w);
        return Probabilities(logits, h, w);
    }

    /// <summary>Segments an object in one image from a box <c>[x0, y0, x1, y1]</c>.</summary>
    public Tensor<T> SegmentWithBox(Tensor<T> image, float[] box)
    {
        var frame = Frame(image, nameof(image));
        var (logits, _, _) = SegmentImage(frame, BoxPrompt(box, frame.Shape[2], frame.Shape[3]), null, out int h, out int w);
        return Probabilities(logits, h, w);
    }

    /// <summary>Refines a mask in one image from a previous prediction's low-resolution logits <c>[S/4, S/4]</c> (S the
    /// encoder size), the image predictor's <c>mask_input</c>.</summary>
    public Tensor<T> SegmentWithMask(Tensor<T> image, Tensor<T> lowResLogits)
    {
        int side = ImageSize / 4;
        if (lowResLogits.Length != side * side) throw new ArgumentException($"The mask input is a {side}x{side} plane of logits.", nameof(lowResLogits));
        var (logits, _, _) = SegmentImage(image, new Sam2Prompt<T>(null, null, null), Engine.Reshape(lowResLogits, new[] { 1, 1, side, side }),
            out int h, out int w);
        return Probabilities(logits, h, w);
    }

    /// <summary>The probability that the clicked object is NOT visible in the image: one minus the sigmoid of the
    /// object-score head.</summary>
    public double GetOcclusionScore(Tensor<T> image, float[,] points, int[] pointLabels)
    {
        var frame = Frame(image, nameof(image));
        var (_, _, score) = SegmentImage(frame, PointPrompt(points, pointLabels, frame.Shape[2], frame.Shape[3]), null, out _, out _);
        return 1.0 - 1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(score[0, 0])));
    }

    // ------------------------------------------------------------------ the tensor contract and training

    /// <summary>
    /// Mask probabilities without prompts: a frame <c>[C, H, W]</c> gives <c>[H, W]</c>, a video <c>[F, C, H, W]</c> gives
    /// <c>[F, H, W]</c> (and <c>[B, F, …]</c> per video). The first frame is decoded from the model's empty prompt (a
    /// padding point) and the object is tracked from it.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input) => Forward(input);

    private Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank == 5)
        {
            var videos = new Tensor<T>[input.Shape[0]];
            for (int b = 0; b < videos.Length; b++)
                videos[b] = Forward(Engine.Reshape(Engine.TensorSlice(input, new[] { b, 0, 0, 0, 0 },
                    new[] { 1, input.Shape[1], input.Shape[2], input.Shape[3], input.Shape[4] }), new[] { input.Shape[1], input.Shape[2], input.Shape[3], input.Shape[4] }));
            return Engine.TensorStack(videos, 0);
        }
        bool single = input.Rank == 3;
        if (!single && input.Rank != 4) throw new ArgumentException("SAM 2 takes a frame [C, H, W] or a video [F, C, H, W].", nameof(input));
        int frames = single ? 1 : input.Shape[0], height = input.Shape[^2], width = input.Shape[^1];
        var tracker = new Sam2VideoTracker<T>(Engine, Network);
        Tensor<T> FrameAt(int t) => single ? Frame(input, nameof(input))
            : Engine.TensorSlice(input, new[] { t, 0, 0, 0 }, new[] { 1, input.Shape[1], height, width });
        var outputs = tracker.Propagate(frames, t => Network.EncodeFeatures(Preprocess(FrameAt(t))),
            new Dictionary<int, Sam2Prompt<T>> { [0] = new Sam2Prompt<T>(null, null, null) });
        var masks = outputs.Select(o => Probabilities(o.HighRes, height, width)).ToArray();
        return single ? masks[0] : Engine.TensorStack(masks, 0);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    /// <summary>The frames (resized and normalized) and their masks (resized to the encoder's square, above 0.5 the
    /// object) of a training example: a frame <c>[C, H, W]</c> with its mask <c>[H, W]</c>, or a video <c>[F, C, H, W]</c>
    /// with <c>[F, H, W]</c>.</summary>
    private (List<Tensor<T>> Images, List<double[]> Masks) Example(Tensor<T> input, Tensor<T> target)
    {
        bool single = input.Rank == 3;
        if (!single && input.Rank != 4) throw new ArgumentException("SAM 2 trains on a frame [C, H, W] or a video [F, C, H, W].", nameof(input));
        int frames = single ? 1 : input.Shape[0], height = input.Shape[^2], width = input.Shape[^1], size = ImageSize;
        if (target.Length != frames * height * width)
            throw new ArgumentException($"The target is one {height}x{width} mask per frame ({frames}).", nameof(target));
        var images = new List<Tensor<T>>();
        var masks = new List<double[]>();
        for (int t = 0; t < frames; t++)
        {
            var frame = single ? Frame(input, nameof(input)) : Engine.TensorSlice(input, new[] { t, 0, 0, 0 }, new[] { 1, input.Shape[1], height, width });
            images.Add(Preprocess(frame));
            var mask = Engine.Reshape(Engine.TensorSlice(Engine.Reshape(target, new[] { frames, height * width }), new[] { t, 0 }, new[] { 1, height * width }),
                new[] { 1, 1, height, width });
            if (height != size || width != size) mask = SpatialResampling.ResizeAntialias(Engine, mask, size, size);
            masks.Add(mask.ToArray().Select(v => NumOps.ToDouble(v) > 0.5 ? 1.0 : 0.0).ToArray());
        }
        return (images, masks);
    }

    private Sam2VideoTraining<T> Trainer() => new(Engine, Network, new Sam2TrainingSettings
    {
        FocalWeight = _options.MaskFocalWeight,
        DiceWeight = _options.MaskDiceWeight,
        IouWeight = _options.IouLossWeight,
        OcclusionWeight = _options.OcclusionLossWeight,
        FocalGamma = _options.FocalGamma,
        FocalAlpha = _options.FocalAlpha,
        CorrectionPoints = _options.CorrectionPoints,
    });

    /// <summary>The trained parameters (read from the network every step, as loading weights may replace them) and the
    /// optimizer, the paper's recipe unless one was given.</summary>
    private (List<Tensor<T>> Parameters, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer) TrainingState()
    {
        var recipe = new Sam2Recipe
        {
            LearningRate = _options.LearningRate,
            ImageEncoderLearningRate = _options.ImageEncoderLearningRate,
            LayerDecay = _options.LayerDecay ?? Sam2Recipe.PaperLayerDecay(SizeIndex),
            WeightDecay = _options.WeightDecay,
            MaxGradientNorm = _options.MaxGradientNorm,
            Steps = _options.TrainingSteps,
            WarmupSteps = _options.WarmupSteps,
        };
        if (_trainingOptimizer is null)
        {
            if (_userOptimizer is not null) _trainingOptimizer = _userOptimizer;
            else _trainingOptimizer = Sam2Optimizer.Create(Network, _networkLayers!, recipe).Optimizer;
        }
        var parameters = _trainingOptimizer is AdamWOptimizer<T, Tensor<T>, Tensor<T>> paper && _userOptimizer is null
            ? Sam2Optimizer.Refresh(paper, Network, _networkLayers!, recipe)
            : Training.TapeTrainingStep<T>.CollectParameters(_networkLayers!).ToList();
        return (parameters, _trainingOptimizer);
    }

    /// <summary>
    /// One step of SAM 2's training (Ravi et al. 2024, Appendix D.2.2) on a frame and its mask, or a video and its masks
    /// for one object: see <see cref="Sam2VideoTraining{T}"/>.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var (images, masks) = Example(input, expectedOutput);
        var trainer = Trainer();
        var plan = trainer.Draw(masks, _trainingRandom);
        var (parameters, optimizer) = TrainingState();
        SetTrainingMode(true);
        try
        {
            Network.TrainingRandom = _trainingRandom;
            LastLoss = Training.TapeTrainingStepper<T>.EagerObjectiveStep(parameters,
                () => trainer.Loss(images, masks, plan, _trainingRandom), optimizer,
                onGradients: gradients => PublishParameterGradients(gradients));
        }
        finally
        {
            SetTrainingMode(false);
        }
        return true;
    }

    /// <summary>SAM 2 trains on a supervised objective: its losses against an object's masks.</summary>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    /// <inheritdoc/>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <summary>The training objective estimated with common random numbers: its mean over four prompt plans, clicks, drop
    /// path and dropout masks drawn from one fixed seed, so two evaluations compare the same objective.</summary>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        const int draws = 4;
        var (images, masks) = Example(input, target);
        var trainer = Trainer();
        var random = new Random(_options.Seed ?? 42);
        bool wasTraining = Network.Training;
        var previous = Network.TrainingRandom;
        Network.Training = true;
        Network.TrainingRandom = random;
        try
        {
            using var _ = new NoGradScope<T>();
            double total = 0;
            for (int i = 0; i < draws; i++)
                total += NumOps.ToDouble(trainer.Loss(images, masks, trainer.Draw(masks, random), random)[0]);
            return NumOps.FromDouble(total / draws);
        }
        finally
        {
            Network.Training = wasTraining;
            Network.TrainingRandom = previous;
        }
    }

    /// <inheritdoc/>
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", _options.Sam21 ? "SAM 2.1" : "SAM 2" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "ImageSize", ImageSize },
            { "ModelSize", _options.ModelSize.ToString() },
            { "MemoryBankSize", _options.MemoryBankSize },
            { "NumLayers", Layers.Count },
        },
        ModelDataProvider = () => SerializeForMetadata()
    };
}

/// <summary>SAM 2's released sizes.</summary>
public enum SAM2ModelSize
{
    /// <summary>Hiera-T (39M parameters).</summary>
    Tiny,

    /// <summary>Hiera-S (46M parameters).</summary>
    Small,

    /// <summary>Hiera-B+ (81M parameters).</summary>
    Base,

    /// <summary>Hiera-L (224M parameters).</summary>
    Large
}
