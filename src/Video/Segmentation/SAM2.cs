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
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _userOptimizer;
    private Sam2Runtime<T>? _runtime;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>SAM 2's masks are probabilities.</summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => UnitIntervalTensorDomain.Value;

    /// <summary>The decoder's mask selection and the memory bank are data-dependent, so training stays on the eager tape.</summary>
    protected override bool SupportsFusedCompiledTraining => false;

    private Sam2Runtime<T> Runtime => _runtime ?? throw new InvalidOperationException("The network is not initialized.");

    /// <summary>The network.</summary>
    internal Sam2Network<T> Network => Runtime.Network;

    /// <summary>The square size frames are resized to.</summary>
    internal int ImageSize => Runtime.ImageSize;

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
        int channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        if (channels != 3) throw new ArgumentException("SAM 2 segments RGB frames (3 channels).", nameof(architecture));
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

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (_options is null || _runtime is not null) return;
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("SAM 2 builds the paper's network; custom layers are not supported.");
        _runtime = new Sam2Runtime<T>(Engine, _options, SizeIndex, _options.Sam21, Math.Max(_height, _width), new Sam2TrainingSettings(), _userOptimizer);
        Layers.AddRange(_runtime.Layers);
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
        if (_runtime is not null) _runtime.Network.Training = isTraining;
    }

    // ------------------------------------------------------------------ video

    /// <summary>Tracks one object through frames <c>[C, H, W]</c> (values in 0..1) from its prompts per frame.</summary>
    private List<Tensor<T>> Track(List<Tensor<T>> frames, Func<int, int, Dictionary<int, Sam2Prompt<T>>> prompts)
    {
        if (frames is null || frames.Count == 0) throw new ArgumentException("No frames.", nameof(frames));
        var batched = frames.Select(f => Sam2Runtime<T>.Frame(f, nameof(frames))).ToList();
        int height = batched[0].Shape[2], width = batched[0].Shape[3];
        if (batched.Any(f => f.Shape[2] != height || f.Shape[3] != width))
            throw new ArgumentException("Every frame must have the same size.", nameof(frames));
        var outputs = Runtime.Track(batched, prompts(height, width));
        return outputs.Select(o => Runtime.VideoProbabilities(o, height, width)).ToList();
    }

    /// <summary>
    /// Segments an object through a video from clicks on its first frame.
    /// </summary>
    /// <param name="frames">The frames <c>[3, H, W]</c>, values in 0..1.</param>
    /// <param name="points">Clicks <c>[k, 2]</c> (x, y in pixels of the first frame).</param>
    /// <param name="pointLabels">Each click's label: 1 on the object, 0 on the background.</param>
    /// <returns>Each frame's mask probabilities <c>[H, W]</c>.</returns>
    public List<Tensor<T>> TrackObject(List<Tensor<T>> frames, float[,] points, int[] pointLabels)
        => Track(frames, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [0] = Runtime.PointPrompt(points, pointLabels, h, w) });

    /// <summary>Segments an object through a video from a box <c>[x0, y0, x1, y1]</c> on its first frame.</summary>
    public List<Tensor<T>> TrackObjectWithBox(List<Tensor<T>> frames, float[] box)
        => Track(frames, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [0] = Runtime.BoxPrompt(box.Select(v => (double)v).ToList(), h, w) });

    /// <summary>Segments an object through a video from its mask <c>[H, W]</c> on the first frame.</summary>
    public List<Tensor<T>> TrackObjectWithMask(List<Tensor<T>> frames, Tensor<T> mask)
        => Track(frames, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [0] = Runtime.MaskPrompt(mask, h, w) });

    /// <summary>
    /// Segments an object through a video from clicks on any frames: each prompted frame is decoded from its clicks, and
    /// the object is propagated from the first prompted frame forward to the end and back to the start.
    /// </summary>
    public List<Tensor<T>> InteractiveVideoSegmentation(List<Tensor<T>> frames, Dictionary<int, (float[,] Points, int[] Labels)> framePrompts)
        => Track(frames, (h, w) => framePrompts.ToDictionary(kv => kv.Key, kv => Runtime.PointPrompt(kv.Value.Points, kv.Value.Labels, h, w)));

    // ------------------------------------------------------------------ images

    private (Tensor<T> Logits, Tensor<T> ObjectScore) SegmentImage(Tensor<T> image, Sam2Prompt<T> prompt, Tensor<T>? lowResLogits)
    {
        var frame = Sam2Runtime<T>.Frame(image, nameof(image));
        bool multimask = (prompt.Labels?.Shape[1] ?? 0) <= 1;
        var (masks, iou, score) = Runtime.DecodeImage(Runtime.EncodeImage(frame), prompt.Points, prompt.Labels, lowResLogits, multimask);
        int best = 0;
        for (int k = 1; k < iou.Shape[1]; k++)
            if (NumOps.ToDouble(iou[0, k]) > NumOps.ToDouble(iou[0, best])) best = k;
        int side = masks.Shape[2];
        return (Engine.TensorSlice(masks, new[] { 0, best, 0, 0 }, new[] { 1, 1, side, side }), score);
    }

    /// <summary>
    /// Segments an object in one image from clicks (the image predictor): with one click the best of the three
    /// ambiguity-aware masks by predicted IoU, otherwise the single mask.
    /// </summary>
    /// <returns>The mask probabilities <c>[H, W]</c>.</returns>
    public Tensor<T> SegmentWithPoints(Tensor<T> image, float[,] points, int[] pointLabels)
    {
        var frame = Sam2Runtime<T>.Frame(image, nameof(image));
        int h = frame.Shape[2], w = frame.Shape[3];
        return Runtime.Probabilities(SegmentImage(frame, Runtime.PointPrompt(points, pointLabels, h, w), null).Logits, h, w);
    }

    /// <summary>Segments an object in one image from a box <c>[x0, y0, x1, y1]</c>.</summary>
    public Tensor<T> SegmentWithBox(Tensor<T> image, float[] box)
    {
        var frame = Sam2Runtime<T>.Frame(image, nameof(image));
        int h = frame.Shape[2], w = frame.Shape[3];
        return Runtime.Probabilities(SegmentImage(frame, Runtime.BoxPrompt(box.Select(v => (double)v).ToList(), h, w), null).Logits, h, w);
    }

    /// <summary>Refines a mask in one image from a previous prediction's low-resolution logits <c>[S/4, S/4]</c> (S the
    /// encoder size), the image predictor's <c>mask_input</c>.</summary>
    public Tensor<T> SegmentWithMask(Tensor<T> image, Tensor<T> lowResLogits)
    {
        int side = ImageSize / 4;
        if (lowResLogits.Length != side * side) throw new ArgumentException($"The mask input is a {side}x{side} plane of logits.", nameof(lowResLogits));
        var frame = Sam2Runtime<T>.Frame(image, nameof(image));
        var (logits, _) = SegmentImage(frame, Sam2Runtime<T>.NoPrompt, Engine.Reshape(lowResLogits, new[] { 1, 1, side, side }));
        return Runtime.Probabilities(logits, frame.Shape[2], frame.Shape[3]);
    }

    /// <summary>The probability that the clicked object is NOT visible in the image: one minus the sigmoid of the
    /// object-score head.</summary>
    public double GetOcclusionScore(Tensor<T> image, float[,] points, int[] pointLabels)
    {
        var frame = Sam2Runtime<T>.Frame(image, nameof(image));
        var (_, score) = SegmentImage(frame, Runtime.PointPrompt(points, pointLabels, frame.Shape[2], frame.Shape[3]), null);
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
        var batched = Enumerable.Range(0, frames).Select(t => single ? Sam2Runtime<T>.Frame(input, nameof(input))
            : Engine.TensorSlice(input, new[] { t, 0, 0, 0 }, new[] { 1, input.Shape[1], height, width })).ToList();
        var outputs = Runtime.Track(batched, new Dictionary<int, Sam2Prompt<T>> { [0] = Sam2Runtime<T>.NoPrompt });
        var masks = outputs.Select(o => Runtime.VideoProbabilities(o, height, width)).ToArray();
        return single ? masks[0] : Engine.TensorStack(masks, 0);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    /// <summary>
    /// One step of SAM 2's training (Ravi et al. 2024, Appendix D.2.2) on a frame <c>[C, H, W]</c> and its mask
    /// <c>[H, W]</c>, or a video <c>[F, C, H, W]</c> and one object's masks <c>[F, H, W]</c>: see
    /// <see cref="Sam2VideoTraining{T}"/>.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            LastLoss = Runtime.Train(Videos(input, expectedOutput), gradients => PublishParameterGradients(gradients));
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

    /// <summary>The training objective estimated with common random numbers (fixed draws), so two evaluations compare the
    /// same objective.</summary>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) => Runtime.Evaluate(Videos(input, target));

    /// <summary>The training examples of an input: one frame or video, or a batch of videos <c>[B, F, C, H, W]</c> with their
    /// masks <c>[B, F, H, W]</c>.</summary>
    private IReadOnlyList<(Tensor<T> Input, Tensor<T> Target)> Videos(Tensor<T> input, Tensor<T> target)
    {
        if (input.Rank != 5) return new[] { (input, target) };
        int b = input.Shape[0], f = input.Shape[1], c = input.Shape[2], h = input.Shape[3], w = input.Shape[4];
        if (target.Length != b * f * h * w) throw new ArgumentException($"The target is one {h}x{w} mask per frame.", nameof(target));
        var flat = Engine.Reshape(target, new[] { b, f * h * w });
        return Enumerable.Range(0, b).Select(i => (
            Engine.Reshape(Engine.TensorSlice(input, new[] { i, 0, 0, 0, 0 }, new[] { 1, f, c, h, w }), new[] { f, c, h, w }),
            Engine.Reshape(Engine.TensorSlice(flat, new[] { i, 0 }, new[] { 1, f * h * w }), new[] { f, h, w }))).ToList();
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
