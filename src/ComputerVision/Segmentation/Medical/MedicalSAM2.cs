using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// Medical SAM 2: SAM 2 for medical images and volumes, segmenting sets of 2D images as pseudo-videos through a self-sorting
/// memory bank.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Prompt one image of a set (for example, click on the optic disc in one retina photo) and Medical SAM 2
/// segments the same structure in all the others, using the most varied, confident earlier results as its memory. For a 3D
/// scan, prompt slices and it tracks the structure through the volume.
/// </para>
/// <para>
/// Medical SAM 2 (Zhu et al. 2024) is SAM 2 (the official code's Hiera-S, at the paper's 256 pixels) on the shared network
/// verified against the official SAM 2 code. For a set of images it replaces SAM 2's recent-frame memory with a
/// self-sorting bank: a confident prediction's memory joins the bank's candidates, the K most mutually dissimilar are kept
/// (eqs. 2-3), and each image attends to K memories drawn by similarity (eqs. 4-5). A volume is tracked as videos along
/// each prompted axis, in both orders, and the orientations' probabilities averaged. Training is SAM 2's (see
/// <see cref="MedicalSAM2Options"/>), with the bank as the memory for image sets.
/// </para>
/// <para>
/// <b>Reference:</b> Zhu et al., "Medical SAM 2: Segment medical images as video via Segment Anything Model 2", arXiv 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 1);
/// var model = new MedicalSAM2&lt;float&gt;(architecture);
/// var masks = model.SegmentImageSet(images, templateIndex: 0, points: new float[,] { { 120, 96 } }, labels: new[] { 1 });
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Medical SAM 2: Segment medical images as video via Segment Anything Model 2", "https://arxiv.org/abs/2408.00874", Year = 2024,
    Authors = "Jiayuan Zhu, Abdullah Hamdi, Yunli Qi, Yueming Jin, Junde Wu")]
[PaperOptimizer(OptimizerKind.AdamW, Beta1 = 0.9, Beta2 = 0.999, Schedule = LearningRateSchedulerType.CosineAnnealing,
                Phase = TrainingPhase.FineTuning,
                Source = "Zhu et al. 2024, section 4.3: AdamW (0.9, 0.999) with a linear learning-rate warmup followed by cosine "
                        + "decay. No rate, warmup length or weight decay is stated; the model fills them from SAM 2's Table 12b "
                        + "and builds the optimizer itself (Sam2Optimizer), with per-parameter rates.")]
public partial class MedicalSAM2<T> : Common.MedicalSegmentationBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly MedicalSAM2Options _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _userOptimizer;
    private Sam2Runtime<T>? _runtime;

    private static readonly string[] Modalities = ["CT", "MRI", "Ultrasound", "Fundus", "Dermoscopy", "Microscopy", "X-ray"];

    public override ModelOptions GetOptions() => _options;

    private Sam2Runtime<T> Runtime => _runtime ?? throw new InvalidOperationException("The network is not initialized.");

    /// <summary>The network.</summary>
    internal Sam2Network<T> Network => Runtime.Network;

    /// <summary>Creates a trainable Medical SAM 2.</summary>
    /// <param name="architecture">The images' size and channels (1 for grayscale, 3 for color).</param>
    /// <param name="optimizer">The optimizer (default: the recipe's AdamW).</param>
    /// <param name="lossFunction">Reported by the base; training minimizes SAM 2's losses.</param>
    /// <param name="options">Model options.</param>
    public MedicalSAM2(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        MedicalSAM2Options? options = null)
        : base(architecture, optimizer, lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>(), (options ??= new MedicalSAM2Options()).NumClasses, Modalities)
    {
        _options = options;
        _options.Validate();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 256;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 256;
        if (_channels is not (1 or 3)) throw new ArgumentException("Medical SAM 2 segments grayscale (1-channel) or RGB (3-channel) images.", nameof(architecture));
        _userOptimizer = optimizer;
        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        if (_options is null || _runtime is not null) return;
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("Medical SAM 2 builds the paper's network; custom layers are not supported.");
        var bank = new Sam2BankSettings(_options.MemoryBankCapacity, _options.ConfidenceThreshold, _options.MemoryBankRule, _options.MemoryResampling);
        _runtime = new Sam2Runtime<T>(Engine, _options, _options.ModelSize == MedicalSAM2ModelSize.Tiny ? 0 : 1, sam21: false,
            _options.ImageSize ?? 256, new Sam2TrainingSettings(), _userOptimizer, bank);
        Layers.AddRange(_runtime.Layers);
        _encoderLayerEnd = _runtime.Layers.Count;
    }

    /// <summary>Loads a checkpoint in Hugging Face's <c>Sam2VideoModel</c> format (an official SAM 2 or Medical SAM 2 <c>.pt</c>
    /// converted with transformers' SAM 2 conversion script).</summary>
    public void LoadPretrainedWeights(string path) => Network.LoadSafetensors(path);

    /// <summary>Loads a Hugging Face <c>Sam2VideoModel</c> state dict.</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        if (_runtime is not null) _runtime.Network.Training = isTraining;
    }

    private int Seed => _options.Seed ?? 42;

    // ------------------------------------------------------------------ image sets

    /// <summary>An image <c>[C, H, W]</c> (C 1 or 3, values in 0..1) as an RGB frame <c>[1, 3, H, W]</c>.</summary>
    private Tensor<T> Rgb(Tensor<T> image)
    {
        var frame = Sam2Runtime<T>.Frame(image, nameof(image));
        if (frame.Shape[1] == 3) return frame;
        if (frame.Shape[1] != 1) throw new ArgumentException("Medical SAM 2 takes grayscale (1-channel) or RGB (3-channel) images.", nameof(image));
        return Engine.TensorConcatenate(new[] { frame, frame, frame }, 1);
    }

    private List<Tensor<T>> SegmentSet(List<Tensor<T>> images, Func<int, int, Dictionary<int, Sam2Prompt<T>>> prompts)
    {
        if (images is null || images.Count == 0) throw new ArgumentException("No images.", nameof(images));
        var frames = images.Select(Rgb).ToList();
        int h = frames[0].Shape[2], w = frames[0].Shape[3];
        if (frames.Any(f => f.Shape[2] != h || f.Shape[3] != w)) throw new ArgumentException("Every image must have the same size.", nameof(images));
        return Runtime.TrackImageSet(frames, prompts(h, w), Seed).Select(o => Runtime.Probabilities(o.HighRes, h, w)).ToList();
    }

    /// <summary>
    /// Segments a structure through a set of images <c>[C, H, W]</c> (values in 0..1) from clicks on one of them (the
    /// one-prompt setting): the images are processed in order as a pseudo-video, each conditioned on the self-sorting bank.
    /// </summary>
    /// <returns>Each image's probabilities <c>[H, W]</c>.</returns>
    public List<Tensor<T>> SegmentImageSet(List<Tensor<T>> images, int templateIndex, float[,] points, int[] labels)
        => SegmentSet(images, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [templateIndex] = Runtime.PointPrompt(points, labels, h, w) });

    /// <summary>Segments a structure through a set of images from a box <c>[x0, y0, x1, y1]</c> on one of them.</summary>
    public List<Tensor<T>> SegmentImageSet(List<Tensor<T>> images, int templateIndex, float[] box)
        => SegmentSet(images, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [templateIndex] = Runtime.BoxPrompt(box.Select(v => (double)v).ToList(), h, w) });

    /// <summary>Segments a structure through a set of images from its mask <c>[H, W]</c> on one of them.</summary>
    public List<Tensor<T>> SegmentImageSet(List<Tensor<T>> images, int templateIndex, Tensor<T> mask)
        => SegmentSet(images, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [templateIndex] = Runtime.MaskPrompt(mask, h, w) });

    // ------------------------------------------------------------------ volumes

    /// <summary>The slices of a volume <c>[C, D, H, W]</c> along an axis (0 axial, 1 coronal, 2 sagittal) as RGB frames.</summary>
    private List<Tensor<T>> Slices(Tensor<T> volume, int axis)
    {
        int c = volume.Shape[0], d = volume.Shape[1], h = volume.Shape[2], w = volume.Shape[3];
        var permuted = axis switch
        {
            0 => volume,
            1 => Engine.TensorPermute(volume, new[] { 0, 2, 1, 3 }),   // [C, H, D, W]
            _ => Engine.TensorPermute(volume, new[] { 0, 3, 1, 2 }),   // [C, W, D, H]
        };
        int count = permuted.Shape[1], rows = permuted.Shape[2], cols = permuted.Shape[3];
        return Enumerable.Range(0, count).Select(t => Rgb(Engine.Reshape(
            Engine.TensorSlice(permuted, new[] { 0, t, 0, 0 }, new[] { c, 1, rows, cols }), new[] { c, rows, cols }))).ToList();
    }

    /// <summary>A video's probabilities <c>[count, rows, cols]</c> back in the volume's layout <c>[D, H, W]</c>.</summary>
    private Tensor<T> Unslice(Tensor<T> probabilities, int axis)
        => axis switch
        {
            0 => probabilities,
            1 => Engine.TensorPermute(probabilities, new[] { 1, 0, 2 }),
            _ => Engine.TensorPermute(probabilities, new[] { 1, 2, 0 }),
        };

    /// <summary>The probabilities <c>[D, H, W]</c> of tracking along an axis in one order from mask prompts on its slices.</summary>
    private Tensor<T> TrackAxis(List<Tensor<T>> slices, IReadOnlyDictionary<int, Sam2Prompt<T>> prompts, bool reversed, int axis)
    {
        int count = slices.Count, rows = slices[0].Shape[2], cols = slices[0].Shape[3];
        var ordered = reversed ? Enumerable.Reverse(slices).ToList() : slices;
        var mapped = prompts.ToDictionary(kv => reversed ? count - 1 - kv.Key : kv.Key, kv => kv.Value);
        var outputs = Runtime.Track(ordered, mapped);
        var probabilities = outputs.Select(o => Runtime.Probabilities(o.HighRes, rows, cols)).ToList();
        if (reversed) probabilities.Reverse();
        return Unslice(Engine.TensorStack(probabilities.ToArray(), 0), axis);
    }

    /// <summary>
    /// Segments a structure through a volume <c>[C, D, H, W]</c> (values in 0..1) from its mask on some slices along each
    /// axis: axial slices <c>[H, W]</c> keyed by depth, and optionally coronal slices <c>[D, W]</c> keyed by row and sagittal
    /// slices <c>[D, H]</c> keyed by column. Each prompted axis is tracked as a video in both orders, and the orientations'
    /// probabilities averaged (the paper's six orientations when all three axes are prompted).
    /// </summary>
    public MedicalSegmentationResult<T> SegmentVolume(Tensor<T> volume, IReadOnlyDictionary<int, Tensor<T>> axialMasks,
        IReadOnlyDictionary<int, Tensor<T>>? coronalMasks = null, IReadOnlyDictionary<int, Tensor<T>>? sagittalMasks = null)
    {
        if (volume.Rank != 4) throw new ArgumentException("A volume is [C, D, H, W].", nameof(volume));
        var axes = new[] { axialMasks, coronalMasks, sagittalMasks };
        Tensor<T>? sum = null;
        int runs = 0;
        for (int axis = 0; axis < 3; axis++)
        {
            if (axes[axis] is not { Count: > 0 } masks) continue;
            var slices = Slices(volume, axis);
            int rows = slices[0].Shape[2], cols = slices[0].Shape[3];
            var prompts = masks.ToDictionary(kv => kv.Key, kv => Runtime.MaskPrompt(kv.Value, rows, cols));
            foreach (bool reversed in new[] { false, true })
            {
                var p = TrackAxis(slices, prompts, reversed, axis);
                sum = sum is null ? p : Engine.TensorAdd(sum, p);
                runs++;
            }
        }
        if (sum is null) throw new ArgumentException("Give at least one prompted slice.", nameof(axialMasks));
        return Result(Engine.TensorMultiplyScalar(sum, NumOps.FromDouble(1.0 / runs)));
    }

    /// <summary>A binary segmentation result from probabilities <c>[..., H, W]</c>: the labels (above 0.5) and one structure.</summary>
    private MedicalSegmentationResult<T> Result(Tensor<T> probabilities)
    {
        var labels = new Tensor<T>(probabilities.Shape.ToArray());
        double area = 0, confidence = 0;
        var values = probabilities.ToArray();
        for (int i = 0; i < values.Length; i++)
        {
            double p = NumOps.ToDouble(values[i]);
            if (p <= 0.5) continue;
            labels[i] = NumOps.One;
            area++;
            confidence += p;
        }
        var structures = area > 0
            ? new List<SegmentedStructure> { new() { ClassId = 1, Name = "Prompted structure", VolumeOrArea = area, MeanConfidence = confidence / area } }
            : new List<SegmentedStructure>();
        return new MedicalSegmentationResult<T>
        {
            Labels = labels,
            Probabilities = Engine.Reshape(probabilities, new[] { 1 }.Concat(probabilities.Shape.ToArray()).ToArray()),
            Structures = structures,
        };
    }

    /// <summary>An image <c>[C, H, W]</c> without a prompt or memory: the decoder on the model's empty prompt.</summary>
    public override MedicalSegmentationResult<T> SegmentSlice(Tensor<T> slice)
    {
        return Result(SegmentSet(new List<Tensor<T>> { slice }, (_, _) => new Dictionary<int, Sam2Prompt<T>>())[0]);
    }

    /// <summary>A volume <c>[C, D, H, W]</c> (or a slice) without a prompt: the axial slices tracked in both orders from the
    /// model's empty prompt on the first slice of each, and averaged.</summary>
    public override MedicalSegmentationResult<T> SegmentVolume(Tensor<T> volume)
    {
        if (volume.Rank <= 3) return SegmentSlice(volume);
        var slices = Slices(volume, 0);
        var prompts = new Dictionary<int, Sam2Prompt<T>> { [0] = Sam2Runtime<T>.NoPrompt };
        var forward = TrackAxis(slices, prompts, reversed: false, axis: 0);
        var backward = TrackAxis(slices, new Dictionary<int, Sam2Prompt<T>> { [slices.Count - 1] = Sam2Runtime<T>.NoPrompt }, reversed: true, axis: 0);
        return Result(Engine.TensorMultiplyScalar(Engine.TensorAdd(forward, backward), NumOps.FromDouble(0.5)));
    }

    // ------------------------------------------------------------------ the tensor contract and training

    /// <summary>The single-mask logits <c>[B, 1, H, W]</c> (or <c>[1, H, W]</c>) of images with no prompt and no memory.</summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        int b = input.Shape[0], h = input.Shape[2], w = input.Shape[3];
        var outputs = new Tensor<T>[b];
        for (int i = 0; i < b; i++)
        {
            var frame = Rgb(Engine.TensorSlice(input, new[] { i, 0, 0, 0 }, new[] { 1, input.Shape[1], h, w }));
            var output = Runtime.TrackImageSet(new[] { frame }, new Dictionary<int, Sam2Prompt<T>>(), Seed)[0];
            outputs[i] = SpatialResampling.Resize(Engine, output.LowRes, h, w);
        }
        var logits = b == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 0);
        return hasBatch ? logits : RemoveBatchDimension(logits);
    }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    /// <summary>An image set <c>[B, C, H, W]</c> (or one image) as RGB frames <c>[B, 3, H, W]</c> with its masks.</summary>
    private (Tensor<T> Images, Tensor<T> Masks) ImageSet(Tensor<T> input, Tensor<T> target)
    {
        var batch = input.Rank == 3 ? AddBatchDimension(input) : input;
        if (batch.Rank != 4) throw new ArgumentException("Medical SAM 2 trains on images [C, H, W] or an image set [B, C, H, W].", nameof(input));
        int b = batch.Shape[0], h = batch.Shape[2], w = batch.Shape[3];
        if (target.Length != b * h * w) throw new ArgumentException($"The target is one {h}x{w} mask per image ({b}).", nameof(target));
        var rgb = Engine.TensorConcatenate(Enumerable.Range(0, b)
            .Select(i => Rgb(Engine.TensorSlice(batch, new[] { i, 0, 0, 0 }, new[] { 1, batch.Shape[1], h, w }))).ToArray(), 0);
        return (rgb, Engine.Reshape(target, new[] { b, h, w }));
    }

    /// <summary>
    /// One step of training on an image <c>[C, H, W]</c> or an image set <c>[B, C, H, W]</c> and its masks: the set is a
    /// pseudo-video with SAM 2's prompt simulation (Zhu et al. 2024, section 4.3), its memory the self-sorting bank. Volumes
    /// train through <see cref="TrainOnVolume"/>.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var (images, masks) = ImageSet(input, expectedOutput);
        Step(new[] { (images, masks) }, bank: true);
        return true;
    }

    /// <summary>One step of SAM 2's video training on a structure through consecutive axial slices of a volume
    /// <c>[C, D, H, W]</c> and its masks <c>[D, H, W]</c> (the paper samples 8-slice sequences).</summary>
    public void TrainOnVolume(Tensor<T> volume, Tensor<T> masks)
    {
        if (volume.Rank != 4) throw new ArgumentException("A volume is [C, D, H, W].", nameof(volume));
        var frames = Slices(volume, 0);
        Step(new[] { (Engine.TensorConcatenate(frames.ToArray(), 0), masks) }, bank: false);
    }

    private void Step(IReadOnlyList<(Tensor<T> Input, Tensor<T> Target)> examples, bool bank)
    {
        SetTrainingMode(true);
        try
        {
            LastLoss = Runtime.Train(examples, gradients => PublishParameterGradients(gradients), bank);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>Medical SAM 2 trains on a supervised objective: its losses against a structure's masks.</summary>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    /// <inheritdoc/>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <summary>The training objective estimated with common random numbers (fixed draws), so two evaluations compare the
    /// same objective.</summary>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var (images, masks) = ImageSet(input, target);
        return Runtime.Evaluate(new[] { (images, masks) }, bank: true);
    }

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
            { "ModelName", "MedicalSAM2" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "ImageSize", Network.Configuration.ImageSize },
            { "ModelSize", _options.ModelSize.ToString() },
            { "MemoryBankCapacity", _options.MemoryBankCapacity },
            { "NumLayers", Layers.Count },
        },
        ModelDataProvider = () => SerializeForMetadata()
    };
}
