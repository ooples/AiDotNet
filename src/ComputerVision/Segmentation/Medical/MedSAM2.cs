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
/// MedSAM2: SAM 2.1 fine-tuned to segment anything in 3D medical images and videos.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Draw a box around a lesion or organ on one slice of a CT or MRI scan (or one frame of an
/// ultrasound or endoscopy video), and MedSAM2 segments it in that slice and follows it through the rest of the volume.
/// </para>
/// <para>
/// MedSAM2 (Ma et al. 2025, bowang-lab/MedSAM2) is SAM2.1-Tiny at 512 pixels (Hiera blocks 1-2-7-2, four memory-attention
/// layers with 2D RoPE on the 32×32 embedding), fine-tuned on medical volumes and videos. It shares SAM 2's network,
/// tracking and training (<see cref="Foundation.SAM21{T}"/>, verified against the official code at 1e-9). A volume's slices
/// are frames: grayscale slices are repeated to three channels, and the box on the key slice is propagated forward to the
/// last slice and, starting again from the key slice, backward to the first, as the official inference does. Training
/// follows the paper (AdamW, 3e-5 for the image encoder and 5e-5 for the rest, weight decay 0.01, focal and dice 20:1,
/// boxes moved by up to 10 pixels) and the official config where the paper is silent (see <see cref="MedSAM2Options"/>).
/// Released checkpoints load with <see cref="LoadPretrainedWeights"/> once converted to Hugging Face's format.
/// </para>
/// <para>
/// <b>Reference:</b> Ma et al., "MedSAM2: Segment Anything in 3D Medical Images and Videos", arXiv 2025.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 1, outputSize: 1);
/// var model = new MedSAM2&lt;float&gt;(architecture);
/// model.LoadPretrainedWeights("MedSAM2/model.safetensors");
/// // volume [1, D, H, W] with values in 0..1 (CT windowed to 0..255, then divided by 255)
/// var result = model.SegmentVolume(volume, keySlice: 42, box: new float[] { 120, 140, 210, 230 });
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MedSAM2: Segment Anything in 3D Medical Images and Videos", "https://arxiv.org/abs/2504.03600", Year = 2025,
    Authors = "Jun Ma, Zongxin Yang, Sumin Kim, Bihui Chen, Mohammed Baharoon, Adibvafa Fallahpour, Reza Asakereh, Hongwei Lyu, Bo Wang")]
[PaperOptimizer(OptimizerKind.AdamW, Beta1 = 0.9, Beta2 = 0.999, WeightDecay = 0.01, MaxGradientNorm = 0.1,
                Schedule = LearningRateSchedulerType.CosineAnnealing, Phase = TrainingPhase.FineTuning,
                Source = "Ma et al. 2025, Methods: AdamW with beta1 0.9, beta2 0.999 and a weight decay of 0.01, at 3.0e-5 "
                        + "for the image encoder and 5.0e-5 for the other components, for 70 epochs. The official config "
                        + "adds a cosine schedule to a tenth, layer-wise decay 0.9 and clipping at 0.1. The model builds this "
                        + "optimizer itself (Sam2Optimizer): no single rate is declared because the rates differ per parameter.")]
public partial class MedSAM2<T> : Common.MedicalSegmentationBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly MedSAM2Options _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _userOptimizer;
    private Sam2Runtime<T>? _runtime;

    private static readonly string[] MedSAM2Modalities = ["CT", "MRI", "PET", "Ultrasound", "Endoscopy"];

    private Sam2Runtime<T> Runtime => _runtime ?? throw new InvalidOperationException("The network is not initialized.");

    /// <summary>The network.</summary>
    internal Sam2Network<T> Network => Runtime.Network;

    /// <summary>Creates a trainable MedSAM2.</summary>
    /// <param name="architecture">The slices' size (512 by default) and channels (1 for grayscale scans, 3 for color video).</param>
    /// <param name="optimizer">The optimizer (default: the paper's AdamW recipe).</param>
    /// <param name="lossFunction">Reported by the base; training minimizes the paper's losses.</param>
    /// <param name="options">Model options.</param>
    public MedSAM2(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        MedSAM2Options? options = null)
        : base(architecture, optimizer, lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>(), (options ??= new MedSAM2Options()).NumClasses, MedSAM2Modalities)
    {
        _options = options;
        _options.Validate();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        if (_channels is not (1 or 3)) throw new ArgumentException("MedSAM2 segments grayscale (1-channel) or RGB (3-channel) images.", nameof(architecture));
        _userOptimizer = optimizer;
        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        if (_options is null || _runtime is not null) return;
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("MedSAM2 builds the paper's network; custom layers are not supported.");
        // SAM2.1-Tiny, at 512 pixels unless the options say otherwise (a smaller canvas follows the architecture).
        int side = _options.ImageSize ?? 512;
        var training = new Sam2TrainingSettings
        {
            PointProbability = 0.5, BoxProbability = 1.0, ReverseProbability = 0.0, BoxNoiseBound = 10.0, CorrectedFramesAsPrompted = true,
        };
        _runtime = new Sam2Runtime<T>(Engine, _options, sizeIndex: 0, sam21: true, side, training, _userOptimizer);
        Layers.AddRange(_runtime.Layers);
        _encoderLayerEnd = _runtime.Layers.Count;
    }

    /// <summary>Loads a checkpoint in Hugging Face's <c>Sam2VideoModel</c> format (MedSAM2's released <c>.pt</c> converted with
    /// transformers' SAM 2 conversion script).</summary>
    public void LoadPretrainedWeights(string path) => Network.LoadSafetensors(path);

    /// <summary>Loads a Hugging Face <c>Sam2VideoModel</c> state dict.</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        if (_runtime is not null) _runtime.Network.Training = isTraining;
    }

    // ------------------------------------------------------------------ slices, volumes and videos

    /// <summary>A slice or frame <c>[C, H, W]</c> (C 1 or 3, values in 0..1) as an RGB frame <c>[1, 3, H, W]</c>: a grayscale
    /// slice is repeated to three channels, as the official preprocessing does.</summary>
    private Tensor<T> Rgb(Tensor<T> slice)
    {
        var frame = Sam2Runtime<T>.Frame(slice, nameof(slice));
        if (frame.Shape[1] == 3) return frame;
        if (frame.Shape[1] != 1) throw new ArgumentException("MedSAM2 takes grayscale (1-channel) or RGB (3-channel) slices.", nameof(slice));
        return Engine.TensorConcatenate(new[] { frame, frame, frame }, 1);
    }

    /// <summary>The slices of a volume <c>[C, D, H, W]</c> (or a video <c>[F, C, H, W]</c> with <paramref name="video"/>) as
    /// RGB frames.</summary>
    private List<Tensor<T>> Slices(Tensor<T> volume, bool video)
    {
        if (volume.Rank != 4) throw new ArgumentException(video ? "A video is [F, C, H, W]." : "A volume is [C, D, H, W].", nameof(volume));
        int channels = video ? volume.Shape[1] : volume.Shape[0], count = video ? volume.Shape[0] : volume.Shape[1];
        int h = volume.Shape[2], w = volume.Shape[3];
        return Enumerable.Range(0, count).Select(t => Rgb(video
            ? Engine.TensorSlice(volume, new[] { t, 0, 0, 0 }, new[] { 1, channels, h, w })
            : Engine.Reshape(Engine.TensorSlice(volume, new[] { 0, t, 0, 0 }, new[] { channels, 1, h, w }), new[] { 1, channels, h, w }))).ToList();
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

    private Tensor<T> Track(List<Tensor<T>> frames, int keyFrame, Sam2Prompt<T> prompt)
    {
        if (keyFrame < 0 || keyFrame >= frames.Count) throw new ArgumentOutOfRangeException(nameof(keyFrame));
        int h = frames[0].Shape[2], w = frames[0].Shape[3];
        var outputs = Runtime.Track(frames, new Dictionary<int, Sam2Prompt<T>> { [keyFrame] = prompt }, freshReverse: true);
        return Engine.TensorStack(outputs.Select(o => Runtime.VideoProbabilities(o, h, w)).ToArray(), 0);
    }

    /// <summary>Segments a structure in one slice <c>[C, H, W]</c> from a box <c>[x0, y0, x1, y1]</c> (the image predictor).</summary>
    public MedicalSegmentationResult<T> SegmentSlice(Tensor<T> slice, float[] box)
    {
        var frame = Rgb(slice);
        int h = frame.Shape[2], w = frame.Shape[3];
        var prompt = Runtime.BoxPrompt(box.Select(v => (double)v).ToList(), h, w);
        var (logits, _, _) = Runtime.DecodeImage(Runtime.EncodeImage(frame), prompt.Points, prompt.Labels, null, multimask: false);
        return Result(Runtime.Probabilities(logits, h, w));
    }

    /// <summary>
    /// Segments a structure through a volume <c>[C, D, H, W]</c> (values in 0..1) from a box <c>[x0, y0, x1, y1]</c> on its key
    /// slice: the box is propagated to the last slice and, starting again from the key slice, to the first.
    /// </summary>
    public MedicalSegmentationResult<T> SegmentVolume(Tensor<T> volume, int keySlice, float[] box)
    {
        var slices = Slices(volume, video: false);
        int h = slices[0].Shape[2], w = slices[0].Shape[3];
        return Result(Track(slices, keySlice, Runtime.BoxPrompt(box.Select(v => (double)v).ToList(), h, w)));
    }

    /// <summary>Segments a structure through a video <c>[F, C, H, W]</c> (values in 0..1) from a box on one frame; each
    /// frame's probabilities <c>[F, H, W]</c>.</summary>
    public Tensor<T> SegmentVideo(Tensor<T> video, int keyFrame, float[] box)
    {
        var frames = Slices(video, video: true);
        int h = frames[0].Shape[2], w = frames[0].Shape[3];
        return Track(frames, keyFrame, Runtime.BoxPrompt(box.Select(v => (double)v).ToList(), h, w));
    }

    /// <summary>A slice <c>[C, H, W]</c> without a prompt: the decoder on the model's empty prompt (a padding point).</summary>
    public override MedicalSegmentationResult<T> SegmentSlice(Tensor<T> slice)
    {
        var frame = Rgb(slice);
        int h = frame.Shape[2], w = frame.Shape[3];
        var (logits, _, _) = Runtime.DecodeImage(Runtime.EncodeImage(frame), null, null, null, multimask: false);
        return Result(Runtime.Probabilities(logits, h, w));
    }

    /// <summary>A volume <c>[C, D, H, W]</c> (or a slice) without a prompt: the middle slice decoded from the model's empty
    /// prompt and propagated both ways.</summary>
    public override MedicalSegmentationResult<T> SegmentVolume(Tensor<T> volume)
    {
        if (volume.Rank <= 3) return SegmentSlice(volume);
        var slices = Slices(volume, video: false);
        return Result(Track(slices, slices.Count / 2, Sam2Runtime<T>.NoPrompt));
    }

    // ------------------------------------------------------------------ the tensor contract and training

    /// <summary>The single-mask logits <c>[B, 1, H, W]</c> (or <c>[1, H, W]</c>) of slices with no prompt.</summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        int b = input.Shape[0], h = input.Shape[2], w = input.Shape[3];
        var outputs = new Tensor<T>[b];
        for (int i = 0; i < b; i++)
        {
            var frame = Rgb(Engine.TensorSlice(input, new[] { i, 0, 0, 0 }, new[] { 1, input.Shape[1], h, w }));
            var (low, _, _) = Runtime.DecodeImage(Runtime.EncodeImage(frame), null, null, null, multimask: false);
            outputs[i] = SpatialResampling.Resize(Engine, low, h, w);
        }
        var logits = b == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 0);
        return hasBatch ? logits : RemoveBatchDimension(logits);
    }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    private IReadOnlyList<(Tensor<T> Input, Tensor<T> Target)> Images(Tensor<T> input, Tensor<T> target)
    {
        if (input.Rank == 3) return new[] { (Engine.Reshape(Rgb(input), new[] { 3, input.Shape[1], input.Shape[2] }), target) };
        if (input.Rank != 4) throw new ArgumentException("MedSAM2 trains on slices [C, H, W] or [B, C, H, W].", nameof(input));
        int b = input.Shape[0], c = input.Shape[1], h = input.Shape[2], w = input.Shape[3];
        if (target.Length != b * h * w) throw new ArgumentException($"The target is one {h}x{w} mask per slice ({b}).", nameof(target));
        var flat = Engine.Reshape(target, new[] { b, h * w });
        return Enumerable.Range(0, b).Select(i => (
            Engine.Reshape(Rgb(Engine.TensorSlice(input, new[] { i, 0, 0, 0 }, new[] { 1, c, h, w })), new[] { 3, h, w }),
            Engine.Reshape(Engine.TensorSlice(flat, new[] { i, 0 }, new[] { 1, h * w }), new[] { h, w }))).ToList();
    }

    /// <summary>
    /// One step of MedSAM2's training on slices <c>[C, H, W]</c> or <c>[B, C, H, W]</c> and their masks: each slice is a
    /// one-frame example (a box prompt with corrective clicks), and the step minimizes the mean loss. Volumes and videos train
    /// through <see cref="TrainOnVolume"/>.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        Step(Images(input, expectedOutput));
        return true;
    }

    /// <summary>One step of MedSAM2's training on a structure through consecutive slices of a volume <c>[C, D, H, W]</c> (or
    /// frames of a video <c>[F, C, H, W]</c> with <paramref name="video"/>) and its masks <c>[D, H, W]</c>; the paper trains on
    /// eight consecutive slices or frames.</summary>
    public void TrainOnVolume(Tensor<T> volume, Tensor<T> masks, bool video = false)
    {
        var frames = Slices(volume, video);
        Step(new[] { (Engine.TensorConcatenate(frames.ToArray(), 0), masks) });
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

    /// <summary>MedSAM2 trains on a supervised objective: its losses against a structure's masks.</summary>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    /// <inheritdoc/>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <summary>The training objective estimated with common random numbers (fixed draws), so two evaluations compare the
    /// same objective.</summary>
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
            { "ModelName", "MedSAM2" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "ImageSize", Network.Configuration.ImageSize },
            { "MemoryBankSize", _options.MemoryBankSize },
            { "NumLayers", Layers.Count },
        },
        ModelDataProvider = () => SerializeForMetadata()
    };
}
