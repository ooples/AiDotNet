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
/// <b>Technical Details:</b> the official network (facebookresearch/segment-anything), checked against the official code:
/// - ViT-B/L/H image encoder on 16x16 patches, windowed and global attention with relative positions, and a neck
/// - Prompt encoder for points, boxes and masks (random Fourier positions)
/// - Two-way transformer mask decoder with an IoU prediction head
/// - Ambiguity-aware: three masks for a single point
/// - SetImage encodes once (longest side resized to the encoder size); prompts then only run the decoder
/// - SegmentEverything is the official automatic mask generator; Train is the paper's interactive training simulation
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
///     options: new SAMOptions { ModelSize = SAMModelSize.ViTHuge });
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
    private readonly SAMOptions _options;
    private SamNetwork<T>? _network;

    /// <summary>Draws the prompts SAM's interactive training simulates.</summary>
    private Random _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);

    // The image set by SetImage: its size, the size the encoder saw it at, and the embedding (the base's field).
    private int _originalHeight, _originalWidth, _inputHeight, _inputWidth;

    internal bool UseNativeMode => _useNativeMode;
    internal SAMModelSize ModelSize => _options.ModelSize;

    /// <summary>The network (image encoder, prompt encoder and mask decoder).</summary>
    internal SamNetwork<T> Network => _network ?? throw new InvalidOperationException("The native network is not initialized.");

    /// <summary>Creates a trainable SAM.</summary>
    /// <param name="architecture">The architecture; its input height and width are the encoder's square size (1024 by
    /// default).</param>
    /// <param name="optimizer">The optimizer (default: the paper's AdamW recipe).</param>
    /// <param name="lossFunction">Reported by the base; training minimizes the paper's focal, dice and IoU losses.</param>
    /// <param name="options">Model options.</param>
    public SAM(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SAMOptions? options = null)
        : base(architecture, optimizer, lossFunction ?? MaskLoss(options ??= new SAMOptions()), 1)
    {
        _options = options ?? new SAMOptions(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        InitializeLayers();
    }

    /// <summary>Creates an inference-only SAM from an ONNX model.</summary>
    public SAM(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SAMOptions? options = null)
        : base(architecture, onnxModelPath, 1)
    {
        _options = options ?? new SAMOptions(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
    }

    /// <summary>The loss the base reports: the paper's focal and dice terms on mask logits.</summary>
    private static ILossFunction<T> MaskLoss(SAMOptions options) => new CompositeLossWithLogits<T>(
        (new FocalLoss<T>(gamma: options.FocalGamma, alpha: options.FocalAlpha), options.MaskFocalWeight),
        (new DiceLoss<T>(), options.MaskDiceWeight));

    /// <summary>
    /// The official configuration of the selected size, its encoder square sized to the architecture's longer side
    /// (rounded up to the 16-pixel patch). As the official predictor, an image of any aspect is resized so its longer
    /// side is the encoder size and zero-padded to the square (<see cref="EncodeImage"/>, ResizeLongestSide and
    /// Sam.preprocess), so a non-square architecture needs no square input.
    /// </summary>
    private SamConfiguration Configuration()
    {
        int longer = Math.Max(_height, _width);
        int imageSize = (longer + 15) / 16 * 16;
        return SamConfiguration.For(_options.ModelSize) with
        {
            ImageSize = imageSize, Channels = _channels, EncoderDim = _options.EncoderDim, EncoderDepth = _options.EncoderDepth,
            EncoderHeads = _options.EncoderHeads, GlobalAttention = _options.GlobalAttentionBlocks,
        };
    }

    protected override void InitializeLayers()
    {
        if (_options is null || _network is not null) return;
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("SAM builds the paper's network; custom layers are not supported.");
        var layers = new List<LayerBase<T>>();
        _network = new SamNetwork<T>(Engine, layers, Configuration());
        int seed = _options.Seed ?? 42;
        _network.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
        _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1);
        Layers.AddRange(layers);
        _encoderLayerEnd = layers.Count;
    }

    /// <summary>Loads an official SAM checkpoint's state dict.</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    // ------------------------------------------------------------------ image setting (SamPredictor)

    /// <summary>The official ResizeLongestSide: the longer side becomes the encoder size, the other scales with it.</summary>
    private (int Height, int Width) ResizedSize(int height, int width)
    {
        double scale = (double)Network.Configuration.ImageSize / Math.Max(height, width);
        return ((int)(height * scale + 0.5), (int)(width * scale + 0.5));
    }

    private Tensor<T> Batched(Tensor<T> image)
    {
        if (image.Rank == 3) return AddBatchDimension(image);
        if (image.Rank != 4 || image.Shape[0] != 1) throw new ArgumentException("SAM takes one image [C,H,W] or [1,C,H,W].", nameof(image));
        return image;
    }

    /// <summary>The embedding of an image (pixel values in 0..255, as the official pixel mean and std expect), resized
    /// so its longer side is the encoder size and zero-padded to a square.</summary>
    protected override Tensor<T> EncodeImage(Tensor<T> image)
    {
        var x = Batched(image);
        int h = x.Shape[2], w = x.Shape[3];
        var (rh, rw) = ResizedSize(h, w);
        if (rh != h || rw != w) x = SpatialResampling.ResizeAntialias(Engine, x, rh, rw);
        return Network.EncodeImage(Network.Preprocess(x));
    }

    /// <inheritdoc />
    public override void SetImage(Tensor<T> image)
    {
        var x = Batched(image);
        _originalHeight = x.Shape[2];
        _originalWidth = x.Shape[3];
        (_inputHeight, _inputWidth) = ResizedSize(_originalHeight, _originalWidth);
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            _imageEmbedding = EncodeImage(x);
        _imageSet = true;
    }

    /// <summary>Original-image coordinates in the encoder's frame (the official <c>apply_coords</c>).</summary>
    private double Scale(double coordinate, bool horizontal)
        => coordinate * (horizontal ? (double)_inputWidth / _originalWidth : (double)_inputHeight / _originalHeight);

    /// <summary>The official stability score: the IoU of the masks thresholded at ±offset around the mask threshold.</summary>
    private static double StabilityScore(double[] logits, double offset)
    {
        int high = 0, low = 0;
        foreach (double v in logits)
        {
            if (v > offset) high++;
            if (v > -offset) low++;
        }
        return low == 0 ? 0.0 : (double)high / low;
    }

    /// <summary>Runs the decoder for prompts in the encoder frame and returns the masks at the original size.</summary>
    private (Tensor<T> Logits, Tensor<T> Iou, Tensor<T> LowRes) Predict(Tensor<T>? points, Tensor<T>? labels, Tensor<T>? boxes, Tensor<T>? masks, bool multimask)
    {
        EnsureImageSet();
        var net = Network;
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var (sparse, dense) = net.EncodePrompts(points, labels, boxes, masks);
        var (low, iou) = net.DecodeMasks(_imageEmbedding!, net.DensePositional(), sparse, dense, multimask);
        return (net.Postprocess(low, _inputHeight, _inputWidth, _originalHeight, _originalWidth), iou, low);
    }

    private PromptedSegmentationResult<T> Result(Tensor<T> logits, Tensor<T> iou, Tensor<T> low)
    {
        int count = logits.Shape[1], h = logits.Shape[2], w = logits.Shape[3];
        var masks = new Tensor<T>(new[] { count, h, w });
        var scores = new double[count];
        var stability = new double[count];
        var values = logits.ToArray();
        for (int m = 0; m < count; m++)
        {
            var plane = new double[h * w];
            for (int i = 0; i < h * w; i++)
            {
                plane[i] = NumOps.ToDouble(values[m * h * w + i]);
                masks[m * h * w + i] = plane[i] > 0 ? NumOps.One : NumOps.Zero;
            }
            scores[m] = NumOps.ToDouble(iou[0, m]);
            stability[m] = StabilityScore(plane, 1.0);
        }
        return new PromptedSegmentationResult<T> { Masks = masks, Scores = scores, StabilityScores = stability, LowResLogits = low };
    }

    /// <summary>The three ambiguity-aware masks for points <c>[k, 2]</c> (x, y in the image) with labels <c>[k]</c>
    /// (1 foreground, 0 background), as SamPredictor gives them with multimask output.</summary>
    public override PromptedSegmentationResult<T> SegmentFromPoints(Tensor<T> points, Tensor<T> labels)
    {
        int k = points.Shape[0];
        var p = new Tensor<T>(new[] { 1, k, 2 });
        var l = new Tensor<T>(new[] { 1, k });
        for (int i = 0; i < k; i++)
        {
            p[0, i, 0] = NumOps.FromDouble(Scale(NumOps.ToDouble(points[i, 0]), true));
            p[0, i, 1] = NumOps.FromDouble(Scale(NumOps.ToDouble(points[i, 1]), false));
            l[0, i] = labels[i];
        }
        var (logits, iou, low) = Predict(p, l, null, null, multimask: true);
        return Result(logits, iou, low);
    }

    /// <summary>The mask for a box <c>[4]</c> (x0, y0, x1, y1 in the image): a single unambiguous mask.</summary>
    public override PromptedSegmentationResult<T> SegmentFromBox(Tensor<T> box)
    {
        var b = new Tensor<T>(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) b[0, i] = NumOps.FromDouble(Scale(NumOps.ToDouble(box[i]), i % 2 == 0));
        var (logits, iou, low) = Predict(null, null, b, null, multimask: false);
        return Result(logits, iou, low);
    }

    /// <summary>
    /// The mask for a mask prompt of logits: either low-resolution logits <c>[4·g, 4·g]</c> (a previous result's
    /// <see cref="PromptedSegmentationResult{T}.LowResLogits"/> plane, SamPredictor's <c>mask_input</c>), or a map of the
    /// image's size <c>[H, W]</c>, which is brought to the prompt resolution the way the image reaches the encoder
    /// (longest side resized, padded to a square) and then downscaled 4×.
    /// </summary>
    public override PromptedSegmentationResult<T> SegmentFromMask(Tensor<T> mask)
    {
        EnsureImageSet();
        int side = 4 * Network.Configuration.Grid, size = Network.Configuration.ImageSize;
        Tensor<T> m;
        if (mask.Length == side * side)
        {
            m = Engine.Reshape(mask, new[] { 1, 1, side, side });
        }
        else if (mask.Length == _originalHeight * _originalWidth)
        {
            m = Engine.Reshape(mask, new[] { 1, 1, _originalHeight, _originalWidth });
            if (_inputHeight != _originalHeight || _inputWidth != _originalWidth)
                m = SpatialResampling.ResizeAntialias(Engine, m, _inputHeight, _inputWidth);
            if (_inputWidth < size) m = Engine.TensorConcatenate(new[] { m, new Tensor<T>(new[] { 1, 1, _inputHeight, size - _inputWidth }) }, 3);
            if (_inputHeight < size) m = Engine.TensorConcatenate(new[] { m, new Tensor<T>(new[] { 1, 1, size - _inputHeight, size }) }, 2);
            m = SpatialResampling.ResizeAntialias(Engine, m, side, side);
        }
        else
        {
            throw new ArgumentException(
                $"A mask prompt is a {side}x{side} plane of low-resolution logits or a {_originalHeight}x{_originalWidth} map of the image.", nameof(mask));
        }
        var (logits, iou, low) = Predict(null, null, null, m, multimask: false);
        return Result(logits, iou, low);
    }

    /// <summary>
    /// The official automatic mask generator with its default settings: a 32×32 grid of foreground points, three masks
    /// per point, kept when the predicted IoU exceeds 0.88 and the stability score (offset 1.0) reaches 0.95, then box
    /// non-maximum suppression at IoU 0.7, highest predicted IoU first.
    /// </summary>
    public override List<PromptedSegmentationResult<T>> SegmentEverything()
    {
        EnsureImageSet();
        const int perSide = 32, perBatch = 64;
        const double iouThreshold = 0.88, stabilityThreshold = 0.95, nmsThreshold = 0.7;
        var candidates = new List<(double Iou, double Stability, double[] Box, Tensor<T> Mask, Tensor<T> Low)>();
        var grid = new List<(double X, double Y)>();
        for (int i = 0; i < perSide; i++)
            for (int j = 0; j < perSide; j++)
            {
                double offset = 1.0 / (2 * perSide), step = (1 - 2 * offset) / (perSide - 1);
                grid.Add(((offset + j * step) * _originalWidth, (offset + i * step) * _originalHeight));
            }
        for (int start = 0; start < grid.Count; start += perBatch)
        {
            int n = Math.Min(perBatch, grid.Count - start);
            var points = new Tensor<T>(new[] { n, 1, 2 });
            var labels = new Tensor<T>(new[] { n, 1 });
            for (int i = 0; i < n; i++)
            {
                points[i, 0, 0] = NumOps.FromDouble(Scale(grid[start + i].X, true));
                points[i, 0, 1] = NumOps.FromDouble(Scale(grid[start + i].Y, false));
                labels[i, 0] = NumOps.One;
            }
            var (logits, iou, low) = Predict(points, labels, null, null, multimask: true);
            int masks = logits.Shape[1], h = logits.Shape[2], w = logits.Shape[3];
            var values = logits.ToArray();
            for (int b = 0; b < n; b++)
                for (int m = 0; m < masks; m++)
                {
                    double predicted = NumOps.ToDouble(iou[b, m]);
                    if (predicted <= iouThreshold) continue;
                    var plane = new double[h * w];
                    Array.Copy(values.Select(v => NumOps.ToDouble(v)).Skip((b * masks + m) * h * w).Take(h * w).ToArray(), plane, h * w);
                    double stability = StabilityScore(plane, 1.0);
                    if (stability < stabilityThreshold) continue;
                    var mask = new Tensor<T>(new[] { 1, h, w });
                    int x0 = int.MaxValue, y0 = int.MaxValue, x1 = -1, y1 = -1;
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++)
                        {
                            if (plane[y * w + x] <= 0) continue;
                            mask[0, y, x] = NumOps.One;
                            x0 = Math.Min(x0, x); y0 = Math.Min(y0, y); x1 = Math.Max(x1, x); y1 = Math.Max(y1, y);
                        }
                    // An empty mask's box is [0, 0, 0, 0], as batched_mask_to_box gives it.
                    var box = x1 < 0 ? new double[4] : new double[] { x0, y0, x1, y1 };
                    var lowPlane = Engine.Reshape(Engine.TensorSlice(low, new[] { b, m, 0, 0 }, new[] { 1, 1, low.Shape[2], low.Shape[3] }),
                        new[] { low.Shape[2], low.Shape[3] });
                    candidates.Add((predicted, stability, box, mask, lowPlane));
                }
        }
        // torchvision's batched_nms with one category: highest score first, drop boxes overlapping a kept one above the
        // threshold (areas without +1, as torchvision computes them).
        var order = Enumerable.Range(0, candidates.Count).OrderByDescending(i => candidates[i].Iou).ToList();
        var kept = new List<int>();
        foreach (int i in order)
        {
            var a = candidates[i].Box;
            bool suppressed = kept.Any(j =>
            {
                var b = candidates[j].Box;
                double iw = Math.Max(0, Math.Min(a[2], b[2]) - Math.Max(a[0], b[0]));
                double ih = Math.Max(0, Math.Min(a[3], b[3]) - Math.Max(a[1], b[1]));
                double inter = iw * ih;
                double union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
                return union > 0 && inter / union > nmsThreshold;
            });
            if (!suppressed) kept.Add(i);
        }
        return kept.Select(i => new PromptedSegmentationResult<T>
        {
            Masks = candidates[i].Mask, Scores = new[] { candidates[i].Iou }, StabilityScores = new[] { candidates[i].Stability },
            LowResLogits = candidates[i].Low,
        }).ToList();
    }

    // ------------------------------------------------------------------ inference without prompts, and training

    /// <summary>The single-mask logits <c>[B, 1, H, W]</c> (or <c>[1, H, W]</c>) of images at the encoder size with no
    /// prompt: the decoder's output tokens alone over the "no mask" embedding.</summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        var net = Network;
        int b = input.Shape[0], h = input.Shape[2], w = input.Shape[3];
        var embedding = net.EncodeImage(net.Preprocess(input));
        var positional = net.DensePositional();
        var outputs = new Tensor<T>[b];
        for (int i = 0; i < b; i++)
        {
            var one = Engine.TensorSlice(embedding, new[] { i, 0, 0, 0 }, new[] { 1, embedding.Shape[1], embedding.Shape[2], embedding.Shape[3] });
            var (sparse, dense) = net.EncodePrompts(null, null, null, null);
            outputs[i] = net.Postprocess(net.DecodeMasks(one, positional, sparse, dense, multimask: false).Logits, h, w, h, w);
        }
        var logits = b == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 0);
        return hasBatch ? logits : RemoveBatchDimension(logits);
    }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    /// <summary>
    /// One step of SAM's training (Kirillov et al. 2023, Appendix A) on an image <c>[C,H,W]</c> (or <c>[1,C,H,W]</c>,
    /// at most the encoder size) and its ground-truth mask (<c>H·W</c> values): see <see cref="SamTraining{T}"/>.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");
        var image = Batched(input);
        var training = new SamTraining<T>(Engine, Network, _options.MaskFocalWeight, _options.MaskDiceWeight, _options.FocalGamma, _options.FocalAlpha);
        var objective = training.Objective(image, expectedOutput, _trainingRandom);
        // The objective runs the whole network itself; TrainWithCustomLoss would also run (and discard) a forward pass.
        TrainWithCustomObjective(image, expectedOutput, (_, _) => objective(), Optimizer);
        return true;
    }

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        var values = input.ToArray();
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(values[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "pixel_values";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "SAM" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "ModelSize", _options.ModelSize.ToString() },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count },
        },
        ModelDataProvider = () => SerializeForMetadata()
    };
}
