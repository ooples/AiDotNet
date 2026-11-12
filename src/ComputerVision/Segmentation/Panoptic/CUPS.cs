using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Panoptic;

/// <summary>
/// CUPS: Scene-Centric Unsupervised Panoptic Segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Unified panoptic segmentation. Semantic-aware scene understanding.
///
/// Common use cases:
/// - Unified panoptic segmentation
/// - Semantic-aware scene understanding
/// - Comprehensive pixel-level labeling
/// - Autonomous driving perception
/// </para>
/// <para>
/// <b>Technical Details:</b> CUPS trains a panoptic network without labels. Semantic pseudo labels come from
/// clustering self-supervised features into 27 pseudo classes and instance pseudo labels from motion in stereo
/// video; <see cref="GeneratePanopticPseudoLabels"/> fuses the two, and <see cref="FitThingStuffSplit"/> decides
/// which pseudo classes are things (more than <c>psi_ts = 0.08</c> of their pixels inside instance masks).
/// The panoptic network is the paper's Panoptic Cascade Mask R-CNN: a semantic head plus a Cascade Mask R-CNN
/// (RPN over an FPN on the backbone's four stages, three cascade box stages at IoU 0.5/0.6/0.7 and a mask head),
/// trained on the pseudo labels with <see cref="TrainPanoptic"/>; <see cref="SegmentPanoptic"/> combines the
/// detected thing instances with the stuff regions the way Panoptic FPN does.
/// </para>
/// <para>
/// <b>Reference:</b> Hahn et al., "Scene-Centric Unsupervised Panoptic Segmentation", CVPR 2025.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a CUPS model for unified panoptic segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 133);
/// var model = new CUPS&lt;double&gt;(architecture);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new CUPS&lt;double&gt;(architecture,
///     options: new CUPSOptions { NumClasses = 27, DropRate = 0.1 });
///
/// // Or load a pre-trained ONNX model for scene understanding
/// var onnxModel = new CUPS&lt;double&gt;(architecture, "cups.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Scene-Centric Unsupervised Panoptic Segmentation",
    "https://arxiv.org/abs/2504.01955",
    Year = 2025,
    Authors = "Oliver Hahn, Christoph Reich, Nikita Araslanov, Daniel Cremers, Christian Rupprecht, Stefan Roth")]
public partial class CUPS<T> : Common.PanopticSegmentationBase<T>
{
    private readonly CUPSOptions _options;

    #region Fields
    // Only CUPS's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from PanopticSegmentationBase -> SegmentationModelBase.
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    // Which pseudo classes are things. Starts from the base's stuff-first split and is replaced by the
    // paper's psi_ts rule once FitThingStuffSplit has seen pseudo labels.
    private readonly bool[] _isThingClass;
    // The Cascade Mask R-CNN instance head (native mode only) and the FPN that feeds it: four lateral 1x1 and four
    // output 3x3 convolutions in Layers after the semantic decoder, which ends at _decoderLayerEnd.
    private readonly CascadeMaskRCNNHead<T>? _instanceHead;
    private int _decoderLayerEnd;
    private const int FpnLevels = 4;
    [AiDotNet.Attributes.Scratch]
    private Random? _trainingRandom;
    private bool _lazyShapesResolved;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth, IsOnnxMode, NumStuffClasses and
    // NumThingClasses are all inherited and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes CUPS in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable CUPS model.
    /// </para>
    /// </remarks>
    public CUPS(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        CUPSOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults `optimizer` lazily via CreateDefaultOptimizer(), so null is passed straight through.
        // The stuff/thing split is the same one/two-thirds rule the explicit interface members used.
        : base(architecture, optimizer, lossFunction, (options ??= new CUPSOptions()).NumClasses,
            Math.Max(1, options.NumClasses / 3), options.NumClasses - Math.Max(1, options.NumClasses / 3))
    {
        _options = options; Options = _options;
        _dropRate = _options.DropRate;
        if (_options.ChannelDimensions.Length != 4 || _options.StageDepths.Length != 4)
            throw new ArgumentException("CUPS requires exactly four encoder stage widths and depths.", nameof(options));
        if (_options.ChannelDimensions.Any(d => d <= 0)
            || _options.StageDepths.Any(d => d <= 0)
            || _options.DecoderDimension <= 0)
            throw new ArgumentException("CUPS encoder widths, depths, and decoder dimension must be positive.", nameof(options));
        _channelDims = (int[])_options.ChannelDimensions.Clone();
        _depths = (int[])_options.StageDepths.Clone();
        _decoderDim = _options.DecoderDimension;
        _isThingClass = InitialThingClasses();
        _instanceHead = new CascadeMaskRCNNHead<T>(_numClasses);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes CUPS in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained CUPS from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public CUPS(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        CUPSOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession.
        : base(architecture, onnxModelPath, (options ??= new CUPSOptions()).NumClasses,
            Math.Max(1, options.NumClasses / 3), options.NumClasses - Math.Max(1, options.NumClasses / 3))
    {
        _options = options; Options = _options;
        _dropRate = _options.DropRate;
        _channelDims = [96, 192, 384, 768];
        _depths = [2, 2, 6, 2];
        _decoderDim = _options.DecoderDimension;
        _isThingClass = InitialThingClasses();
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    /// <summary>
    /// Runs a forward pass to produce segmentation logits.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Segmentation logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass an image to get a per-pixel class prediction map.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => Optimizer ?? base.GetOrCreateBaseOptimizer();

    /// <summary>
    /// One training step of the Panoptic Cascade Mask R-CNN on a panoptic pseudo label (Hahn et al. 2025, Sec. 3.3).
    /// </summary>
    /// <param name="image">The image <c>[C, H, W]</c> or <c>[1, C, H, W]</c>.</param>
    /// <param name="semanticLabels">Semantic pseudo labels <c>[H, W]</c>; <c>-1</c> is ignored (the output of
    /// <see cref="GeneratePanopticPseudoLabels"/>).</param>
    /// <param name="instanceIds">Instance ids <c>[H, W]</c>; <c>0</c> is no instance.</param>
    /// <returns>The step's loss.</returns>
    /// <remarks>
    /// <para>
    /// The objective is the semantic head's cross-entropy over the labelled pixels (the labels nearest-resampled to
    /// the head's resolution) plus the instance head's losses: the RPN loss, the three cascade stage losses and the
    /// Mask R-CNN mask loss. Every instance is one object whose class is the majority semantic label of its pixels and
    /// whose box encloses its mask. Generic <c>Train(input, target)</c> trains the semantic head only, since a
    /// semantic target carries no instances.
    /// </para>
    /// </remarks>
    public T TrainPanoptic(Tensor<T> image, Tensor<T> semanticLabels, Tensor<T> instanceIds)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (semanticLabels is null) throw new ArgumentNullException(nameof(semanticLabels));
        if (instanceIds is null) throw new ArgumentNullException(nameof(instanceIds));
        var head = _instanceHead;
        if (!_useNativeMode || head is null) throw new NotSupportedException("Training is not supported in ONNX inference mode.");
        var batched = image.Rank == 4 ? image : AddBatchDimension(image);
        if (batched.Rank != 4 || batched.Shape[0] != 1) throw new ArgumentException("TrainPanoptic takes one image.", nameof(image));
        int h = batched.Shape[2], w = batched.Shape[3];
        if (semanticLabels.Rank != 2 || semanticLabels.Shape[0] != h || semanticLabels.Shape[1] != w)
            throw new ArgumentException($"Semantic labels must be [{h}, {w}].", nameof(semanticLabels));
        if (instanceIds.Rank != 2 || instanceIds.Shape[0] != h || instanceIds.Shape[1] != w)
            throw new ArgumentException($"Instance ids must be [{h}, {w}].", nameof(instanceIds));

        var (gold, goldClasses, goldMasks) = InstanceTargets(semanticLabels, instanceIds);
        var random = _trainingRandom ??= _options.Seed is int seed
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        return TrainWithCustomObjective(batched, semanticLabels, (x, labels) =>
        {
            var (encoded, taps) = RunBackbone(x);
            var semantic = SemanticCrossEntropy(Decode(encoded), labels);
            if (gold.Count == 0) return semantic;
            return Engine.TensorAdd(semantic, head.Loss(Fpn(taps), h, w, gold, goldClasses, goldMasks, random));
        });
    }

    /// <summary>
    /// The instance targets of a panoptic pseudo label: each instance's box (x1, y1, x2, y2; the end exclusive), its
    /// class (the majority semantic label of its labelled pixels) and its mask <c>[M, 1, H, W]</c>.
    /// </summary>
    internal (List<double[]> Boxes, List<int> Classes, Tensor<T> Masks) InstanceTargets(Tensor<T> semanticLabels, Tensor<T> instanceIds)
    {
        int h = instanceIds.Shape[0], w = instanceIds.Shape[1], hw = h * w;
        var ids = Engine.Reshape(instanceIds, new[] { hw }).Cast<int>();
        int[] idCounts = Engine.TensorBinCount(ids, null).ToArray();
        var present = new List<int>();
        for (int id = 1; id < idCounts.Length; id++)
        {
            if (idCounts[id] > 0) present.Add(id);
        }

        if (present.Count == 0) return (new List<double[]>(), new List<int>(), new Tensor<T>(new[] { 0, 1, h, w }));
        int m = present.Count;
        var masks = Engine.TensorIndexSelect(
            Engine.TensorTranspose(Engine.TensorOneHot<T>(ids, idCounts.Length)),
            new Tensor<int>(present.ToArray(), new[] { m }), 0);                                       // [M, HW]
        var labels = Engine.Reshape(semanticLabels, new[] { hw }).Cast<int>();
        var labelled = Engine.TensorGreaterEqual(labels, Tensor<int>.CreateDefault(new[] { hw }, 0)); // 1 where labelled
        var classOneHot = Engine.TensorMultiply(
            Engine.TensorOneHot<T>(Engine.TensorMultiply(labels, labelled), _numClasses),
            Engine.TensorBroadcastTo(Engine.Reshape(labelled.Cast<T>(), new[] { hw, 1 }), new[] { hw, _numClasses }));
        int[] classes = Engine.TensorArgMax(Engine.TensorMatMul(masks, classOneHot), 1).ToArray();  // majority class

        var grid = Engine.Reshape(masks, new[] { m, h, w });
        var rows = Engine.ReduceMax(grid, new[] { 2 }, false);                                         // [M, H]
        var columns = Engine.ReduceMax(grid, new[] { 1 }, false);                                      // [M, W]
        int[] top = Engine.TensorArgMax(rows, 1).ToArray();
        int[] bottomGap = Engine.TensorArgMax(Engine.TensorFlip(rows, new[] { 1 }), 1).ToArray();
        int[] left = Engine.TensorArgMax(columns, 1).ToArray();
        int[] rightGap = Engine.TensorArgMax(Engine.TensorFlip(columns, new[] { 1 }), 1).ToArray();
        var boxes = new List<double[]>(m);
        for (int i = 0; i < m; i++) boxes.Add(new double[] { left[i], top[i], w - rightGap[i], h - bottomGap[i] });
        return (boxes, classes.ToList(), Engine.Reshape(masks, new[] { m, 1, h, w }));
    }

    /// <summary>Mean cross-entropy of the semantic logits over the labelled pixels (labels below zero are ignored).</summary>
    private Tensor<T> SemanticCrossEntropy(Tensor<T> logits, Tensor<T> labels)
    {
        int k = logits.Shape[1], lh = logits.Shape[2], lw = logits.Shape[3], n = lh * lw;
        var resized = Engine.Reshape(
            Engine.Interpolate(Engine.Reshape(labels, new[] { 1, 1, labels.Shape[0], labels.Shape[1] }),
                new[] { lh, lw }, InterpolateMode.Nearest, false), new[] { n }).Cast<int>();
        var labelled = Engine.TensorGreaterEqual(resized, Tensor<int>.CreateDefault(new[] { n }, 0));
        var target = Engine.TensorMultiply(
            Engine.TensorOneHot<T>(Engine.TensorMultiply(resized, labelled), k),
            Engine.TensorBroadcastTo(Engine.Reshape(labelled.Cast<T>(), new[] { n, 1 }), new[] { n, k })); // [n, K]
        var logProbabilities = Engine.TensorTranspose(
            Engine.Reshape(Engine.TensorLogSoftmax(logits, 1), new[] { k, n }));                          // [n, K]
        int count = Math.Max(1, Engine.TensorSum(labelled));
        return Engine.TensorMultiplyScalar(
            Engine.TensorNegate(Engine.ReduceSum(Engine.TensorMultiply(target, logProbabilities), null)),
            NumOps.FromDouble(1.0 / count));
    }

    #endregion

    #region Private Methods
    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        features = Decode(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

    /// <summary>The backbone's output and its stage outputs (finest first) for a batched input.</summary>
    private (Tensor<T> Encoded, List<Tensor<T>> Stages) RunBackbone(Tensor<T> batched)
        => Common.FeaturePyramidOps.RunAndTapStages(Layers, 0, _encoderLayerEnd, batched);

    /// <summary>The semantic head on the backbone's output.</summary>
    private Tensor<T> Decode(Tensor<T> encoded)
    {
        var features = encoded;
        for (int i = _encoderLayerEnd; i < _decoderLayerEnd; i++) features = Layers[i].Forward(features);
        return features;
    }

    /// <summary>The FPN levels P2-P5 (256 channels, finest first) on the backbone's last four stages.</summary>
    private List<Tensor<T>> Fpn(List<Tensor<T>> stages)
    {
        if (stages.Count < FpnLevels)
            throw new InvalidOperationException(
                $"CUPS's Cascade Mask R-CNN reads four backbone stages (strides 4 to 32), but the backbone produced {stages.Count} resolutions.");
        var lateral = new List<Func<Tensor<T>, Tensor<T>>>(FpnLevels);
        var output = new List<Func<Tensor<T>, Tensor<T>>>(FpnLevels);
        for (int i = 0; i < FpnLevels; i++)
        {
            lateral.Add(Layers[_decoderLayerEnd + i].Forward);
            output.Add(Layers[_decoderLayerEnd + FpnLevels + i].Forward);
        }

        return Common.FeaturePyramidOps.TopDown(stages.Skip(stages.Count - FpnLevels).ToList(), lateral, output);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        NumOps.ToFloatSpan(input.Data.Span.Slice(0, input.Length), inputData);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        NumOps.FromFloatSpan(outputTensor.ToDenseTensor().Buffer.Span, outputData);
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result); return result;
    }
    #endregion

    #region Abstract Implementation
    /// <summary>
    /// Initializes the encoder and decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the neural network layers.
    /// In ONNX mode, no layers are created.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Architecture.Layers.Count / 2; }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateCUPSEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateCUPSDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
            Layers.AddRange(decoderLayers);
        }

        // The FPN under the Cascade Mask R-CNN: lateral 1x1 and output 3x3 convolutions to 256 channels.
        _decoderLayerEnd = Layers.Count;
        var identity = new AiDotNet.ActivationFunctions.IdentityActivation<T>() as IActivationFunction<T>;
        for (int i = 0; i < FpnLevels; i++) Layers.Add(new ConvolutionalLayer<T>(CascadeMaskRCNNHead<T>.Channels, 1, 1, 0, identity));
        for (int i = 0; i < FpnLevels; i++) Layers.Add(new ConvolutionalLayer<T>(CascadeMaskRCNNHead<T>.Channels, 3, 1, 1, identity));
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent("cascade-mask-rcnn", _instanceHead);
    }

    /// <summary>
    /// Resolves the lazy layers through the real forward: the FPN reads backbone stages rather than its sequential
    /// predecessor, so the base class's sequential shape walk would size it wrongly.
    /// </summary>
    protected override void ResolveLazyLayerShapes()
    {
        if (_lazyShapesResolved) return;
        if (Layers is null || Layers.Count == 0) return;
        if (!_useNativeMode) { _lazyShapesResolved = true; return; }
        using (InferenceMode.Enter())
        {
            var (encoded, taps) = RunBackbone(new Tensor<T>(new[] { 1, _channels, _height, _width }));
            Decode(encoded);
            if (taps.Count >= FpnLevels) Fpn(taps);
        }

        _lazyShapesResolved = true;
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Collects metadata describing this model's configuration.
    /// </summary>
    /// <returns>Model metadata.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary for saving or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "CUPS" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelDataProvider = () => SerializeForMetadata()
    };
    #endregion

    #region IPanopticSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, NumStuffClasses and NumThingClasses
    // all come from the base; only the model-specific override remains.

    /// <summary>Whether pseudo class <paramref name="classId"/> is currently treated as a thing.</summary>
    /// <param name="classId">A class id in <c>[0, NumClasses)</c>.</param>
    public bool IsThingClass(int classId)
    {
        if (classId < 0 || classId >= _numClasses) throw new ArgumentOutOfRangeException(nameof(classId));
        return _isThingClass[classId];
    }

    private bool[] InitialThingClasses()
    {
        var flags = new bool[_numClasses];
        for (int c = _numStuffClasses; c < _numClasses; c++) flags[c] = true;
        return flags;
    }

    /// <summary>
    /// Decides which pseudo classes are things from a set of pseudo-labelled images (Hahn et al. 2025, Sec. 3.3):
    /// a class is a thing when the share of its pixels, over all images, that lie inside an instance pseudo mask
    /// exceeds <see cref="CUPSOptions.ThingStuffThreshold"/>; otherwise it is stuff.
    /// </summary>
    /// <param name="semanticPseudoLabels">Per image, the pseudo class map <c>[H, W]</c> (integral ids).</param>
    /// <param name="instanceMasks">Per image, the binary instance pseudo masks <c>[M, H, W]</c> (M may be 0).</param>
    /// <returns>The thing flag of every pseudo class.</returns>
    public IReadOnlyList<bool> FitThingStuffSplit(IReadOnlyList<Tensor<T>> semanticPseudoLabels, IReadOnlyList<Tensor<T>> instanceMasks)
    {
        if (semanticPseudoLabels is null) throw new ArgumentNullException(nameof(semanticPseudoLabels));
        if (instanceMasks is null) throw new ArgumentNullException(nameof(instanceMasks));
        if (semanticPseudoLabels.Count != instanceMasks.Count)
            throw new ArgumentException("Give one instance-mask set per semantic pseudo label map.", nameof(instanceMasks));

        var total = new long[_numClasses];
        var inside = new long[_numClasses];
        for (int i = 0; i < semanticPseudoLabels.Count; i++)
        {
            var (labels, covered) = PseudoLabelPixels(semanticPseudoLabels[i], instanceMasks[i]);
            int[] all = Engine.TensorBinCount(labels, _numClasses).ToArray();
            // Pixels outside every instance move to an extra bin.
            var binned = Engine.TensorAdd(
                Engine.TensorMultiply(labels, covered),
                Engine.TensorMultiplyScalar(Engine.ScalarMinusTensor(1, covered), _numClasses));
            int[] within = Engine.TensorBinCount(binned, _numClasses + 1).ToArray();
            for (int c = 0; c < _numClasses; c++)
            {
                total[c] += all[c];
                inside[c] += within[c];
            }
        }

        int things = 0;
        for (int c = 0; c < _numClasses; c++)
        {
            _isThingClass[c] = total[c] > 0 && (double)inside[c] / total[c] > _options.ThingStuffThreshold;
            if (_isThingClass[c]) things++;
        }

        _numThingClasses = things;
        _numStuffClasses = _numClasses - things;
        return _isThingClass.ToArray();
    }

    /// <summary>
    /// Fuses semantic and instance pseudo labels into a panoptic pseudo label (Hahn et al. 2025, Sec. 3.3): each
    /// instance mask takes the most frequent pseudo class under it, and pixels of thing classes that no instance
    /// covers are ignored.
    /// </summary>
    /// <param name="semanticPseudoLabels">Pseudo class map <c>[H, W]</c> (integral ids).</param>
    /// <param name="instanceMasks">Binary instance pseudo masks <c>[M, H, W]</c>; where masks overlap, the first wins.</param>
    /// <returns>The semantic label map <c>[H, W]</c> (<c>-1</c> = ignore) and the instance id map <c>[H, W]</c>
    /// (<c>0</c> = no instance, otherwise the mask's 1-based index).</returns>
    public (Tensor<T> SemanticLabels, Tensor<T> InstanceIds) GeneratePanopticPseudoLabels(
        Tensor<T> semanticPseudoLabels, Tensor<T> instanceMasks)
    {
        var (labels, covered) = PseudoLabelPixels(semanticPseudoLabels, instanceMasks);
        int h = semanticPseudoLabels.Shape[0], w = semanticPseudoLabels.Shape[1], hw = h * w;
        int m = instanceMasks.Rank == 3 ? instanceMasks.Shape[0] : 0;

        // Ignore thing pixels that no instance covers.
        var thingPixel = Engine.TensorIndexSelect(Common.MaskClassificationOps.BoolTable<T>(_isThingClass), labels, 0).Cast<int>();
        var ignored = Engine.TensorMultiply(thingPixel, Engine.ScalarMinusTensor(1, covered));
        var semantic = labels;
        var instanceIds = new Tensor<int>(new[] { hw });
        if (m > 0)
        {
            var flatMasks = Engine.Reshape(Engine.TensorGreaterThan(instanceMasks, NumOps.FromDouble(0.5)), new[] { m, hw });
            // Majority class of every mask: argmax over K of the per-mask class histogram [M, K].
            var histogram = Engine.TensorMatMul(flatMasks, Engine.TensorOneHot<T>(labels, _numClasses));
            var maskClass = Engine.TensorArgMax(histogram, 1);                                    // [M]
            // The first covering mask of each pixel (argmax returns the first maximum).
            var owner = Engine.TensorArgMax(flatMasks, 0);                                        // [HW]
            var ownerClass = Engine.TensorIndexSelect(maskClass, owner, 0);
            semantic = Engine.TensorAdd(
                Engine.TensorMultiply(ownerClass, covered),
                Engine.TensorMultiply(labels, Engine.ScalarMinusTensor(1, covered)));
            instanceIds = Engine.TensorMultiply(Engine.TensorAddScalar(owner, 1), covered);
        }

        // -1 where ignored: semantic * (1 - ignored) - ignored.
        var withIgnore = Engine.TensorSubtract(Engine.TensorMultiply(semantic, Engine.ScalarMinusTensor(1, ignored)), ignored);
        return (Engine.Reshape(withIgnore.Cast<T>(), new[] { h, w }), Engine.Reshape(instanceIds.Cast<T>(), new[] { h, w }));
    }

    /// <summary>Flattened integer pseudo classes <c>[HW]</c> and whether an instance covers each pixel.</summary>
    private (Tensor<int> Labels, Tensor<int> Covered) PseudoLabelPixels(Tensor<T> semanticPseudoLabels, Tensor<T> instanceMasks)
    {
        if (semanticPseudoLabels is null) throw new ArgumentNullException(nameof(semanticPseudoLabels));
        if (instanceMasks is null) throw new ArgumentNullException(nameof(instanceMasks));
        if (semanticPseudoLabels.Rank != 2)
            throw new ArgumentException("Semantic pseudo labels must be [H, W].", nameof(semanticPseudoLabels));
        int h = semanticPseudoLabels.Shape[0], w = semanticPseudoLabels.Shape[1], hw = h * w;
        if (instanceMasks.Length != 0 && (instanceMasks.Rank != 3 || instanceMasks.Shape[1] != h || instanceMasks.Shape[2] != w))
            throw new ArgumentException($"Instance masks must be [M, {h}, {w}].", nameof(instanceMasks));
        var labels = Engine.Reshape(semanticPseudoLabels, new[] { hw }).Cast<int>();
        int[] range = Engine.TensorBinCount(labels, _numClasses).ToArray();
        if (range.Length > _numClasses)
            throw new ArgumentException($"Pseudo class ids must lie in [0, {_numClasses}).", nameof(semanticPseudoLabels));
        if (instanceMasks.Length == 0) return (labels, new Tensor<int>(new[] { hw }));
        var binary = Engine.TensorGreaterThan(instanceMasks, NumOps.FromDouble(0.5));
        var covered = Engine.Reshape(Engine.ReduceMax(binary, new[] { 0 }, false), new[] { hw }).Cast<int>();
        return (labels, covered);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// The semantic head's logits, bilinearly upsampled to the input, give the class map. The Cascade Mask R-CNN
    /// head's detections (cascade-averaged class scores, class-aware NMS at IoU 0.5, at most 100) of thing classes
    /// are the instance candidates; they are pasted by descending score while
    /// the score reaches <see cref="CUPSOptions.InstanceScoreThreshold"/> (0.4) and at most
    /// <see cref="CUPSOptions.OverlapThreshold"/> (0.5) of the candidate is already claimed, then every stuff
    /// class whose unclaimed area reaches <see cref="CUPSOptions.StuffAreaThreshold"/> (4096 input pixels)
    /// becomes one segment (Panoptic FPN's combine step, which CUPS's panoptic network uses). Unassigned pixels
    /// are void: <c>-1</c> in the semantic and panoptic maps.
    /// </para>
    /// </remarks>
    public override PanopticSegmentationResult<T> SegmentPanoptic(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var head = _instanceHead;
        if (!_useNativeMode || head is null)
            throw new NotSupportedException("CUPS panoptic inference needs the native Cascade Mask R-CNN head; ONNX mode exports the semantic head only.");
        var batched = image.Rank == 4 ? image : AddBatchDimension(image);
        int h = batched.Shape[2], w = batched.Shape[3], hw = h * w;
        Tensor<T> logits;
        List<Tensor<T>> taps;
        using (InferenceMode.Enter())
        {
            var (encoded, stages) = RunBackbone(batched);
            taps = stages;
            logits = Engine.Interpolate(Decode(encoded), new[] { h, w }, InterpolateMode.Bilinear, false);
        }

        int k = logits.Shape[1];
        var probabilities = Engine.Reshape(Engine.TensorSoftmax(logits, 1), new[] { k, hw });
        var semantic = Engine.TensorArgMax(probabilities, 0);                                     // [HW]

        Tensor<int> detectedMasks;
        int[] detectedClasses;
        Tensor<T> detectedScores;
        using (InferenceMode.Enter())
        {
            (detectedMasks, detectedClasses, detectedScores) = head.Detect(Fpn(taps), h, w);
        }

        // Only thing-class detections are instances; stuff comes from the semantic head.
        var keep = new List<int>();
        for (int i = 0; i < detectedClasses.Length; i++)
        {
            if (detectedClasses[i] >= 0 && detectedClasses[i] < k && _isThingClass[detectedClasses[i]]) keep.Add(i);
        }

        var index = new Tensor<int>(keep.ToArray(), new[] { keep.Count });
        var instanceMasks = keep.Count == 0 ? new Tensor<int>(new[] { 0, hw }) : Engine.TensorIndexSelect(detectedMasks, index, 0);
        var instanceScores = keep.Count == 0 ? new Tensor<T>(new[] { 0 }) : Engine.TensorIndexSelect(detectedScores, index, 0);
        var classes = keep.Select(i => detectedClasses[i]).ToArray();
        var settings = new Common.PanopticFpnSettings(
            _options.InstanceScoreThreshold, _options.OverlapThreshold, _options.StuffAreaThreshold);
        return Common.MaskClassificationOps.MergePanopticFpn(
            semantic, probabilities, instanceMasks, classes, instanceScores, _isThingClass, h, w, settings, 1.0);
    }
    #endregion
}
