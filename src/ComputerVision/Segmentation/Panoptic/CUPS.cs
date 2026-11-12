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
/// The network predicts semantic scores; <see cref="SegmentPanoptic"/> takes the connected regions of each thing
/// class as instance candidates and combines them with the stuff regions the way Panoptic FPN does.
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

    #endregion

    #region Private Methods
    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
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
    /// The semantic head's argmax gives the class map. Every connected region of a thing class is an instance
    /// candidate scored by the mean probability of its class; candidates are pasted by descending score while
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
        var logits = Common.SegmentationTensorOps.EnsureUnbatched(Predict(image));
        int k = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2], hw = h * w;
        var probabilities = Engine.Reshape(Engine.TensorSoftmax(logits, 0), new[] { k, hw });
        var semantic = Engine.TensorArgMax(probabilities, 0);                                     // [HW]
        var semanticMap = Engine.Reshape(semantic, new[] { h, w });

        var masks = new List<Tensor<int>>();
        var scores = new List<Tensor<T>>();
        var classes = new List<int>();
        for (int cls = 0; cls < k; cls++)
        {
            if (!_isThingClass[cls]) continue;
            var (componentMap, count) = Common.SegmentationTensorOps.LabelConnectedComponents(semanticMap, cls);
            if (count == 0) continue;
            var components = Engine.Reshape(componentMap, new[] { hw });
            // Rows 1..count of the one-hot component map are the instance masks.
            var oneHot = Engine.TensorPermute(Engine.TensorOneHot<int>(components, count + 1), new[] { 1, 0 });
            masks.Add(Engine.TensorSlice(oneHot, new[] { 1, 0 }, new[] { count, hw }));
            var classProbability = Engine.Reshape(Engine.TensorSlice(probabilities, new[] { cls, 0 }, new[] { 1, hw }), new[] { hw });
            var sums = Engine.TensorScatterAdd(new Tensor<T>(new[] { count + 1 }), components, classProbability, 0);
            var areas = Engine.TensorBinCount(components, count + 1).Cast<T>();
            scores.Add(Engine.TensorSlice(Engine.TensorDivide(sums, Engine.TensorClampMin(areas, NumOps.One)), new[] { 1 }, new[] { count }));
            for (int c = 0; c < count; c++) classes.Add(cls);
        }

        var instanceMasks = masks.Count == 0 ? new Tensor<int>(new[] { 0, hw }) : Engine.TensorConcatenate(masks.ToArray(), 0);
        var instanceScores = scores.Count == 0 ? new Tensor<T>(new[] { 0 }) : Engine.TensorConcatenate(scores.ToArray(), 0);
        int imageHeight = image.Shape[image.Rank - 2], imageWidth = image.Shape[image.Rank - 1];
        double pixelArea = (double)imageHeight * imageWidth / hw;
        var settings = new Common.PanopticFpnSettings(
            _options.InstanceScoreThreshold, _options.OverlapThreshold, _options.StuffAreaThreshold);
        return Common.MaskClassificationOps.MergePanopticFpn(
            semantic, probabilities, instanceMasks, classes.ToArray(), instanceScores, _isThingClass, h, w, settings, pixelArea);
    }
    #endregion
}
