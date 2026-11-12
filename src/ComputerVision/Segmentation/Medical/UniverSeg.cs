using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// UniverSeg: Universal Medical Image Segmentation via cross-attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Few-shot medical segmentation without fine-tuning. Cross-domain medical image segmentation.
///
/// Common use cases:
/// - Few-shot medical segmentation without fine-tuning
/// - Cross-domain medical image segmentation
/// - Label-efficient medical AI
/// - New task adaptation from a few examples
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - CrossBlock: each (query, support example) pair is convolved jointly and the query stream takes the
///   interaction averaged over the support set
/// - A U-Net of CrossBlocks (64 features at four encoder and three decoder levels, LeakyReLU)
/// - No fine-tuning needed for new segmentation tasks; set the support set with SetSupportSet
/// - Trained on diverse medical segmentation datasets
/// </para>
/// <para>
/// <b>Reference:</b> Butoi et al., "UniverSeg: Universal Medical Image Segmentation", ICCV 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a UniverSeg model for few-shot medical segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 128, inputWidth: 128, inputDepth: 1, outputSize: 1);
/// var model = new UniverSeg&lt;double&gt;(architecture);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new UniverSeg&lt;double&gt;(architecture,
///     options: new UniverSegOptions { NumClasses = 1, DropRate = 0 });
///
/// // Or load a pre-trained ONNX model for cross-domain medical segmentation
/// var onnxModel = new UniverSeg&lt;double&gt;(architecture, "universeg.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("UniverSeg: Universal Medical Image Segmentation", "https://arxiv.org/abs/2304.06131", Year = 2023, Authors = "Butoi et al.")]
public partial class UniverSeg<T> : Common.MedicalSegmentationBase<T>
{
    private readonly UniverSegOptions _options;

    #region Fields
    // Only UniverSeg's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from MedicalSegmentationBase -> SegmentationModelBase.
    private static readonly string[] ModalitiesSupported =
        ["CT", "MRI_T1", "Xray", "Ultrasound", "Dermoscopy", "Microscopy"];

    // Each CrossBlock owns three layers in Layers: the cross convolution over (target, support_i) pairs, then
    // the target and the support convolutions. The final 1x1 convolution follows the last decoder block.
    private const int LayersPerCrossBlock = 3;
    private readonly int[] _encoderFeatures;
    private readonly int[] _decoderFeatures;
    private bool _lazyShapesResolved;

    // The support set the network conditions on: images [S, C, H, W] and binary labels [S, 1, H, W].
    private Tensor<T>? _supportImages;
    private Tensor<T>? _supportLabels;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, SupportedModalities
    // and Supports2D are all supplied identically by the base.
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>UniverSeg operates on 2D slices only.</summary>
    public override bool Supports3D => false;

    /// <summary>UniverSeg is a few-shot model: its whole point is segmenting from a support set.</summary>
    public override bool SupportsFewShot => true;

    /// <inheritdoc />
    /// <remarks>
    /// The CrossBlock U-Net decodes back to the input resolution, so the output keeps the input's spatial size.
    /// </remarks>
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => SpatialStrideContract(inputRank, 1);
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes UniverSeg in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: segmentation loss for the class count).</param>
    /// <param name="options">Optional model options.</param>
    public UniverSeg(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        UniverSegOptions? options = null)
        // `optimizer` is passed straight through - INCLUDING null. The base resolves the default
        // AdamW lazily via CreateDefaultOptimizer(), which a base-constructor argument cannot do.
        : base(architecture, optimizer, lossFunction, (options ??= new UniverSegOptions()).NumClasses, ModalitiesSupported)
    {
        _options = options; Options = _options;
        // UniverSeg defaults to 128x128, not the base's 512x512, so the geometry fallback stays here.
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 128;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 128;
        (_encoderFeatures, _decoderFeatures) = ValidateFeatures(_options);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes UniverSeg in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the ONNX model. A graph with three inputs receives the target
    /// <c>[B, C, H, W]</c>, the support images <c>[B, S, C, H, W]</c> and the support labels
    /// <c>[B, S, 1, H, W]</c>, in that order (the official UniverSeg signature).</param>
    /// <param name="options">Optional model options.</param>
    public UniverSeg(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        UniverSegOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession - the same twenty lines this used to repeat.
        : base(architecture, onnxModelPath, (options ??= new UniverSegOptions()).NumClasses, ModalitiesSupported)
    {
        _options = options; Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 128;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 128;
        (_encoderFeatures, _decoderFeatures) = ValidateFeatures(_options);
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    /// <summary>
    /// Sets the support set every prediction is conditioned on.
    /// </summary>
    /// <param name="images">Support images <c>[S, C, H, W]</c> (or one image <c>[C, H, W]</c>), with the
    /// query's channel count.</param>
    /// <param name="labels">Binary support labels <c>[S, H, W]</c>, <c>[S, 1, H, W]</c> or <c>[H, W]</c>.</param>
    /// <remarks>
    /// <para>
    /// UniverSeg segments a query by comparing it with labelled examples of the task (Butoi et al. 2023).
    /// Supports are resampled to the query's size when the two differ. The support set is part of the model's
    /// state: it is saved and restored with the model. The paper defines the network only for a support set of at
    /// least one labelled pair, so predicting before a support set is given throws.
    /// </para>
    /// </remarks>
    public void SetSupportSet(Tensor<T> images, Tensor<T> labels)
    {
        var (supportImages, supportLabels) = NormalizeSupport(images, labels);
        _supportImages = supportImages;
        _supportLabels = supportLabels;
    }

    /// <summary>Removes the support set; predictions then throw until a new one is set.</summary>
    public void ClearSupportSet()
    {
        _supportImages = null;
        _supportLabels = null;
    }

    /// <summary>
    /// Runs the CrossBlock network on a query image conditioned on the current support set.
    /// </summary>
    /// <param name="input">The query <c>[C, H, W]</c> or <c>[B, C, H, W]</c>.</param>
    /// <returns>Logits <c>[NumClasses, H, W]</c> (or batched); a sigmoid gives the mask probability.</returns>
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

    #endregion

    #region Private Methods
    private static (int[] Encoder, int[] Decoder) ValidateFeatures(UniverSegOptions options)
    {
        var encoder = options.EncoderFeatures?.ToArray() ?? Array.Empty<int>();
        var decoder = options.DecoderFeatures?.ToArray() ?? Array.Empty<int>();
        if (encoder.Length < 1 || encoder.Any(f => f <= 0))
            throw new ArgumentException("UniverSeg needs at least one positive encoder CrossBlock width.", nameof(options));
        if (decoder.Length != encoder.Length - 1 || decoder.Any(f => f <= 0))
            throw new ArgumentException(
                $"UniverSeg needs {encoder.Length - 1} positive decoder CrossBlock widths, one per encoder skip.", nameof(options));
        return (encoder, decoder);
    }

    /// <summary>Normalizes a support set to images <c>[S, C, H, W]</c> and labels <c>[S, 1, H, W]</c>.</summary>
    private (Tensor<T> Images, Tensor<T> Labels) NormalizeSupport(Tensor<T> images, Tensor<T> labels)
    {
        if (images is null) throw new ArgumentNullException(nameof(images));
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        var supportImages = images.Rank == 3 ? AddBatchDimension(images) : images;
        if (supportImages.Rank != 4)
            throw new ArgumentException("Support images must be [S, C, H, W] or [C, H, W].", nameof(images));
        int s = supportImages.Shape[0], h = supportImages.Shape[2], w = supportImages.Shape[3];
        Tensor<T> supportLabels = labels.Rank switch
        {
            2 => Engine.Reshape(labels, new[] { 1, 1, labels.Shape[0], labels.Shape[1] }),
            3 => Engine.Reshape(labels, new[] { labels.Shape[0], 1, labels.Shape[1], labels.Shape[2] }),
            4 => labels,
            _ => throw new ArgumentException("Support labels must be [S, H, W], [S, 1, H, W] or [H, W].", nameof(labels)),
        };
        if (supportLabels.Shape[0] != s || supportLabels.Shape[1] != 1 || supportLabels.Shape[2] != h || supportLabels.Shape[3] != w)
            throw new ArgumentException($"Give one {h}x{w} label per support image ({s}).", nameof(labels));
        return (supportImages, supportLabels);
    }

    /// <summary>
    /// The support pairs for a query of size <paramref name="height"/> x <paramref name="width"/>: the current
    /// support set resampled to that size.
    /// </summary>
    /// <exception cref="InvalidOperationException">No support set has been given.</exception>
    /// <remarks>
    /// UniverSeg is <c>f(x_t, S)</c> with <c>S = {(x_i, y_i)}, i = 1..N</c> and the CrossBlock interaction averaged
    /// over the <c>N</c> pairs (Butoi et al. 2023, Sec. 3); an empty support set is not defined, so it is rejected
    /// rather than replaced by an invented pair.
    /// </remarks>
    private (Tensor<T> Images, Tensor<T> Labels) SupportFor(int channels, int height, int width)
    {
        if (_supportImages is null || _supportLabels is null)
            throw new InvalidOperationException(
                "UniverSeg segments a query by comparing it with a labelled support set, and the paper defines it only for " +
                "at least one support pair. Call SetSupportSet (or SegmentFewShot) before predicting or training.");
        if (_supportImages.Shape[1] != channels)
            throw new InvalidOperationException(
                $"The support images have {_supportImages.Shape[1]} channels but the query has {channels}.");
        if (_supportImages.Shape[2] == height && _supportImages.Shape[3] == width)
            return (_supportImages, _supportLabels);
        var size = new[] { height, width };
        return (Engine.Interpolate(_supportImages, size, InterpolateMode.Bilinear, false),
            Engine.Interpolate(_supportLabels, size, InterpolateMode.Bilinear, false));
    }

    /// <summary>
    /// UniverSeg's CrossBlock (Butoi et al. 2023, Sec. 3): every (target, support_i) pair is concatenated and
    /// convolved, <c>z_i = LReLU(Conv([u, v_i]))</c>; the target becomes <c>LReLU(Conv(mean_i z_i))</c>, the pairwise
    /// interaction averaged over the support set, and each support entry <c>LReLU(Conv(z_i))</c>.
    /// </summary>
    /// <param name="layer">Index of the block's cross convolution in <c>Layers</c>.</param>
    /// <param name="target">Target features <c>[B, Ct, H, W]</c>.</param>
    /// <param name="support">Support features <c>[B*S, Cs, H, W]</c> (support-major within each batch item).</param>
    /// <param name="batch">B.</param>
    /// <param name="supportSize">S.</param>
    private (Tensor<T> Target, Tensor<T> Support) CrossBlock(int layer, Tensor<T> target, Tensor<T> support, int batch, int supportSize)
    {
        int ct = target.Shape[1], h = target.Shape[2], w = target.Shape[3];
        var paired = Engine.Reshape(
            Engine.TensorTile(Engine.Reshape(target, new[] { batch, 1, ct, h, w }), new[] { 1, supportSize, 1, 1, 1 }),
            new[] { batch * supportSize, ct, h, w });
        var interaction = Layers[layer].Forward(Engine.TensorConcatenate(new[] { paired, support }, 1));        // [B*S, F, H, W]
        int f = interaction.Shape[1];
        var averaged = Engine.ReduceMean(Engine.Reshape(interaction, new[] { batch, supportSize, f, h, w }), new[] { 1 }, false);
        return (Layers[layer + 1].Forward(averaged), Layers[layer + 2].Forward(interaction));
    }

    /// <summary>
    /// The UniverSeg network: an encoder of CrossBlocks with 2x2 max pooling between levels, a decoder that
    /// upsamples (bilinear), concatenates the matching encoder outputs on both streams and applies a CrossBlock,
    /// and a 1x1 convolution on the target stream.
    /// </summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            var sequential = input;
            for (int i = 0; i < Layers.Count; i++) sequential = Layers[i].Forward(sequential);
            if (!hasBatch) sequential = RemoveBatchDimension(sequential); return sequential;
        }

        var (images, labels) = SupportFor(input.Shape[1], input.Shape[2], input.Shape[3]);
        var output = CrossBlockNetwork(input, images, labels);
        if (!hasBatch) output = RemoveBatchDimension(output); return output;
    }

    /// <summary>The CrossBlock U-Net on a batched query and its support pairs (images <c>[S, C, H, W]</c>, labels
    /// <c>[S, 1, H, W]</c>).</summary>
    private Tensor<T> CrossBlockNetwork(Tensor<T> input, Tensor<T> images, Tensor<T> labels)
    {
        int b = input.Shape[0], h = input.Shape[2], w = input.Shape[3];
        int s = images.Shape[0];
        var pairs = Engine.TensorConcatenate(new[] { images, labels }, 1);                                     // [S, C+1, H, W]
        int cs = pairs.Shape[1];
        var support = Engine.Reshape(
            Engine.TensorTile(Engine.Reshape(pairs, new[] { 1, s, cs, h, w }), new[] { b, 1, 1, 1, 1 }),
            new[] { b * s, cs, h, w });
        var target = input;

        var skips = new List<(Tensor<T> Target, Tensor<T> Support)>();
        int layer = 0;
        for (int level = 0; level < _encoderFeatures.Length; level++)
        {
            (target, support) = CrossBlock(layer, target, support, b, s);
            layer += LayersPerCrossBlock;
            if (level < _encoderFeatures.Length - 1)
            {
                skips.Add((target, support));
                target = Engine.MaxPool2D(target, 2, 2, 0);
                support = Engine.MaxPool2D(support, 2, 2, 0);
            }
        }

        for (int level = 0; level < _decoderFeatures.Length; level++)
        {
            var skip = skips[skips.Count - 1 - level];
            var size = new[] { skip.Target.Shape[2], skip.Target.Shape[3] };
            target = Engine.TensorConcatenate(new[] { Engine.Interpolate(target, size, InterpolateMode.Bilinear, false), skip.Target }, 1);
            support = Engine.TensorConcatenate(new[] { Engine.Interpolate(support, size, InterpolateMode.Bilinear, false), skip.Support }, 1);
            (target, support) = CrossBlock(layer, target, support, b, s);
            layer += LayersPerCrossBlock;
        }

        return Layers[layer].Forward(target);
    }

    /// <summary>
    /// Resolves the lazy convolutions' input channels with one real forward: the cross convolutions see the
    /// concatenated (target, support) channels, which a sequential shape walk cannot model.
    /// </summary>
    protected override void ResolveLazyLayerShapes()
    {
        if (_lazyShapesResolved) return;
        if (Layers is null || Layers.Count == 0) return;
        if (!_useNativeMode || (Architecture.Layers != null && Architecture.Layers.Count > 0))
        {
            _lazyShapesResolved = true;
            return;
        }

        // A shape probe, not a prediction: one zero support pair sizes the cross convolutions, whose input channels
        // do not depend on the support values or count.
        using (InferenceMode.Enter())
        {
            CrossBlockNetwork(new Tensor<T>(new[] { 1, _channels, _height, _width }),
                new Tensor<T>(new[] { 1, _channels, _height, _width }), new Tensor<T>(new[] { 1, 1, _height, _width }));
        }

        _lazyShapesResolved = true;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(
            _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images", ToOnnx(input, input._shape)) };
        if (_onnxSession.InputMetadata.Count >= 3)
        {
            int b = input.Shape[0], c = input.Shape[1], h = input.Shape[2], w = input.Shape[3];
            var (images, labels) = SupportFor(c, h, w);
            int s = images.Shape[0];
            var names = _onnxSession.InputMetadata.Keys.ToArray();
            var tiledImages = Engine.TensorTile(Engine.Reshape(images, new[] { 1, s, c, h, w }), new[] { b, 1, 1, 1, 1 });
            var tiledLabels = Engine.TensorTile(Engine.Reshape(labels, new[] { 1, s, 1, h, w }), new[] { b, 1, 1, 1, 1 });
            inputs.Add(NamedOnnxValue.CreateFromTensor(names[1], ToOnnx(tiledImages, tiledImages._shape)));
            inputs.Add(NamedOnnxValue.CreateFromTensor(names[2], ToOnnx(tiledLabels, tiledLabels._shape)));
        }

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        NumOps.FromFloatSpan(outputTensor.ToDenseTensor().Buffer.Span, outputData);
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result); return result;
    }

    private OnnxTensors.DenseTensor<float> ToOnnx(Tensor<T> tensor, int[] shape)
    {
        var data = new float[tensor.Length];
        NumOps.ToFloatSpan(tensor.Data.Span.Slice(0, tensor.Length), data);
        return new OnnxTensors.DenseTensor<float>(data, shape);
    }

    // AddBatchDimension / RemoveBatchDimension are inherited from SegmentationModelBase; the copies
    // that used to live here were line-for-line identical apart from the base's extra rank guards.
    #endregion

    #region Abstract Implementation
    /// <summary>
    /// Builds the CrossBlock layers (three per block, encoder then decoder) and the output convolution.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Architecture.Layers.Count / 2; }
        else
        {
            // Every convolution is 3x3 with LeakyReLU, as in the official CrossConv/ConvOp blocks.
            IActivationFunction<T> leaky = new LeakyReLUActivation<T>();
            IActivationFunction<T> identity = new IdentityActivation<T>();
            foreach (int features in _encoderFeatures.Concat(_decoderFeatures))
            {
                Layers.Add(new ConvolutionalLayer<T>(features, 3, 1, 1, leaky));
                Layers.Add(new ConvolutionalLayer<T>(features, 3, 1, 1, leaky));
                Layers.Add(new ConvolutionalLayer<T>(features, 3, 1, 1, leaky));
            }

            _encoderLayerEnd = _encoderFeatures.Length * LayersPerCrossBlock;
            Layers.Add(new ConvolutionalLayer<T>(_numClasses, 1, 1, 0, identity));
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Collects metadata describing this model's configuration.
    /// </summary>
    /// <returns>Model metadata.</returns>
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "UniverSeg" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelDataProvider = () => SerializeForMetadata()
    };

    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and flips _disposed,
    // and UniverSeg owns no other unmanaged resource.
    #endregion

    #region IMedicalSegmentation Implementation
    /// <summary>Segments a single 2D medical slice against the current support set.</summary>
    public override MedicalSegmentationResult<T> SegmentSlice(Tensor<T> slice)
    {
        var output = Common.SegmentationTensorOps.EnsureUnbatched(Predict(slice));
        var probs = Engine.TensorSoftmax(output, 0);
        var labels = Engine.TensorArgMax(output, 0).Cast<T>();
        var structures = Common.SegmentationStructureStatistics.Summarize(labels, probs);
        return new MedicalSegmentationResult<T> { Labels = labels, Probabilities = probs, Structures = structures };
    }

    /// <summary>Segments a volume by treating it as a single slice (UniverSeg is a 2D model).</summary>
    public override MedicalSegmentationResult<T> SegmentVolume(Tensor<T> volume) => SegmentSlice(volume);

    /// <summary>
    /// Few-shot segmentation: the network is run on <paramref name="queryImage"/> conditioned on the given
    /// support set (Butoi et al. 2023). The first output channel's sigmoid is the foreground probability.
    /// </summary>
    /// <returns>Labels <c>[H, W]</c> (1 where the probability is at least 0.5), probabilities
    /// <c>[2, H, W]</c> (background, foreground) and the foreground structure.</returns>
    protected override MedicalSegmentationResult<T> SegmentFewShotInternal(
        Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
    {
        var (images, labels) = NormalizeSupport(supportImages, supportMasks);
        var previousImages = _supportImages;
        var previousLabels = _supportLabels;
        Tensor<T> logits;
        try
        {
            _supportImages = images;
            _supportLabels = labels;
            logits = Common.SegmentationTensorOps.EnsureUnbatched(Predict(queryImage));
        }
        finally
        {
            _supportImages = previousImages;
            _supportLabels = previousLabels;
        }

        int h = logits.Shape[1], w = logits.Shape[2];
        var foreground = Engine.TensorSigmoid(Engine.TensorSlice(logits, new[] { 0, 0, 0 }, new[] { 1, h, w }));
        var mask = Engine.TensorGreaterEqual(foreground, Tensor<T>.CreateDefault(new[] { 1, h, w }, NumOps.FromDouble(0.5)));
        var probs = Engine.TensorConcatenate(new[] { Engine.ScalarMinusTensor(NumOps.One, foreground), foreground }, 0);
        var structures = new List<SegmentedStructure>();
        int area = Engine.TensorSum(mask.Cast<int>());
        if (area > 0)
        {
            double confidence = Engine.TensorDivide(
                Engine.ReduceSum(Engine.Reshape(Engine.TensorMultiply(foreground, mask), new[] { h * w }), new[] { 0 }, true),
                Tensor<T>.CreateDefault(new[] { 1 }, NumOps.FromDouble(area))).Cast<double>().ToArray()[0];
            structures.Add(new SegmentedStructure { ClassId = 1, Name = "FewShot_Target", VolumeOrArea = area, MeanConfidence = confidence });
        }

        return new MedicalSegmentationResult<T>
        {
            Labels = Engine.Reshape(mask, new[] { h, w }), Probabilities = probs, Structures = structures,
        };
    }
    #endregion
}
