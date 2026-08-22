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

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// EoMT: Encoder-only Mask Transformer for universal image segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> EoMT dramatically simplifies segmentation by removing the complex pixel
/// decoder and transformer decoder used by models like Mask2Former. Instead, mask queries are
/// inserted directly into a plain Vision Transformer (DINOv2), making EoMT 4.4x faster while
/// maintaining competitive accuracy. Think of it as the "minimalist" approach to segmentation.
///
/// Common use cases:
/// - Real-time panoptic segmentation
/// - Latency-sensitive deployment (4.4x faster than Mask2Former)
/// - Research into simpler segmentation architectures
/// - Any scenario where speed matters more than peak accuracy
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Uses DINOv2 as frozen backbone (ViT-S/B/L)
/// - Queries inserted at intermediate ViT layers, processed alongside image tokens
/// - No separate pixel decoder or transformer decoder needed
/// - Query-to-mask via dot product with intermediate ViT features
/// - 4.4x faster than Mask2Former-Swin-L with competitive results
/// </para>
/// <para>
/// <b>Reference:</b> Saporta et al., "Encoder-only Mask Transformer", CVPR 2025 Highlight.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an EoMT model for fast panoptic segmentation (4.4x faster than Mask2Former)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 150);
/// var model = new EoMT&lt;double&gt;(architecture, numClasses: 150);
///
/// // Or load a pre-trained ONNX model for encoder-only segmentation
/// var onnxModel = new EoMT&lt;double&gt;(architecture, "eomt.onnx", numClasses: 150);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Your ViT is Secretly an Image Segmentation Model", "https://arxiv.org/abs/2503.19108", Year = 2025, Authors = "Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, Daan de Geus")]
public partial class EoMT<T> : Common.PanopticSegmentationBase<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Downsamples by 16, not the family's 32 - measured: [1,3,64,64] returns [1,C,4,4].
    /// </remarks>
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => SpatialStrideContract(inputRank, 16);

    private readonly EoMTOptions _options;

    /// <summary>
    /// Gets the configuration options for this EoMT model.
    /// </summary>
    /// <returns>The <see cref="EoMTOptions"/> for this model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Options control model behavior including random seed for reproducibility.
    /// </para>
    /// </remarks>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    // Only EoMT's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from PanopticSegmentationBase -> SegmentationModelBase.
    private readonly int _numQueries;
    private readonly EoMTModelSize _modelSize;
    private readonly int _embedDim;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether using native mode (trainable) or ONNX mode (inference only).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;
    internal EoMTModelSize ModelSize => _modelSize;

    #endregion

    /// <summary>
    /// EoMT's stuff/thing split: the first third of the class list is "stuff" (amorphous regions
    /// like sky or road), the rest are countable "things". Kept as a static helper so both
    /// constructors can hand the same split to the panoptic base before any field is assigned.
    /// </summary>
    private static int StuffClassCount(int numClasses) => Math.Max(1, numClasses / 3);

    #region Constructors

    /// <summary>
    /// Initializes EoMT in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW as in the paper).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of output classes (default: 150 for ADE20K).</param>
    /// <param name="numQueries">Number of mask queries inserted into ViT (default: 100).</param>
    /// <param name="modelSize">DINOv2 backbone size (default: Base).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable EoMT. The queries are injected directly into the
    /// ViT layers — no separate decoder architecture needed, which is why EoMT is so much faster.
    /// </para>
    /// </remarks>
    public EoMT(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        int numQueries = 100,
        EoMTModelSize modelSize = EoMTModelSize.Base,
        double dropRate = 0.1,
        EoMTOptions? options = null)
        : base(architecture, optimizer, lossFunction, numClasses,
               StuffClassCount(numClasses), numClasses - StuffClassCount(numClasses))
    {
        _options = options ?? new EoMTOptions();
        Options = _options;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _dropRate = dropRate;

        (_embedDim, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes EoMT in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of output classes (default: 150).</param>
    /// <param name="numQueries">Number of mask queries (default: 100).</param>
    /// <param name="modelSize">DINOv2 backbone size for metadata (default: Base).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained EoMT from ONNX for fast inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public EoMT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        int numQueries = 100,
        EoMTModelSize modelSize = EoMTModelSize.Base,
        EoMTOptions? options = null)
        : base(architecture, onnxModelPath, numClasses,
               StuffClassCount(numClasses), numClasses - StuffClassCount(numClasses))
    {
        _options = options ?? new EoMTOptions();
        Options = _options;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _dropRate = 0.0;

        (_embedDim, _depths, _decoderDim) = GetModelConfig(modelSize);

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    // PredictCore's mode dispatch (ONNX -> PredictOnnx, native -> Forward) is inherited from
    // SegmentationModelBase; both branches are overridden below.

    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trains EoMT by comparing predictions to ground truth. Only native mode.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException(
                "Training is not supported in ONNX mode. Use the native mode constructor for training.");

        if (input.Shape.Length == 3) input = AddBatchDimension(input);
        if (expectedOutput.Shape.Length == 3) expectedOutput = AddBatchDimension(expectedOutput);
        if (input.Shape.Length != 4) throw new ArgumentException($"Tape-based training requires rank 3 (CHW) or rank 4 (NCHW), got rank {input.Shape.Length}.", nameof(input));
        if (expectedOutput.Shape.Length != 4) throw new ArgumentException($"Tape-based training target requires rank 3 (CHW) or rank 4 (NCHW), got rank {expectedOutput.Shape.Length}.", nameof(expectedOutput));
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, Optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    #endregion

    #region Private Methods

    private static (int EmbedDim, int[] Depths, int DecoderDim) GetModelConfig(EoMTModelSize modelSize)
    {
        return modelSize switch
        {
            EoMTModelSize.Small => (384, [12, 0, 0, 0], 256),
            EoMTModelSize.Base => (768, [12, 0, 0, 0], 256),
            EoMTModelSize.Large => (1024, [24, 0, 0, 0], 256),
            _ => (768, [12, 0, 0, 0], 256)
        };
    }

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            features = Layers[i].Forward(features);

        if (!hasBatch) features = RemoveBatchDimension(features);
        return features;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "pixel_values";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        var result = new Tensor<T>(outputShape, new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
    }

    // AddBatchDimension / RemoveBatchDimension are inherited from SegmentationModelBase.

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the encoder-only layers for EoMT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the DINOv2 ViT backbone with embedded mask queries.
    /// In ONNX mode, no layers are created.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Architecture.Layers.Count / 2;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateEoMTEncoderLayers(
                _channels, _height, _width, _embedDim, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int featureH = _height / 16;
            int featureW = _width / 16;
            var decoderLayers = LayerHelper<T>.CreateEoMTDecoderLayers(
                _embedDim, _decoderDim, _numClasses, featureH, featureW);
            Layers.AddRange(decoderLayers);
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Collects metadata describing this EoMT model's configuration.
    /// </summary>
    /// <returns>Model metadata.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary for saving or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "EoMT" }, { "InputHeight", _height }, { "InputWidth", _width },
                { "InputChannels", _channels }, { "NumClasses", _numClasses },
                { "NumQueries", _numQueries }, { "ModelSize", _modelSize.ToString() },
                { "EmbedDim", _embedDim }, { "DecoderDim", _decoderDim },
                { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <summary>
    /// Writes EoMT configuration to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves model configuration for later reconstruction.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Reads EoMT configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads model configuration when restoring a saved model.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Creates a new EoMT instance with the same configuration but fresh weights.
    /// </summary>
    /// <returns>A new <see cref="EoMT{T}"/> model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a copy for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return _useNativeMode
            ? new EoMT<T>(Architecture, _optimizer, LossFunction, _numClasses, _numQueries, _modelSize, _dropRate, _options)
            : new EoMT<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _numQueries, _modelSize, _options);
    }

    /// <summary>
    // Dispose of the ONNX session and the _disposed latch are handled by SegmentationModelBase.

    #endregion

    #region IPanopticSegmentation Implementation

    // NumClasses / InputHeight / InputWidth / IsOnnxMode / Segment / NumStuffClasses /
    // NumThingClasses all arrive from PanopticSegmentationBase.

    /// <inheritdoc />
    public override PanopticSegmentationResult<T> SegmentPanoptic(Tensor<T> image)
    {
        var logits = Common.SegmentationTensorOps.EnsureUnbatched(Predict(image));
        var probMap = Common.SegmentationTensorOps.SoftmaxAlongClassDim(logits);
        var semanticMap = Common.SegmentationTensorOps.ArgmaxAlongClassDim(logits);
        int h = semanticMap.Shape[0], w = semanticMap.Shape[1];
        int numStuff = NumStuffClasses;
        var instanceMap = new Tensor<T>([h, w]);
        var panopticMap = new Tensor<T>([h, w]);
        var segments = new List<PanopticSegment<T>>();
        int nextInstId = 1;
        for (int cls = 0; cls < numStuff; cls++)
        {
            int area = 0; double sumConf = 0;
            for (int row = 0; row < h; row++)
                for (int col = 0; col < w; col++)
                    if (NumOps.Compare(semanticMap[row, col], NumOps.FromDouble(cls)) == 0)
                    { panopticMap[row, col] = NumOps.FromDouble(cls * 1000); area++; sumConf += NumOps.ToDouble(probMap[cls, row, col]); }
            if (area > 0) segments.Add(new PanopticSegment<T> { SegmentId = cls, ClassId = cls, IsThing = false, Confidence = sumConf / area, Area = area });
        }
        for (int cls = numStuff; cls < _numClasses; cls++)
        {
            var (labelMap, count) = Common.SegmentationTensorOps.LabelConnectedComponents(semanticMap, cls);
            for (int comp = 1; comp <= count; comp++)
            {
                int instId = nextInstId++;
                int area = 0; double sumConf = 0; var compMask = new Tensor<T>([h, w]);
                for (int row = 0; row < h; row++)
                    for (int col = 0; col < w; col++)
                        if (NumOps.Compare(labelMap[row, col], NumOps.FromDouble(comp)) == 0)
                        { instanceMap[row, col] = NumOps.FromDouble(instId); panopticMap[row, col] = NumOps.FromDouble(cls * 1000 + instId); compMask[row, col] = NumOps.FromDouble(1.0); area++; sumConf += NumOps.ToDouble(probMap[cls, row, col]); }
                if (area > 0) segments.Add(new PanopticSegment<T> { SegmentId = instId, ClassId = cls, IsThing = true, Confidence = sumConf / area, Area = area, Mask = compMask });
            }
        }
        return new PanopticSegmentationResult<T> { SemanticMap = semanticMap, InstanceMap = instanceMap, PanopticMap = panopticMap, Segments = segments };
    }

    #endregion
}
