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

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> InternImage is a semantic segmentation model that proves CNNs can compete
/// with Vision Transformers when using modern deformable convolutions. It uses DCNv3 (Deformable
/// Convolution v3) which can adaptively adjust where it "looks" in the image based on the content,
/// allowing it to focus on relevant regions for better segmentation.
///
/// Common use cases:
/// - Large-scale scene parsing (Cityscapes, ADE20K)
/// - Object detection and segmentation pipelines
/// - Foundation model applications requiring dense predictions
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - DCNv3 operator with multi-group deformable attention
/// - 4-stage hierarchical architecture (like ConvNeXt/Swin but with DCNv3)
/// - UPerNet decoder for multi-scale feature aggregation
/// - Scales from 30M (Tiny) to 1.08B (Huge) parameters
/// - Competitive with ViT-based models on ADE20K and COCO
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "InternImage: Exploring Large-Scale Vision Foundation Models
/// with Deformable Convolutions", CVPR 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an InternImage model for large-scale semantic segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 150);
/// var model = new InternImage&lt;double&gt;(architecture, numClasses: 150);
///
/// // Or load a pre-trained ONNX model for scene parsing
/// var onnxModel = new InternImage&lt;double&gt;(architecture, "internimage.onnx", numClasses: 150);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions", "https://arxiv.org/abs/2211.05778", Year = 2023, Authors = "Wang et al.")]
public partial class InternImage<T> : Common.SemanticSegmentationBase<T>
{
    private readonly InternImageOptions _options;

    /// <summary>
    /// Gets the configuration options for this InternImage model.
    /// </summary>
    /// <returns>The <see cref="InternImageOptions"/> used to configure this model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Options control model behavior including random seed for reproducibility.
    /// </para>
    /// </remarks>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    // Only InternImage's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from SemanticSegmentationBase -> SegmentationModelBase.
    private readonly InternImageModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;

    #endregion

    #region Properties

    // SupportsTraining and NumClasses are inherited from SegmentationModelBase and say exactly the
    // same thing, so re-declaring them here would only create two sources of one fact.
    internal bool UseNativeMode => _useNativeMode;
    internal InternImageModelSize ModelSize => _modelSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of InternImage in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration defining
    /// input dimensions (height, width, channels).</param>
    /// <param name="optimizer">The gradient-based optimizer (default: AdamW, as used in the
    /// InternImage paper for all experiments on ADE20K and COCO).</param>
    /// <param name="lossFunction">The loss function (default: CrossEntropyLoss for multi-class segmentation).</param>
    /// <param name="numClasses">Number of semantic classes (default: 150 for ADE20K).</param>
    /// <param name="modelSize">Model size variant (default: Tiny, 30M params).</param>
    /// <param name="dropRate">Dropout rate for regularization (default: 0.1).</param>
    /// <param name="options">Optional model options including random seed.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable InternImage from scratch. The model uses
    /// deformable convolutions that adaptively adjust their sampling positions based on
    /// image content, giving CNN-level efficiency with transformer-level accuracy.
    /// Start with Tiny for experiments, then scale to Base or larger for production.
    /// </para>
    /// </remarks>
    public InternImage(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        InternImageModelSize modelSize = InternImageModelSize.Tiny,
        double dropRate = 0.1,
        InternImageOptions? options = null)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new InternImageOptions();
        Options = _options;
        _modelSize = modelSize;
        _dropRate = dropRate;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        InitializeLayers();
    }

    /// <summary>
    /// Creates InternImage's paper-specified AdamW default when the caller supplies no optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the exact hyper-parameters the constructor used to inline
    /// (<c>optimizer ?? new AdamWOptimizer&lt;...&gt;(this, ...)</c>). They live here because a
    /// base-constructor argument cannot reference <c>this</c>; the base resolves this lazily after
    /// construction, which is what makes the default expressible at all.
    /// </para>
    /// </remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
        => new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 6e-5,
                WeightDecay = 0.05,
                EnableGradientClipping = true,
                MaxGradientNorm = 5.0,
            });

    /// <summary>
    /// Initializes a new instance of InternImage in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of semantic classes the ONNX model predicts (default: 150).</param>
    /// <param name="modelSize">Model size variant for metadata (default: Tiny).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained InternImage from an ONNX file for fast inference.
    /// ONNX mode does not support training. Use the native constructor for fine-tuning.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if ONNX path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if ONNX file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if ONNX runtime fails to load the model.</exception>
    public InternImage(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        InternImageModelSize modelSize = InternImageModelSize.Tiny,
        InternImageOptions? options = null)
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new InternImageOptions();
        Options = _options;
        _modelSize = modelSize;
        _dropRate = 0.0;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through InternImage to produce per-pixel segmentation logits.
    /// </summary>
    /// <param name="input">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel class logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass in an image and get back a map where each pixel has scores for
    /// every class. The DCNv3 encoder adaptively focuses on relevant regions, while the UPerNet
    /// decoder aggregates multi-scale features for accurate predictions.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return !_useNativeMode ? PredictOnnx(input) : Forward(input);
    }

    /// <summary>
    /// Performs one training step with forward pass, loss computation, backward pass, and update.
    /// </summary>
    /// <param name="input">Input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation map.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each training step processes an image through the DCNv3 encoder,
    /// compares the prediction to ground truth, and updates weights to improve future predictions.
    /// Only available in native mode.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown in ONNX mode.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException(
                "Training is not supported in ONNX mode. Use the native mode constructor for training.");

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

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(
        InternImageModelSize modelSize)
    {
        return modelSize switch
        {
            InternImageModelSize.Tiny => ([64, 128, 256, 512], [4, 4, 18, 4], 512),
            InternImageModelSize.Small => ([80, 160, 320, 640], [4, 4, 21, 4], 512),
            InternImageModelSize.Base => ([112, 224, 448, 896], [4, 4, 21, 4], 512),
            InternImageModelSize.XL => ([192, 384, 768, 1536], [4, 4, 21, 4], 1024),
            InternImageModelSize.Huge => ([320, 640, 1280, 2560], [6, 6, 32, 6], 1024),
            _ => ([64, 128, 256, 512], [4, 4, 18, 4], 512)
        };
    }

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
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

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

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the DCNv3 encoder and UPerNet decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, creates the 4-stage DCNv3 encoder that uses deformable
    /// convolutions to adaptively sample relevant image regions, followed by a UPerNet decoder that
    /// aggregates multi-scale features for segmentation. In ONNX mode, no layers are created.
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
            var encoderLayers = LayerHelper<T>.CreateInternImageEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();

            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int[] patchKernels = [7, 3, 3, 3];
            int[] patchStrides = [4, 2, 2, 2];
            int[] patchPaddings = [3, 1, 1, 1];
            int featureH = _height, featureW = _width;
            for (int stage = 0; stage < 4; stage++)
            {
                featureH = (featureH + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
                featureW = (featureW + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
            }

            Layers.AddRange(LayerHelper<T>.CreateInternImageDecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, featureH, featureW));
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Collects metadata describing this InternImage model.
    /// </summary>
    /// <returns>Model metadata with type, configuration, and serialized data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary of the model including its type, dimensions,
    /// class count, and serialized weights for saving or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "InternImage" },
                { "Description", "InternImage Semantic Segmentation (DCNv3)" },
                { "InputHeight", _height }, { "InputWidth", _width }, { "InputChannels", _channels },
                { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() },
                { "DecoderDim", _decoderDim }, { "DropRate", _dropRate },
                { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <summary>
    /// Serializes InternImage-specific configuration to a binary stream.
    /// </summary>
    /// <param name="writer">Binary writer for persistence.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves the model's configuration so it can be restored later.
    /// The order must match <see cref="DeserializeNetworkSpecificData"/>.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Deserializes InternImage-specific configuration from a binary stream.
    /// </summary>
    /// <param name="reader">Binary reader for loading.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reads back the saved configuration in the same order it was written.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Creates a new InternImage with the same config but fresh weights.
    /// </summary>
    /// <returns>A new model instance with reinitialized weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Used for cross-validation or ensemble training where multiple
    /// independent copies of the same architecture are needed.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return _useNativeMode
            ? new InternImage<T>(Architecture, null, LossFunction, _numClasses, _modelSize, _dropRate, _options)
            : new InternImage<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, _options);
    }

    #endregion
}
