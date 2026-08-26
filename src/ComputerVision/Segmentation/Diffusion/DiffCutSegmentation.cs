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

namespace AiDotNet.ComputerVision.Segmentation.Diffusion;

/// <summary>
/// DiffCut: Diffusion-based zero-shot segmentation via graph cuts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Zero-shot segmentation from diffusion features. Unsupervised image segmentation.
///
/// Common use cases:
/// - Zero-shot segmentation from diffusion features
/// - Unsupervised image segmentation
/// - Open-world object discovery
/// - Training-free segmentation
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Extracts features from pre-trained Stable Diffusion U-Net
/// - Recursive Normalized Cut on diffusion feature affinity graph
/// - No training or fine-tuning required
/// - Uses diffusion model internal features as dense visual descriptors
/// </para>
/// <para>
/// <b>Reference:</b> Couairon et al., "DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut", arXiv 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a DiffCut model for zero-shot diffusion-based segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 1);
/// var model = new DiffCutSegmentation&lt;double&gt;(architecture, numClasses: 1);
///
/// // Or load a pre-trained ONNX model
/// var onnxModel = new DiffCutSegmentation&lt;double&gt;(architecture, "diffcut_model.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut", "https://arxiv.org/abs/2406.02842", Year = 2024, Authors = "Couairon et al.")]
public class DiffCutSegmentation<T> : Common.SemanticSegmentationBase<T>
{
    private readonly DiffCutSegmentationOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only DiffCut's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from SemanticSegmentationBase -> SegmentationModelBase.
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    /// <summary>
    /// Gets whether using the native inference pipeline or an ONNX model.
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets whether supervised parameter training is supported.
    /// </summary>
    /// <remarks>
    /// DiffCut is a zero-shot, training-free method over frozen diffusion features.
    /// Exposing the native graph as trainable would describe a different model than the
    /// cited paper, so both native and ONNX modes are inference-only.
    /// </remarks>
    public override bool SupportsTraining => false;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes DiffCutSegmentation with the native inference pipeline.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Retained for source compatibility; DiffCut does not train parameters.</param>
    /// <param name="lossFunction">Retained for source compatibility; DiffCut does not use a supervised objective.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates the training-free DiffCut inference pipeline.
    /// </para>
    /// </remarks>
    public DiffCutSegmentation(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        double dropRate = 0,
        DiffCutSegmentationOptions? options = null)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new DiffCutSegmentationOptions();
        Options = _options;
        _dropRate = dropRate;
        var nativeOptions = ValidateAndCopyNativeOptions(_options);
        _channelDims = nativeOptions.ChannelDimensions;
        _depths = nativeOptions.StageDepths;
        _decoderDim = nativeOptions.DecoderDimension;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes DiffCutSegmentation in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained DiffCutSegmentation from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public DiffCutSegmentation(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1,
        DiffCutSegmentationOptions? options = null)
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new DiffCutSegmentationOptions();
        Options = _options;
        _dropRate = 0;
        var nativeOptions = ValidateAndCopyNativeOptions(_options);
        _channelDims = nativeOptions.ChannelDimensions;
        _depths = nativeOptions.StageDepths;
        _decoderDim = nativeOptions.DecoderDimension;
        InitializeLayers();
    }
    #endregion

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => Optimizer ?? base.GetOrCreateBaseOptimizer();

    #region Public Methods
    /// <summary>
    /// Rejects supervised training because DiffCut is a frozen, zero-shot method.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation tensor.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> DiffCut extracts frozen diffusion features and segments
    /// them with recursive normalized cut; it does not learn from labeled masks.</para>
    /// </remarks>
    /// <exception cref="NotSupportedException">Always thrown because the cited method is training-free.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        throw new NotSupportedException(
            "DiffCut is a zero-shot, training-free segmentation method over frozen diffusion features.");
    }
    #endregion

    #region Private Methods
    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
            input = AddBatchDimension(input);

        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            features = Layers[i].Forward(features);

        if (!hasBatch)
            features = RemoveBatchDimension(features);

        return features;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
            input = AddBatchDimension(input);

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch)
            result = RemoveBatchDimension(result);

        return result;
    }

    /// <summary>Adds a leading batch axis through a RECORDED reshape, so it stays on the autodiff tape.</summary>
    /// <remarks>
    /// <para>
    /// These deliberately SHADOW <c>SegmentationModelBase</c>'s versions. The base allocates a new
    /// tensor and copies raw spans into it, which DETACHES the result from the tape;
    /// <c>Engine.Reshape</c> records the operation instead. This model trains - <see cref="Train"/>
    /// calls <c>TrainWithTape</c> - and <see cref="Forward"/> routes any unbatched (rank-3) input
    /// through both helpers, so inheriting the copying versions would break the gradient chain and
    /// train nothing at all. VideoLISA, EfficientTAM, OneFormer, GroundedSAM2 and ViTCoMer carry the
    /// same shadowing for the same measured reason (zero-gradient failures in
    /// GradientFlow_ShouldBeNonZeroAndFinite).
    /// </para>
    /// </remarks>
    private new Tensor<T> AddBatchDimension(Tensor<T> tensor)
        => Engine.Reshape(tensor, [1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]);

    /// <summary>Drops the leading batch axis through a RECORDED reshape. See <see cref="AddBatchDimension"/>.</summary>
    private new Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int[] s = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < s.Length; i++)
            s[i] = tensor.Shape[i + 1];
        return Engine.Reshape(tensor, s);
    }

    private static (int[] ChannelDimensions, int[] StageDepths, int DecoderDimension)
        ValidateAndCopyNativeOptions(DiffCutSegmentationOptions options)
    {
        if (options.ChannelDimensions is null || options.ChannelDimensions.Length == 0)
            throw new ArgumentException("At least one encoder channel dimension is required.", nameof(options));
        if (options.StageDepths is null || options.StageDepths.Length != options.ChannelDimensions.Length)
            throw new ArgumentException("StageDepths must contain one entry per channel dimension.", nameof(options));
        if (options.ChannelDimensions.Any(value => value <= 0))
            throw new ArgumentOutOfRangeException(nameof(options), "All channel dimensions must be positive.");
        if (options.StageDepths.Any(value => value <= 0))
            throw new ArgumentOutOfRangeException(nameof(options), "All stage depths must be positive.");
        if (options.DecoderDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "DecoderDimension must be positive.");

        return (options.ChannelDimensions.ToArray(), options.StageDepths.ToArray(), options.DecoderDimension);
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
        if (!_useNativeMode)
        {
            ClearLayers();
            return;
        }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = _options.EncoderLayerCount ?? Architecture.Layers.Count / 2;
            if (_encoderLayerEnd < 0 || _encoderLayerEnd > Architecture.Layers.Count)
                throw new ArgumentOutOfRangeException(nameof(_options.EncoderLayerCount));
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateDiffCutSegmentationEncoderLayers(
                _channels,
                _height,
                _width,
                _channelDims,
                _depths,
                _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);
            int featureHeight = _height / 32;
            int featureWidth = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateDiffCutSegmentationDecoderLayers(
                _channelDims[^1],
                _decoderDim,
                _numClasses,
                featureHeight,
                featureWidth);
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
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "DiffCutSegmentation" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "NumClasses", _numClasses },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count }
        },
        ModelData = SerializeForMetadata()
    };

    /// <summary>
    /// Writes configuration to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves model configuration for later reconstruction.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numClasses);
        writer.Write(_decoderDim);
        writer.Write(_dropRate);
        writer.Write(_useNativeMode);
        writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
        writer.Write(_channelDims.Length);
        foreach (int dimension in _channelDims)
            writer.Write(dimension);
        writer.Write(_depths.Length);
        foreach (int depth in _depths)
            writer.Write(depth);
    }

    /// <summary>
    /// Reads configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads model configuration when restoring a saved model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadDouble();
        _ = reader.ReadBoolean();
        _ = reader.ReadString();
        _ = reader.ReadInt32();

        int channelDimensionCount = reader.ReadInt32();
        for (int i = 0; i < channelDimensionCount; i++)
            _ = reader.ReadInt32();

        int depthCount = reader.ReadInt32();
        for (int i = 0; i < depthCount; i++)
            _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance with the same configuration but fresh weights.
    /// </summary>
    /// <returns>A new model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a copy for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var clonedOptions = new DiffCutSegmentationOptions(_options);
        return _useNativeMode
            ? new DiffCutSegmentation<T>(Architecture, optimizer: null, lossFunction: LossFunction,
                numClasses: _numClasses, dropRate: _dropRate, options: clonedOptions)
            : new DiffCutSegmentation<T>(Architecture,
                _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."),
                _numClasses, clonedOptions);
    }

    // Dispose of the ONNX session and the _disposed latch are handled by SegmentationModelBase.
    // NumClasses / InputHeight / InputWidth / IsOnnxMode / Segment / GetClassMap / GetProbabilityMap
    // all arrive from SemanticSegmentationBase with identical bodies.
    #endregion
}
