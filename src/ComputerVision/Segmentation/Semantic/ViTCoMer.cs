using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ViT-CoMer is a hybrid model that combines a CNN branch with a Vision
/// Transformer branch, getting the best of both worlds. CNNs excel at capturing fine local details
/// (edges, textures), while transformers capture global context (relationships between distant objects).
/// By fusing them, ViT-CoMer produces segmentation maps with excellent boundary quality.
///
/// Common use cases:
/// - High-precision boundary segmentation (medical imaging, industrial inspection)
/// - Scene understanding where both local and global context matter
/// - Applications where ViTs alone miss fine details at object boundaries
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Parallel CNN and transformer branches with cross-branch feature interaction
/// - CNN branch provides multi-scale local features at each ViT stage
/// - Bidirectional feature interaction module fuses CNN and transformer features
/// - Improved boundary quality over pure ViT or pure CNN approaches
/// </para>
/// <para>
/// <b>Reference:</b> Xia et al., "ViT-CoMer: Vision Transformer with Convolutional Multi-scale
/// Feature Interaction for Dense Predictions", CVPR 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a ViT-CoMer hybrid CNN-Transformer model for high-quality segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 150);
/// var model = new ViTCoMer&lt;double&gt;(architecture, numClasses: 150);
///
/// // Or load a pre-trained ONNX model for inference
/// var onnxModel = new ViTCoMer&lt;double&gt;(architecture, "vitcomer.onnx", numClasses: 150);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions", "https://arxiv.org/abs/2403.07392", Year = 2024, Authors = "Xia et al.")]
public partial class ViTCoMer<T> : Common.SemanticSegmentationBase<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Downsamples by 4, not the family's 32 - measured: [1,3,64,64] returns [1,C,16,16]. Its parallel
    /// CNN branch keeps far more spatial resolution than a plain /32 backbone.
    /// </remarks>
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => SpatialStrideContract(inputRank, 4);

    private readonly ViTCoMerOptions _options;

    /// <summary>
    /// Gets the configuration options for this ViT-CoMer model.
    /// </summary>
    /// <returns>The <see cref="ViTCoMerOptions"/> for this model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Options control model behavior including random seed for reproducibility.
    /// </para>
    /// </remarks>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    // Only ViT-CoMer's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from SemanticSegmentationBase -> SegmentationModelBase.
    private readonly ViTCoMerModelSize _modelSize;
    private readonly int _embedDim;
    private readonly int[] _cnnChannels;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;

    #endregion

    #region Properties

    // SupportsTraining and NumClasses are inherited from SegmentationModelBase and say exactly the
    // same thing, so re-declaring them here would only create two sources of one fact.
    internal bool UseNativeMode => _useNativeMode;
    internal ViTCoMerModelSize ModelSize => _modelSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of ViT-CoMer in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW, as used in the paper).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of semantic classes (default: 150 for ADE20K).</param>
    /// <param name="modelSize">Model size variant (default: Small).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable ViT-CoMer model that runs CNN and transformer
    /// branches in parallel. The CNN branch captures local details while the transformer captures
    /// global context, and they exchange information through cross-branch interaction modules.
    /// </para>
    /// </remarks>
    public ViTCoMer(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        ViTCoMerModelSize modelSize = ViTCoMerModelSize.Small,
        double dropRate = 0.1,
        ViTCoMerOptions? options = null)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new ViTCoMerOptions();
        Options = _options;
        _modelSize = modelSize;
        _dropRate = dropRate;

        (_embedDim, _cnnChannels, _depths, _decoderDim) = GetModelConfig(modelSize);

        InitializeLayers();
    }

    /// <summary>
    /// Creates ViT-CoMer's AdamW default (paper settings, learning rate from options) when the
    /// caller supplies no optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the exact hyper-parameters the constructor used to inline
    /// (<c>optimizer ?? new AdamWOptimizer&lt;...&gt;(this, ...)</c>). They live here because a
    /// base-constructor argument cannot reference <c>this</c>; the base resolves this lazily after
    /// construction, by which point <c>_options</c> is assigned.
    /// </para>
    /// </remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
        => new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                UseAdaptiveLearningRate = false
            });

    /// <summary>
    /// Initializes a new instance of ViT-CoMer in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of classes (default: 150).</param>
    /// <param name="modelSize">Model size for metadata (default: Small).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained ViT-CoMer for fast inference. Does not support training.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if file not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if ONNX load fails.</exception>
    public ViTCoMer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        ViTCoMerModelSize modelSize = ViTCoMerModelSize.Small,
        ViTCoMerOptions? options = null)
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new ViTCoMerOptions();
        Options = _options;
        _modelSize = modelSize;
        _dropRate = 0.0;

        (_embedDim, _cnnChannels, _depths, _decoderDim) = GetModelConfig(modelSize);

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through the hybrid CNN-transformer model.
    /// </summary>
    /// <param name="input">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel class logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The image is processed through parallel CNN and transformer branches
    /// that exchange information, producing segmentation maps with excellent boundary quality.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return !_useNativeMode ? PredictOnnx(input) : Forward(input);
    }

    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">Input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation map.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trains both the CNN and transformer branches simultaneously.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown in ONNX mode.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

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

    private static (int EmbedDim, int[] CnnChannels, int[] Depths, int DecoderDim) GetModelConfig(
        ViTCoMerModelSize modelSize)
    {
        return modelSize switch
        {
            ViTCoMerModelSize.Small => (384, [64, 128, 320, 512], [2, 2, 6, 2], 256),
            ViTCoMerModelSize.Base => (768, [64, 128, 320, 512], [2, 2, 6, 2], 512),
            ViTCoMerModelSize.Large => (1024, [96, 192, 384, 768], [2, 2, 6, 2], 768),
            _ => (384, [64, 128, 320, 512], [2, 2, 6, 2], 256)
        };
    }

    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features);
        return features;
    }

    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "pixel_values";
        using var results = _onnxSession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) });
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
    }

    /// <summary>Adds a leading batch axis through a RECORDED reshape, so it stays on the autodiff tape.</summary>
    /// <remarks>
    /// <para>
    /// These deliberately SHADOW <c>SegmentationModelBase</c>'s versions rather than inheriting them, and
    /// that is load-bearing. The base copies raw spans into a freshly allocated tensor, which detaches the
    /// result from the tape; <c>Engine.Reshape</c> records the operation instead. This model trains -
    /// <see cref="Train"/> calls <c>TrainWithTape</c> - and <see cref="Forward"/> routes unbatched input
    /// through both helpers, so inheriting the copying versions would break the gradient chain and train
    /// nothing. VideoLISA and EfficientTAM carry the same shadowing for the same measured reason
    /// (zero-gradient failures in GradientFlow_ShouldBeNonZeroAndFinite).
    /// </para>
    /// </remarks>
    private new Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        return Engine.Reshape(tensor, [1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]);
    }

    /// <summary>Drops the leading batch axis through a RECORDED reshape. See <see cref="AddBatchDimension"/>.</summary>
    private new Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++) newShape[i] = tensor.Shape[i + 1];
        return Engine.Reshape(tensor, newShape);
    }

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the hybrid CNN-transformer encoder and decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates parallel CNN and transformer processing stages with
    /// cross-branch feature interaction, followed by a decoder for classification. ONNX mode
    /// skips layer creation.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = _options.EncoderLayerCount ?? Architecture.Layers.Count;
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateViTCoMerLayers(
                _channels, _height, _width, _embedDim, _cnnChannels, _depths,
                _decoderDim, _numClasses, _dropRate));
            _encoderLayerEnd = Layers.Count;
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Collects model metadata.
    /// </summary>
    /// <returns>Model metadata.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Summary of the model for saving, comparing, or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "ViTCoMer" }, { "Description", "ViT-CoMer Hybrid CNN-Transformer Segmentation" },
                { "InputHeight", _height }, { "InputWidth", _width }, { "InputChannels", _channels },
                { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() },
                { "EmbedDim", _embedDim }, { "DecoderDim", _decoderDim }, { "DropRate", _dropRate },
                { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <summary>
    /// Serializes configuration for persistence.
    /// </summary>
    /// <param name="writer">Binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves config so the model can be restored later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height); writer.Write(_width); writer.Write(_channels);
        writer.Write(_numClasses); writer.Write((int)_modelSize);
        writer.Write(_embedDim); writer.Write(_decoderDim); writer.Write(_dropRate);
        writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
        writer.Write(_options.LearningRate);
        writer.Write(_cnnChannels.Length);
        foreach (int c in _cnnChannels) writer.Write(c);
        writer.Write(_depths.Length);
        foreach (int d in _depths) writer.Write(d);
    }

    /// <summary>
    /// Deserializes configuration.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reads saved configuration matching the write order.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadDouble();
        _ = reader.ReadBoolean(); _ = reader.ReadString();
        _ = reader.ReadInt32();
        _ = reader.ReadDouble();
        int cc = reader.ReadInt32(); for (int i = 0; i < cc; i++) _ = reader.ReadInt32();
        int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new ViT-CoMer with same config but fresh weights.
    /// </summary>
    /// <returns>New model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Used for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return _useNativeMode
            ? new ViTCoMer<T>(Architecture, optimizer: null, LossFunction, _numClasses, _modelSize, _dropRate, new ViTCoMerOptions(_options))
            : new ViTCoMer<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, new ViTCoMerOptions(_options));
    }

    #endregion
}
