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
/// MedNeXt: Transformer-driven scaling of ConvNets for medical segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Medical image segmentation with efficient ConvNet design. CT and MRI organ segmentation.
///
/// Common use cases:
/// - Medical image segmentation with efficient ConvNet design
/// - CT and MRI organ segmentation
/// - 3D medical volume analysis
/// - Resource-efficient medical AI deployment
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - ConvNeXt-inspired blocks adapted for medical imaging
/// - Large kernel sizes (up to 7x7x7) for capturing global context
/// - UpKern: compound scaling strategy for depth, width, and kernel size
/// - Achieves transformer-level performance with pure convolutions
/// </para>
/// <para>
/// <b>Reference:</b> Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation", MICCAI 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a MedNeXt model for CT/MRI organ segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 256, inputWidth: 256, inputDepth: 1, outputSize: 14);
/// var model = new MedNeXt&lt;double&gt;(architecture, numClasses: 14);
///
/// // Or load a pre-trained ONNX model for medical image analysis
/// var onnxModel = new MedNeXt&lt;double&gt;(architecture, "mednext.onnx", numClasses: 14);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation", "https://arxiv.org/abs/2303.09975", Year = 2023, Authors = "Roy et al.")]
public class MedNeXt<T> : Common.MedicalSegmentationBase<T>
{
    private readonly MedNeXtOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only MedNeXt's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from MedicalSegmentationBase -> SegmentationModelBase, as do SupportedModalities,
    // Supports2D and Supports3D.
    private readonly MedNeXtModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;

    /// <summary>
    /// The imaging modalities MedNeXt was trained on, passed to the base constructor.
    /// </summary>
    private static readonly string[] MedNeXtModalities = ["CT", "MRI_T1", "MRI_T2"];
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal MedNeXtModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes MedNeXt in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyWithLogitsLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 14).</param>
    /// <param name="modelSize">Model size variant (default: Small).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable MedNeXt model.
    /// </para>
    /// </remarks>
    public MedNeXt(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 14,
        MedNeXtModelSize modelSize = MedNeXtModelSize.Small, double dropRate = 0,
        MedNeXtOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture,
        // defaults the loss to CrossEntropyWithLogitsLoss and stores the modality list - exactly
        // what the deleted lines did by hand. `optimizer` is passed straight through INCLUDING
        // null; the base's lazy CreateDefaultOptimizer() produces the same
        // `new AdamWOptimizer<...>(this)` default, which could never be written as a
        // base-constructor argument because `this` is unavailable there.
        : base(architecture, optimizer, lossFunction, numClasses, MedNeXtModalities)
    {
        _options = options ?? new MedNeXtOptions(); Options = _options;
        ApplyMedNeXtInputFallback(architecture);
        _modelSize = modelSize; _dropRate = dropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Re-applies MedNeXt's 128x128 fallback for architectures that carry no input geometry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SegmentationModelBase falls back to 512x512 when the architecture supplies no input height
    /// or width. MedNeXt's documented fallback is 128x128 (the volumetric patch size used for
    /// medical training), so it is restored here for that unset case only - when the architecture
    /// does specify dimensions, the base's value already matches and nothing changes.
    /// </para>
    /// </remarks>
    private void ApplyMedNeXtInputFallback(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.InputHeight <= 0) _height = 128;
        if (architecture.InputWidth <= 0) _width = 128;
    }

    /// <summary>
    /// Initializes MedNeXt in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 14).</param>
    /// <param name="modelSize">Model size for metadata (default: Small).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained MedNeXt from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public MedNeXt(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 14, MedNeXtModelSize modelSize = MedNeXtModelSize.Small,
        MedNeXtOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry, stores the modality list and opens the InferenceSession - the same lines this
        // used to repeat.
        : base(architecture, onnxModelPath, numClasses, MedNeXtModalities)
    {
        _options = options ?? new MedNeXtOptions(); Options = _options;
        ApplyMedNeXtInputFallback(architecture);
        _modelSize = modelSize; _dropRate = 0;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    // PredictCore is inherited from SegmentationModelBase and dispatches to Forward / PredictOnnx
    // exactly as the deleted override did.

    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trains the model. Only available in native mode.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");
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
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(MedNeXtModelSize modelSize) => modelSize switch
    {
        MedNeXtModelSize.Small => ([32, 64, 128, 256], [2, 2, 2, 2], 256),
        MedNeXtModelSize.Base => ([32, 64, 128, 256], [2, 2, 2, 2], 256),
        MedNeXtModelSize.Medium => ([32, 64, 128, 256], [3, 4, 4, 4], 256),
        MedNeXtModelSize.Large => ([32, 64, 128, 256], [4, 8, 8, 8], 256),
        _ => ([32, 64, 128, 256], [2, 2, 2, 2], 256)
    };

    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result); return result;
    }

    // AddBatchDimension and RemoveBatchDimension are inherited from SegmentationModelBase.
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
            var encoderLayers = LayerHelper<T>.CreateMedNeXtEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateMedNeXtDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "MedNeXt" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
    { writer.Write(_height); writer.Write(_width); writer.Write(_channels); writer.Write(_numClasses); writer.Write((int)_modelSize); writer.Write(_decoderDim); writer.Write(_dropRate); writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty); writer.Write(_encoderLayerEnd); writer.Write(_channelDims.Length); foreach (int d in _channelDims) writer.Write(d); writer.Write(_depths.Length); foreach (int d in _depths) writer.Write(d); }

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
    { _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadDouble(); _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32(); int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32(); int dd = reader.ReadInt32(); for (int i = 0; i < dd; i++) _ = reader.ReadInt32(); }

    /// <summary>
    /// Creates a new instance with the same configuration but fresh weights.
    /// </summary>
    /// <returns>A new model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a copy for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => _useNativeMode
        ? new MedNeXt<T>(Architecture, Optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new MedNeXt<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, _options);

    // Dispose is inherited from SegmentationModelBase, which already disposes the ONNX session.
    // MedNeXt owns no further unmanaged resources.
    #endregion

    #region IMedicalSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase.
    // SupportedModalities is supplied to the base constructor via MedNeXtModalities, and
    // Supports3D (true), Supports2D (true) and SupportsFewShot (false) are MedicalSegmentationBase's
    // defaults already - re-declaring them here would only create two sources of one fact.

    /// <inheritdoc/>
    public override MedicalSegmentationResult<T> SegmentSlice(Tensor<T> slice)
    {
        var output = Predict(slice);
        var labels = Common.SegmentationTensorOps.ArgmaxAlongClassDim(output);
        var probs = Common.SegmentationTensorOps.SoftmaxAlongClassDim(output);
        int h = labels.Shape[0], w = labels.Shape[1];
        int numC = probs.Shape[0];
        var structures = new List<SegmentedStructure>();
        for (int c = 0; c < numC; c++)
        {
            int area = 0; double confSum = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if ((int)NumOps.ToDouble(labels[y, x]) == c) { area++; confSum += NumOps.ToDouble(probs[c, y, x]); }
            if (area > 0)
                structures.Add(new SegmentedStructure { ClassId = c, Name = $"Class_{c}", VolumeOrArea = area, MeanConfidence = confSum / area });
        }
        return new MedicalSegmentationResult<T> { Labels = labels, Probabilities = probs, Structures = structures };
    }
    /// <inheritdoc/>
    public override MedicalSegmentationResult<T> SegmentVolume(Tensor<T> volume)
    {
        if (volume.Rank <= 3)
            return SegmentSlice(volume);
        int numC = volume.Shape[0], depth = volume.Shape[1], h = volume.Shape[2], w = volume.Shape[3];
        var volLabels = new Tensor<T>([depth, h, w]);
        var volProbs = new Tensor<T>([numC, depth, h, w]);
        var structAccum = new Dictionary<int, (double area, double confSum)>();
        for (int d = 0; d < depth; d++)
        {
            var slice = new Tensor<T>([numC, h, w]);
            for (int c = 0; c < numC; c++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        slice[c, y, x] = volume[c, d, y, x];
            var result = SegmentSlice(slice);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    volLabels[d, y, x] = result.Labels[y, x];
            for (int c = 0; c < numC; c++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        volProbs[c, d, y, x] = result.Probabilities[c, y, x];
            foreach (var s in result.Structures)
            {
                if (structAccum.TryGetValue(s.ClassId, out var existing))
                    structAccum[s.ClassId] = (existing.area + s.VolumeOrArea, existing.confSum + s.MeanConfidence * s.VolumeOrArea);
                else
                    structAccum[s.ClassId] = (s.VolumeOrArea, s.MeanConfidence * s.VolumeOrArea);
            }
        }
        var structures = new List<SegmentedStructure>();
        foreach (var kvp in structAccum)
            structures.Add(new SegmentedStructure { ClassId = kvp.Key, Name = $"Class_{kvp.Key}", VolumeOrArea = kvp.Value.area, MeanConfidence = kvp.Value.confSum / kvp.Value.area });
        return new MedicalSegmentationResult<T> { Labels = volLabels, Probabilities = volProbs, Structures = structures };
    }
    /// <summary>
    /// Segments a query image, ignoring the support set.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Overrides the base rather than inheriting it deliberately. MedicalSegmentationBase's
    /// SegmentFewShot throws NotSupportedException whenever SupportsFewShot is false, which it is
    /// for MedNeXt; this model instead falls back to plain slice segmentation, and that behaviour
    /// is preserved exactly as it was before re-parenting.
    /// </para>
    /// </remarks>
    public override MedicalSegmentationResult<T> SegmentFewShot(Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
        => SegmentSlice(queryImage);
    #endregion
}
