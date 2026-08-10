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
/// U-Mamba: Hybrid CNN-Mamba architecture for medical segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Medical image segmentation with long-range dependencies. CT and MRI organ segmentation.
///
/// Common use cases:
/// - Medical image segmentation with long-range dependencies
/// - CT and MRI organ segmentation
/// - 3D medical volume segmentation
/// - Efficient medical AI with linear complexity
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Hybrid CNN + Mamba (State Space Model) blocks in U-Net
/// - Linear complexity for processing long-range dependencies
/// - Captures both local CNN features and global SSM context
/// - U-Net architecture with Mamba blocks replacing transformer layers
/// </para>
/// <para>
/// <b>Reference:</b> Ma et al., "U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation", arXiv 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a U-Mamba model for medical image segmentation with long-range dependencies
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 256, inputWidth: 256, inputDepth: 1, outputSize: 14);
/// var model = new UMamba&lt;double&gt;(architecture, numClasses: 14);
///
/// // Or load a pre-trained ONNX model for CT/MRI organ segmentation
/// var onnxModel = new UMamba&lt;double&gt;(architecture, "umamba.onnx", numClasses: 14);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation", "https://arxiv.org/abs/2401.04722", Year = 2024, Authors = "Ma et al.")]
public class UMamba<T> : Common.MedicalSegmentationBase<T>
{
    private readonly UMambaOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only UMamba's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from MedicalSegmentationBase -> SegmentationModelBase.
    private static readonly string[] ModalitiesSupported = ["CT", "MRI_T1", "MRI_T2"];
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, SupportedModalities,
    // Supports3D, Supports2D and SupportsFewShot are all supplied identically by the base.
    internal bool UseNativeMode => _useNativeMode;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes UMamba in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyWithLogitsLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 14).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable UMamba model.
    /// </para>
    /// </remarks>
    public UMamba(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 14,
        double dropRate = 0,
        UMambaOptions? options = null)
        // `optimizer` is passed straight through - INCLUDING null. The base resolves the default
        // AdamW lazily via CreateDefaultOptimizer(), which a base-constructor argument cannot do.
        : base(architecture, optimizer, lossFunction, numClasses, ModalitiesSupported)
    {
        _options = options ?? new UMambaOptions(); Options = _options;
        // U-Mamba defaults to 256x256, not the base's 512x512, so the geometry fallback stays here.
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 256;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 256;
        _dropRate = dropRate;
        _channelDims = [32, 64, 128, 256];
        _depths = [2, 2, 2, 2];
        _decoderDim = 256;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes UMamba in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 14).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained UMamba from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public UMamba(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 14,
        UMambaOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession - the same twenty lines this used to repeat.
        : base(architecture, onnxModelPath, numClasses, ModalitiesSupported)
    {
        _options = options ?? new UMambaOptions(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 256;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 256;
        _dropRate = 0;
        _channelDims = [32, 64, 128, 256];
        _depths = [2, 2, 2, 2];
        _decoderDim = 256;
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
        if (input.Shape.Length == 3) { input = AddLeadingBatchDimension(input); expectedOutput = AddLeadingBatchDimension(expectedOutput); } else if (input.Shape.Length != 4 && input.Shape.Length != 5) throw new ArgumentException($"UMamba supports 2D [C,H,W]/[B,C,H,W] and 3D [C,D,H,W]/[B,C,D,H,W]. Got rank {input.Shape.Length}.", nameof(input));
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
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4 || input.Rank == 5; if (!hasBatch) input = AddLeadingBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4 || input.Rank == 5; if (!hasBatch) input = AddLeadingBatchDimension(input);
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

    // RemoveBatchDimension comes from SegmentationModelBase (identical, plus a Shape[0] == 1 guard).
    // AddBatchDimension does NOT: the base's promotes rank-3 [C,H,W] only, while U-Mamba promotes
    // 2D masks and 3D volumes too, so the rank-agnostic version keeps its own name.
    private static Tensor<T> AddLeadingBatchDimension(Tensor<T> tensor)
    { var s = new int[tensor.Shape.Length + 1]; s[0] = 1; for (int i = 0; i < tensor.Shape.Length; i++) s[i + 1] = tensor.Shape[i]; var result = new Tensor<T>(s); tensor.Data.Span.CopyTo(result.Data.Span); return result; }
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
            var encoderLayers = LayerHelper<T>.CreateUMambaEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateUMambaDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "UMamba" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
    { writer.Write(_height); writer.Write(_width); writer.Write(_channels); writer.Write(_numClasses); writer.Write(_decoderDim); writer.Write(_dropRate); writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty); writer.Write(_encoderLayerEnd); writer.Write(_channelDims.Length); foreach (int d in _channelDims) writer.Write(d); writer.Write(_depths.Length); foreach (int d in _depths) writer.Write(d); }

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
    { _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadDouble(); _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32(); int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32(); int dd = reader.ReadInt32(); for (int i = 0; i < dd; i++) _ = reader.ReadInt32(); }

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
        ? new UMamba<T>(Architecture, _optimizer, LossFunction, _numClasses, _dropRate, _options)
        : new UMamba<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _options);

    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and flips _disposed,
    // and UMamba owns no other unmanaged resource.
    #endregion

    #region IMedicalSegmentation Implementation
    /// <summary>Segments a single 2D medical slice.</summary>
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
    /// <summary>Segments a 3D volume slice-by-slice and aggregates the per-structure statistics.</summary>
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
    /// <inheritdoc/>
    public override MedicalSegmentationResult<T> SegmentFewShot(
        Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
        => SegmentSlice(queryImage);
    #endregion
}
