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
/// BiomedParse: Biomedical image parsing with text prompts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Text-prompted biomedical image segmentation. Multi-modality biomedical parsing.
///
/// Common use cases:
/// - Text-prompted biomedical image segmentation
/// - Multi-modality biomedical parsing
/// - Detection and recognition in biomedical images
/// - Joint segmentation-detection-recognition
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Text-prompted segmentation for biomedical images
/// - Joint segmentation, detection, and recognition in one model
/// - Trained on 6M+ triples across 9 imaging modalities
/// - GPT-4 assisted harmonization of biomedical datasets
/// </para>
/// <para>
/// <b>Reference:</b> Zhao et al., "BiomedParse: a biomedical foundation model for image parsing of everything everywhere all at once", Nature Methods 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a BiomedParse model for text-prompted biomedical segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new BiomedParse&lt;double&gt;(architecture, numClasses: 1);
///
/// // Or load a pre-trained ONNX model for multi-modality biomedical parsing
/// var onnxModel = new BiomedParse&lt;double&gt;(architecture, "biomedparse.onnx", numClasses: 1);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("BiomedParse: a Biomedical Foundation Model for Image Parsing of Everything Everywhere All at Once", "https://arxiv.org/abs/2405.12971", Year = 2024, Authors = "Zhao et al.")]
public class BiomedParse<T> : Common.MedicalSegmentationBase<T>
{
    private readonly BiomedParseOptions _options;
    public override ModelOptions GetOptions() => _options;

    // BiomedParse paper defaults: Swin-B backbone (Zhao et al., Nature Methods 2024)
    private static readonly int[] DefaultChannelDims = [96, 192, 384, 768];
    private static readonly int[] DefaultDepths = [2, 2, 6, 2];
    private const int DefaultDecoderDim = 256;

    /// <summary>
    /// The imaging modalities BiomedParse was trained on, passed to the base constructor.
    /// </summary>
    private static readonly string[] BiomedParseModalities =
        ["CT", "MRI_T1", "MRI_T2", "Xray", "Ultrasound", "Pathology", "Dermoscopy", "Fundus", "Microscopy"];

    #region Fields
    // Only BiomedParse's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from MedicalSegmentationBase -> SegmentationModelBase, as do SupportedModalities and
    // Supports2D.
    private int[] _channelDims;
    private int _decoderDim;
    private int[] _depths;
    private double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// BiomedParse is a 2D slice model; it does not process volumes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Overrides MedicalSegmentationBase's <c>true</c> default. Supports2D and SupportsFewShot
    /// already match the base's defaults and so are not re-declared.
    /// </para>
    /// </remarks>
    public override bool Supports3D => false;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes BiomedParse in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyWithLogitsLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable BiomedParse model.
    /// </para>
    /// </remarks>
    public BiomedParse(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        double dropRate = 0.1,
        BiomedParseOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture,
        // defaults the loss to CrossEntropyWithLogitsLoss and stores the modality list - exactly
        // what the deleted lines did by hand. `optimizer` is passed straight through INCLUDING
        // null; the base's lazy CreateDefaultOptimizer() produces the same
        // `new AdamWOptimizer<...>(this)` default, which could never be written as a
        // base-constructor argument because `this` is unavailable there.
        : base(architecture, optimizer, lossFunction, numClasses, BiomedParseModalities)
    {
        _options = options ?? new BiomedParseOptions(); Options = _options;
        ApplyBiomedParseInputFallback(architecture);
        _dropRate = dropRate;
        _channelDims = DefaultChannelDims;
        _depths = DefaultDepths;
        _decoderDim = DefaultDecoderDim;
        InitializeLayers();
    }

    /// <summary>
    /// Re-applies BiomedParse's 1024x1024 fallback for architectures that carry no input geometry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SegmentationModelBase falls back to 512x512 when the architecture supplies no input height
    /// or width. BiomedParse's documented fallback is 1024x1024, so it is restored here for that
    /// unset case only - when the architecture does specify dimensions, the base's value already
    /// matches and nothing changes.
    /// </para>
    /// </remarks>
    private void ApplyBiomedParseInputFallback(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.InputHeight <= 0) _height = 1024;
        if (architecture.InputWidth <= 0) _width = 1024;
    }

    /// <summary>
    /// Initializes BiomedParse in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained BiomedParse from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public BiomedParse(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1,
        BiomedParseOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry, stores the modality list and opens the InferenceSession - the same lines this
        // used to repeat.
        : base(architecture, onnxModelPath, numClasses, BiomedParseModalities)
    {
        _options = options ?? new BiomedParseOptions(); Options = _options;
        ApplyBiomedParseInputFallback(architecture);
        _dropRate = 0.1;
        _channelDims = DefaultChannelDims;
        _depths = DefaultDepths;
        _decoderDim = DefaultDecoderDim;
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
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");

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
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var features = input;
        for (int i = 0; i < Layers.Count; i++)
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
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
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
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = _options.EncoderLayerCount ?? Architecture.Layers.Count / 2;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateBiomedParseEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateBiomedParseDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
            Layers.AddRange(decoderLayers);
        }
    }

    /// <summary>
    /// Updates all trainable parameters from a flat parameter vector.
    /// </summary>
    /// <param name="parameters">Flat vector of all model parameters.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Replaces all model weights with new values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int totalRequired = 0;
        foreach (var l in Layers)
            totalRequired += l.GetParameters().Length;

        if (parameters.Length < totalRequired)
            throw new ArgumentException(
                $"Parameter vector length {parameters.Length} is less than required {totalRequired}.",
                nameof(parameters));

        int offset = 0;
        foreach (var layer in Layers)
        {
            int count = layer.GetParameters().Length;
            var newParams = new Vector<T>(count);
            for (int i = 0; i < count; i++)
                newParams[i] = parameters[offset + i];
            layer.UpdateParameters(newParams);
            offset += count;
        }
    }

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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "BiomedParse" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
    {
        _height = reader.ReadInt32();
        _width = reader.ReadInt32();
        _channels = reader.ReadInt32();
        _numClasses = reader.ReadInt32();
        _decoderDim = reader.ReadInt32();
        _dropRate = reader.ReadDouble();
        _useNativeMode = reader.ReadBoolean();
        _onnxModelPath = reader.ReadString();
        _encoderLayerEnd = reader.ReadInt32();
        int dc = reader.ReadInt32();
        _channelDims = new int[dc];
        for (int i = 0; i < dc; i++) _channelDims[i] = reader.ReadInt32();
        int dd = reader.ReadInt32();
        _depths = new int[dd];
        for (int i = 0; i < dd; i++) _depths[i] = reader.ReadInt32();
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => _useNativeMode
        ? new BiomedParse<T>(Architecture, Optimizer, LossFunction, _numClasses, _dropRate, _options)
        : new BiomedParse<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _options);

    // Dispose is inherited from SegmentationModelBase, which already disposes the ONNX session.
    // BiomedParse owns no further unmanaged resources.
    #endregion

    #region IMedicalSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase.
    // SupportedModalities is supplied to the base constructor via BiomedParseModalities; Supports2D
    // (true) and SupportsFewShot (false) are MedicalSegmentationBase's defaults already, and
    // Supports3D is overridden to false alongside the other properties above.

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
        => throw new NotSupportedException("BiomedParse does not support 3D volumetric segmentation. Use SegmentSlice for 2D slices.");

    /// <summary>
    /// Not supported: BiomedParse has no few-shot pathway.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Overridden rather than inherited so the BiomedParse-specific message is preserved verbatim.
    /// MedicalSegmentationBase's SegmentFewShot would also throw NotSupportedException here (since
    /// SupportsFewShot is false), but with a generic message.
    /// </para>
    /// </remarks>
    public override MedicalSegmentationResult<T> SegmentFewShot(Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
        => throw new NotSupportedException("BiomedParse does not support few-shot segmentation. Use SegmentSlice for standard inference.");
    #endregion
}
