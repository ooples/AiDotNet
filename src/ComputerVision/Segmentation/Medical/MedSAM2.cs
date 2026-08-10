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
/// MedSAM-2: SAM 2 adapted for medical image and video segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Medical video segmentation (endoscopy, ultrasound). 3D volumetric medical segmentation (treat slices as video).
///
/// Common use cases:
/// - Medical video segmentation (endoscopy, ultrasound)
/// - 3D volumetric medical segmentation (treat slices as video)
/// - One-click medical segmentation across frames
/// - Temporal-consistent medical image analysis
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - SAM 2 architecture with memory attention for temporal consistency
/// - Treats 3D medical volumes as video sequences
/// - Point and box prompts propagated across frames/slices
/// - Hiera (Hierarchical) image encoder backbone
/// </para>
/// <para>
/// <b>Reference:</b> Zhu et al., "Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2", arXiv 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a MedSAM2 model for medical video and volumetric segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new MedSAM2&lt;double&gt;(architecture, numClasses: 1);
///
/// // Or load a pre-trained ONNX model for endoscopy and 3D volume segmentation
/// var onnxModel = new MedSAM2&lt;double&gt;(architecture, "medsam2.onnx", numClasses: 1);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2", "https://arxiv.org/abs/2408.00874", Year = 2024, Authors = "Zhu et al.")]
public class MedSAM2<T> : Common.MedicalSegmentationBase<T>
{
    private readonly MedSAM2Options _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only MedSAM2's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from MedicalSegmentationBase -> SegmentationModelBase, as do SupportedModalities,
    // Supports2D and Supports3D.
    private readonly MedSAM2ModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;

    // MedSAM2's own optimizer bookkeeping. NOTE: this model deliberately reads the base's raw
    // _optimizer FIELD (the un-defaulted constructor argument) rather than the base's Optimizer
    // PROPERTY. The property would lazily materialize CreateDefaultOptimizer(), and passing that
    // non-null value into TrainWithTape would bypass GetOrCreateBaseOptimizer below - the tuned
    // warmup path this model needs to stay finite.
    private readonly bool _hasUserSuppliedOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _baseTapeOptimizer;

    /// <summary>
    /// The imaging modalities MedSAM2 was trained on, passed to the base constructor.
    /// </summary>
    private static readonly string[] MedSAM2Modalities =
        ["CT", "MRI_T1", "MRI_T2", "Ultrasound", "Endoscopy"];
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal MedSAM2ModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes MedSAM2 in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyWithLogitsLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="modelSize">Model size variant (default: Tiny).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable MedSAM2 model.
    /// </para>
    /// </remarks>
    public MedSAM2(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        MedSAM2ModelSize modelSize = MedSAM2ModelSize.Tiny, double dropRate = 0,
        MedSAM2Options? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture,
        // defaults the loss to CrossEntropyWithLogitsLoss and stores the modality list - exactly
        // what the deleted lines did by hand. `optimizer` is passed straight through INCLUDING
        // null, and the base stores it verbatim in _optimizer; MedSAM2 never asks for the base's
        // lazy default, so a null here still routes training through GetOrCreateBaseOptimizer.
        : base(architecture, optimizer, lossFunction, numClasses, MedSAM2Modalities)
    {
        _options = options ?? new MedSAM2Options(); Options = _options;
        ApplyMedSAM2InputFallback(architecture);
        _modelSize = modelSize; _dropRate = dropRate;
        _hasUserSuppliedOptimizer = optimizer is not null;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize, _options);
        InitializeLayers();
    }

    /// <summary>
    /// Re-applies MedSAM2's 1024x1024 fallback for architectures that carry no input geometry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SegmentationModelBase falls back to 512x512 when the architecture supplies no input height
    /// or width. MedSAM2 inherits SAM 2's 1024x1024 input resolution, so that fallback is restored
    /// here for the unset case only - when the architecture does specify dimensions, the base's
    /// value already matches and nothing changes.
    /// </para>
    /// </remarks>
    private void ApplyMedSAM2InputFallback(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.InputHeight <= 0) _height = 1024;
        if (architecture.InputWidth <= 0) _width = 1024;
    }

    /// <summary>
    /// Initializes MedSAM2 in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="modelSize">Model size for metadata (default: Tiny).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained MedSAM2 from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public MedSAM2(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1, MedSAM2ModelSize modelSize = MedSAM2ModelSize.Tiny,
        MedSAM2Options? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry, stores the modality list and opens the InferenceSession - the same lines this
        // used to repeat.
        : base(architecture, onnxModelPath, numClasses, MedSAM2Modalities)
    {
        _options = options ?? new MedSAM2Options(); Options = _options;
        ApplyMedSAM2InputFallback(architecture);
        _modelSize = modelSize; _dropRate = 0;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize, _options);
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
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!_useNativeMode) return PredictOnnx(input);
        SetTrainingMode(false); // eval mode: sync inference-weight cache to latest weights
        return Forward(input);
    }

    /// <summary>
    /// Supplies the training optimizer: a linear-warmup Adam at 1e-4 (honoring a
    /// constructor-supplied optimizer when present) so the from-scratch SAM-style stack
    /// does not diverge — the default optimizer was never consulted and MoreData saw the
    /// loss climb (22 -> 58) over successive iterations. Mirrors the MedSAM / UniVS / ODISE fix.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
    {
        if (_hasUserSuppliedOptimizer && _optimizer is not null)
            return _optimizer;

        return _baseTapeOptimizer ??= new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                UseAMSGrad = false,
                InitialLearningRate = 0.0001,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.LinearWarmupScheduler(
                    baseLearningRate: 0.0001,
                    warmupSteps: 5,
                    totalSteps: 300,
                    warmupInitLr: 0.00001)
            });
    }

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
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }
    #endregion

    #region Private Methods
    /// <summary>
    /// Resolves the encoder configuration, preferring explicit caller overrides over the paper preset.
    /// </summary>
    /// <remarks>
    /// The presets below are the published Hiera encoders and remain the default in every case, so
    /// production behaviour is unchanged. Previously they were the ONLY reachable configurations, so
    /// even MedSAM2ModelSize.Tiny built a full 96/192/384/768 encoder — no caller could construct the
    /// model at a bounded size for a test or a memory-constrained deployment. Options.ChannelDims /
    /// Depths / DecoderDim now allow that; leaving them null selects the preset exactly as before.
    /// </remarks>
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(
        MedSAM2ModelSize modelSize, MedSAM2Options? options)
    {
        (int[] ChannelDims, int[] Depths, int DecoderDim) preset = modelSize switch
        {
            MedSAM2ModelSize.Tiny => ([96, 192, 384, 768], [2, 2, 6, 2], 256),
            MedSAM2ModelSize.Base => ([112, 224, 448, 896], [2, 3, 16, 3], 256),
            MedSAM2ModelSize.Large => ([144, 288, 576, 1152], [2, 6, 36, 4], 256),
            _ => ([96, 192, 384, 768], [2, 2, 6, 2], 256)
        };

        int[] dims = options?.ChannelDims is { Length: > 0 } d ? d : preset.ChannelDims;
        int[] depths = options?.Depths is { Length: > 0 } p ? p : preset.Depths;
        if (dims.Length != depths.Length)
        {
            throw new ArgumentException(
                $"MedSAM2Options.ChannelDims ({dims.Length}) and Depths ({depths.Length}) must have the " +
                "same number of stages.", nameof(options));
        }

        return (dims, depths, options?.DecoderDim ?? preset.DecoderDim);
    }

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
            var encoderLayers = LayerHelper<T>.CreateMedSAM2EncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateMedSAM2DecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "MedSAM2" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
        ? new MedSAM2<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new MedSAM2<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, _options);

    // Dispose is inherited from SegmentationModelBase, which already disposes the ONNX session.
    // MedSAM2 owns no further unmanaged resources.
    #endregion

    #region IMedicalSegmentation Implementation
    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase.
    // SupportedModalities is supplied to the base constructor via MedSAM2Modalities, and
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
    /// for MedSAM2; this model instead falls back to plain slice segmentation, and that behaviour
    /// is preserved exactly as it was before re-parenting.
    /// </para>
    /// </remarks>
    public override MedicalSegmentationResult<T> SegmentFewShot(Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
        => SegmentSlice(queryImage);
    #endregion
}
