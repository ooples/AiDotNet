using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// Swin UNETR: Swin Transformer encoder for 3D medical segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> 3D medical volume segmentation. Brain tumor segmentation from MRI.
///
/// Common use cases:
/// - 3D medical volume segmentation
/// - Brain tumor segmentation from MRI
/// - CT organ segmentation
/// - Self-supervised pre-trained medical segmentation
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Swin Transformer encoder with shifted window attention
/// - U-Net style decoder with skip connections from encoder stages
/// - Designed for 3D volumetric medical data
/// - Self-supervised pre-training on large medical datasets
/// </para>
/// <para>
/// <b>Reference:</b> Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images", BrainLes 2022.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SwinUNETR model for 3D brain tumor segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 256, inputWidth: 256, inputDepth: 1, outputSize: 14);
/// var model = new SwinUNETR&lt;double&gt;(architecture, numClasses: 14);
///
/// // Or load a pre-trained ONNX model for brain MRI segmentation
/// var onnxModel = new SwinUNETR&lt;double&gt;(architecture, "swinunetr.onnx", numClasses: 14);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images", "https://arxiv.org/abs/2201.01266", Year = 2022, Authors = "Ali Hatamizadeh, Vishwesh Nath, Yucheng Tang, Dong Yang, Holger R. Roth, Daguang Xu")]
public partial class SwinUNETR<T> : Common.MedicalSegmentationBase<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Does NOT downsample: measured [1,3,64,64] -> [1,C,64,64]. A UNETR decoder upsamples back to the
    /// input grid, which is the point of the architecture.
    /// </remarks>
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => SpatialStrideContract(inputRank, 1);

    private readonly SwinUNETROptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only SwinUNETR's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from MedicalSegmentationBase -> SegmentationModelBase.
    private static readonly string[] ModalitiesSupported = ["MRI_T1", "MRI_T2", "MRI_FLAIR", "CT"];
    private readonly SwinUNETRModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, SupportedModalities,
    // Supports3D, Supports2D and SupportsFewShot are all supplied identically by the base.
    internal bool UseNativeMode => _useNativeMode;
    internal SwinUNETRModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes SwinUNETR in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyWithLogitsLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 14).</param>
    /// <param name="modelSize">Model size variant (default: Tiny).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable SwinUNETR model.
    /// </para>
    /// </remarks>
    public SwinUNETR(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 14,
        SwinUNETRModelSize modelSize = SwinUNETRModelSize.Tiny, double dropRate = 0.1,
        SwinUNETROptions? options = null)
        // `optimizer` is passed straight through - INCLUDING null. The base resolves the default
        // lazily via CreateDefaultOptimizer(), overridden below to keep the paper's AdamW recipe.
        : base(architecture, optimizer, lossFunction, numClasses, ModalitiesSupported)
    {
        _options = options ?? new SwinUNETROptions(); Options = _options;
        // Swin UNETR defaults to 96x96 crops, not the base's 512x512, so the fallback stays here.
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 96;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 96;
        _modelSize = modelSize; _dropRate = dropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Per Hatamizadeh 2022 §4.2 ("Training Setup"): AdamW with peak learning rate 8e-4, cosine
    /// decay, and weight decay 1e-5. A short linear warmup is the standard transformer/PyTorch
    /// recipe for stabilizing randomly initialized dense heads on tiny batches while preserving
    /// the paper's optimizer, peak LR, and decay shape.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
        => new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 0.0008,
                LearningRateScheduler =
                    new AiDotNet.LearningRateSchedulers.LinearWarmupScheduler(
                        baseLearningRate: 0.0008,
                        warmupSteps: 100,
                        totalSteps: 5000,
                        warmupInitLr: 0.00008,
                        decayMode: AiDotNet.LearningRateSchedulers.LinearWarmupScheduler.DecayMode.Cosine,
                        endLr: 0.0),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
                WeightDecay = 1e-5,
            });

    /// <summary>
    /// Initializes SwinUNETR in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 14).</param>
    /// <param name="modelSize">Model size for metadata (default: Tiny).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained SwinUNETR from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public SwinUNETR(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 14, SwinUNETRModelSize modelSize = SwinUNETRModelSize.Tiny,
        SwinUNETROptions? options = null)
        // The base validates the path, sets ONNX mode and opens the InferenceSession.
        : base(architecture, onnxModelPath, numClasses, ModalitiesSupported)
    {
        _options = options ?? new SwinUNETROptions(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 96;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 96;
        _modelSize = modelSize; _dropRate = 0.1;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
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
        bool inputWasUnbatched = input.Shape.Length == 3;
        if (inputWasUnbatched) input = AddLeadingBatchDimension(input);
        else if (input.Shape.Length != 4 && input.Shape.Length != 5) throw new ArgumentException($"SwinUNETR supports 2D [C,H,W]/[B,C,H,W] and 3D [C,D,H,W]/[B,C,D,H,W]. Got rank {input.Shape.Length}.", nameof(input));

        expectedOutput = NormalizeTrainingTarget(expectedOutput, inputWasUnbatched);
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
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(SwinUNETRModelSize modelSize) => modelSize switch
    {
        SwinUNETRModelSize.Tiny => ([48, 96, 192, 384], [2, 2, 6, 2], 256),
        SwinUNETRModelSize.Small => ([96, 192, 384, 768], [2, 2, 6, 2], 256),
        SwinUNETRModelSize.Base => ([128, 256, 512, 1024], [2, 2, 6, 2], 256),
        _ => ([48, 96, 192, 384], [2, 2, 6, 2], 256)
    };

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
    // AddBatchDimension does NOT: the base's promotes rank-3 [C,H,W] only, while SwinUNETR promotes
    // rank-2/3/4 tensors (2D masks, 3D volumes), so the rank-agnostic version keeps its own name.
    private static Tensor<T> AddLeadingBatchDimension(Tensor<T> tensor)
    { var s = new int[tensor.Shape.Length + 1]; s[0] = 1; for (int i = 0; i < tensor.Shape.Length; i++) s[i + 1] = tensor.Shape[i]; var result = new Tensor<T>(s); tensor.Data.Span.CopyTo(result.Data.Span); return result; }

    private Tensor<T> NormalizeTrainingTarget(Tensor<T> target, bool inputWasUnbatched)
    {
        // PyTorch-style dense segmentation targets are class-index masks:
        //   logits [B, C, H, W] + target [B, H, W]
        // Also keep backward compatibility with one-hot/probability masks:
        //   logits [B, C, H, W] + target [B, C, H, W]
        if (target.Shape.Length == 2)
            return AddLeadingBatchDimension(target);

        if (inputWasUnbatched && target.Shape.Length == 3 && target.Shape[0] == _numClasses)
            return AddLeadingBatchDimension(target);

        if (target.Shape.Length == 3 || target.Shape.Length == 4 || target.Shape.Length == 5)
            return target;

        throw new ArgumentException(
            $"SwinUNETR target must be a class-index mask [H,W]/[B,H,W] or a one-hot/probability mask [C,H,W]/[B,C,H,W]. Got rank {target.Shape.Length}.",
            nameof(target));
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
            var encoderLayers = LayerHelper<T>.CreateSwinUNETREncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateSwinUNETRDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
            Layers.AddRange(decoderLayers);
        }
    }

    /// <summary>
    /// Use the AdamW optimizer the constructor stored in <c>_optimizer</c>
    /// for the base class's tape-training path. The default
    /// <see cref="GetOrCreateBaseOptimizer"/> returns plain Adam with
    /// lr=0.001, which is too conservative for the per-pixel CE-with-logits
    /// memorization task — the network only achieves ≈0.25% loss decrease
    /// over the 100-step probe, well under the 1% threshold. The
    /// constructor-supplied AdamW (default lr=0.001 but with decoupled weight
    /// decay) is the paper-recommended optimizer (Hatamizadeh 2022); routing
    /// it through the base path so TrainWithTape actually steps it.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
    {
        return Optimizer ?? base.GetOrCreateBaseOptimizer();
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "SwinUNETR" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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


    /// <summary>
    /// Reads configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads model configuration when restoring a saved model.
    /// </para>
    /// </remarks>


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
        ? new SwinUNETR<T>(Architecture, optimizer: null, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new SwinUNETR<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, _options);

    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and flips _disposed,
    // and SwinUNETR owns no other unmanaged resource.
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
