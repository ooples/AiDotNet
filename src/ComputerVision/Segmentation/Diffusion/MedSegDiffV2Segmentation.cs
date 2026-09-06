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
/// MedSegDiff-V2 Segmentation: Diffusion-based medical image segmentation pipeline.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Diffusion-based medical segmentation. High-precision medical boundary delineation.
///
/// Common use cases:
/// - Diffusion-based medical segmentation
/// - High-precision medical boundary delineation
/// - Stochastic segmentation for uncertainty estimation
/// - Multi-modal medical image segmentation
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Conditional diffusion model for segmentation mask denoising
/// - Spectrum-Space Former for frequency and spatial features
/// - Iterative refinement via reverse diffusion process
/// - Handles ambiguous medical image boundaries
/// </para>
/// <para>
/// <b>Reference:</b> Wu et al., "MedSegDiff-V2: Diffusion-based Medical Image Segmentation with Transformer", AAAI 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var trainingMasks = Tensor&lt;double&gt;.CreateRandom(1, 3, 32, 32);
/// var medicalImage = Tensor&lt;double&gt;.CreateRandom(1, 3, 32, 32);
/// var trainingImages = Tensor&lt;double&gt;.CreateRandom(4, 3, 32, 32);
/// // Use AiModelBuilder facade for medical image segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 1);
///
/// var builder = new AiModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
///     .ConfigureModel(new MedSegDiffV2Segmentation&lt;double&gt;(architecture, numClasses: 1));
///
/// var result = builder.Build(trainingImages, trainingMasks);
/// var segmentation = result.Predict(medicalImage);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MedSegDiff-V2: Diffusion-based Medical Image Segmentation with Transformer", "https://arxiv.org/abs/2301.11798", Year = 2023, Authors = "Junde Wu, Wei Ji, Huazhu Fu, Min Xu, Yueming Jin, Yanwu Xu")]
public partial class MedSegDiffV2Segmentation<T> : Common.MedicalSegmentationBase<T>
{
    private readonly MedSegDiffV2SegmentationOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only MedSegDiff-V2's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from MedicalSegmentationBase -> SegmentationModelBase.
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;

    /// <summary>Modalities this diffusion segmentation pipeline was trained on.</summary>
    private static readonly string[] _modalities = ["CT", "MRI_T1", "MRI_T2", "Ultrasound"];
    #endregion

    #region Properties
    /// <summary>
    /// Gets whether using native mode (trainable) or ONNX mode (inference only).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    /// <inheritdoc />
    public override bool Supports3D => false;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes MedSegDiffV2Segmentation in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable MedSegDiffV2Segmentation model.
    /// </para>
    /// </remarks>
    public MedSegDiffV2Segmentation(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        double dropRate = 0,
        MedSegDiffV2SegmentationOptions? options = null)
        : base(architecture, optimizer, lossFunction, numClasses, _modalities)
    {
        _options = options ?? new MedSegDiffV2SegmentationOptions(); Options = _options;
        ApplyMedSegDiffDefaultGeometry(architecture);
        _dropRate = dropRate;
        _channelDims = [64, 128, 256, 512];
        _depths = [2, 2, 2, 2];
        _decoderDim = 256;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes MedSegDiffV2Segmentation in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained MedSegDiffV2Segmentation from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public MedSegDiffV2Segmentation(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1,
        MedSegDiffV2SegmentationOptions? options = null)
        : base(architecture, onnxModelPath, numClasses, _modalities)
    {
        _options = options ?? new MedSegDiffV2SegmentationOptions(); Options = _options;
        ApplyMedSegDiffDefaultGeometry(architecture);
        _dropRate = 0;
        _channelDims = [64, 128, 256, 512];
        _depths = [2, 2, 2, 2];
        _decoderDim = 256;
        InitializeLayers();
    }

    /// <summary>
    /// Restores MedSegDiff-V2's own 256x256 fallback for unspecified input geometry.
    /// </summary>
    /// <remarks>
    /// SegmentationModelBase falls back to 512x512 when the architecture leaves the input size
    /// unset; MedSegDiff-V2 has always fallen back to 256x256, so that stays the model's own rule.
    /// </remarks>
    private void ApplyMedSegDiffDefaultGeometry(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.InputHeight <= 0) _height = 256;
        if (architecture.InputWidth <= 0) _width = 256;
    }
    #endregion

    #region Public Methods
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
    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

    /// <inheritdoc />
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

    // AddBatchDimension / RemoveBatchDimension are inherited from SegmentationModelBase.
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
            var encoderLayers = LayerHelper<T>.CreateMedSegDiffV2SegmentationEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateMedSegDiffV2SegmentationDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "MedSegDiffV2Segmentation" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelData = SerializeForMetadata()
    };

    // Dispose of the ONNX session and the _disposed latch are handled by SegmentationModelBase.
    #endregion

    #region IMedicalSegmentation Implementation
    // NumClasses / InputHeight / InputWidth / IsOnnxMode / Segment / SupportedModalities /
    // Supports2D / SupportsFewShot all arrive from MedicalSegmentationBase.

    /// <inheritdoc />
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
    /// <inheritdoc />
    public override MedicalSegmentationResult<T> SegmentVolume(Tensor<T> volume) => SegmentSlice(volume);

    /// <inheritdoc />
    public override MedicalSegmentationResult<T> SegmentFewShot(
        Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
        => SegmentSlice(queryImage);
    #endregion
}
