using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// UniverSeg: Universal Medical Image Segmentation via cross-attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Few-shot medical segmentation without fine-tuning. Cross-domain medical image segmentation.
///
/// Common use cases:
/// - Few-shot medical segmentation without fine-tuning
/// - Cross-domain medical image segmentation
/// - Label-efficient medical AI
/// - New task adaptation from a few examples
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - CrossBlock mechanism for support-query feature interaction
/// - No fine-tuning needed for new segmentation tasks
/// - Uses a small labeled support set at inference time
/// - Trained on diverse medical segmentation datasets
/// </para>
/// <para>
/// <b>Reference:</b> Butoi et al., "UniverSeg: Universal Medical Image Segmentation", ICCV 2023.
/// </para>
/// </remarks>
public class UniverSeg<T> : NeuralNetworkBase<T>, IMedicalSegmentation<T>
{
    private readonly UniverSegOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    private readonly int _height, _width, _channels, _numClasses;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    private readonly bool _useNativeMode;
    private readonly string? _onnxModelPath;
    private InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;
    private int _encoderLayerEnd;
    #endregion

    #region Properties
    /// <summary>
    /// Gets whether this UniverSeg instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns <c>true</c> in native mode, <c>false</c> in ONNX mode.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal int NumClasses => _numClasses;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes UniverSeg in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable UniverSeg model.
    /// </para>
    /// </remarks>
    public UniverSeg(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        double dropRate = 0,
        UniverSegOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new UniverSegOptions(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 128;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 128;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses; _dropRate = dropRate;
        _useNativeMode = true; _onnxModelPath = null;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _channelDims = [64, 128, 256, 512];
        _depths = [2, 2, 2, 2];
        _decoderDim = 256;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes UniverSeg in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained UniverSeg from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public UniverSeg(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1,
        UniverSegOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new UniverSegOptions(); Options = _options;
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"UniverSeg ONNX model not found: {onnxModelPath}");
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 128;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 128;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses; _dropRate = 0;
        _useNativeMode = false; _onnxModelPath = onnxModelPath; _optimizer = null;
        _channelDims = [64, 128, 256, 512];
        _depths = [2, 2, 2, 2];
        _decoderDim = 256;
        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load UniverSeg ONNX model: {ex.Message}", ex); }
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
    public override Tensor<T> Predict(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

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
        var predicted = Forward(input);
        var lossGradient = predicted.Transform((v, idx) => NumOps.Subtract(v, expectedOutput.Data.Span[idx]));
        BackwardPass(lossGradient); _optimizer?.UpdateParameters(Layers);
    }
    #endregion

    #region Private Methods
    private Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result); return result;
    }

    private void BackwardPass(Tensor<T> gradient)
    { if (!_useNativeMode || Layers.Count == 0) return; for (int i = Layers.Count - 1; i >= 0; i--) gradient = Layers[i].Backward(gradient); }

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    { var result = new Tensor<T>([1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]); tensor.Data.Span.CopyTo(result.Data.Span); return result; }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    { int[] s = new int[tensor.Shape.Length - 1]; for (int i = 0; i < s.Length; i++) s[i] = tensor.Shape[i + 1]; var r = new Tensor<T>(s); tensor.Data.Span.CopyTo(r.Data.Span); return r; }
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
            var encoderLayers = LayerHelper<T>.CreateUniverSegEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateUniverSegDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
    { int o = 0; foreach (var l in Layers) { var p = l.GetParameters(); int c = p.Length; if (o + c <= parameters.Length) { var n = new Vector<T>(c); for (int i = 0; i < c; i++) n[i] = parameters[o + i]; l.UpdateParameters(n); o += c; } } }

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
        ModelType = ModelType.SemanticSegmentation,
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "UniverSeg" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelData = this.Serialize()
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
        ? new UniverSeg<T>(Architecture, _optimizer, LossFunction, _numClasses, _dropRate, _options)
        : new UniverSeg<T>(Architecture, _onnxModelPath!, _numClasses, _options);

    /// <summary>
    /// Releases managed resources including the ONNX inference session.
    /// </summary>
    /// <param name="disposing">True when called from Dispose().</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frees memory used by the ONNX runtime.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    { if (!_disposed) { if (disposing) { _onnxSession?.Dispose(); _onnxSession = null; } _disposed = true; } base.Dispose(disposing); }
    #endregion

    #region IMedicalSegmentation Implementation
    int ISegmentationModel<T>.NumClasses => _numClasses;
    int ISegmentationModel<T>.InputHeight => _height;
    int ISegmentationModel<T>.InputWidth => _width;
    bool ISegmentationModel<T>.IsOnnxMode => !_useNativeMode;
    Tensor<T> ISegmentationModel<T>.Segment(Tensor<T> image) => Predict(image);
    IReadOnlyList<string> IMedicalSegmentation<T>.SupportedModalities => ["CT", "MRI_T1", "Xray", "Ultrasound", "Dermoscopy", "Microscopy"];
    bool IMedicalSegmentation<T>.Supports3D => false;
    bool IMedicalSegmentation<T>.Supports2D => true;
    bool IMedicalSegmentation<T>.SupportsFewShot => true;
    MedicalSegmentationResult<T> IMedicalSegmentation<T>.SegmentSlice(Tensor<T> slice)
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
    MedicalSegmentationResult<T> IMedicalSegmentation<T>.SegmentVolume(Tensor<T> volume)
        => ((IMedicalSegmentation<T>)this).SegmentSlice(volume);
    MedicalSegmentationResult<T> IMedicalSegmentation<T>.SegmentFewShot(Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
    {
        var queryFeatures = Predict(queryImage);
        int numC = queryFeatures.Shape[0], h = queryFeatures.Shape[1], w = queryFeatures.Shape[2];
        int numSupport = supportImages.Rank == 4 ? supportImages.Shape[0] : 1;
        int sC = supportImages.Shape[supportImages.Rank - 3];
        int sH = supportImages.Shape[supportImages.Rank - 2];
        int sW = supportImages.Shape[supportImages.Rank - 1];
        var prototype = new double[numC];
        for (int s = 0; s < numSupport; s++)
        {
            var sSlice = supportImages.Rank == 4 ? new Tensor<T>([sC, sH, sW]) : supportImages;
            if (supportImages.Rank == 4)
                for (int c = 0; c < sC; c++)
                    for (int y = 0; y < sH; y++)
                        for (int x = 0; x < sW; x++)
                            sSlice[c, y, x] = supportImages[s, c, y, x];
            var sFeatures = Predict(sSlice);
            int fC = sFeatures.Shape[0], fH = sFeatures.Shape[1], fW = sFeatures.Shape[2];
            var sMean = new double[numC];
            int maskCount = 0;
            for (int y = 0; y < fH; y++)
                for (int x = 0; x < fW; x++)
                {
                    double m = supportMasks.Rank >= 3 ? NumOps.ToDouble(supportMasks[s, y, x]) : NumOps.ToDouble(supportMasks[y, x]);
                    if (m >= 0.5)
                    {
                        maskCount++;
                        for (int c = 0; c < fC && c < numC; c++)
                            sMean[c] += NumOps.ToDouble(sFeatures[c, y, x]);
                    }
                }
            if (maskCount > 0)
                for (int c = 0; c < numC; c++)
                    prototype[c] += sMean[c] / maskCount;
        }
        if (numSupport > 1)
            for (int c = 0; c < numC; c++)
                prototype[c] /= numSupport;
        var scoreMap = Common.SegmentationTensorOps.WeightedChannelSum(queryFeatures, prototype);
        var probMap = Common.SegmentationTensorOps.Sigmoid(scoreMap);
        var labels = new Tensor<T>([h, w]);
        var probs = new Tensor<T>([2, h, w]);
        var structures = new List<SegmentedStructure>();
        int fgArea = 0; double confTotal = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double p = NumOps.ToDouble(probMap[y, x]);
                if (p >= 0.5) { labels[y, x] = NumOps.FromDouble(1); fgArea++; confTotal += p; }
                probs[0, y, x] = NumOps.FromDouble(1.0 - p);
                probs[1, y, x] = NumOps.FromDouble(p);
            }
        if (fgArea > 0)
            structures.Add(new SegmentedStructure { ClassId = 1, Name = "FewShot_Target", VolumeOrArea = fgArea, MeanConfidence = confTotal / fgArea });
        return new MedicalSegmentationResult<T> { Labels = labels, Probabilities = probs, Structures = structures };
    }
    #endregion
}
