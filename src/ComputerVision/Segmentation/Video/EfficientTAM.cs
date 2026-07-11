using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Video;

/// <summary>
/// EfficientTAM: Efficient Track Anything Model for edge video segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Edge-device video segmentation. Mobile video object tracking.
///
/// Common use cases:
/// - Edge-device video segmentation
/// - Mobile video object tracking
/// - Real-time interactive video editing
/// - Low-latency video analytics
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Lightweight encoder replacing SAM2 heavy backbone
/// - Efficient memory mechanism for temporal propagation
/// - Designed for mobile and edge deployment
/// - Compatible with SAM2 prompt interface
/// </para>
/// <para>
/// <b>Reference:</b> Xiong et al., "Efficient Track Anything", arXiv:2411.18933 (2024).
/// The image encoder is a plain, non-hierarchical ViT (ViT-Tiny/-Small, 16x16 patches) that
/// replaces SAM 2's hierarchical Hiera — the model's core efficiency contribution.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an EfficientTAM model for lightweight edge-device video segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 480, inputWidth: 480, inputDepth: 3, outputSize: 1);
/// var model = new EfficientTAM&lt;double&gt;(architecture, numClasses: 1);
///
/// // Or load a pre-trained ONNX model for mobile video tracking
/// var onnxModel = new EfficientTAM&lt;double&gt;(architecture, "efficienttam.onnx", numClasses: 1);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Tracking)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Efficient Track Anything", "https://arxiv.org/abs/2411.18933", Year = 2024, Authors = "Yunyang Xiong, Chong Zhou, Xiaoyu Xiang, Lemeng Wu, Chenchen Zhu, Zechun Liu, Saksham Suri, Balakrishnan Varadarajan, Ramya Akula, Forrest Iandola, Raghuraman Krishnamoorthi, Bilge Soran, Vikas Chandra")]
public class EfficientTAM<T> : NeuralNetworkBase<T>, IVideoSegmentation<T>
{
    private readonly EfficientTAMOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    private readonly int _height, _width, _channels, _numClasses;
    private readonly EfficientTAMModelSize _modelSize;
    // Paper-faithful plain-ViT image-encoder config (Xiong et al. 2024, arXiv 2411.18933):
    // ViT-Tiny/-Small with a 16x16 patch embed + pre-norm transformer blocks.
    private readonly int _embedDim;
    private readonly int _numEncoderLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _decoderDim;
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
    /// Gets whether this EfficientTAM instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns <c>true</c> in native mode, <c>false</c> in ONNX mode.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal EfficientTAMModelSize ModelSize => _modelSize;
    internal int NumClasses => _numClasses;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes EfficientTAM in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="modelSize">Model size variant (default: Tiny).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable EfficientTAM model.
    /// </para>
    /// </remarks>
    public EfficientTAM(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        EfficientTAMModelSize modelSize = EfficientTAMModelSize.Tiny, double dropRate = 0,
        EfficientTAMOptions? options = null)
        // Single-class mask (numClasses == 1): regress the raw mask logit against the target with MSE.
        // NOTE ON LOSS CHOICE: softmax CrossEntropyWithLogitsLoss is DEGENERATE for one class
        // (log_softmax over a single logit is identically 0 -> zero gradient). BinaryCrossEntropy-
        // WithLogitsLoss trains, but it drives logits toward +-inf, which AMPLIFIES a tiny CPU-vs-GPU
        // forward numerical difference in the plain-ViT encoder (patch-embed + attention + LayerNorm)
        // into a full training divergence on the CPU engine (loss climbs, logits explode) while the
        // GPU stays stable — see the tracked Tensors numerical-parity issue. MSE keeps logits bounded
        // (~[0,1]) so that numerical difference never amplifies: the ViT trains stably on BOTH engines
        // (verified CPU and GPU converge to ~identical loss). Multi-class masks keep softmax CE.
        : base(architecture, lossFunction ?? (numClasses <= 1
            ? new MeanSquaredErrorLoss<T>()
            : new CrossEntropyWithLogitsLoss<T>()))
    {
        _options = options ?? new EfficientTAMOptions(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses; _modelSize = modelSize; _dropRate = dropRate;
        _useNativeMode = true; _onnxModelPath = null;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        (_embedDim, _numEncoderLayers, _numHeads, _patchSize, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes EfficientTAM in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="modelSize">Model size for metadata (default: Tiny).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained EfficientTAM from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public EfficientTAM(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1, EfficientTAMModelSize modelSize = EfficientTAMModelSize.Tiny,
        EfficientTAMOptions? options = null)
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new EfficientTAMOptions(); Options = _options;
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"EfficientTAM ONNX model not found: {onnxModelPath}");
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses; _modelSize = modelSize; _dropRate = 0;
        _useNativeMode = false; _onnxModelPath = onnxModelPath; _optimizer = null;
        (_embedDim, _numEncoderLayers, _numHeads, _patchSize, _decoderDim) = GetModelConfig(modelSize);
        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load EfficientTAM ONNX model: {ex.Message}", ex); }
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
    /// Routes the training forward through the SAME <see cref="Forward"/> the inference path uses,
    /// so training adds the leading batch dim (the encoder's ConvolutionalLayers expect rank-4
    /// [B,C,H,W]) and runs the encoder→decoder chain identically. The base
    /// <see cref="NeuralNetworkBase{T}.ForwardForTraining"/> fed the raw rank-3 [C,H,W] test input
    /// straight to the layers, so the forward it recorded on the tape did not match the trained
    /// architecture and no gradient reached the parameters ("No parameters changed after
    /// training"). The batch reshape is tape-safe (Engine.Reshape), so the gradient flows end to
    /// end.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

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
            TrainWithTape(input, expectedOutput);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }
    #endregion

    #region Private Methods
    // Paper (Xiong et al. 2024, arXiv 2411.18933): EfficientTAM's image encoder is a plain,
    // non-hierarchical ViT — ViT-Tiny (embed 192, depth 12, 3 heads) or ViT-Small (embed 384,
    // depth 12, 6 heads) with 16x16 patches — replacing SAM 2's hierarchical Hiera. Returns
    // (embedDim, numEncoderLayers, numHeads, patchSize, decoderDim).
    private static (int EmbedDim, int NumLayers, int NumHeads, int PatchSize, int DecoderDim) GetModelConfig(EfficientTAMModelSize modelSize) => modelSize switch
    {
        EfficientTAMModelSize.Tiny => (192, 12, 3, 16, 256),   // ViT-Tiny
        EfficientTAMModelSize.Small => (384, 12, 6, 16, 256),  // ViT-Small
        _ => (192, 12, 3, 16, 256)
    };

    private Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        int batch = input.Shape[0], h = input.Shape[2], w = input.Shape[3];
        // ViT encoder: image [B, C, H, W] -> patch tokens [B, (H/P)*(W/P), embedDim].
        var tokens = input;
        for (int i = 0; i < _encoderLayerEnd; i++) tokens = Layers[i].Forward(tokens);
        // Reshape the token sequence back to a spatial feature map so the conv mask decoder can
        // run: [B, gh*gw, D] -> [B, gh, gw, D] -> [B, D, gh, gw]. Engine.Reshape/TensorPermute
        // keep this on the autodiff tape so gradients flow through to the ViT encoder.
        int gh = h / _patchSize, gw = w / _patchSize, d = tokens.Shape[tokens.Rank - 1];
        var grid = Engine.Reshape(tokens, new[] { batch, gh, gw, d });
        var features = Engine.TensorPermute(grid, new[] { 0, 3, 1, 2 });
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

    /// <summary>
    /// Captures per-layer activations. Overrides the base flat walk because EfficientTAM's forward is
    /// NOT a straight <c>Layers</c> chain: the plain-ViT encoder emits token sequences [B, N, embedDim]
    /// that must be reshaped back to a spatial map [B, embedDim, H/P, W/P] before the conv mask decoder.
    /// The base walk feeds the raw token tensor to the first decoder conv and throws "Expected input
    /// depth {embedDim}, but got {N}". Mirror <see cref="Forward"/> so the boundary reshape is applied.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode) return base.GetNamedLayerActivations(input);
        var activations = new Dictionary<string, Tensor<T>>();
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        int batch = input.Shape[0], h = input.Shape[2], w = input.Shape[3];
        var current = input;
        for (int i = 0; i < _encoderLayerEnd; i++) { current = Layers[i].Forward(current); activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone(); }
        int gh = h / _patchSize, gw = w / _patchSize, d = current.Shape[current.Rank - 1];
        current = Engine.TensorPermute(Engine.Reshape(current, new[] { batch, gh, gw, d }), new[] { 0, 3, 1, 2 });
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) { current = Layers[i].Forward(current); activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone(); }
        return activations;
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
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

    // Tape-safe add/remove of the leading batch dim: Engine.Reshape records the reshape on the
    // autodiff tape, so gradients flow through it. The prior raw `new Tensor<T>(...)` +
    // Data.Span.CopyTo SEVERED the tape — any training path that funnels through Forward would get
    // zero gradient upstream of the reshape.
    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
        => Engine.Reshape(tensor, new[] { 1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2] });

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int[] s = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < s.Length; i++) s[i] = tensor.Shape[i + 1];
        return Engine.Reshape(tensor, s);
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
            // Paper-faithful plain ViT image encoder (Xiong et al. 2024, arXiv 2411.18933):
            // patch-embed -> N PRE-NORM RESIDUAL transformer blocks, producing patch tokens
            // [B, (H/P)*(W/P), embedDim]. This is EfficientTAM's core contribution — a plain,
            // non-hierarchical ViT replacing SAM 2's Hiera. Built from TransformerEncoderLayer
            // (each block is x = x + MHA(LN(x)); x = x + FFN(LN(x)), Dosovitskiy et al. 2021 §3.1):
            //   - the previous hierarchical Conv-BN-ReLU CNN diverged from the paper AND dead-ReLU'd
            //     constant-input signal to zero (DifferentInputs collapse), and
            //   - a NON-residual flat ViT (LayerHelper.CreateDefaultViTLayers) exploded to a 2800x
            //     loss within a few steps — a deep transformer has no gradient highway without the
            //     residual skips, so the standard residual block is both correct AND trainable.
            var encoderLayers = new List<ILayer<T>> { new PatchEmbeddingLayer<T>(_patchSize, _embedDim) };
            for (int i = 0; i < _numEncoderLayers; i++)
                encoderLayers.Add(new TransformerEncoderLayer<T>(_numHeads, _embedDim * 4, _embedDim));
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            // Mask decoder: the encoder's token grid is reshaped back to a spatial feature map
            // [B, embedDim, H/P, W/P] in Forward, then this lightweight conv head produces the
            // per-pixel class logits.
            int fH = _height / _patchSize, fW = _width / _patchSize;
            var decoderLayers = LayerHelper<T>.CreateEfficientTAMDecoderLayers(_embedDim, _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "EfficientTAM" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
    { writer.Write(_height); writer.Write(_width); writer.Write(_channels); writer.Write(_numClasses); writer.Write((int)_modelSize); writer.Write(_decoderDim); writer.Write(_dropRate); writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty); writer.Write(_encoderLayerEnd); writer.Write(_embedDim); writer.Write(_numEncoderLayers); writer.Write(_numHeads); writer.Write(_patchSize); }

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
    { _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadDouble(); _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); }

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
        ? new EfficientTAM<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new EfficientTAM<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, _options);

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

    #region IVideoSegmentation Implementation
    private Tensor<T>? _trackingFeatures;
    private Tensor<T>? _trackingMasks;
    private int[]? _trackedObjectIds;
    private int _frameIndex;
    private readonly Dictionary<int, Tensor<T>> _corrections = [];
    int ISegmentationModel<T>.NumClasses => _numClasses;
    int ISegmentationModel<T>.InputHeight => _height;
    int ISegmentationModel<T>.InputWidth => _width;
    bool ISegmentationModel<T>.IsOnnxMode => !_useNativeMode;
    Tensor<T> ISegmentationModel<T>.Segment(Tensor<T> image) => Predict(image);
    int IVideoSegmentation<T>.MaxTrackedObjects => 64;
    bool IVideoSegmentation<T>.SupportsStreaming => true;
    void IVideoSegmentation<T>.InitializeTracking(Tensor<T> frame, Tensor<T> masks, int[]? objectIds)
    {
        _trackingFeatures = Common.SegmentationTensorOps.EnsureUnbatched(Predict(frame));
        _trackingMasks = masks;
        int numObj = masks.Rank >= 3 ? masks.Shape[0] : 1;
        _trackedObjectIds = objectIds ?? Enumerable.Range(1, numObj).ToArray();
        _frameIndex = 0;
        _corrections.Clear();
    }
    VideoSegmentationResult<T> IVideoSegmentation<T>.PropagateToFrame(Tensor<T> frame)
    {
        _frameIndex++;
        var currentFeatures = Common.SegmentationTensorOps.EnsureUnbatched(Predict(frame));
        int h = currentFeatures.Shape[1], w = currentFeatures.Shape[2];
        var ids = _trackedObjectIds ?? [1];
        int numObj = ids.Length;
        Tensor<T> masks;
        if (_trackingFeatures != null && _trackingMasks != null && _trackingMasks.Rank == 3)
        {
            var affinity = Common.SegmentationTensorOps.PixelAffinity(_trackingFeatures, currentFeatures);
            masks = Common.SegmentationTensorOps.WarpMasksByAffinity(_trackingMasks, affinity);
        }
        else
        {
            masks = new Tensor<T>([numObj, h, w]);
        }
        foreach (var kvp in _corrections)
        {
            int idx = Array.IndexOf(ids, kvp.Key);
            if (idx >= 0)
            {
                int mH = Math.Min(kvp.Value.Shape[0], h), mW = Math.Min(kvp.Value.Shape[1], w);
                for (int y = 0; y < mH; y++)
                    for (int x = 0; x < mW; x++)
                        masks[idx, y, x] = kvp.Value[y, x];
            }
        }
        _corrections.Clear();
        var confidences = new double[numObj];
        var isVisible = new bool[numObj];
        for (int obj = 0; obj < numObj; obj++)
        {
            int area = 0; double confSum = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double v = NumOps.ToDouble(masks[obj, y, x]);
                    if (v >= 0.5) { area++; confSum += v; }
                }
            confidences[obj] = area > 0 ? confSum / area : 0.0;
            isVisible[obj] = area >= 4;
        }
        _trackingFeatures = currentFeatures;
        _trackingMasks = masks;
        return new VideoSegmentationResult<T>
        {
            Masks = masks, ObjectIds = ids, Confidences = confidences,
            FrameIndex = _frameIndex, IsVisible = isVisible
        };
    }
    void IVideoSegmentation<T>.AddCorrection(int objectId, Tensor<T> correctionMask)
    {
        _corrections[objectId] = correctionMask;
    }
    void IVideoSegmentation<T>.ResetTracking()
    {
        _trackingFeatures = null; _trackingMasks = null; _trackedObjectIds = null;
        _frameIndex = 0; _corrections.Clear();
    }
    #endregion
}
