using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.OpenVocabulary;

/// <summary>
/// Grounded SAM 2: Text-grounded tracking and segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Text-grounded video segmentation. Open-world object tracking.
///
/// Common use cases:
/// - Text-grounded video segmentation
/// - Open-world object tracking
/// - Natural language video search
/// - Automatic annotation from text descriptions
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Grounding DINO for text-to-box detection + SAM 2 for segmentation
/// - Combines open-set detection with promptable segmentation
/// - Video tracking with text-specified targets
/// - Hiera backbone with memory attention for temporal consistency
/// </para>
/// <para>
/// <b>Reference:</b> Ren et al., "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks", arXiv 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Grounded SAM 2 model for text-grounded video segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var model = new GroundedSAM2&lt;double&gt;(architecture, numClasses: 1);
///
/// // Or load a pre-trained ONNX model for open-world object tracking
/// var onnxModel = new GroundedSAM2&lt;double&gt;(architecture, "groundedsam2.onnx", numClasses: 1);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Tracking)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks", "https://arxiv.org/abs/2401.14159", Year = 2024, Authors = "Ren et al.")]
public class GroundedSAM2<T> : NeuralNetworkBase<T>, IOpenVocabSegmentation<T>
{
    private readonly GroundedSAM2Options _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    private readonly int _height, _width, _channels, _numClasses;
    private readonly int _decoderDim;
    private readonly double _dropRate;
    private readonly bool _useNativeMode;
    private readonly string? _onnxModelPath;
    private InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;

    // Native image->mask pipeline configuration (paper-backed, sourced from GroundedSAM2Options).
    private readonly int _patchSize;
    private readonly int _visionDim;
    private readonly int _numEncoderLayers;
    private readonly int _numDecoderLayers;
    private readonly int _numHeads;

    // Layer-index boundaries into the flat Layers list for the structured Forward pass:
    //   [0]                          : patch-embedding conv -> [B, visionDim, H/p, W/p]
    //   [1 .. _maskConvIndex)        : positional encoding + encoder/decoder transformer blocks (token space)
    //   [_maskConvIndex]             : 1x1 conv -> numClasses mask logits at token-grid resolution
    //   [_upsampleIndex]             : upsample by patchSize -> full input resolution
    private int _encoderLayerEnd;    // encoder boundary, retained for serialization round-trip compatibility
    private int _maskConvIndex;
    private int _upsampleIndex;
    private bool _customLayers;
    private bool _lazyShapesWarmed;
    #endregion

    #region Properties
    /// <summary>
    /// Gets whether this GroundedSAM2 instance supports training.
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
    /// Initializes GroundedSAM2 in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyWithLogitsLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable GroundedSAM2 model.
    /// </para>
    /// </remarks>
    public GroundedSAM2(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 1,
        double dropRate = 0,
        GroundedSAM2Options? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new GroundedSAM2Options(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses; _dropRate = dropRate;
        _useNativeMode = true; _onnxModelPath = null;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _decoderDim = _options.DecoderDim > 0 ? _options.DecoderDim : 256;
        _visionDim = _options.VisionDim > 0 ? _options.VisionDim : 256;
        _numEncoderLayers = _options.NumVisionLayers > 0 ? _options.NumVisionLayers : 6;
        _numDecoderLayers = _options.NumDecoderLayers > 0 ? _options.NumDecoderLayers : 6;
        _numHeads = _options.NumHeads > 0 ? _options.NumHeads : 8;
        _patchSize = _options.PatchSize > 0 ? _options.PatchSize : 16;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes GroundedSAM2 in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained GroundedSAM2 from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public GroundedSAM2(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 1,
        GroundedSAM2Options? options = null)
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new GroundedSAM2Options(); Options = _options;
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"GroundedSAM2 ONNX model not found: {onnxModelPath}");
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses; _dropRate = 0;
        _useNativeMode = false; _onnxModelPath = onnxModelPath; _optimizer = null;
        _decoderDim = _options.DecoderDim > 0 ? _options.DecoderDim : 256;
        _visionDim = _options.VisionDim > 0 ? _options.VisionDim : 256;
        _numEncoderLayers = _options.NumVisionLayers > 0 ? _options.NumVisionLayers : 6;
        _numDecoderLayers = _options.NumDecoderLayers > 0 ? _options.NumDecoderLayers : 6;
        _numHeads = _options.NumHeads > 0 ? _options.NumHeads : 8;
        _patchSize = _options.PatchSize > 0 ? _options.PatchSize : 16;
        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load GroundedSAM2 ONNX model: {ex.Message}", ex); }
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
    private Tensor<T> Forward(Tensor<T> input)
    {
        // Native GroundedSAM2 is an image -> pixel-mask model: it takes an RGB image [C,H,W] / [B,C,H,W]
        // and returns per-pixel class logits [B, numClasses, H, W] (matching the ONNX contract). The image
        // is tokenized by a patch-embedding conv, refined by the encoder/decoder transformer stack in token
        // space, then projected back to a spatial mask and upsampled to the input resolution. The previous
        // body forwarded the raw input straight into the token transformers, which required a pre-tokenized
        // [B, tokens, dim] tensor and never produced pixel-level masks — inconsistent with ONNX mode.
        if (input.Rank != 3 && input.Rank != 4)
            throw new ArgumentException(
                "GroundedSAM2 expects an image tensor [C, H, W] or [B, C, H, W].", nameof(input));

        bool unbatched = input.Rank == 3;
        var x = unbatched ? AddBatchDimension(input) : input;   // [B, C, H, W]

        // A user-supplied custom layer stack is run straight through without the structured reshapes.
        if (_customLayers)
        {
            var custom = x;
            foreach (var layer in Layers) custom = layer.Forward(custom);
            return unbatched ? RemoveBatchDimension(custom) : custom;
        }

        // 1. Patch embedding -> [B, visionDim, gridH, gridW].
        var feat = Layers[0].Forward(x);
        int b = feat.Shape[0], d = feat.Shape[1], gh = feat.Shape[2], gw = feat.Shape[3];

        // 2. Flatten the spatial grid to a token sequence [B, gridH*gridW, visionDim] (tape-aware).
        var tokens = Engine.Reshape(Engine.TensorPermute(feat, new[] { 0, 2, 3, 1 }), new[] { b, gh * gw, d });

        // 3. Positional encoding (Layers[1]) + encoder + decoder transformer blocks, all in token space
        //    (everything between the patch conv and the mask head).
        for (int i = 1; i < _maskConvIndex; i++)
            tokens = Layers[i].Forward(tokens);

        // 4. Reshape the refined tokens back to a spatial feature map [B, visionDim, gridH, gridW].
        var spatial = Engine.TensorPermute(Engine.Reshape(tokens, new[] { b, gh, gw, d }), new[] { 0, 3, 1, 2 });

        // 5. Mask head: 1x1 conv -> numClasses logits at grid resolution, then upsample by patchSize to the
        //    full input resolution.
        var maskLogits = Layers[_maskConvIndex].Forward(spatial);   // [B, numClasses, gridH, gridW]
        maskLogits = Layers[_upsampleIndex].Forward(maskLogits);    // [B, numClasses, H, W]

        return unbatched ? RemoveBatchDimension(maskLogits) : maskLogits;
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
        {
            Layers.AddRange(Architecture.Layers);
            _customLayers = true;
            return;
        }

        // Grounded SAM = Grounding DINO + SAM (both transformer). The default image->mask pipeline
        // (patch-embedding tokenizer -> positional encoding -> ViT encoder + mask-decoder transformer
        // stack -> mask head + upsample) is built by LayerHelper, exactly like every other model's default
        // layers; all fixed dimensions, head counts and depths come from GroundedSAM2Options.
        Layers.AddRange(LayerHelper<T>.CreateGroundedSAM2Layers(
            visionDim: _visionDim,
            numHeads: _numHeads,
            numEncoderLayers: _numEncoderLayers,
            numDecoderLayers: _numDecoderLayers,
            patchSize: _patchSize,
            numClasses: _numClasses,
            imageHeight: _height,
            imageWidth: _width,
            dropRate: _dropRate));

        // The mask head is always the last two layers (mask conv + upsample); everything between the patch
        // conv (index 0) and it runs in token space. Indexing the head from the end keeps this correct
        // regardless of how many dropout layers the drop-rate inserted among the transformer blocks.
        _encoderLayerEnd = 2 + _numEncoderLayers;   // encoder boundary in the no-dropout default (serialized)
        _upsampleIndex = Layers.Count - 1;
        _maskConvIndex = Layers.Count - 2;
    }

    /// <summary>
    /// Resolves every lazy layer's shape by running ONE dummy image through the custom <see cref="Forward"/>.
    /// GroundedSAM2's forward is not a plain sequential walk of <c>Layers</c> — it reshapes between the
    /// patch-embedding conv (spatial) and the transformer stack (token space) and back before the mask
    /// head. The base per-layer shape inference walks the layers in order and would feed the conv's spatial
    /// output straight into the first transformer ("embedding dimension (2) does not match weight dimension
    /// (256)"). Running one eval-mode warm forward at the real [C, H, W] page shape materializes every lazy
    /// weight through the correct reshape path (mirrors UDOP's override).
    /// </summary>
    protected override void ResolveLazyLayerShapes()
    {
        if (_lazyShapesWarmed) return;
        _lazyShapesWarmed = true;
        if (!_useNativeMode || _customLayers) return;

        var dummy = new Tensor<T>([_channels, _height, _width]);
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try { _ = Forward(dummy); }
        catch { /* best-effort; a real forward failure surfaces on the actual Train/Predict */ }
        finally { if (wasTraining) SetTrainingMode(true); }
    }

    /// <summary>
    /// Training forward. Routes through the SAME custom <see cref="Forward"/> as inference (patch embed ->
    /// token-space transformer stack -> mask head, with the spatial&lt;-&gt;token reshapes), NOT the base
    /// sequential walk over <c>Layers</c>. The base walk would feed the patch conv's spatial output
    /// straight into the first transformer and throw an embedding-dimension mismatch; sharing Forward keeps
    /// the train and inference graphs identical and every op tape-tracked.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        EnsureLayerRandomSeedsWired();
        return Forward(input);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "GroundedSAM2" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
    { writer.Write(_height); writer.Write(_width); writer.Write(_channels); writer.Write(_numClasses); writer.Write(_decoderDim); writer.Write(_dropRate); writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty); writer.Write(_encoderLayerEnd); }

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
    { _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadDouble(); _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32(); }

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
        ? new GroundedSAM2<T>(Architecture, _optimizer, LossFunction, _numClasses, _dropRate, _options)
        : new GroundedSAM2<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _options);

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

    #region IOpenVocabSegmentation Implementation
    int ISegmentationModel<T>.NumClasses => _numClasses;
    int ISegmentationModel<T>.InputHeight => _height;
    int ISegmentationModel<T>.InputWidth => _width;
    bool ISegmentationModel<T>.IsOnnxMode => !_useNativeMode;
    Tensor<T> ISegmentationModel<T>.Segment(Tensor<T> image) => Predict(image);
    int IOpenVocabSegmentation<T>.MaxCategories => 256;
    int IOpenVocabSegmentation<T>.MaxPromptLength => 77;

    OpenVocabSegmentationResult<T> IOpenVocabSegmentation<T>.SegmentWithText(Tensor<T> image, IReadOnlyList<string> classNames)
    {
        var logits = Common.SegmentationTensorOps.EnsureUnbatched(Predict(image));
        int numC = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2];
        int numText = classNames.Count;
        var masks = new Tensor<T>([numText, h, w]);
        var scores = new double[numText];
        var semanticMap = new Tensor<T>([h, w]);
        var textProbs = new double[numText][];
        for (int t = 0; t < numText; t++)
        {
            var weights = Common.SegmentationTensorOps.TextToWeights(classNames[t], numC);
            var scoreMap = Common.SegmentationTensorOps.WeightedChannelSum(logits, weights);
            var probMap = Common.SegmentationTensorOps.Sigmoid(scoreMap);
            double area = 0, confSum = 0;
            textProbs[t] = new double[h * w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double v = NumOps.ToDouble(probMap[y, x]);
                    textProbs[t][y * w + x] = v;
                    if (v >= 0.5) { masks[t, y, x] = NumOps.FromDouble(1.0); area++; confSum += v; }
                }
            scores[t] = area > 0 ? confSum / area : 0;
        }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int best = 0; double bestV = -1;
                for (int t = 0; t < numText; t++) { double v = textProbs[t][y * w + x]; if (v > bestV) { bestV = v; best = t; } }
                semanticMap[y, x] = NumOps.FromDouble(best);
            }
        return new OpenVocabSegmentationResult<T> { Masks = masks, ClassNames = classNames.ToArray(), Scores = scores, SemanticMap = semanticMap };
    }

    OpenVocabSegmentationResult<T> IOpenVocabSegmentation<T>.SegmentWithPrompt(Tensor<T> image, string prompt)
        => ((IOpenVocabSegmentation<T>)this).SegmentWithText(image, new[] { prompt });
    #endregion
}
