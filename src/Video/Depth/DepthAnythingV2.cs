using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Depth;

/// <summary>
/// Depth Anything V2 for monocular depth estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Depth Anything V2 is a state-of-the-art model for estimating depth maps
/// from single images (monocular depth estimation). Given an RGB image, it predicts the relative
/// distance of each pixel from the camera. This is useful for:
/// - 3D scene understanding
/// - Augmented reality applications
/// - Autonomous driving
/// - Video editing and VFX
/// - Object detection and segmentation
///
/// Unlike stereo depth estimation which requires two cameras, Depth Anything works with
/// a single image by learning depth cues from large-scale training data.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Vision Transformer (ViT) based encoder with DINOv2 initialization
/// - Efficient multi-scale decoder for dense prediction
/// - Scale-invariant depth loss for robust training
/// - Supports various backbone sizes (Small, Base, Large)
/// </para>
/// <para>
/// <b>Reference:</b> Yang et al., "Depth Anything V2" 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Depth Anything V2 model for monocular depth estimation
/// var depthModel = new DepthAnythingV2&lt;double&gt;();
///
/// // Or configure with a specific backbone size
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 1);
/// var model = new DepthAnythingV2&lt;double&gt;(architecture, modelSize: ModelSize.Large);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Depth Anything V2",
    "https://arxiv.org/abs/2406.09414",
    Year = 2024,
    Authors = "Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao")]
public class DepthAnythingV2<T> : NeuralNetworkBase<T>
{
    private readonly DepthAnythingV2Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Enums

    /// <summary>
    /// Model size variants for Depth Anything V2.
    /// </summary>
    public enum ModelSize
    {
        /// <summary>Small model (faster, less accurate)</summary>
        Small,
        /// <summary>Base model (balanced)</summary>
        Base,
        /// <summary>Large model (slower, more accurate)</summary>
        Large
    }

    #endregion

    #region Fields

    private int _height;
    private int _width;
    private int _channels;
    private ModelSize _modelSize;
    private int _numFeatures;
    private int _patchSize;
    private int _numEncoderBlocks;
    private bool _useNativeMode;
    private string? _onnxModelPath;
    private readonly InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether training is supported.
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the input height for frames.
    /// </summary>
    internal int InputHeight => _height;

    /// <summary>
    /// Gets the input width for frames.
    /// </summary>
    internal int InputWidth => _width;

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    internal int InputChannels => _channels;

    /// <summary>
    /// Gets the model size variant.
    /// </summary>
    internal ModelSize Size => _modelSize;

    /// <summary>
    /// Gets whether using native mode (trainable) or ONNX mode (inference only).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the DepthAnythingV2 class in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function (default: ScaleInvariantDepthLoss).</param>
    /// <param name="modelSize">The model size variant.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a trainable Depth Anything V2 model.
    /// Use this when you want to train or fine-tune the model on your own depth data.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public DepthAnythingV2()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputHeight: 256, inputWidth: 256, inputDepth: 3,
            outputSize: 1))
    {
    }

    public DepthAnythingV2(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        ModelSize modelSize = ModelSize.Base,
        DepthAnythingV2Options? options = null)
        : base(architecture, lossFunction ?? new ScaleInvariantDepthLoss<T>())
    {
        _options = options ?? new DepthAnythingV2Options();
        Options = _options;

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _modelSize = modelSize;
        _patchSize = 16;
        _useNativeMode = true;
        _onnxModelPath = null;
        _optimizer = optimizer;

        _numFeatures = modelSize switch
        {
            ModelSize.Small => 384,
            ModelSize.Base => 768,
            ModelSize.Large => 1024,
            _ => 768
        };

        _numEncoderBlocks = modelSize switch
        {
            ModelSize.Small => 12,
            ModelSize.Base => 12,
            ModelSize.Large => 24,
            _ => 12
        };

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the DepthAnythingV2 class in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="modelSize">The model size variant for configuration.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor loads a pre-trained Depth Anything V2 model from ONNX format.
    /// Use this for fast inference when you don't need to train the model.
    /// </para>
    /// </remarks>
    public DepthAnythingV2(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ModelSize modelSize = ModelSize.Base,
        DepthAnythingV2Options? options = null)
        : base(architecture, new ScaleInvariantDepthLoss<T>())
    {
        _options = options ?? new DepthAnythingV2Options();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"Depth Anything V2 ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _modelSize = modelSize;
        _patchSize = 16;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        _numFeatures = modelSize switch
        {
            ModelSize.Small => 384,
            ModelSize.Base => 768,
            ModelSize.Large => 1024,
            _ => 768
        };

        _numEncoderBlocks = modelSize switch
        {
            ModelSize.Small => 12,
            ModelSize.Base => 12,
            ModelSize.Large => 24,
            _ => 12
        };

        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load Depth Anything V2 ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Estimates depth from an RGB image.
    /// </summary>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Depth map tensor [H, W] or [B, 1, H, W] with relative depth values.</returns>
    public Tensor<T> EstimateDepth(Tensor<T> image)
    {
        bool hasBatch = image.Rank == 4;
        if (!hasBatch)
        {
            image = AddBatchDimension(image);
        }

        Tensor<T> depth;
        if (_useNativeMode)
        {
            depth = RunForward(image);
        }
        else
        {
            depth = PredictOnnx(image);
        }

        if (!hasBatch)
        {
            depth = RemoveBatchDimension(depth);
        }

        return depth;
    }

    /// <summary>
    /// Performs ONNX inference for depth estimation.
    /// </summary>
    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        // Convert input tensor to float array
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        // Create ONNX input tensor
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };

        // Run inference
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Convert output to Tensor<T>
        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <summary>
    /// Estimates depth for a sequence of video frames.
    /// </summary>
    /// <param name="frames">List of video frames.</param>
    /// <returns>List of depth maps for each frame.</returns>
    public List<Tensor<T>> EstimateVideoDepth(List<Tensor<T>> frames)
    {
        var depths = new List<Tensor<T>>();
        foreach (var frame in frames)
        {
            depths.Add(EstimateDepth(frame));
        }
        return depths;
    }

    /// <summary>
    /// Gets the relative depth value at a specific point.
    /// </summary>
    /// <param name="depthMap">The depth map tensor.</param>
    /// <param name="x">X coordinate.</param>
    /// <param name="y">Y coordinate.</param>
    /// <returns>Relative depth value at the specified point.</returns>
    public double GetDepthAtPoint(Tensor<T> depthMap, int x, int y)
    {
        int h = depthMap.Shape[^2];
        int w = depthMap.Shape[^1];

        if (x < 0 || x >= w || y < 0 || y >= h)
        {
            throw new ArgumentOutOfRangeException($"Point ({x}, {y}) is outside depth map bounds ({w}x{h})");
        }

        // Handle both [H, W] and [B, 1, H, W] formats
        if (depthMap.Rank == 2)
        {
            return Convert.ToDouble(depthMap[y, x]);
        }
        else if (depthMap.Rank == 4)
        {
            return Convert.ToDouble(depthMap[0, 0, y, x]);
        }
        else
        {
            return Convert.ToDouble(depthMap[0, y, x]);
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return EstimateDepth(input);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        // Route training through the SAME EstimateDepth path as inference so the two
        // cannot diverge (the base ForwardForTraining iterates the flat Layers list,
        // which would feed the encoder's rank-3 token output straight into the decoder's
        // rank-4 conv and skip the DPT reassemble). EstimateDepth adds/removes the batch
        // dim and calls the tape-aware RunForward, so the GradientTape captures the full
        // ViT encoder + DPT decoder and gradients reach every parameter.
        return EstimateDepth(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use native mode constructor for training.");
        }

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

    // Runs the full DINOv2 ViT encoder + DPT decoder in a single tape-aware pass so
    // inference (PredictCore) and training (ForwardForTraining) share EXACTLY the same
    // forward — there is no Predict-vs-train divergence. Every step is a layer.Forward
    // or a tape-aware Engine op (Reshape / TensorPermute), so gradients flow end-to-end.
    // Expects a batched image [B, C, H, W] and returns a depth map [B, 1, H, W].
    private Tensor<T> RunForward(Tensor<T> image)
    {
        // The token grid is derived from the ACTUAL input spatial dims, not the model's
        // configured _height/_width: the test harness runs at a reduced resolution
        // (e.g. 32x32), so the encoder produces (H/P)*(W/P) tokens for THIS input and the
        // reassemble must match. The DPT decoder upsamples by exactly the patch factor
        // (four 2x upsamples = 16x = the patch size), so the output resolution tracks the
        // input at any size.
        int inputH = image.Shape[image.Rank - 2];
        int inputW = image.Shape[image.Rank - 1];
        int gridH = inputH / _patchSize;
        int gridW = inputW / _patchSize;
        var tokens = EncodeImage(image);                        // [B, N, numFeatures]
        var featureMap = ReassembleTokensToMap(tokens, gridH, gridW); // [B, numFeatures, gridH, gridW]
        return DecodeDepth(featureMap);                         // [B, 1, H, W]
    }

    // Encoder: patch-embed (Layers[0]) -> transformer blocks -> final LayerNorm, returning
    // the token sequence [B, N, numFeatures]. Activations live INSIDE the layers
    // (TransformerEncoderLayer's GELU MLP), so there is NO manual activation here — the
    // previous ApplyGELU pass double-activated the conv layers AND severed the tape
    // (Tensor.Transform is a scalar loop the autodiff graph can't see through).
    private Tensor<T> EncodeImage(Tensor<T> image)
    {
        var features = image;
        int encoderEndIdx = _numEncoderBlocks + 1; // index of the encoder's final LayerNorm
        for (int i = 0; i <= encoderEndIdx && i < Layers.Count; i++)
        {
            features = Layers[i].Forward(features);
        }
        return features;
    }

    // DPT "reassemble": turn the encoder token sequence [B, N, numFeatures] back into a
    // 2-D feature map [B, numFeatures, gridH, gridW] (N == gridH*gridW; PatchEmbeddingLayer
    // adds no CLS token). Uses tape-aware Engine.Reshape + TensorPermute so the operation
    // is differentiable.
    private Tensor<T> ReassembleTokensToMap(Tensor<T> tokens, int gridH, int gridW)
    {
        int batch = tokens.Shape[0];
        int embed = tokens.Shape[tokens.Rank - 1];
        // Token count actually produced by the patch embedding.
        int n = tokens.Shape[1];
        if (gridH <= 0 || gridW <= 0 || gridH * gridW != n)
        {
            // Fall back to a square grid inferred from the token count (covers any
            // padding the PatchEmbeddingLayer applied to make the image patch-divisible).
            int side = System.Math.Max(1, (int)System.Math.Round(System.Math.Sqrt(n)));
            gridH = side;
            gridW = System.Math.Max(1, n / side);
        }
        // [B, N, D] -> [B, gridH, gridW, D] -> [B, D, gridH, gridW]
        var grid = Engine.Reshape(tokens, new[] { batch, gridH, gridW, embed });
        return Engine.TensorPermute(grid, new[] { 0, 3, 1, 2 });
    }

    // Decoder: run the DPT conv/upsample stages on the reassembled feature map. The
    // 1-channel sigmoid depth head is the last layer, so the output is already in [0, 1] —
    // no manual sigmoid (which would sever the tape) is applied. Upsampling is done only by
    // the UpsamplingLayers in the stack; the previous manual Upsample2x between layers
    // double-upsampled and threw IndexOutOfRange in CpuEngine.Upsample.
    private Tensor<T> DecodeDepth(Tensor<T> featureMap)
    {
        var decoded = featureMap;
        int decoderStartIdx = _numEncoderBlocks + 2; // first layer after the encoder LayerNorm
        for (int i = decoderStartIdx; i < Layers.Count; i++)
        {
            decoded = Layers[i].Forward(decoded);
        }
        return decoded;
    }

    // Tape-aware batch add/remove. These MUST go through Engine.Reshape (not a raw
    // new Tensor + Data.CopyTo, which severs the autodiff tape): RemoveBatchDimension is
    // the LAST op of the training forward, so a severed tape there would stop the loss
    // gradient from ever reaching the ViT/DPT parameters (zero-grad training collapse).
    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        int c = tensor.Shape[0];
        int h = tensor.Shape[1];
        int w = tensor.Shape[2];
        return Engine.Reshape(tensor, new[] { 1, c, h, w });
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int c = tensor.Shape[1];
        int h = tensor.Shape[2];
        int w = tensor.Shape[3];
        return Engine.Reshape(tensor, new[] { c, h, w });
    }

    #endregion

    #region Abstract Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            ClearLayers();
            return;
        }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultDepthAnythingV2Layers(
                _channels,
                _height,
                _width,
                _numFeatures,
                _numEncoderBlocks));
        }

        // EncodeImage / DecodeDepth split Layers positionally as
        //   [0] = patch embed, [1 .. _numEncoderBlocks] = ViT encoder blocks,
        //   [_numEncoderBlocks + 1] = encoder LayerNorm, [_numEncoderBlocks + 2 ..] = DPT decoder.
        // InitializeLayers also accepts a caller-supplied Architecture.Layers, which never has to
        // honor that contract. Validate it here so a reordered / custom layer list fails loudly
        // instead of silently routing the wrong layers through the encoder/decoder split.
        ValidateLayerContract();
    }

    /// <summary>
    /// Verifies that <see cref="Layers"/> matches the positional contract the
    /// <c>EncodeImage</c> / <c>DecodeDepth</c> split depends on. Throws if the count is too small or
    /// the layer types at the encoder indices are wrong (e.g. a custom <c>Architecture.Layers</c>).
    /// </summary>
    private void ValidateLayerContract()
    {
        int encoderLnIdx = _numEncoderBlocks + 1;
        int minLayers = encoderLnIdx + 2; // patch embed + encoder blocks + LayerNorm + >= 1 decoder layer
        if (Layers.Count < minLayers)
        {
            throw new InvalidOperationException(
                $"DepthAnythingV2 requires at least {minLayers} layers in the order " +
                $"[patch embed, {_numEncoderBlocks} encoder blocks, LayerNorm, decoder...], but got " +
                $"{Layers.Count}. The Encode/Decode split relies on this ordering.");
        }
        if (Layers[0] is not AiDotNet.NeuralNetworks.Layers.PatchEmbeddingLayer<T>)
        {
            throw new InvalidOperationException(
                $"DepthAnythingV2 layer[0] must be a PatchEmbeddingLayer<T> (the ViT patch embedding), " +
                $"but was {Layers[0].GetType().Name}.");
        }
        for (int i = 1; i <= _numEncoderBlocks; i++)
        {
            if (Layers[i] is not AiDotNet.NeuralNetworks.Layers.TransformerEncoderLayer<T>)
            {
                throw new InvalidOperationException(
                    $"DepthAnythingV2 layer[{i}] must be a TransformerEncoderLayer<T> (ViT encoder block), " +
                    $"but was {Layers[i].GetType().Name}.");
            }
        }
        if (Layers[encoderLnIdx] is not AiDotNet.NeuralNetworks.Layers.LayerNormalizationLayer<T>)
        {
            throw new InvalidOperationException(
                $"DepthAnythingV2 layer[{encoderLnIdx}] must be a LayerNormalizationLayer<T> " +
                $"(the encoder output norm), but was {Layers[encoderLnIdx].GetType().Name}.");
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (offset + layerParamCount <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.UpdateParameters(newParams);
                offset += layerParamCount;
            }
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "DepthAnythingV2" },
            { "Description", "Monocular Depth Estimation" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "InputChannels", _channels },
            { "ModelSize", _modelSize.ToString() },
            { "NumFeatures", _numFeatures },
            { "PatchSize", _patchSize },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count }
        };

        return new ModelMetadata<T>
        {
            AdditionalInfo = additionalInfo,
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write((int)_modelSize);
        writer.Write(_useNativeMode);
        writer.Write(_onnxModelPath ?? string.Empty);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _height = reader.ReadInt32();
        _width = reader.ReadInt32();
        _channels = reader.ReadInt32();
        _modelSize = (ModelSize)reader.ReadInt32();
        _useNativeMode = reader.ReadBoolean();
        _onnxModelPath = reader.ReadString();
        if (string.IsNullOrEmpty(_onnxModelPath)) _onnxModelPath = null;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new DepthAnythingV2<T>(Architecture, _optimizer, LossFunction, _modelSize);
        }
        else
        {
            return new DepthAnythingV2<T>(Architecture, _onnxModelPath!, _modelSize);
        }
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes of managed resources, including the ONNX inference session.
    /// </summary>
    /// <param name="disposing">True if disposing, false if finalizing.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _onnxSession?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
