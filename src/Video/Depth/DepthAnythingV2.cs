using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
            var features = EncodeImage(image);
            depth = DecodeDepth(features);
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
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return EstimateDepth(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use native mode constructor for training.");
        }

        var predicted = Predict(input);
        var lossGradient = predicted.Transform((v, idx) =>
            NumOps.Subtract(v, expectedOutput.Data.Span[idx]));

        BackwardPass(lossGradient);

        if (_optimizer != null)
        {
            _optimizer.UpdateParameters(Layers);
        }
    }

    #endregion

    #region Private Methods

    private Tensor<T> EncodeImage(Tensor<T> image)
    {
        // Apply patch embedding (layer 0)
        var features = Layers[0].Forward(image);
        features = ApplyGELU(features);

        // Apply encoder blocks (layers 1 to numEncoderBlocks)
        int encoderEndIdx = 1 + _numEncoderBlocks;
        for (int i = 1; i < Math.Min(encoderEndIdx, Layers.Count); i++)
        {
            features = Layers[i].Forward(features);
            features = ApplyGELU(features);
        }

        return features;
    }

    private Tensor<T> DecodeDepth(Tensor<T> features)
    {
        // Apply decoder blocks
        var decoded = features;
        int decoderStartIdx = 1 + _numEncoderBlocks;

        for (int i = decoderStartIdx; i < Layers.Count; i++)
        {
            // Upsample between decoder stages
            if (i > decoderStartIdx)
            {
                decoded = Upsample2x(decoded);
            }
            decoded = Layers[i].Forward(decoded);
            decoded = ApplyGELU(decoded);
        }

        // Final upsample to match input resolution
        while (decoded.Shape[2] < _height || decoded.Shape[3] < _width)
        {
            decoded = Upsample2x(decoded);
        }

        // Apply sigmoid to normalize depth to [0, 1]
        decoded = ApplySigmoid(decoded);

        return decoded;
    }

    private Tensor<T> Upsample2x(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];
        int outHeight = height * 2;
        int outWidth = width * 2;

        var output = new Tensor<T>([batchSize, channels, outHeight, outWidth]);
        var inputMemory = input.Data;
        var outputMemory = output.Data;

        // Parallel nearest-neighbor upsampling for efficiency
        Parallel.For(0, batchSize * channels, idx =>
        {
            var inputData = inputMemory.Span;
            var outputData = outputMemory.Span;
            int b = idx / channels;
            int c = idx % channels;
            int inBaseIdx = (b * channels + c) * height * width;
            int outBaseIdx = (b * channels + c) * outHeight * outWidth;

            for (int h = 0; h < height; h++)
            {
                int outH = h * 2;
                for (int w = 0; w < width; w++)
                {
                    int outW = w * 2;
                    T val = inputData[inBaseIdx + h * width + w];

                    // Write 2x2 block
                    outputData[outBaseIdx + outH * outWidth + outW] = val;
                    outputData[outBaseIdx + outH * outWidth + outW + 1] = val;
                    outputData[outBaseIdx + (outH + 1) * outWidth + outW] = val;
                    outputData[outBaseIdx + (outH + 1) * outWidth + outW + 1] = val;
                }
            }
        });

        return output;
    }

    private Tensor<T> ApplyGELU(Tensor<T> input)
    {
        return input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            double c = Math.Sqrt(2.0 / Math.PI);
            double gelu = 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
            return NumOps.FromDouble(gelu);
        });
    }

    private Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        return input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            double sigmoid = 1.0 / (1.0 + Math.Exp(-x));
            return NumOps.FromDouble(sigmoid);
        });
    }

    private Tensor<T> ApplyGELUGradient(Tensor<T> gradient, Tensor<T> input)
    {
        return gradient.Transform((g, idx) =>
        {
            double x = Convert.ToDouble(input.Data.Span[idx]);
            double c = Math.Sqrt(2.0 / Math.PI);
            double inner = c * (x + 0.044715 * x * x * x);
            double tanh_inner = Math.Tanh(inner);
            double sech2 = 1.0 - tanh_inner * tanh_inner;
            double d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
            double grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;
            return NumOps.Multiply(g, NumOps.FromDouble(grad));
        });
    }

    private Tensor<T> ApplySigmoidGradient(Tensor<T> gradient, Tensor<T> output)
    {
        return gradient.Transform((g, idx) =>
        {
            double s = Convert.ToDouble(output.Data.Span[idx]);
            double grad = s * (1.0 - s);
            return NumOps.Multiply(g, NumOps.FromDouble(grad));
        });
    }

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        int c = tensor.Shape[0];
        int h = tensor.Shape[1];
        int w = tensor.Shape[2];

        var result = new Tensor<T>([1, c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int c = tensor.Shape[1];
        int h = tensor.Shape[2];
        int w = tensor.Shape[3];

        var result = new Tensor<T>([c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private void BackwardPass(Tensor<T> gradient)
    {
        if (!_useNativeMode || Layers.Count == 0)
        {
            return;
        }

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }
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
            ModelType = ModelType.VideoDepthEstimation,
            AdditionalInfo = additionalInfo,
            ModelData = this.Serialize()
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
