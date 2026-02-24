using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Abstract base class for all segmentation models, providing common dual-mode (native + ONNX)
/// infrastructure, batch handling, forward/backward passes, and serialization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is the foundation for all segmentation models in the library.
/// It handles the plumbing that every segmentation model needs:
/// - Loading pre-trained ONNX models for fast inference
/// - Native mode for training from scratch or fine-tuning
/// - Converting images between batched and unbatched formats
/// - Saving and loading model weights
///
/// You don't use this class directly — instead, create a concrete model like SegFormer, Mask2Former,
/// or SAM that extends this base class.
/// </para>
/// </remarks>
public abstract class SegmentationModelBase<T> : NeuralNetworkBase<T>, ISegmentationModel<T>
{
    #region Fields

    /// <summary>
    /// Input image height in pixels.
    /// </summary>
    protected int _height;

    /// <summary>
    /// Input image width in pixels.
    /// </summary>
    protected int _width;

    /// <summary>
    /// Number of input channels (typically 3 for RGB).
    /// </summary>
    protected int _channels;

    /// <summary>
    /// Number of segmentation output classes.
    /// </summary>
    protected int _numClasses;

    /// <summary>
    /// Whether the model is running in native (trainable) mode or ONNX (inference-only) mode.
    /// </summary>
    protected bool _useNativeMode;

    /// <summary>
    /// Path to the ONNX model file (null in native mode).
    /// </summary>
    protected string? _onnxModelPath;

    /// <summary>
    /// ONNX runtime inference session (null in native mode).
    /// </summary>
    protected InferenceSession? _onnxSession;

    /// <summary>
    /// Gradient-based optimizer for training (null in ONNX mode).
    /// </summary>
    protected readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// Whether this instance has been disposed.
    /// </summary>
    protected bool _disposed;

    /// <summary>
    /// Index separating encoder layers from decoder layers in the Layers list.
    /// </summary>
    protected int _encoderLayerEnd;

    #endregion

    #region ISegmentationModel Implementation

    /// <inheritdoc/>
    public int NumClasses => _numClasses;

    /// <inheritdoc/>
    public int InputHeight => _height;

    /// <inheritdoc/>
    public int InputWidth => _width;

    /// <inheritdoc/>
    public bool IsOnnxMode => !_useNativeMode;

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer for training.</param>
    /// <param name="lossFunction">Loss function for training.</param>
    /// <param name="numClasses">Number of segmentation classes.</param>
    protected SegmentationModelBase(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction,
        int numClasses)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "numClasses must be > 0.");
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _useNativeMode = true;
        _onnxModelPath = null;
        _optimizer = optimizer;
    }

    /// <summary>
    /// Initializes the base in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes the model predicts.</param>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    protected SegmentationModelBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "numClasses must be > 0.");
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex);
        }
    }

    #endregion

    #region Segmentation Methods

    /// <inheritdoc/>
    public virtual Tensor<T> Segment(Tensor<T> image)
    {
        return Predict(image);
    }

    /// <summary>
    /// Runs a forward pass, dispatching to ONNX or native mode.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (!_useNativeMode)
        {
            return PredictOnnx(input);
        }

        return Forward(input);
    }

    /// <summary>
    /// Performs one training step: forward pass, loss, backward pass, and parameter update.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException(
                "Training is not supported in ONNX mode. Use the native mode constructor for training.");
        }

        var predicted = Forward(input);
        var gradVec = LossFunction.CalculateDerivative(predicted.ToVector(), expectedOutput.ToVector());
        var lossGradient = Tensor<T>.FromVector(gradVec);

        BackwardPass(lossGradient);

        _optimizer?.UpdateParameters(Layers);
    }

    #endregion

    #region Forward / Backward

    /// <summary>
    /// Executes the full forward pass through encoder and decoder layers.
    /// </summary>
    protected virtual Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank != 3 && input.Rank != 4)
            throw new ArgumentException("Input must be rank 3 [C,H,W] or rank 4 [N,C,H,W].", nameof(input));
        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
        {
            input = AddBatchDimension(input);
        }

        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++)
        {
            features = Layers[i].Forward(features);
        }

        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
        {
            features = Layers[i].Forward(features);
        }

        if (!hasBatch)
        {
            features = RemoveBatchDimension(features);
        }

        return features;
    }

    /// <summary>
    /// Runs ONNX inference.
    /// </summary>
    protected virtual Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
        {
            input = AddBatchDimension(input);
        }

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.FirstOrDefault() ?? "pixel_values";
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        var result = new Tensor<T>(outputShape, new Vector<T>(outputData));

        if (!hasBatch)
        {
            result = RemoveBatchDimension(result);
        }

        return result;
    }

    /// <summary>
    /// Propagates gradients backward through all layers.
    /// </summary>
    protected virtual void BackwardPass(Tensor<T> gradient)
    {
        if (!_useNativeMode || Layers.Count == 0) return;

        if (gradient.Rank == 3) gradient = AddBatchDimension(gradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }
    }

    #endregion

    #region Tensor Helpers

    /// <summary>
    /// Adds a batch dimension to a [C, H, W] tensor, producing [1, C, H, W].
    /// </summary>
    protected Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        if (tensor.Rank != 3)
            throw new ArgumentException("Expected rank-3 tensor [C,H,W].", nameof(tensor));
        int c = tensor.Shape[0];
        int h = tensor.Shape[1];
        int w = tensor.Shape[2];

        var result = new Tensor<T>([1, c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    /// <summary>
    /// Removes the batch dimension from a [1, ...] tensor.
    /// </summary>
    protected Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        if (tensor.Rank < 1 || tensor.Shape[0] != 1)
            throw new ArgumentException("Expected batch dimension of 1 to remove.", nameof(tensor));
        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++)
        {
            newShape[i] = tensor.Shape[i + 1];
        }

        var result = new Tensor<T>(newShape);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    #endregion

    #region Parameter Updates

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (offset + layerParamCount > parameters.Length)
            {
                throw new ArgumentException(
                    $"Parameter vector is too short: need {offset + layerParamCount} elements, but got {parameters.Length}.",
                    nameof(parameters));
            }

            var newParams = new Vector<T>(layerParamCount);
            for (int i = 0; i < layerParamCount; i++)
            {
                newParams[i] = parameters[offset + i];
            }
            layer.UpdateParameters(newParams);
            offset += layerParamCount;
        }
    }

    #endregion

    #region Serialization Helpers

    /// <summary>
    /// Writes common segmentation fields to a binary stream.
    /// </summary>
    protected void SerializeSegmentationBaseData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numClasses);
        writer.Write(_useNativeMode);
        writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
    }

    /// <summary>
    /// Reads common segmentation fields from a binary stream.
    /// </summary>
    protected void DeserializeSegmentationBaseData(BinaryReader reader)
    {
        _height = reader.ReadInt32();
        _width = reader.ReadInt32();
        _channels = reader.ReadInt32();
        _numClasses = reader.ReadInt32();
        _useNativeMode = reader.ReadBoolean();
        var onnxPath = reader.ReadString();
        _onnxModelPath = string.IsNullOrEmpty(onnxPath) ? null : onnxPath;
        _encoderLayerEnd = reader.ReadInt32();
    }

    #endregion

    #region Dispose

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _onnxSession?.Dispose();
                _onnxSession = null;
            }
            _disposed = true;
        }
        base.Dispose(disposing);
    }

    #endregion
}
