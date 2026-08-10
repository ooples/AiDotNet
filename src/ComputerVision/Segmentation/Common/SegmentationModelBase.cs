using System.IO;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds to ours when this import shadows them from a nearer
// scope. Without it the attribute silently resolves to the wrong type and ADNSHAPE003 reports this
// contract as having no input layout - which is exactly what happened before this line was added.
using AiDotNet.Attributes;
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
// MEASURED, not assumed. Every axis below was falsified by building each model under four profiles
// that differ in exactly one variable and comparing what Predict returned (ModelFamilyLawTests):
//
//   classes 7 -> 13   moved axis 1 only          => Fixed(_numClasses)
//   extent 64 -> 128  moved axes 2,3: 2 -> 4     => Scaled(1/32), a /32-stride encoder
//   batch 1 -> 2      moved axis 0 only          => Same(Batch)
//
// 10 models probed, 0 skipped, unanimous. The geometries STRADDLE the stride deliberately: a first
// attempt at 8 and 16 reported every spatial axis as a constant 1, because /32 floors both to 1 - so
// it could not tell Fixed(1) from Scaled(1/32), and declaring Fixed(1) would have been wrong for all
// 69 models inheriting this. Probing below the stride cannot falsify a spatial constant.
//
// RANK 4 ONLY. Forward also accepts rank-3 [C,H,W] by promoting it, but no rank-3 output was measured,
// so OutputAxesFor declines that rank rather than guessing - see its remarks.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input,
    Note = "A batched image. Forward rejects every rank but 3 and 4.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Classes, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output,
    Note = "Per-class logits over a /32 feature grid; the class axis IS the channel axis of the output.")]
public abstract class SegmentationModelBase<T> : NeuralNetworkBase<T>, ISegmentationModel<T>, IShapeContract
{
    /// <summary>
    /// The output axes for a segmentation model, shared by every model in the family.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Declared ONCE here rather than on each of the 69 derived models, which is the whole point of a
    /// family contract: [TensorLayout] is inherited and this method is virtual, so a model whose law
    /// genuinely differs overrides it and every other model needs no shape code at all.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> segmentation labels every pixel, so the output has one value per class per
    /// position. The class count comes from the model's configuration, and the spatial grid is 32x
    /// smaller than the input because the encoder downsamples.
    /// </para>
    /// <para>
    /// Returns null for any rank but 4, and null when the class count is unresolved - declining is the
    /// honest answer where nothing was measured, and a contract that cannot fire is better than one
    /// that fires wrongly.
    /// </para>
    /// </remarks>
    public virtual IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => SpatialStrideContract(inputRank, 32);

    /// <summary>
    /// The family contract at a given encoder stride: [Batch, Classes, H/stride, W/stride].
    /// </summary>
    /// <param name="inputRank">Rank of the incoming shape; only rank 4 is declared.</param>
    /// <param name="stride">Total downsampling factor of this model's encoder.</param>
    /// <remarks>
    /// <para>
    /// The STRIDE is the only thing that varies across the family, so it is the only thing an override
    /// has to supply. Measured across all 69 models: 59 are /32 (the default here), SAM/SAMHQ/EoMT/
    /// EfficientTAM are /16, ViTCoMer is /4, and four models do not downsample at all (stride 1).
    /// Batch and the class axis are unanimous, so they live here once rather than in ten overrides.
    /// </para>
    /// <para>
    /// Every one of those numbers came from the conformance sweep comparing the declaration against a
    /// real Predict, not from reading encoders. The first attempt declared /32 for the whole family and
    /// the sweep rejected it for exactly these ten models.
    /// </para>
    /// </remarks>
    protected IReadOnlyList<OutputAxisContract>? SpatialStrideContract(int inputRank, int stride)
    {
        if (inputRank != 4 || _numClasses <= 0 || stride <= 0) return null;

        AxisRelation Spatial(TensorAxis axis)
            => stride == 1 ? AxisRelation.Same(axis) : AxisRelation.Scaled(axis, 1, stride);

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Classes, AxisRelation.Fixed(_numClasses)),
            new OutputAxisContract(TensorAxis.Height, Spatial(TensorAxis.Height)),
            new OutputAxisContract(TensorAxis.Width, Spatial(TensorAxis.Width)),
        };
    }

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
    /// Gradient-based optimizer for training (null in ONNX mode until <see cref="Optimizer"/> resolves it).
    /// </summary>
    /// <remarks>
    /// <para>
    /// NOT readonly, and that is what makes this base adoptable at all. It was <c>protected readonly</c>,
    /// assignable only from a base-constructor parameter - and the default every segmentation model
    /// actually wants is <c>new AdamWOptimizer&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;(this)</c>, which
    /// a derived type CANNOT pass to a base constructor because <c>this</c> is not available in a
    /// constructor initializer (CS0027), and CANNOT assign afterwards because the field was readonly
    /// (CS0191). So a model wanting the standard default had no way to express it through this base and
    /// had to keep its own field and derive from NeuralNetworkBase instead. All 70 concrete segmentation
    /// models did exactly that, which is why this base and its 8 family bases had zero users.
    /// </para>
    /// </remarks>
    protected IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

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

    /// <summary>
    /// The optimizer used for training, created on first use if the constructor was given none.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Resolving LAZILY is the point: it runs after construction completes, so <c>this</c> is a fully
    /// built model and the default optimizer can bind to it. That is the one thing a constructor
    /// parameter cannot do, and its absence is what made this base unusable.
    /// </para>
    /// <para>
    /// Null in ONNX mode, where training is not supported and asking for an optimizer is a caller error
    /// rather than something to satisfy with a default.
    /// </para>
    /// </remarks>
    protected IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer
        => _useNativeMode ? _optimizer ??= CreateDefaultOptimizer() : null;

    /// <summary>
    /// Creates the optimizer used when the constructor was given none. Override to change the default.
    /// </summary>
    protected virtual IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
        => new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

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
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>())
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
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
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
    protected override Tensor<T> PredictCore(Tensor<T> input)
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

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
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
