using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SegNeXt is a semantic segmentation model that labels every pixel
/// in an image with a category (e.g., road, tree, building). Unlike transformer-based models,
/// SegNeXt uses a purely convolutional backbone called MSCAN (Multi-Scale Convolutional Attention
/// Network) that achieves better accuracy than many transformers while being simpler and faster.
///
/// Common use cases:
/// - Autonomous driving (road, lane, sidewalk parsing)
/// - Drone/satellite imagery analysis
/// - Indoor scene understanding
/// - Real-time segmentation on resource-constrained devices
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - MSCAN backbone with multi-scale convolutional attention (no self-attention needed)
/// - Multi-branch depth-wise strip convolutions capture multi-scale context
/// - Attention weights are computed via 1x1 convolutions on concatenated multi-scale features
/// - Hamburger decoder uses matrix decomposition for global context aggregation
/// - Four model sizes (Tiny to Large) from 4.3M to 48.9M parameters
/// </para>
/// <para>
/// <b>Reference:</b> Guo et al., "SegNeXt: Rethinking Convolutional Attention Design for
/// Semantic Segmentation", NeurIPS 2022.
/// </para>
/// </remarks>
public class SegNeXt<T> : NeuralNetworkBase<T>
{
    private readonly SegNeXtOptions _options;

    /// <summary>
    /// Gets the configuration options for this SegNeXt model.
    /// </summary>
    /// <returns>The <see cref="SegNeXtOptions"/> used to configure this model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Options control how the model behaves, including the random seed
    /// for reproducibility. This returns the options that were passed in when creating the model,
    /// or the defaults if none were provided.
    /// </para>
    /// </remarks>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numClasses;
    private readonly SegNeXtModelSize _modelSize;
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
    /// Gets whether this SegNeXt instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SegNeXt can run in two modes: native mode (supports both training
    /// and inference) and ONNX mode (inference only, using a pre-trained model file). This property
    /// returns <c>true</c> when the model was created with the native constructor and <c>false</c>
    /// when loaded from an ONNX file. If you need to train SegNeXt on your own data,
    /// use the native constructor.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets whether using native mode (trainable) or ONNX mode (inference only).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the model size variant (Tiny through Large).
    /// </summary>
    internal SegNeXtModelSize ModelSize => _modelSize;

    /// <summary>
    /// Gets the number of semantic classes this model predicts.
    /// </summary>
    internal int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of SegNeXt in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration that defines
    /// input dimensions (height, width, channels). For a standard SegNeXt setup, use
    /// ThreeDimensional input type with your image dimensions (e.g., 512x512x3).</param>
    /// <param name="optimizer">The gradient-based optimizer used to update model weights during
    /// training (default: AdamW with weight decay, as specified in the SegNeXt paper). AdamW is
    /// the standard choice for modern segmentation models because it combines adaptive learning
    /// rates with decoupled weight decay for better generalization.</param>
    /// <param name="lossFunction">The loss function used to measure prediction error during training
    /// (default: CrossEntropyLoss, the standard for multi-class segmentation tasks).</param>
    /// <param name="numClasses">Number of semantic classes to predict. Set this to match your dataset
    /// (default: 150 for the ADE20K benchmark, use 19 for Cityscapes, or your custom class count).</param>
    /// <param name="modelSize">Model size variant controlling the number of parameters and
    /// accuracy (default: Tiny, the smallest and fastest variant with 4.3M parameters).</param>
    /// <param name="dropRate">Dropout rate applied for regularization (default: 0.1). Higher values
    /// reduce overfitting but may slow convergence.</param>
    /// <param name="options">Optional model options including random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a trainable SegNeXt model from scratch.
    /// Use this when you want to train or fine-tune on your own segmentation dataset.
    /// Start with the Tiny variant for quick prototyping, then scale up to Base or Large
    /// for production accuracy.
    ///
    /// SegNeXt processes images through a 4-stage convolutional encoder where each stage uses
    /// multi-scale depth-wise strip convolutions to capture context at different scales, followed
    /// by a Hamburger decoder that aggregates global context using matrix decomposition.
    /// </para>
    /// </remarks>
    public SegNeXt(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        SegNeXtModelSize modelSize = SegNeXtModelSize.Tiny,
        double dropRate = 0.1,
        SegNeXtOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new SegNeXtOptions();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _modelSize = modelSize;
        _dropRate = dropRate;
        _useNativeMode = true;
        _onnxModelPath = null;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of SegNeXt in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration. Input dimensions
    /// should match the ONNX model's expected input (typically 512x512x3 or 1024x1024x3).</param>
    /// <param name="onnxModelPath">Absolute or relative path to the pre-trained ONNX model file.
    /// Pre-trained SegNeXt ONNX models can be exported from the MMSegmentation framework.</param>
    /// <param name="numClasses">Number of semantic classes the ONNX model was trained to predict
    /// (default: 150 for ADE20K). This must match the model's training configuration.</param>
    /// <param name="modelSize">Model size variant for metadata purposes (default: Tiny). This should
    /// match the ONNX model's architecture so metadata accurately reflects the model.</param>
    /// <param name="options">Optional model options including random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor loads a pre-trained SegNeXt model from an ONNX file.
    /// Use this for fast inference when you don't need to train the model. ONNX mode is typically
    /// faster than native mode because the ONNX runtime applies hardware-specific optimizations.
    ///
    /// Note: ONNX mode does not support training. If you call Train() on an ONNX model,
    /// it will throw an exception. To fine-tune, use the native constructor instead.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public SegNeXt(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        SegNeXtModelSize modelSize = SegNeXtModelSize.Tiny,
        SegNeXtOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new SegNeXtOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"SegNeXt ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _modelSize = modelSize;
        _dropRate = 0.0;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load SegNeXt ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through the SegNeXt model to produce a per-pixel segmentation map.
    /// </summary>
    /// <param name="input">The input image tensor. Accepts either a single image as [C, H, W]
    /// or a batched input as [B, C, H, W]. Pixel values should be normalized to [0, 1]
    /// or standardized with ImageNet statistics.</param>
    /// <returns>
    /// A tensor containing per-pixel class logits. The output spatial resolution matches
    /// the decoder output (typically 1/4 of input). Take argmax along the class dimension
    /// to get the predicted class per pixel.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for segmenting an image. Pass in your image
    /// tensor and get back a map where each pixel has a score for every possible class.
    /// In native mode, the image goes through the MSCAN encoder and Hamburger decoder.
    /// In ONNX mode, the ONNX runtime handles everything for optimized inference.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (!_useNativeMode)
        {
            return PredictOnnx(input);
        }

        return Forward(input);
    }

    /// <summary>
    /// Performs one training step: forward pass, loss computation, backward pass, and parameter update.
    /// </summary>
    /// <param name="input">The input image tensor [B, C, H, W] or [C, H, W].</param>
    /// <param name="expectedOutput">The ground-truth segmentation map tensor matching the model output shape.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training teaches the model to produce better segmentation maps by comparing
    /// its predictions to correct answers. Each call to Train():
    /// 1. Runs the input through the MSCAN encoder and Hamburger decoder (forward pass)
    /// 2. Compares prediction to ground truth to compute the error (loss)
    /// 3. Calculates how each weight contributed to the error (backward pass)
    /// 4. Updates weights using AdamW optimizer to reduce future errors
    ///
    /// Training is only available in native mode. ONNX models will throw an exception.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException(
                "Training is not supported in ONNX mode. Use the native mode constructor for training.");
        }

        var predicted = Forward(input);
        var lossGradient = predicted.Transform((v, idx) =>
            NumOps.Subtract(v, expectedOutput.Data.Span[idx]));

        BackwardPass(lossGradient);

        _optimizer?.UpdateParameters(Layers);
    }

    #endregion

    #region Private Methods

    /// <summary>
    /// Returns the architecture configuration for a given SegNeXt model size.
    /// </summary>
    /// <param name="modelSize">The SegNeXt model size variant.</param>
    /// <returns>A tuple containing channel dimensions per stage, depths per stage,
    /// and the Hamburger decoder hidden dimension.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each SegNeXt size uses different channel widths and encoder depths.
    /// The "channel dims" control feature richness at each of the 4 stages, and the "depths"
    /// control how many MSCAN blocks are stacked per stage. Larger models use more channels
    /// and deeper encoders for higher accuracy.
    /// </para>
    /// </remarks>
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(
        SegNeXtModelSize modelSize)
    {
        return modelSize switch
        {
            SegNeXtModelSize.Tiny => ([32, 64, 160, 256], [3, 3, 5, 2], 256),
            SegNeXtModelSize.Small => ([64, 128, 320, 512], [2, 2, 4, 2], 256),
            SegNeXtModelSize.Base => ([64, 128, 320, 512], [3, 3, 12, 3], 512),
            SegNeXtModelSize.Large => ([64, 128, 320, 512], [3, 5, 27, 3], 1024),
            _ => ([32, 64, 160, 256], [3, 3, 5, 2], 256)
        };
    }

    /// <summary>
    /// Executes the full forward pass through the SegNeXt MSCAN encoder and Hamburger decoder.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>The segmentation logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The forward pass has two stages:
    /// 1. <b>MSCAN Encoder</b>: Processes the image through 4 stages of multi-scale convolutional
    ///    attention blocks. Each stage uses depth-wise strip convolutions at multiple scales
    ///    to capture both local details and wider context, producing features at decreasing
    ///    spatial resolutions (1/4, 1/8, 1/16, 1/32).
    /// 2. <b>Hamburger Decoder</b>: Aggregates the encoder features using matrix decomposition
    ///    (Non-negative Matrix Factorization) for global context, then produces per-pixel predictions.
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
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
    /// Runs inference using the ONNX runtime session.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
    /// <returns>The segmentation logits tensor from the ONNX model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When using a pre-trained ONNX model, this method converts the input
    /// to float arrays for the ONNX runtime, runs the optimized model, and converts the output
    /// back to the library's tensor format.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX session was not initialized.</exception>
    private Tensor<T> PredictOnnx(Tensor<T> input)
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
    /// Propagates gradients backward through all layers from decoder to encoder.
    /// </summary>
    /// <param name="gradient">The loss gradient tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation calculates how much each weight contributed to the
    /// prediction error, working backward from the decoder output through the MSCAN encoder.
    /// These gradients are then used by AdamW to adjust weights and improve future predictions.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Adds a batch dimension to an unbatched [C, H, W] tensor, producing [1, C, H, W].
    /// </summary>
    /// <param name="tensor">An unbatched image tensor.</param>
    /// <returns>A batched tensor with shape [1, C, H, W].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks process images in batches. When you pass a single image,
    /// this wraps it in a batch of size 1 so the layers can process it correctly.
    /// </para>
    /// </remarks>
    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        int c = tensor.Shape[0];
        int h = tensor.Shape[1];
        int w = tensor.Shape[2];

        var result = new Tensor<T>([1, c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    /// <summary>
    /// Removes the batch dimension from a [1, C, H, W] tensor, producing [C, H, W].
    /// </summary>
    /// <param name="tensor">A batched tensor with shape [1, C, H, W].</param>
    /// <returns>An unbatched tensor with shape [C, H, W].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After processing a single image, this strips the batch dimension
    /// to return output in the same format as the original input.
    /// </para>
    /// </remarks>
    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
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

    #region Abstract Implementation

    /// <summary>
    /// Initializes the MSCAN encoder and Hamburger decoder layers for the SegNeXt model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method builds the internal structure of the SegNeXt model.
    /// In native mode, it creates:
    /// 1. <b>MSCAN encoder layers</b>: 4 stages of multi-scale convolutional attention blocks
    ///    with depth-wise strip convolutions that capture context at multiple scales.
    /// 2. <b>Hamburger decoder layers</b>: A decoder that uses matrix decomposition to aggregate
    ///    global context, then produces per-pixel class predictions.
    ///
    /// In ONNX mode, no layers are created because the ONNX runtime handles everything internally.
    /// </para>
    /// </remarks>
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
            _encoderLayerEnd = Architecture.Layers.Count / 2;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateSegNeXtEncoderLayers(
                _channels, _height, _width,
                _channelDims, _depths, _dropRate).ToList();

            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int[] patchKernels = [7, 3, 3, 3];
            int[] patchStrides = [4, 2, 2, 2];
            int[] patchPaddings = [3, 1, 1, 1];
            int featureH = _height;
            int featureW = _width;
            for (int stage = 0; stage < 4; stage++)
            {
                featureH = (featureH + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
                featureW = (featureW + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
            }

            int encoderOutputChannels = _channelDims[^1];
            var decoderLayers = LayerHelper<T>.CreateSegNeXtDecoderLayers(
                encoderOutputChannels, _decoderDim, _numClasses,
                featureH, featureW);

            Layers.AddRange(decoderLayers);
        }
    }

    /// <summary>
    /// Updates all trainable parameters across the model from a flat parameter vector.
    /// </summary>
    /// <param name="parameters">A flat vector containing new values for all model parameters,
    /// ordered sequentially by layer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method replaces all current weights with new values from a flat vector.
    /// It walks through each layer in order, slicing out the correct number of parameters and updating
    /// them. This is used internally during optimization and when loading saved model weights.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Collects metadata describing this SegNeXt model's configuration and state.
    /// </summary>
    /// <returns>A <see cref="ModelMetadata{T}"/> containing model type, configuration, and serialized data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Metadata is a summary of everything about the model — its type,
    /// input dimensions, class count, model size, and serialized weights. Useful for saving models,
    /// comparing configurations, or displaying model info in a dashboard.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "SegNeXt" },
            { "Description", "SegNeXt Semantic Segmentation Model" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "InputChannels", _channels },
            { "NumClasses", _numClasses },
            { "ModelSize", _modelSize.ToString() },
            { "DecoderDim", _decoderDim },
            { "DropRate", _dropRate },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count },
            { "EncoderLayerEnd", _encoderLayerEnd }
        };

        return new ModelMetadata<T>
        {
            ModelType = ModelType.SemanticSegmentation,
            AdditionalInfo = additionalInfo,
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Writes all SegNeXt-specific configuration values to a binary stream for persistence.
    /// </summary>
    /// <param name="writer">The binary writer to serialize configuration data into.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When saving a SegNeXt model, the base class handles layer weights,
    /// and this method saves the model's configuration (dimensions, classes, model size, etc.).
    /// The data is written in a specific order matching <see cref="DeserializeNetworkSpecificData"/>.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numClasses);
        writer.Write((int)_modelSize);
        writer.Write(_decoderDim);
        writer.Write(_dropRate);
        writer.Write(_useNativeMode);
        writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);

        writer.Write(_channelDims.Length);
        foreach (int dim in _channelDims)
        {
            writer.Write(dim);
        }

        writer.Write(_depths.Length);
        foreach (int depth in _depths)
        {
            writer.Write(depth);
        }
    }

    /// <summary>
    /// Reads SegNeXt-specific configuration values from a binary stream during model loading.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize configuration data from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the counterpart to <see cref="SerializeNetworkSpecificData"/>.
    /// When loading a saved model, this reads back configuration values in the exact same order
    /// they were written.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // height
        _ = reader.ReadInt32(); // width
        _ = reader.ReadInt32(); // channels
        _ = reader.ReadInt32(); // numClasses
        _ = reader.ReadInt32(); // modelSize
        _ = reader.ReadInt32(); // decoderDim
        _ = reader.ReadDouble(); // dropRate
        _ = reader.ReadBoolean(); // useNativeMode
        _ = reader.ReadString(); // onnxModelPath
        _ = reader.ReadInt32(); // encoderLayerEnd

        int channelCount = reader.ReadInt32();
        for (int i = 0; i < channelCount; i++)
        {
            _ = reader.ReadInt32();
        }

        int depthCount = reader.ReadInt32();
        for (int i = 0; i < depthCount; i++)
        {
            _ = reader.ReadInt32();
        }
    }

    /// <summary>
    /// Creates a new SegNeXt instance with the same configuration but freshly initialized weights.
    /// </summary>
    /// <returns>A new <see cref="SegNeXt{T}"/> with the same settings but reinitialized weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a "copy" of the model configuration with fresh random weights.
    /// Used internally for cross-validation or ensemble training where multiple independent copies
    /// of the same architecture are needed.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new SegNeXt<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options);
        }
        else
        {
            return new SegNeXt<T>(Architecture, _onnxModelPath!, _numClasses, _modelSize, _options);
        }
    }

    /// <summary>
    /// Releases managed resources held by this SegNeXt instance.
    /// </summary>
    /// <param name="disposing"><c>true</c> when called from Dispose(); <c>false</c> from finalizer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When done using a SegNeXt model, calling Dispose() frees memory
    /// and file handles used by the ONNX runtime session. Use a <c>using</c> statement:
    /// <c>using var model = new SegNeXt&lt;float&gt;(...);</c>
    /// </para>
    /// </remarks>
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
