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
/// SegFormer: Simple and Efficient Semantic Segmentation with Transformers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SegFormer is a semantic segmentation model that classifies every pixel
/// in an image into a category (e.g., road, sky, person, building). It uses a hierarchical
/// transformer encoder (Mix Transformer / MiT) and a lightweight MLP decoder — no complex
/// decoders or positional encodings needed.
///
/// Common use cases:
/// - Autonomous driving (road, lane, obstacle detection)
/// - Medical imaging (organ/lesion segmentation)
/// - Scene understanding (indoor/outdoor parsing)
/// - Agriculture (crop/weed detection from drone images)
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - 4-stage hierarchical encoder producing multi-scale features (1/4, 1/8, 1/16, 1/32)
/// - Overlapping patch embedding replaces non-overlapping ViT patches
/// - Efficient Self-Attention with spatial reduction for computational savings
/// - Mix-FFN uses 3x3 depthwise convolutions instead of positional encodings
/// - Lightweight All-MLP decode head for fast inference
/// - Six model sizes (B0–B5) from 3.8M to 82.0M parameters
/// </para>
/// <para>
/// <b>Reference:</b> Xie et al., "SegFormer: Simple and Efficient Design for Semantic
/// Segmentation with Transformers", NeurIPS 2021.
/// </para>
/// </remarks>
public class SegFormer<T> : NeuralNetworkBase<T>, ISemanticSegmentation<T>
{
    private readonly SegFormerOptions _options;

    /// <summary>
    /// Gets the configuration options for this SegFormer model.
    /// </summary>
    /// <returns>The <see cref="SegFormerOptions"/> used to configure this model instance.</returns>
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
    private readonly SegFormerModelSize _modelSize;
    private readonly int[] _embedDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly int[] _numHeads;
    private readonly double _dropRate;
    private readonly bool _useNativeMode;
    private readonly string? _onnxModelPath;
    private InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;

    // Boundary between encoder and decoder layers in the Layers list
    private int _encoderLayerEnd;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether this SegFormer instance supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SegFormer can run in two modes: native mode (supports both training
    /// and inference) and ONNX mode (inference only, using a pre-trained model file). This property
    /// returns <c>true</c> when the model was created with the native constructor and <c>false</c>
    /// when loaded from an ONNX file. If you need to fine-tune SegFormer on your own data,
    /// make sure you create it using the native constructor.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets whether using native mode (trainable) or ONNX mode (inference only).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the model size variant (B0 through B5).
    /// </summary>
    internal SegFormerModelSize ModelSize => _modelSize;

    /// <summary>
    /// Gets the number of semantic classes this model predicts.
    /// </summary>
    internal int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of SegFormer in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration that defines
    /// input dimensions (height, width, channels). For a standard SegFormer setup, use
    /// ThreeDimensional input type with your image dimensions (e.g., 512x512x3).</param>
    /// <param name="optimizer">The gradient-based optimizer used to update model weights during
    /// training (default: AdamW with weight decay, as specified in the SegFormer paper). AdamW is
    /// the standard choice for transformer-based segmentation models because it combines adaptive
    /// learning rates with decoupled weight decay for better generalization.</param>
    /// <param name="lossFunction">The loss function used to measure prediction error during training
    /// (default: CrossEntropyLoss, the standard for multi-class segmentation tasks).</param>
    /// <param name="numClasses">Number of semantic classes to predict. Set this to match your dataset
    /// (default: 150 for the ADE20K benchmark, use 21 for Pascal VOC, or your custom class count).</param>
    /// <param name="modelSize">Model size variant B0–B5 controlling the number of parameters and
    /// accuracy (default: B0, the smallest and fastest variant with 3.8M parameters).</param>
    /// <param name="dropRate">Dropout rate applied within transformer blocks for regularization
    /// (default: 0.1). Higher values reduce overfitting but may slow convergence.</param>
    /// <param name="options">Optional model options including random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a trainable SegFormer model from scratch.
    /// Use this when you want to train or fine-tune on your own segmentation dataset.
    /// Start with B0 for experimentation since it trains quickly, then scale up to B2–B5
    /// for production use when you need higher accuracy.
    ///
    /// The model works by first encoding your image through a 4-stage transformer encoder that
    /// extracts features at multiple scales, then decoding those features into a per-pixel
    /// class prediction map through a lightweight MLP decoder.
    /// </para>
    /// </remarks>
    public SegFormer(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        SegFormerModelSize modelSize = SegFormerModelSize.B0,
        double dropRate = 0.1,
        SegFormerOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new SegFormerOptions();
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

        (_embedDims, _depths, _numHeads, _decoderDim) = GetModelConfig(modelSize);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of SegFormer in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration. Input dimensions
    /// should match the ONNX model's expected input (typically 512x512x3 or 1024x1024x3).</param>
    /// <param name="onnxModelPath">Absolute or relative path to the pre-trained ONNX model file.
    /// Pre-trained SegFormer ONNX models can be exported from Hugging Face or NVIDIA's model zoo.</param>
    /// <param name="numClasses">Number of semantic classes the ONNX model was trained to predict
    /// (default: 150 for ADE20K). This must match the model's training configuration.</param>
    /// <param name="modelSize">Model size variant for metadata purposes (default: B0). This should
    /// match the ONNX model's architecture so metadata accurately reflects the model.</param>
    /// <param name="options">Optional model options including random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor loads a pre-trained SegFormer model from an ONNX file.
    /// Use this for fast inference when you don't need to train the model — just load and predict.
    /// ONNX mode is typically faster than native mode because the ONNX runtime applies hardware-specific
    /// optimizations automatically.
    ///
    /// Note: ONNX mode does not support training. If you call Train() on an ONNX model,
    /// it will throw an exception. To fine-tune a model, use the native constructor instead.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found at the specified path.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load or parse the model file.</exception>
    public SegFormer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        SegFormerModelSize modelSize = SegFormerModelSize.B0,
        SegFormerOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new SegFormerOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"SegFormer ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numClasses = numClasses;
        _modelSize = modelSize;
        _dropRate = 0.0;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        (_embedDims, _depths, _numHeads, _decoderDim) = GetModelConfig(modelSize);

        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load SegFormer ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through the SegFormer model to produce a per-pixel segmentation map.
    /// </summary>
    /// <param name="input">The input image tensor. Accepts either a single image as [C, H, W]
    /// (channels, height, width) or a batched input as [B, C, H, W] (batch, channels, height, width).
    /// Pixel values should be normalized to [0, 1] or standardized with ImageNet statistics.</param>
    /// <returns>
    /// A tensor containing per-pixel class logits. For batched input [B, C, H, W], the output shape
    /// is [B, numClasses, outH, outW]. For unbatched input [C, H, W], the batch dimension is removed
    /// and the output shape is [numClasses, outH, outW]. To get the predicted class for each pixel,
    /// take the argmax along the class dimension.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method you call to segment an image. Pass in your image
    /// tensor and get back a map where each pixel has a score for every possible class. The class with
    /// the highest score at each pixel is the model's prediction for what that pixel represents.
    ///
    /// In native mode, this runs the image through the transformer encoder and MLP decoder layers.
    /// In ONNX mode, this delegates to the ONNX runtime for optimized inference.
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
    /// <param name="input">The input image tensor [B, C, H, W] or [C, H, W] containing the training image(s).</param>
    /// <param name="expectedOutput">The ground-truth segmentation map tensor with the same spatial dimensions
    /// as the model output. Each value represents the correct class label for that pixel position.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training teaches the model to produce better segmentation maps by comparing
    /// its predictions to the correct answers (ground truth). Each call to Train():
    /// 1. Runs the input image through the model (forward pass)
    /// 2. Compares the prediction to the expected output to compute the error (loss)
    /// 3. Calculates how each weight contributed to the error (backward pass / backpropagation)
    /// 4. Updates the weights using the optimizer (e.g., Adam) to reduce future errors
    ///
    /// You typically call this method many times in a loop over your training dataset (epochs).
    /// Training is only available in native mode — ONNX models are inference-only and will throw
    /// an <see cref="InvalidOperationException"/> if you attempt to train them.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model,
    /// which does not support training.</exception>
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
    /// Returns the architecture configuration (embedding dimensions, transformer depths,
    /// attention head counts, and decoder dimension) for a given SegFormer model size.
    /// </summary>
    /// <param name="modelSize">The SegFormer model size variant (B0–B5).</param>
    /// <returns>A tuple containing the embed dims per stage, depths per stage, heads per stage,
    /// and the decoder hidden dimension. These values come directly from the SegFormer paper.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each SegFormer size (B0 through B5) uses different numbers of
    /// transformer layers and feature dimensions. B0 is the smallest and fastest (3.8M parameters),
    /// while B5 is the largest and most accurate (82M parameters). The "embed dims" control
    /// how many features the model extracts at each of its 4 processing stages, and the "depths"
    /// control how many transformer blocks are stacked at each stage.
    /// </para>
    /// </remarks>
    private static (int[] EmbedDims, int[] Depths, int[] NumHeads, int DecoderDim) GetModelConfig(
        SegFormerModelSize modelSize)
    {
        return modelSize switch
        {
            SegFormerModelSize.B0 => ([32, 64, 160, 256], [2, 2, 2, 2], [1, 2, 5, 8], 256),
            SegFormerModelSize.B1 => ([64, 128, 320, 512], [2, 2, 2, 2], [1, 2, 5, 8], 256),
            SegFormerModelSize.B2 => ([64, 128, 320, 512], [3, 4, 6, 3], [1, 2, 5, 8], 768),
            SegFormerModelSize.B3 => ([64, 128, 320, 512], [3, 4, 18, 3], [1, 2, 5, 8], 768),
            SegFormerModelSize.B4 => ([64, 128, 320, 512], [3, 8, 27, 3], [1, 2, 5, 8], 768),
            SegFormerModelSize.B5 => ([64, 128, 320, 512], [3, 6, 40, 3], [1, 2, 5, 8], 768),
            _ => ([32, 64, 160, 256], [2, 2, 2, 2], [1, 2, 5, 8], 256)
        };
    }

    /// <summary>
    /// Executes the full forward pass through the SegFormer encoder and decoder in native mode.
    /// </summary>
    /// <param name="input">The input image tensor. Can be [C, H, W] (unbatched) or [B, C, H, W] (batched).</param>
    /// <returns>The segmentation logits tensor with shape [B, numClasses, outH, outW] or [numClasses, outH, outW]
    /// depending on whether the input was batched.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method processes an image through two main stages:
    /// 1. <b>Encoder</b>: The Mix Transformer (MiT) encoder processes the image through 4 stages,
    ///    each extracting features at a progressively lower resolution. This is similar to how your
    ///    eyes first see the big picture, then focus on finer details.
    /// 2. <b>Decoder</b>: The MLP decode head projects the encoder's features to the target number
    ///    of classes, producing a per-pixel prediction map.
    ///
    /// If the input doesn't have a batch dimension, one is temporarily added and then removed
    /// from the output for convenience.
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
        {
            input = AddBatchDimension(input);
        }

        // Encoder: extract multi-scale features through 4 transformer stages
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++)
        {
            features = Layers[i].Forward(features);
        }

        // Decoder: MLP decode head projects features to per-pixel class predictions
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
    /// Runs inference using the ONNX runtime session for optimized prediction.
    /// </summary>
    /// <param name="input">The input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>The segmentation logits tensor from the ONNX model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When using a pre-trained ONNX model, this method converts the input
    /// tensor to the format expected by the ONNX runtime (float arrays), runs the model, and
    /// converts the output back to the library's tensor format. The ONNX runtime automatically
    /// applies hardware-specific optimizations (e.g., CPU vectorization, GPU acceleration)
    /// for fast inference.
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

        // Convert input tensor to float array for ONNX
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        // Create ONNX input tensor
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputMeta = _onnxSession.InputMetadata;

        string inputName = inputMeta.Keys.FirstOrDefault() ?? "pixel_values";
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        // Run inference
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Convert output to our tensor format
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
    /// <param name="gradient">The loss gradient tensor computed from the difference between
    /// predicted and expected outputs.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation is how neural networks learn. After comparing the model's
    /// prediction to the correct answer, this method works backward through every layer — from the
    /// decoder output all the way back to the first encoder layer — calculating how much each weight
    /// contributed to the error. These gradients are then used by the optimizer to adjust the weights
    /// and improve future predictions.
    /// </para>
    /// </remarks>
    private void BackwardPass(Tensor<T> gradient)
    {
        if (!_useNativeMode || Layers.Count == 0)
        {
            return;
        }

        if (gradient.Rank == 3) gradient = AddBatchDimension(gradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }
    }

    /// <summary>
    /// Adds a batch dimension to an unbatched [C, H, W] tensor, producing [1, C, H, W].
    /// </summary>
    /// <param name="tensor">An unbatched image tensor with shape [channels, height, width].</param>
    /// <returns>A batched tensor with shape [1, channels, height, width] containing the same data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks typically process images in batches (groups). When you
    /// pass a single image, this method wraps it in a batch of size 1 so the network's layers
    /// can process it correctly. The batch dimension is removed from the output afterward.
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
    /// <param name="tensor">A batched tensor with shape [1, channels, height, width].</param>
    /// <returns>An unbatched tensor with shape [channels, height, width] containing the same data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After the network processes a single image (which was wrapped in a
    /// batch of size 1), this method strips the batch dimension off to return the output in the
    /// same format as the original input — just [channels, height, width].
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
    /// Initializes the encoder and decoder layers for the SegFormer model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method builds the internal structure of the SegFormer model.
    /// In native mode, it creates two groups of layers:
    /// 1. <b>Encoder layers</b> (Mix Transformer): A 4-stage hierarchy of overlapping patch embeddings
    ///    and transformer blocks that extract increasingly abstract features from the input image.
    /// 2. <b>Decoder layers</b> (MLP decode head): A lightweight set of projection and classification
    ///    layers that convert the encoder's features into per-pixel class predictions.
    ///
    /// In ONNX mode, no layers are created because the ONNX runtime handles everything internally.
    /// If custom layers are provided via the architecture, those are used instead of the defaults.
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
            // Assume user-provided layers have encoder as first half
            _encoderLayerEnd = Architecture.Layers.Count / 2;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateSegFormerEncoderLayers(
                _channels, _height, _width,
                _embedDims, _depths, _numHeads, _dropRate).ToList();

            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            // Compute encoder output spatial dimensions
            // Strides per stage: [4, 2, 2, 2] → total downscale = 32x
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

            int encoderOutputChannels = _embedDims[^1]; // last stage output
            var decoderLayers = LayerHelper<T>.CreateSegFormerDecoderLayers(
                encoderOutputChannels, _decoderDim, _numClasses,
                featureH, featureW);

            Layers.AddRange(decoderLayers);
        }
    }

    /// <summary>
    /// Updates all trainable parameters across the encoder and decoder layers from a flat parameter vector.
    /// </summary>
    /// <param name="parameters">A flat vector containing new values for all model parameters,
    /// ordered sequentially by layer (encoder layers first, then decoder layers).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A neural network's "parameters" are the numerical weights that determine
    /// its behavior. This method replaces all current weights with new values from a flat vector.
    /// It walks through each layer in order, slicing out the correct number of parameters for that
    /// layer and updating them. This is used internally during optimization and when loading
    /// saved model weights.
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
    /// Collects metadata describing this SegFormer model's configuration and current state.
    /// </summary>
    /// <returns>A <see cref="ModelMetadata{T}"/> object containing the model type, architecture
    /// configuration (input dimensions, number of classes, model size, decoder dimension, etc.),
    /// and serialized model data for persistence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Metadata is a summary of everything about the model — its type,
    /// input dimensions, how many classes it predicts, which size variant it is, and a serialized
    /// copy of its weights. This is useful for saving models to disk, comparing different model
    /// configurations, or displaying model information in a dashboard.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "SegFormer" },
            { "Description", "SegFormer Semantic Segmentation Model" },
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
    /// Writes all SegFormer-specific configuration values to a binary stream for model persistence.
    /// </summary>
    /// <param name="writer">The binary writer to serialize configuration data into.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you save a SegFormer model, the base class handles saving the
    /// layer weights, and this method saves the model's configuration — input dimensions, number
    /// of classes, model size, dropout rate, etc. All of these values are needed to reconstruct
    /// the model when loading it back later. The data is written in a specific order that must
    /// match the reading order in <see cref="DeserializeNetworkSpecificData"/>.
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

        // Write embed dims
        writer.Write(_embedDims.Length);
        foreach (int dim in _embedDims)
        {
            writer.Write(dim);
        }

        // Write depths
        writer.Write(_depths.Length);
        foreach (int depth in _depths)
        {
            writer.Write(depth);
        }

        // Write num heads
        writer.Write(_numHeads.Length);
        foreach (int head in _numHeads)
        {
            writer.Write(head);
        }
    }

    /// <summary>
    /// Reads SegFormer-specific configuration values from a binary stream during model loading.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize configuration data from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the counterpart to <see cref="SerializeNetworkSpecificData"/>.
    /// When loading a saved SegFormer model, this method reads back all the configuration values
    /// (input dimensions, classes, model size, etc.) in the exact same order they were written.
    /// The base class uses the read-only fields set during construction, so the values read here
    /// are consumed to advance the reader position.
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

        // Read embed dims
        int embedCount = reader.ReadInt32();
        for (int i = 0; i < embedCount; i++)
        {
            _ = reader.ReadInt32();
        }

        // Read depths
        int depthCount = reader.ReadInt32();
        for (int i = 0; i < depthCount; i++)
        {
            _ = reader.ReadInt32();
        }

        // Read num heads
        int headCount = reader.ReadInt32();
        for (int i = 0; i < headCount; i++)
        {
            _ = reader.ReadInt32();
        }
    }

    /// <summary>
    /// Creates a new SegFormer instance with the same configuration as this one but freshly
    /// initialized weights.
    /// </summary>
    /// <returns>A new <see cref="SegFormer{T}"/> model with the same architecture, model size,
    /// number of classes, and other settings, but with reinitialized layer weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a "copy" of the model's configuration (same size,
    /// same number of classes, same input dimensions) but with fresh random weights. It's used
    /// internally by the framework for operations like cross-validation, where multiple independent
    /// copies of the same model architecture need to be trained separately. In native mode it
    /// creates a new trainable model; in ONNX mode it reloads from the same ONNX file.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new SegFormer<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options);
        }
        else
        {
            return new SegFormer<T>(Architecture, _onnxModelPath!, _numClasses, _modelSize, _options);
        }
    }

    /// <summary>
    /// Releases managed resources held by this SegFormer instance, including the ONNX inference session.
    /// </summary>
    /// <param name="disposing"><c>true</c> when called from <see cref="IDisposable.Dispose"/>;
    /// <c>false</c> when called from a finalizer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you're done using a SegFormer model (especially in ONNX mode),
    /// calling Dispose() frees the memory and file handles used by the ONNX runtime session.
    /// In native mode there are no special resources to release, but it's still good practice
    /// to dispose the model when finished. Use a <c>using</c> statement to ensure disposal happens
    /// automatically: <c>using var model = new SegFormer&lt;float&gt;(...);</c>
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

    #region ISemanticSegmentation Implementation

    int ISegmentationModel<T>.NumClasses => _numClasses;
    int ISegmentationModel<T>.InputHeight => _height;
    int ISegmentationModel<T>.InputWidth => _width;
    bool ISegmentationModel<T>.IsOnnxMode => !_useNativeMode;
    Tensor<T> ISegmentationModel<T>.Segment(Tensor<T> image) => Predict(image);

    Tensor<T> ISemanticSegmentation<T>.GetClassMap(Tensor<T> image)
        => Common.SegmentationTensorOps.ArgmaxAlongClassDim(Predict(image));

    Tensor<T> ISemanticSegmentation<T>.GetProbabilityMap(Tensor<T> image)
        => Common.SegmentationTensorOps.SoftmaxAlongClassDim(Predict(image));

    #endregion
}
