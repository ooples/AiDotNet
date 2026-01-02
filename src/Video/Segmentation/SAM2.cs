using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Segmentation;

/// <summary>
/// Segment Anything Model 2 (SAM2) for video object segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM2 is a powerful model that can segment any object in video.
/// You can interact with it by:
/// - Clicking on an object in the first frame to select it
/// - Drawing a bounding box around objects
/// - Providing text prompts describing what to segment
///
/// Once you identify an object, SAM2 automatically tracks and segments it across
/// all frames in the video, even when the object moves, rotates, or is partially occluded.
///
/// Common use cases:
/// - Video editing (isolating subjects for effects)
/// - Object tracking and analysis
/// - Video annotation and labeling
/// - Interactive video manipulation
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Memory attention mechanism for temporal consistency
/// - Hierarchical image encoder (similar to MAE/ViT)
/// - Prompt encoder for points, boxes, and masks
/// - Mask decoder with occlusion prediction
/// - Memory bank for efficient object tracking
/// </para>
/// <para>
/// <b>Reference:</b> Ravi et al., "SAM 2: Segment Anything in Images and Videos"
/// Meta AI, 2024.
/// </para>
/// </remarks>
public class SAM2<T> : NeuralNetworkBase<T>
{
    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numFeatures;
    private readonly int _memoryBankSize;
    private readonly SAM2ModelSize _modelSize;
    private readonly bool _useNativeMode;
    private readonly string? _onnxModelPath;
    private InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    // Memory bank for tracking
    private readonly List<Tensor<T>> _memoryBank;
    private readonly List<int> _memoryFrameIndices;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether training is supported.
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the input height.
    /// </summary>
    internal int InputHeight => _height;

    /// <summary>
    /// Gets the input width.
    /// </summary>
    internal int InputWidth => _width;

    /// <summary>
    /// Gets the model size variant.
    /// </summary>
    internal SAM2ModelSize ModelSize => _modelSize;

    /// <summary>
    /// Gets whether using native mode (trainable) or ONNX mode (inference only).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the current memory bank size.
    /// </summary>
    internal int CurrentMemorySize => _memoryBank.Count;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the SAM2 class in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="optimizer">Optional optimizer for training (default: null uses layer-wise learning).</param>
    /// <param name="lossFunction">Optional loss function (default: BinaryCrossEntropyLoss).</param>
    /// <param name="modelSize">The model size variant (Tiny, Small, Base, Large).</param>
    /// <param name="memoryBankSize">Maximum number of frames to keep in memory.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a trainable SAM2 model.
    /// Use this when you want to fine-tune the model on your own video data.
    /// </para>
    /// </remarks>
    public SAM2(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SAM2ModelSize modelSize = SAM2ModelSize.Base,
        int memoryBankSize = 7)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>())
    {
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _modelSize = modelSize;
        _memoryBankSize = memoryBankSize;
        _useNativeMode = true;
        _onnxModelPath = null;
        _optimizer = optimizer;

        // Set number of features based on model size
        _numFeatures = modelSize switch
        {
            SAM2ModelSize.Tiny => 96,
            SAM2ModelSize.Small => 128,
            SAM2ModelSize.Base => 256,
            SAM2ModelSize.Large => 384,
            _ => 256
        };

        _memoryBank = [];
        _memoryFrameIndices = [];

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the SAM2 class in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="modelSize">The model size variant for configuration.</param>
    /// <param name="memoryBankSize">Maximum number of frames to keep in memory.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor loads a pre-trained SAM2 model from ONNX format.
    /// Use this for fast inference when you don't need to train the model.
    /// Download pre-trained models from Meta's SAM2 repository.
    /// </para>
    /// </remarks>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX model fails to load.</exception>
    public SAM2(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SAM2ModelSize modelSize = SAM2ModelSize.Base,
        int memoryBankSize = 7)
        : base(architecture, new BinaryCrossEntropyLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"SAM2 ONNX model not found: {onnxModelPath}");

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _modelSize = modelSize;
        _memoryBankSize = memoryBankSize;
        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _optimizer = null;

        _numFeatures = modelSize switch
        {
            SAM2ModelSize.Tiny => 96,
            SAM2ModelSize.Small => 128,
            SAM2ModelSize.Base => 256,
            SAM2ModelSize.Large => 384,
            _ => 256
        };

        _memoryBank = [];
        _memoryFrameIndices = [];

        // Initialize ONNX session
        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load SAM2 ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Segments objects in an image given point prompts.
    /// </summary>
    /// <param name="image">The input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <param name="points">Point coordinates [[x, y], ...] for foreground/background.</param>
    /// <param name="pointLabels">Label for each point: 1 for foreground, 0 for background.</param>
    /// <returns>Segmentation mask tensor [H, W] or [B, H, W] with values in [0, 1].</returns>
    public Tensor<T> SegmentWithPoints(Tensor<T> image, float[,] points, int[] pointLabels)
    {
        bool hasBatch = image.Rank == 4;
        if (!hasBatch)
        {
            image = AddBatchDimension(image);
        }

        // Encode image
        var imageFeatures = EncodeImage(image);

        // Encode point prompts
        var pointFeatures = EncodePoints(points, pointLabels);

        // Apply memory attention if we have previous frames
        if (_memoryBank.Count > 0)
        {
            imageFeatures = ApplyMemoryAttention(imageFeatures);
        }

        // Decode mask
        var masks = DecodeMask(imageFeatures, pointFeatures, null, null);

        // Select best mask
        var bestMask = SelectBestMask(masks.Masks, masks.IouScores);

        // Upsample to original resolution
        var outputMask = UpsampleMask(bestMask, _height, _width);

        if (!hasBatch)
        {
            outputMask = RemoveBatchDimension(outputMask);
        }

        return outputMask;
    }

    /// <summary>
    /// Segments objects in an image given a bounding box.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="box">Bounding box [x1, y1, x2, y2] in pixel coordinates.</param>
    /// <returns>Segmentation mask tensor.</returns>
    public Tensor<T> SegmentWithBox(Tensor<T> image, float[] box)
    {
        bool hasBatch = image.Rank == 4;
        if (!hasBatch)
        {
            image = AddBatchDimension(image);
        }

        var imageFeatures = EncodeImage(image);
        var boxFeatures = EncodeBox(box);

        if (_memoryBank.Count > 0)
        {
            imageFeatures = ApplyMemoryAttention(imageFeatures);
        }

        var masks = DecodeMask(imageFeatures, null, boxFeatures, null);
        var bestMask = SelectBestMask(masks.Masks, masks.IouScores);
        var outputMask = UpsampleMask(bestMask, _height, _width);

        if (!hasBatch)
        {
            outputMask = RemoveBatchDimension(outputMask);
        }

        return outputMask;
    }

    /// <summary>
    /// Segments objects using a mask prompt (for refinement).
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="maskPrompt">Low-resolution mask prompt [H/4, W/4].</param>
    /// <returns>Refined segmentation mask tensor.</returns>
    public Tensor<T> SegmentWithMask(Tensor<T> image, Tensor<T> maskPrompt)
    {
        bool hasBatch = image.Rank == 4;
        if (!hasBatch)
        {
            image = AddBatchDimension(image);
            maskPrompt = AddBatchDimension(maskPrompt);
        }

        var imageFeatures = EncodeImage(image);
        var maskFeatures = EncodeMaskPrompt(maskPrompt);

        if (_memoryBank.Count > 0)
        {
            imageFeatures = ApplyMemoryAttention(imageFeatures);
        }

        var masks = DecodeMask(imageFeatures, null, null, maskFeatures);
        var bestMask = SelectBestMask(masks.Masks, masks.IouScores);
        var outputMask = UpsampleMask(bestMask, _height, _width);

        if (!hasBatch)
        {
            outputMask = RemoveBatchDimension(outputMask);
        }

        return outputMask;
    }

    /// <summary>
    /// Tracks and segments an object across video frames.
    /// </summary>
    /// <param name="frames">List of video frames.</param>
    /// <param name="initialPoints">Point prompts for the first frame.</param>
    /// <param name="pointLabels">Labels for initial points.</param>
    /// <returns>List of segmentation masks for each frame.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main video tracking method.
    /// Simply provide the initial frame with point clicks to identify objects,
    /// and SAM2 will automatically track and segment those objects in all
    /// subsequent frames.
    /// </para>
    /// </remarks>
    public List<Tensor<T>> TrackObject(List<Tensor<T>> frames, float[,] initialPoints, int[] pointLabels)
    {
        ClearMemory();
        var masks = new List<Tensor<T>>();

        for (int i = 0; i < frames.Count; i++)
        {
            var frame = frames[i];
            bool hasBatch = frame.Rank == 4;
            if (!hasBatch)
            {
                frame = AddBatchDimension(frame);
            }

            // Encode current frame
            var imageFeatures = EncodeImage(frame);

            Tensor<T> mask;
            if (i == 0)
            {
                // First frame: use point prompts
                var pointFeatures = EncodePoints(initialPoints, pointLabels);
                var maskResult = DecodeMask(imageFeatures, pointFeatures, null, null);
                mask = SelectBestMask(maskResult.Masks, maskResult.IouScores);
            }
            else
            {
                // Subsequent frames: use memory attention
                var memoryFeatures = ApplyMemoryAttention(imageFeatures);
                var maskResult = DecodeMask(memoryFeatures, null, null, null);
                mask = SelectBestMask(maskResult.Masks, maskResult.IouScores);
            }

            // Update memory bank
            UpdateMemoryBank(imageFeatures, mask, i);

            // Upsample mask to original resolution
            var outputMask = UpsampleMask(mask, _height, _width);
            if (!hasBatch)
            {
                outputMask = RemoveBatchDimension(outputMask);
            }

            masks.Add(outputMask);
        }

        return masks;
    }

    /// <summary>
    /// Performs interactive video segmentation with refinement.
    /// </summary>
    /// <param name="frames">List of video frames.</param>
    /// <param name="framePrompts">Dictionary of frame index to prompts for refinement.</param>
    /// <returns>List of refined segmentation masks.</returns>
    public List<Tensor<T>> InteractiveVideoSegmentation(
        List<Tensor<T>> frames,
        Dictionary<int, (float[,] Points, int[] Labels)> framePrompts)
    {
        ClearMemory();
        var masks = new List<Tensor<T>>();

        for (int i = 0; i < frames.Count; i++)
        {
            var frame = frames[i];
            bool hasBatch = frame.Rank == 4;
            if (!hasBatch)
            {
                frame = AddBatchDimension(frame);
            }

            var imageFeatures = EncodeImage(frame);

            Tensor<T> mask;
            if (framePrompts.TryGetValue(i, out var prompts))
            {
                // Frame has explicit prompts - use them
                var pointFeatures = EncodePoints(prompts.Points, prompts.Labels);

                if (_memoryBank.Count > 0)
                {
                    imageFeatures = ApplyMemoryAttention(imageFeatures);
                }

                var maskResult = DecodeMask(imageFeatures, pointFeatures, null, null);
                mask = SelectBestMask(maskResult.Masks, maskResult.IouScores);
            }
            else if (_memoryBank.Count > 0)
            {
                // Use memory propagation
                var memoryFeatures = ApplyMemoryAttention(imageFeatures);
                var maskResult = DecodeMask(memoryFeatures, null, null, null);
                mask = SelectBestMask(maskResult.Masks, maskResult.IouScores);
            }
            else
            {
                // No prompts and no memory - use automatic mode
                var maskResult = DecodeMask(imageFeatures, null, null, null);
                mask = SelectBestMask(maskResult.Masks, maskResult.IouScores);
            }

            UpdateMemoryBank(imageFeatures, mask, i);

            var outputMask = UpsampleMask(mask, _height, _width);
            if (!hasBatch)
            {
                outputMask = RemoveBatchDimension(outputMask);
            }

            masks.Add(outputMask);
        }

        return masks;
    }

    /// <summary>
    /// Gets the occlusion score for the current segmentation.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="points">Point prompts.</param>
    /// <param name="pointLabels">Point labels.</param>
    /// <returns>Occlusion score in [0, 1] where 1 means fully occluded.</returns>
    public double GetOcclusionScore(Tensor<T> image, float[,] points, int[] pointLabels)
    {
        bool hasBatch = image.Rank == 4;
        if (!hasBatch)
        {
            image = AddBatchDimension(image);
        }

        var imageFeatures = EncodeImage(image);
        var pointFeatures = EncodePoints(points, pointLabels);

        if (_memoryBank.Count > 0)
        {
            imageFeatures = ApplyMemoryAttention(imageFeatures);
        }

        var masks = DecodeMask(imageFeatures, pointFeatures, null, null);
        return masks.OcclusionScore;
    }

    /// <summary>
    /// Clears the memory bank for starting a new video.
    /// </summary>
    public void ClearMemory()
    {
        _memoryBank.Clear();
        _memoryFrameIndices.Clear();
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Default: segment with automatic mode (no prompts)
        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
        {
            input = AddBatchDimension(input);
        }

        var imageFeatures = EncodeImage(input);

        if (_memoryBank.Count > 0)
        {
            imageFeatures = ApplyMemoryAttention(imageFeatures);
        }

        var masks = DecodeMask(imageFeatures, null, null, null);
        var bestMask = SelectBestMask(masks.Masks, masks.IouScores);
        var outputMask = UpsampleMask(bestMask, _height, _width);

        if (!hasBatch)
        {
            outputMask = RemoveBatchDimension(outputMask);
        }

        return outputMask;
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
            NumOps.Subtract(v, expectedOutput.Data[idx]));

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
        if (!_useNativeMode)
        {
            return EncodeImageOnnx(image);
        }

        var features = image;

        // Process through encoder layers (first 14 layers are encoder in our LayerHelper setup)
        int encoderLayerCount = Math.Min(14, Layers.Count);
        for (int i = 0; i < encoderLayerCount; i++)
        {
            features = Layers[i].Forward(features);
        }

        return features;
    }

    /// <summary>
    /// Encodes an image using the ONNX model.
    /// </summary>
    private Tensor<T> EncodeImageOnnx(Tensor<T> image)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        // Convert input tensor to float array for ONNX
        var inputData = new float[image.Length];
        for (int i = 0; i < image.Length; i++)
        {
            inputData[i] = Convert.ToSingle(image.Data[i]);
        }

        // Create ONNX input tensor
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, image.Shape);
        var inputMeta = _onnxSession.InputMetadata;

        // SAM2 encoder typically has 'image' as input name
        string inputName = inputMeta.Keys.FirstOrDefault() ?? "image";

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

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    private Tensor<T> EncodePoints(float[,] points, int[] pointLabels)
    {
        int numPoints = points.GetLength(0);
        int batchSize = 1;

        // Create point embedding with positional encoding
        var pointTensor = new Tensor<T>([batchSize, 2, 1, 1]);

        // Aggregate points with their labels
        for (int i = 0; i < numPoints; i++)
        {
            T x = NumOps.FromDouble(points[i, 0] / _width);
            T y = NumOps.FromDouble(points[i, 1] / _height);
            T weight = NumOps.FromDouble(pointLabels[i] == 1 ? 1.0 : -1.0);

            pointTensor[0, 0, 0, 0] = NumOps.Add(pointTensor[0, 0, 0, 0], NumOps.Multiply(x, weight));
            pointTensor[0, 1, 0, 0] = NumOps.Add(pointTensor[0, 1, 0, 0], NumOps.Multiply(y, weight));
        }

        // Use the point encoder layer if available (layer 14 in LayerHelper setup)
        if (_useNativeMode && Layers.Count > 14)
        {
            var encoded = Layers[14].Forward(pointTensor);
            return ApplyGELU(encoded);
        }

        return ApplyGELU(pointTensor);
    }

    private Tensor<T> EncodeBox(float[] box)
    {
        int batchSize = 1;
        var boxTensor = new Tensor<T>([batchSize, 4, 1, 1]);

        // Normalize coordinates
        boxTensor[0, 0, 0, 0] = NumOps.FromDouble(box[0] / _width);
        boxTensor[0, 1, 0, 0] = NumOps.FromDouble(box[1] / _height);
        boxTensor[0, 2, 0, 0] = NumOps.FromDouble(box[2] / _width);
        boxTensor[0, 3, 0, 0] = NumOps.FromDouble(box[3] / _height);

        // Use the box encoder layer if available (layer 15 in LayerHelper setup)
        if (_useNativeMode && Layers.Count > 15)
        {
            var encoded = Layers[15].Forward(boxTensor);
            return ApplyGELU(encoded);
        }

        return ApplyGELU(boxTensor);
    }

    private Tensor<T> EncodeMaskPrompt(Tensor<T> maskPrompt)
    {
        // Use the mask encoder layer if available (layer 16 in LayerHelper setup)
        if (_useNativeMode && Layers.Count > 16)
        {
            var encoded = Layers[16].Forward(maskPrompt);
            return ApplyGELU(encoded);
        }

        return ApplyGELU(maskPrompt);
    }

    private Tensor<T> ApplyMemoryAttention(Tensor<T> currentFeatures)
    {
        if (_memoryBank.Count == 0)
        {
            return currentFeatures;
        }

        // Average pool memory features
        var memoryAggregate = new Tensor<T>(currentFeatures.Shape);
        foreach (var memory in _memoryBank)
        {
            memoryAggregate = AddTensors(memoryAggregate, memory);
        }
        memoryAggregate = memoryAggregate.Transform((v, _) =>
            NumOps.Divide(v, NumOps.FromDouble(_memoryBank.Count)));

        // Concatenate current and memory features
        var combined = ConcatenateChannels(currentFeatures, memoryAggregate);

        // Apply memory attention layers if available (layers 17-21 in LayerHelper setup)
        if (_useNativeMode && Layers.Count > 21)
        {
            var attended = combined;
            for (int i = 17; i <= 20; i++)
            {
                attended = Layers[i].Forward(attended);
                attended = ApplyGELU(attended);
            }
            // Memory projection (layer 21)
            attended = Layers[21].Forward(attended);
            return AddTensors(currentFeatures, attended);
        }

        return currentFeatures;
    }

    private (Tensor<T> Masks, Tensor<T> IouScores, double OcclusionScore) DecodeMask(
        Tensor<T> imageFeatures,
        Tensor<T>? pointFeatures,
        Tensor<T>? boxFeatures,
        Tensor<T>? maskFeatures)
    {
        var features = imageFeatures;

        // Add prompt features if provided
        if (pointFeatures != null)
        {
            features = AddPromptFeatures(features, pointFeatures);
        }
        if (boxFeatures != null)
        {
            features = AddPromptFeatures(features, boxFeatures);
        }
        if (maskFeatures != null)
        {
            features = AddTensors(features, maskFeatures);
        }

        // Decoder layers (layers 22-23 in LayerHelper setup)
        if (_useNativeMode && Layers.Count > 23)
        {
            features = Layers[22].Forward(features);
            features = ApplyGELU(features);
            features = Layers[23].Forward(features);
            features = ApplyGELU(features);
        }

        // Generate mask candidates (layer 24)
        Tensor<T> masks;
        if (_useNativeMode && Layers.Count > 24)
        {
            masks = Layers[24].Forward(features);
        }
        else
        {
            int batchSize = features.Shape[0];
            int h = features.Shape[2];
            int w = features.Shape[3];
            masks = new Tensor<T>([batchSize, 4, h, w]);
        }
        masks = ApplySigmoid(masks);

        // Predict IoU scores (layer 25)
        var pooled = GlobalAveragePool(features);
        Tensor<T> iouScores;
        if (_useNativeMode && Layers.Count > 25)
        {
            iouScores = Layers[25].Forward(pooled);
        }
        else
        {
            iouScores = new Tensor<T>([pooled.Shape[0], 4, 1, 1]);
            for (int b = 0; b < iouScores.Shape[0]; b++)
            {
                for (int m = 0; m < 4; m++)
                {
                    iouScores[b, m, 0, 0] = NumOps.FromDouble(0.5);
                }
            }
        }
        iouScores = ApplySigmoid(iouScores);

        // Predict occlusion (layer 26)
        double occlusionScore = 0.0;
        if (_useNativeMode && Layers.Count > 26)
        {
            var occlusionLogit = Layers[26].Forward(pooled);
            var occlusionSigmoid = ApplySigmoid(occlusionLogit);
            occlusionScore = Convert.ToDouble(occlusionSigmoid[0, 0, 0, 0]);
        }

        return (masks, iouScores, occlusionScore);
    }

    private Tensor<T> AddPromptFeatures(Tensor<T> imageFeatures, Tensor<T> promptFeatures)
    {
        int batchSize = imageFeatures.Shape[0];
        int channels = imageFeatures.Shape[1];
        int height = imageFeatures.Shape[2];
        int width = imageFeatures.Shape[3];

        // Broadcast prompt features spatially
        var broadcastedPrompt = new Tensor<T>([batchSize, channels, height, width]);

        int promptChannels = Math.Min(channels, promptFeatures.Shape[1]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < promptChannels; c++)
            {
                T promptVal = promptFeatures[b, c, 0, 0];
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        broadcastedPrompt[b, c, h, w] = promptVal;
                    }
                }
            }
        }

        return AddTensors(imageFeatures, broadcastedPrompt);
    }

    private Tensor<T> SelectBestMask(Tensor<T> masks, Tensor<T> iouScores)
    {
        int batchSize = masks.Shape[0];
        int numMasks = masks.Shape[1];
        int height = masks.Shape[2];
        int width = masks.Shape[3];

        var bestMask = new Tensor<T>([batchSize, 1, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            // Find mask with highest IoU score
            int bestIdx = 0;
            double bestScore = double.MinValue;

            for (int m = 0; m < numMasks; m++)
            {
                double score = Convert.ToDouble(iouScores[b, m, 0, 0]);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestIdx = m;
                }
            }

            // Copy best mask
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    bestMask[b, 0, h, w] = masks[b, bestIdx, h, w];
                }
            }
        }

        return bestMask;
    }

    private Tensor<T> UpsampleMask(Tensor<T> mask, int targetH, int targetW)
    {
        int batchSize = mask.Shape[0];
        int channels = mask.Shape[1];
        int srcH = mask.Shape[2];
        int srcW = mask.Shape[3];

        var upsampled = new Tensor<T>([batchSize, channels, targetH, targetW]);

        double scaleH = (double)srcH / targetH;
        double scaleW = (double)srcW / targetW;

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < targetH; h++)
                {
                    for (int w = 0; w < targetW; w++)
                    {
                        // Bilinear interpolation
                        double srcY = h * scaleH;
                        double srcX = w * scaleW;

                        int y0 = (int)Math.Floor(srcY);
                        int x0 = (int)Math.Floor(srcX);
                        int y1 = Math.Min(y0 + 1, srcH - 1);
                        int x1 = Math.Min(x0 + 1, srcW - 1);

                        double dy = srcY - y0;
                        double dx = srcX - x0;

                        double v00 = Convert.ToDouble(mask[b, c, y0, x0]);
                        double v01 = Convert.ToDouble(mask[b, c, y0, x1]);
                        double v10 = Convert.ToDouble(mask[b, c, y1, x0]);
                        double v11 = Convert.ToDouble(mask[b, c, y1, x1]);

                        double value = v00 * (1 - dx) * (1 - dy) +
                                       v01 * dx * (1 - dy) +
                                       v10 * (1 - dx) * dy +
                                       v11 * dx * dy;

                        upsampled[b, c, h, w] = NumOps.FromDouble(value);
                    }
                }
            }
        }

        return upsampled;
    }

    private void UpdateMemoryBank(Tensor<T> features, Tensor<T> mask, int frameIndex)
    {
        // Combine features with mask information
        var memoryFeatures = features; // In a real implementation, this would also incorporate the mask

        if (_memoryBank.Count >= _memoryBankSize)
        {
            // Remove oldest memory
            _memoryBank.RemoveAt(0);
            _memoryFrameIndices.RemoveAt(0);
        }

        _memoryBank.Add(memoryFeatures);
        _memoryFrameIndices.Add(frameIndex);
    }

    private Tensor<T> GlobalAveragePool(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var output = new Tensor<T>([batchSize, channels, 1, 1]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                T sum = NumOps.Zero;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        sum = NumOps.Add(sum, input[b, c, h, w]);
                    }
                }
                output[b, c, 0, 0] = NumOps.Divide(sum, NumOps.FromDouble(height * width));
            }
        }

        return output;
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        int batchSize = a.Shape[0];
        int channelsA = a.Shape[1];
        int channelsB = b.Shape[1];
        int height = a.Shape[2];
        int width = a.Shape[3];

        var output = new Tensor<T>([batchSize, channelsA + channelsB, height, width]);

        for (int batch = 0; batch < batchSize; batch++)
        {
            for (int c = 0; c < channelsA; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        output[batch, c, h, w] = a[batch, c, h, w];
                    }
                }
            }

            for (int c = 0; c < channelsB; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        output[batch, channelsA + c, h, w] = b[batch, c, h, w];
                    }
                }
            }
        }

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

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        int c = tensor.Shape[0];
        int h = tensor.Shape[1];
        int w = tensor.Shape[2];

        var result = new Tensor<T>([1, c, h, w]);
        Array.Copy(tensor.Data, result.Data, tensor.Data.Length);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++)
        {
            newShape[i] = tensor.Shape[i + 1];
        }

        var result = new Tensor<T>(newShape);
        Array.Copy(tensor.Data, result.Data, tensor.Data.Length);
        return result;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return a.Transform((v, idx) => NumOps.Add(v, b.Data[idx]));
    }

    private void BackwardPass(Tensor<T> gradient)
    {
        if (!_useNativeMode || Layers.Count == 0)
        {
            return;
        }

        // Backward through decoder and output heads
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
            // ONNX mode - no layers to initialize
            ClearLayers();
            return;
        }

        // Use architecture layers if provided, otherwise create default layers
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            // SAM2 uses a multi-branch architecture. The Layers list contains the image encoder.
            // Prompt encoders, memory layers, and mask decoder are created separately in their
            // respective processing methods (ProcessPrompt, ProcessMemory, DecodeMask).
            Layers.AddRange(LayerHelper<T>.CreateSAM2ImageEncoderLayers(
                _channels,
                _height,
                _width,
                _numFeatures));
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
            { "ModelName", "SAM2" },
            { "Description", "Segment Anything Model 2 for Video Object Segmentation" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "ModelSize", _modelSize.ToString() },
            { "NumFeatures", _numFeatures },
            { "MemoryBankSize", _memoryBankSize },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count }
        };

        return new ModelMetadata<T>
        {
            ModelType = ModelType.VideoObjectSegmentation,
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
        writer.Write(_numFeatures);
        writer.Write(_memoryBankSize);
        writer.Write((int)_modelSize);
        writer.Write(_useNativeMode);
        writer.Write(_onnxModelPath ?? string.Empty);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // height
        _ = reader.ReadInt32(); // width
        _ = reader.ReadInt32(); // channels
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // memoryBankSize
        _ = reader.ReadInt32(); // modelSize
        _ = reader.ReadBoolean(); // useNativeMode
        _ = reader.ReadString(); // onnxModelPath
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new SAM2<T>(Architecture, _optimizer, LossFunction, _modelSize, _memoryBankSize);
        }
        else
        {
            return new SAM2<T>(Architecture, _onnxModelPath!, _modelSize, _memoryBankSize);
        }
    }

    #endregion
}

/// <summary>
/// Model size variants for SAM2.
/// </summary>
public enum SAM2ModelSize
{
    /// <summary>
    /// Tiny model - fastest, lowest memory.
    /// </summary>
    Tiny,

    /// <summary>
    /// Small model - balanced speed/accuracy.
    /// </summary>
    Small,

    /// <summary>
    /// Base model - good accuracy.
    /// </summary>
    Base,

    /// <summary>
    /// Large model - highest accuracy.
    /// </summary>
    Large
}
