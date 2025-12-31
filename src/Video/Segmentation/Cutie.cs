using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Segmentation;

/// <summary>
/// Cutie: Cutting-edge Video Instance Segmentation with transformer memory.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Cutie is designed for semi-supervised video object segmentation (VOS).
/// Given a mask for an object in the first frame, Cutie tracks and segments that object
/// throughout the entire video with high accuracy.
/// </para>
/// <para>
/// <b>For Beginners:</b> Cutie tracks objects in video. You provide a mask showing
/// where the object is in the first frame, and Cutie finds that same object in all
/// following frames - even when it moves, changes shape, or gets partially hidden.
///
/// Key features:
/// - Object permanence understanding (tracks objects even when briefly occluded)
/// - Efficient memory management for long videos
/// - High-quality mask predictions
/// - Multi-object tracking support
///
/// Example usage (native mode for training):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 480, inputWidth: 854, inputDepth: 3);
/// var model = new Cutie&lt;double&gt;(arch);
/// var masks = model.TrackObject(videoFrames, initialMask);
/// </code>
///
/// Example usage (ONNX mode for inference):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 480, inputWidth: 854, inputDepth: 3);
/// var model = new Cutie&lt;double&gt;(arch, "cutie.onnx");
/// var masks = model.TrackObject(videoFrames, initialMask);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Putting the Object Back into Video Object Segmentation"
/// https://arxiv.org/abs/2310.12982
/// </para>
/// </remarks>
public class Cutie<T> : NeuralNetworkBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this model uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for the model.
    /// </summary>
    private readonly InferenceSession? _onnxSession;

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// The optimizer used for training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// The loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Input frame height.
    /// </summary>
    private readonly int _inputHeight;

    /// <summary>
    /// Input frame width.
    /// </summary>
    private readonly int _inputWidth;

    /// <summary>
    /// Number of input channels.
    /// </summary>
    private readonly int _inputChannels;

    /// <summary>
    /// Feature dimension for the model.
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// Maximum size of the memory bank.
    /// </summary>
    private readonly int _memorySize;

    /// <summary>
    /// Memory bank storing key-value pairs for object tracking.
    /// </summary>
    private readonly List<(Tensor<T> Key, Tensor<T> Value)> _memoryBank;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether this model uses native mode (true) or ONNX mode (false).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets whether training is supported (only in native mode).
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the input height.
    /// </summary>
    internal int InputHeight => _inputHeight;

    /// <summary>
    /// Gets the input width.
    /// </summary>
    internal int InputWidth => _inputWidth;

    /// <summary>
    /// Gets the maximum memory size.
    /// </summary>
    internal int MemorySize => _memorySize;

    /// <summary>
    /// Gets the current number of items in the memory bank.
    /// </summary>
    internal int CurrentMemoryCount => _memoryBank.Count;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Cutie model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">Architecture for the segmentation network.</param>
    /// <param name="optimizer">Optional optimizer for training. Default: Adam.</param>
    /// <param name="lossFunction">Optional loss function. Default: Binary cross-entropy.</param>
    /// <param name="numFeatures">Feature dimension (default: 256).</param>
    /// <param name="memorySize">Maximum memory bank size (default: 50).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a trainable Cutie model:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 480, inputWidth: 854, inputDepth: 3);
    /// var model = new Cutie&lt;double&gt;(arch);
    /// </code>
    /// </para>
    /// </remarks>
    public Cutie(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numFeatures = 256,
        int memorySize = 50)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>())
    {
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), numFeatures, "Feature dimension must be at least 1.");
        if (memorySize < 1)
            throw new ArgumentOutOfRangeException(nameof(memorySize), memorySize, "Memory size must be at least 1.");

        _useNativeMode = true;
        _inputHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _inputWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 854;
        _inputChannels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = numFeatures;
        _memorySize = memorySize;
        _memoryBank = [];

        _lossFunction = lossFunction ?? new BinaryCrossEntropyLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a Cutie model using a pretrained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model.</param>
    /// <param name="memorySize">Maximum memory bank size (default: 50).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained model
    /// in ONNX format. Training is not supported in ONNX mode.
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 480, inputWidth: 854, inputDepth: 3);
    /// var model = new Cutie&lt;double&gt;(arch, "cutie.onnx");
    /// var masks = model.TrackObject(frames, initialMask);
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    public Cutie(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int memorySize = 50)
        : base(architecture, new BinaryCrossEntropyLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"Cutie ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _inputHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _inputWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 854;
        _inputChannels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = 256;
        _memorySize = memorySize;
        _memoryBank = [];
        _lossFunction = new BinaryCrossEntropyLoss<T>();

        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Tracks and segments an object across video frames.
    /// </summary>
    /// <param name="frames">List of video frames.</param>
    /// <param name="initialMask">Object mask for the first frame.</param>
    /// <returns>List of segmentation masks for each frame.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for tracking objects.
    /// Provide video frames and a mask showing the object in the first frame.
    /// The method returns masks for the object in all frames.
    /// </para>
    /// </remarks>
    public List<Tensor<T>> TrackObject(List<Tensor<T>> frames, Tensor<T> initialMask)
    {
        if (frames is null || frames.Count == 0)
            throw new ArgumentException("Frames list cannot be null or empty.", nameof(frames));
        if (initialMask is null)
            throw new ArgumentNullException(nameof(initialMask));

        ClearMemory();
        var masks = new List<Tensor<T>>();

        for (int i = 0; i < frames.Count; i++)
        {
            var frame = frames[i];
            bool hasBatch = frame.Rank == 4;
            if (!hasBatch) frame = AddBatchDimension(frame);

            Tensor<T> mask;
            if (i == 0)
            {
                mask = ProcessFirstFrame(frame, initialMask);
            }
            else
            {
                mask = PropagateFromMemory(frame);
            }

            UpdateMemory(frame, mask);

            if (!hasBatch) mask = RemoveBatchDimension(mask);
            masks.Add(mask);
        }

        return masks;
    }

    /// <summary>
    /// Segments a single frame using the current memory state.
    /// </summary>
    /// <param name="frame">Input video frame.</param>
    /// <returns>Segmentation mask for the frame.</returns>
    public Tensor<T> SegmentFrame(Tensor<T> frame)
    {
        if (frame is null)
            throw new ArgumentNullException(nameof(frame));

        bool hasBatch = frame.Rank == 4;
        if (!hasBatch) frame = AddBatchDimension(frame);

        var mask = PropagateFromMemory(frame);

        if (!hasBatch) mask = RemoveBatchDimension(mask);
        return mask;
    }

    /// <summary>
    /// Adds a new object to track by storing its features in memory.
    /// </summary>
    /// <param name="frame">Frame containing the object.</param>
    /// <param name="mask">Mask indicating the object location.</param>
    public void AddObject(Tensor<T> frame, Tensor<T> mask)
    {
        if (frame is null)
            throw new ArgumentNullException(nameof(frame));
        if (mask is null)
            throw new ArgumentNullException(nameof(mask));

        if (frame.Rank == 3) frame = AddBatchDimension(frame);
        if (mask.Rank == 3) mask = AddBatchDimension(mask);

        var features = EncodeImage(frame);
        var objectFeatures = EncodeObject(features, mask);

        // Store key-value pair in memory
        _memoryBank.Add((objectFeatures, objectFeatures));
        TrimMemory();
    }

    /// <summary>
    /// Clears the memory bank.
    /// </summary>
    public void ClearMemory() => _memoryBank.Clear();

    #endregion

    #region Inference

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var result = input;
        foreach (var layer in Layers)
        {
            result = layer.Forward(result);
        }
        return result;
    }

    /// <summary>
    /// Performs inference using the ONNX model.
    /// </summary>
    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

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

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return SegmentFrame(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use native mode for training.");

        var prediction = Predict(input);
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

        var currentGradient = outputGradientTensor;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
        }

        if (_optimizer != null)
        {
            _optimizer.UpdateParameters(Layers);
        }
    }

    #endregion

    #region Private Methods

    private Tensor<T> ProcessFirstFrame(Tensor<T> frame, Tensor<T> mask)
    {
        var features = EncodeImage(frame);

        if (mask.Rank == 3) mask = AddBatchDimension(mask);
        var downsampledMask = DownsampleMask(mask, features.Shape[2], features.Shape[3]);

        var objectFeatures = EncodeObject(features, downsampledMask);
        _memoryBank.Add((objectFeatures, objectFeatures));

        return UpsampleMask(downsampledMask, _inputHeight, _inputWidth);
    }

    private Tensor<T> PropagateFromMemory(Tensor<T> frame)
    {
        if (_useNativeMode)
        {
            var features = EncodeImage(frame);
            var attended = AttendToMemory(features);
            return DecodeMask(attended);
        }
        else
        {
            return PredictOnnx(frame);
        }
    }

    private void UpdateMemory(Tensor<T> frame, Tensor<T> mask)
    {
        var features = EncodeImage(frame);
        var downsampledMask = DownsampleMask(mask, features.Shape[2], features.Shape[3]);
        var objectFeatures = EncodeObject(features, downsampledMask);

        _memoryBank.Add((objectFeatures, objectFeatures));
        TrimMemory();
    }

    private void TrimMemory()
    {
        while (_memoryBank.Count > _memorySize)
            _memoryBank.RemoveAt(0);
    }

    private Tensor<T> EncodeImage(Tensor<T> frame)
    {
        // Use first 5 layers (image encoder)
        var features = frame;
        for (int i = 0; i < 5 && i < Layers.Count; i++)
        {
            features = Layers[i].Forward(features);
        }
        return features;
    }

    private Tensor<T> EncodeObject(Tensor<T> imageFeatures, Tensor<T> mask)
    {
        var concat = ConcatenateChannels(imageFeatures, mask);

        // Use object encoder layers (indices 5-6)
        var features = concat;
        for (int i = 5; i < 7 && i < Layers.Count; i++)
        {
            features = Layers[i].Forward(features);
        }
        return features;
    }

    private Tensor<T> AttendToMemory(Tensor<T> query)
    {
        if (_memoryBank.Count == 0) return query;

        int batchSize = query.Shape[0];
        int channels = query.Shape[1];
        int height = query.Shape[2];
        int width = query.Shape[3];

        var attended = new Tensor<T>(query.Shape);

        foreach (var (key, value) in _memoryBank)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        double score = 0;
                        for (int c = 0; c < channels; c++)
                            score += Convert.ToDouble(query[b, c, h, w]) * Convert.ToDouble(key[b, c, h, w]);

                        double weight = Math.Exp(score / Math.Sqrt(channels));

                        for (int c = 0; c < channels; c++)
                        {
                            double v = Convert.ToDouble(value[b, c, h, w]);
                            attended[b, c, h, w] = NumOps.Add(attended[b, c, h, w], NumOps.FromDouble(v * weight));
                        }
                    }
                }
            }
        }

        // Normalize
        double norm = _memoryBank.Count;
        attended = attended.Transform((v, _) => NumOps.FromDouble(Convert.ToDouble(v) / norm));

        // Process through attention layers (indices 10-13)
        for (int i = 10; i < 14 && i < Layers.Count; i++)
        {
            attended = Layers[i].Forward(attended);
        }

        return attended;
    }

    private Tensor<T> DecodeMask(Tensor<T> features)
    {
        // Use decoder layers (indices 14+)
        var decoded = features;
        for (int i = 14; i < Layers.Count; i++)
        {
            if (i < Layers.Count - 1)
            {
                decoded = Upsample2x(decoded);
            }
            decoded = Layers[i].Forward(decoded);
        }

        // Final upsampling if needed
        while (decoded.Shape[2] < _inputHeight || decoded.Shape[3] < _inputWidth)
            decoded = Upsample2x(decoded);

        return decoded;
    }

    private Tensor<T> DownsampleMask(Tensor<T> mask, int targetH, int targetW)
    {
        int batchSize = mask.Shape[0];
        int srcH = mask.Shape[2];
        int srcW = mask.Shape[3];

        var downsampled = new Tensor<T>([batchSize, 1, targetH, targetW]);
        double scaleH = (double)srcH / targetH;
        double scaleW = (double)srcW / targetW;

        for (int b = 0; b < batchSize; b++)
            for (int h = 0; h < targetH; h++)
                for (int w = 0; w < targetW; w++)
                {
                    int srcY = Math.Min((int)(h * scaleH), srcH - 1);
                    int srcX = Math.Min((int)(w * scaleW), srcW - 1);
                    downsampled[b, 0, h, w] = mask[b, 0, srcY, srcX];
                }

        return downsampled;
    }

    private Tensor<T> UpsampleMask(Tensor<T> mask, int targetH, int targetW)
    {
        int batchSize = mask.Shape[0];
        int srcH = mask.Shape[2];
        int srcW = mask.Shape[3];

        var upsampled = new Tensor<T>([batchSize, 1, targetH, targetW]);
        double scaleH = (double)srcH / targetH;
        double scaleW = (double)srcW / targetW;

        for (int b = 0; b < batchSize; b++)
            for (int h = 0; h < targetH; h++)
                for (int w = 0; w < targetW; w++)
                {
                    double srcY = h * scaleH;
                    double srcX = w * scaleW;
                    int y0 = Math.Max(0, Math.Min((int)srcY, srcH - 1));
                    int x0 = Math.Max(0, Math.Min((int)srcX, srcW - 1));
                    upsampled[b, 0, h, w] = mask[b, 0, y0, x0];
                }

        return upsampled;
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
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        output[batch, c, h, w] = a[batch, c, h, w];

            for (int c = 0; c < channelsB; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        output[batch, channelsA + c, h, w] = b[batch, c, h, w];
        }

        return output;
    }

    private Tensor<T> Upsample2x(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var output = new Tensor<T>([batchSize, channels, height * 2, width * 2]);

        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        T val = input[b, c, h, w];
                        output[b, c, h * 2, w * 2] = val;
                        output[b, c, h * 2, w * 2 + 1] = val;
                        output[b, c, h * 2 + 1, w * 2] = val;
                        output[b, c, h * 2 + 1, w * 2 + 1] = val;
                    }

        return output;
    }

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        var result = new Tensor<T>([1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]);
        Array.Copy(tensor.Data, result.Data, tensor.Data.Length);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        var result = new Tensor<T>([tensor.Shape[1], tensor.Shape[2], tensor.Shape[3]]);
        Array.Copy(tensor.Data, result.Data, result.Data.Length);
        return result;
    }

    #endregion

    #region Layer Initialization

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
            Layers.AddRange(LayerHelper<T>.CreateDefaultCutieLayers(
                inputChannels: _inputChannels,
                inputHeight: _inputHeight,
                inputWidth: _inputWidth,
                numFeatures: _numFeatures));
        }
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");

        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            var layerParameters = parameters.Slice(index, layerParameterCount);
            layer.UpdateParameters(layerParameters);
            index += layerParameterCount;
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "Cutie" },
            { "Description", "Cutting-edge Video Instance Segmentation" },
            { "InputHeight", _inputHeight },
            { "InputWidth", _inputWidth },
            { "InputChannels", _inputChannels },
            { "NumFeatures", _numFeatures },
            { "MemorySize", _memorySize },
            { "UseNativeMode", _useNativeMode }
        };

        if (!_useNativeMode && _onnxModelPath != null)
        {
            additionalInfo["OnnxModelPath"] = _onnxModelPath;
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.VideoObjectSegmentation,
            AdditionalInfo = additionalInfo,
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        writer.Write(_inputHeight);
        writer.Write(_inputWidth);
        writer.Write(_inputChannels);
        writer.Write(_numFeatures);
        writer.Write(_memorySize);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        _ = reader.ReadInt32(); // inputHeight
        _ = reader.ReadInt32(); // inputWidth
        _ = reader.ReadInt32(); // inputChannels
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // memorySize
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Cutie<T>(
            Architecture,
            _optimizer,
            _lossFunction,
            _numFeatures,
            _memorySize);
    }

    #endregion
}
