using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Video.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.ActionRecognition;

/// <summary>
/// SlowFast Networks for Video Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SlowFast is a two-pathway network that processes video at two
/// different frame rates simultaneously:
/// - Slow pathway: Processes fewer frames (e.g., 4 fps) but with more channels to capture spatial details
/// - Fast pathway: Processes more frames (e.g., 32 fps) but with fewer channels to capture motion
///
/// This design is inspired by how human vision has:
/// - Parvo cells: Slow but detailed spatial processing
/// - Magno cells: Fast but coarse motion processing
///
/// Example usage:
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3);
/// var model = new SlowFast&lt;double&gt;(arch, numClasses: 400);
/// var predictions = model.Classify(videoFrames);
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Two-pathway design with lateral connections
/// - Slow pathway: T frames, C channels
/// - Fast pathway: αT frames, βC channels (α=8, β=1/8 typically)
/// - Lateral connections fuse information between pathways
/// </para>
/// <para>
/// <b>Reference:</b> "SlowFast Networks for Video Recognition" ICCV 2019
/// https://arxiv.org/abs/1812.03982
/// </para>
/// </remarks>
public class SlowFast<T> : NeuralNetworkBase<T>
{
    private readonly SlowFastOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Execution Mode

    private readonly bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region ONNX Mode Fields

    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    // Training components - private set to allow deserialization restoration
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private ILossFunction<T> _lossFunction;
    private IActivationFunction<T> _probabilityActivation;

    // Configuration fields - mutable to support deserialization
    private int _numClasses;
    private int _slowFrames;
    private int _fastFrames;
    private int _slowChannels;
    private int _fastChannels;
    private int _alpha;
    private int _imageSize;

    /// <summary>
    /// Fast pathway layers (high temporal resolution, low channel capacity).
    /// </summary>
    private readonly List<ILayer<T>> _fastLayers = [];

    /// <summary>
    /// Fusion layers that combine slow and fast pathway outputs for classification.
    /// </summary>
    private readonly List<ILayer<T>> _fusionLayers = [];

    /// <summary>
    /// Custom fast pathway layers provided by user (null = use default).
    /// </summary>
    private IReadOnlyList<ILayer<T>>? _customFastLayers;

    /// <summary>
    /// Custom fusion layers provided by user (null = use default).
    /// </summary>
    private IReadOnlyList<ILayer<T>>? _customFusionLayers;

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    public override bool SupportsTraining => _useNativeMode;
    internal int NumClasses => _numClasses;
    internal int SlowFrames => _slowFrames;
    internal int FastFrames => _fastFrames;
    internal int Alpha => _alpha;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SlowFast model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The network architecture configuration. If Architecture.Layers is provided,
    /// it will be used as the slow pathway and customFastLayers/customFusionLayers must also be provided.</param>
    /// <param name="numClasses">Number of action classes (default: 400 for Kinetics-400).</param>
    /// <param name="optimizer">Optimizer for training (default: Adam).</param>
    /// <param name="lossFunction">Loss function for training (default: CrossEntropy).</param>
    /// <param name="probabilityActivation">Activation for converting logits to probabilities (default: Softmax).</param>
    /// <param name="customFastLayers">Custom fast pathway layers (required if Architecture.Layers is provided).</param>
    /// <param name="customFusionLayers">Custom fusion layers (required if Architecture.Layers is provided).</param>
    /// <param name="slowFrames">Number of frames for slow pathway (default: 4).</param>
    /// <param name="slowChannels">Base channels for slow pathway (default: 64).</param>
    /// <param name="fastChannels">Base channels for fast pathway (default: 8).</param>
    /// <param name="alpha">Frame rate ratio between fast and slow pathways (default: 8).</param>
    public SlowFast(
        NeuralNetworkArchitecture<T> architecture,
        int numClasses = 400,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        IActivationFunction<T>? probabilityActivation = null,
        IReadOnlyList<ILayer<T>>? customFastLayers = null,
        IReadOnlyList<ILayer<T>>? customFusionLayers = null,
        int slowFrames = 4,
        int slowChannels = 64,
        int fastChannels = 8,
        int alpha = 8,
        SlowFastOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new SlowFastOptions();
        Options = _options;

        if (numClasses < 1)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be at least 1.");
        if (slowFrames < 1)
            throw new ArgumentOutOfRangeException(nameof(slowFrames), "Slow frames must be at least 1.");
        if (slowChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(slowChannels), "Slow channels must be at least 1.");
        if (fastChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(fastChannels), "Fast channels must be at least 1.");
        if (alpha < 1)
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be at least 1.");

        // Validate custom layer consistency - if any custom layers provided, all three pathways must be specified
        bool hasCustomSlowLayers = architecture.Layers != null && architecture.Layers.Count > 0;
        bool hasCustomFastLayers = customFastLayers != null && customFastLayers.Count > 0;
        bool hasCustomFusionLayers = customFusionLayers != null && customFusionLayers.Count > 0;

        if (hasCustomSlowLayers || hasCustomFastLayers || hasCustomFusionLayers)
        {
            if (!hasCustomSlowLayers || !hasCustomFastLayers || !hasCustomFusionLayers)
            {
                throw new ArgumentException(
                    "SlowFast requires all three pathway layer sets when customizing. " +
                    "Provide Architecture.Layers (slow pathway), customFastLayers, and customFusionLayers together, " +
                    "or leave all null to use default initialization.");
            }
        }

        _useNativeMode = true;
        _numClasses = numClasses;
        _slowFrames = slowFrames;
        _fastFrames = slowFrames * alpha;
        _slowChannels = slowChannels;
        _fastChannels = fastChannels;
        _alpha = alpha;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 224;

        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _probabilityActivation = probabilityActivation ?? new SoftmaxActivation<T>();
        _customFastLayers = customFastLayers;
        _customFusionLayers = customFusionLayers;

        InitializeLayers();
    }

    /// <summary>
    /// Creates a SlowFast model using a pretrained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model file.</param>
    /// <param name="numClasses">Number of action classes (default: 400 for Kinetics-400).</param>
    /// <param name="probabilityActivation">Activation for converting logits to probabilities (default: Softmax).</param>
    /// <param name="slowFrames">Number of frames for slow pathway (default: 4).</param>
    /// <param name="slowChannels">Base channels for slow pathway (default: 64).</param>
    /// <param name="fastChannels">Base channels for fast pathway (default: 8).</param>
    /// <param name="alpha">Frame rate ratio between fast and slow pathways (default: 8).</param>
    public SlowFast(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 400,
        IActivationFunction<T>? probabilityActivation = null,
        int slowFrames = 4,
        int slowChannels = 64,
        int fastChannels = 8,
        int alpha = 8,
        SlowFastOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        _options = options ?? new SlowFastOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"SlowFast ONNX model not found: {onnxModelPath}");
        if (numClasses < 1)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be at least 1.");
        if (slowFrames < 1)
            throw new ArgumentOutOfRangeException(nameof(slowFrames), "Slow frames must be at least 1.");
        if (slowChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(slowChannels), "Slow channels must be at least 1.");
        if (fastChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(fastChannels), "Fast channels must be at least 1.");
        if (alpha < 1)
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be at least 1.");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _numClasses = numClasses;
        _slowFrames = slowFrames;
        _fastFrames = slowFrames * alpha;
        _slowChannels = slowChannels;
        _fastChannels = fastChannels;
        _alpha = alpha;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _lossFunction = new CrossEntropyLoss<T>();
        _probabilityActivation = probabilityActivation ?? new SoftmaxActivation<T>();

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
    /// Classifies video frames into action categories.
    /// </summary>
    public Tensor<T> Classify(Tensor<T> videoFrames)
    {
        if (videoFrames is null)
            throw new ArgumentNullException(nameof(videoFrames));

        return _useNativeMode ? Forward(videoFrames) : PredictOnnx(videoFrames);
    }

    /// <summary>
    /// Gets top-K predictions with probabilities.
    /// </summary>
    /// <param name="videoFrames">Input video frames tensor.</param>
    /// <param name="topK">Number of top predictions to return (default: 5).</param>
    /// <returns>List of (ClassIndex, Probability) tuples sorted by probability descending.</returns>
    public List<(int ClassIndex, double Probability)> GetTopKPredictions(Tensor<T> videoFrames, int topK = 5)
    {
        var logits = Classify(videoFrames);
        var probabilities = _probabilityActivation.Activate(logits);

        var results = new List<(int, double)>();
        for (int i = 0; i < probabilities.Length; i++)
        {
            results.Add((i, Convert.ToDouble(probabilities.Data.Span[i])));
        }

        return results.OrderByDescending(x => x.Item2).Take(topK).ToList();
    }

    #endregion

    #region Inference

    private Tensor<T> Forward(Tensor<T> input)
    {
        // Validate dual pathways are initialized
        if (Layers.Count == 0)
            throw new InvalidOperationException("Slow pathway not initialized. Layers collection is empty.");
        if (_fastLayers.Count == 0)
            throw new InvalidOperationException("Fast pathway not initialized. Fast layers collection is empty.");
        if (_fusionLayers.Count == 0)
            throw new InvalidOperationException("Fusion layers not initialized. Fusion layers collection is empty.");

        // SlowFast dual-pathway architecture:
        // Input is expected as [batch, channels, frames, height, width] or [batch, channels * frames, height, width]
        // Slow pathway: subsampled frames (every alpha-th frame)
        // Fast pathway: all frames

        // Run slow pathway (Layers contains slow pathway layers)
        var slowInput = SubsampleFrames(input, _alpha);
        var slowResult = slowInput;
        foreach (var layer in Layers)
        {
            slowResult = layer.Forward(slowResult);
        }

        // Run fast pathway
        var fastResult = input;
        foreach (var layer in _fastLayers)
        {
            fastResult = layer.Forward(fastResult);
        }

        // Concatenate slow and fast pathway outputs along channel dimension
        var fused = ConcatenateTensors(slowResult, fastResult);

        // Apply fusion layers for classification
        foreach (var layer in _fusionLayers)
        {
            fused = layer.Forward(fused);
        }

        return fused;
    }

    /// <summary>
    /// Subsamples frames by taking every n-th frame.
    /// </summary>
    private Tensor<T> SubsampleFrames(Tensor<T> input, int subsampleRate)
    {
        if (input.Rank < 4)
            return input;

        // Assume input is [batch, channels * frames, height, width] or [batch, channels, frames, height, width]
        // For simplicity, if channels are flattened with frames, subsample along channel dimension
        int batch = input.Shape[0];
        int totalChannels = input.Shape[1];
        int h = input.Shape[2];
        int w = input.Shape[3];

        // Calculate subsampled channels using actual input depth from architecture (supports RGB, grayscale, RGBA, etc.)
        int channelsPerFrame = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
        int framesInInput = totalChannels / channelsPerFrame;
        int subsampledFrames = (framesInInput + subsampleRate - 1) / subsampleRate;
        int subsampledChannels = subsampledFrames * channelsPerFrame;

        if (subsampledFrames == framesInInput || subsampleRate == 1)
            return input;

        var result = new Tensor<T>([batch, subsampledChannels, h, w]);
        int srcFrameSize = channelsPerFrame * h * w;
        int dstFrameSize = channelsPerFrame * h * w;

        for (int b = 0; b < batch; b++)
        {
            int dstFrame = 0;
            for (int srcFrame = 0; srcFrame < framesInInput && dstFrame < subsampledFrames; srcFrame += subsampleRate)
            {
                int srcOffset = b * totalChannels * h * w + srcFrame * channelsPerFrame * h * w;
                int dstOffset = b * subsampledChannels * h * w + dstFrame * channelsPerFrame * h * w;
                int copyLen = Math.Min(srcFrameSize, dstFrameSize);
                input.Data.Span.Slice(srcOffset, copyLen).CopyTo(result.Data.Span.Slice(dstOffset, copyLen));
                dstFrame++;
            }
        }

        return result;
    }

    /// <summary>
    /// Concatenates two tensors along the channel dimension.
    /// </summary>
    private Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
    {
        if (a.Rank != 4 || b.Rank != 4)
            throw new ArgumentException("Both tensors must be 4D [batch, channels, height, width].");

        int batch = a.Shape[0];
        int aChannels = a.Shape[1];
        int bChannels = b.Shape[1];
        int h = a.Shape[2];
        int w = a.Shape[3];

        if (batch != b.Shape[0] || h != b.Shape[2] || w != b.Shape[3])
            throw new ArgumentException(
                $"Tensor dimensions must match. Expected batch={batch}, height={h}, width={w}, " +
                $"but got batch={b.Shape[0]}, height={b.Shape[2]}, width={b.Shape[3]}.");

        var result = new Tensor<T>([batch, aChannels + bChannels, h, w]);
        int aSliceSize = aChannels * h * w;
        int bSliceSize = bChannels * h * w;
        int resultSliceSize = (aChannels + bChannels) * h * w;

        for (int bi = 0; bi < batch; bi++)
        {
            a.Data.Span.Slice(bi * aSliceSize, aSliceSize).CopyTo(result.Data.Span.Slice(bi * resultSliceSize, aSliceSize));
            b.Data.Span.Slice(bi * bSliceSize, bSliceSize).CopyTo(result.Data.Span.Slice(bi * resultSliceSize + aSliceSize, bSliceSize));
        }

        return result;
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(SlowFast<T>));
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        // Validate input shape matches ONNX model expectations
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();
        var expectedShape = inputMeta[inputName].Dimensions;

        if (expectedShape.Length != input.Rank)
            throw new ArgumentException(
                $"Input rank mismatch: ONNX model expects {expectedShape.Length}D tensor, got {input.Rank}D.",
                nameof(input));

        // Validate dimensions (skip dynamic dimensions marked as -1)
        for (int i = 0; i < expectedShape.Length; i++)
        {
            if (expectedShape[i] > 0 && expectedShape[i] != input.Shape[i])
                throw new ArgumentException(
                    $"Input shape mismatch at dimension {i}: ONNX model expects {expectedShape[i]}, got {input.Shape[i]}.",
                    nameof(input));
        }

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };

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

    public override Tensor<T> Predict(Tensor<T> input) => Classify(input);

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        var prediction = Predict(input);
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

        // Backward through fusion layers
        var fusionGradient = outputGradientTensor;
        for (int i = _fusionLayers.Count - 1; i >= 0; i--)
        {
            fusionGradient = _fusionLayers[i].Backward(fusionGradient);
        }

        // Split gradient for slow and fast pathways
        var (slowGradient, fastGradient) = SplitGradient(fusionGradient);

        // Backward through slow pathway
        var slowCurrentGradient = slowGradient;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            slowCurrentGradient = Layers[i].Backward(slowCurrentGradient);
        }

        // Backward through fast pathway
        var fastCurrentGradient = fastGradient;
        for (int i = _fastLayers.Count - 1; i >= 0; i--)
        {
            fastCurrentGradient = _fastLayers[i].Backward(fastCurrentGradient);
        }

        // Update all parameters
        var allLayers = new List<ILayer<T>>();
        allLayers.AddRange(Layers);
        allLayers.AddRange(_fastLayers);
        allLayers.AddRange(_fusionLayers);
        _optimizer?.UpdateParameters(allLayers);
    }

    /// <summary>
    /// Splits the gradient tensor for slow and fast pathways.
    /// </summary>
    private (Tensor<T> slowGradient, Tensor<T> fastGradient) SplitGradient(Tensor<T> fusedGradient)
    {
        if (fusedGradient.Rank != 4)
            throw new ArgumentException("Gradient must be 4D [batch, channels, height, width].");

        int batch = fusedGradient.Shape[0];
        int totalChannels = fusedGradient.Shape[1];
        int h = fusedGradient.Shape[2];
        int w = fusedGradient.Shape[3];

        // Slow pathway outputs _slowChannels * 8 (after 3 downsampling stages in ResNet-like arch)
        // Fast pathway outputs _fastChannels * 8
        int slowOutputChannels = _slowChannels * 8;
        int fastOutputChannels = _fastChannels * 8;

        // Validate channel count
        if (totalChannels != slowOutputChannels + fastOutputChannels)
        {
            // Fall back to proportional split based on slow/fast channel ratio
            slowOutputChannels = totalChannels * _slowChannels / (_slowChannels + _fastChannels);
            fastOutputChannels = totalChannels - slowOutputChannels;
        }

        var slowGrad = new Tensor<T>([batch, slowOutputChannels, h, w]);
        var fastGrad = new Tensor<T>([batch, fastOutputChannels, h, w]);

        int slowSliceSize = slowOutputChannels * h * w;
        int fastSliceSize = fastOutputChannels * h * w;
        int totalSliceSize = totalChannels * h * w;

        for (int bi = 0; bi < batch; bi++)
        {
            fusedGradient.Data.Span.Slice(bi * totalSliceSize, slowSliceSize).CopyTo(slowGrad.Data.Span.Slice(bi * slowSliceSize, slowSliceSize));
            fusedGradient.Data.Span.Slice(bi * totalSliceSize + slowSliceSize, fastSliceSize).CopyTo(fastGrad.Data.Span.Slice(bi * fastSliceSize, fastSliceSize));
        }

        return (slowGrad, fastGrad);
    }

    #endregion

    #region Layer Initialization

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            ClearLayers();
            _fastLayers.Clear();
            _fusionLayers.Clear();
            return;
        }

        // Check if custom layers are provided (validation already done in constructor)
        bool hasCustomLayers = Architecture.Layers != null && Architecture.Layers.Count > 0;

        if (hasCustomLayers && Architecture.Layers != null && _customFastLayers != null && _customFusionLayers != null)
        {
            // Use custom layers for all three pathways
            Layers.AddRange(Architecture.Layers);
            _fastLayers.AddRange(_customFastLayers);
            _fusionLayers.AddRange(_customFusionLayers);
        }
        else
        {
            // Use default LayerHelper initialization
            int inputChannels = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            int inputHeight = Architecture.InputHeight > 0 ? Architecture.InputHeight : 224;
            int inputWidth = Architecture.InputWidth > 0 ? Architecture.InputWidth : 224;

            // SlowFast dual-pathway architecture:
            // 1. Slow pathway: low temporal resolution (subsampled frames), high channel capacity
            Layers.AddRange(LayerHelper<T>.CreateSlowFastSlowPathwayLayers(
                inputChannels, inputHeight, inputWidth, _slowChannels));

            // 2. Fast pathway: high temporal resolution (all frames), low channel capacity
            _fastLayers.AddRange(LayerHelper<T>.CreateSlowFastFastPathwayLayers(
                inputChannels, inputHeight, inputWidth, _fastChannels));

            // 3. Fusion layers: combine slow and fast pathway outputs for classification
            // Feature map size after pathways (typically 14x14 for 224x224 input after 4 downsampling stages)
            int featureHeight = inputHeight / 16;
            int featureWidth = inputWidth / 16;
            _fusionLayers.AddRange(LayerHelper<T>.CreateSlowFastFusionLayers(
                _slowChannels, _fastChannels, featureHeight, featureWidth, _numClasses));
        }
    }

    #endregion

    #region Serialization

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");

        int offset = 0;

        // Update slow pathway layers
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int paramCount = layerParams.Length;
            if (paramCount > 0 && offset + paramCount <= parameters.Length)
            {
                var slice = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++) slice[i] = parameters[offset + i];
                layer.SetParameters(slice);
                offset += paramCount;
            }
        }

        // Update fast pathway layers
        foreach (var layer in _fastLayers)
        {
            var layerParams = layer.GetParameters();
            int paramCount = layerParams.Length;
            if (paramCount > 0 && offset + paramCount <= parameters.Length)
            {
                var slice = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++) slice[i] = parameters[offset + i];
                layer.SetParameters(slice);
                offset += paramCount;
            }
        }

        // Update fusion layers
        foreach (var layer in _fusionLayers)
        {
            var layerParams = layer.GetParameters();
            int paramCount = layerParams.Length;
            if (paramCount > 0 && offset + paramCount <= parameters.Length)
            {
                var slice = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++) slice[i] = parameters[offset + i];
                layer.SetParameters(slice);
                offset += paramCount;
            }
        }
    }

    /// <summary>
    /// Gets metadata about this model for serialization.
    /// </summary>
    /// <remarks>
    /// Serializes model architecture configuration, layer weights, and training component types
    /// (optimizer, loss function, probability activation). After deserialization, training components
    /// are recreated from their type names using reflection. Custom layer definitions are NOT preserved -
    /// default LayerHelper layers are used unless custom layers are re-provided after deserialization.
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        ModelType = ModelType.VideoActionRecognition,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "SlowFast" },
            { "NumClasses", _numClasses },
            { "SlowFrames", _slowFrames },
            { "FastFrames", _fastFrames },
            { "Alpha", _alpha },
            { "UseNativeMode", _useNativeMode }
        },
        ModelData = _useNativeMode ? this.Serialize() : []
    };

    /// <summary>
    /// Serializes SlowFast-specific configuration data including training component types.
    /// </summary>
    /// <remarks>
    /// Serializes configuration parameters and type names for training components
    /// (optimizer, loss function, probability activation). Custom layer definitions
    /// are NOT serialized - after deserialization, default LayerHelper layers are used
    /// unless custom layers are re-provided.
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        // Configuration parameters
        writer.Write(_numClasses);
        writer.Write(_slowFrames);
        writer.Write(_fastFrames);
        writer.Write(_slowChannels);
        writer.Write(_fastChannels);
        writer.Write(_alpha);
        writer.Write(_imageSize);

        // Training component type names for restoration
        writer.Write(_lossFunction.GetType().AssemblyQualifiedName ?? typeof(CrossEntropyLoss<T>).AssemblyQualifiedName!);
        writer.Write(_probabilityActivation.GetType().AssemblyQualifiedName ?? typeof(SoftmaxActivation<T>).AssemblyQualifiedName!);

        // Optimizer type (can be null for ONNX mode or after certain operations)
        bool hasOptimizer = _optimizer != null;
        writer.Write(hasOptimizer);
        if (hasOptimizer)
        {
            writer.Write(_optimizer!.GetType().AssemblyQualifiedName ?? typeof(AdamOptimizer<T, Tensor<T>, Tensor<T>>).AssemblyQualifiedName!);
        }
    }

    /// <summary>
    /// Deserializes SlowFast-specific configuration data and reinitializes layers.
    /// </summary>
    /// <remarks>
    /// Restores configuration parameters and recreates training components from serialized type names.
    /// Custom layer definitions are NOT restored - default LayerHelper layers are used after deserialization.
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        // Restore configuration values
        _numClasses = reader.ReadInt32();
        _slowFrames = reader.ReadInt32();
        _fastFrames = reader.ReadInt32();
        _slowChannels = reader.ReadInt32();
        _fastChannels = reader.ReadInt32();
        _alpha = reader.ReadInt32();
        _imageSize = reader.ReadInt32();

        // Restore training component types
        string lossFunctionTypeName = reader.ReadString();
        string probabilityActivationTypeName = reader.ReadString();

        // Recreate loss function from type name
        var lossFunctionType = Type.GetType(lossFunctionTypeName);
        if (lossFunctionType != null)
        {
            _lossFunction = (ILossFunction<T>?)Activator.CreateInstance(lossFunctionType) ?? new CrossEntropyLoss<T>();
        }
        else
        {
            _lossFunction = new CrossEntropyLoss<T>();
        }

        // Recreate probability activation from type name
        var activationType = Type.GetType(probabilityActivationTypeName);
        if (activationType != null)
        {
            _probabilityActivation = (IActivationFunction<T>?)Activator.CreateInstance(activationType) ?? new SoftmaxActivation<T>();
        }
        else
        {
            _probabilityActivation = new SoftmaxActivation<T>();
        }

        // Restore optimizer if it was serialized
        bool hasOptimizer = reader.ReadBoolean();
        if (hasOptimizer)
        {
            string optimizerTypeName = reader.ReadString();
            var optimizerType = Type.GetType(optimizerTypeName);

            // Optimizer requires 'this' network instance, so we need to handle construction specially
            if (optimizerType != null)
            {
                // Try to find constructor that takes IFullModel parameter (used by optimizers)
                var constructor = optimizerType.GetConstructor([typeof(IFullModel<T, Tensor<T>, Tensor<T>>)]);
                if (constructor != null)
                {
                    _optimizer = (IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>?)constructor.Invoke([this]);
                }
                else
                {
                    // Fall back to default Adam optimizer
                    _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
                }
            }
            else
            {
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            }
        }
        else
        {
            _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        }

        // Clear custom layer references (not serialized)
        _customFastLayers = null;
        _customFusionLayers = null;

        // Reinitialize layers with restored configuration
        ClearLayers();
        _fastLayers.Clear();
        _fusionLayers.Clear();
        InitializeLayers();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new SlowFast<T>(Architecture, _numClasses, _optimizer, _lossFunction, _probabilityActivation, _customFastLayers, _customFusionLayers, _slowFrames, _slowChannels, _fastChannels, _alpha);

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases the unmanaged resources and optionally releases the managed resources.
    /// </summary>
    /// <param name="disposing">True to release both managed and unmanaged resources; false to release only unmanaged.</param>
    /// <remarks>
    /// Disposes the ONNX inference session if one was created. This is important for
    /// releasing native ONNX runtime handles and memory when using pretrained models.
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _onnxSession?.Dispose();
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
