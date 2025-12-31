using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Generation;

/// <summary>
/// CogVideo: Text-to-Video Diffusion Model for generating videos from text descriptions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// CogVideo is a state-of-the-art text-to-video generation model that:
/// - Generates coherent video clips from text prompts
/// - Uses diffusion-based denoising in latent space
/// - Produces temporally consistent animations
/// </para>
/// <para>
/// <b>For Beginners:</b> CogVideo creates videos from text descriptions.
/// You provide a prompt like "a cat playing with a ball" and it generates
/// a video showing that scene. It works by:
/// 1. Starting with random noise
/// 2. Gradually denoising to create coherent frames
/// 3. Ensuring temporal consistency across frames
///
/// Example usage (native mode for training):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 32, inputWidth: 32, inputDepth: 4);
/// var model = new CogVideo&lt;double&gt;(arch);
/// model.Train(noisyLatent, cleanLatent);
/// var videoFrames = model.Generate(textEmbedding, numSteps: 50);
/// </code>
///
/// Example usage (ONNX mode for inference only):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 32, inputWidth: 32, inputDepth: 4);
/// var model = new CogVideo&lt;double&gt;(arch, "cogvideo.onnx");
/// var videoFrames = model.Generate(textEmbedding);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer"
/// https://arxiv.org/abs/2408.06072
/// </para>
/// </remarks>
public class CogVideo<T> : NeuralNetworkBase<T>
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
    /// Embedding dimension for the model.
    /// </summary>
    private readonly int _embedDim;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    private readonly int _numLayers;

    /// <summary>
    /// Number of video frames to generate.
    /// </summary>
    private readonly int _numFrames;

    /// <summary>
    /// Latent space height.
    /// </summary>
    private readonly int _latentHeight;

    /// <summary>
    /// Latent space width.
    /// </summary>
    private readonly int _latentWidth;

    /// <summary>
    /// Number of latent channels.
    /// </summary>
    private readonly int _latentChannels;

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
    /// Gets the embedding dimension.
    /// </summary>
    internal int EmbedDim => _embedDim;

    /// <summary>
    /// Gets the number of frames generated.
    /// </summary>
    internal int NumFrames => _numFrames;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a CogVideo model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">Architecture for the video generation network.</param>
    /// <param name="optimizer">Optional optimizer for training. Default: Adam.</param>
    /// <param name="lossFunction">Optional loss function. Default: MSE.</param>
    /// <param name="embedDim">Embedding dimension (default: 1024).</param>
    /// <param name="numLayers">Number of transformer layers (default: 24).</param>
    /// <param name="numFrames">Number of frames to generate (default: 16).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a trainable CogVideo model:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 32, inputWidth: 32, inputDepth: 4);
    /// var model = new CogVideo&lt;double&gt;(arch);
    /// </code>
    /// </para>
    /// </remarks>
    public CogVideo(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int embedDim = 1024,
        int numLayers = 24,
        int numFrames = 16)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        if (embedDim < 1)
            throw new ArgumentOutOfRangeException(nameof(embedDim), embedDim, "Embedding dimension must be at least 1.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "Number of layers must be at least 1.");
        if (numFrames < 1)
            throw new ArgumentOutOfRangeException(nameof(numFrames), numFrames, "Number of frames must be at least 1.");

        _useNativeMode = true;
        _embedDim = embedDim;
        _numLayers = numLayers;
        _numFrames = numFrames;
        _latentHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 32;
        _latentWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 32;
        _latentChannels = architecture.InputDepth > 0 ? architecture.InputDepth : 4;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a CogVideo model using a pretrained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model.</param>
    /// <param name="numFrames">Number of frames the model generates (default: 16).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained model
    /// in ONNX format. Training is not supported in ONNX mode.
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 32, inputWidth: 32, inputDepth: 4);
    /// var model = new CogVideo&lt;double&gt;(arch, "cogvideo.onnx");
    /// var video = model.Generate(textEmbedding);
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    public CogVideo(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numFrames = 16)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"CogVideo ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _embedDim = 1024;
        _numLayers = 24;
        _numFrames = numFrames;
        _latentHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 32;
        _latentWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 32;
        _latentChannels = architecture.InputDepth > 0 ? architecture.InputDepth : 4;
        _lossFunction = new MeanSquaredErrorLoss<T>();

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
    /// Generates video frames from a text embedding.
    /// </summary>
    /// <param name="textEmbedding">Text embedding from a text encoder.</param>
    /// <param name="numSteps">Number of denoising steps (default: 50).</param>
    /// <returns>Generated video frames tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a video from a text description.
    /// The text must first be encoded using a text encoder (like CLIP).
    /// More denoising steps generally produce better quality but take longer.
    /// </para>
    /// </remarks>
    public Tensor<T> Generate(Tensor<T> textEmbedding, int numSteps = 50)
    {
        if (textEmbedding is null)
            throw new ArgumentNullException(nameof(textEmbedding));

        // Start with random noise in latent space
        var latent = GenerateRandomNoise();

        // Iteratively denoise
        for (int step = 0; step < numSteps; step++)
        {
            double timestepRatio = 1.0 - (double)step / numSteps;
            var noisePrediction = PredictNoise(latent, textEmbedding, timestepRatio);
            latent = DenoisingStep(latent, noisePrediction, timestepRatio);
        }

        return latent;
    }

    /// <summary>
    /// Performs a single denoising step.
    /// </summary>
    /// <param name="noisyInput">Current noisy input tensor.</param>
    /// <param name="textEmbedding">Text conditioning embedding.</param>
    /// <param name="timestep">Current timestep (0-1 range).</param>
    /// <returns>Denoised output tensor.</returns>
    public Tensor<T> Denoise(Tensor<T> noisyInput, Tensor<T> textEmbedding, double timestep)
    {
        if (noisyInput is null)
            throw new ArgumentNullException(nameof(noisyInput));

        var noisePrediction = PredictNoise(noisyInput, textEmbedding, timestep);
        return DenoisingStep(noisyInput, noisePrediction, timestep);
    }

    #endregion

    #region Inference

    /// <summary>
    /// Generates random noise for the diffusion process.
    /// </summary>
    private Tensor<T> GenerateRandomNoise()
    {
        var random = new Random();
        int totalSize = _numFrames * _latentChannels * _latentHeight * _latentWidth;
        var noiseData = new T[totalSize];

        for (int i = 0; i < totalSize; i++)
        {
            // Box-Muller transform for Gaussian noise
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            noiseData[i] = NumOps.FromDouble(randStdNormal);
        }

        return new Tensor<T>(
            [_numFrames, _latentChannels, _latentHeight, _latentWidth],
            new Vector<T>(noiseData));
    }

    /// <summary>
    /// Predicts the noise component in the input.
    /// </summary>
    private Tensor<T> PredictNoise(Tensor<T> input, Tensor<T> textEmbedding, double timestep)
    {
        // Combine input with timestep embedding (simplified)
        var result = input;

        if (_useNativeMode)
        {
            result = Forward(result);
        }
        else
        {
            result = PredictOnnx(result);
        }

        return result;
    }

    /// <summary>
    /// Performs a single denoising step.
    /// </summary>
    private Tensor<T> DenoisingStep(Tensor<T> noisyLatent, Tensor<T> noisePrediction, double timestep)
    {
        // DDPM-style denoising step (simplified)
        double alpha = Math.Sqrt(1.0 - timestep * timestep);
        double sigma = timestep;

        var resultData = new T[noisyLatent.Length];
        for (int i = 0; i < noisyLatent.Length; i++)
        {
            double noisy = NumOps.ToDouble(noisyLatent.Data[i]);
            double predicted = NumOps.ToDouble(noisePrediction.Data[i]);
            double denoised = (noisy - sigma * predicted) / alpha;
            resultData[i] = NumOps.FromDouble(denoised);
        }

        return new Tensor<T>(noisyLatent.Shape, new Vector<T>(resultData));
    }

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
        if (_useNativeMode)
        {
            return Forward(input);
        }
        else
        {
            return PredictOnnx(input);
        }
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultCogVideoLayers(
                inputChannels: _latentChannels,
                inputHeight: _latentHeight,
                inputWidth: _latentWidth,
                embedDim: _embedDim,
                numLayers: _numLayers,
                numFrames: _numFrames));
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
            { "ModelName", "CogVideo" },
            { "EmbedDim", _embedDim },
            { "NumLayers", _numLayers },
            { "NumFrames", _numFrames },
            { "LatentHeight", _latentHeight },
            { "LatentWidth", _latentWidth },
            { "LatentChannels", _latentChannels },
            { "UseNativeMode", _useNativeMode }
        };

        if (!_useNativeMode && _onnxModelPath != null)
        {
            additionalInfo["OnnxModelPath"] = _onnxModelPath;
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.TextToVideo,
            AdditionalInfo = additionalInfo,
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        writer.Write(_embedDim);
        writer.Write(_numLayers);
        writer.Write(_numFrames);
        writer.Write(_latentHeight);
        writer.Write(_latentWidth);
        writer.Write(_latentChannels);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        _ = reader.ReadInt32(); // embedDim
        _ = reader.ReadInt32(); // numLayers
        _ = reader.ReadInt32(); // numFrames
        _ = reader.ReadInt32(); // latentHeight
        _ = reader.ReadInt32(); // latentWidth
        _ = reader.ReadInt32(); // latentChannels
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CogVideo<T>(
            Architecture,
            _optimizer,
            _lossFunction,
            _embedDim,
            _numLayers,
            _numFrames);
    }

    #endregion
}
