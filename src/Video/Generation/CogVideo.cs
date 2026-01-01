using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
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
    private int _embedDim;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    private int _numLayers;

    /// <summary>
    /// Number of video frames to generate.
    /// </summary>
    private int _numFrames;

    /// <summary>
    /// Latent space height.
    /// </summary>
    private int _latentHeight;

    /// <summary>
    /// Latent space width.
    /// </summary>
    private int _latentWidth;

    /// <summary>
    /// Number of latent channels.
    /// </summary>
    private int _latentChannels;

    /// <summary>
    /// Number of diffusion timesteps.
    /// </summary>
    private int _numTimesteps;

    /// <summary>
    /// Beta schedule (variance at each timestep).
    /// </summary>
    private double[]? _betas;

    /// <summary>
    /// Alpha values (1 - beta).
    /// </summary>
    private double[]? _alphas;

    /// <summary>
    /// Cumulative product of alphas (alpha_bar).
    /// </summary>
    private double[]? _alphaBars;

    /// <summary>
    /// Cumulative product of alphas for t-1 (alpha_bar_prev).
    /// </summary>
    private double[]? _alphaBarsPrev;

    /// <summary>
    /// Square root of alpha_bar.
    /// </summary>
    private double[]? _sqrtAlphaBars;

    /// <summary>
    /// Square root of 1 - alpha_bar.
    /// </summary>
    private double[]? _sqrtOneMinusAlphaBars;

    /// <summary>
    /// Posterior variance for DDPM sampling.
    /// </summary>
    private double[]? _posteriorVariances;

    /// <summary>
    /// Posterior mean coefficient 1: sqrt(alpha_bar_prev) * beta / (1 - alpha_bar).
    /// </summary>
    private double[]? _posteriorMeanCoef1;

    /// <summary>
    /// Posterior mean coefficient 2: sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar).
    /// </summary>
    private double[]? _posteriorMeanCoef2;

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
        int numFrames = 16,
        int numTimesteps = 1000)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        if (embedDim < 1)
            throw new ArgumentOutOfRangeException(nameof(embedDim), embedDim, "Embedding dimension must be at least 1.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "Number of layers must be at least 1.");
        if (numFrames < 1)
            throw new ArgumentOutOfRangeException(nameof(numFrames), numFrames, "Number of frames must be at least 1.");
        if (numTimesteps < 1)
            throw new ArgumentOutOfRangeException(nameof(numTimesteps), numTimesteps, "Number of timesteps must be at least 1.");

        _useNativeMode = true;
        _embedDim = embedDim;
        _numLayers = numLayers;
        _numFrames = numFrames;
        _numTimesteps = numTimesteps;
        _latentHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 32;
        _latentWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 32;
        _latentChannels = architecture.InputDepth > 0 ? architecture.InputDepth : 4;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeNoiseSchedule();
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
        int numFrames = 16,
        int numTimesteps = 1000)
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
        _numTimesteps = numTimesteps;
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

        InitializeNoiseSchedule();
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
        var random = RandomHelper.CreateSecureRandom();
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
        // Create timestep embedding
        var timestepEmbed = CreateTimestepEmbedding(timestep);

        // Combine input with text embedding and timestep embedding
        var conditioned = CombineWithCondition(input, textEmbedding, timestepEmbed);

        if (_useNativeMode)
        {
            return Forward(conditioned);
        }
        else
        {
            return PredictOnnx(conditioned);
        }
    }

    /// <summary>
    /// Creates a timestep embedding using sinusoidal encoding.
    /// </summary>
    /// <remarks>
    /// Handles both even and odd embedding dimensions properly:
    /// - For even dimensions: sin fills first half, cos fills second half
    /// - For odd dimensions: sin/cos pairs interleaved, last element uses cos at highest frequency
    /// </remarks>
    private Tensor<T> CreateTimestepEmbedding(double timestep)
    {
        // Use sinusoidal position encoding for timestep
        int embedDim = _latentChannels;
        var embedding = new T[embedDim];

        // Handle both even and odd dimensions properly
        int halfDim = embedDim / 2;
        bool isOdd = embedDim % 2 == 1;

        for (int i = 0; i < halfDim; i++)
        {
            // Frequency for this position (exponential decay from 1 to 1/10000)
            double freq = Math.Exp(-Math.Log(10000.0) * i / Math.Max(1, halfDim - 1));
            embedding[i] = NumOps.FromDouble(Math.Sin(timestep * freq));
            embedding[i + halfDim] = NumOps.FromDouble(Math.Cos(timestep * freq));
        }

        // For odd dimensions, fill the last element with highest frequency cosine
        if (isOdd)
        {
            double highestFreq = Math.Exp(-Math.Log(10000.0) * halfDim / Math.Max(1, halfDim));
            embedding[embedDim - 1] = NumOps.FromDouble(Math.Cos(timestep * highestFreq));
        }

        return new Tensor<T>([1, embedDim], new Vector<T>(embedding));
    }

    /// <summary>
    /// Combines input with text and timestep conditioning.
    /// </summary>
    private Tensor<T> CombineWithCondition(Tensor<T> input, Tensor<T> textEmbedding, Tensor<T> timestepEmbed)
    {
        // Scale the input by the timestep embedding and modulate by text
        var result = new Tensor<T>(input.Shape);

        // Get timestep scale factor (average of timestep embedding)
        double timestepScale = 0.0;
        for (int i = 0; i < timestepEmbed.Length; i++)
        {
            timestepScale += NumOps.ToDouble(timestepEmbed.Data[i]);
        }
        timestepScale = 1.0 + timestepScale / timestepEmbed.Length;

        // Apply text conditioning modulation
        int textLen = textEmbedding.Length;
        for (int i = 0; i < input.Length; i++)
        {
            double val = NumOps.ToDouble(input.Data[i]);
            double textMod = textLen > 0 ? NumOps.ToDouble(textEmbedding.Data[i % textLen]) : 0.0;
            result.Data[i] = NumOps.FromDouble(val * timestepScale + textMod * 0.1);
        }

        return result;
    }

    /// <summary>
    /// Initializes the DDPM noise schedule using a linear beta schedule.
    /// </summary>
    /// <remarks>
    /// The noise schedule follows the DDPM paper (Ho et al., 2020):
    /// - Linear beta schedule from beta_start to beta_end
    /// - Precomputed alpha, alpha_bar, and posterior coefficients
    /// </remarks>
    private void InitializeNoiseSchedule()
    {
        const double betaStart = 0.0001;
        const double betaEnd = 0.02;

        _betas = new double[_numTimesteps];
        _alphas = new double[_numTimesteps];
        _alphaBars = new double[_numTimesteps];
        _alphaBarsPrev = new double[_numTimesteps];
        _sqrtAlphaBars = new double[_numTimesteps];
        _sqrtOneMinusAlphaBars = new double[_numTimesteps];
        _posteriorVariances = new double[_numTimesteps];
        _posteriorMeanCoef1 = new double[_numTimesteps];
        _posteriorMeanCoef2 = new double[_numTimesteps];

        // Linear beta schedule
        for (int t = 0; t < _numTimesteps; t++)
        {
            _betas[t] = betaStart + (betaEnd - betaStart) * t / (_numTimesteps - 1);
            _alphas[t] = 1.0 - _betas[t];
        }

        // Compute cumulative products of alphas (alpha_bar)
        double cumulativeProduct = 1.0;
        for (int t = 0; t < _numTimesteps; t++)
        {
            cumulativeProduct *= _alphas[t];
            _alphaBars[t] = cumulativeProduct;
            _alphaBarsPrev[t] = t > 0 ? _alphaBars[t - 1] : 1.0;
            _sqrtAlphaBars[t] = Math.Sqrt(_alphaBars[t]);
            _sqrtOneMinusAlphaBars[t] = Math.Sqrt(1.0 - _alphaBars[t]);
        }

        // Compute posterior coefficients for DDPM sampling
        // β̃_t = β_t * (1 - α̅_{t-1}) / (1 - α̅_t)
        // μ̃_t = √(α̅_{t-1}) * β_t / (1 - α̅_t) * x_0 + √(α_t) * (1 - α̅_{t-1}) / (1 - α̅_t) * x_t
        for (int t = 0; t < _numTimesteps; t++)
        {
            double oneMinusAlphaBar = 1.0 - _alphaBars[t];
            double oneMinusAlphaBarPrev = 1.0 - _alphaBarsPrev[t];

            // Posterior variance: β̃_t = β_t * (1 - α̅_{t-1}) / (1 - α̅_t)
            // For t=0, use a small value to avoid division issues
            if (t == 0)
            {
                _posteriorVariances[t] = _betas[t];
            }
            else
            {
                _posteriorVariances[t] = _betas[t] * oneMinusAlphaBarPrev / Math.Max(oneMinusAlphaBar, 1e-20);
            }

            // Posterior mean coefficient 1: sqrt(alpha_bar_prev) * beta / (1 - alpha_bar)
            _posteriorMeanCoef1[t] = Math.Sqrt(_alphaBarsPrev[t]) * _betas[t] / Math.Max(oneMinusAlphaBar, 1e-20);

            // Posterior mean coefficient 2: sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
            _posteriorMeanCoef2[t] = Math.Sqrt(_alphas[t]) * oneMinusAlphaBarPrev / Math.Max(oneMinusAlphaBar, 1e-20);
        }
    }

    /// <summary>
    /// Performs a single DDPM denoising step following Ho et al. (2020).
    /// </summary>
    /// <param name="noisyLatent">The noisy latent at timestep t.</param>
    /// <param name="noisePrediction">The predicted noise ε_θ(x_t, t).</param>
    /// <param name="timestep">The normalized timestep (0 to 1).</param>
    /// <returns>The denoised latent at timestep t-1.</returns>
    /// <remarks>
    /// DDPM denoising follows:
    /// 1. Predict x_0 from x_t: x_0 = (x_t - √(1-α̅_t) * ε_θ) / √(α̅_t)
    /// 2. Compute posterior mean: μ̃_t = coef1 * x_0 + coef2 * x_t
    /// 3. Sample: x_{t-1} = μ̃_t + σ_t * z (z ~ N(0,I), except at t=0)
    /// </remarks>
    private Tensor<T> DenoisingStep(Tensor<T> noisyLatent, Tensor<T> noisePrediction, double timestep)
    {
        if (_betas == null || _alphaBars == null || _sqrtAlphaBars == null ||
            _sqrtOneMinusAlphaBars == null || _posteriorVariances == null ||
            _posteriorMeanCoef1 == null || _posteriorMeanCoef2 == null)
        {
            throw new InvalidOperationException("Noise schedule not initialized. Call InitializeNoiseSchedule first.");
        }

        // Convert normalized timestep to discrete index
        int t = (int)Math.Round(timestep * (_numTimesteps - 1));
        t = Math.Max(0, Math.Min(t, _numTimesteps - 1));

        double sqrtAlphaBar = _sqrtAlphaBars[t];
        double sqrtOneMinusAlphaBar = _sqrtOneMinusAlphaBars[t];
        double posteriorMeanCoef1 = _posteriorMeanCoef1[t];
        double posteriorMeanCoef2 = _posteriorMeanCoef2[t];
        double posteriorVariance = _posteriorVariances[t];

        var resultData = new T[noisyLatent.Length];

        // Step 1: Predict x_0 from x_t and ε_θ
        // x_0 = (x_t - √(1-α̅_t) * ε_θ) / √(α̅_t)
        // Step 2: Compute posterior mean
        // μ̃_t = posteriorMeanCoef1 * x_0 + posteriorMeanCoef2 * x_t
        for (int i = 0; i < noisyLatent.Length; i++)
        {
            double xt = NumOps.ToDouble(noisyLatent.Data[i]);
            double eps = NumOps.ToDouble(noisePrediction.Data[i]);

            // Predict x_0
            double x0 = (xt - sqrtOneMinusAlphaBar * eps) / Math.Max(sqrtAlphaBar, 1e-20);

            // Clip x0 to reasonable range for stability
            x0 = MathHelper.Clamp(x0, -10.0, 10.0);

            // Compute posterior mean
            double mean = posteriorMeanCoef1 * x0 + posteriorMeanCoef2 * xt;

            resultData[i] = NumOps.FromDouble(mean);
        }

        // Step 3: Add noise (stochastic sampling) except at t=0
        if (t > 0 && posteriorVariance > 1e-20)
        {
            double stdDev = Math.Sqrt(posteriorVariance);
            var random = new Random();

            for (int i = 0; i < resultData.Length; i++)
            {
                // Sample from standard normal using Box-Muller transform
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();
                double z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-20))) * Math.Cos(2.0 * Math.PI * u2);

                double mean = NumOps.ToDouble(resultData[i]);
                resultData[i] = NumOps.FromDouble(mean + stdDev * z);
            }
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
    /// <remarks>
    /// Restores all model configuration fields and reinitializes layers to match
    /// the deserialized state. This ensures the model structure is properly
    /// reconstructed after loading from a serialized format.
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        // Read serialized configuration values
        _embedDim = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numFrames = reader.ReadInt32();
        _latentHeight = reader.ReadInt32();
        _latentWidth = reader.ReadInt32();
        _latentChannels = reader.ReadInt32();

        // Reinitialize layers with the restored configuration
        // This is necessary because the layer structure depends on these parameters
        ClearLayers();
        InitializeLayers();
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
