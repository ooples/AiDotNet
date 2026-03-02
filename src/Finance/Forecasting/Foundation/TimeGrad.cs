using System.IO;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// TimeGrad — Autoregressive Denoising Diffusion Model for Time Series Forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeGrad combines an autoregressive RNN with a conditional diffusion process for
/// probabilistic multi-step forecasting. It generates multiple forecast samples to
/// provide well-calibrated uncertainty estimates.
/// </para>
/// <para>
/// <b>Reference:</b> Rasul et al., "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting", ICML 2021.
/// </para>
/// </remarks>
public class TimeGrad<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _rnnEncoder;
    private readonly List<ILayer<T>> _denoisingLayers = [];
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TimeGradOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _hiddenDimension;
    private int _numRnnLayers;
    private int _numDiffusionSteps;
    private int _denoisingNetworkDim;
    private int _numSamples;
    private double _dropout;
    private double _betaStart;
    private double _betaEnd;

    // DDPM noise schedule arrays (precomputed for efficiency)
    private double[] _betas = Array.Empty<double>();
    private double[] _alphas = Array.Empty<double>();
    private double[] _alphasCumprod = Array.Empty<double>();
    private double[] _sqrtAlphasCumprod = Array.Empty<double>();
    private double[] _sqrtOneMinusAlphasCumprod = Array.Empty<double>();

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;
    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;
    /// <inheritdoc/>
    public override int NumFeatures => 1;
    /// <inheritdoc/>
    public override int PatchSize => 1;
    /// <inheritdoc/>
    public override int Stride => 1;
    /// <inheritdoc/>
    public override bool IsChannelIndependent => false;
    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;
    /// <inheritdoc/>
    public override FoundationModelSize ModelSize => FoundationModelSize.Small;
    /// <inheritdoc/>
    public override int MaxContextLength => _contextLength;
    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TimeGrad model using a pretrained ONNX model.
    /// </summary>
    public TimeGrad(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TimeGradOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TimeGradOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
    }

    /// <summary>
    /// Creates a TimeGrad model in native mode for training.
    /// </summary>
    public TimeGrad(
        NeuralNetworkArchitecture<T> architecture,
        TimeGradOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TimeGradOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
        InitializeLayers();
    }

    private void CopyOptionsToFields(TimeGradOptions<T> options)
    {
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numRnnLayers = options.NumRnnLayers;
        _numDiffusionSteps = options.NumDiffusionSteps;
        _denoisingNetworkDim = options.DenoisingNetworkDim;
        _numSamples = options.NumSamples;
        _dropout = options.DropoutRate;
        _betaStart = options.BetaStart;
        _betaEnd = options.BetaEnd;

        ComputeNoiseSchedule();
    }

    /// <summary>
    /// Precomputes the DDPM noise schedule arrays for the diffusion process.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The noise schedule controls how quickly noise is added/removed:
    /// - beta_t: noise added at step t
    /// - alpha_t: 1 - beta_t (signal retained)
    /// - alpha_bar_t: cumulative product of alphas (total signal remaining at step t)
    /// These are precomputed once and reused during every forward/sampling pass.
    /// </remarks>
    private void ComputeNoiseSchedule()
    {
        if (_numDiffusionSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(_numDiffusionSteps), "DiffusionSteps must be positive.");

        _betas = new double[_numDiffusionSteps];
        _alphas = new double[_numDiffusionSteps];
        _alphasCumprod = new double[_numDiffusionSteps];
        _sqrtAlphasCumprod = new double[_numDiffusionSteps];
        _sqrtOneMinusAlphasCumprod = new double[_numDiffusionSteps];

        // Linear beta schedule: beta_t linearly interpolates from betaStart to betaEnd
        for (int t = 0; t < _numDiffusionSteps; t++)
        {
            _betas[t] = _betaStart + (_betaEnd - _betaStart) * t / Math.Max(1, _numDiffusionSteps - 1);
            _alphas[t] = 1.0 - _betas[t];
        }

        // Cumulative product of alphas
        _alphasCumprod[0] = _alphas[0];
        for (int t = 1; t < _numDiffusionSteps; t++)
        {
            _alphasCumprod[t] = _alphasCumprod[t - 1] * _alphas[t];
        }

        // Precompute sqrt values used in sampling
        for (int t = 0; t < _numDiffusionSteps; t++)
        {
            _sqrtAlphasCumprod[t] = Math.Sqrt(_alphasCumprod[t]);
            _sqrtOneMinusAlphasCumprod[t] = Math.Sqrt(1.0 - _alphasCumprod[t]);
        }
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ExtractLayerReferences();
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultTimeGradLayers(
                Architecture, _contextLength, _forecastHorizon, _hiddenDimension,
                _numRnnLayers, _denoisingNetworkDim, _dropout));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;

        // RNN encoder
        if (idx < Layers.Count)
            _rnnEncoder = Layers[idx++];

        // Denoising network layers
        _denoisingLayers.Clear();
        while (idx < Layers.Count - 1)
            _denoisingLayers.Add(Layers[idx++]);

        // Output projection
        if (idx < Layers.Count)
            _outputProjection = Layers[idx++];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <b>For Beginners:</b> DDPM training objective: at each step, we
    /// 1. Pick a random diffusion timestep t
    /// 2. Add noise to the target at level t
    /// 3. Have the network predict the noise
    /// 4. Loss = MSE(predicted_noise, actual_noise)
    /// This trains the network to denoise, which enables sampling at inference.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);
        try
        {
            var (predicted, noiseTarget) = ForwardTraining(input, target);

            // Loss between predicted noise and actual noise
            int evalLen = Math.Min(predicted.Length, noiseTarget.Length);
            var predSlice = new Tensor<T>(new[] { evalLen });
            var targetSlice = new Tensor<T>(new[] { evalLen });
            for (int i = 0; i < evalLen; i++)
            {
                predSlice.Data.Span[i] = i < predicted.Length ? predicted[i] : NumOps.Zero;
                targetSlice.Data.Span[i] = i < noiseTarget.Length ? noiseTarget[i] : NumOps.Zero;
            }

            LastLoss = _lossFunction.CalculateLoss(predSlice.ToVector(), targetSlice.ToVector());

            var gradient = _lossFunction.CalculateDerivative(predSlice.ToVector(), targetSlice.ToVector());
            BackwardNative(Tensor<T>.FromVector(gradient, predicted.Shape));

            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TimeGrad" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "HiddenDimension", _hiddenDimension },
                { "NumRnnLayers", _numRnnLayers },
                { "NumDiffusionSteps", _numDiffusionSteps },
                { "DenoisingNetworkDim", _denoisingNetworkDim },
                { "UseNativeMode", _useNativeMode }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TimeGrad<T>(Architecture, new TimeGradOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            HiddenDimension = _hiddenDimension,
            NumRnnLayers = _numRnnLayers,
            NumDiffusionSteps = _numDiffusionSteps,
            DenoisingNetworkDim = _denoisingNetworkDim,
            NumSamples = _numSamples,
            DropoutRate = _dropout,
            BetaStart = _betaStart,
            BetaEnd = _betaEnd
        });
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_hiddenDimension);
        writer.Write(_numRnnLayers);
        writer.Write(_numDiffusionSteps);
        writer.Write(_denoisingNetworkDim);
        writer.Write(_numSamples);
        writer.Write(_dropout);
        writer.Write(_betaStart);
        writer.Write(_betaEnd);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numRnnLayers = reader.ReadInt32();
        _numDiffusionSteps = reader.ReadInt32();
        _denoisingNetworkDim = reader.ReadInt32();
        _numSamples = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _betaStart = reader.ReadDouble();
        _betaEnd = reader.ReadDouble();

        ComputeNoiseSchedule();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        // TimeGrad is inherently autoregressive (step-by-step diffusion)
        var predictions = new List<Tensor<T>>();
        var currentInput = input;
        int stepsRemaining = steps;

        while (stepsRemaining > 0)
        {
            var prediction = Forecast(currentInput, null);
            predictions.Add(prediction);
            int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining);
            stepsRemaining -= stepsUsed;

            if (stepsRemaining > 0)
                currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed);
        }

        return ConcatenatePredictions(predictions, steps);
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();
        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
            count++;
        }

        if (count > 0)
        {
            mse = NumOps.Divide(mse, NumOps.FromDouble(count));
            mae = NumOps.Divide(mae, NumOps.FromDouble(count));
        }

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = NumOps.Sqrt(mse);
        return metrics;
    }

    /// <inheritdoc/>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        var result = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            T mean = NumOps.Zero;
            for (int t = 0; t < seqLen; t++)
            {
                int idx = b * seqLen + t;
                if (idx < input.Length)
                    mean = NumOps.Add(mean, input[idx]);
            }
            mean = NumOps.Divide(mean, NumOps.FromDouble(seqLen));

            T variance = NumOps.Zero;
            for (int t = 0; t < seqLen; t++)
            {
                int idx = b * seqLen + t;
                if (idx < input.Length)
                {
                    var diff = NumOps.Subtract(input[idx], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
            }
            variance = NumOps.Divide(variance, NumOps.FromDouble(seqLen));
            T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));

            for (int t = 0; t < seqLen; t++)
            {
                int idx = b * seqLen + t;
                if (idx < input.Length && idx < result.Length)
                    result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumDiffusionSteps"] = NumOps.FromDouble(_numDiffusionSteps),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs inference via the DDPM reverse process (iterative denoising).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the core DDPM sampling algorithm:
    /// 1. Encode historical data with the RNN to get a conditioning hidden state
    /// 2. Start from pure Gaussian noise (x_T)
    /// 3. Iteratively denoise: for t = T, T-1, ..., 1:
    ///    a. Predict the noise in x_t using the denoising network (conditioned on RNN state)
    ///    b. Remove predicted noise to get x_{t-1}
    ///    c. Add a small amount of fresh noise (except at t=1)
    /// 4. The final x_0 is the forecast
    /// </remarks>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var normalized = ApplyInstanceNormalization(input);
        var current = normalized;

        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        // Step 1: Encode historical data with RNN to get conditioning hidden state
        Tensor<T> hiddenState;
        if (_rnnEncoder is not null)
            hiddenState = _rnnEncoder.Forward(current);
        else
            hiddenState = current;

        // Step 2: Generate multiple samples via DDPM reverse process and average
        var sampleAccumulator = new double[_forecastHorizon];
        var rand = RandomHelper.CreateSecureRandom();
        int effectiveSamples = Math.Max(1, _numSamples);

        for (int s = 0; s < effectiveSamples; s++)
        {
            // Start from pure noise: x_T ~ N(0, I)
            var xt = new Tensor<T>(new[] { 1, _forecastHorizon });
            for (int i = 0; i < _forecastHorizon; i++)
                xt.Data.Span[i] = NumOps.FromDouble(SampleStandardNormal(rand));

            // Iterative denoising: t = T-1, T-2, ..., 0
            for (int t = _numDiffusionSteps - 1; t >= 0; t--)
            {
                // Concatenate x_t with hidden state as conditioning for denoising network
                var denoisingInput = ConcatenateForDenoising(xt, hiddenState, t);

                // Predict noise epsilon_theta(x_t, t, h) using the denoising network
                var predictedNoise = denoisingInput;
                foreach (var layer in _denoisingLayers)
                    predictedNoise = layer.Forward(predictedNoise);

                if (_outputProjection is not null)
                    predictedNoise = _outputProjection.Forward(predictedNoise);

                // DDPM reverse step: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta) + sigma_t * z
                double alpha_t = _alphas[t];
                double alpha_bar_t = _alphasCumprod[t];
                double beta_t = _betas[t];
                double sqrtAlpha = Math.Sqrt(alpha_t);
                double noiseCoeff = beta_t / Math.Max(1e-10, Math.Sqrt(1.0 - alpha_bar_t));

                for (int i = 0; i < _forecastHorizon && i < xt.Length; i++)
                {
                    double xtVal = NumOps.ToDouble(xt[i]);
                    double epsVal = i < predictedNoise.Length ? NumOps.ToDouble(predictedNoise[i]) : 0.0;

                    // Mean of p(x_{t-1} | x_t)
                    double mean = (xtVal - noiseCoeff * epsVal) / Math.Max(1e-10, sqrtAlpha);

                    // Add noise for t > 0 (no noise at final step)
                    double sigma = t > 0 ? Math.Sqrt(beta_t) : 0.0;
                    double z = t > 0 ? SampleStandardNormal(rand) : 0.0;

                    xt.Data.Span[i] = NumOps.FromDouble(mean + sigma * z);
                }
            }

            // Accumulate this sample
            for (int i = 0; i < _forecastHorizon && i < xt.Length; i++)
                sampleAccumulator[i] += NumOps.ToDouble(xt[i]);
        }

        // Average all samples to get point forecast
        var result = new Tensor<T>(new[] { 1, _forecastHorizon });
        for (int i = 0; i < _forecastHorizon; i++)
            result.Data.Span[i] = NumOps.FromDouble(sampleAccumulator[i] / effectiveSamples);

        if (addedBatchDim && result.Rank == 2 && result.Shape[0] == 1)
            result = result.Reshape(new[] { result.Shape[1] });

        return result;
    }

    /// <summary>
    /// Concatenates the noisy sample x_t with the RNN hidden state and diffusion timestep
    /// to form the input for the denoising network.
    /// </summary>
    private Tensor<T> ConcatenateForDenoising(Tensor<T> xt, Tensor<T> hiddenState, int timestep)
    {
        // Create input that includes: [x_t values, hidden state summary, timestep embedding]
        // The denoising layers expect a fixed-size input, so we project to _denoisingNetworkDim
        int xtLen = Math.Min(xt.Length, _forecastHorizon);
        int hiddenLen = Math.Min(hiddenState.Length, _hiddenDimension);
        int totalLen = xtLen + hiddenLen + 1; // +1 for sinusoidal timestep

        var combined = new Tensor<T>(new[] { 1, totalLen });

        // Copy x_t values
        for (int i = 0; i < xtLen; i++)
            combined.Data.Span[i] = xt[i];

        // Copy hidden state (first _hiddenDimension elements)
        for (int i = 0; i < hiddenLen; i++)
            combined.Data.Span[xtLen + i] = hiddenState[i];

        // Sinusoidal timestep embedding (normalized to [-1, 1])
        double normalizedT = (_numDiffusionSteps > 1)
            ? 2.0 * timestep / (_numDiffusionSteps - 1) - 1.0
            : 0.0;
        combined.Data.Span[xtLen + hiddenLen] = NumOps.FromDouble(Math.Sin(normalizedT * Math.PI));

        return combined;
    }

    /// <summary>
    /// Samples from a standard normal distribution using Box-Muller transform.
    /// </summary>
    private double SampleStandardNormal(Random rand)
    {
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Training forward pass: adds noise to target at random timestep, predicts it.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> During training, the DDPM objective is:
    /// 1. Pick a random timestep t
    /// 2. Add noise to the target according to the schedule at step t
    /// 3. Have the denoising network predict the noise
    /// 4. Loss = MSE between predicted and actual noise
    /// </remarks>
    private (Tensor<T> Prediction, Tensor<T> NoiseTarget) ForwardTraining(Tensor<T> input, Tensor<T> target)
    {
        var normalized = ApplyInstanceNormalization(input);
        var current = normalized;

        if (current.Rank == 1)
            current = current.Reshape(new[] { 1, current.Length });

        // Encode history
        Tensor<T> hiddenState;
        if (_rnnEncoder is not null)
            hiddenState = _rnnEncoder.Forward(current);
        else
            hiddenState = current;

        // Sample random timestep
        var rand = RandomHelper.CreateSecureRandom();
        int t = rand.Next(_numDiffusionSteps);

        // Generate noise
        var noise = new Tensor<T>(target.Shape);
        for (int i = 0; i < noise.Length; i++)
            noise.Data.Span[i] = NumOps.FromDouble(SampleStandardNormal(rand));

        // Create noisy target: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        var noisyTarget = new Tensor<T>(target.Shape);
        for (int i = 0; i < target.Length; i++)
        {
            double x0 = NumOps.ToDouble(target[i]);
            double eps = NumOps.ToDouble(noise[i]);
            double xt = _sqrtAlphasCumprod[t] * x0 + _sqrtOneMinusAlphasCumprod[t] * eps;
            noisyTarget.Data.Span[i] = NumOps.FromDouble(xt);
        }

        // Predict noise
        var denoisingInput = ConcatenateForDenoising(noisyTarget, hiddenState, t);
        var predicted = denoisingInput;
        foreach (var layer in _denoisingLayers)
            predicted = layer.Forward(predicted);

        if (_outputProjection is not null)
            predicted = _outputProjection.Forward(predicted);

        return (predicted, noise);
    }

    private Tensor<T> BackwardNative(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        if (_outputProjection is not null)
            current = _outputProjection.Backward(current);

        for (int i = _denoisingLayers.Count - 1; i >= 0; i--)
            current = _denoisingLayers[i].Backward(current);

        if (_rnnEncoder is not null)
            current = _rnnEncoder.Backward(current);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
            current = current.Reshape(new[] { current.Shape[1] });

        return current;
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession == null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        int features = input.Shape.Length > 2 ? input.Shape[2] : 1;

        var inputData = new float[batchSize * seqLen * features];
        for (int i = 0; i < input.Length && i < inputData.Length; i++)
            inputData[i] = (float)NumOps.ToDouble(input[i]);

        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData, new[] { batchSize, seqLen, features });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var output = new Tensor<T>(outputShape);

        int totalElements = 1;
        foreach (var dim in outputShape) totalElements *= dim;

        for (int i = 0; i < totalElements && i < output.Length; i++)
            output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return output;
    }

    #endregion
}
