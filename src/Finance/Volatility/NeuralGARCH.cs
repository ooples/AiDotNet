using AiDotNet.Finance.Base;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Finance.Volatility;

/// <summary>
/// Neural GARCH model for forecasting asset volatility.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// NeuralGARCH extends the classic GARCH idea with a neural network, allowing
/// non-linear relationships between recent returns and future volatility.
/// </para>
/// <para>
/// <b>For Beginners:</b> GARCH is a popular statistical model for volatility.
/// This neural version learns the same idea from data instead of fixed formulas.
/// It looks at recent returns and predicts how bouncy prices will be next.
/// </para>
/// </remarks>
public class NeuralGARCH<T> : FinancialModelBase<T>, IVolatilityModel<T>
{
    #region Native Mode Fields

    private ILayer<T>? _outputLayer;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly NeuralGARCHOptions<T> _options;
    private readonly int _numAssets;
    private readonly int _lookbackWindow;
    private readonly int _hiddenSize;
    private readonly int _numLayers;
    private readonly double _dropoutRate;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a NeuralGARCH model using a pretrained ONNX model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you already have a trained ONNX file
    /// and only need fast volatility predictions.</para>
    /// </remarks>
    public NeuralGARCH(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        NeuralGARCHOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            onnxModelPath,
            options?.LookbackWindow ?? 60,
            options?.ForecastHorizon ?? 1,
            options?.NumAssets ?? 1)
    {
        _options = options ?? new NeuralGARCHOptions<T>();
        _options.Validate();

        _numAssets = _options.NumAssets;
        _lookbackWindow = _options.LookbackWindow;
        _hiddenSize = _options.HiddenSize;
        _numLayers = _options.NumLayers;
        _dropoutRate = _options.DropoutRate;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a NeuralGARCH model in native mode for training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you want to train volatility forecasting
    /// on your own returns data.</para>
    /// </remarks>
    public NeuralGARCH(
        NeuralNetworkArchitecture<T> architecture,
        NeuralGARCHOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            options?.LookbackWindow ?? 60,
            options?.ForecastHorizon ?? 1,
            options?.NumAssets ?? 1,
            lossFunction)
    {
        _options = options ?? new NeuralGARCHOptions<T>();
        _options.Validate();

        _numAssets = _options.NumAssets;
        _lookbackWindow = _options.LookbackWindow;
        _hiddenSize = _options.HiddenSize;
        _numLayers = _options.NumLayers;
        _dropoutRate = _options.DropoutRate;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the NeuralGARCH layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you provide custom layers, those are used.
    /// Otherwise, a default multi-layer perceptron is created for volatility prediction.</para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (UseNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultNeuralGARCHLayers(
                Architecture,
                _lookbackWindow,
                _numAssets,
                _hiddenSize,
                _numLayers,
                _dropoutRate));
        }

        _outputLayer = Layers.LastOrDefault();
    }

    /// <summary>
    /// Validates custom layers for the NeuralGARCH model.
    /// </summary>
    /// <param name="layers">Custom layer list.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This ensures your custom architecture has at least
    /// one layer to produce a volatility output.</para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 1)
            throw new ArgumentException("NeuralGARCH requires at least one layer.");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Runs a forward pass through the network.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how the model turns input returns into
    /// predicted volatility.</para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Forecasts volatility using native layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This routes the input through the network to get a forecast.</para>
    /// </remarks>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        return Predict(input);
    }

    /// <summary>
    /// Validates input shape for the volatility model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensures the data has the expected sequence length and asset count.</para>
    /// </remarks>
    protected override void ValidateInputShape(Tensor<T> input)
    {
        if (input.Rank != 2 && input.Rank != 3)
            throw new ArgumentException("Input must be 2D [sequence, assets] or 3D [batch, sequence, assets].", nameof(input));

        int sequenceLength = input.Rank == 3 ? input.Shape[1] : input.Shape[0];
        int assetCount = input.Rank == 3 ? input.Shape[2] : input.Shape[1];

        if (sequenceLength != _lookbackWindow)
            throw new ArgumentException($"Sequence length {sequenceLength} does not match expected {_lookbackWindow}.", nameof(input));
        if (assetCount != _numAssets)
            throw new ArgumentException($"Asset count {assetCount} does not match expected {_numAssets}.", nameof(input));
    }

    /// <summary>
    /// Core training logic for NeuralGARCH.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This computes gradients and updates weights
    /// so the model gets better at predicting volatility.</para>
    /// </remarks>
    protected override void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output)
    {
        SetTrainingMode(true);
        var outputGradient = LossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        var currentGrad = Tensor<T>.FromVector(outputGradient, output.Shape);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGrad = Layers[i].Backward(currentGrad);
        }

        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates model parameters from a flat vector.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This lets optimizers update every weight at once.</para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            layer.SetParameters(parameters.Slice(offset, layerParams.Length));
            offset += layerParams.Length;
        }
    }

    /// <summary>
    /// Returns metadata describing the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Metadata helps track configuration like lookback
    /// size and number of assets.</para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelType", "NeuralGARCH" },
                { "NumAssets", _numAssets },
                { "LookbackWindow", _lookbackWindow },
                { "ForecastHorizon", PredictionHorizon }
            }
        };
    }

    /// <summary>
    /// Creates a new instance for cloning.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is used internally to copy the model.</para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new NeuralGARCHOptions<T>
        {
            NumAssets = _numAssets,
            LookbackWindow = _lookbackWindow,
            ForecastHorizon = PredictionHorizon,
            HiddenSize = _hiddenSize,
            NumLayers = _numLayers,
            DropoutRate = _dropoutRate
        };

        return new NeuralGARCH<T>(Architecture, options, _optimizer, _lossFunction);
    }

    #endregion

    #region IVolatilityModel Implementation

    /// <summary>
    /// Forecasts future volatility.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This produces the volatility estimate for the next
    /// few steps based on recent returns.</para>
    /// </remarks>
    public Tensor<T> ForecastVolatility(Tensor<T> historicalReturns, int horizon)
    {
        var forecast = Forecast(historicalReturns);
        return EnsureHorizonShape(forecast, horizon);
    }

    /// <summary>
    /// Estimates the current volatility from recent returns.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a quick "right now" volatility estimate.</para>
    /// </remarks>
    public Tensor<T> EstimateCurrentVolatility(Tensor<T> recentReturns)
    {
        return CalculateRealizedVolatility(recentReturns);
    }

    /// <summary>
    /// Computes the correlation matrix.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Correlation tells you which assets move together.</para>
    /// </remarks>
    public Tensor<T> ComputeCorrelationMatrix(Tensor<T> returns)
    {
        var covariance = ComputeCovarianceMatrix(returns);
        int assets = covariance.Shape[0];
        var data = new T[assets * assets];

        for (int i = 0; i < assets; i++)
        {
            T varI = covariance.Data.Span[i * assets + i];
            for (int j = 0; j < assets; j++)
            {
                T varJ = covariance.Data.Span[j * assets + j];
                T denom = NumOps.Multiply(NumOps.Sqrt(varI), NumOps.Sqrt(varJ));
                data[(i * assets) + j] = NumOps.Divide(covariance.Data.Span[(i * assets) + j], denom);
            }
        }

        return new Tensor<T>(new[] { assets, assets }, new Vector<T>(data));
    }

    /// <summary>
    /// Computes the covariance matrix.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Covariance measures how assets vary together in size.</para>
    /// </remarks>
    public Tensor<T> ComputeCovarianceMatrix(Tensor<T> returns)
    {
        GetReturnMatrixShape(returns, out int samples, out int assets);
        var means = new T[assets];

        for (int a = 0; a < assets; a++)
        {
            T sum = NumOps.Zero;
            for (int s = 0; s < samples; s++)
                sum = NumOps.Add(sum, GetReturnAt(returns, s, a, assets));
            means[a] = NumOps.Divide(sum, NumOps.FromDouble(samples));
        }

        var cov = new T[assets * assets];
        double denom = Math.Max(1, samples - 1);

        for (int i = 0; i < assets; i++)
        {
            for (int j = 0; j < assets; j++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < samples; s++)
                {
                    T xi = NumOps.Subtract(GetReturnAt(returns, s, i, assets), means[i]);
                    T xj = NumOps.Subtract(GetReturnAt(returns, s, j, assets), means[j]);
                    sum = NumOps.Add(sum, NumOps.Multiply(xi, xj));
                }
                cov[(i * assets) + j] = NumOps.Divide(sum, NumOps.FromDouble(denom));
            }
        }

        return new Tensor<T>(new[] { assets, assets }, new Vector<T>(cov));
    }

    /// <summary>
    /// Calculates realized volatility from high-frequency returns.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the square root of the average squared returns.</para>
    /// </remarks>
    public Tensor<T> CalculateRealizedVolatility(Tensor<T> highFrequencyReturns)
    {
        GetReturnMatrixShape(highFrequencyReturns, out int samples, out int assets);
        var data = new T[assets];

        for (int a = 0; a < assets; a++)
        {
            T sumSq = NumOps.Zero;
            for (int s = 0; s < samples; s++)
            {
                T r = GetReturnAt(highFrequencyReturns, s, a, assets);
                sumSq = NumOps.Add(sumSq, NumOps.Multiply(r, r));
            }
            T meanSq = NumOps.Divide(sumSq, NumOps.FromDouble(samples));
            data[a] = NumOps.Sqrt(meanSq);
        }

        return new Tensor<T>(new[] { assets }, new Vector<T>(data));
    }

    /// <summary>
    /// Gets volatility-specific metrics.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns a small report card for the volatility model.</para>
    /// </remarks>
    public Dictionary<string, T> GetVolatilityMetrics()
    {
        var metrics = GetFinancialMetrics();
        metrics["NumAssets"] = NumOps.FromDouble(_numAssets);
        metrics["LookbackWindow"] = NumOps.FromDouble(_lookbackWindow);
        return metrics;
    }

    /// <summary>
    /// Gets overall financial metrics for the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This forwards to the volatility-specific metrics.</para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics() => GetVolatilityMetrics();

    #endregion

    #region Helper Methods

    /// <summary>
    /// Ensures the forecast tensor has the requested horizon.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If the model predicts only one step, we repeat it
    /// to cover the requested horizon.</para>
    /// </remarks>
    private Tensor<T> EnsureHorizonShape(Tensor<T> forecast, int horizon)
    {
        if (forecast.Rank == 2 && forecast.Shape[0] == horizon && forecast.Shape[1] == _numAssets)
            return forecast;

        var vector = forecast.ToVector();
        var data = new T[horizon * _numAssets];
        int count = Math.Min(_numAssets, vector.Length);

        for (int h = 0; h < horizon; h++)
        {
            for (int a = 0; a < _numAssets; a++)
            {
                data[(h * _numAssets) + a] = vector[a % count];
            }
        }

        return new Tensor<T>(new[] { horizon, _numAssets }, new Vector<T>(data));
    }

    /// <summary>
    /// Gets the sample and asset dimensions for return tensors.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts 3D batches into a simple list of samples.</para>
    /// </remarks>
    private static void GetReturnMatrixShape(Tensor<T> returns, out int samples, out int assets)
    {
        if (returns.Rank == 2)
        {
            samples = returns.Shape[0];
            assets = returns.Shape[1];
            return;
        }

        if (returns.Rank == 3)
        {
            samples = returns.Shape[0] * returns.Shape[1];
            assets = returns.Shape[2];
            return;
        }

        throw new ArgumentException("Returns tensor must be 2D or 3D.");
    }

    /// <summary>
    /// Reads a return value at the given sample and asset index.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This handles both 2D and 3D return tensors.</para>
    /// </remarks>
    private static T GetReturnAt(Tensor<T> returns, int sampleIndex, int assetIndex, int assets)
    {
        if (returns.Rank == 2)
        {
            return returns.Data.Span[(sampleIndex * assets) + assetIndex];
        }

        int seqLen = returns.Shape[1];
        int batch = sampleIndex / seqLen;
        int step = sampleIndex % seqLen;
        int idx = (batch * seqLen * assets) + (step * assets) + assetIndex;
        return returns.Data.Span[idx];
    }

    #endregion
}
