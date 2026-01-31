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
/// Realized Volatility Transformer for attention-based volatility forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This model applies transformer attention to recent returns to forecast volatility.
/// </para>
/// <para>
/// <b>For Beginners:</b> Transformers learn which past time points matter most.
/// This helps the model focus on recent shocks or patterns when predicting volatility.
/// </para>
/// </remarks>
public class RealizedVolatilityTransformer<T> : FinancialModelBase<T>, IVolatilityModel<T>
{
    #region Native Mode Fields

    private ILayer<T>? _outputLayer;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly RealizedVolatilityTransformerOptions<T> _options;
    private readonly int _numAssets;
    private readonly int _lookbackWindow;
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numLayers;
    private readonly double _dropoutRate;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a RealizedVolatilityTransformer using a pretrained ONNX model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you want fast inference from an ONNX file.</para>
    /// </remarks>
    public RealizedVolatilityTransformer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        RealizedVolatilityTransformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            onnxModelPath,
            options?.LookbackWindow ?? 90,
            options?.ForecastHorizon ?? 5,
            options?.NumAssets ?? 1)
    {
        _options = options ?? new RealizedVolatilityTransformerOptions<T>();
        _options.Validate();

        _numAssets = _options.NumAssets;
        _lookbackWindow = _options.LookbackWindow;
        _hiddenSize = _options.HiddenSize;
        _numHeads = _options.NumHeads;
        _numLayers = _options.NumLayers;
        _dropoutRate = _options.DropoutRate;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a RealizedVolatilityTransformer in native mode for training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you want to train the model on your own data.</para>
    /// </remarks>
    public RealizedVolatilityTransformer(
        NeuralNetworkArchitecture<T> architecture,
        RealizedVolatilityTransformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            options?.LookbackWindow ?? 90,
            options?.ForecastHorizon ?? 5,
            options?.NumAssets ?? 1,
            lossFunction)
    {
        _options = options ?? new RealizedVolatilityTransformerOptions<T>();
        _options.Validate();

        _numAssets = _options.NumAssets;
        _lookbackWindow = _options.LookbackWindow;
        _hiddenSize = _options.HiddenSize;
        _numHeads = _options.NumHeads;
        _numLayers = _options.NumLayers;
        _dropoutRate = _options.DropoutRate;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes transformer layers for volatility forecasting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If no custom layers are supplied, the model creates
    /// a default transformer stack with attention blocks.</para>
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultRealizedVolatilityTransformerLayers(
                Architecture,
                _lookbackWindow,
                _numAssets,
                _hiddenSize,
                _numHeads,
                _numLayers,
                _dropoutRate));
        }

        _outputLayer = Layers.LastOrDefault();
    }

    /// <summary>
    /// Validates custom layers for the transformer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This ensures the architecture has at least one layer.</para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 1)
            throw new ArgumentException("RealizedVolatilityTransformer requires at least one layer.");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Runs a forward pass through the network.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This turns historical returns into a volatility forecast.</para>
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
    /// Forecasts volatility using native transformer layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calls the transformer to produce volatility predictions.</para>
    /// </remarks>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        return Predict(input);
    }

    /// <summary>
    /// Validates input shape for the transformer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensures the returns tensor matches the expected size.</para>
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
    /// Core training logic for the transformer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This updates model weights based on prediction error.</para>
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
    /// <para><b>For Beginners:</b> This lets optimizers update all weights at once.</para>
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
    /// <para><b>For Beginners:</b> Metadata records configuration like lookback and heads.</para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelType", "RealizedVolatilityTransformer" },
                { "NumAssets", _numAssets },
                { "LookbackWindow", _lookbackWindow },
                { "ForecastHorizon", PredictionHorizon },
                { "NumHeads", _numHeads },
                { "NumLayers", _numLayers }
            }
        };
    }

    /// <summary>
    /// Creates a new instance for cloning.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Used internally to make a full copy of the model.</para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new RealizedVolatilityTransformerOptions<T>
        {
            NumAssets = _numAssets,
            LookbackWindow = _lookbackWindow,
            ForecastHorizon = PredictionHorizon,
            HiddenSize = _hiddenSize,
            NumHeads = _numHeads,
            NumLayers = _numLayers,
            DropoutRate = _dropoutRate
        };

        return new RealizedVolatilityTransformer<T>(Architecture, options, _optimizer, _lossFunction);
    }

    #endregion

    #region IVolatilityModel Implementation

    /// <summary>
    /// Forecasts future volatility.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Produces the volatility estimate for the next horizon steps.</para>
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
    /// <para><b>For Beginners:</b> This is a quick estimate based on the latest data.</para>
    /// </remarks>
    public Tensor<T> EstimateCurrentVolatility(Tensor<T> recentReturns)
    {
        return CalculateRealizedVolatility(recentReturns);
    }

    /// <summary>
    /// Computes the correlation matrix.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Correlation shows which assets move together.</para>
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
    /// <para><b>For Beginners:</b> Covariance measures how returns move together in size.</para>
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
    /// <para><b>For Beginners:</b> This is the square root of average squared returns.</para>
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
    /// <para><b>For Beginners:</b> Returns a summary of settings and current status.</para>
    /// </remarks>
    [ThreadStatic]
    private static bool _inGetVolatilityMetrics;

    public Dictionary<string, T> GetVolatilityMetrics()
    {
        // Guard against infinite recursion - return empty metrics if already in this method
        if (_inGetVolatilityMetrics)
        {
            return new Dictionary<string, T>();
        }

        try
        {
            _inGetVolatilityMetrics = true;

            // Build base metrics directly to avoid infinite recursion with GetFinancialMetrics
            var metrics = new Dictionary<string, T>
            {
                ["NumAssets"] = NumOps.FromDouble(_numAssets),
                ["LookbackWindow"] = NumOps.FromDouble(_lookbackWindow),
                ["ForecastHorizon"] = NumOps.FromDouble(PredictionHorizon),
                ["HiddenSize"] = NumOps.FromDouble(_hiddenSize),
                ["NumHeads"] = NumOps.FromDouble(_numHeads),
                ["NumLayers"] = NumOps.FromDouble(_numLayers),
                ["ParameterCount"] = NumOps.FromDouble(ParameterCount)
            };

            return metrics;
        }
        finally
        {
            _inGetVolatilityMetrics = false;
        }
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
    /// <para><b>For Beginners:</b> If the model outputs one step, we repeat it to fill the horizon.</para>
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
