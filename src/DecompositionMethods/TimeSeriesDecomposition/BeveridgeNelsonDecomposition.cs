namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class BeveridgeNelsonDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly BeveridgeNelsonAlgorithm _algorithm;
    private readonly ARIMAOptions<T> _arimaOptions;
    private readonly int _forecastHorizon;
    private readonly Matrix<T> _multivariateSeries;

    public BeveridgeNelsonDecomposition(Vector<T> timeSeries, 
        BeveridgeNelsonAlgorithm algorithm = BeveridgeNelsonAlgorithm.Standard,
        ARIMAOptions<T>? arimaOptions = null,
        int forecastHorizon = 100,
        Matrix<T>? multivariateSeries = null) 
        : base(timeSeries)
    {
        _algorithm = algorithm;
        _arimaOptions = arimaOptions ?? new ARIMAOptions<T> { P = 1, D = 1, Q = 1 };
        _forecastHorizon = forecastHorizon;
        _multivariateSeries = multivariateSeries ?? new Matrix<T>(TimeSeries.Length, 1);
    }

    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case BeveridgeNelsonAlgorithm.Standard:
                DecomposeStandard();
                break;
            case BeveridgeNelsonAlgorithm.ARIMA:
                DecomposeARIMA();
                break;
            case BeveridgeNelsonAlgorithm.Multivariate:
                DecomposeMultivariate();
                break;
            default:
                throw new NotImplementedException($"Beveridge-Nelson decomposition algorithm {_algorithm} is not implemented.");
        }
    }

    private void DecomposeStandard()
    {
        // Implement standard Beveridge-Nelson decomposition
        Vector<T> trend = CalculateStandardTrend();
        Vector<T> cycle = CalculateStandardCycle(trend);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    private void DecomposeARIMA()
    {
        // Implement ARIMA-based Beveridge-Nelson decomposition
        var arimaModel = new ARIMAModel<T>(_arimaOptions);
        arimaModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        Vector<T> trend = CalculateARIMATrend(arimaModel);
        Vector<T> cycle = CalculateARIMACycle(trend);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    private void DecomposeMultivariate()
    {
        int n = _multivariateSeries.Rows;
        int m = _multivariateSeries.Columns;

        // Step 1: Estimate VAR model
        var varOptions = new VARModelOptions<T> { Lag = 1, OutputDimension = m }; // Assuming first-order VAR
        var varModel = new VectorAutoRegressionModel<T>(varOptions);
        varModel.Train(_multivariateSeries, Vector<T>.Empty()); // Passing an empty vector as y since it's not used in VAR

        // Step 2: Calculate long-run impact matrix
        Matrix<T> longRunImpact = CalculateLongRunImpactMatrix(varModel, varOptions);

        // Step 3: Calculate permanent component
        Matrix<T> permanent = new Matrix<T>(n, m);
        for (int i = 0; i < n; i++)
        {
            Vector<T> cumSum = new Vector<T>(m);
            for (int j = 0; j <= i; j++)
            {
                cumSum = cumSum.Add(longRunImpact.Multiply(_multivariateSeries.GetRow(j)));
            }

            permanent.SetRow(i, cumSum);
        }

        // Step 4: Calculate transitory component
        Matrix<T> transitory = _multivariateSeries.Subtract(permanent);

        // Add components
        AddComponent(DecompositionComponentType.Trend, permanent.GetColumn(0));
        AddComponent(DecompositionComponentType.Cycle, transitory.GetColumn(0));

        // If there are additional series, add them as separate components
        for (int i = 1; i < m; i++)
        {
            AddComponent((DecompositionComponentType)((int)DecompositionComponentType.Trend + i), permanent.GetColumn(i));
            AddComponent((DecompositionComponentType)((int)DecompositionComponentType.Cycle + i), transitory.GetColumn(i));
        }
    }

    private Matrix<T> CalculateLongRunImpactMatrix(VectorAutoRegressionModel<T> varModel, VARModelOptions<T> varOptions)
    {
        int m = _multivariateSeries.Columns;
        Matrix<T> identity = Matrix<T>.CreateIdentity(m);
        Matrix<T> coeffSum = new Matrix<T>(m, m, NumOps);

        // Assuming Coefficients is a single matrix representing all lags
        Matrix<T> coefficients = varModel.Coefficients;

        // Sum up the coefficient matrices for each lag
        for (int i = 0; i < varOptions.Lag; i++)
        {
            int startRow = i * m;
            Matrix<T> lagCoeff = coefficients.Slice(startRow, m);
            coeffSum = coeffSum.Add(lagCoeff);
        }

        return identity.Subtract(coeffSum).Inverse();
    }

    private Vector<T> CalculateStandardTrend()
    {
        int n = TimeSeries.Length;
        var trend = new Vector<T>(n, NumOps);
        var differenced = new Vector<T>(n - 1, NumOps);

        // Calculate first differences
        for (int i = 1; i < n; i++)
        {
            differenced[i - 1] = NumOps.Subtract(TimeSeries[i], TimeSeries[i - 1]);
        }

        // Calculate mean of differenced series
        T meanDiff = StatisticsHelper<T>.CalculateMean(differenced);

        // Calculate autocovariance of differenced series
        var autocovariance = new Vector<T>(n, NumOps);
        for (int k = 0; k < n - 1; k++)
        {
            T sum = NumOps.Zero;
            for (int i = k; i < n - 1; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(
                    NumOps.Subtract(differenced[i], meanDiff),
                    NumOps.Subtract(differenced[i - k], meanDiff)
                ));
            }
            autocovariance[k] = NumOps.Divide(sum, NumOps.FromDouble(n - 1 - k));
        }

        // Calculate long-run variance
        T longRunVariance = autocovariance[0];
        for (int k = 1; k < n - 1; k++)
        {
            longRunVariance = NumOps.Add(longRunVariance, NumOps.Multiply(NumOps.FromDouble(2), autocovariance[k]));
        }

        // Calculate trend
        trend[0] = TimeSeries[0];
        for (int i = 1; i < n; i++)
        {
            T adjustment = NumOps.Multiply(NumOps.FromDouble(i), NumOps.Divide(longRunVariance, autocovariance[0]));
            trend[i] = NumOps.Add(TimeSeries[i], NumOps.Multiply(adjustment, meanDiff));
        }

        return trend;
    }

    private Vector<T> CalculateStandardCycle(Vector<T> trend)
    {
        // Calculate cycle as the difference between the time series and the trend
        return TimeSeries.Subtract(trend);
    }

    private Vector<T> CalculateARIMATrend(ARIMAModel<T> model)
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n, NumOps);

        // Calculate input dimension based on ARIMA order
        int inputDimension = Math.Max(_arimaOptions.P, _arimaOptions.Q + 1);
        if (_arimaOptions.SeasonalPeriod > 0)
        {
            inputDimension = Math.Max(inputDimension, 
                Math.Max(_arimaOptions.P + _arimaOptions.SeasonalP * _arimaOptions.SeasonalPeriod, 
                         _arimaOptions.Q + _arimaOptions.SeasonalQ * _arimaOptions.SeasonalPeriod + 1));
        }

        // Create input matrix for prediction
        Matrix<T> input = new Matrix<T>(1, inputDimension);

        // Calculate the long-run forecast for each time point
        for (int i = 0; i < n; i++)
        {
            // Set the input values based on the available data
            for (int j = 0; j < inputDimension; j++)
            {
                if (i - j >= 0)
                {
                    input[0, j] = TimeSeries[i - j];
                }
                else
                {
                    input[0, j] = NumOps.Zero;
                }
            }

            // Predict future values
            Vector<T> forecast = model.Predict(input);

            // Calculate the long-run forecast (Beveridge-Nelson trend)
            T longRunForecast = forecast[0];
            for (int k = 1; k < forecast.Length; k++)
            {
                longRunForecast = NumOps.Add(longRunForecast, NumOps.Subtract(forecast[k], forecast[k - 1]));
            }

            trend[i] = longRunForecast;
        }

        return trend;
    }

    private Vector<T> CalculateARIMACycle(Vector<T> trend)
    {
        // Calculate cycle as the difference between the time series and the trend
        return TimeSeries.Subtract(trend);
    }
}