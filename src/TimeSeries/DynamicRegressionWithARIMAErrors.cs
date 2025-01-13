namespace AiDotNet.TimeSeries;

public class DynamicRegressionWithARIMAErrors<T> : TimeSeriesModelBase<T>
{
    private DynamicRegressionWithARIMAErrorsOptions<T> _arimaOptions;
    private IRegularization<T> _regularization;
    private Vector<T> _regressionCoefficients;
    private Vector<T> _arCoefficients;
    private Vector<T> _maCoefficients;
    private Vector<T> _differenced;
    private T _intercept;

    public DynamicRegressionWithARIMAErrors(DynamicRegressionWithARIMAErrorsOptions<T> options) : base(options)
    {
        _arimaOptions = options;
        _regressionCoefficients = new Vector<T>(options.ExternalRegressors, NumOps);
        _arCoefficients = new Vector<T>(options.AROrder, NumOps);
        _maCoefficients = new Vector<T>(options.MAOrder, NumOps);
        _differenced = new Vector<T>(0, NumOps);
        _intercept = NumOps.Zero;
        _regularization = options.Regularization ?? new NoRegularization<T>();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Step 1: Perform differencing if necessary
        Vector<T> diffY = DifferenceTimeSeries(y, _arimaOptions.DifferenceOrder);

        // Step 2: Fit regression model
        FitRegressionModel(x, diffY);

        // Step 3: Extract residuals
        Vector<T> residuals = ExtractResiduals(x, diffY);

        // Step 4: Fit ARIMA model to residuals
        FitARIMAModel(residuals);

        // Step 5: Update model parameters
        UpdateModelParameters();
    }

    public override Vector<T> Predict(Matrix<T> xNew)
    {
        Vector<T> predictions = new Vector<T>(xNew.Rows, NumOps);

        for (int t = 0; t < xNew.Rows; t++)
        {
            T prediction = _intercept;

            // Apply regression component
            for (int i = 0; i < xNew.Columns; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(xNew[t, i], _regressionCoefficients[i]));
            }

            // Apply ARIMA component
            for (int p = 0; p < _arimaOptions.AROrder; p++)
            {
                if (t - p - 1 >= 0)
                {
                    prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[p], NumOps.Subtract(predictions[t - p - 1], _intercept)));
                }
            }

            for (int q = 0; q < _arimaOptions.MAOrder; q++)
            {
                if (t - q - 1 >= 0)
                {
                    T error = NumOps.Subtract(predictions[t - q - 1], xNew[t - q - 1, 0]); // Assuming the first column is the target variable
                    prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[q], error));
                }
            }

            predictions[t] = prediction;
        }

        // Reverse differencing if necessary
        if (_arimaOptions.DifferenceOrder > 0)
        {
            predictions = InverseDifferenceTimeSeries(predictions, _differenced);
        }

        return predictions;
    }

    private Vector<T> DifferenceTimeSeries(Vector<T> y, int order)
    {
        Vector<T> diffY = y;
        for (int d = 0; d < order; d++)
        {
            Vector<T> temp = new Vector<T>(diffY.Length - 1, NumOps);
            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = NumOps.Subtract(diffY[i + 1], diffY[i]);
            }
            _differenced = new Vector<T>(diffY.Take(order));
            diffY = temp;
        }

        return diffY;
    }

    private Vector<T> InverseDifferenceTimeSeries(Vector<T> diffY, Vector<T> original)
    {
        Vector<T> y = diffY;
        for (int d = _arimaOptions.DifferenceOrder - 1; d >= 0; d--)
        {
            Vector<T> temp = new Vector<T>(y.Length + 1, NumOps);
            temp[0] = original[d];
            for (int i = 1; i < temp.Length; i++)
            {
                temp[i] = NumOps.Add(temp[i - 1], y[i - 1]);
            }
            y = temp;
        }

        return y;
    }

    private void FitRegressionModel(Matrix<T> x, Vector<T> y)
    {
        // Use OLS or other regression method to fit the model
        Matrix<T> xT = x.Transpose();
        Matrix<T> xTx = xT * x;
        Vector<T> xTy = xT * y;

        _regressionCoefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _arimaOptions.DecompositionType);
        _intercept = NumOps.Divide(y.Sum(), NumOps.FromDouble(y.Length));
    }

    private Vector<T> ExtractResiduals(Matrix<T> x, Vector<T> y)
    {
        Vector<T> predictions = x * _regressionCoefficients;
        return y - predictions;
    }

    private void FitARIMAModel(Vector<T> residuals)
    {
        int p = _arimaOptions.AROrder;
        int q = _arimaOptions.MAOrder;
        int maxLag = Math.Max(p, q);

        // Calculate autocorrelations
        T[] autocorrelations = CalculateAutocorrelations(residuals, maxLag);

        // Estimate AR coefficients using Yule-Walker equations
        EstimateARCoefficients(autocorrelations, p);

        // Estimate MA coefficients using innovation algorithm
        EstimateMACoefficients(residuals, autocorrelations, q);

        // Perform joint optimization of AR and MA coefficients
        OptimizeARMACoefficients(residuals);
    }

    private void EstimateARCoefficients(T[] autocorrelations, int p)
    {
        Matrix<T> R = new Matrix<T>(p, p, NumOps);
        Vector<T> r = new Vector<T>(p, NumOps);

        for (int i = 0; i < p; i++)
        {
            r[i] = autocorrelations[i + 1];
            for (int j = 0; j < p; j++)
            {
                R[i, j] = autocorrelations[Math.Abs(i - j)];
            }
        }

        try
        {
            _arCoefficients = MatrixSolutionHelper.SolveLinearSystem(R, r, _arimaOptions.DecompositionType);
        }
        catch (Exception ex)
        {
            // Handle potential numerical instability
            Console.WriteLine($"Error in AR coefficient estimation: {ex.Message}");
            _arCoefficients = new Vector<T>(p, NumOps); // Initialize with zeros
        }
    }

    private void EstimateMACoefficients(Vector<T> residuals, T[] autocorrelations, int q)
    {
        _maCoefficients = new Vector<T>(q, NumOps);
        Vector<T> v = new Vector<T>(q + 1, NumOps);
        v[0] = autocorrelations[0];

        for (int k = 1; k <= q; k++)
        {
            T sum = NumOps.Zero;
            for (int j = 1; j < k; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_maCoefficients[j - 1], v[k - j]));
            }
            _maCoefficients[k - 1] = NumOps.Divide(NumOps.Subtract(autocorrelations[k], sum), v[0]);
        
            for (int j = 1; j <= k; j++)
            {
                v[j] = NumOps.Subtract(v[j], NumOps.Multiply(_maCoefficients[k - 1], v[k - j]));
            }
            v[k] = NumOps.Multiply(_maCoefficients[k - 1], v[0]);
        }
    }

    private void OptimizeARMACoefficients(Vector<T> residuals)
    {
        int maxIterations = 100;
        T tolerance = NumOps.FromDouble(1e-6);
        T prevLikelihood = NumOps.MinValue;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Compute model residuals
            Vector<T> modelResiduals = ComputeModelResiduals(residuals);

            // Compute log-likelihood
            T likelihood = ComputeLogLikelihood(modelResiduals);

            // Check for convergence
            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(likelihood, prevLikelihood)), tolerance))
            {
                break;
            }

            prevLikelihood = likelihood;

            // Update AR coefficients
            UpdateARCoefficients(modelResiduals);

            // Update MA coefficients
            UpdateMACoefficients(modelResiduals);
        }
    }

    private Vector<T> ComputeModelResiduals(Vector<T> residuals)
    {
        int n = residuals.Length;
        Vector<T> modelResiduals = new Vector<T>(n, NumOps);

        for (int t = 0; t < n; t++)
        {
            T prediction = NumOps.Zero;

            for (int i = 0; i < _arimaOptions.AROrder && t - i - 1 >= 0; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[i], residuals[t - i - 1]));
            }

            for (int i = 0; i < _arimaOptions.MAOrder && t - i - 1 >= 0; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[i], modelResiduals[t - i - 1]));
            }

            modelResiduals[t] = NumOps.Subtract(residuals[t], prediction);
        }

        return modelResiduals;
    }

    private T ComputeLogLikelihood(Vector<T> modelResiduals)
    {
        T sumSquaredResiduals = NumOps.Zero;
        foreach (T residual in modelResiduals)
        {
            sumSquaredResiduals = NumOps.Add(sumSquaredResiduals, NumOps.Multiply(residual, residual));
        }

        T variance = NumOps.Divide(sumSquaredResiduals, NumOps.FromDouble(modelResiduals.Length));
        T logLikelihood = NumOps.Multiply(NumOps.FromDouble(-0.5 * modelResiduals.Length), 
            NumOps.Add(NumOps.Log(NumOps.Multiply(NumOps.FromDouble(2 * Math.PI), variance)), NumOps.One));

        return logLikelihood;
    }

    private void UpdateARCoefficients(Vector<T> modelResiduals)
    {
        // Implement a gradient descent step for AR coefficients
        T learningRate = NumOps.FromDouble(0.01);
        for (int i = 0; i < _arimaOptions.AROrder; i++)
        {
            T gradient = ComputeARGradient(modelResiduals, i);
            _arCoefficients[i] = NumOps.Add(_arCoefficients[i], NumOps.Multiply(learningRate, gradient));
        }
    }

    private void UpdateMACoefficients(Vector<T> modelResiduals)
    {
        // Implement a gradient descent step for MA coefficients
        T learningRate = NumOps.FromDouble(0.01);
        for (int i = 0; i < _arimaOptions.MAOrder; i++)
        {
            T gradient = ComputeMAGradient(modelResiduals, i);
            _maCoefficients[i] = NumOps.Add(_maCoefficients[i], NumOps.Multiply(learningRate, gradient));
        }
    }

    private T ComputeARGradient(Vector<T> modelResiduals, int lag)
    {
        T gradient = NumOps.Zero;
        for (int t = lag + 1; t < modelResiduals.Length; t++)
        {
            gradient = NumOps.Add(gradient, NumOps.Multiply(modelResiduals[t], modelResiduals[t - lag - 1]));
        }

        return NumOps.Multiply(NumOps.FromDouble(-2), gradient);
    }

    private T ComputeMAGradient(Vector<T> modelResiduals, int lag)
    {
        T gradient = NumOps.Zero;
        for (int t = lag + 1; t < modelResiduals.Length; t++)
        {
            T prevError = NumOps.Zero;
            for (int i = 0; i < _arimaOptions.MAOrder && t - i - 1 >= 0; i++)
            {
                prevError = NumOps.Add(prevError, NumOps.Multiply(_maCoefficients[i], modelResiduals[t - i - 1]));
            }
            gradient = NumOps.Add(gradient, NumOps.Multiply(modelResiduals[t], prevError));
        }

        return NumOps.Multiply(NumOps.FromDouble(-2), gradient);
    }

    private T[] CalculateAutocorrelations(Vector<T> y, int maxLag)
    {
        T[] autocorrelations = new T[maxLag + 1];
        T mean = StatisticsHelper<T>.CalculateMean(y);
        T variance = StatisticsHelper<T>.CalculateVariance(y);

        for (int lag = 0; lag <= maxLag; lag++)
        {
            T sum = NumOps.Zero;
            int n = y.Length - lag;

            for (int t = 0; t < n; t++)
            {
                T diff1 = NumOps.Subtract(y[t], mean);
                T diff2 = NumOps.Subtract(y[t + lag], mean);
                sum = NumOps.Add(sum, NumOps.Multiply(diff1, diff2));
            }

            autocorrelations[lag] = NumOps.Divide(sum, NumOps.Multiply(NumOps.FromDouble(n), variance));
        }

        return autocorrelations;
    }

    private void UpdateModelParameters()
    {
        // 1. Ensure stationarity of AR process
        EnsureARStationarity();

        // 2. Ensure invertibility of MA process
        EnsureMAInvertibility();

        // 3. Normalize regression coefficients
        NormalizeRegressionCoefficients();

        // 4. Apply regularization to prevent overfitting
        ApplyRegularization();

        // 5. Update intercept
        UpdateIntercept();
    }

    private void ApplyRegularization()
    {
        _regressionCoefficients = _regularization.RegularizeCoefficients(_regressionCoefficients);
    }

    private void EnsureARStationarity()
    {
        // Use the constraint that the roots of the AR polynomial should lie outside the unit circle
        // A simple approximation: ensure the sum of absolute AR coefficients is less than 1
        T sum = NumOps.Zero;
        for (int i = 0; i < _arCoefficients.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Abs(_arCoefficients[i]));
        }

        if (NumOps.GreaterThan(sum, NumOps.One))
        {
            T scaleFactor = NumOps.Divide(NumOps.FromDouble(0.99), sum);
            for (int i = 0; i < _arCoefficients.Length; i++)
            {
                _arCoefficients[i] = NumOps.Multiply(_arCoefficients[i], scaleFactor);
            }
        }
    }

    private void EnsureMAInvertibility()
    {
        // Similar to AR stationarity, ensure the sum of absolute MA coefficients is less than 1
        T sum = NumOps.Zero;
        for (int i = 0; i < _maCoefficients.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Abs(_maCoefficients[i]));
        }

        if (NumOps.GreaterThan(sum, NumOps.One))
        {
            T scaleFactor = NumOps.Divide(NumOps.FromDouble(0.99), sum);
            for (int i = 0; i < _maCoefficients.Length; i++)
            {
                _maCoefficients[i] = NumOps.Multiply(_maCoefficients[i], scaleFactor);
            }
        }
    }

    private void NormalizeRegressionCoefficients()
    {
        // Normalize regression coefficients to have unit norm
        T norm = NumOps.Sqrt(_regressionCoefficients.DotProduct(_regressionCoefficients));
        if (!NumOps.Equals(norm, NumOps.Zero))
        {
            for (int i = 0; i < _regressionCoefficients.Length; i++)
            {
                _regressionCoefficients[i] = NumOps.Divide(_regressionCoefficients[i], norm);
            }
        }
    }

    private void UpdateIntercept()
    {
        // Adjust intercept based on the mean of the differenced series
        if (_arimaOptions.DifferenceOrder > 0 && _differenced.Length > 0)
        {
            T diffMean = StatisticsHelper<T>.CalculateMean(_differenced);
            _intercept = NumOps.Add(_intercept, diffMean);
        }
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>
        {
            ["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions),
            ["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions),
            ["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions),
            ["MAPE"] = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(yTest, predictions)
        };

        return metrics;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_regressionCoefficients.Length);
        for (int i = 0; i < _regressionCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_regressionCoefficients[i]));

        writer.Write(_arCoefficients.Length);
        for (int i = 0; i < _arCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_arCoefficients[i]));

        writer.Write(_maCoefficients.Length);
        for (int i = 0; i < _maCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_maCoefficients[i]));

        writer.Write(_differenced.Length);
        for (int i = 0; i < _differenced.Length; i++)
            writer.Write(Convert.ToDouble(_differenced[i]));

        writer.Write(Convert.ToDouble(_intercept));

        writer.Write(JsonConvert.SerializeObject(_options));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        int regressionCoefficientsLength = reader.ReadInt32();
        _regressionCoefficients = new Vector<T>(regressionCoefficientsLength, NumOps);
        for (int i = 0; i < regressionCoefficientsLength; i++)
            _regressionCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int arCoefficientsLength = reader.ReadInt32();
        _arCoefficients = new Vector<T>(arCoefficientsLength, NumOps);
        for (int i = 0; i < arCoefficientsLength; i++)
            _arCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int maCoeffientsLength = reader.ReadInt32();
        _maCoefficients = new Vector<T>(maCoeffientsLength, NumOps);
        for (int i = 0; i < maCoeffientsLength; i++)
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int differencedLength = reader.ReadInt32();
        _differenced = new Vector<T>(differencedLength, NumOps);
        for (int i = 0; i < differencedLength; i++)
            _differenced[i] = NumOps.FromDouble(reader.ReadDouble());

        _intercept = NumOps.FromDouble(reader.ReadDouble());

        string optionsJson = reader.ReadString();
        _options = JsonConvert.DeserializeObject<DynamicRegressionWithARIMAErrorsOptions<T>>(optionsJson) ?? new();
    }
}