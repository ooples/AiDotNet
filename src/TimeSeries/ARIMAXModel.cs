namespace AiDotNet.TimeSeries;

public class ARIMAXModel<T> : TimeSeriesModelBase<T>
{
    private ARIMAXModelOptions<T> _arimaxOptions;
    private Vector<T> _arCoefficients;
    private Vector<T> _maCoefficients;
    private Vector<T> _exogenousCoefficients;
    private Vector<T> _differenced;
    private T _intercept;

    public ARIMAXModel(ARIMAXModelOptions<T>? options = null) : base(options ?? new ARIMAXModelOptions<T>())
    {
        _arimaxOptions = options ?? new();
        _arCoefficients = new Vector<T>(_arimaxOptions.AROrder, NumOps);
        _maCoefficients = new Vector<T>(_arimaxOptions.MAOrder, NumOps);
        _exogenousCoefficients = new Vector<T>(_arimaxOptions.ExogenousVariables, NumOps);
        _differenced = new Vector<T>(0, NumOps);
        _intercept = NumOps.Zero;
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Step 1: Perform differencing if necessary
        Vector<T> diffY = DifferenceTimeSeries(y, _arimaxOptions.DifferenceOrder);

        // Step 2: Fit ARIMAX model
        FitARIMAXModel(x, diffY);

        // Step 3: Update model parameters
        UpdateModelParameters();
    }

    public override Vector<T> Predict(Matrix<T> xNew)
    {
        Vector<T> predictions = new Vector<T>(xNew.Rows, NumOps);

        for (int t = 0; t < xNew.Rows; t++)
        {
            T prediction = _intercept;

            // Apply exogenous component
            for (int i = 0; i < xNew.Columns; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(xNew[t, i], _exogenousCoefficients[i]));
            }

            // Apply AR component
            for (int p = 0; p < _arimaxOptions.AROrder; p++)
            {
                if (t - p - 1 >= 0)
                {
                    prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[p], NumOps.Subtract(predictions[t - p - 1], _intercept)));
                }
            }

            // Apply MA component
            for (int q = 0; q < _arimaxOptions.MAOrder; q++)
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
        if (_arimaxOptions.DifferenceOrder > 0)
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
            _differenced = new Vector<T>(order, NumOps);
            for (int i = 0; i < order; i++)
            {
                _differenced[i] = diffY[i];
            }
            diffY = temp;
        }

        return diffY;
    }

    private Vector<T> InverseDifferenceTimeSeries(Vector<T> diffY, Vector<T> original)
    {
        Vector<T> y = diffY;
        for (int d = _arimaxOptions.DifferenceOrder - 1; d >= 0; d--)
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

    private void FitARIMAXModel(Matrix<T> x, Vector<T> y)
    {
        // Implement ARIMAX model fitting
        // This is a simplified version and may need to be expanded for more accurate results

        // Fit exogenous variables
        Matrix<T> xT = x.Transpose();
        Matrix<T> xTx = xT * x;
        Vector<T> xTy = xT * y;

        _exogenousCoefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _arimaxOptions.DecompositionType);

        // Extract residuals
        Vector<T> residuals = y - (x * _exogenousCoefficients);

        // Fit ARMA model to residuals
        FitARMAModel(residuals);

        _intercept = NumOps.Divide(y.Sum(), NumOps.FromDouble(y.Length));
    }

    private void FitARMAModel(Vector<T> residuals)
    {
        int p = _arimaxOptions.AROrder;
        int q = _arimaxOptions.MAOrder;

        // Calculate autocorrelations
        T[] autocorrelations = CalculateAutocorrelations(residuals, Math.Max(p, q));

        // Update AR coefficients using Yule-Walker equations
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

        // Solve Yule-Walker equations
        _arCoefficients = MatrixSolutionHelper.SolveLinearSystem(R, r, _arimaxOptions.DecompositionType);

        // Update MA coefficients using a simple method
        for (int i = 0; i < q; i++)
        {
            _maCoefficients[i] = NumOps.Multiply(NumOps.FromDouble(0.5), autocorrelations[i + 1]);
        }
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
        // Implement any necessary parameter updates or constraints
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
        writer.Write(_arCoefficients.Length);
        for (int i = 0; i < _arCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_arCoefficients[i]));

        writer.Write(_maCoefficients.Length);
        for (int i = 0; i < _maCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_maCoefficients[i]));

        writer.Write(_exogenousCoefficients.Length);
        for (int i = 0; i < _exogenousCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_exogenousCoefficients[i]));

        writer.Write(_differenced.Length);
        for (int i = 0; i < _differenced.Length; i++)
            writer.Write(Convert.ToDouble(_differenced[i]));

        writer.Write(Convert.ToDouble(_intercept));

        writer.Write(JsonConvert.SerializeObject(_arimaxOptions));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        int arCoefficientsLength = reader.ReadInt32();
        _arCoefficients = new Vector<T>(arCoefficientsLength, NumOps);
        for (int i = 0; i < arCoefficientsLength; i++)
            _arCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int maCoefficientsLength = reader.ReadInt32();
        _maCoefficients = new Vector<T>(maCoefficientsLength, NumOps);
        for (int i = 0; i < maCoefficientsLength; i++)
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int exogenousCoefficientsLength = reader.ReadInt32();
        _exogenousCoefficients = new Vector<T>(exogenousCoefficientsLength, NumOps);
        for (int i = 0; i < exogenousCoefficientsLength; i++)
            _exogenousCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int differencedLength = reader.ReadInt32();
        _differenced = new Vector<T>(differencedLength, NumOps);
        for (int i = 0; i < differencedLength; i++)
            _differenced[i] = NumOps.FromDouble(reader.ReadDouble());

        _intercept = NumOps.FromDouble(reader.ReadDouble());

        string optionsJson = reader.ReadString();
        _arimaxOptions = JsonConvert.DeserializeObject<ARIMAXModelOptions<T>>(optionsJson) ?? new();
    }
}