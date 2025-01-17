namespace AiDotNet.TimeSeries;

public class VectorAutoRegressionModel<T> : TimeSeriesModelBase<T>
{
    private readonly VARModelOptions<T> _varOptions;
    private Matrix<T> _coefficients;
    private Vector<T> _intercepts;
    private Matrix<T> _residuals;

    public VectorAutoRegressionModel(VARModelOptions<T> options) : base(options)
    {
        _varOptions = options;
        _coefficients = new Matrix<T>(options.OutputDimension, options.OutputDimension * options.Lag, NumOps);
        _intercepts = new Vector<T>(options.OutputDimension, NumOps);
        _residuals = new Matrix<T>(0, options.OutputDimension, NumOps);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Columns != _varOptions.OutputDimension)
        {
            throw new ArgumentException("The number of columns in x must match the OutputDimension.");
        }

        int n = x.Rows;
        int m = x.Columns;

        if (n <= _varOptions.Lag)
        {
            throw new ArgumentException($"Not enough data points. Need more than {_varOptions.Lag} observations.");
        }

        // Prepare lagged data
        Matrix<T> laggedData = PrepareLaggedData(x);

        // Estimate coefficients using OLS for each equation
        for (int i = 0; i < m; i++)
        {
            Vector<T> yi = x.GetColumn(i).Slice(_varOptions.Lag, n - _varOptions.Lag);
            Vector<T> coeffs = EstimateOLS(laggedData, yi);
            _intercepts[i] = coeffs[0];
            for (int j = 0; j < m * _varOptions.Lag; j++)
            {
                _coefficients[i, j] = coeffs[j + 1];
            }
        }

        // Calculate residuals
        _residuals = CalculateResiduals(x);
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        if (input.Columns != _varOptions.OutputDimension)
        {
            throw new ArgumentException("Input dimensions do not match the model.");
        }

        Vector<T> prediction = new Vector<T>(_varOptions.OutputDimension, NumOps);
        Vector<T> laggedValues = input.GetRow(input.Rows - 1);

        for (int i = 0; i < _varOptions.OutputDimension; i++)
        {
            prediction[i] = _intercepts[i];
            for (int j = 0; j < _varOptions.OutputDimension * _varOptions.Lag; j++)
            {
                prediction[i] = NumOps.Add(prediction[i], NumOps.Multiply(_coefficients[i, j], laggedValues[j]));
            }
        }

        return prediction;
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
        // Serialize VARModelOptions
        writer.Write(_varOptions.Lag);
        writer.Write(_varOptions.OutputDimension);

        // Serialize _coefficients
        writer.Write(_coefficients.Rows);
        writer.Write(_coefficients.Columns);
        for (int i = 0; i < _coefficients.Rows; i++)
            for (int j = 0; j < _coefficients.Columns; j++)
                writer.Write(Convert.ToDouble(_coefficients[i, j]));

        // Serialize _intercepts
        writer.Write(_intercepts.Length);
        for (int i = 0; i < _intercepts.Length; i++)
            writer.Write(Convert.ToDouble(_intercepts[i]));

        // Serialize _residuals
        writer.Write(_residuals.Rows);
        writer.Write(_residuals.Columns);
        for (int i = 0; i < _residuals.Rows; i++)
            for (int j = 0; j < _residuals.Columns; j++)
                writer.Write(Convert.ToDouble(_residuals[i, j]));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Deserialize VARModelOptions
        _varOptions.Lag = reader.ReadInt32();
        _varOptions.OutputDimension = reader.ReadInt32();

        // Deserialize _coefficients
        int coeffRows = reader.ReadInt32();
        int coeffCols = reader.ReadInt32();
        _coefficients = new Matrix<T>(coeffRows, coeffCols, NumOps);
        for (int i = 0; i < coeffRows; i++)
            for (int j = 0; j < coeffCols; j++)
                _coefficients[i, j] = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize _intercepts
        int interceptsLength = reader.ReadInt32();
        _intercepts = new Vector<T>(interceptsLength, NumOps);
        for (int i = 0; i < interceptsLength; i++)
            _intercepts[i] = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize _residuals
        int residualsRows = reader.ReadInt32();
        int residualsCols = reader.ReadInt32();
        _residuals = new Matrix<T>(residualsRows, residualsCols, NumOps);
        for (int i = 0; i < residualsRows; i++)
            for (int j = 0; j < residualsCols; j++)
                _residuals[i, j] = NumOps.FromDouble(reader.ReadDouble());
    }

    private Matrix<T> PrepareLaggedData(Matrix<T> x)
    {
        int n = x.Rows;
        int m = x.Columns;
        Matrix<T> laggedData = new Matrix<T>(n - _varOptions.Lag, m * _varOptions.Lag + 1, NumOps);

        for (int i = _varOptions.Lag; i < n; i++)
        {
            laggedData[i - _varOptions.Lag, 0] = NumOps.One;
            for (int j = 0; j < _varOptions.Lag; j++)
            {
                for (int k = 0; k < m; k++)
                {
                    laggedData[i - _varOptions.Lag, j * m + k + 1] = x[i - j - 1, k];
                }
            }
        }

        return laggedData;
    }

    private Vector<T> EstimateOLS(Matrix<T> x, Vector<T> y)
    {
        Matrix<T> xTx = x.Transpose().Multiply(x);
        Vector<T> xTy = x.Transpose().Multiply(y);

        return MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _varOptions.DecompositionType);
    }

    private Matrix<T> CalculateResiduals(Matrix<T> x)
    {
        int n = x.Rows;
        int m = x.Columns;
        Matrix<T> residuals = new Matrix<T>(n - _varOptions.Lag, m, NumOps);

        for (int i = _varOptions.Lag; i < n; i++)
        {
            Vector<T> predicted = Predict(x.Slice(i - _varOptions.Lag, _varOptions.Lag));
            Vector<T> actual = x.GetRow(i);
            residuals.SetRow(i - _varOptions.Lag, actual.Subtract(predicted));
        }

        return residuals;
    }
}