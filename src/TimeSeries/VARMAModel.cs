namespace AiDotNet.TimeSeries;

public class VARMAModel<T> : VectorAutoRegressionModel<T>
{
    private readonly VARMAModelOptions<T> _varmaOptions;
    private Matrix<T> _maCoefficients;
    private Matrix<T> _residuals;

    public VARMAModel(VARMAModelOptions<T> options) : base(options)
    {
        _varmaOptions = options;
        _maCoefficients = new Matrix<T>(options.OutputDimension, options.OutputDimension * options.MaLag);
        _residuals = Matrix<T>.Empty();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        base.Train(x, y); // Train VAR part

        // Calculate residuals
        _residuals = CalculateResiduals(x, y);

        // Estimate MA coefficients using residuals
        EstimateMACoefficients();
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> arPrediction = base.Predict(input);
        Vector<T> maPrediction = PredictMA();

        return arPrediction.Add(maPrediction);
    }

    private void EstimateMACoefficients()
    {
        int n = _residuals.Rows;
        int m = _residuals.Columns;

        if (n <= _varmaOptions.MaLag)
        {
            throw new InvalidOperationException($"Not enough residuals. Need more than {_varmaOptions.MaLag} observations.");
        }

        // Prepare lagged residuals
        Matrix<T> laggedResiduals = PrepareLaggedResiduals();

        // Estimate MA coefficients using OLS for each equation
        for (int i = 0; i < m; i++)
        {
            Vector<T> yi = _residuals.GetColumn(i).Slice(_varmaOptions.MaLag, n - _varmaOptions.MaLag);
            Vector<T> coeffs = SolveOLS(laggedResiduals, yi);
            for (int j = 0; j < m * _varmaOptions.MaLag; j++)
            {
                _maCoefficients[i, j] = coeffs[j];
            }
        }
    }

    private Matrix<T> PrepareLaggedResiduals()
    {
        int n = _residuals.Rows;
        int m = _residuals.Columns;
        Matrix<T> laggedResiduals = new Matrix<T>(n - _varmaOptions.MaLag, m * _varmaOptions.MaLag);

        for (int i = _varmaOptions.MaLag; i < n; i++)
        {
            for (int j = 0; j < _varmaOptions.MaLag; j++)
            {
                for (int k = 0; k < m; k++)
                {
                    laggedResiduals[i - _varmaOptions.MaLag, j * m + k] = _residuals[i - j - 1, k];
                }
            }
        }

        return laggedResiduals;
    }

    private Vector<T> PredictMA()
    {
        Vector<T> maPrediction = new Vector<T>(_varmaOptions.OutputDimension);
        Vector<T> lastResiduals = _residuals.GetRow(_residuals.Rows - 1);

        for (int i = 0; i < _varmaOptions.OutputDimension; i++)
        {
            for (int j = 0; j < _varmaOptions.OutputDimension * _varmaOptions.MaLag; j++)
            {
                maPrediction[i] = NumOps.Add(maPrediction[i], NumOps.Multiply(_maCoefficients[i, j], lastResiduals[j]));
            }
        }

        return maPrediction;
    }

    private Matrix<T> CalculateResiduals(Matrix<T> x, Vector<T> y)
    {
        Vector<T> predictions = base.Predict(x);
        return Matrix<T>.FromColumns(y.Subtract(predictions));
    }

    private Vector<T> SolveOLS(Matrix<T> x, Vector<T> y)
    {
        return MatrixSolutionHelper.SolveLinearSystem(x.Transpose().Multiply(x), x.Transpose().Multiply(y), _varmaOptions.DecompositionType);
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        base.SerializeCore(writer);

        // Serialize VARMAModelOptions
        writer.Write(_varmaOptions.MaLag);

        // Serialize _maCoefficients
        writer.Write(_maCoefficients.Rows);
        writer.Write(_maCoefficients.Columns);
        for (int i = 0; i < _maCoefficients.Rows; i++)
            for (int j = 0; j < _maCoefficients.Columns; j++)
                writer.Write(Convert.ToDouble(_maCoefficients[i, j]));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        base.DeserializeCore(reader);

        // Deserialize VARMAModelOptions
        _varmaOptions.MaLag = reader.ReadInt32();

        // Deserialize _maCoefficients
        int maCoeffRows = reader.ReadInt32();
        int maCoeffCols = reader.ReadInt32();
        _maCoefficients = new Matrix<T>(maCoeffRows, maCoeffCols);
        for (int i = 0; i < maCoeffRows; i++)
            for (int j = 0; j < maCoeffCols; j++)
                _maCoefficients[i, j] = NumOps.FromDouble(reader.ReadDouble());
    }
}