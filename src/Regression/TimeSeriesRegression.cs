namespace AiDotNet.Regression;

public class TimeSeriesRegression<T> : RegressionBase<T>
{
    private readonly TimeSeriesRegressionOptions<T> _options;
    private ITimeSeriesModel<T> _timeSeriesModel;
    private readonly IRegularization<T> _regularization;

    public TimeSeriesRegression(TimeSeriesRegressionOptions<T> options, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options;
        _regularization = regularization ?? new NoRegularization<T>();
        _timeSeriesModel = TimeSeriesModelFactory<T>.CreateModel(options.ModelType, options);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Prepare the data
        Matrix<T> preparedX = PrepareInputData(x, y);
        Vector<T> preparedY = PrepareTargetData(y);

        // Apply regularization to the prepared input data
        if (Regularization != null)
        {
            preparedX = Regularization.RegularizeMatrix(preparedX);
        }

        // Train the time series model
        _timeSeriesModel.Train(preparedX, preparedY);

        // Extract coefficients and apply regularization
        ExtractCoefficients();
        ApplyRegularization();

        if (_options.AutocorrelationCorrection)
        {
            ApplyAutocorrelationCorrection(preparedX, preparedY);
        }
    }

    private void ApplyRegularization()
    {
        if (Coefficients != null && Regularization != null)
        {
            Coefficients = Regularization.RegularizeCoefficients(Coefficients);
        }
    }

    private void ApplyAutocorrelationCorrection(Matrix<T> x, Vector<T> y)
    {
        const int maxIterations = 20;
        const double convergenceThreshold = 1e-5;
        T previousAutocorrelation = NumOps.Zero;
        T autocorrelation = NumOps.Zero;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Calculate residuals using the time series model
            Vector<T> predictions = _timeSeriesModel.Predict(x);
            Vector<T> residuals = y.Subtract(predictions);

            // Calculate autocorrelation
            autocorrelation = CalculateAutocorrelation(residuals);

            // Check for convergence
            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(autocorrelation, previousAutocorrelation)), NumOps.FromDouble(convergenceThreshold)))
            {
                break;
            }

            // Apply Cochrane-Orcutt transformation
            Matrix<T> correctedX = new(x.Rows - 1, x.Columns);
            Vector<T> correctedY = new(y.Length - 1);

            for (int i = 1; i < x.Rows; i++)
            {
                for (int j = 0; j < x.Columns; j++)
                {
                    correctedX[i - 1, j] = NumOps.Subtract(x[i, j], NumOps.Multiply(autocorrelation, x[i - 1, j]));
                }
                correctedY[i - 1] = NumOps.Subtract(y[i], NumOps.Multiply(autocorrelation, y[i - 1]));
            }

            // Retrain the time series model with corrected data
            _timeSeriesModel.Train(correctedX, correctedY);

            previousAutocorrelation = autocorrelation;
        }

        // Apply final correction to the original data
        if (!MathHelper.AlmostEqual(autocorrelation, NumOps.Zero))
        {
            Matrix<T> finalCorrectedX = new(x.Rows, x.Columns);
            Vector<T> finalCorrectedY = new(y.Length);

            for (int j = 0; j < x.Columns; j++)
            {
                finalCorrectedX[0, j] = x[0, j];
            }
            finalCorrectedY[0] = y[0];

            for (int i = 1; i < x.Rows; i++)
            {
                for (int j = 0; j < x.Columns; j++)
                {
                    finalCorrectedX[i, j] = NumOps.Subtract(x[i, j], NumOps.Multiply(autocorrelation, x[i - 1, j]));
                }
                finalCorrectedY[i] = NumOps.Subtract(y[i], NumOps.Multiply(autocorrelation, y[i - 1]));
            }

            // Final retraining of the time series model with the fully corrected data
            _timeSeriesModel.Train(finalCorrectedX, finalCorrectedY);
        }
    }

    private Matrix<T> PrepareInputData(Matrix<T> x, Vector<T> y)
    {
        int n = y.Length;
        int laggedFeatures = _options.LagOrder * (x.Columns + 1); // +1 for lagged y
        int trendFeatures = _options.IncludeTrend ? 1 : 0;
        int seasonalFeatures = _options.SeasonalPeriod > 0 ? _options.SeasonalPeriod - 1 : 0;
        int totalFeatures = x.Columns + laggedFeatures + trendFeatures + seasonalFeatures;

        Matrix<T> preparedX = new(n - _options.LagOrder, totalFeatures);

        // Add original features
        for (int i = _options.LagOrder; i < n; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                preparedX[i - _options.LagOrder, j] = x[i, j];
            }
        }

        // Add lagged features
        int column = x.Columns;
        for (int lag = 1; lag <= _options.LagOrder; lag++)
        {
            for (int i = _options.LagOrder; i < n; i++)
            {
                for (int j = 0; j < x.Columns; j++)
                {
                    preparedX[i - _options.LagOrder, column] = x[i - lag, j];
                    column++;
                }
                preparedX[i - _options.LagOrder, column] = y[i - lag];
                column++;
            }
        }

        // Add trend feature
        if (_options.IncludeTrend)
        {
            for (int i = 0; i < preparedX.Rows; i++)
            {
                preparedX[i, column] = NumOps.FromDouble(i + 1);
            }
            column++;
        }

        // Add seasonal features
        if (_options.SeasonalPeriod > 0)
        {
            for (int i = 0; i < preparedX.Rows; i++)
            {
                for (int s = 1; s < _options.SeasonalPeriod; s++)
                {
                    preparedX[i, column] = NumOps.FromDouble((i + _options.LagOrder) % _options.SeasonalPeriod == s ? 1 : 0);
                    column++;
                }
            }
        }

        // Apply regularization
        if (_regularization != null)
        {
            preparedX = _regularization.RegularizeMatrix(preparedX);
        }

        return preparedX;
    }

    private Vector<T> PrepareTargetData(Vector<T> y)
    {
        return new Vector<T>([.. y.Skip(_options.LagOrder)]);
    }

    private void ExtractCoefficients()
    {
        int originalFeatures = Coefficients.Length - (_options.LagOrder * (Coefficients.Length + 1) + (_options.IncludeTrend ? 1 : 0) + (_options.SeasonalPeriod > 0 ? _options.SeasonalPeriod - 1 : 0));

        // Remove trend and seasonal coefficients from the main Coefficients vector
        Coefficients = new Vector<T>([.. Coefficients.Take(originalFeatures)]);
    }

    private T CalculateAutocorrelation(Vector<T> residuals)
    {
        T numerator = NumOps.Zero;
        T denominator = NumOps.Zero;

        for (int i = 1; i < residuals.Length; i++)
        {
            numerator = NumOps.Add(numerator, NumOps.Multiply(residuals[i], residuals[i - 1]));
            denominator = NumOps.Add(denominator, NumOps.Multiply(residuals[i - 1], residuals[i - 1]));
        }

        return NumOps.Divide(numerator, denominator);
    }

    private Vector<T> ExtractTrendCoefficients()
    {
        if (_options.IncludeTrend)
        {
            int trendIndex = Coefficients.Length - 1;
            if (_options.SeasonalPeriod > 0)
            {
                trendIndex -= (_options.SeasonalPeriod - 1);
            }
            return new Vector<T>(new[] { Coefficients[trendIndex] });
        }
        return new Vector<T>(0);
    }

    private Vector<T> ExtractSeasonalCoefficients()
    {
        if (_options.SeasonalPeriod > 0)
        {
            return new Vector<T>(Coefficients.Skip(Coefficients.Length - (_options.SeasonalPeriod - 1)).ToArray());
        }
        return new Vector<T>(0);
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        Matrix<T> preparedInput = PrepareInputData(input, new Vector<T>(input.Rows)); // Dummy y vector
        Vector<T> predictions = base.Predict(preparedInput);

        Vector<T> trendCoefficients = ExtractTrendCoefficients();
        Vector<T> seasonalCoefficients = ExtractSeasonalCoefficients();

        // Add trend and seasonality components
        for (int i = 0; i < predictions.Length; i++)
        {
            if (_options.IncludeTrend)
            {
                predictions[i] = NumOps.Add(predictions[i], NumOps.Multiply(trendCoefficients[0], NumOps.FromDouble(i + 1)));
            }

            if (_options.SeasonalPeriod > 0)
            {
                int seasonIndex = i % _options.SeasonalPeriod;
                if (seasonIndex > 0)
                {
                    predictions[i] = NumOps.Add(predictions[i], seasonalCoefficients[seasonIndex - 1]);
                }
            }
        }

        return predictions;
    }

    protected override ModelType GetModelType() => ModelType.TimeSeriesRegression;

    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize TimeSeriesRegression specific data
            writer.Write(_options.LagOrder);
            writer.Write(_options.IncludeTrend);
            writer.Write(_options.SeasonalPeriod);
            writer.Write(_options.AutocorrelationCorrection);
            writer.Write((int)_options.ModelType);

            // Serialize the time series model
            byte[] modelData = _timeSeriesModel.Serialize();
            writer.Write(modelData.Length);
            writer.Write(modelData);

            return ms.ToArray();
        }
    }

    public override void Deserialize(byte[] modelData)
    {
        using (MemoryStream ms = new MemoryStream(modelData))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize TimeSeriesRegression specific data
            _options.LagOrder = reader.ReadInt32();
            _options.IncludeTrend = reader.ReadBoolean();
            _options.SeasonalPeriod = reader.ReadInt32();
            _options.AutocorrelationCorrection = reader.ReadBoolean();
            _options.ModelType = (TimeSeriesModelType)reader.ReadInt32();

            // Deserialize the time series model
            int modelDataLength = reader.ReadInt32();
            byte[] timeSeriesModelData = reader.ReadBytes(modelDataLength);
            _timeSeriesModel = TimeSeriesModelFactory<T>.CreateModel(_options.ModelType, _options);
            _timeSeriesModel.Deserialize(timeSeriesModelData);
        }
    }
}