namespace AiDotNet.TimeSeries;

public class GARCHModel<T> : TimeSeriesModelBase<T>
{
    private GARCHModelOptions<T> _garchOptions;
    private ITimeSeriesModel<T> _meanModel;
    private Vector<T> _omega; // Constant term in variance equation
    private Vector<T> _alpha; // ARCH coefficients
    private Vector<T> _beta;  // GARCH coefficients
    private Vector<T> _residuals;
    private Vector<T> _conditionalVariances;

    public GARCHModel(GARCHModelOptions<T>? options = null) : base(options ?? new GARCHModelOptions<T>())
    {
        _garchOptions = (GARCHModelOptions<T>)_options;
        _meanModel = _garchOptions.MeanModel ?? new ARIMAModel<T>();
        _omega = new Vector<T>(1);
        _alpha = new Vector<T>(_garchOptions.ARCHOrder);
        _beta = new Vector<T>(_garchOptions.GARCHOrder);
        _residuals = new Vector<T>(0);
        _conditionalVariances = new Vector<T>(0);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Step 1: Train the mean model
        _meanModel.Train(x, y);

        // Step 2: Calculate residuals from the mean model
        Vector<T> meanPredictions = _meanModel.Predict(x);
        _residuals = y.Subtract(meanPredictions);

        // Step 3: Initialize GARCH parameters
        InitializeParameters();

        // Step 4: Estimate GARCH parameters using Maximum Likelihood Estimation
        EstimateParameters(_residuals);

        // Step 5: Calculate final residuals and conditional variances
        CalculateResidualsAndVariances(_residuals);
    }

    public override Vector<T> Predict(Matrix<T> xNew)
    {
        int forecastHorizon = xNew.Rows;
        Vector<T> predictions = new Vector<T>(forecastHorizon);
        Vector<T> variances = new Vector<T>(forecastHorizon);

        // Predict mean using the mean model
        Vector<T> meanPredictions = _meanModel.Predict(xNew);

        // Initialize with the last known values
        T lastVariance = _conditionalVariances[_conditionalVariances.Length - 1];
        T lastResidual = _residuals[_residuals.Length - 1];

        for (int t = 0; t < forecastHorizon; t++)
        {
            // Calculate conditional variance
            T variance = _omega[0];
            for (int i = 0; i < _garchOptions.ARCHOrder; i++)
            {
                if (t - i - 1 >= 0)
                {
                    variance = NumOps.Add(variance, NumOps.Multiply(_alpha[i], NumOps.Multiply(lastResidual, lastResidual)));
                }
                else
                {
                    variance = NumOps.Add(variance, NumOps.Multiply(_alpha[i], NumOps.Multiply(lastResidual, lastResidual)));
                }
            }
            for (int i = 0; i < _garchOptions.GARCHOrder; i++)
            {
                if (t - i - 1 >= 0)
                {
                    variance = NumOps.Add(variance, NumOps.Multiply(_beta[i], variances[t - i - 1]));
                }
                else
                {
                    variance = NumOps.Add(variance, NumOps.Multiply(_beta[i], lastVariance));
                }
            }

            variances[t] = variance;
            lastVariance = variance;

            // Generate prediction
            T standardNormal = GenerateStandardNormal();
            T residual = NumOps.Multiply(NumOps.Sqrt(variance), standardNormal);
            predictions[t] = NumOps.Add(meanPredictions[t], residual);

            // Update last residual for the next iteration
            lastResidual = residual;
        }

        return predictions;
    }

    private T GenerateStandardNormal()
    {
        // Box-Muller transform to generate standard normal random variable
        Random random = new Random();
        double u1 = random.NextDouble();
        double u2 = random.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);

        return NumOps.FromDouble(z);
    }

    private void InitializeParameters()
    {
        // Initialize parameters with small positive values
        _omega[0] = NumOps.FromDouble(0.01);
        for (int i = 0; i < _alpha.Length; i++)
        {
            _alpha[i] = NumOps.FromDouble(0.05);
        }
        for (int i = 0; i < _beta.Length; i++)
        {
            _beta[i] = NumOps.FromDouble(0.85);
        }
    }

    private void EstimateParameters(Vector<T> y)
    {
        int maxIterations = 10000;
        T initialLearningRate = NumOps.FromDouble(0.01);
        T minLearningRate = NumOps.FromDouble(1e-6);
        T convergenceThreshold = NumOps.FromDouble(1e-6);
        T momentumFactor = NumOps.FromDouble(0.9);

        Vector<T> previousOmega = _omega.Copy();
        Vector<T> previousAlpha = _alpha.Copy();
        Vector<T> previousBeta = _beta.Copy();

        Vector<T> velocityOmega = new Vector<T>(_omega.Length);
        Vector<T> velocityAlpha = new Vector<T>(_alpha.Length);
        Vector<T> velocityBeta = new Vector<T>(_beta.Length);

        T previousLogLikelihood = CalculateLogLikelihood(y);
        T currentLearningRate = initialLearningRate;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            Vector<T> gradientOmega = CalculateGradient(y, GradientType.Omega);
            Vector<T> gradientAlpha = CalculateGradient(y, GradientType.Alpha);
            Vector<T> gradientBeta = CalculateGradient(y, GradientType.Beta);

            // Update velocities (momentum)
            velocityOmega = velocityOmega.Multiply(momentumFactor).Add(gradientOmega.Multiply(currentLearningRate));
            velocityAlpha = velocityAlpha.Multiply(momentumFactor).Add(gradientAlpha.Multiply(currentLearningRate));
            velocityBeta = velocityBeta.Multiply(momentumFactor).Add(gradientBeta.Multiply(currentLearningRate));

            // Update parameters
            _omega = _omega.Subtract(velocityOmega);
            _alpha = _alpha.Subtract(velocityAlpha);
            _beta = _beta.Subtract(velocityBeta);

            // Ensure parameters stay within valid ranges
            ConstrainParameters();

            // Check for convergence
            T currentLogLikelihood = CalculateLogLikelihood(y);
            T improvement = NumOps.Subtract(currentLogLikelihood, previousLogLikelihood);

            if (NumOps.LessThan(NumOps.Abs(improvement), convergenceThreshold))
            {
                break;
            }

            // Adaptive learning rate
            if (NumOps.GreaterThan(improvement, NumOps.Zero))
            {
                currentLearningRate = NumOps.Multiply(currentLearningRate, NumOps.FromDouble(1.05)); // Increase learning rate
                previousOmega = _omega.Copy();
                previousAlpha = _alpha.Copy();
                previousBeta = _beta.Copy();
                previousLogLikelihood = currentLogLikelihood;
            }
            else
            {
                currentLearningRate = NumOps.Multiply(currentLearningRate, NumOps.FromDouble(0.5)); // Decrease learning rate
                _omega = previousOmega.Copy();
                _alpha = previousAlpha.Copy();
                _beta = previousBeta.Copy();

                if (NumOps.LessThan(currentLearningRate, minLearningRate))
                {
                    break; // Stop if learning rate becomes too small
                }
            }
        }
    }

    private Vector<T> CalculateGradient(Vector<T> y, GradientType gradientType)
    {
        T epsilon = NumOps.FromDouble(1e-6); // Small value for numerical differentiation
        T twoEpsilon = NumOps.Multiply(epsilon, NumOps.FromDouble(2));
        T logLikelihood = CalculateLogLikelihood(y);

        if (gradientType == GradientType.Omega)
        {
            Vector<T> gradient = new Vector<T>(1);
            T originalOmega = _omega[0];

            _omega[0] = NumOps.Add(originalOmega, epsilon);
            T logLikelihoodPlus = CalculateLogLikelihood(y);

            _omega[0] = NumOps.Subtract(originalOmega, epsilon);
            T logLikelihoodMinus = CalculateLogLikelihood(y);

            gradient[0] = NumOps.Divide(NumOps.Subtract(logLikelihoodPlus, logLikelihoodMinus), twoEpsilon);
            _omega[0] = originalOmega; // Restore original value

            return gradient;
        }
        else if (gradientType == GradientType.Alpha)
        {
            Vector<T> gradient = new Vector<T>(_garchOptions.ARCHOrder);
            for (int i = 0; i < _garchOptions.ARCHOrder; i++)
            {
                T originalAlpha = _alpha[i];

                _alpha[i] = NumOps.Add(originalAlpha, epsilon);
                T logLikelihoodPlus = CalculateLogLikelihood(y);

                _alpha[i] = NumOps.Subtract(originalAlpha, epsilon);
                T logLikelihoodMinus = CalculateLogLikelihood(y);

                gradient[i] = NumOps.Divide(NumOps.Subtract(logLikelihoodPlus, logLikelihoodMinus), twoEpsilon);
                _alpha[i] = originalAlpha; // Restore original value
            }
            return gradient;
        }
        else // beta
        {
            Vector<T> gradient = new Vector<T>(_garchOptions.GARCHOrder);
            for (int i = 0; i < _garchOptions.GARCHOrder; i++)
            {
                T originalBeta = _beta[i];

                _beta[i] = NumOps.Add(originalBeta, epsilon);
                T logLikelihoodPlus = CalculateLogLikelihood(y);

                _beta[i] = NumOps.Subtract(originalBeta, epsilon);
                T logLikelihoodMinus = CalculateLogLikelihood(y);

                gradient[i] = NumOps.Divide(NumOps.Subtract(logLikelihoodPlus, logLikelihoodMinus), twoEpsilon);
                _beta[i] = originalBeta; // Restore original value
            }
            return gradient;
        }
    }

    private T CalculateLogLikelihood(Vector<T> y)
    {
        T logLikelihood = NumOps.Zero;
        int n = y.Length;
        Vector<T> conditionalVariances = CalculateConditionalVariances(y);

        for (int t = Math.Max(_garchOptions.ARCHOrder, _garchOptions.GARCHOrder); t < n; t++)
        {
            T variance = conditionalVariances[t];
            T residual = y[t]; // Assuming zero mean for simplicity
            T term = NumOps.Add(NumOps.Log(variance), NumOps.Divide(NumOps.Multiply(residual, residual), variance));
            logLikelihood = NumOps.Add(logLikelihood, term);
        }

        return NumOps.Multiply(NumOps.FromDouble(-0.5), logLikelihood);
    }

    private Vector<T> CalculateConditionalVariances(Vector<T> y)
    {
        int n = y.Length;
        Vector<T> conditionalVariances = new Vector<T>(n);
        T unconditionalVariance = CalculateUnconditionalVariance(y);

        // Initialize with unconditional variance
        for (int t = 0; t < Math.Max(_garchOptions.ARCHOrder, _garchOptions.GARCHOrder); t++)
        {
            conditionalVariances[t] = unconditionalVariance;
        }

        // Calculate conditional variances
        for (int t = Math.Max(_garchOptions.ARCHOrder, _garchOptions.GARCHOrder); t < n; t++)
        {
            T variance = _omega[0];
            for (int i = 0; i < _garchOptions.ARCHOrder; i++)
            {
                variance = NumOps.Add(variance, NumOps.Multiply(_alpha[i], NumOps.Multiply(y[t - i - 1], y[t - i - 1])));
            }
            for (int i = 0; i < _garchOptions.GARCHOrder; i++)
            {
                variance = NumOps.Add(variance, NumOps.Multiply(_beta[i], conditionalVariances[t - i - 1]));
            }
            conditionalVariances[t] = variance;
        }

        return conditionalVariances;
    }

    private void ConstrainParameters()
    {
        // Ensure all parameters are non-negative
        _omega[0] = MathHelper.Max(_omega[0], NumOps.Zero);
        for (int i = 0; i < _alpha.Length; i++)
        {
            _alpha[i] = MathHelper.Max(_alpha[i], NumOps.Zero);
        }
        for (int i = 0; i < _beta.Length; i++)
        {
            _beta[i] = MathHelper.Max(_beta[i], NumOps.Zero);
        }

        // Ensure the sum of ARCH and GARCH coefficients is less than 1 for stationarity
        T sum = NumOps.Add(_alpha.Sum(), _beta.Sum());
        if (NumOps.GreaterThan(sum, NumOps.One))
        {
            T scaleFactor = NumOps.Divide(NumOps.FromDouble(0.99), sum);
            _alpha = _alpha.Multiply(scaleFactor);
            _beta = _beta.Multiply(scaleFactor);
        }
    }

    private void CalculateResidualsAndVariances(Vector<T> residuals)
    {
        int n = residuals.Length;
        _conditionalVariances = new Vector<T>(n);

        // Initialize with unconditional variance
        T unconditionalVariance = CalculateUnconditionalVariance(residuals);
        for (int t = 0; t < Math.Max(_garchOptions.ARCHOrder, _garchOptions.GARCHOrder); t++)
        {
            _conditionalVariances[t] = unconditionalVariance;
        }

        // Calculate conditional variances
        for (int t = Math.Max(_garchOptions.ARCHOrder, _garchOptions.GARCHOrder); t < n; t++)
        {
            T variance = _omega[0];
            for (int i = 0; i < _garchOptions.ARCHOrder; i++)
            {
                variance = NumOps.Add(variance, NumOps.Multiply(_alpha[i], NumOps.Multiply(residuals[t - i - 1], residuals[t - i - 1])));
            }
            for (int i = 0; i < _garchOptions.GARCHOrder; i++)
            {
                variance = NumOps.Add(variance, NumOps.Multiply(_beta[i], _conditionalVariances[t - i - 1]));
            }
            _conditionalVariances[t] = variance;
        }
    }

    private T CalculateUnconditionalVariance(Vector<T> y)
    {
        return StatisticsHelper<T>.CalculateVariance(y);
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
        SerializationHelper<T>.SerializeVector(writer, _omega);
        SerializationHelper<T>.SerializeVector(writer, _alpha);
        SerializationHelper<T>.SerializeVector(writer, _beta);
        SerializationHelper<T>.SerializeVector(writer, _residuals);
        SerializationHelper<T>.SerializeVector(writer, _conditionalVariances);

        writer.Write(JsonConvert.SerializeObject(_garchOptions));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _omega = SerializationHelper<T>.DeserializeVector(reader);
        _alpha = SerializationHelper<T>.DeserializeVector(reader);
        _beta = SerializationHelper<T>.DeserializeVector(reader);
        _residuals = SerializationHelper<T>.DeserializeVector(reader);
        _conditionalVariances = SerializationHelper<T>.DeserializeVector(reader);

        string optionsJson = reader.ReadString();
        _garchOptions = JsonConvert.DeserializeObject<GARCHModelOptions<T>>(optionsJson) ?? new();
        _options = _garchOptions;
    }
}