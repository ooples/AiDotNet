namespace AiDotNet.TimeSeries;

public class MAModel<T> : TimeSeriesModelBase<T>
{
    private Vector<T> _maCoefficients;
    private int _maOrder;
    private readonly double _learningRate;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    public MAModel(MAModelOptions<T> options) : base(options)
    {
        _maOrder = options.MAOrder;
        _learningRate = options.LearningRate;
        _maxIterations = options.MaxIterations;
        _tolerance = options.Tolerance;
        _maCoefficients = Vector<T>.Empty();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Initialize coefficients
        _maCoefficients = new Vector<T>(_maOrder);

        Vector<T> prevGradMA = new Vector<T>(_maOrder);
        Vector<T> errors = new Vector<T>(y.Length);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            Vector<T> residuals = CalculateResiduals(y, errors);
            Vector<T> gradMA = CalculateGradients(errors, residuals);

            // Update coefficients using gradient descent
            for (int i = 0; i < _maOrder; i++)
            {
                _maCoefficients[i] = NumOps.Subtract(_maCoefficients[i], NumOps.Multiply(NumOps.FromDouble(_learningRate), gradMA[i]));
            }

            // Check for convergence
            if (CheckConvergence(gradMA, prevGradMA))
            {
                break;
            }

            prevGradMA = gradMA;
        }
    }

    private Vector<T> CalculateResiduals(Vector<T> y, Vector<T> errors)
    {
        Vector<T> residuals = new Vector<T>(y.Length);
        for (int t = 0; t < y.Length; t++)
        {
            T yHat = Predict(errors, t);
            residuals[t] = NumOps.Subtract(y[t], yHat);
            errors[t] = residuals[t];
        }

        return residuals;
    }

    private Vector<T> CalculateGradients(Vector<T> errors, Vector<T> residuals)
    {
        Vector<T> gradMA = new Vector<T>(_maOrder);

        for (int t = _maOrder; t < errors.Length; t++)
        {
            for (int i = 0; i < _maOrder; i++)
            {
                gradMA[i] = NumOps.Add(gradMA[i], NumOps.Multiply(residuals[t], errors[t - i - 1]));
            }
        }

        return gradMA;
    }

    private bool CheckConvergence(Vector<T> gradMA, Vector<T> prevGradMA)
    {
        T diffMA = gradMA.Subtract(prevGradMA).Norm();
        return NumOps.LessThan(diffMA, NumOps.FromDouble(_tolerance));
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new Vector<T>(input.Rows);
        Vector<T> errors = new Vector<T>(input.Rows);
        for (int t = 0; t < input.Rows; t++)
        {
            predictions[t] = Predict(errors, t);
            if (t < input.Rows - 1)
            {
                errors[t] = NumOps.Subtract(input[t + 1, 0], predictions[t]);
            }
        }

        return predictions;
    }

    private T Predict(Vector<T> errors, int t)
    {
        T prediction = NumOps.Zero;
        for (int i = 0; i < _maOrder && t - i - 1 >= 0; i++)
        {
            prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[i], errors[t - i - 1]));
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
        writer.Write(_maOrder);
        for (int i = 0; i < _maOrder; i++)
        {
            writer.Write(Convert.ToDouble(_maCoefficients[i]));
        }
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _maOrder = reader.ReadInt32();
        _maCoefficients = new Vector<T>(_maOrder);
        for (int i = 0; i < _maOrder; i++)
        {
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}