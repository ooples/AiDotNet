namespace AiDotNet.TimeSeries;

public class ARModel<T> : TimeSeriesModelBase<T>
{
    private Vector<T> _arCoefficients;
    private int _arOrder;
    private readonly double _learningRate;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    public ARModel(ARModelOptions<T> options) : base(options)
    {
        _arOrder = options.AROrder;
        _learningRate = options.LearningRate;
        _maxIterations = options.MaxIterations;
        _tolerance = options.Tolerance;
        _arCoefficients = Vector<T>.Empty();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Initialize coefficients
        _arCoefficients = new Vector<T>(_arOrder);

        Vector<T> prevGradAR = new Vector<T>(_arOrder);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            Vector<T> residuals = CalculateResiduals(y);
            Vector<T> gradAR = CalculateGradients(y, residuals);

            // Update coefficients using gradient descent
            for (int i = 0; i < _arOrder; i++)
            {
                _arCoefficients[i] = NumOps.Subtract(_arCoefficients[i], NumOps.Multiply(NumOps.FromDouble(_learningRate), gradAR[i]));
            }

            // Check for convergence
            if (CheckConvergence(gradAR, prevGradAR))
            {
                break;
            }

            prevGradAR = gradAR;
        }
    }

    private Vector<T> CalculateResiduals(Vector<T> y)
    {
        Vector<T> residuals = new Vector<T>(y.Length);
        for (int t = _arOrder; t < y.Length; t++)
        {
            T yHat = Predict(y, t);
            residuals[t] = NumOps.Subtract(y[t], yHat);
        }

        return residuals;
    }

    private Vector<T> CalculateGradients(Vector<T> y, Vector<T> residuals)
    {
        Vector<T> gradAR = new Vector<T>(_arOrder);

        for (int t = _arOrder; t < y.Length; t++)
        {
            for (int i = 0; i < _arOrder; i++)
            {
                gradAR[i] = NumOps.Add(gradAR[i], NumOps.Multiply(residuals[t], y[t - i - 1]));
            }
        }

        return gradAR;
    }

    private bool CheckConvergence(Vector<T> gradAR, Vector<T> prevGradAR)
    {
        T diffAR = gradAR.Subtract(prevGradAR).Norm();
        return NumOps.LessThan(diffAR, NumOps.FromDouble(_tolerance));
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new Vector<T>(input.Rows);
        for (int t = 0; t < input.Rows; t++)
        {
            predictions[t] = Predict(input.GetRow(t), t);
        }

        return predictions;
    }

    private T Predict(Vector<T> y, int t)
    {
        T prediction = NumOps.Zero;
        for (int i = 0; i < _arOrder && t - i - 1 >= 0; i++)
        {
            prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[i], y[t - i - 1]));
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
        writer.Write(_arOrder);
        for (int i = 0; i < _arOrder; i++)
        {
            writer.Write(Convert.ToDouble(_arCoefficients[i]));
        }
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _arOrder = reader.ReadInt32();
        _arCoefficients = new Vector<T>(_arOrder);
        for (int i = 0; i < _arOrder; i++)
        {
            _arCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}