namespace AiDotNet.TimeSeries;

public class ARMAModel<T> : TimeSeriesModelBase<T>
{
    private Vector<T> _arCoefficients;
    private Vector<T> _maCoefficients;
    private int _arOrder;
    private int _maOrder;
    private readonly double _learningRate;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    public ARMAModel(ARMAOptions<T> options) : base(options)
    {
        _arOrder = options.AROrder;
        _maOrder = options.MAOrder;
        _learningRate = options.LearningRate;
        _maxIterations = options.MaxIterations;
        _tolerance = options.Tolerance;
        _arCoefficients = Vector<T>.Empty();
        _maCoefficients = Vector<T>.Empty();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Initialize coefficients
        _arCoefficients = new Vector<T>(_arOrder);
        _maCoefficients = new Vector<T>(_maOrder);

        Vector<T> prevGradAR = new Vector<T>(_arOrder);
        Vector<T> prevGradMA = new Vector<T>(_maOrder);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            Vector<T> residuals = CalculateResiduals(y);
            (Vector<T> gradAR, Vector<T> gradMA) = CalculateGradients(y, residuals);

            // Update coefficients using gradient descent
            for (int i = 0; i < _arOrder; i++)
            {
                _arCoefficients[i] = NumOps.Subtract(_arCoefficients[i], NumOps.Multiply(NumOps.FromDouble(_learningRate), gradAR[i]));
            }
            for (int i = 0; i < _maOrder; i++)
            {
                _maCoefficients[i] = NumOps.Subtract(_maCoefficients[i], NumOps.Multiply(NumOps.FromDouble(_learningRate), gradMA[i]));
            }

            // Check for convergence
            if (CheckConvergence(gradAR, gradMA, prevGradAR, prevGradMA))
            {
                break;
            }

            prevGradAR = gradAR;
            prevGradMA = gradMA;
        }
    }

    private Vector<T> CalculateResiduals(Vector<T> y)
    {
        Vector<T> residuals = new Vector<T>(y.Length);
        for (int t = Math.Max(_arOrder, _maOrder); t < y.Length; t++)
        {
            T yHat = Predict(y, t);
            residuals[t] = NumOps.Subtract(y[t], yHat);
        }

        return residuals;
    }

    private (Vector<T>, Vector<T>) CalculateGradients(Vector<T> y, Vector<T> residuals)
    {
        Vector<T> gradAR = new Vector<T>(_arOrder);
        Vector<T> gradMA = new Vector<T>(_maOrder);

        for (int t = Math.Max(_arOrder, _maOrder); t < y.Length; t++)
        {
            for (int i = 0; i < _arOrder; i++)
            {
                gradAR[i] = NumOps.Add(gradAR[i], NumOps.Multiply(residuals[t], y[t - i - 1]));
            }
            for (int i = 0; i < _maOrder; i++)
            {
                gradMA[i] = NumOps.Add(gradMA[i], NumOps.Multiply(residuals[t], residuals[t - i - 1]));
            }
        }

        return (gradAR, gradMA);
    }

    private bool CheckConvergence(Vector<T> gradAR, Vector<T> gradMA, Vector<T> prevGradAR, Vector<T> prevGradMA)
    {
        T diffAR = gradAR.Subtract(prevGradAR).Norm();
        T diffMA = gradMA.Subtract(prevGradMA).Norm();

        return NumOps.LessThan(diffAR, NumOps.FromDouble(_tolerance)) && NumOps.LessThan(diffMA, NumOps.FromDouble(_tolerance));
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
        for (int i = 0; i < _maOrder && t - i - 1 >= 0; i++)
        {
            T residual = NumOps.Subtract(y[t - i - 1], Predict(y, t - i - 1));
            prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[i], residual));
        }

        return prediction;
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new()
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
        writer.Write(_maOrder);
        for (int i = 0; i < _arOrder; i++)
        {
            writer.Write(Convert.ToDouble(_arCoefficients[i]));
        }
        for (int i = 0; i < _maOrder; i++)
        {
            writer.Write(Convert.ToDouble(_maCoefficients[i]));
        }
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _arOrder = reader.ReadInt32();
        _maOrder = reader.ReadInt32();
        _arCoefficients = new Vector<T>(_arOrder);
        _maCoefficients = new Vector<T>(_maOrder);
        for (int i = 0; i < _arOrder; i++)
        {
            _arCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        for (int i = 0; i < _maOrder; i++)
        {
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}