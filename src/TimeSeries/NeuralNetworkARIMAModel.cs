global using AiDotNet.NeuralNetworks;
global using AiDotNet.ActivationFunctions;

namespace AiDotNet.TimeSeries;

public class NeuralNetworkARIMAModel<T> : TimeSeriesModelBase<T>
{
    private readonly NeuralNetworkARIMAOptions<T> _nnarimaOptions;
    private Vector<T> _arParameters;
    private Vector<T> _maParameters;
    private Vector<T> _residuals;
    private Vector<T> _fitted;
    private readonly IOptimizer<T> _optimizer;
    private Vector<T> _y;
    private readonly INeuralNetwork<T> _neuralNetwork;

    public NeuralNetworkARIMAModel(NeuralNetworkARIMAOptions<T>? options = null) : base(options ?? new())
    {
        _nnarimaOptions = options ?? new NeuralNetworkARIMAOptions<T>();
        _optimizer = _nnarimaOptions.Optimizer ?? new LBFGSOptimizer<T>();
        _arParameters = Vector<T>.Empty();
        _maParameters = Vector<T>.Empty();
        _residuals = Vector<T>.Empty();
        _fitted = Vector<T>.Empty();
        _y = Vector<T>.Empty();
        _neuralNetwork = _nnarimaOptions.NeuralNetwork ?? CreateDefaultNeuralNetwork();
    }

    private NeuralNetwork<T> CreateDefaultNeuralNetwork()
    {
        var defaultArchitecture = new NeuralNetworkArchitecture<T>
        {
            LayerSizes = new List<int> 
            { 
                _nnarimaOptions.LaggedPredictions + _nnarimaOptions.ExogenousVariables, 
                10, // Hidden layer with 10 neurons
                1   // Output layer with 1 neuron
            },
            ActivationFunctions = new List<IActivationFunction<T>>
            {
                new ReLUActivation<T>(),
            },
            VectorActivationFunctions = new List<IVectorActivationFunction<T>>
            {
                new SoftmaxActivation<T>(),
            }
        };

        return new NeuralNetwork<T>(defaultArchitecture);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Input matrix rows must match output vector length.");
        }

        _y = y;

        InitializeParameters();
        OptimizeParameters(x, _y);
        ComputeResiduals(x, _y);
    }

    private void InitializeParameters()
    {
        int p = _nnarimaOptions.AROrder;
        int q = _nnarimaOptions.MAOrder;

        _arParameters = new Vector<T>(p);
        _maParameters = new Vector<T>(q);

        // Initialize with small random values
        Random rand = new();
        for (int i = 0; i < p; i++) _arParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
        for (int i = 0; i < q; i++) _maParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
    }

    private void OptimizeParameters(Matrix<T> x, Vector<T> y)
    {
        var inputData = new OptimizationInputData<T>
        {
            XTrain = x,
            YTrain = y
        };

        OptimizationResult<T> result = _optimizer.Optimize(inputData);
        UpdateModelParameters(result.BestSolution.Coefficients);
    }

    private void UpdateModelParameters(Vector<T> optimizedParameters)
    {
        int paramIndex = 0;

        // Update AR parameters
        for (int i = 0; i < _arParameters.Length; i++)
        {
            _arParameters[i] = optimizedParameters[paramIndex++];
        }

        // Update MA parameters
        for (int i = 0; i < _maParameters.Length; i++)
        {
            _maParameters[i] = optimizedParameters[paramIndex++];
        }

        // Calculate the length of neural network parameters
        int nnParamsLength = optimizedParameters.Length - paramIndex;

        // Update neural network parameters
        _neuralNetwork.UpdateParameters(optimizedParameters.Slice(paramIndex, nnParamsLength));
    }

    private void ComputeResiduals(Matrix<T> x, Vector<T> y)
    {
        _fitted = Predict(x);
        _residuals = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            _residuals[i] = NumOps.Subtract(y[i], _fitted[i]);
        }
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        int n = input.Rows;
        Vector<T> predictions = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(predictions, input.GetRow(i), i);
        }

        return predictions;
    }

    private T PredictSingle(Vector<T> predictions, Vector<T> inputRow, int index)
    {
        T prediction = NumOps.Zero;

        // Add AR terms
        for (int i = 0; i < _arParameters.Length; i++)
        {
            if (index - i - 1 >= 0)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_arParameters[i], predictions[index - i - 1]));
            }
        }

        // Add MA terms
        for (int i = 0; i < _maParameters.Length; i++)
        {
            if (index - i - 1 >= 0 && _residuals != null)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_maParameters[i], _residuals[index - i - 1]));
            }
        }

        // Add neural network prediction
        Vector<T> nnInput = CreateNeuralNetworkInput(predictions, inputRow, index);
        T nnPrediction = _neuralNetwork.Predict(nnInput)[0];
        prediction = NumOps.Add(prediction, nnPrediction);

        return prediction;
    }

    private Vector<T> CreateNeuralNetworkInput(Vector<T> predictions, Vector<T> inputRow, int index)
    {
        List<T> nnInputList = new List<T>();

        // Add lagged predictions
        for (int i = 1; i <= _nnarimaOptions.LaggedPredictions; i++)
        {
            if (index - i >= 0)
            {
                nnInputList.Add(predictions[index - i]);
            }
            else
            {
                nnInputList.Add(NumOps.Zero);
            }
        }

        // Add exogenous variables
        nnInputList.AddRange(inputRow);

        return new Vector<T>(nnInputList);
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>();

        // Mean Absolute Error (MAE)
        metrics["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions);

        // Mean Squared Error (MSE)
        metrics["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions);

        // Root Mean Squared Error (RMSE)
        metrics["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions);

        // R-squared (R2)
        metrics["R2"] = StatisticsHelper<T>.CalculateR2(yTest, predictions);

        return metrics;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write model parameters
        SerializationHelper<T>.SerializeVector(writer, _arParameters);
        SerializationHelper<T>.SerializeVector(writer, _maParameters);

        // Write neural network parameters
        _neuralNetwork.Serialize(writer);

        // Write options
        writer.Write(_nnarimaOptions.AROrder);
        writer.Write(_nnarimaOptions.MAOrder);
        writer.Write(_nnarimaOptions.LaggedPredictions);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read model parameters
        _arParameters = SerializationHelper<T>.DeserializeVector(reader);
        _maParameters = SerializationHelper<T>.DeserializeVector(reader);

        // Read neural network parameters
        _neuralNetwork.Deserialize(reader);

        // Read options
        _nnarimaOptions.AROrder = reader.ReadInt32();
        _nnarimaOptions.MAOrder = reader.ReadInt32();
        _nnarimaOptions.LaggedPredictions = reader.ReadInt32();
    }
}