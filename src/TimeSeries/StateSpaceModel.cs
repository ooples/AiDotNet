namespace AiDotNet.TimeSeries;

public class StateSpaceModel<T> : TimeSeriesModelBase<T>
{
    private Matrix<T> _transitionMatrix;
    private Matrix<T> _observationMatrix;
    private Matrix<T> _processNoise;
    private Matrix<T> _observationNoise;
    private Vector<T> _initialState;
    private int _stateSize;
    private int _observationSize;
    private double _learningRate;
    private int _maxIterations;
    private double _tolerance;
    private Matrix<T> _previousTransitionMatrix;
    private Matrix<T> _previousObservationMatrix;
    private double _convergenceThreshold = 1e-6;

    public StateSpaceModel(StateSpaceModelOptions<T> options) : base(options)
    {
        _stateSize = options.StateSize;
        _observationSize = options.ObservationSize;
        _learningRate = options.LearningRate;
        _maxIterations = options.MaxIterations;
        _tolerance = options.Tolerance;
        _transitionMatrix = Matrix<T>.CreateIdentity(_stateSize);
        _observationMatrix = Matrix<T>.CreateIdentity(_observationSize);
        _processNoise = Matrix<T>.CreateIdentity(_stateSize);
        _observationNoise = Matrix<T>.CreateIdentity(_observationSize);
        _initialState = new Vector<T>(_stateSize);
        _previousTransitionMatrix = Matrix<T>.CreateIdentity(_stateSize);
        _previousObservationMatrix = Matrix<T>.CreateIdentity(_observationSize);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Combine x and y into a single matrix of observations
        Matrix<T> observations = x.AddColumn(y);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var (filteredStates, predictedStates) = KalmanFilter(observations);
            var (smoothedStates, smoothedCovariances) = KalmanSmoother(filteredStates, predictedStates);

            UpdateParameters(observations, smoothedStates, smoothedCovariances);

            if (CheckConvergence())
            {
                break;
            }
        }
    }

    private (List<Vector<T>>, List<Vector<T>>) KalmanFilter(Matrix<T> observations)
    {
        var filteredStates = new List<Vector<T>>();
        var predictedStates = new List<Vector<T>>();

        var currentState = _initialState;
        var currentCovariance = Matrix<T>.CreateIdentity(_stateSize);

        for (int t = 0; t < observations.Rows; t++)
        {
            // Predict
            var predictedState = _transitionMatrix.Multiply(currentState);
            var predictedCovariance = _transitionMatrix.Multiply(currentCovariance).Multiply(_transitionMatrix.Transpose()).Add(_processNoise);

            predictedStates.Add(predictedState);

            // Update
            var observationVector = observations.GetRow(t);
            var predictedObservation = _observationMatrix.Multiply(predictedState);
            var innovation = VectorHelper.Subtract(observationVector, predictedObservation);
            var innovationCovariance = _observationMatrix.Multiply(predictedCovariance).Multiply(_observationMatrix.Transpose()).Add(_observationNoise);
            var kalmanGain = predictedCovariance.Multiply(_observationMatrix.Transpose()).Multiply(innovationCovariance.Inverse());

            currentState = VectorHelper.Add(predictedState, kalmanGain.Multiply(innovation));
            currentCovariance = predictedCovariance.Subtract(kalmanGain.Multiply(_observationMatrix).Multiply(predictedCovariance));

            filteredStates.Add(currentState);
        }

        return (filteredStates, predictedStates);
    }

    private (List<Vector<T>>, List<Matrix<T>>) KalmanSmoother(List<Vector<T>> filteredStates, List<Vector<T>> predictedStates)
    {
        var smoothedStates = new List<Vector<T>>(filteredStates);
        var smoothedCovariances = new List<Matrix<T>>();

        var currentSmoothedState = filteredStates[filteredStates.Count - 1];
        var currentSmoothedCovariance = Matrix<T>.CreateIdentity(_stateSize);

        for (int t = filteredStates.Count - 2; t >= 0; t--)
        {
            var predictedCovariance = _transitionMatrix.Multiply(currentSmoothedCovariance).Multiply(_transitionMatrix.Transpose()).Add(_processNoise);
        
            var smoothingGain = filteredStates[t].OuterProduct(filteredStates[t])
                .Multiply(_transitionMatrix.Transpose())
                .Multiply(predictedCovariance.Inverse());

            currentSmoothedState = VectorHelper.Add(
                filteredStates[t], 
                smoothingGain.Multiply(VectorHelper.Subtract(currentSmoothedState, predictedStates[t + 1]))
            );

            currentSmoothedCovariance = filteredStates[t].OuterProduct(filteredStates[t]).Add(
                smoothingGain.Multiply(
                    currentSmoothedCovariance.Subtract(predictedCovariance)
                ).Multiply(smoothingGain.Transpose())
            );

            smoothedStates[t] = currentSmoothedState;
            smoothedCovariances.Insert(0, currentSmoothedCovariance);
        }

        return (smoothedStates, smoothedCovariances);
    }

    private void UpdateParameters(Matrix<T> observations, List<Vector<T>> smoothedStates, List<Matrix<T>> smoothedCovariances)
    {
        // Update transition matrix
        Matrix<T> sumXX = Matrix<T>.CreateZeros(_stateSize, _stateSize);
        Matrix<T> sumXY = Matrix<T>.CreateZeros(_stateSize, _stateSize);

        for (int t = 1; t < smoothedStates.Count; t++)
        {
            sumXX = sumXX.Add(smoothedCovariances[t - 1].Add(VectorHelper.OuterProduct(smoothedStates[t - 1], smoothedStates[t - 1])));
            sumXY = sumXY.Add(VectorHelper.OuterProduct(smoothedStates[t], smoothedStates[t - 1]));
        }

        _transitionMatrix = sumXY.Multiply(sumXX.Inverse());

        // Update observation matrix
        Matrix<T> sumYX = Matrix<T>.CreateZeros(_observationSize, _stateSize);
        Matrix<T> sumXX_obs = Matrix<T>.CreateZeros(_stateSize, _stateSize);

        for (int t = 0; t < observations.Rows; t++)
        {
            sumYX = sumYX.Add(VectorHelper.OuterProduct(observations.GetRow(t), smoothedStates[t]));
            sumXX_obs = sumXX_obs.Add(smoothedCovariances[t].Add(VectorHelper.OuterProduct(smoothedStates[t], smoothedStates[t])));
        }

        _observationMatrix = sumYX.Multiply(sumXX_obs.Inverse());

        // Update process noise
        _processNoise = Matrix<T>.CreateZeros(_stateSize, _stateSize);
        for (int t = 1; t < smoothedStates.Count; t++)
        {
            var diff = VectorHelper.Subtract(smoothedStates[t], _transitionMatrix.Multiply(smoothedStates[t - 1]));
            _processNoise = _processNoise.Add(VectorHelper.OuterProduct(diff, diff)).Add(smoothedCovariances[t]);
            _processNoise = _processNoise.Subtract(_transitionMatrix.Multiply(smoothedCovariances[t - 1]).Multiply(_transitionMatrix.Transpose()));
        }
        _processNoise = _processNoise.Divide(NumOps.FromDouble(smoothedStates.Count - 1));

        // Update observation noise
        _observationNoise = Matrix<T>.CreateZeros(_observationSize, _observationSize);
        for (int t = 0; t < observations.Rows; t++)
        {
            var diff = VectorHelper.Subtract(observations.GetRow(t), _observationMatrix.Multiply(smoothedStates[t]));
            _observationNoise = _observationNoise.Add(VectorHelper.OuterProduct(diff, diff));
            _observationNoise = _observationNoise.Add(_observationMatrix.Multiply(smoothedCovariances[t]).Multiply(_observationMatrix.Transpose()));
        }
        _observationNoise = _observationNoise.Divide(NumOps.FromDouble(observations.Rows));
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new Vector<T>(input.Rows);
        Vector<T> currentState = _initialState;

        for (int t = 0; t < input.Rows; t++)
        {
            currentState = _transitionMatrix.Multiply(currentState);
            predictions[t] = _observationMatrix.Multiply(currentState)[0];
        }

        return predictions;
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

    private bool CheckConvergence()
    {
        if (_previousTransitionMatrix == null || _previousObservationMatrix == null)
        {
            _previousTransitionMatrix = _transitionMatrix.Copy();
            _previousObservationMatrix = _observationMatrix.Copy();
            return false;
        }

        double transitionDiff = CalculateMatrixDifference(_transitionMatrix, _previousTransitionMatrix);
        double observationDiff = CalculateMatrixDifference(_observationMatrix, _previousObservationMatrix);

        _previousTransitionMatrix = _transitionMatrix.Copy();
        _previousObservationMatrix = _observationMatrix.Copy();

        return transitionDiff < _convergenceThreshold && observationDiff < _convergenceThreshold;
    }

    private double CalculateMatrixDifference(Matrix<T> matrix1, Matrix<T> matrix2)
    {
        double sum = 0;
        for (int i = 0; i < matrix1.Rows; i++)
        {
            for (int j = 0; j < matrix1.Columns; j++)
            {
                sum += Math.Pow(Convert.ToDouble(NumOps.Subtract(matrix1[i, j], matrix2[i, j])), 2);
            }
        }

        return Math.Sqrt(sum);
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Serialize dimensions
        writer.Write(_stateSize);
        writer.Write(_observationSize);

        // Serialize matrices
        SerializationHelper<T>.SerializeMatrix(writer, _transitionMatrix);
        SerializationHelper<T>.SerializeMatrix(writer, _observationMatrix);
        SerializationHelper<T>.SerializeMatrix(writer, _processNoise);
        SerializationHelper<T>.SerializeMatrix(writer, _observationNoise);

        // Serialize vector
        SerializationHelper<T>.SerializeVector(writer, _initialState);

        // Serialize other parameters
        writer.Write(_learningRate);
        writer.Write(_maxIterations);
        writer.Write(_tolerance);
        writer.Write(_convergenceThreshold);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Deserialize dimensions
        _stateSize = reader.ReadInt32();
        _observationSize = reader.ReadInt32();

        // Deserialize matrices
        _transitionMatrix = SerializationHelper<T>.DeserializeMatrix(reader, _stateSize, _stateSize);
        _observationMatrix = SerializationHelper<T>.DeserializeMatrix(reader, _observationSize, _stateSize);
        _processNoise = SerializationHelper<T>.DeserializeMatrix(reader, _stateSize, _stateSize);
        _observationNoise = SerializationHelper<T>.DeserializeMatrix(reader, _observationSize, _observationSize);

        // Deserialize vector
        _initialState = SerializationHelper<T>.DeserializeVector(reader, _stateSize);

        // Deserialize other parameters
        _learningRate = reader.ReadDouble();
        _maxIterations = reader.ReadInt32();
        _tolerance = reader.ReadDouble();
        _convergenceThreshold = reader.ReadDouble();
    }
}