namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements a State Space Model for time series analysis and forecasting.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// State Space Models represent time series data as a system with hidden states that evolve over time
/// according to probabilistic rules. They are powerful tools for modeling complex dynamic systems
/// and can handle missing data, multiple variables, and non-stationary patterns.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// A State Space Model is like tracking the position of a moving object when you can only see its shadow.
/// The actual position (state) is hidden, but you can observe its effects (the shadow).
/// 
/// For example, if you're tracking the economy, you might not directly observe the "true state" of the economy,
/// but you can see indicators like GDP, unemployment rates, etc. The State Space Model helps infer the hidden
/// state from these observations and predict future values.
/// 
/// The model has two main components:
/// 1. A transition equation that describes how the hidden state evolves over time
/// 2. An observation equation that relates the hidden state to what we actually observe
/// 
/// This implementation uses the Kalman filter and smoother algorithms to estimate the hidden states
/// and learn the model parameters from data.
/// </para>
/// </remarks>
public class StateSpaceModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// The state transition matrix that describes how the hidden state evolves from one time step to the next.
    /// </summary>
    private Matrix<T> _transitionMatrix;

    /// <summary>
    /// The observation matrix that relates the hidden state to the observed measurements.
    /// </summary>
    private Matrix<T> _observationMatrix;

    /// <summary>
    /// The covariance matrix of the process noise, representing uncertainty in the state transition.
    /// </summary>
    private Matrix<T> _processNoise;

    /// <summary>
    /// The covariance matrix of the observation noise, representing measurement uncertainty.
    /// </summary>
    private Matrix<T> _observationNoise;

    /// <summary>
    /// The initial state vector at time t=0.
    /// </summary>
    private Vector<T> _initialState;

    /// <summary>
    /// The dimension of the state vector.
    /// </summary>
    private int _stateSize;

    /// <summary>
    /// The dimension of the observation vector.
    /// </summary>
    private int _observationSize;

    /// <summary>
    /// The learning rate for parameter updates during training.
    /// </summary>
    private double _learningRate;

    /// <summary>
    /// The maximum number of iterations for the EM algorithm during training.
    /// </summary>
    private int _maxIterations;

    /// <summary>
    /// The convergence tolerance for the EM algorithm.
    /// </summary>
    private double _tolerance;

    /// <summary>
    /// The transition matrix from the previous iteration, used to check convergence.
    /// </summary>
    private Matrix<T> _previousTransitionMatrix;

    /// <summary>
    /// The observation matrix from the previous iteration, used to check convergence.
    /// </summary>
    private Matrix<T> _previousObservationMatrix;

    /// <summary>
    /// The threshold for determining when the parameter updates have converged.
    /// </summary>
    private double _convergenceThreshold = 1e-6;

    /// <summary>
    /// Initializes a new instance of the StateSpaceModel class with the specified options.
    /// </summary>
    /// <param name="options">The configuration options for the state space model.</param>
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

    /// <summary>
    /// Trains the state space model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// The training process uses the Expectation-Maximization (EM) algorithm:
    /// 1. E-step: Estimate the hidden states using the Kalman filter and smoother
    /// 2. M-step: Update the model parameters based on the estimated states
    /// 3. Repeat until convergence or maximum iterations reached
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Training this model is like learning to track an object by its shadow. The process involves:
    /// 
    /// 1. First, we make an initial guess about how the object moves and how its shadow appears
    /// 2. Then we use the Kalman filter to estimate where the object was at each time point based on our current understanding
    /// 3. Next, we use the Kalman smoother to refine these estimates by looking at all the data at once
    /// 4. Finally, we update our understanding of how the object moves and creates shadows based on these estimates
    /// 5. We repeat steps 2-4 until our understanding stops improving significantly
    /// 
    /// After training, the model will have learned the patterns in your data and can make predictions.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Applies the Kalman filter to estimate the hidden states based on observations.
    /// </summary>
    /// <param name="observations">The matrix of observations.</param>
    /// <returns>A tuple containing the filtered states and predicted states.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The Kalman filter is like a GPS navigation system that continuously updates its estimate of your position.
    /// It works in two steps:
    /// 
    /// 1. Predict: Based on how we think the object moves, predict where it should be now
    /// 2. Update: Compare our prediction with the actual measurement (the shadow), and adjust our estimate
    /// 
    /// This method processes the observations one by one, updating the state estimate at each time step.
    /// The result is our best guess of where the object was at each moment, based only on observations up to that point.
    /// </para>
    /// </remarks>
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
            var innovation = observationVector.Subtract(predictedObservation);
            var innovationCovariance = _observationMatrix.Multiply(predictedCovariance).Multiply(_observationMatrix.Transpose()).Add(_observationNoise);
            var kalmanGain = predictedCovariance.Multiply(_observationMatrix.Transpose()).Multiply(innovationCovariance.Inverse());

            currentState = predictedState.Add(kalmanGain.Multiply(innovation));
            currentCovariance = predictedCovariance.Subtract(kalmanGain.Multiply(_observationMatrix).Multiply(predictedCovariance));

            filteredStates.Add(currentState);
        }

        return (filteredStates, predictedStates);
    }

    /// <summary>
    /// Applies the Kalman smoother to refine the state estimates using all observations.
    /// </summary>
    /// <param name="filteredStates">The states estimated by the Kalman filter.</param>
    /// <param name="predictedStates">The predicted states from the Kalman filter.</param>
    /// <returns>A tuple containing the smoothed states and their covariances.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// While the Kalman filter only uses past observations to estimate the current state,
    /// the Kalman smoother uses all observations (past and future) to improve these estimates.
    /// 
    /// It's like watching a video of someone walking behind a fence with gaps. The Kalman filter
    /// is like watching in real-time and guessing where the person is when they're behind the fence.
    /// The smoother is like rewatching the video and using the knowledge of where they appeared later
    /// to better guess where they were when hidden.
    /// 
    /// This method works backward in time, refining each state estimate using information from future observations.
    /// </para>
    /// </remarks>
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

            currentSmoothedState = filteredStates[t].Add(smoothingGain.Multiply(currentSmoothedState.Subtract(predictedStates[t + 1]))
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

    /// <summary>
    /// Updates the model parameters based on the estimated states.
    /// </summary>
    /// <param name="observations">The matrix of observations.</param>
    /// <param name="smoothedStates">The states estimated by the Kalman smoother.</param>
    /// <param name="smoothedCovariances">The covariances of the smoothed states.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is the "learning" part of the model, where it updates its understanding of:
    /// 
    /// 1. How the hidden state evolves over time (transition matrix)
    /// 2. How the hidden state relates to what we observe (observation matrix)
    /// 3. How much random variation there is in the state evolution (process noise)
    /// 4. How much measurement error there is in our observations (observation noise)
    /// 
    /// It's like a scientist refining their theory based on experimental data. The model
    /// adjusts its parameters to better explain the patterns seen in the data.
    /// </para>
    /// </remarks>
    private void UpdateParameters(Matrix<T> observations, List<Vector<T>> smoothedStates, List<Matrix<T>> smoothedCovariances)
    {
        // Update transition matrix
        Matrix<T> sumXX = Matrix<T>.CreateZeros(_stateSize, _stateSize);
        Matrix<T> sumXY = Matrix<T>.CreateZeros(_stateSize, _stateSize);

        for (int t = 1; t < smoothedStates.Count; t++)
        {
            sumXX = sumXX.Add(smoothedCovariances[t - 1].Add(smoothedStates[t - 1].OuterProduct(smoothedStates[t - 1])));
            sumXY = sumXY.Add(smoothedStates[t].OuterProduct(smoothedStates[t - 1]));
        }

        _transitionMatrix = sumXY.Multiply(sumXX.Inverse());

        // Update observation matrix
        Matrix<T> sumYX = Matrix<T>.CreateZeros(_observationSize, _stateSize);
        Matrix<T> sumXX_obs = Matrix<T>.CreateZeros(_stateSize, _stateSize);

        for (int t = 0; t < observations.Rows; t++)
        {
            sumYX = sumYX.Add(observations.GetRow(t).OuterProduct(smoothedStates[t]));
            sumXX_obs = sumXX_obs.Add(smoothedCovariances[t].Add(smoothedStates[t].OuterProduct(smoothedStates[t])));
        }

        _observationMatrix = sumYX.Multiply(sumXX_obs.Inverse());

        // Update process noise
        _processNoise = Matrix<T>.CreateZeros(_stateSize, _stateSize);
        for (int t = 1; t < smoothedStates.Count; t++)
        {
            var diff = smoothedStates[t].Subtract(_transitionMatrix.Multiply(smoothedStates[t - 1]));
            _processNoise = _processNoise.Add(diff.OuterProduct(diff)).Add(smoothedCovariances[t]);
            _processNoise = _processNoise.Subtract(_transitionMatrix.Multiply(smoothedCovariances[t - 1]).Multiply(_transitionMatrix.Transpose()));
        }

        _processNoise = _processNoise.Divide(NumOps.FromDouble(smoothedStates.Count - 1));

        // Update observation noise
        _observationNoise = Matrix<T>.CreateZeros(_observationSize, _observationSize);
        for (int t = 0; t < observations.Rows; t++)
        {
            var diff = observations.GetRow(t).Subtract(_observationMatrix.Multiply(smoothedStates[t]));
                        _observationNoise = _observationNoise.Add(diff.OuterProduct(diff));
            _observationNoise = _observationNoise.Add(_observationMatrix.Multiply(smoothedCovariances[t]).Multiply(_observationMatrix.Transpose()));
        }

        _observationNoise = _observationNoise.Divide(NumOps.FromDouble(observations.Rows));
    }

    /// <summary>
    /// Generates predictions using the trained state space model.
    /// </summary>
    /// <param name="input">The input features matrix for which predictions are to be made.</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method predicts future values by:
    /// 
    /// 1. Starting with our initial state estimate
    /// 2. Using the transition matrix to predict how the state will evolve at each time step
    /// 3. Using the observation matrix to convert these state predictions into observable predictions
    /// 
    /// It's like predicting where the shadow will appear based on our understanding of how the object moves.
    /// We don't directly predict the shadow; we predict the object's position and then calculate where
    /// its shadow would be.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Evaluates the performance of the trained model on test data.
    /// </summary>
    /// <param name="xTest">The input features matrix for testing.</param>
    /// <param name="yTest">The actual target values for testing.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method tests how well our model performs by comparing its predictions to actual values.
    /// It calculates several error metrics:
    /// 
    /// - MSE (Mean Squared Error): The average of the squared differences between predictions and actual values
    /// - RMSE (Root Mean Squared Error): The square root of MSE, which gives an error in the same units as the data
    /// - MAE (Mean Absolute Error): The average of the absolute differences between predictions and actual values
    /// - MAPE (Mean Absolute Percentage Error): The average percentage difference between predictions and actual values
    /// 
    /// Lower values for these metrics indicate better model performance.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Checks if the model parameters have converged during training.
    /// </summary>
    /// <returns>True if the parameters have converged; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method checks if the model has finished learning by comparing the current parameter values
    /// to those from the previous iteration. If they're very similar (the difference is below a threshold),
    /// we consider the model to have "converged" and stop the training process.
    /// 
    /// It's like stopping practice when you're no longer improving significantly - there's no need to
    /// continue if you're not getting any better.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Calculates the Frobenius norm of the difference between two matrices.
    /// </summary>
    /// <param name="matrix1">The first matrix.</param>
    /// <param name="matrix2">The second matrix.</param>
    /// <returns>The Frobenius norm of the difference between the matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method measures how different two matrices are by calculating the square root of the sum
    /// of the squared differences between corresponding elements. It's like measuring the straight-line
    /// distance between two points in a multi-dimensional space.
    /// 
    /// A smaller value means the matrices are more similar, which indicates that the model parameters
    /// aren't changing much between iterations.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Serializes the model's core parameters to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization is the process of converting the model's state into a format that can be saved to disk.
    /// This allows you to save a trained model and load it later without having to retrain it.
    /// 
    /// This method saves:
    /// - The dimensions of the state and observation vectors
    /// - The transition and observation matrices
    /// - The process and observation noise matrices
    /// - The initial state vector
    /// - Other parameters like learning rate and convergence settings
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Deserializes the model's core parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the process of loading a previously saved model from disk.
    /// This method reads the model's parameters from a file and reconstructs the model
    /// exactly as it was when it was saved.
    /// 
    /// This allows you to train a model once and then use it many times without retraining.
    /// It's like writing down a recipe so you can make the same dish again later without
    /// having to figure out the ingredients and proportions from scratch.
    /// </para>
    /// </remarks>
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