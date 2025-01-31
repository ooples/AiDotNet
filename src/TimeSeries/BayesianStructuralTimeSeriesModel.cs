namespace AiDotNet.TimeSeries;

public class BayesianStructuralTimeSeriesModel<T> : TimeSeriesModelBase<T>
{
    private readonly BayesianStructuralTimeSeriesOptions<T> _bayesianOptions;
    private T _level;
    private T _trend;
    private List<Vector<T>> _seasonalComponents;
    private Matrix<T> _stateCovariance;
    private T _observationVariance;
    private Vector<T>? _regression;

    public BayesianStructuralTimeSeriesModel(BayesianStructuralTimeSeriesOptions<T>? options = null) 
    : base(options ?? new BayesianStructuralTimeSeriesOptions<T>())
    {
        _bayesianOptions = options ?? new BayesianStructuralTimeSeriesOptions<T>();

        // Initialize model components
        _level = NumOps.FromDouble(_bayesianOptions.InitialLevelValue);
        _trend = _bayesianOptions.IncludeTrend ? NumOps.FromDouble(_bayesianOptions.InitialTrendValue) : NumOps.Zero;
        _seasonalComponents = [];
        foreach (int period in _bayesianOptions.SeasonalPeriods)
        {
            _seasonalComponents.Add(new Vector<T>(period));
        }
        _observationVariance = NumOps.FromDouble(_bayesianOptions.InitialObservationVariance);
    
        int stateSize = GetStateSize();
        _stateCovariance = new Matrix<T>(stateSize, stateSize);

        // Initialize regression component if included
        if (_bayesianOptions.IncludeRegression)
        {
            // We'll initialize the regression vector later when we have the actual input data
            _regression = null;
        }
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = y.Length;
        Matrix<T> states = new Matrix<T>(n, GetStateSize());

        // Initialize or update regression component if included
        if (_bayesianOptions.IncludeRegression)
        {
            if (_regression == null || _regression.Length != x.Columns)
            {
                _regression = new Vector<T>(x.Columns);
            }
    
            // Initialize regression coefficients using Ordinary Least Squares (OLS)
            InitializeRegressionCoefficients(x, y);
        }
        
        // Kalman filter
        for (int t = 0; t < n; t++)
        {
            // Prediction step
            Vector<T> predictedState = PredictState(x.GetRow(t));
            Matrix<T> predictedCovariance = PredictCovariance();

            // Update step
            T innovation = CalculateInnovation(y[t], predictedState);
            var kalmanGain = CalculateKalmanGain(predictedCovariance);
            UpdateState(predictedState, kalmanGain, innovation);
            UpdateCovariance(predictedCovariance, kalmanGain);

            // Store state
            states.SetRow(t, GetCurrentState());
        }

        // Backward smoothing (optional)
        if (_bayesianOptions.PerformBackwardSmoothing)
        {
            PerformBackwardSmoothing(states);
        }

        // Parameter estimation (e.g., EM algorithm or variational inference)
        EstimateParameters(x, y, states);
    }

    private void InitializeRegressionCoefficients(Matrix<T> x, Vector<T> y)
    {
        if (_regression == null)
        {
            throw new InvalidOperationException("Regression vector is not initialized.");
        }

        // Perform OLS to initialize regression coefficients
        Matrix<T> xTranspose = x.Transpose();
        Matrix<T> xTx = xTranspose.Multiply(x);
        Vector<T> xTy = xTranspose.Multiply(y);

        try
        {
            // Solve the normal equations: (X^T * X) * beta = X^T * y
            Vector<T> olsCoefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _bayesianOptions.RegressionDecompositionType);

            // Apply shrinkage to prevent overfitting
            T shrinkageFactor = NumOps.FromDouble(0.95); // You might want to make this configurable
            for (int i = 0; i < _regression.Length; i++)
            {
                _regression[i] = NumOps.Multiply(olsCoefficients[i], shrinkageFactor);
            }
        }
        catch (Exception)
        {
            // If matrix is singular or near-singular, use ridge regression
            T ridgeParameter = NumOps.FromDouble(_bayesianOptions.RidgeParameter);
            Vector<T> ridgeDiagonal = new Vector<T>(x.Columns);
            for (int i = 0; i < x.Columns; i++)
            {
                ridgeDiagonal[i] = ridgeParameter;
            }
            Matrix<T> ridgeMatrix = Matrix<T>.CreateDiagonal(ridgeDiagonal);
            Matrix<T> regularizedXTX = xTx.Add(ridgeMatrix);
            Vector<T> ridgeCoefficients = MatrixSolutionHelper.SolveLinearSystem(regularizedXTX, xTy, _bayesianOptions.RegressionDecompositionType);

            // Apply shrinkage to prevent overfitting
            T shrinkageFactor = NumOps.FromDouble(0.95); // You might want to make this configurable
            for (int i = 0; i < _regression.Length; i++)
            {
                _regression[i] = NumOps.Multiply(ridgeCoefficients[i], shrinkageFactor);
            }
        }
    }

    private Vector<T> PredictState(Vector<T> x)
    {
        int stateSize = GetStateSize();
        Vector<T> predictedState = new Vector<T>(stateSize);
        int index = 0;

        // Level
        predictedState[index] = _level;
        index++;

        // Trend
        if (_bayesianOptions.IncludeTrend)
        {
            predictedState[index] = NumOps.Add(_level, _trend);
            index++;
        }

        // Seasonal components
        foreach (var seasonalComponent in _seasonalComponents)
        {
            for (int i = 0; i < seasonalComponent.Length; i++)
            {
                predictedState[index + i] = seasonalComponent[i];
            }
            index += seasonalComponent.Length;
        }

        // Regression component
        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                predictedState[index + i] = NumOps.Multiply(x[i], _regression[i]);
            }
        }

        return predictedState;
    }

    private Matrix<T> PredictCovariance()
    {
        int stateSize = GetStateSize();
        Matrix<T> transitionMatrix = CreateTransitionMatrix();

        return transitionMatrix.Multiply(_stateCovariance).Multiply(transitionMatrix.Transpose()) + CreateProcessNoiseMatrix();
    }

    private T CalculateInnovation(T observation, Vector<T> predictedState)
    {
        return NumOps.Subtract(observation, CalculatePrediction(predictedState));
    }

    private Vector<T> CalculateKalmanGain(Matrix<T> predictedCovariance)
    {
        Vector<T> observationVector = CreateObservationVector();
        T denominator = NumOps.Add(
            observationVector.DotProduct(predictedCovariance.Multiply(observationVector)),
            _observationVariance
        );

        return predictedCovariance.Multiply(observationVector).Divide(denominator);
    }

    private void UpdateState(Vector<T> predictedState, Vector<T> kalmanGain, T innovation)
    {
        if (predictedState.Length != kalmanGain.Length)
        {
            throw new ArgumentException("Predicted state and Kalman gain must have the same length");
        }

        // Calculate the update vector
        Vector<T> update = kalmanGain.Multiply(innovation);

        // Apply the update to the entire state vector
        Vector<T> updatedState = predictedState.Add(update);

        // Update individual components
        int index = 0;

        // Update level
        _level = updatedState[index++];

        // Update trend if included
        if (_bayesianOptions.IncludeTrend)
        {
            _trend = updatedState[index++];
        }

        // Update seasonal components
        for (int i = 0; i < _seasonalComponents.Count; i++)
        {
            for (int j = 0; j < _seasonalComponents[i].Length; j++)
            {
                _seasonalComponents[i][j] = updatedState[index++];
            }
        }

        // Update regression components if included
        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                _regression[i] = updatedState[index++];
            }
        }

        // Sanity check
        if (index != updatedState.Length)
        {
            throw new InvalidOperationException("State update mismatch: not all components were updated");
        }
    }

    private void UpdateCovariance(Matrix<T> predictedCovariance, Vector<T> kalmanGain)
    {
        Vector<T> observationVector = CreateObservationVector();
        Matrix<T> identity = Matrix<T>.CreateIdentity(predictedCovariance.Rows);
        Matrix<T> kalmanGainMatrix = kalmanGain.ToColumnMatrix();
        _stateCovariance = identity.Subtract(kalmanGainMatrix.Multiply(observationVector.ToRowMatrix())).Multiply(predictedCovariance);
    }

    private Vector<T> GetCurrentState()
    {
        int stateSize = GetStateSize();
        Vector<T> currentState = new Vector<T>(stateSize);
        int index = 0;

        currentState[index++] = _level;
        if (_bayesianOptions.IncludeTrend) currentState[index++] = _trend;

        foreach (var seasonalComponent in _seasonalComponents)
        {
            for (int i = 0; i < seasonalComponent.Length; i++)
            {
                currentState[index++] = seasonalComponent[i];
            }
        }

        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                currentState[index++] = _regression[i];
            }
        }

        return currentState;
    }

    private void PerformBackwardSmoothing(Matrix<T> states)
    {
        int n = states.Rows;
        int stateSize = states.Columns;
        Matrix<T> smoothedStates = new Matrix<T>(n, stateSize);
        Matrix<T> transitionMatrix = CreateTransitionMatrix();

        // Initialize with the last state
        smoothedStates.SetRow(n - 1, states.GetRow(n - 1));

        for (int t = n - 2; t >= 0; t--)
        {
            Vector<T> currentState = states.GetRow(t);
            Vector<T> nextState = states.GetRow(t + 1);
            Vector<T> smoothedNextState = smoothedStates.GetRow(t + 1);

            Matrix<T> predictedCovariance = PredictCovariance();
            Matrix<T> smoothingGain = _stateCovariance
                .Multiply(transitionMatrix.Transpose())
                .Multiply(predictedCovariance.Inverse());

            Vector<T> stateDiff = smoothedNextState.Subtract(nextState);
            Vector<T> smoothedState = currentState.Add(smoothingGain.Multiply(stateDiff));

            smoothedStates.SetRow(t, smoothedState);
        }

        // Update model components with smoothed states
        UpdateModelComponentsFromSmoothedStates(smoothedStates);
    }

    private void EstimateParameters(Matrix<T> x, Vector<T> y, Matrix<T> states)
    {
        T previousLogLikelihood = NumOps.MinValue;
    
        for (int iteration = 0; iteration < _bayesianOptions.MaxIterations; iteration++)
        {
            // E-step: Run Kalman filter and smoother
            Matrix<T> smoothedStates = RunKalmanFilterAndSmoother(x, y);

            // M-step: Update model parameters
            T currentLogLikelihood = UpdateModelParameters(x, y, smoothedStates);

            // Check for convergence
            if (CheckConvergence(previousLogLikelihood, currentLogLikelihood))
            {
                break;
            }
        
            previousLogLikelihood = currentLogLikelihood;
        }
    }

    private Matrix<T> RunKalmanFilterAndSmoother(Matrix<T> x, Vector<T> y)
    {
        int n = y.Length;
        int stateSize = GetStateSize();
        Matrix<T> filteredStates = new Matrix<T>(n, stateSize);
    
        // Forward pass (Kalman filter)
        for (int t = 0; t < n; t++)
        {
            Vector<T> predictedState = PredictState(x.GetRow(t));
            Matrix<T> predictedCovariance = PredictCovariance();
            T innovation = CalculateInnovation(y[t], predictedState);
            var kalmanGain = CalculateKalmanGain(predictedCovariance);
            UpdateState(predictedState, kalmanGain, innovation);
            UpdateCovariance(predictedCovariance, kalmanGain);
            filteredStates.SetRow(t, GetCurrentState());
        }
    
        // Backward pass (smoother)
        if (_bayesianOptions.PerformBackwardSmoothing)
        {
            PerformBackwardSmoothing(filteredStates);
            return filteredStates;
        }
    
        return filteredStates;
    }

    private T UpdateModelParameters(Matrix<T> x, Vector<T> y, Matrix<T> smoothedStates)
    {
        T logLikelihood = NumOps.Zero;
        int n = y.Length;
        int stateSize = GetStateSize();

        // Update level and trend variances
        T levelVariance = NumOps.Zero;
        T trendVariance = NumOps.Zero;
        for (int t = 1; t < n; t++)
        {
            Vector<T> prevState = smoothedStates.GetRow(t - 1);
            Vector<T> currState = smoothedStates.GetRow(t);
            levelVariance = NumOps.Add(levelVariance, NumOps.Square(NumOps.Subtract(currState[0], prevState[0])));
            if (_bayesianOptions.IncludeTrend)
            {
                trendVariance = NumOps.Add(trendVariance, NumOps.Square(NumOps.Subtract(currState[1], prevState[1])));
            }
        }
        levelVariance = NumOps.Divide(levelVariance, NumOps.FromDouble(n - 1));
        _stateCovariance[0, 0] = levelVariance;
        if (_bayesianOptions.IncludeTrend)
        {
            trendVariance = NumOps.Divide(trendVariance, NumOps.FromDouble(n - 1));
            _stateCovariance[1, 1] = trendVariance;
        }

        // Update seasonal variances
        int seasonalIndex = _bayesianOptions.IncludeTrend ? 2 : 1;
        foreach (var seasonalComponent in _seasonalComponents)
        {
            T seasonalVariance = NumOps.Zero;
            for (int t = 1; t < n; t++)
            {
                Vector<T> prevState = smoothedStates.GetRow(t - 1);
                Vector<T> currState = smoothedStates.GetRow(t);
                for (int i = 0; i < seasonalComponent.Length; i++)
                {
                    seasonalVariance = NumOps.Add(seasonalVariance, NumOps.Square(NumOps.Subtract(currState[seasonalIndex + i], prevState[seasonalIndex + i])));
                }
            }
            seasonalVariance = NumOps.Divide(seasonalVariance, NumOps.FromDouble((n - 1) * seasonalComponent.Length));
            for (int i = 0; i < seasonalComponent.Length; i++)
            {
                _stateCovariance[seasonalIndex + i, seasonalIndex + i] = seasonalVariance;
            }
            seasonalIndex += seasonalComponent.Length;
        }

        // Update observation variance
        T totalVariance = NumOps.Zero;
        for (int t = 0; t < n; t++)
        {
            T prediction = CalculatePrediction(smoothedStates.GetRow(t));
            T error = NumOps.Subtract(y[t], prediction);
            totalVariance = NumOps.Add(totalVariance, NumOps.Square(error));
            logLikelihood = NumOps.Add(logLikelihood, NumOps.Log(NumOps.Abs(error)));
        }
        _observationVariance = NumOps.Divide(totalVariance, NumOps.FromDouble(n));

        return logLikelihood;
    }

    private bool CheckConvergence(T previousLogLikelihood, T currentLogLikelihood)
    {
        T difference = NumOps.Abs(NumOps.Subtract(currentLogLikelihood, previousLogLikelihood));
        T threshold = NumOps.FromDouble(_bayesianOptions.ConvergenceTolerance);

        return NumOps.LessThanOrEquals(difference, threshold);
    }

    private Matrix<T> CreateTransitionMatrix()
    {
        int stateSize = GetStateSize();
        Matrix<T> transitionMatrix = Matrix<T>.CreateIdentity(stateSize);

        // Add trend component if included
        if (_bayesianOptions.IncludeTrend)
        {
            transitionMatrix[0, 1] = NumOps.FromDouble(1.0);
        }

        // Add seasonal components
        int index = _bayesianOptions.IncludeTrend ? 2 : 1;
        foreach (var seasonalComponent in _seasonalComponents)
        {
            for (int i = 0; i < seasonalComponent.Length - 1; i++)
            {
                transitionMatrix[index + i, index + i + 1] = NumOps.FromDouble(1.0);
            }
            transitionMatrix[index + seasonalComponent.Length - 1, index] = NumOps.FromDouble(-1.0);
            index += seasonalComponent.Length;
        }

        return transitionMatrix;
    }

    private Matrix<T> CreateProcessNoiseMatrix()
    {
        int stateSize = GetStateSize();
        Matrix<T> processNoiseMatrix = new Matrix<T>(stateSize, stateSize);

        // Set variances for level and trend
        processNoiseMatrix[0, 0] = NumOps.FromDouble(_bayesianOptions.LevelSmoothingPrior);
        if (_bayesianOptions.IncludeTrend)
        {
            processNoiseMatrix[1, 1] = NumOps.FromDouble(_bayesianOptions.TrendSmoothingPrior);
        }

        // Set variances for seasonal components
        int index = _bayesianOptions.IncludeTrend ? 2 : 1;
        foreach (var seasonalComponent in _seasonalComponents)
        {
            for (int i = 0; i < seasonalComponent.Length; i++)
            {
                processNoiseMatrix[index + i, index + i] = NumOps.FromDouble(_bayesianOptions.SeasonalSmoothingPrior);
            }
            index += seasonalComponent.Length;
        }

        return processNoiseMatrix;
    }

    private Vector<T> CreateObservationVector()
    {
        int stateSize = GetStateSize();
        Vector<T> observationVector = new Vector<T>(stateSize);
        observationVector[0] = NumOps.FromDouble(1.0); // Level component

        int index = 1;
        if (_bayesianOptions.IncludeTrend)
        {
            observationVector[index] = NumOps.FromDouble(1.0); // Trend component
            index++;
        }

        // Seasonal components
        foreach (var seasonalComponent in _seasonalComponents)
        {
            observationVector[index] = NumOps.FromDouble(1.0);
            index += seasonalComponent.Length;
        }

        // Regression components
        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                observationVector[index + i] = NumOps.FromDouble(1.0);
            }
        }

        return observationVector;
    }

    private void UpdateModelComponentsFromSmoothedStates(Matrix<T> smoothedStates)
    {
        Vector<T> lastState = smoothedStates.GetRow(smoothedStates.Rows - 1);
        int index = 0;

        _level = lastState[index++];
        if (_bayesianOptions.IncludeTrend) _trend = lastState[index++];

        for (int i = 0; i < _seasonalComponents.Count; i++)
        {
            for (int j = 0; j < _seasonalComponents[i].Length; j++)
            {
                _seasonalComponents[i][j] = lastState[index++];
            }
        }

        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                _regression[i] = lastState[index++];
            }
        }
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        int horizon = input.Rows;
        Vector<T> predictions = new Vector<T>(horizon);

        for (int t = 0; t < horizon; t++)
        {
            Vector<T> state = PredictState(input.GetRow(t));
            predictions[t] = CalculatePrediction(state);
        }

        return predictions;
    }

    private T CalculatePrediction(Vector<T> state)
    {
        // In a Bayesian Structural Time Series model, the prediction is typically
        // the sum of the level, trend, and seasonal components
        T prediction = state[0]; // Level is always the first component

        int index = 1;
        if (_bayesianOptions.IncludeTrend)
        {
            prediction = NumOps.Add(prediction, state[index]);
            index++;
        }

        // Add seasonal components
        foreach (var seasonalComponent in _seasonalComponents)
        {
            prediction = NumOps.Add(prediction, state[index]);
            index += seasonalComponent.Length;
        }

        // Add regression component if present
        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(state[index + i], _regression[i]));
            }
        }

        return prediction;
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = [];

        // Calculate Mean Squared Error (MSE)
        T mse = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions);
        metrics["MSE"] = mse;

        // Calculate Root Mean Squared Error (RMSE)
        T rmse = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions);
        metrics["RMSE"] = rmse;

        // Calculate Mean Absolute Error (MAE)
        T mae = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions);
        metrics["MAE"] = mae;

        // Calculate Mean Absolute Percentage Error (MAPE)
        T mape = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(yTest, predictions);
        metrics["MAPE"] = mape;

        return metrics;
    }

    private int GetStateSize()
    {
        int size = 1; // Always include level
        if (_bayesianOptions.IncludeTrend) size++;
        size += _seasonalComponents.Sum(s => s.Length);
        if (_bayesianOptions.IncludeRegression && _regression != null) size += _regression.Length;

        return size;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Serialize model parameters
        writer.Write(Convert.ToDouble(_level));

        if (_bayesianOptions.IncludeTrend)
        {
            writer.Write(Convert.ToDouble(_trend));
        }

        writer.Write(_seasonalComponents.Count);
        foreach (var component in _seasonalComponents)
        {
            writer.Write(component.Length);
            foreach (var val in component) writer.Write(Convert.ToDouble(val));
        }

        writer.Write(_stateCovariance.Rows);
        writer.Write(_stateCovariance.Columns);
        for (int i = 0; i < _stateCovariance.Rows; i++)
            for (int j = 0; j < _stateCovariance.Columns; j++)
                writer.Write(Convert.ToDouble(_stateCovariance[i, j]));

        writer.Write(Convert.ToDouble(_observationVariance));

        // Serialize options
        writer.Write(_bayesianOptions.IncludeTrend);
        writer.Write(_bayesianOptions.IncludeRegression);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Deserialize model parameters
        _level = NumOps.FromDouble(reader.ReadDouble());

        if (_bayesianOptions.IncludeTrend)
        {
            _trend = NumOps.FromDouble(reader.ReadDouble());
        }

        int seasonalComponentsCount = reader.ReadInt32();
        _seasonalComponents = new List<Vector<T>>();
        for (int i = 0; i < seasonalComponentsCount; i++)
        {
            int componentLength = reader.ReadInt32();
            Vector<T> component = new Vector<T>(componentLength);
            for (int j = 0; j < componentLength; j++) component[j] = NumOps.FromDouble(reader.ReadDouble());
            _seasonalComponents.Add(component);
        }

        int covarianceRows = reader.ReadInt32();
        int covarianceColumns = reader.ReadInt32();
        _stateCovariance = new Matrix<T>(covarianceRows, covarianceColumns);
        for (int i = 0; i < covarianceRows; i++)
            for (int j = 0; j < covarianceColumns; j++)
                _stateCovariance[i, j] = NumOps.FromDouble(reader.ReadDouble());

        _observationVariance = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize options
        _bayesianOptions.IncludeTrend = reader.ReadBoolean();
        _bayesianOptions.IncludeRegression = reader.ReadBoolean();
    }
}