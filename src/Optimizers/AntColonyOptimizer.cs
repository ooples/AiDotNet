namespace AiDotNet.Optimizers;

public class AntColonyOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private readonly AntColonyOptimizationOptions _antColonyOptions;

    public AntColonyOptimizer(AntColonyOptimizationOptions? options = null)
        : base(options)
    {
        _random = new Random();
        _antColonyOptions = options ?? new AntColonyOptimizationOptions();
    }

    public override OptimizationResult<T> Optimize(
        Matrix<T> XTrain,
        Vector<T> yTrain,
        Matrix<T> XVal,
        Vector<T> yVal,
        Matrix<T> XTest,
        Vector<T> yTest,
        PredictionModelOptions modelOptions,
        IRegression<T> regressionMethod,
        IRegularization<T> regularization,
        INormalizer<T> normalizer,
        NormalizationInfo<T> normInfo,
        IFitnessCalculator<T> fitnessCalculator,
        IFitDetector<T> fitDetector)
    {
        int dimensions = XTrain.Columns;
        var pheromones = InitializePheromones(dimensions);
        var bestSolution = Vector<T>.Empty();
        T bestFitness = _numOps.MaxValue;
        T bestIntercept = _numOps.Zero;
        FitDetectorResult<T> bestFitDetectionResult = new();
        Vector<T> bestTrainingPredictions = new(yTrain.Length, _numOps);
        Vector<T> bestValidationPredictions = new(yVal.Length, _numOps);
        Vector<T> bestTestPredictions = new(yTest.Length, _numOps);
        ErrorStats<T> bestTrainingErrorStats = ErrorStats<T>.Empty();
        ErrorStats<T> bestValidationErrorStats = ErrorStats<T>.Empty();
        ErrorStats<T> bestTestErrorStats = ErrorStats<T>.Empty();
        BasicStats<T> bestTrainingActualBasicStats = BasicStats<T>.Empty();
        BasicStats<T> bestTrainingPredictedBasicStats = BasicStats<T>.Empty();
        BasicStats<T> bestValidationActualBasicStats = BasicStats<T>.Empty();
        BasicStats<T> bestValidationPredictedBasicStats = BasicStats<T>.Empty();
        BasicStats<T> bestTestActualBasicStats = BasicStats<T>.Empty();
        BasicStats<T> bestTestPredictedBasicStats = BasicStats<T>.Empty();
        PredictionStats<T> bestTrainingPredictionStats = PredictionStats<T>.Empty();
        PredictionStats<T> bestValidationPredictionStats = PredictionStats<T>.Empty();
        PredictionStats<T> bestTestPredictionStats = PredictionStats<T>.Empty();
        List<Vector<T>> bestSelectedFeatures = [];
        Matrix<T> bestTestFeatures = new(XTest.Rows, XTest.Columns, _numOps);
        Matrix<T> bestTrainingFeatures = new(XTrain.Rows, XTrain.Columns, _numOps);
        Matrix<T> bestValidationFeatures = new(XVal.Rows, XVal.Columns, _numOps);
        List<int> bestSelectedFeatureIndices = [];

        var fitnessHistory = new List<T>();
        var iterationHistory = new List<OptimizationIterationInfo<T>>();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var solutions = new List<Vector<T>>();
            var fitnesses = new List<T>();

            for (int ant = 0; ant < _antColonyOptions.AntCount; ant++)
            {
                var solution = ConstructSolution(pheromones, XTrain);
                solutions.Add(solution);

                var selectedFeatures = OptimizerHelper.GetSelectedFeatures(solution);
                var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
                var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
                var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

                var (currentFitnessScore, fitDetectionResult, 
                     trainingPredictions, validationPredictions, testPredictions,
                     trainingErrorStats, validationErrorStats, testErrorStats,
                     trainingActualBasicStats, trainingPredictedBasicStats,
                     validationActualBasicStats, validationPredictedBasicStats,
                     testActualBasicStats, testPredictedBasicStats,
                     trainingPredictionStats, validationPredictionStats, testPredictionStats) = 
                    EvaluateSolution(
                        XTrainSubset, XValSubset, XTestSubset,
                        yTrain, yVal, yTest,
                        regressionMethod, normalizer, normInfo,
                        fitnessCalculator, fitDetector, selectedFeatures.Count);

                fitnesses.Add(currentFitnessScore);

                UpdateBestSolution(
                    currentFitnessScore,
                    solution,
                    regressionMethod.Intercept,
                    fitDetectionResult,
                    trainingPredictions,
                    validationPredictions,
                    testPredictions,
                    trainingErrorStats,
                    validationErrorStats,
                    testErrorStats,
                    trainingActualBasicStats,
                    trainingPredictedBasicStats,
                    validationActualBasicStats,
                    validationPredictedBasicStats,
                    testActualBasicStats,
                    testPredictedBasicStats,
                    trainingPredictionStats,
                    validationPredictionStats,
                    testPredictionStats,
                    selectedFeatures,
                    XTrain,
                    XTestSubset,
                    XTrainSubset,
                    XValSubset,
                    fitnessCalculator,
                    ref bestFitness,
                    ref bestSolution,
                    ref bestIntercept,
                    ref bestFitDetectionResult,
                    ref bestTrainingPredictions,
                    ref bestValidationPredictions,
                    ref bestTestPredictions,
                    ref bestTrainingErrorStats,
                    ref bestValidationErrorStats,
                    ref bestTestErrorStats,
                    ref bestTrainingActualBasicStats,
                    ref bestTrainingPredictedBasicStats,
                    ref bestValidationActualBasicStats,
                    ref bestValidationPredictedBasicStats,
                    ref bestTestActualBasicStats,
                    ref bestTestPredictedBasicStats,
                    ref bestTrainingPredictionStats,
                    ref bestValidationPredictionStats,
                    ref bestTestPredictionStats,
                    ref bestSelectedFeatures,
                    ref bestTestFeatures,
                    ref bestTrainingFeatures,
                    ref bestValidationFeatures);
            }

            UpdatePheromones(pheromones, solutions, fitnesses);
            if (UpdateIterationHistoryAndCheckEarlyStopping(fitnessHistory, iterationHistory, iteration, bestFitness, bestFitDetectionResult, fitnessCalculator))
            {
                break; // Early stopping criteria met, exit the loop
            }
        }

        return OptimizerHelper.CreateOptimizationResult(
        bestSolution,
        bestIntercept,
        bestFitness,
        fitnessHistory,
        bestSelectedFeatures,
        new OptimizationResult<T>.DatasetResult
        {
            X = bestTrainingFeatures,
            Y = yTrain,
            Predictions = bestTrainingPredictions,
            ErrorStats = bestTrainingErrorStats,
            ActualBasicStats = bestTrainingActualBasicStats,
            PredictedBasicStats = bestTrainingPredictedBasicStats,
            PredictionStats = bestTrainingPredictionStats
        },
        new OptimizationResult<T>.DatasetResult
        {
            X = bestValidationFeatures,
            Y = yVal,
            Predictions = bestValidationPredictions,
            ErrorStats = bestValidationErrorStats,
            ActualBasicStats = bestValidationActualBasicStats,
            PredictedBasicStats = bestValidationPredictedBasicStats,
            PredictionStats = bestValidationPredictionStats
        },
        new OptimizationResult<T>.DatasetResult
        {
            X = bestTestFeatures,
            Y = yTest,
            Predictions = bestTestPredictions,
            ErrorStats = bestTestErrorStats,
            ActualBasicStats = bestTestActualBasicStats,
            PredictedBasicStats = bestTestPredictedBasicStats,
            PredictionStats = bestTestPredictionStats
        },
        bestFitDetectionResult,
        iterationHistory.Count,
        _numOps
    );
    }

    private Matrix<T> InitializePheromones(int dimensions)
    {
        var pheromones = new Matrix<T>(dimensions, dimensions, _numOps);
        for (int i = 0; i < dimensions; i++)
        {
            for (int j = 0; j < dimensions; j++)
            {
                pheromones[i, j] = _numOps.One;
            }
        }
        return pheromones;
    }

    private Vector<T> ConstructSolution(Matrix<T> pheromones, Matrix<T> XTrain)
    {
        var solution = new Vector<T>(XTrain.Columns, _numOps);
        var visited = new bool[XTrain.Columns];
        int current = _random.Next(XTrain.Columns);
        visited[current] = true;

        for (int i = 1; i < XTrain.Columns; i++)
        {
            int next = SelectNextFeature(current, pheromones, XTrain, visited);
            solution[next] = _numOps.FromDouble(_random.NextDouble() * 2 - 1); // Random value between -1 and 1
            visited[next] = true;
            current = next;
        }

        return solution;
    }

    private int SelectNextFeature(int current, Matrix<T> pheromones, Matrix<T> XTrain, bool[] visited)
    {
        var probabilities = new Vector<T>(XTrain.Columns, _numOps);
        T total = _numOps.Zero;

        for (int i = 0; i < XTrain.Columns; i++)
        {
            if (!visited[i])
            {
                T pheromone = pheromones[current, i];
                T heuristic = _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, _numOps.Abs(XTrain[current, i])));
                T probability = _numOps.Multiply(
                    _numOps.Power(pheromone, _numOps.FromDouble(_antColonyOptions.Alpha)),
                    _numOps.Power(heuristic, _numOps.FromDouble(_antColonyOptions.Beta))
                );
                probabilities[i] = probability;
                total = _numOps.Add(total, probability);
            }
        }

        T random = _numOps.Multiply(_numOps.FromDouble(_random.NextDouble()), total);
        T sum = _numOps.Zero;
        for (int i = 0; i < XTrain.Columns; i++)
        {
            if (!visited[i])
            {
                sum = _numOps.Add(sum, probabilities[i]);
                if (_numOps.GreaterThanOrEquals(sum, random))
                {
                    return i;
                }
            }
        }

        return -1; // This should never happen
    }

    private void UpdatePheromones(Matrix<T> pheromones, List<Vector<T>> solutions, List<T> fitnesses)
    {
        // Evaporation
        for (int i = 0; i < pheromones.Rows; i++)
        {
            for (int j = 0; j < pheromones.Columns; j++)
            {
                pheromones[i, j] = _numOps.Multiply(pheromones[i, j], _numOps.FromDouble(1 - _antColonyOptions.EvaporationRate));
            }
        }

        // Deposit
        for (int k = 0; k < solutions.Count; k++)
        {
            T deposit = _numOps.Divide(_numOps.One, fitnesses[k]);
            for (int i = 0; i < solutions[k].Length; i++)
            {
                for (int j = 0; j < solutions[k].Length; j++)
                {
                    if (i != j)
                    {
                        pheromones[i, j] = _numOps.Add(pheromones[i, j], _numOps.Multiply(deposit, _numOps.Abs(solutions[k][i])));
                    }
                }
            }
        }
    }
}