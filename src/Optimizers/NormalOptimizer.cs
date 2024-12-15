namespace AiDotNet.Optimizers;

public class NormalOptimizer<T> : IOptimizationAlgorithm<T>
{
    private readonly Random _random = new();
    private readonly INumericOperations<T> _numOps;
    private readonly OptimizationAlgorithmOptions _optimizationOptions;

    public NormalOptimizer(OptimizationAlgorithmOptions? optimizationOptions = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _optimizationOptions = optimizationOptions ?? new OptimizationAlgorithmOptions();
    }

    public OptimizationResult<T> Optimize(
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
        var bestSolution = new Vector<T>(XTrain.Columns, _numOps);
        var bestIntercept = _numOps.Zero;
        T bestFitness = _optimizationOptions.MaximizeFitness ? _numOps.MinValue : _numOps.MaxValue;
        var fitnessHistory = new List<T>();
        var iterationHistory = new List<OptimizationIterationInfo<T>>();
        var bestSelectedFeatures = new List<Vector<T>>();
        FitDetectorResult<T>? bestFitDetectionResult = null;
        Vector<T>? bestTrainingPredictions = null;
        Vector<T>? bestValidationPredictions = null;
        Vector<T>? bestTestPredictions = null;
        Matrix<T>? bestTrainingFeatures = null;
        Matrix<T>? bestValidationFeatures = null;
        Matrix<T>? bestTestFeatures = null;
        ErrorStats<T>? bestTrainingErrorStats = null;
        ErrorStats<T>? bestValidationErrorStats = null;
        ErrorStats<T>? bestTestErrorStats = null;
        BasicStats<T>? bestTrainingActualBasicStats = null;
        BasicStats<T>? bestTrainingPredictedBasicStats = null;
        BasicStats<T>? bestValidationActualBasicStats = null;
        BasicStats<T>? bestValidationPredictedBasicStats = null;
        BasicStats<T>? bestTestActualBasicStats = null;
        BasicStats<T>? bestTestPredictedBasicStats = null;
        PredictionStats<T>? bestTrainingPredictionStats = null;
        PredictionStats<T>? bestValidationPredictionStats = null;
        PredictionStats<T>? bestTestPredictionStats = null;

        for (int iteration = 0; iteration < _optimizationOptions.MaxIterations; iteration++)
        {
            // Randomly select features
            var selectedFeatures = RandomlySelectFeatures(XTrain.Columns, modelOptions.MinimumFeatures, modelOptions.MaximumFeatures);

            // Create subsets of the data with selected features
            var XTrainSubset = XTrain.SubMatrix(0, XTrain.Rows - 1, selectedFeatures);
            var XValSubset = XVal.SubMatrix(0, XVal.Rows - 1, selectedFeatures);
            var XTestSubset = XTest.SubMatrix(0, XTest.Rows - 1, selectedFeatures);

            // Fit the model
            regressionMethod.Fit(XTrainSubset, yTrain);

            // Denormalize coefficients and intercept
            var denormalizedCoefficients = normalizer.DenormalizeCoefficients(regressionMethod.Coefficients, normInfo.XParams, normInfo.YParams);
            var denormalizedIntercept = normalizer.DenormalizeYIntercept(XTrainSubset, yTrain, regressionMethod.Coefficients, normInfo.XParams, normInfo.YParams);

            // Calculate predictions for all sets
            var trainingPredictions = XTrainSubset.Multiply(denormalizedCoefficients).Add(denormalizedIntercept);
            var validationPredictions = XValSubset.Multiply(denormalizedCoefficients).Add(denormalizedIntercept);
            var testPredictions = XTestSubset.Multiply(denormalizedCoefficients).Add(denormalizedIntercept);

            // Calculate error stats, basic stats, and prediction stats for all sets
            var featureCount = selectedFeatures.Count + 1;
            var trainingErrorStats = new ErrorStats<T>(yTrain, trainingPredictions, featureCount);
            var validationErrorStats = new ErrorStats<T>(yVal, validationPredictions, featureCount);
            var testErrorStats = new ErrorStats<T>(yTest, testPredictions, featureCount);

            var trainingActualBasicStats = new BasicStats<T>(yTrain);
            var trainingPredictedBasicStats = new BasicStats<T>(trainingPredictions);
            var validationActualBasicStats = new BasicStats<T>(yVal);
            var validationPredictedBasicStats = new BasicStats<T>(validationPredictions);
            var testActualBasicStats = new BasicStats<T>(yTest);
            var testPredictedBasicStats = new BasicStats<T>(testPredictions);

            var trainingPredictionStats = new PredictionStats<T>(yTrain, trainingPredictions, featureCount, _numOps.FromDouble(_optimizationOptions.ConfidenceLevel), _numOps);
            var validationPredictionStats = new PredictionStats<T>(yVal, validationPredictions, featureCount, _numOps.FromDouble(_optimizationOptions.ConfidenceLevel), _numOps);
            var testPredictionStats = new PredictionStats<T>(yTest, testPredictions, featureCount, _numOps.FromDouble(_optimizationOptions.ConfidenceLevel), _numOps);

            // Detect fit type
            var fitDetectionResult = fitDetector.DetectFit(
                trainingErrorStats, validationErrorStats, testErrorStats,
                trainingActualBasicStats, trainingPredictedBasicStats,
                validationActualBasicStats, validationPredictedBasicStats,
                testActualBasicStats, testPredictedBasicStats,
                trainingPredictionStats, validationPredictionStats, testPredictionStats);

            // Calculate fitness score
            T currentFitnessScore = fitnessCalculator.CalculateFitnessScore(
                validationErrorStats, validationActualBasicStats, validationPredictedBasicStats,
                yVal, validationPredictions, XValSubset, validationPredictionStats);

            // Update best solution if necessary
            if (fitnessCalculator.IsBetterFitness(currentFitnessScore, bestFitness))
            {
                bestFitness = currentFitnessScore;
                bestSolution = denormalizedCoefficients;
                bestIntercept = denormalizedIntercept;
                bestFitDetectionResult = fitDetectionResult;
                bestTrainingPredictions = trainingPredictions;
                bestValidationPredictions = validationPredictions;
                bestTestPredictions = testPredictions;
                bestTrainingErrorStats = trainingErrorStats;
                bestValidationErrorStats = validationErrorStats;
                bestTestErrorStats = testErrorStats;
                bestTrainingActualBasicStats = trainingActualBasicStats;
                bestTrainingPredictedBasicStats = trainingPredictedBasicStats;
                bestValidationActualBasicStats = validationActualBasicStats;
                bestValidationPredictedBasicStats = validationPredictedBasicStats;
                bestTestActualBasicStats = testActualBasicStats;
                bestTestPredictedBasicStats = testPredictedBasicStats;
                bestTrainingPredictionStats = trainingPredictionStats;
                bestValidationPredictionStats = validationPredictionStats;
                bestTestPredictionStats = testPredictionStats;
                bestSelectedFeatures = [.. selectedFeatures.Select(XTrain.GetColumn)];
                bestTestFeatures = XTestSubset;
                bestTrainingFeatures = XTrainSubset;
                bestValidationFeatures = XValSubset;
            }

            fitnessHistory.Add(currentFitnessScore);
            iterationHistory.Add(new OptimizationIterationInfo<T>
            {
                Iteration = iteration,
                Fitness = currentFitnessScore,
                FitDetectionResult = fitDetectionResult
            });

            // Check for early stopping
            if (_optimizationOptions.UseEarlyStopping && ShouldEarlyStop(iterationHistory, fitnessCalculator))
            {
                break;
            }
        }

        return new OptimizationResult<T>
        {
            BestCoefficients = bestSolution,
            BestIntercept = bestIntercept,
            FitnessScore = bestFitness,
            Iterations = iterationHistory.Count,
            FitnessHistory = new Vector<T>([.. fitnessHistory], _numOps),
            SelectedFeatures = bestSelectedFeatures,
    
            TrainingResult = new OptimizationResult<T>.DatasetResult
            {
                Predictions = bestTrainingPredictions ?? Vector<T>.Empty(),
                ErrorStats = bestTrainingErrorStats ?? ErrorStats<T>.Empty(),
                ActualBasicStats = bestTrainingActualBasicStats ?? BasicStats<T>.Empty(),
                PredictedBasicStats = bestTrainingPredictedBasicStats ?? BasicStats<T>.Empty(),
                PredictionStats = bestTrainingPredictionStats ?? PredictionStats<T>.Empty(),
                X = bestTrainingFeatures ?? Matrix<T>.Empty(),
                Y = yTrain
            },
            ValidationResult = new OptimizationResult<T>.DatasetResult
            {
                Predictions = bestValidationPredictions ?? Vector<T>.Empty(),
                ErrorStats = bestValidationErrorStats ?? ErrorStats<T>.Empty(),
                ActualBasicStats = bestValidationActualBasicStats ?? BasicStats<T>.Empty(),
                PredictedBasicStats = bestValidationPredictedBasicStats ?? BasicStats<T>.Empty(),
                PredictionStats = bestValidationPredictionStats ?? PredictionStats<T>.Empty(),
                X = bestValidationFeatures ?? Matrix<T>.Empty(),
                Y = yVal
            },
            TestResult = new OptimizationResult<T>.DatasetResult
            {
                Predictions = bestTestPredictions ?? Vector<T>.Empty(),
                ErrorStats = bestTestErrorStats ?? ErrorStats<T>.Empty(),
                ActualBasicStats = bestTestActualBasicStats ?? BasicStats<T>.Empty(),
                PredictedBasicStats = bestTestPredictedBasicStats ?? BasicStats<T>.Empty(),
                PredictionStats = bestTestPredictionStats ?? PredictionStats<T>.Empty(),
                X = bestTestFeatures ?? Matrix<T>.Empty(),
                Y = yTest
            },
    
            FitDetectionResult = bestFitDetectionResult ?? new FitDetectorResult<T>(),
    
            CoefficientLowerBounds = Vector<T>.Empty(),
            CoefficientUpperBounds = Vector<T>.Empty()
        };
    }

    public bool ShouldEarlyStop(List<OptimizationIterationInfo<T>> iterationHistory, IFitnessCalculator<T> fitnessCalculator)
    {
        if (iterationHistory.Count < _optimizationOptions.EarlyStoppingPatience)
        {
            return false;
        }

        var recentIterations = iterationHistory.Skip(Math.Max(0, iterationHistory.Count - _optimizationOptions.EarlyStoppingPatience)).ToList();

        // Find the best fitness score
        T bestFitness = iterationHistory[0].Fitness;
        foreach (var iteration in iterationHistory)
        {
            if (fitnessCalculator.IsBetterFitness(iteration.Fitness, bestFitness))
            {
                bestFitness = iteration.Fitness;
            }
        }

        // Check for improvement in recent iterations
        bool noImprovement = true;
        foreach (var iteration in recentIterations)
        {
            if (fitnessCalculator.IsBetterFitness(iteration.Fitness, bestFitness))
            {
                noImprovement = false;
                break;
            }
        }

        // Check for consecutive bad fits
        int consecutiveBadFits = 0;
        foreach (var iteration in recentIterations.Reverse<OptimizationIterationInfo<T>>())
        {
            if (iteration.FitDetectionResult.FitType != FitType.Good)
            {
                consecutiveBadFits++;
            }
            else
            {
                break;
            }
        }

        return noImprovement || consecutiveBadFits >= _optimizationOptions.BadFitPatience;
    }

    private List<int> RandomlySelectFeatures(int totalFeatures, int minFeatures, int maxFeatures)
    {
        int numFeatures = _random.Next(minFeatures, Math.Min(maxFeatures, totalFeatures) + 1);
        return [.. Enumerable.Range(0, totalFeatures).OrderBy(x => _random.Next()).Take(numFeatures)];
    }
}