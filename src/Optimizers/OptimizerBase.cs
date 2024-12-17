namespace AiDotNet.Optimizers;

public abstract class OptimizerBase<T> : IOptimizationAlgorithm<T>
{
    protected readonly INumericOperations<T> _numOps;
    protected readonly OptimizationAlgorithmOptions _options;

    protected OptimizerBase(OptimizationAlgorithmOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new OptimizationAlgorithmOptions();
    }

    public abstract OptimizationResult<T> Optimize(
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
        IFitDetector<T> fitDetector);

    protected (Vector<T> TrainingPredictions, Vector<T> ValidationPredictions, Vector<T> TestPredictions)
    CalculatePredictions(Matrix<T> XTrainSubset, Matrix<T> XValSubset, Matrix<T> XTestSubset, 
                         Vector<T> denormalizedCoefficients, T denormalizedIntercept)
    {
        var trainingPredictions = XTrainSubset.Multiply(denormalizedCoefficients).Add(denormalizedIntercept);
        var validationPredictions = XValSubset.Multiply(denormalizedCoefficients).Add(denormalizedIntercept);
        var testPredictions = XTestSubset.Multiply(denormalizedCoefficients).Add(denormalizedIntercept);
        return (trainingPredictions, validationPredictions, testPredictions);
    }

    protected (ErrorStats<T> TrainingErrorStats, ErrorStats<T> ValidationErrorStats, ErrorStats<T> TestErrorStats)
    CalculateErrorStats(Vector<T> yTrain, Vector<T> yVal, Vector<T> yTest, 
                        Vector<T> trainingPredictions, Vector<T> validationPredictions, Vector<T> testPredictions, 
                        int featureCount)
    {
        var trainingErrorStats = new ErrorStats<T>(yTrain, trainingPredictions, featureCount);
        var validationErrorStats = new ErrorStats<T>(yVal, validationPredictions, featureCount);
        var testErrorStats = new ErrorStats<T>(yTest, testPredictions, featureCount);
        return (trainingErrorStats, validationErrorStats, testErrorStats);
    }

    protected (BasicStats<T> TrainingActual, BasicStats<T> TrainingPredicted, 
                BasicStats<T> ValidationActual, BasicStats<T> ValidationPredicted, 
                BasicStats<T> TestActual, BasicStats<T> TestPredicted)
    CalculateBasicStats(Vector<T> yTrain, Vector<T> yVal, Vector<T> yTest, 
                        Vector<T> trainingPredictions, Vector<T> validationPredictions, Vector<T> testPredictions)
    {
        return (
            new BasicStats<T>(yTrain),
            new BasicStats<T>(trainingPredictions),
            new BasicStats<T>(yVal),
            new BasicStats<T>(validationPredictions),
            new BasicStats<T>(yTest),
            new BasicStats<T>(testPredictions)
        );
    }

    protected (PredictionStats<T> TrainingPredictionStats, PredictionStats<T> ValidationPredictionStats, PredictionStats<T> TestPredictionStats)
    CalculatePredictionStats(Vector<T> yTrain, Vector<T> yVal, Vector<T> yTest, 
                             Vector<T> trainingPredictions, Vector<T> validationPredictions, Vector<T> testPredictions, 
                             int featureCount)
    {
        var confidenceLevel = _numOps.FromDouble(_options.ConfidenceLevel);
        return (
            new PredictionStats<T>(yTrain, trainingPredictions, featureCount, confidenceLevel, _numOps),
            new PredictionStats<T>(yVal, validationPredictions, featureCount, confidenceLevel, _numOps),
            new PredictionStats<T>(yTest, testPredictions, featureCount, confidenceLevel, _numOps)
        );
    }

    protected (T CurrentFitnessScore, FitDetectorResult<T> FitDetectionResult, 
                Vector<T> TrainingPredictions, Vector<T> ValidationPredictions, Vector<T> TestPredictions,
                ErrorStats<T> TrainingErrorStats, ErrorStats<T> ValidationErrorStats, ErrorStats<T> TestErrorStats,
                BasicStats<T> TrainingActualBasicStats, BasicStats<T> TrainingPredictedBasicStats,
                BasicStats<T> ValidationActualBasicStats, BasicStats<T> ValidationPredictedBasicStats,
                BasicStats<T> TestActualBasicStats, BasicStats<T> TestPredictedBasicStats,
                PredictionStats<T> TrainingPredictionStats, PredictionStats<T> ValidationPredictionStats, PredictionStats<T> TestPredictionStats)
    EvaluateSolution(
        Matrix<T> XTrainSubset, Matrix<T> XValSubset, Matrix<T> XTestSubset,
        Vector<T> yTrain, Vector<T> yVal, Vector<T> yTest,
        IRegression<T> regressionMethod, INormalizer<T> normalizer, NormalizationInfo<T> normInfo,
        IFitnessCalculator<T> fitnessCalculator, IFitDetector<T> fitDetector, int featureCount)
    {
        // Fit the model
        regressionMethod.Fit(XTrainSubset, yTrain);

        // Denormalize coefficients and intercept
        var denormalizedCoefficients = normalizer.DenormalizeCoefficients(regressionMethod.Coefficients, normInfo.XParams, normInfo.YParams);
        var denormalizedIntercept = normalizer.DenormalizeYIntercept(XTrainSubset, yTrain, regressionMethod.Coefficients, normInfo.XParams, normInfo.YParams);

        // Calculate predictions for all sets
        var (trainingPredictions, validationPredictions, testPredictions) = CalculatePredictions(
            XTrainSubset, XValSubset, XTestSubset, denormalizedCoefficients, denormalizedIntercept);

        // Calculate error stats, basic stats, and prediction stats for all sets
        var (trainingErrorStats, validationErrorStats, testErrorStats) = CalculateErrorStats(
            yTrain, yVal, yTest, trainingPredictions, validationPredictions, testPredictions, featureCount);

        var (trainingActualBasicStats, trainingPredictedBasicStats,
             validationActualBasicStats, validationPredictedBasicStats,
             testActualBasicStats, testPredictedBasicStats) = CalculateBasicStats(
            yTrain, yVal, yTest, trainingPredictions, validationPredictions, testPredictions);

        var (trainingPredictionStats, validationPredictionStats, testPredictionStats) = CalculatePredictionStats(
            yTrain, yVal, yTest, trainingPredictions, validationPredictions, testPredictions, featureCount);

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

        return (currentFitnessScore, fitDetectionResult, 
                trainingPredictions, validationPredictions, testPredictions,
                trainingErrorStats, validationErrorStats, testErrorStats,
                trainingActualBasicStats, trainingPredictedBasicStats,
                validationActualBasicStats, validationPredictedBasicStats,
                testActualBasicStats, testPredictedBasicStats,
                trainingPredictionStats, validationPredictionStats, testPredictionStats);
    }

    protected void UpdateBestSolution(
        T currentFitnessScore,
        Vector<T> denormalizedCoefficients,
        T denormalizedIntercept,
        FitDetectorResult<T> fitDetectionResult,
        Vector<T> trainingPredictions,
        Vector<T> validationPredictions,
        Vector<T> testPredictions,
        ErrorStats<T> trainingErrorStats,
        ErrorStats<T> validationErrorStats,
        ErrorStats<T> testErrorStats,
        BasicStats<T> trainingActualBasicStats,
        BasicStats<T> trainingPredictedBasicStats,
        BasicStats<T> validationActualBasicStats,
        BasicStats<T> validationPredictedBasicStats,
        BasicStats<T> testActualBasicStats,
        BasicStats<T> testPredictedBasicStats,
        PredictionStats<T> trainingPredictionStats,
        PredictionStats<T> validationPredictionStats,
        PredictionStats<T> testPredictionStats,
        List<int> selectedFeatures,
        Matrix<T> XTrain,
        Matrix<T> XTestSubset,
        Matrix<T> XTrainSubset,
        Matrix<T> XValSubset,
        IFitnessCalculator<T> fitnessCalculator,
        ref T bestFitness,
        ref Vector<T> bestSolution,
        ref T bestIntercept,
        ref FitDetectorResult<T> bestFitDetectionResult,
        ref Vector<T> bestTrainingPredictions,
        ref Vector<T> bestValidationPredictions,
        ref Vector<T> bestTestPredictions,
        ref ErrorStats<T> bestTrainingErrorStats,
        ref ErrorStats<T> bestValidationErrorStats,
        ref ErrorStats<T> bestTestErrorStats,
        ref BasicStats<T> bestTrainingActualBasicStats,
        ref BasicStats<T> bestTrainingPredictedBasicStats,
        ref BasicStats<T> bestValidationActualBasicStats,
        ref BasicStats<T> bestValidationPredictedBasicStats,
        ref BasicStats<T> bestTestActualBasicStats,
        ref BasicStats<T> bestTestPredictedBasicStats,
        ref PredictionStats<T> bestTrainingPredictionStats,
        ref PredictionStats<T> bestValidationPredictionStats,
        ref PredictionStats<T> bestTestPredictionStats,
        ref List<Vector<T>> bestSelectedFeatures,
        ref Matrix<T> bestTestFeatures,
        ref Matrix<T> bestTrainingFeatures,
        ref Matrix<T> bestValidationFeatures)
    {
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
    }

    protected bool UpdateIterationHistoryAndCheckEarlyStopping(
    List<T> fitnessHistory,
    List<OptimizationIterationInfo<T>> iterationHistory,
    int iteration,
    T currentFitnessScore,
    FitDetectorResult<T> fitDetectionResult,
    IFitnessCalculator<T> fitnessCalculator)
    {
        fitnessHistory.Add(currentFitnessScore);
        iterationHistory.Add(new OptimizationIterationInfo<T>
        {
            Iteration = iteration,
            Fitness = currentFitnessScore,
            FitDetectionResult = fitDetectionResult
        });

        // Check for early stopping
        if (_options.UseEarlyStopping && ShouldEarlyStop(iterationHistory, fitnessCalculator))
        {
            return true; // Signal to stop the optimization
        }

        return false; // Continue optimization
    }

    public virtual bool ShouldEarlyStop(List<OptimizationIterationInfo<T>> iterationHistory, IFitnessCalculator<T> fitnessCalculator)
    {
        if (iterationHistory.Count < _options.EarlyStoppingPatience)
        {
            return false;
        }

        var recentIterations = iterationHistory.Skip(Math.Max(0, iterationHistory.Count - _options.EarlyStoppingPatience)).ToList();

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

        return noImprovement || consecutiveBadFits >= _options.BadFitPatience;
    }
}