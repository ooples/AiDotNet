namespace AiDotNet.Optimizers;

public abstract class OptimizerBase<T> : IOptimizationAlgorithm<T>
{
    protected readonly INumericOperations<T> _numOps;
    protected readonly OptimizationAlgorithmOptions _options;
    protected readonly PredictionStatsOptions _predictionOptions;
    protected readonly ModelStatsOptions _modelStatsOptions;

    protected OptimizerBase(OptimizationAlgorithmOptions? options = null, 
                        PredictionStatsOptions? predictionOptions = null,
                        ModelStatsOptions? modelOptions = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new OptimizationAlgorithmOptions();
        _predictionOptions = predictionOptions ?? new PredictionStatsOptions();
        _modelStatsOptions = modelOptions ?? new ModelStatsOptions();
    }

    public abstract OptimizationResult<T> Optimize(
        Matrix<T> XTrain,
        Vector<T> yTrain,
        Matrix<T> XVal,
        Vector<T> yVal,
        Matrix<T> XTest,
        Vector<T> yTest,
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

    protected (BasicStats<T> TrainingActual, BasicStats<T> TrainingPredicted, 
            BasicStats<T> ValidationActual, BasicStats<T> ValidationPredicted, 
            BasicStats<T> TestActual, BasicStats<T> TestPredicted)
    CalculateBasicStats(Vector<T> yTrain, Vector<T> yVal, Vector<T> yTest,
                        Vector<T> trainingPredictions, Vector<T> validationPredictions, Vector<T> testPredictions)
    {
        return (
            new BasicStats<T>(new BasicStatsInputs<T> { Values = yTrain }),
            new BasicStats<T>(new BasicStatsInputs<T> { Values = trainingPredictions }),
            new BasicStats<T>(new BasicStatsInputs<T> { Values = yVal }),
            new BasicStats<T>(new BasicStatsInputs<T> { Values = validationPredictions }),
            new BasicStats<T>(new BasicStatsInputs<T> { Values = yTest }),
            new BasicStats<T>(new BasicStatsInputs<T> { Values = testPredictions })
        );
    }

    protected (ErrorStats<T> Training, ErrorStats<T> Validation, ErrorStats<T> Test)
    CalculateErrorStats(Vector<T> yTrain, Vector<T> yVal, Vector<T> yTest, 
                        Vector<T> trainingPredictions, Vector<T> validationPredictions, Vector<T> testPredictions, 
                        int featureCount)
    {
        return (
            new ErrorStats<T>(new ErrorStatsInputs<T> { Actual = yTrain, Predicted = trainingPredictions, FeatureCount = featureCount }),
            new ErrorStats<T>(new ErrorStatsInputs<T> { Actual = yVal, Predicted = validationPredictions, FeatureCount = featureCount }),
            new ErrorStats<T>(new ErrorStatsInputs<T> { Actual = yTest, Predicted = testPredictions, FeatureCount = featureCount })
        );
    }

    protected (PredictionStats<T> Training, PredictionStats<T> Validation, PredictionStats<T> Test)
    CalculatePredictionStats(Vector<T> yTrain, Vector<T> yVal, Vector<T> yTest, 
                             Vector<T> trainingPredictions, Vector<T> validationPredictions, Vector<T> testPredictions, 
                             int featureCount)
    {
        return (
            new PredictionStats<T>(new PredictionStatsInputs<T> 
            { 
                Actual = yTrain, 
                Predicted = trainingPredictions, 
                NumberOfParameters = featureCount, 
                ConfidenceLevel = _predictionOptions.ConfidenceLevel, 
                LearningCurveSteps = _predictionOptions.LearningCurveSteps 
            }),
            new PredictionStats<T>(new PredictionStatsInputs<T> 
            { 
                Actual = yVal, 
                Predicted = validationPredictions, 
                NumberOfParameters = featureCount, 
                ConfidenceLevel = _predictionOptions.ConfidenceLevel, 
                LearningCurveSteps = _predictionOptions.LearningCurveSteps 
            }),
            new PredictionStats<T>(new PredictionStatsInputs<T> 
            { 
                Actual = yTest, 
                Predicted = testPredictions, 
                NumberOfParameters = featureCount, 
                ConfidenceLevel = _predictionOptions.ConfidenceLevel, 
                LearningCurveSteps = _predictionOptions.LearningCurveSteps 
            })
        );
    }

    protected ModelStats<T> CalculateModelStats(Matrix<T> X, int featureCount)
    {
        return new ModelStats<T>(new ModelStatsInputs<T>
        {
            XMatrix = X,
            FeatureCount = featureCount
        });
    }

    protected (T CurrentFitnessScore, FitDetectorResult<T> FitDetectionResult, Vector<T> TrainingPredictions, Vector<T> ValidationPredictions, 
        Vector<T> TestPredictions, ModelEvaluationData<T> evaluationData)
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

        var modelStats = CalculateModelStats(XTrainSubset, featureCount);

        var evaluationData = new ModelEvaluationData<T>()
        {
            TrainingErrorStats = trainingErrorStats,
            ValidationErrorStats = validationErrorStats,
            TestErrorStats = testErrorStats,
            TrainingPredictedBasicStats = trainingPredictedBasicStats,
            ValidationPredictedBasicStats = validationPredictedBasicStats,
            TestPredictedBasicStats = testPredictedBasicStats,
            TrainingActualBasicStats = trainingActualBasicStats,
            ValidationActualBasicStats = validationActualBasicStats,
            TestActualBasicStats = testActualBasicStats,
            TrainingPredictionStats = trainingPredictionStats,
            ValidationPredictionStats = validationPredictionStats,
            TestPredictionStats = testPredictionStats,
            ModelStats = modelStats
        };

        // Detect fit type
        var fitDetectionResult = fitDetector.DetectFit(evaluationData);

        // Calculate fitness score
        T currentFitnessScore = fitnessCalculator.CalculateFitnessScore(
            evaluationData.ValidationErrorStats, validationActualBasicStats, validationPredictedBasicStats,
            yVal, validationPredictions, XValSubset, evaluationData.ValidationPredictionStats);

        return (currentFitnessScore, fitDetectionResult, 
            trainingPredictions, validationPredictions, testPredictions, evaluationData);
    }

    protected void UpdateBestSolution(
        T currentFitnessScore,
        Vector<T> denormalizedCoefficients,
        T denormalizedIntercept,
        FitDetectorResult<T> fitDetectionResult,
        Vector<T> trainingPredictions,
        Vector<T> validationPredictions,
        Vector<T> testPredictions,
        ModelEvaluationData<T> evaluationData,
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
        ref ModelEvaluationData<T> bestEvaluationData,
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
            bestEvaluationData = evaluationData;
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