global using AiDotNet.Models.Results;
global using AiDotNet.Models.Inputs;

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
        IFullModel<T> regressionMethod,
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

    protected ModelStats<T> CalculateModelStats(Matrix<T> X, int featureCount, IPredictiveModel<T> model)
    {
        return new ModelStats<T>(new ModelStatsInputs<T>
        {
            XMatrix = X,
            FeatureCount = featureCount,
            Model = model
        });
    }

    protected (T CurrentFitnessScore, FitDetectorResult<T> FitDetectionResult, Vector<T> TrainingPredictions, Vector<T> ValidationPredictions, 
        Vector<T> TestPredictions, ModelEvaluationData<T> evaluationData)
    EvaluateSolution(
        ISymbolicModel<T> model,
        Matrix<T> XTrain, Matrix<T> XVal, Matrix<T> XTest,
        Vector<T> yTrain, Vector<T> yVal, Vector<T> yTest,
        INormalizer<T> normalizer, NormalizationInfo<T> normInfo,
        IFitnessCalculator<T> fitnessCalculator, IFitDetector<T> fitDetector)
    {
        // Calculate predictions for all sets
        var trainingPredictions = PredictWithSymbolicModel(model, XTrain);
        var validationPredictions = PredictWithSymbolicModel(model, XVal);
        var testPredictions = PredictWithSymbolicModel(model, XTest);

        int featureCount = XTrain.Columns;

        // Calculate error stats, basic stats, and prediction stats for all sets
        var (trainingErrorStats, validationErrorStats, testErrorStats) = CalculateErrorStats(
            yTrain, yVal, yTest, trainingPredictions, validationPredictions, testPredictions, featureCount);

        var (trainingActualBasicStats, trainingPredictedBasicStats,
             validationActualBasicStats, validationPredictedBasicStats,
             testActualBasicStats, testPredictedBasicStats) = CalculateBasicStats(
            yTrain, yVal, yTest, trainingPredictions, validationPredictions, testPredictions);

        var (trainingPredictionStats, validationPredictionStats, testPredictionStats) = CalculatePredictionStats(
            yTrain, yVal, yTest, trainingPredictions, validationPredictions, testPredictions, featureCount);

        var predictionModelResult = new PredictionModelResult<T>((IFullModel<T>)model, new OptimizationResult<T>(), normInfo);

        var modelStats = CalculateModelStats(XTrain, featureCount, predictionModelResult);
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
            yVal, validationPredictions, XVal, evaluationData.ValidationPredictionStats);

        return (currentFitnessScore, fitDetectionResult, 
            trainingPredictions, validationPredictions, testPredictions, evaluationData);
    }

    private Vector<T> PredictWithSymbolicModel(ISymbolicModel<T> model, Matrix<T> X)
    {
        var predictions = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            predictions[i] = model.Evaluate(X.GetRow(i));
        }

        return predictions;
    }

    protected void UpdateBestSolution(
        ModelResult<T> currentResult,
        ModelResult<T> bestResult,
        Matrix<T> XTrain,
        Matrix<T> XTest,
        Matrix<T> XVal,
        IFitnessCalculator<T> fitnessCalculator)
    {
        if (fitnessCalculator.IsBetterFitness(currentResult.Fitness, bestResult.Fitness))
        {
            bestResult.Solution = currentResult.Solution;
            bestResult.Fitness = currentResult.Fitness;
            bestResult.TrainingPredictions = currentResult.TrainingPredictions;
            bestResult.ValidationPredictions = currentResult.ValidationPredictions;
            bestResult.TestPredictions = currentResult.TestPredictions;
            bestResult.EvaluationData = currentResult.EvaluationData;
            bestResult.SelectedFeatures = currentResult.SelectedFeatures;
            bestResult.FitDetectionResult = currentResult.FitDetectionResult;
            bestResult.TrainingFeatures = OptimizerHelper.SelectFeatures(XTrain, currentResult.SelectedFeatures);
            bestResult.ValidationFeatures = OptimizerHelper.SelectFeatures(XVal, currentResult.SelectedFeatures);
            bestResult.TestFeatures = OptimizerHelper.SelectFeatures(XTest, currentResult.SelectedFeatures);
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
            if (iteration.FitDetectionResult.FitType != FitType.GoodFit)
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