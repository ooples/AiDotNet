namespace AiDotNet.Optimizers;

public class NormalOptimizer : IOptimizationAlgorithm
{
    private readonly Random _random = new Random();

    public OptimizationResult Optimize(
    Matrix<double> XTrain,
    Vector<double> yTrain,
    Matrix<double> XVal,
    Vector<double> yVal,
    Matrix<double> XTest,
    Vector<double> yTest,
    PredictionModelOptions modelOptions,
    OptimizationAlgorithmOptions optimizationOptions,
    IRegression regressionMethod,
    IRegularization regularization,
    INormalizer normalizer,
    NormalizationInfo normInfo,
    IFitnessCalculator fitnessCalculator,
    IFitDetector fitDetector)
    {
        var bestSolution = new Vector<double>(XTrain.Columns);
        var bestIntercept = 0.0;
        double bestFitness = optimizationOptions.MaximizeFitness ? double.MinValue : double.MaxValue;
        var fitnessHistory = new List<double>();
        var iterationHistory = new List<OptimizationIteration>();
        FitDetectorResult? bestFitDetectionResult = null;
        Vector<double>? bestTrainingPredictions = null;
        Vector<double>? bestValidationPredictions = null;
        Vector<double>? bestTestPredictions = null;
        ErrorStats? bestTrainingErrorStats = null;
        ErrorStats? bestValidationErrorStats = null;
        ErrorStats? bestTestErrorStats = null;
        BasicStats? bestTrainingBasicStats = null;
        BasicStats? bestValidationBasicStats = null;
        BasicStats? bestTestBasicStats = null;

        for (int iteration = 0; iteration < optimizationOptions.MaxIterations; iteration++)
        {
            // Randomly select features
            var selectedFeatures = RandomlySelectFeatures(XTrain.Columns, modelOptions.MinimumFeatures, modelOptions.MaximumFeatures);

            // Create subsets of the data with selected features
            var XTrainSubset = XTrain.SubMatrix(0, XTrain.Rows - 1, selectedFeatures);
            var XValSubset = XVal.SubMatrix(0, XVal.Rows - 1, selectedFeatures);
            var XTestSubset = XTest.SubMatrix(0, XTest.Rows - 1, selectedFeatures);

            // Fit the model
            regressionMethod.Fit(XTrainSubset, yTrain, regularization);

            // Denormalize coefficients and intercept
            var denormalizedCoefficients = normalizer.DenormalizeCoefficients(regressionMethod.Coefficients, normInfo.XParams, normInfo.YParams);
            var denormalizedIntercept = normalizer.DenormalizeYIntercept(XTrainSubset, yTrain, regressionMethod.Coefficients, normInfo.XParams, normInfo.YParams);

            // Calculate predictions for all sets
            var trainingPredictions = XTrainSubset.Multiply(denormalizedCoefficients).Add(denormalizedIntercept);
            var validationPredictions = XValSubset.Multiply(denormalizedCoefficients).Add(denormalizedIntercept);
            var testPredictions = XTestSubset.Multiply(denormalizedCoefficients).Add(denormalizedIntercept);

            // Calculate error stats and basic stats for all sets
            var featureCount = selectedFeatures.Count + 1;
            var trainingErrorStats = new ErrorStats(yTrain, trainingPredictions, featureCount); // +1 for intercept
            var validationErrorStats = new ErrorStats(yVal, validationPredictions, featureCount);
            var testErrorStats = new ErrorStats(yTest, testPredictions, featureCount);

            var trainingBasicStats = new BasicStats(yTrain, trainingPredictions, featureCount);
            var validationBasicStats = new BasicStats(yVal, validationPredictions, featureCount);
            var testBasicStats = new BasicStats(yTest, testPredictions, featureCount);

            // Detect fit type
            var fitDetectionResult = fitDetector.DetectFit(trainingErrorStats, validationErrorStats, testErrorStats, trainingBasicStats, validationBasicStats, testBasicStats);

            // Update best solution if necessary
            if ((optimizationOptions.MaximizeFitness && validationBasicStats.R2 > bestFitness) ||
                (!optimizationOptions.MaximizeFitness && validationBasicStats.R2 < bestFitness))
            {
                bestFitness = validationBasicStats.R2;
                bestSolution = denormalizedCoefficients;
                bestIntercept = denormalizedIntercept;
                bestFitDetectionResult = fitDetectionResult;
                bestTrainingPredictions = trainingPredictions;
                bestValidationPredictions = validationPredictions;
                bestTestPredictions = testPredictions;
                bestTrainingErrorStats = trainingErrorStats;
                bestValidationErrorStats = validationErrorStats;
                bestTestErrorStats = testErrorStats;
                bestTrainingBasicStats = trainingBasicStats;
                bestValidationBasicStats = validationBasicStats;
                bestTestBasicStats = testBasicStats;
            }

            fitnessHistory.Add(validationBasicStats.R2);
            iterationHistory.Add(new OptimizationIteration
            {
                Iteration = iteration,
                Fitness = validationBasicStats.R2,
                FitDetectionResult = fitDetectionResult
            });

            // Check for early stopping
            if (optimizationOptions.UseEarlyStopping && ShouldEarlyStop(iterationHistory, optimizationOptions))
            {
                break;
            }
        }

        return new OptimizationResult
        {
            Regression = 
            FitnessScore = bestFitness,
            Iterations = iterationHistory.Count,
            FitnessHistory = new Vector<double>([.. fitnessHistory]),
            TrainingPredictions = bestTrainingPredictions,
            ValidationPredictions = bestValidationPredictions,
            TestPredictions = bestTestPredictions,
            FitDetectionResult = bestFitDetectionResult,
            TrainingErrorStats = bestTrainingErrorStats,
            ValidationErrorStats = bestValidationErrorStats,
            TestErrorStats = bestTestErrorStats,
            TrainingBasicStats = bestTrainingBasicStats,
            ValidationBasicStats = bestValidationBasicStats,
            TestBasicStats = bestTestBasicStats,
            LowerBounds = lowerBounds,
            UpperBounds = upperBounds
        };
    }

    public bool ShouldEarlyStop(List<OptimizationIteration> iterationHistory, OptimizationAlgorithmOptions options)
    {
        if (iterationHistory.Count < options.EarlyStoppingPatience)
        {
            return false;
        }

        var recentIterations = iterationHistory.Skip(Math.Max(0, iterationHistory.Count - options.EarlyStoppingPatience)).ToList();

        // Check for improvement in recent iterations
        var bestFitness = options.MaximizeFitness ? 
            iterationHistory.Max(i => i.Fitness) : 
            iterationHistory.Min(i => i.Fitness);
    
        bool noImprovement = true;
        foreach (var iteration in recentIterations)
        {
            if ((options.MaximizeFitness && iteration.Fitness > bestFitness + options.EarlyStoppingMinDelta) ||
                (!options.MaximizeFitness && iteration.Fitness < bestFitness - options.EarlyStoppingMinDelta))
            {
                noImprovement = false;
                break;
            }
        }

        // Check for consecutive bad fits
        int consecutiveBadFits = 0;
        foreach (var iteration in recentIterations.Reverse<OptimizationIteration>())
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

        return noImprovement || consecutiveBadFits >= options.BadFitPatience;
    }

    private List<int> RandomlySelectFeatures(int totalFeatures, int minFeatures, int maxFeatures)
    {
        int numFeatures = _random.Next(minFeatures, Math.Min(maxFeatures, totalFeatures) + 1);
        return Enumerable.Range(0, totalFeatures).OrderBy(x => _random.Next()).Take(numFeatures).ToList();
    }
}