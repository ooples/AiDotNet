namespace AiDotNet.Optimizers;

public class NormalOptimizer : IOptimizationAlgorithm
{
    private readonly Random _random = new();

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
        var bestSelectedFeatures = new List<Vector<double>>();
        FitDetectorResult? bestFitDetectionResult = null;
        Vector<double>? bestTrainingPredictions = null;
        Vector<double>? bestValidationPredictions = null;
        Vector<double>? bestTestPredictions = null;
        Matrix<double>? bestTrainingFeatures = null;
        Matrix<double>? bestValidationFeatures = null;
        Matrix<double>? bestTestFeatures = null;
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

            // Calculate fitness score
            double currentFitnessScore = fitnessCalculator.CalculateFitnessScore(validationErrorStats, validationBasicStats, yVal, validationPredictions, XValSubset);

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
                bestTrainingBasicStats = trainingBasicStats;
                bestValidationBasicStats = validationBasicStats;
                bestTestBasicStats = testBasicStats;
                bestSelectedFeatures = [.. selectedFeatures.Select(XTrain.GetColumn)];
                bestTestFeatures = XTestSubset;
                bestTrainingFeatures = XTrainSubset;
                bestValidationFeatures = XValSubset;
            }

            fitnessHistory.Add(currentFitnessScore);
            iterationHistory.Add(new OptimizationIteration
            {
                Iteration = iteration,
                Fitness = currentFitnessScore,
                FitDetectionResult = fitDetectionResult
            });

            // Check for early stopping
            if (optimizationOptions.UseEarlyStopping && ShouldEarlyStop(iterationHistory, optimizationOptions, fitnessCalculator))
            {
                break;
            }
        }

        return new OptimizationResult
        {
            BestCoefficients = bestSolution,
            BestIntercept = bestIntercept,
            FitnessScore = bestFitness,
            Iterations = iterationHistory.Count,
            FitnessHistory = new Vector<double>([.. fitnessHistory]),
            SelectedFeatures = bestSelectedFeatures,
    
            TrainingResult = new OptimizationResult.DatasetResult
            {
                Predictions = bestTrainingPredictions ?? Vector<double>.Empty(),
                ErrorStats = bestTrainingErrorStats ?? ErrorStats.Empty(),
                BasicStats = bestTrainingBasicStats ?? BasicStats.Empty(),
                X = bestTrainingFeatures ?? Matrix<double>.Empty(),
                Y = yTrain
            },
            ValidationResult = new OptimizationResult.DatasetResult
            {
                Predictions = bestValidationPredictions ?? Vector<double>.Empty(),
                ErrorStats = bestValidationErrorStats ?? ErrorStats.Empty(),
                BasicStats = bestValidationBasicStats ?? BasicStats.Empty(),
                X = bestValidationFeatures ?? Matrix<double>.Empty(),
                Y = yVal
            },
            TestResult = new OptimizationResult.DatasetResult
            {
                Predictions = bestTestPredictions ?? Vector<double>.Empty(),
                ErrorStats = bestTestErrorStats ?? ErrorStats.Empty(),
                BasicStats = bestTestBasicStats ?? BasicStats.Empty(),
                X = bestTestFeatures ?? Matrix<double>.Empty(),
                Y = yTest
            },
    
            FitDetectionResult = bestFitDetectionResult ?? new FitDetectorResult(),
    
            CoefficientLowerBounds = Vector<double>.Empty(),
            CoefficientUpperBounds = Vector<double>.Empty()
        };
    }

    public bool ShouldEarlyStop(List<OptimizationIteration> iterationHistory, OptimizationAlgorithmOptions options, IFitnessCalculator fitnessCalculator)
    {
        if (iterationHistory.Count < options.EarlyStoppingPatience)
        {
            return false;
        }

        var recentIterations = iterationHistory.Skip(Math.Max(0, iterationHistory.Count - options.EarlyStoppingPatience)).ToList();

        // Find the best fitness score
        double bestFitness = iterationHistory[0].Fitness;
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
        return [.. Enumerable.Range(0, totalFeatures).OrderBy(x => _random.Next()).Take(numFeatures)];
    }
}