namespace AiDotNet.Optimizers;

public class DifferentialEvolutionOptimizer<T> : OptimizerBase<T>
{
    private readonly DifferentialEvolutionOptions _deOptions;
    private readonly Random _random;

    public DifferentialEvolutionOptimizer(DifferentialEvolutionOptions? options = null)
        : base(options)
    {
        _deOptions = options ?? new DifferentialEvolutionOptions();
        _random = new Random();
    }

    public override OptimizationResult<T> Optimize(
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
            IFitDetector<T> fitDetector)
    {
        int dimensions = XTrain.Columns;
        var population = InitializePopulation(dimensions, _deOptions.PopulationSize);
        Vector<T> bestSolution = new(dimensions, _numOps);
        T bestIntercept = _numOps.Zero;
        T bestFitness = fitnessCalculator.IsHigherScoreBetter ? _numOps.MinValue : _numOps.MaxValue;
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

        var fitnessHistory = new List<T>();
        var iterationHistory = new List<OptimizationIterationInfo<T>>();

        for (int generation = 0; generation < _options.MaxIterations; generation++)
        {
            for (int i = 0; i < _deOptions.PopulationSize; i++)
            {
                var trial = GenerateTrialVector(population, i, dimensions);
                var selectedFeatures = OptimizerHelper.GetSelectedFeatures(trial);
                var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
                var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
                var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

                var (currentFitnessScore, fitDetectionResult, trainingPredictions, validationPredictions, testPredictions,
                    trainingErrorStats, validationErrorStats, testErrorStats,
                    trainingActualBasicStats, trainingPredictedBasicStats,
                    validationActualBasicStats, validationPredictedBasicStats,
                    testActualBasicStats, testPredictedBasicStats,
                    trainingPredictionStats, validationPredictionStats, testPredictionStats) = EvaluateSolution(
                        XTrainSubset, XValSubset, XTestSubset,
                        yTrain, yVal, yTest,
                        regressionMethod, normalizer, normInfo,
                        fitnessCalculator, fitDetector, selectedFeatures.Count);

                UpdateBestSolution(
                    currentFitnessScore, trial, _numOps.Zero, fitDetectionResult,
                    trainingPredictions, validationPredictions, testPredictions,
                    trainingErrorStats, validationErrorStats, testErrorStats,
                    trainingActualBasicStats, trainingPredictedBasicStats,
                    validationActualBasicStats, validationPredictedBasicStats,
                    testActualBasicStats, testPredictedBasicStats,
                    trainingPredictionStats, validationPredictionStats, testPredictionStats,
                    selectedFeatures, XTrain, XTestSubset, XTrainSubset, XValSubset, fitnessCalculator,
                    ref bestFitness, ref bestSolution, ref bestIntercept, ref bestFitDetectionResult,
                    ref bestTrainingPredictions, ref bestValidationPredictions, ref bestTestPredictions,
                    ref bestTrainingErrorStats, ref bestValidationErrorStats, ref bestTestErrorStats,
                    ref bestTrainingActualBasicStats, ref bestTrainingPredictedBasicStats,
                    ref bestValidationActualBasicStats, ref bestValidationPredictedBasicStats,
                    ref bestTestActualBasicStats, ref bestTestPredictedBasicStats,
                    ref bestTrainingPredictionStats, ref bestValidationPredictionStats, ref bestTestPredictionStats,
                    ref bestSelectedFeatures, ref bestTestFeatures, ref bestTrainingFeatures, ref bestValidationFeatures);

                population[i] = trial;
            }

            if (UpdateIterationHistoryAndCheckEarlyStopping(fitnessHistory, iterationHistory, generation, bestFitness, bestFitDetectionResult, fitnessCalculator))
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
            OptimizerHelper.CreateDatasetResult(bestTrainingPredictions, bestTrainingErrorStats, bestTrainingActualBasicStats, bestTrainingPredictedBasicStats, bestTrainingPredictionStats, bestTrainingFeatures, yTrain),
            OptimizerHelper.CreateDatasetResult(bestValidationPredictions, bestValidationErrorStats, bestValidationActualBasicStats, bestValidationPredictedBasicStats, bestValidationPredictionStats, bestValidationFeatures, yVal),
            OptimizerHelper.CreateDatasetResult(bestTestPredictions, bestTestErrorStats, bestTestActualBasicStats, bestTestPredictedBasicStats, bestTestPredictionStats, bestTestFeatures, yTest),
            bestFitDetectionResult,
            fitnessHistory.Count,
            _numOps);
    }

    private List<Vector<T>> InitializePopulation(int dimensions, int populationSize)
    {
        var population = new List<Vector<T>>();
        for (int i = 0; i < populationSize; i++)
        {
            var individual = new Vector<T>(dimensions, _numOps);
            for (int j = 0; j < dimensions; j++)
            {
                individual[j] = _numOps.FromDouble(_random.NextDouble() * 2 - 1); // Random values between -1 and 1
            }
            population.Add(individual);
        }

        return population;
    }

    private Vector<T> GenerateTrialVector(List<Vector<T>> population, int currentIndex, int dimensions)
    {
        int a, b, c;
        do
        {
            a = _random.Next(population.Count);
        } while (a == currentIndex);

        do
        {
            b = _random.Next(population.Count);
        } while (b == currentIndex || b == a);

        do
        {
            c = _random.Next(population.Count);
        } while (c == currentIndex || c == a || c == b);

        var trial = population[currentIndex].Copy();
        int R = _random.Next(dimensions);

        for (int i = 0; i < dimensions; i++)
        {
            if (_random.NextDouble() < _deOptions.CrossoverRate || i == R)
            {
                trial[i] = _numOps.Add(population[a][i],
                    _numOps.Multiply(_numOps.FromDouble(_deOptions.MutationFactor),
                        _numOps.Subtract(population[b][i], population[c][i])));
            }
        }

        return trial;
    }
}