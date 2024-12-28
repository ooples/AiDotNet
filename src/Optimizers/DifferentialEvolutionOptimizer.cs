using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

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
            IFullModel<T> regressionMethod,
            IRegularization<T> regularization,
            INormalizer<T> normalizer,
            NormalizationInfo<T> normInfo,
            IFitnessCalculator<T> fitnessCalculator,
            IFitDetector<T> fitDetector)
    {
        int dimensions = XTrain.Columns;
        var population = InitializePopulation(dimensions, _deOptions.PopulationSize);
        var bestSolution = SymbolicModelFactory<T>.CreateEmptyModel(_options.UseExpressionTrees, XTrain.Columns, _numOps);
        T bestFitness = fitnessCalculator.IsHigherScoreBetter ? _numOps.MinValue : _numOps.MaxValue;
        FitDetectorResult<T> bestFitDetectionResult = new();
        Vector<T> bestTrainingPredictions = new(yTrain.Length, _numOps);
        Vector<T> bestValidationPredictions = new(yVal.Length, _numOps);
        Vector<T> bestTestPredictions = new(yTest.Length, _numOps);
        ModelEvaluationData<T> bestEvaluationData = new();
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
                var trial = GenerateTrialModel(population, i, dimensions);
                var selectedFeatures = OptimizerHelper.GetSelectedFeatures<T>(trial);
                var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
                var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
                var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

                var (currentFitnessScore, fitDetectionResult, trainingPredictions, validationPredictions, testPredictions, evaluationData) = EvaluateSolution(
                    trial, XTrainSubset, XValSubset, XTestSubset,
                    yTrain, yVal, yTest,
                    normalizer, normInfo,
                    fitnessCalculator, fitDetector);

                var currentResult = new ModelResult<T>
                {
                    Solution = trial,
                    Fitness = currentFitnessScore,
                    FitDetectionResult = fitDetectionResult,
                    TrainingPredictions = trainingPredictions,
                    ValidationPredictions = validationPredictions,
                    TestPredictions = testPredictions,
                    EvaluationData = evaluationData,
                    SelectedFeatures = selectedFeatures.ToVectorList<T>()
                };

                var bestResult = new ModelResult<T>
                {
                    Solution = bestSolution ?? new VectorModel<T>(Vector<T>.Empty(), _numOps),
                    Fitness = bestFitness,
                    FitDetectionResult = bestFitDetectionResult,
                    TrainingPredictions = bestTrainingPredictions,
                    ValidationPredictions = bestValidationPredictions,
                    TestPredictions = bestTestPredictions,
                    EvaluationData = bestEvaluationData,
                    SelectedFeatures = bestSelectedFeatures
                };

                OptimizerHelper.UpdateAndApplyBestSolution(
                    currentResult,
                    ref bestResult,
                    XTrainSubset,
                    XTestSubset,
                    XValSubset,
                    fitnessCalculator
                );

                population[i] = trial;
            }

            if (UpdateIterationHistoryAndCheckEarlyStopping(fitnessHistory, iterationHistory, generation, bestFitness, bestFitDetectionResult, fitnessCalculator))
            {
                break; // Early stopping criteria met, exit the loop
            }
        }

        return OptimizerHelper.CreateOptimizationResult(
            bestSolution ?? SymbolicModelFactory<T>.CreateEmptyModel(_options.UseExpressionTrees, XTrain.Columns, _numOps),
            bestFitness,
            fitnessHistory,
            bestSelectedFeatures,
            new OptimizationResult<T>.DatasetResult
            {
                X = bestTrainingFeatures,
                Y = yTrain,
                Predictions = bestTrainingPredictions,
                ErrorStats = bestEvaluationData.TrainingErrorStats,
                ActualBasicStats = bestEvaluationData.TrainingActualBasicStats,
                PredictedBasicStats = bestEvaluationData.TrainingPredictedBasicStats,
                PredictionStats = bestEvaluationData.TrainingPredictionStats
            },
            new OptimizationResult<T>.DatasetResult
            {
                X = bestValidationFeatures,
                Y = yVal,
                Predictions = bestValidationPredictions,
                ErrorStats = bestEvaluationData.ValidationErrorStats,
                ActualBasicStats = bestEvaluationData.ValidationActualBasicStats,
                PredictedBasicStats = bestEvaluationData.ValidationPredictedBasicStats,
                PredictionStats = bestEvaluationData.ValidationPredictionStats
            },
            new OptimizationResult<T>.DatasetResult
            {
                X = bestTestFeatures,
                Y = yTest,
                Predictions = bestTestPredictions,
                ErrorStats = bestEvaluationData.TestErrorStats,
                ActualBasicStats = bestEvaluationData.TestActualBasicStats,
                PredictedBasicStats = bestEvaluationData.TestPredictedBasicStats,
                PredictionStats = bestEvaluationData.TestPredictionStats
            },
            bestFitDetectionResult,
            fitnessHistory.Count,
            _numOps);
    }

    private List<ISymbolicModel<T>> InitializePopulation(int dimensions, int populationSize)
    {
        var population = new List<ISymbolicModel<T>>();
        for (int i = 0; i < populationSize; i++)
        {
            var individual = new Vector<T>(dimensions, _numOps);
            for (int j = 0; j < dimensions; j++)
            {
                individual[j] = _numOps.FromDouble(_random.NextDouble() * 2 - 1); // Random values between -1 and 1
            }
            population.Add(new VectorModel<T>(individual, _numOps));
        }

        return population;
    }

    private ISymbolicModel<T> GenerateTrialModel(List<ISymbolicModel<T>> population, int currentIndex, int dimensions)
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

        var currentModel = population[currentIndex];
        var trialVector = new Vector<T>(dimensions, _numOps);
        int R = _random.Next(dimensions);

        for (int i = 0; i < dimensions; i++)
        {
            if (_random.NextDouble() < _deOptions.CrossoverRate || i == R)
            {
                var aValue = ((VectorModel<T>)population[a]).Coefficients[i];
                var bValue = ((VectorModel<T>)population[b]).Coefficients[i];
                var cValue = ((VectorModel<T>)population[c]).Coefficients[i];

                trialVector[i] = _numOps.Add(aValue,
                    _numOps.Multiply(_numOps.FromDouble(_deOptions.MutationFactor),
                        _numOps.Subtract(bValue, cValue)));
            }
            else
            {
                trialVector[i] = ((VectorModel<T>)currentModel).Coefficients[i];
            }
        }

        return new VectorModel<T>(trialVector, _numOps);
    }
}