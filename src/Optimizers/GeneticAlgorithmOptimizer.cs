using AiDotNet.Models.Options;

namespace AiDotNet.Optimizers;

public class GeneticAlgorithmOptimizer<T> : OptimizerBase<T>
{
    private readonly GeneticAlgorithmOptions _geneticOptions;
    private readonly Random _random;

    public GeneticAlgorithmOptimizer(GeneticAlgorithmOptions? options = null) : base(options)
    {
        _geneticOptions = options ?? new GeneticAlgorithmOptions();
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
        int populationSize = _geneticOptions.PopulationSize;
        double mutationRate = _geneticOptions.MutationRate;
        double crossoverRate = _geneticOptions.CrossoverRate;
        var population = InitializePopulation(XTrain.Columns, populationSize);
        var bestSolution = Vector<T>.Empty();
        T bestFitness = fitnessCalculator.IsHigherScoreBetter ? _numOps.MinValue : _numOps.MaxValue;
        var bestIntercept = _numOps.Zero;
        var fitnessHistory = new List<T>();
        var iterationHistory = new List<OptimizationIterationInfo<T>>();

        var bestFitDetectionResult = new FitDetectorResult<T>();
        var bestTrainingPredictions = Vector<T>.Empty();
        var bestValidationPredictions = Vector<T>.Empty();
        var bestTestPredictions = Vector<T>.Empty();
        var bestEvaluationData = new ModelEvaluationData<T>();
        var bestSelectedFeatures = new List<Vector<T>>();
        var bestTestFeatures = Matrix<T>.Empty();
        var bestTrainingFeatures = Matrix<T>.Empty();
        var bestValidationFeatures = Matrix<T>.Empty();

        for (int generation = 0; generation < _options.MaxIterations; generation++)
        {
            for (int i = 0; i < populationSize; i++)
            {
                var currentSolution = population[i];

                var (currentFitness, fitDetectionResult, trainingPredictions, validationPredictions, testPredictions, evaluationData) = 
                    EvaluateSolution(
                        XTrain, XVal, XTest,
                        yTrain, yVal, yTest,
                        regressionMethod, normalizer, normInfo,
                        fitnessCalculator, fitDetector, XTrain.Columns
                    );

                UpdateBestSolution(
                    currentFitness,
                    currentSolution,
                    regressionMethod.Intercept,
                    fitDetectionResult,
                    trainingPredictions,
                    validationPredictions,
                    testPredictions,
                    evaluationData,
                    [.. Enumerable.Range(0, XTrain.Columns)],
                    XTrain,
                    XTest,
                    XTrain,
                    XVal,
                    fitnessCalculator,
                    ref bestFitness,
                    ref bestSolution,
                    ref bestIntercept,
                    ref bestFitDetectionResult,
                    ref bestTrainingPredictions,
                    ref bestValidationPredictions,
                    ref bestTestPredictions,
                    ref bestEvaluationData,
                    ref bestSelectedFeatures,
                    ref bestTestFeatures,
                    ref bestTrainingFeatures,
                    ref bestValidationFeatures
                );
            }

            fitnessHistory.Add(bestFitness);
            iterationHistory.Add(new OptimizationIterationInfo<T> { Iteration = generation, Fitness = bestFitness, FitDetectionResult = bestFitDetectionResult });

            if (ShouldEarlyStop(iterationHistory, fitnessCalculator))
            {
                break;
            }

            population = PerformSelection(population, fitnessCalculator, XTrain, yTrain, XVal, yVal, regressionMethod, regularization, fitDetector, normalizer, normInfo);
            population = PerformCrossover(population, crossoverRate);
            population = PerformMutation(population, mutationRate);
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
            iterationHistory.Count,
            _numOps
        );
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

    private List<Vector<T>> PerformSelection(List<Vector<T>> population, IFitnessCalculator<T> fitnessCalculator, 
        Matrix<T> XTrain, Vector<T> yTrain, Matrix<T> XVal, Vector<T> yVal, 
        IRegression<T> regressionMethod, IRegularization<T> regularization, IFitDetector<T> fitnessDetector,
        INormalizer<T> normalizer, NormalizationInfo<T> normInfo)
    {
        var fitnesses = population.Select(individual => 
            EvaluateSolution(XTrain, XVal, XVal, yTrain, yVal, yVal, 
                regressionMethod, normalizer, normInfo, 
                fitnessCalculator, fitnessDetector, XTrain.Columns).CurrentFitnessScore).ToList();

        return TournamentSelection(population, fitnesses, population.Count);
    }

    private List<Vector<T>> TournamentSelection(List<Vector<T>> population, List<T> fitnesses, int selectionSize)
    {
        var selected = new List<Vector<T>>();
        for (int i = 0; i < selectionSize; i++)
        {
            int index1 = _random.Next(population.Count);
            int index2 = _random.Next(population.Count);
            selected.Add(_numOps.LessThan(fitnesses[index1], fitnesses[index2]) ? population[index1] : population[index2]);
        }
        return selected;
    }

    private List<Vector<T>> PerformCrossover(List<Vector<T>> population, double crossoverRate)
    {
        var newPopulation = new List<Vector<T>>();
        for (int i = 0; i < population.Count; i += 2)
        {
            var parent1 = population[i];
            var parent2 = i + 1 < population.Count ? population[i + 1] : population[0];

            var (child1, child2) = Crossover(parent1, parent2, crossoverRate);
            newPopulation.Add(child1);
            newPopulation.Add(child2);
        }
        return newPopulation;
    }

    private (Vector<T>, Vector<T>) Crossover(Vector<T> parent1, Vector<T> parent2, double crossoverRate)
    {
        if (_random.NextDouble() < crossoverRate)
        {
            int crossoverPoint = _random.Next(parent1.Length);
            var child1 = new Vector<T>(parent1.Length);
            var child2 = new Vector<T>(parent1.Length);

            for (int i = 0; i < parent1.Length; i++)
            {
                if (i < crossoverPoint)
                {
                    child1[i] = parent1[i];
                    child2[i] = parent2[i];
                }
                else
                {
                    child1[i] = parent2[i];
                    child2[i] = parent1[i];
                }
            }

            return (child1, child2);
        }
        else
        {
            return (parent1, parent2);
        }
    }

    private List<Vector<T>> PerformMutation(List<Vector<T>> population, double mutationRate)
    {
        return population.Select(individual => Mutate(individual, mutationRate)).ToList();
    }

    private Vector<T> Mutate(Vector<T> individual, double mutationRate)
    {
        var mutated = individual.Copy();
        for (int i = 0; i < mutated.Length; i++)
        {
            if (_random.NextDouble() < mutationRate)
            {
                mutated[i] = _numOps.Add(mutated[i], _numOps.FromDouble(_random.NextDouble() * 0.2 - 0.1)); // Small random change
            }
        }

        return mutated;
    }
}