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
        Matrix<T> XTrain, Vector<T> yTrain,
        Matrix<T> XVal, Vector<T> yVal,
        Matrix<T> XTest, Vector<T> yTest,
        IFullModel<T> regressionMethod,
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
        var bestSolution = SymbolicModelFactory<T>.CreateEmptyModel(_options.UseExpressionTrees, XTrain.Columns, _numOps);
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
                var selectedFeatures = OptimizerHelper.GetSelectedFeatures(currentSolution);
                var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
                var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
                var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

                var (currentFitness, fitDetectionResult, trainingPredictions, validationPredictions, testPredictions, evaluationData) = 
                EvaluateSolution(currentSolution,
                    XTrainSubset, XValSubset, XTestSubset,
                    yTrain, yVal, yTest,
                    normalizer, normInfo,
                    fitnessCalculator, fitDetector
                );

                int featureCount = selectedFeatures.Count;
                Vector<T>? coefficients = null;
                T? intercept = default;
                bool hasIntercept = false;

                if (regressionMethod is ILinearModel<T> linearModel)
                {
                    hasIntercept = linearModel.HasIntercept;
                    coefficients = linearModel.Coefficients;
                    intercept = linearModel.Intercept;
                    featureCount += hasIntercept ? 1 : 0;
                }

                var currentResult = new ModelResult<T>
                {
                    Solution = currentSolution,
                    Fitness = currentFitness,
                    FitDetectionResult = fitDetectionResult,
                    TrainingPredictions = trainingPredictions,
                    ValidationPredictions = validationPredictions,
                    TestPredictions = testPredictions,
                    EvaluationData = evaluationData,
                    SelectedFeatures = selectedFeatures.ToVectorList<T>()
                };

                var bestResult = new ModelResult<T>
                {
                    Solution = bestSolution,
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
            }

            fitnessHistory.Add(bestFitness);
            iterationHistory.Add(new OptimizationIterationInfo<T> { Iteration = generation, Fitness = bestFitness, FitDetectionResult = bestFitDetectionResult });

            if (ShouldEarlyStop(iterationHistory, fitnessCalculator))
            {
                break;
            }

            population = PerformSelection(population, fitnessCalculator, XTrain, yTrain, XVal, yVal, XTest, yTest, regressionMethod, regularization, fitDetector, normalizer, normInfo);
            population = PerformCrossover(population, crossoverRate);
            population = PerformMutation(population, mutationRate);
        }

        return OptimizerHelper.CreateOptimizationResult(
            bestSolution,
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

    private List<ISymbolicModel<T>> InitializePopulation(int dimensions, int populationSize)
    {
        var population = new List<ISymbolicModel<T>>();
        for (int i = 0; i < populationSize; i++)
        {
            population.Add(SymbolicModelFactory<T>.CreateRandomModel(_options.UseExpressionTrees, dimensions, _numOps));
        }

        return population;
    }

    private List<ISymbolicModel<T>> PerformSelection(List<ISymbolicModel<T>> population, IFitnessCalculator<T> fitnessCalculator, 
    Matrix<T> XTrain, Vector<T> yTrain, Matrix<T> XVal, Vector<T> yVal, Matrix<T> XTest, Vector<T> yTest, 
    IFullModel<T> regressionMethod, IRegularization<T> regularization, IFitDetector<T> fitDetector,
    INormalizer<T> normalizer, NormalizationInfo<T> normInfo)
    {
        var fitnesses = new List<T>();

        foreach (var individual in population)
        {
            individual.Fit(XTrain, yTrain);
            var trainingPredictions = new Vector<T>(XTrain.Rows, _numOps);
            var validationPredictions = new Vector<T>(XVal.Rows, _numOps);
            var testingPredictions = new Vector<T>(XTest.Rows, _numOps);

            // Evaluate each row individually
            for (int i = 0; i < XTrain.Rows; i++)
            {
                trainingPredictions[i] = individual.Evaluate(XTrain.GetRow(i));
            }
            for (int i = 0; i < XVal.Rows; i++)
            {
                validationPredictions[i] = individual.Evaluate(XVal.GetRow(i));
            }
            for (int i = 0; i < XTest.Rows; i++)
            {
                testingPredictions[i] = individual.Evaluate(XTest.GetRow(i));
            }

            var (trainingErrorStats, validationErrorStats, _) = CalculateErrorStats(
                yTrain, yVal, yTest, 
                trainingPredictions, validationPredictions, testingPredictions, 
                XTrain.Columns);

            var (trainingActualStats, trainingPredictedStats, 
                 validationActualStats, validationPredictedStats, _, _) = CalculateBasicStats(
                yTrain, yVal, yTest,
                trainingPredictions, validationPredictions, testingPredictions);

            var (trainingPredictionStats, validationPredictionStats, _) = CalculatePredictionStats(
                yTrain, yVal, yTest, 
                trainingPredictions, validationPredictions, testingPredictions, 
                XTrain.Columns);

            var currentFitness = fitnessCalculator.CalculateFitnessScore(
                trainingErrorStats,
                trainingActualStats,
                trainingPredictedStats,
                yTrain,
                trainingPredictions,
                XTrain,
                trainingPredictionStats
            );

            fitnesses.Add(currentFitness);
        }

        return TournamentSelection(population, fitnesses, population.Count);
    }

    private List<ISymbolicModel<T>> TournamentSelection(List<ISymbolicModel<T>> population, List<T> fitnesses, int selectionSize)
    {
        var selected = new List<ISymbolicModel<T>>();
        for (int i = 0; i < selectionSize; i++)
        {
            int index1 = _random.Next(population.Count);
            int index2 = _random.Next(population.Count);
            selected.Add(_numOps.LessThan(fitnesses[index1], fitnesses[index2]) ? population[index1] : population[index2]);
        }

        return selected;
    }

    private List<ISymbolicModel<T>> PerformCrossover(List<ISymbolicModel<T>> population, double crossoverRate)
    {
        var newPopulation = new List<ISymbolicModel<T>>();
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

    private (ISymbolicModel<T>, ISymbolicModel<T>) Crossover(ISymbolicModel<T> parent1, ISymbolicModel<T> parent2, double crossoverRate)
    {
        if (_random.NextDouble() < crossoverRate)
        {
            return SymbolicModelFactory<T>.Crossover(parent1, parent2, crossoverRate, _numOps);
        }
        else
        {
            return (parent1, parent2);
        }
    }

    private List<ISymbolicModel<T>> PerformMutation(List<ISymbolicModel<T>> population, double mutationRate)
    {
        return [.. population.Select(individual => Mutate(individual, mutationRate))];
    }

    private ISymbolicModel<T> Mutate(ISymbolicModel<T> individual, double mutationRate)
    {
        if (_random.NextDouble() < mutationRate)
        {
            return SymbolicModelFactory<T>.Mutate(individual, mutationRate, _numOps);
        }

        return individual;
    }
}