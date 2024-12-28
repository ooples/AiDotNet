namespace AiDotNet.Optimizers;

public class TabuSearchOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private readonly TabuSearchOptions _tabuOptions;

    public TabuSearchOptimizer(TabuSearchOptions? options = null) : base(options)
    {
        _random = new Random();
        _tabuOptions = options ?? new TabuSearchOptions();
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
        var currentSolution = InitializeRandomSolution(XTrain.Columns);
        var bestSolution = SymbolicModelFactory<T>.CreateEmptyModel(_options.UseExpressionTrees, XTrain.Columns, _numOps);
        var tabuList = new Queue<ISymbolicModel<T>>(_tabuOptions.TabuListSize);
        T bestIntercept = _numOps.Zero;
        T bestFitness = fitnessCalculator.IsHigherScoreBetter ? _numOps.MinValue : _numOps.MaxValue;
        FitDetectorResult<T> bestFitDetectionResult = new();
        Vector<T> bestTrainingPredictions = Vector<T>.Empty();
        Vector<T> bestValidationPredictions = Vector<T>.Empty();
        Vector<T> bestTestPredictions = Vector<T>.Empty();
        ModelEvaluationData<T> bestEvaluationData = new();
        List<Vector<T>> bestSelectedFeatures = [];
        Matrix<T> bestTestFeatures = Matrix<T>.Empty();
        Matrix<T> bestTrainingFeatures = Matrix<T>.Empty();
        Matrix<T> bestValidationFeatures = Matrix<T>.Empty();

        var fitnessHistory = new List<T>();
        var iterationHistory = new List<OptimizationIterationInfo<T>>();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var neighbors = GenerateNeighbors(currentSolution);
            var bestNeighbor = neighbors
                .Where(n => !IsTabu(n, tabuList))
                .OrderByDescending(n => EvaluateSolutionFitness(n, XTrain, yTrain, XVal, yVal, XTest, yTest, regressionMethod, regularization, normalizer, normInfo, fitnessCalculator, fitDetector))
                .FirstOrDefault() ?? neighbors.First();

            currentSolution = bestNeighbor;

            var selectedFeatures = OptimizerHelper.GetSelectedFeatures(currentSolution);
            var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
            var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
            var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

            var (currentFitnessScore, fitDetectionResult, trainingPredictions, validationPredictions, testPredictions, evaluationData) = EvaluateSolution(
                        currentSolution, XTrainSubset, XValSubset, XTestSubset,
                        yTrain, yVal, yTest,
                        normalizer, normInfo,
                        fitnessCalculator, fitDetector);

            var currentResult = new ModelResult<T>
            {
                Solution = currentSolution,
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

            UpdateTabuList(tabuList, currentSolution);

            if (UpdateIterationHistoryAndCheckEarlyStopping(fitnessHistory, iterationHistory, iteration, bestFitness, bestFitDetectionResult, fitnessCalculator))
            {
                break; // Early stopping criteria met, exit the loop
            }
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
            fitnessHistory.Count,
            _numOps);
    }

    private ISymbolicModel<T> InitializeRandomSolution(int dimensions)
    {
        if (_options.UseExpressionTrees)
        {
            return GenerateRandomExpressionTree(dimensions, 3); // 3 is the max depth, adjust as needed
        }
        else
        {
            var solution = new Vector<T>(dimensions);
            for (int i = 0; i < dimensions; i++)
            {
                solution[i] = _numOps.FromDouble(_random.NextDouble());
            }
            return new VectorModel<T>(solution, _numOps);
        }
    }

    private List<ISymbolicModel<T>> GenerateNeighbors(ISymbolicModel<T> currentSolution)
    {
        var neighbors = new List<ISymbolicModel<T>>();
        for (int i = 0; i < _tabuOptions.NeighborhoodSize; i++)
        {
            if (currentSolution is ExpressionTree<T> expressionTree)
            {
                neighbors.Add((ExpressionTree<T>)expressionTree.Mutate(_tabuOptions.MutationRate, _numOps));
            }
            else if (currentSolution is VectorModel<T> vectorModel)
            {
                var neighbor = vectorModel.Coefficients.Copy();
                int index = _random.Next(neighbor.Length);
                neighbor[index] = _numOps.Add(neighbor[index], _numOps.FromDouble(_random.NextDouble() * _tabuOptions.PerturbationFactor - _tabuOptions.PerturbationFactor / 2));
                neighbors.Add(new VectorModel<T>(neighbor, _numOps));
            }
        }
        return neighbors;
    }

    private ExpressionTree<T> GenerateRandomExpressionTree(int maxVariables, int maxDepth)
    {
        return GenerateRandomExpressionTreeRecursive(maxVariables, maxDepth, 0);
    }

    private ExpressionTree<T> GenerateRandomExpressionTreeRecursive(int maxVariables, int maxDepth, int currentDepth)
    {
        if (currentDepth == maxDepth || _random.NextDouble() < 0.3) // 30% chance to stop early
        {
            if (_random.NextDouble() < 0.5)
            {
                // Generate a constant
                return new ExpressionTree<T>(NodeType.Constant, _numOps.FromDouble(_random.NextDouble() * 10 - 5));
            }
            else
            {
                // Generate a variable
                return new ExpressionTree<T>(NodeType.Variable, _numOps.FromDouble(_random.Next(maxVariables)));
            }
        }
        else
        {
            // Generate an operation node
            NodeType nodeType = (NodeType)_random.Next(2, 6); // Add, Subtract, Multiply, or Divide
            var left = GenerateRandomExpressionTreeRecursive(maxVariables, maxDepth, currentDepth + 1);
            var right = GenerateRandomExpressionTreeRecursive(maxVariables, maxDepth, currentDepth + 1);
            return new ExpressionTree<T>(nodeType, default, left, right);
        }
    }

    private bool IsTabu(ISymbolicModel<T> solution, Queue<ISymbolicModel<T>> tabuList)
    {
        return tabuList.Any(tabuSolution => tabuSolution.Equals(solution));
    }

    private void UpdateTabuList(Queue<ISymbolicModel<T>> tabuList, ISymbolicModel<T> solution)
    {
        if (tabuList.Count >= _tabuOptions.TabuListSize)
        {
            tabuList.Dequeue();
        }

        tabuList.Enqueue(solution);
    }

    private T EvaluateSolutionFitness(ISymbolicModel<T> solution, Matrix<T> XTrain, Vector<T> yTrain, Matrix<T> XVal, Vector<T> yVal, Matrix<T> XTest, Vector<T> yTest,
        IFullModel<T> regressionMethod, IRegularization<T> regularization, INormalizer<T> normalizer, NormalizationInfo<T> normInfo,
        IFitnessCalculator<T> fitnessCalculator, IFitDetector<T> fitDetector)
    {
        var selectedFeatures = OptimizerHelper.GetSelectedFeatures(solution);
        var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
        var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
        var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

        var (currentFitnessScore, fitDetectionResult, trainingPredictions, validationPredictions, testPredictions, evaluationData) = EvaluateSolution(
                        solution, XTrainSubset, XValSubset, XTestSubset,
                        yTrain, yVal, yTest,
                        normalizer, normInfo,
                        fitnessCalculator, fitDetector);

        return currentFitnessScore;
    }
}