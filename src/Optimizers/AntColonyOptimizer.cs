using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.Optimizers;

public class AntColonyOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private readonly AntColonyOptimizationOptions _antColonyOptions;

    public AntColonyOptimizer(AntColonyOptimizationOptions? options = null)
        : base(options)
    {
        _random = new Random();
        _antColonyOptions = options ?? new AntColonyOptimizationOptions();
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
        var pheromones = InitializePheromones(dimensions);
        var bestSolution = SymbolicModelFactory<T>.CreateEmptyModel(_options.UseExpressionTrees, XTrain.Columns, _numOps);
        T bestIntercept = _numOps.Zero;
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
        List<int> bestSelectedFeatureIndices = [];

        var fitnessHistory = new List<T>();
        var iterationHistory = new List<OptimizationIterationInfo<T>>();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var solutions = new List<ISymbolicModel<T>>();
            var fitnesses = new List<T>();

            for (int ant = 0; ant < _antColonyOptions.AntCount; ant++)
            {
                var solution = ConstructSolution(pheromones, XTrain);
                solutions.Add(solution);

                var selectedFeatures = OptimizerHelper.GetSelectedFeatures(solution);
                var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
                var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
                var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

                var (currentFitnessScore, fitDetectionResult, 
                 trainingPredictions, validationPredictions, testPredictions, evaluationData) = 
                EvaluateSolution(
                    solution, XTrainSubset, XValSubset, XTestSubset,
                    yTrain, yVal, yTest,
                    normalizer, normInfo,
                    fitnessCalculator, fitDetector);

                fitnesses.Add(currentFitnessScore);

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
                    Solution = solution,
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
            }

            UpdatePheromones(pheromones, solutions, fitnesses);
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
            iterationHistory.Count,
            _numOps
        );
    }

    private Matrix<T> InitializePheromones(int dimensions)
    {
        var pheromones = new Matrix<T>(dimensions, dimensions, _numOps);
        for (int i = 0; i < dimensions; i++)
        {
            for (int j = 0; j < dimensions; j++)
            {
                pheromones[i, j] = _numOps.One;
            }
        }
        return pheromones;
    }

private ISymbolicModel<T> ConstructSolution(Matrix<T> pheromones, Matrix<T> XTrain)
{
    var dimensions = XTrain.Columns;
    var model = SymbolicModelFactory<T>.CreateEmptyModel(_options.UseExpressionTrees, dimensions, _numOps);
    var visited = new bool[dimensions];
    int current = _random.Next(dimensions);
    visited[current] = true;

    for (int i = 0; i < dimensions; i++)
    {
        int next = SelectNextFeature(current, pheromones, XTrain, visited);
        if (next == -1) break; // No more unvisited features

        T coefficient = _numOps.FromDouble(_random.NextDouble() * 2 - 1); // Random value between -1 and 1

        if (model is VectorModel<T> vectorModel)
        {
            vectorModel.Coefficients[next] = coefficient;
        }
        else if (model is ExpressionTree<T> expressionTree)
        {
            var featureNode = new ExpressionTree<T>(NodeType.Variable, _numOps.FromDouble(next));
            var coefficientNode = new ExpressionTree<T>(NodeType.Constant, coefficient);
            var termNode = new ExpressionTree<T>(NodeType.Multiply, default, coefficientNode, featureNode);

            if (i == 0)
            {
                expressionTree.SetLeft(termNode);
            }
            else
            {
                expressionTree.SetType(NodeType.Add);
                expressionTree.SetRight(termNode);
                var newRoot = new ExpressionTree<T>(NodeType.Add, default, expressionTree, null);
                expressionTree = newRoot;
            }
        }

        visited[next] = true;
        current = next;
    }

    return model;
}

    private int SelectNextFeature(int current, Matrix<T> pheromones, Matrix<T> XTrain, bool[] visited)
    {
        var probabilities = new Vector<T>(XTrain.Columns, _numOps);
        T total = _numOps.Zero;

        for (int i = 0; i < XTrain.Columns; i++)
        {
            if (!visited[i])
            {
                T pheromone = pheromones[current, i];
                T heuristic = _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, _numOps.Abs(XTrain[current, i])));
                T probability = _numOps.Multiply(
                    _numOps.Power(pheromone, _numOps.FromDouble(_antColonyOptions.Alpha)),
                    _numOps.Power(heuristic, _numOps.FromDouble(_antColonyOptions.Beta))
                );
                probabilities[i] = probability;
                total = _numOps.Add(total, probability);
            }
        }

        T random = _numOps.Multiply(_numOps.FromDouble(_random.NextDouble()), total);
        T sum = _numOps.Zero;
        for (int i = 0; i < XTrain.Columns; i++)
        {
            if (!visited[i])
            {
                sum = _numOps.Add(sum, probabilities[i]);
                if (_numOps.GreaterThanOrEquals(sum, random))
                {
                    return i;
                }
            }
        }

        return -1; // This should never happen
    }

    private void UpdatePheromones(Matrix<T> pheromones, List<ISymbolicModel<T>> solutions, List<T> fitnesses)
    {
        // Evaporation
        for (int i = 0; i < pheromones.Rows; i++)
        {
            for (int j = 0; j < pheromones.Columns; j++)
            {
                pheromones[i, j] = _numOps.Multiply(pheromones[i, j], _numOps.FromDouble(1 - _antColonyOptions.EvaporationRate));
            }
        }

        // Deposit
        for (int k = 0; k < solutions.Count; k++)
        {
            T deposit = _numOps.Divide(_numOps.One, fitnesses[k]);
            var model = solutions[k];

            if (model is VectorModel<T> vectorModel)
            {
                for (int i = 0; i < vectorModel.Coefficients.Length; i++)
                {
                    for (int j = 0; j < vectorModel.Coefficients.Length; j++)
                    {
                        if (i != j)
                        {
                            pheromones[i, j] = _numOps.Add(pheromones[i, j], _numOps.Multiply(deposit, _numOps.Abs(vectorModel.Coefficients[i])));
                        }
                    }
                }
            }
            else if (model is ExpressionTree<T> expressionTree)
            {
                UpdatePheromonesForExpressionTree(pheromones, expressionTree, deposit);
            }
        }
    }

    private void UpdatePheromonesForExpressionTree(Matrix<T> pheromones, ExpressionTree<T>? tree, T deposit)
    {
        if (tree == null) return;

        if (tree.Type == NodeType.Variable)
        {
            int variableIndex = _numOps.ToInt32(tree.Value);
            for (int j = 0; j < pheromones.Columns; j++)
            {
                if (variableIndex != j)
                {
                    pheromones[variableIndex, j] = _numOps.Add(pheromones[variableIndex, j], deposit);
                }
            }
        }

        UpdatePheromonesForExpressionTree(pheromones, tree.Left, deposit);
        UpdatePheromonesForExpressionTree(pheromones, tree.Right, deposit);
    }
}