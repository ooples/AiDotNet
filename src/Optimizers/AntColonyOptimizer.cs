namespace AiDotNet.Optimizers;

public class AntColonyOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private AntColonyOptimizationOptions _antColonyOptions;

    public AntColonyOptimizer(AntColonyOptimizationOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator)
    {
        _random = new Random();
        _antColonyOptions = options ?? new AntColonyOptimizationOptions();
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        int dimensions = inputData.XTrain.Columns;
        var pheromones = InitializePheromones(dimensions);
        var bestStepData = new OptimizationStepData<T>();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            var solutions = new List<ISymbolicModel<T>>();

            for (int ant = 0; ant < _antColonyOptions.AntCount; ant++)
            {
                var solution = ConstructSolution(pheromones, inputData.XTrain);
                solutions.Add(solution);

                var currentStepData = PrepareAndEvaluateSolution(solution, inputData);
                UpdateBestSolution(currentStepData, ref bestStepData);
            }

            UpdatePheromones(pheromones, solutions);
            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break; // Early stopping criteria met, exit the loop
            }
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private Matrix<T> InitializePheromones(int dimensions)
    {
        var pheromones = new Matrix<T>(dimensions, dimensions, NumOps);
        for (int i = 0; i < dimensions; i++)
        {
            for (int j = 0; j < dimensions; j++)
            {
                pheromones[i, j] = NumOps.One;
            }
        }

        return pheromones;
    }

private ISymbolicModel<T> ConstructSolution(Matrix<T> pheromones, Matrix<T> XTrain)
{
    var dimensions = XTrain.Columns;
    var model = SymbolicModelFactory<T>.CreateEmptyModel(Options.UseExpressionTrees, dimensions, NumOps);
    var visited = new bool[dimensions];
    int current = _random.Next(dimensions);
    visited[current] = true;

    for (int i = 0; i < dimensions; i++)
    {
        int next = SelectNextFeature(current, pheromones, XTrain, visited);
        if (next == -1) break; // No more unvisited features

        T coefficient = NumOps.FromDouble(_random.NextDouble() * 2 - 1); // Random value between -1 and 1

        if (model is VectorModel<T> vectorModel)
        {
            vectorModel.Coefficients[next] = coefficient;
        }
        else if (model is ExpressionTree<T> expressionTree)
        {
            var featureNode = new ExpressionTree<T>(NodeType.Variable, NumOps.FromDouble(next));
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
        var probabilities = new Vector<T>(XTrain.Columns, NumOps);
        T total = NumOps.Zero;

        for (int i = 0; i < XTrain.Columns; i++)
        {
            if (!visited[i])
            {
                T pheromone = pheromones[current, i];
                T heuristic = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Abs(XTrain[current, i])));
                T probability = NumOps.Multiply(
                    NumOps.Power(pheromone, NumOps.FromDouble(_antColonyOptions.Alpha)),
                    NumOps.Power(heuristic, NumOps.FromDouble(_antColonyOptions.Beta))
                );
                probabilities[i] = probability;
                total = NumOps.Add(total, probability);
            }
        }

        T random = NumOps.Multiply(NumOps.FromDouble(_random.NextDouble()), total);
        T sum = NumOps.Zero;
        for (int i = 0; i < XTrain.Columns; i++)
        {
            if (!visited[i])
            {
                sum = NumOps.Add(sum, probabilities[i]);
                if (NumOps.GreaterThanOrEquals(sum, random))
                {
                    return i;
                }
            }
        }

        return -1; // This should never happen
    }

    private void UpdatePheromones(Matrix<T> pheromones, List<ISymbolicModel<T>> solutions)
    {
        // Evaporation
        for (int i = 0; i < pheromones.Rows; i++)
        {
            for (int j = 0; j < pheromones.Columns; j++)
            {
                pheromones[i, j] = NumOps.Multiply(pheromones[i, j], NumOps.FromDouble(1 - _antColonyOptions.EvaporationRate));
            }
        }

        // Deposit
        for (int k = 0; k < solutions.Count; k++)
        {
            T deposit = NumOps.Divide(NumOps.One, _fitnessList[k]);
            var model = solutions[k];

            if (model is VectorModel<T> vectorModel)
            {
                for (int i = 0; i < vectorModel.Coefficients.Length; i++)
                {
                    for (int j = 0; j < vectorModel.Coefficients.Length; j++)
                    {
                        if (i != j)
                        {
                            pheromones[i, j] = NumOps.Add(pheromones[i, j], NumOps.Multiply(deposit, NumOps.Abs(vectorModel.Coefficients[i])));
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
            int variableIndex = NumOps.ToInt32(tree.Value);
            for (int j = 0; j < pheromones.Columns; j++)
            {
                if (variableIndex != j)
                {
                    pheromones[variableIndex, j] = NumOps.Add(pheromones[variableIndex, j], deposit);
                }
            }
        }

        UpdatePheromonesForExpressionTree(pheromones, tree.Left, deposit);
        UpdatePheromonesForExpressionTree(pheromones, tree.Right, deposit);
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is AntColonyOptimizationOptions antColonyOptions)
        {
            _antColonyOptions = antColonyOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AntColonyOptimizerOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _antColonyOptions;
    }

    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize AntColonyOptimizerOptions
            string optionsJson = JsonConvert.SerializeObject(_antColonyOptions);
            writer.Write(optionsJson);

            return ms.ToArray();
        }
    }

    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize AntColonyOptimizerOptions
            string optionsJson = reader.ReadString();
            _antColonyOptions = JsonConvert.DeserializeObject<AntColonyOptimizationOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }
}