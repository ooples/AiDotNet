namespace AiDotNet.Optimizers;

public class ParticleSwarmOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private readonly ParticleSwarmOptimizationOptions _psoOptions;

    public ParticleSwarmOptimizer(ParticleSwarmOptimizationOptions? options = null)
        : base(options)
    {
        _random = new Random();
        _psoOptions = options ?? new ParticleSwarmOptimizationOptions();
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
        var swarm = InitializeSwarm(XTrain.Columns, _psoOptions.SwarmSize);
        var velocities = InitializeVelocities(XTrain.Columns, _psoOptions.SwarmSize);
        T bestFitness = fitnessCalculator.IsHigherScoreBetter ? _numOps.MinValue : _numOps.MaxValue;
        var bestSolution = SymbolicModelFactory<T>.CreateEmptyModel(_options.UseExpressionTrees, XTrain.Columns, _numOps);
        T bestIntercept = _numOps.Zero;
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

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            for (int i = 0; i < _psoOptions.SwarmSize; i++)
            {
                var particle = swarm[i];
                var selectedFeatures = OptimizerHelper.GetSelectedFeatures(particle);
                var XTrainSubset = OptimizerHelper.SelectFeatures(XTrain, selectedFeatures);
                var XValSubset = OptimizerHelper.SelectFeatures(XVal, selectedFeatures);
                var XTestSubset = OptimizerHelper.SelectFeatures(XTest, selectedFeatures);

                var (currentFitnessScore, fitDetectionResult, trainingPredictions, validationPredictions, testPredictions, evaluationData) = EvaluateSolution(
                    particle, XTrainSubset, XValSubset, XTestSubset,
                    yTrain, yVal, yTest,
                    normalizer, normInfo,
                    fitnessCalculator, fitDetector);

                var currentResult = new ModelResult<T>
                {
                    Solution = particle,
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

                // Update velocity and position
                if (_options.UseExpressionTrees)
                {
                    UpdateParticle((ExpressionTree<T>)swarm[i], (ExpressionTreeVelocity<T>)velocities[i], (ExpressionTree<T>)particle, (ExpressionTree<T>)bestSolution);
                }
                else
                {
                    UpdateParticle((Vector<T>)swarm[i], (Vector<T>)velocities[i], (Vector<T>)particle, (Vector<T>)bestSolution);
                }
            }

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

    private List<ISymbolicModel<T>> InitializeSwarm(int dimensions, int swarmSize)
    {
        var swarm = new List<ISymbolicModel<T>>();
        for (int i = 0; i < swarmSize; i++)
        {
            var particle = SymbolicModelFactory<T>.CreateRandomModel(_options.UseExpressionTrees, dimensions, _numOps);
        
            // If using VectorModel, we need to initialize it with random values
            if (particle is VectorModel<T> vectorModel)
            {
                for (int j = 0; j < dimensions; j++)
                {
                    vectorModel.Coefficients[j] = _numOps.FromDouble(_random.NextDouble() * 2 - 1); // Random values between -1 and 1
                }
            }
            // If using ExpressionTree, it's already randomly initialized by the factory

            swarm.Add(particle);
        }

        return swarm;
    }

    private List<object> InitializeVelocities(int dimensions, int swarmSize)
    {
        var velocities = new List<object>();
        for (int i = 0; i < swarmSize; i++)
        {
            if (_options.UseExpressionTrees)
            {
                velocities.Add(new ExpressionTreeVelocity<T>());
            }
            else
            {
                var vectorVelocity = new Vector<T>(dimensions, _numOps);
                for (int j = 0; j < dimensions; j++)
                {
                    vectorVelocity[j] = _numOps.FromDouble(_random.NextDouble() * 0.1 - 0.05); // Small random values
                }
                velocities.Add(vectorVelocity);
            }
        }

        return velocities;
    }

    private void UpdateParticle(Vector<T> position, Vector<T> velocity, Vector<T> personalBest, Vector<T> globalBest)
    {
        for (int j = 0; j < position.Length; j++)
        {
            T r1 = _numOps.FromDouble(_random.NextDouble());
            T r2 = _numOps.FromDouble(_random.NextDouble());

            velocity[j] = _numOps.Add(
                _numOps.Multiply(_numOps.FromDouble(_psoOptions.InertiaWeight), velocity[j]),
                _numOps.Add(
                    _numOps.Multiply(_numOps.FromDouble(_psoOptions.CognitiveParameter), _numOps.Multiply(r1, _numOps.Subtract(personalBest[j], position[j]))),
                    _numOps.Multiply(_numOps.FromDouble(_psoOptions.SocialParameter), _numOps.Multiply(r2, _numOps.Subtract(globalBest[j], position[j])))
                )
            );

            position[j] = _numOps.Add(position[j], velocity[j]);
        }
    }

    private void UpdateParticle(ExpressionTree<T> position, ExpressionTreeVelocity<T> velocity, ExpressionTree<T> personalBest, ExpressionTree<T> globalBest)
    {
        // Update node values
        foreach (var nodeChange in velocity.NodeValueChanges)
        {
            var node = position.FindNodeById(nodeChange.Key);
            if (node != null && node.Type == NodeType.Constant)
            {
                node.SetValue(_numOps.Add(node.Value, nodeChange.Value));
            }
        }

        // Apply structure changes
        foreach (var change in velocity.StructureChanges)
        {
            ApplyStructureChange(position, change);
        }

        // Calculate new velocity
        var newVelocity = new ExpressionTreeVelocity<T>();

        // Update node values
        var allNodes = position.GetAllNodes();
        foreach (var node in allNodes)
        {
            if (node.Type == NodeType.Constant)
            {
                T r1 = _numOps.FromDouble(_random.NextDouble());
                T r2 = _numOps.FromDouble(_random.NextDouble());

                T inertia = _numOps.Multiply(
                    _numOps.FromDouble(_psoOptions.InertiaWeight), 
                    velocity.NodeValueChanges.TryGetValue(node.Id, out var existingVelocity) ? existingVelocity : _numOps.Zero
                );

                T cognitive = _numOps.Multiply(
                    _numOps.FromDouble(_psoOptions.CognitiveParameter), 
                    _numOps.Multiply(r1, _numOps.Subtract(GetNodeValueOrDefault(personalBest, node.Id), node.Value))
                );

                T social = _numOps.Multiply(
                    _numOps.FromDouble(_psoOptions.SocialParameter), 
                    _numOps.Multiply(r2, _numOps.Subtract(GetNodeValueOrDefault(globalBest, node.Id), node.Value))
                );

                newVelocity.NodeValueChanges[node.Id] = _numOps.Add(_numOps.Add(inertia, cognitive), social);
            }
        }

        // Update structure changes
        if (_random.NextDouble() < 0.1) // 10% chance of structure change
        {
            newVelocity.StructureChanges.Add(GenerateRandomStructureChange());
        }

        // Apply new velocity
        velocity.NodeValueChanges = newVelocity.NodeValueChanges;
        velocity.StructureChanges = newVelocity.StructureChanges;
    }

    private T GetNodeValueOrDefault(ExpressionTree<T> tree, int nodeId)
    {
        var node = tree.FindNodeById(nodeId);
        return node != null ? node.Value : _numOps.Zero;
    }

    private void ApplyStructureChange(ExpressionTree<T> tree, NodeModification change)
    {
        var node = tree.FindNodeById(change.NodeId);
        if (node == null) return;

        switch (change.Type)
        {
            case ModificationType.AddNode:
                var newNode = new ExpressionTree<T>((NodeType)_random.Next(0, 6), _numOps.FromDouble(_random.NextDouble()));
                if (_random.NextDouble() < 0.5)
                    node.SetLeft(newNode);
                else
                    node.SetRight(newNode);
                break;
            case ModificationType.RemoveNode:
                if (node.Parent != null)
                {
                    if (node.Parent.Left == node)
                        node.Parent.SetLeft(null);
                    else
                        node.Parent.SetRight(null);
                }
                break;
            case ModificationType.ChangeNodeType:
                if (change.NewNodeType.HasValue)
                    node.SetType(change.NewNodeType.Value);
                break;
        }
    }

    private NodeModification GenerateRandomStructureChange()
    {
        return new NodeModification
        {
            NodeId = _random.Next(0, 1000), // Assume max 1000 nodes
            Type = (ModificationType)_random.Next(0, 3),
            NewNodeType = _random.NextDouble() < 0.5 ? null : (NodeType)_random.Next(0, 6)
        };
    }
}