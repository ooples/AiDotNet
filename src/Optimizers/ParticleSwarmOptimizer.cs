namespace AiDotNet.Optimizers;

public class ParticleSwarmOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private ParticleSwarmOptimizationOptions _psoOptions;

    public ParticleSwarmOptimizer(
        ParticleSwarmOptimizationOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator)
    {
        _random = new Random();
        _psoOptions = options ?? new ParticleSwarmOptimizationOptions();
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        int dimensions = inputData.XTrain.Columns;
        var swarm = InitializeSwarm(dimensions, _psoOptions.SwarmSize);
        var velocities = InitializeVelocities(dimensions, _psoOptions.SwarmSize);
        var personalBests = new List<OptimizationStepData<T>>();
        var globalBest = new OptimizationStepData<T>();
        var evaluatedSolutions = new Dictionary<ISymbolicModel<T>, OptimizationStepData<T>>();

        // Initialize personal bests
        for (int i = 0; i < _psoOptions.SwarmSize; i++)
        {
            var stepData = EvaluateParticle(swarm[i], inputData, evaluatedSolutions);
            personalBests.Add(stepData);
            UpdateGlobalBest(stepData, ref globalBest);
        }

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            for (int i = 0; i < _psoOptions.SwarmSize; i++)
            {
                var particle = swarm[i];

                // Update velocity and position
                if (Options.UseExpressionTrees)
                {
                    UpdateParticle((ExpressionTree<T>)particle, (ExpressionTreeVelocity<T>)velocities[i], 
                                   (ExpressionTree<T>)personalBests[i].Solution, (ExpressionTree<T>)globalBest.Solution);
                }
                else
                {
                    UpdateParticle((Vector<T>)particle, (Vector<T>)velocities[i], 
                                   (Vector<T>)personalBests[i].Solution, (Vector<T>)globalBest.Solution);
                }

                // Evaluate the updated particle
                var stepData = EvaluateParticle(particle, inputData, evaluatedSolutions);

                // Update personal best if necessary
                if (_fitnessCalculator.IsBetterFitness(stepData.FitnessScore, personalBests[i].FitnessScore))
                {
                    personalBests[i] = stepData;
                }

                // Update global best if necessary
                UpdateGlobalBest(stepData, ref globalBest);
            }

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, globalBest))
            {
                break;
            }
        }

        return CreateOptimizationResult(globalBest, inputData);
    }

    private OptimizationStepData<T> EvaluateParticle(ISymbolicModel<T> particle, OptimizationInputData<T> inputData, 
        Dictionary<ISymbolicModel<T>, OptimizationStepData<T>> evaluatedSolutions)
    {
        if (evaluatedSolutions.TryGetValue(particle, out var cachedStepData))
        {
            return cachedStepData;
        }

        var stepData = PrepareAndEvaluateSolution(particle, inputData);
        evaluatedSolutions[particle] = stepData;

        return stepData;
    }

    private void UpdateGlobalBest(OptimizationStepData<T> stepData, ref OptimizationStepData<T> globalBest)
    {
        if (globalBest.Solution == null || _fitnessCalculator.IsBetterFitness(stepData.FitnessScore, globalBest.FitnessScore))
        {
            globalBest = stepData;
        }
    }

    private List<ISymbolicModel<T>> InitializeSwarm(int dimensions, int swarmSize)
    {
        var swarm = new List<ISymbolicModel<T>>();
        for (int i = 0; i < swarmSize; i++)
        {
            var particle = SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, dimensions, NumOps);
            
            if (particle is VectorModel<T> vectorModel)
            {
                for (int j = 0; j < dimensions; j++)
                {
                    vectorModel.Coefficients[j] = NumOps.FromDouble(_random.NextDouble() * 2 - 1);
                }
            }

            swarm.Add(particle);
        }

        return swarm;
    }

    private List<object> InitializeVelocities(int dimensions, int swarmSize)
    {
        var velocities = new List<object>();
        for (int i = 0; i < swarmSize; i++)
        {
            if (Options.UseExpressionTrees)
            {
                velocities.Add(new ExpressionTreeVelocity<T>());
            }
            else
            {
                var vectorVelocity = new Vector<T>(dimensions, NumOps);
                for (int j = 0; j < dimensions; j++)
                {
                    vectorVelocity[j] = NumOps.FromDouble(_random.NextDouble() * 0.1 - 0.05);
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
            T r1 = NumOps.FromDouble(_random.NextDouble());
            T r2 = NumOps.FromDouble(_random.NextDouble());

            velocity[j] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(_psoOptions.InertiaWeight), velocity[j]),
                NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(_psoOptions.CognitiveParameter), NumOps.Multiply(r1, NumOps.Subtract(personalBest[j], position[j]))),
                    NumOps.Multiply(NumOps.FromDouble(_psoOptions.SocialParameter), NumOps.Multiply(r2, NumOps.Subtract(globalBest[j], position[j])))
                )
            );

            position[j] = NumOps.Add(position[j], velocity[j]);
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
                node.SetValue(NumOps.Add(node.Value, nodeChange.Value));
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
                T r1 = NumOps.FromDouble(_random.NextDouble());
                T r2 = NumOps.FromDouble(_random.NextDouble());

                T inertia = NumOps.Multiply(
                    NumOps.FromDouble(_psoOptions.InertiaWeight), 
                    velocity.NodeValueChanges.TryGetValue(node.Id, out var existingVelocity) ? existingVelocity : NumOps.Zero
                );

                T cognitive = NumOps.Multiply(
                    NumOps.FromDouble(_psoOptions.CognitiveParameter), 
                    NumOps.Multiply(r1, NumOps.Subtract(GetNodeValueOrDefault(personalBest, node.Id), node.Value))
                );

                T social = NumOps.Multiply(
                    NumOps.FromDouble(_psoOptions.SocialParameter), 
                    NumOps.Multiply(r2, NumOps.Subtract(GetNodeValueOrDefault(globalBest, node.Id), node.Value))
                );

                newVelocity.NodeValueChanges[node.Id] = NumOps.Add(NumOps.Add(inertia, cognitive), social);
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
        return node != null ? node.Value : NumOps.Zero;
    }

    private void ApplyStructureChange(ExpressionTree<T> tree, NodeModification change)
    {
        var node = tree.FindNodeById(change.NodeId);
        if (node == null) return;

        switch (change.Type)
        {
            case ModificationType.AddNode:
                var newNode = new ExpressionTree<T>((NodeType)_random.Next(0, 6), NumOps.FromDouble(_random.NextDouble()));
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

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is ParticleSwarmOptimizationOptions psoOptions)
        {
            _psoOptions = psoOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected ParticleSwarmOptimizationOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _psoOptions;
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

            // Serialize ParticleSwarmOptimizationOptions
            string optionsJson = JsonConvert.SerializeObject(_psoOptions);
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

            // Deserialize ParticleSwarmOptimizationOptions
            string optionsJson = reader.ReadString();
            _psoOptions = JsonConvert.DeserializeObject<ParticleSwarmOptimizationOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }
}