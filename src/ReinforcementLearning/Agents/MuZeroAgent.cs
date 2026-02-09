using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.ReinforcementLearning.Agents.MuZero;

/// <summary>
/// MuZero agent combining tree search with learned models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MuZero combines tree search (like AlphaZero) with learned dynamics.
/// It masters games without knowing the rules, learning its own internal model.
/// </para>
/// <para><b>For Beginners:</b>
/// MuZero is DeepMind's breakthrough that achieved superhuman performance in
/// Atari, Go, Chess, and Shogi without being told the rules. It learns its own
/// "internal model" of the game and uses tree search to plan ahead.
///
/// Three key networks:
/// - **Representation**: Observation -> hidden state
/// - **Dynamics**: (hidden state, action) -> (next hidden state, reward)
/// - **Prediction**: hidden state -> (policy, value)
///
/// Plus tree search (MCTS) for planning using the learned model.
///
/// Think of it as: Learning chess by watching games, figuring out the rules
/// yourself, then planning moves by mentally simulating the game.
///
/// Famous for: Superhuman Atari/board games without knowing rules
/// </para>
/// </remarks>
public class MuZeroAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private MuZeroOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    // Three core networks
    private NeuralNetwork<T> _representationNetwork;  // h = f(observation)
    private NeuralNetwork<T> _dynamicsNetwork;  // (h', r) = g(h, action)
    private NeuralNetwork<T> _predictionNetwork;  // (p, v) = f(h)

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private int _updateCount;

    public MuZeroAgent(MuZeroOptions<T> options) : base(new ReinforcementLearningOptions<T>
    {
        LearningRate = options.LearningRate,
        DiscountFactor = options.DiscountFactor,
        LossFunction = new MeanSquaredErrorLoss<T>(),
        Seed = options.Seed
    })
    {
        _options = options;
        _updateCount = 0;

        // Initialize networks directly in constructor
        // Representation function: observation -> hidden state
        _representationNetwork = CreateNetwork(_options.ObservationSize, _options.LatentStateSize, _options.RepresentationLayers);

        // Dynamics function: (hidden state, action) -> (next hidden state, reward)
        _dynamicsNetwork = CreateNetwork(_options.LatentStateSize + _options.ActionSize, _options.LatentStateSize + 1, _options.DynamicsLayers);

        // Prediction function: hidden state -> (policy, value)
        _predictionNetwork = CreateNetwork(_options.LatentStateSize, _options.ActionSize + 1, _options.PredictionLayers);

        // Initialize replay buffer
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize, _options.Seed);

        // Initialize Networks list for base class (used by GetParameters/SetParameters)
        Networks = new List<INeuralNetwork<T>>
        {
            _representationNetwork,
            _dynamicsNetwork,
            _predictionNetwork
        };
    }

    private NeuralNetwork<T> CreateNetwork(int inputSize, int outputSize, List<int> hiddenLayers)
    {
        var layers = new List<ILayer<T>>();
        int previousSize = inputSize;

        foreach (var layerSize in hiddenLayers)
        {
            layers.Add(new DenseLayer<T>(previousSize, layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            previousSize = layerSize;
        }

        layers.Add(new DenseLayer<T>(previousSize, outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture);
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize);
    }

    public override Vector<T> SelectAction(Vector<T> observation, bool training = true)
    {
        // Encode observation to hidden state
        var obsTensor = Tensor<T>.FromVector(observation);
        var hiddenStateTensorOutput = _representationNetwork.Predict(obsTensor);
        var hiddenState = hiddenStateTensorOutput.ToVector();

        if (!training)
        {
            // Greedy: just use policy network
            var policyValueTensor = Tensor<T>.FromVector(hiddenState);
            var policyValueTensorOutput = _predictionNetwork.Predict(policyValueTensor);
            var policyValue = policyValueTensorOutput.ToVector();
            int bestAction = ArgMax(ExtractPolicy(policyValue));
            var action = new Vector<T>(_options.ActionSize);
            action[bestAction] = NumOps.One;
            return action;
        }

        // Run MCTS to select action
        int selectedAction = RunMCTS(hiddenState);

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

    private int RunMCTS(Vector<T> rootHiddenState)
    {
        var root = new MCTSNode<T> { HiddenState = rootHiddenState };

        // Initialize root
        var rootPredictionTensor = Tensor<T>.FromVector(rootHiddenState);
        var rootPredictionTensorOutput = _predictionNetwork.Predict(rootPredictionTensor);
        var rootPrediction = rootPredictionTensorOutput.ToVector();
        root.Value = ExtractValue(rootPrediction);

        // Run simulations
        for (int sim = 0; sim < _options.NumSimulations; sim++)
        {
            SimulateFromNode(root);
        }

        // Select action with highest visit count
        int bestAction = 0;
        int maxVisits = 0;

        foreach (var kvp in root.VisitCounts)
        {
            if (kvp.Value > maxVisits)
            {
                maxVisits = kvp.Value;
                bestAction = kvp.Key;
            }
        }

        return bestAction;
    }

    private void SimulateFromNode(MCTSNode<T> node)
    {
        // Selection: traverse tree using PUCT
        var path = new List<(MCTSNode<T> node, int action)>();
        var currentNode = node;

        while (currentNode.Children.Count > 0)
        {
            int action = SelectActionPUCT(currentNode);
            path.Add((currentNode, action));

            if (!currentNode.Children.ContainsKey(action))
            {
                break;
            }

            currentNode = currentNode.Children[action];
        }

        // Expansion: if not terminal, expand
        if (path.Count < _options.UnrollSteps)
        {
            int action = SelectActionPUCT(currentNode);
            var child = ExpandNode(currentNode, action);
            currentNode.Children[action] = child;
            path.Add((currentNode, action));
            currentNode = child;
        }

        // Evaluation: get value from prediction network
        T value = currentNode.Value;

        // Backup: propagate value up the tree with rewards
        // CRITICAL: Must compute backed-up value BEFORE updating Q-values
        for (int i = path.Count - 1; i >= 0; i--)
        {
            var (pathNode, pathAction) = path[i];

            // Compute the backed-up value first (reward + gamma * child_value)
            // This is the value we'll use to update Q
            T backedUpValue = value;
            if (pathNode.Rewards.ContainsKey(pathAction))
            {
                var reward = pathNode.Rewards[pathAction];
                backedUpValue = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, value));
            }
            else
            {
                // If no reward stored, just discount (for root node initial actions)
                backedUpValue = NumOps.Multiply(DiscountFactor, value);
            }

            // Initialize visit counts and Q-values if this is first visit
            // This should only happen for root node on first simulation
            if (!pathNode.VisitCounts.ContainsKey(pathAction))
            {
                pathNode.VisitCounts[pathAction] = 0;
                pathNode.QValues[pathAction] = NumOps.Zero;
            }

            // Increment visit counts
            pathNode.VisitCounts[pathAction]++;
            pathNode.TotalVisits++;

            // Update Q-value using incremental mean: Q_new = Q_old + (backed_up_value - Q_old) / n
            // This is mathematically equivalent to: Q = (Q * (n-1) + backed_up_value) / n
            var oldQ = pathNode.QValues[pathAction];
            var n = NumOps.FromDouble(pathNode.VisitCounts[pathAction]);
            var diff = NumOps.Subtract(backedUpValue, oldQ);
            var update = NumOps.Divide(diff, n);
            pathNode.QValues[pathAction] = NumOps.Add(oldQ, update);

            // Propagate the backed-up value to parent for next iteration
            value = backedUpValue;
        }
    }

    private int SelectActionPUCT(MCTSNode<T> node)
    {
        // PUCT formula: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        var predictionTensor = Tensor<T>.FromVector(node.HiddenState);
        var predictionOutput = _predictionNetwork.Predict(predictionTensor);
        var prediction = predictionOutput.ToVector();
        var policy = ExtractPolicy(prediction);

        double bestScore = double.NegativeInfinity;
        int bestAction = 0;

        double sqrtTotalVisits = Math.Sqrt(node.TotalVisits + 1);

        for (int action = 0; action < _options.ActionSize; action++)
        {
            double qValue = 0;
            if (node.QValues.ContainsKey(action))
            {
                qValue = NumOps.ToDouble(node.QValues[action]);
            }

            int visitCount = node.VisitCounts.ContainsKey(action) ? node.VisitCounts[action] : 0;
            double prior = NumOps.ToDouble(policy[action]);

            double puctScore = qValue + _options.PUCTConstant * prior * sqrtTotalVisits / (1 + visitCount);

            if (puctScore > bestScore)
            {
                bestScore = puctScore;
                bestAction = action;
            }
        }

        return bestAction;
    }

    private MCTSNode<T> ExpandNode(MCTSNode<T> parent, int action)
    {
        // Use dynamics network to predict next hidden state and reward
        var actionVec = new Vector<T>(_options.ActionSize);
        actionVec[action] = NumOps.One;

        var dynamicsInput = ConcatenateVectors(parent.HiddenState, actionVec);
        var dynamicsInputTensor = Tensor<T>.FromVector(dynamicsInput);
        var dynamicsOutputTensor = _dynamicsNetwork.Predict(dynamicsInputTensor);
        var dynamicsOutput = dynamicsOutputTensor.ToVector();

        // Extract next hidden state and reward
        // Dynamics output: [hidden_state (latentStateSize), reward (1)]
        var nextHiddenState = new Vector<T>(_options.LatentStateSize);
        for (int i = 0; i < _options.LatentStateSize; i++)
        {
            nextHiddenState[i] = dynamicsOutput[i];
        }

        // Extract predicted reward (last element of dynamics output)
        var predictedReward = dynamicsOutput[_options.LatentStateSize];

        // Get value from prediction network
        var predictionTensor = Tensor<T>.FromVector(nextHiddenState);
        var predictionTensorOutput = _predictionNetwork.Predict(predictionTensor);
        var prediction = predictionTensorOutput.ToVector();
        var value = ExtractValue(prediction);

        // Store reward in parent node for this action
        parent.Rewards[action] = predictedReward;

        return new MCTSNode<T>
        {
            HiddenState = nextHiddenState,
            Value = value,
            TotalVisits = 0
        };
    }

    private Vector<T> ExtractPolicy(Vector<T> predictionOutput)
    {
        var policy = new Vector<T>(_options.ActionSize);
        for (int i = 0; i < _options.ActionSize; i++)
        {
            policy[i] = predictionOutput[i];
        }
        return policy;
    }

    private T ExtractValue(Vector<T> predictionOutput)
    {
        return predictionOutput[_options.ActionSize];
    }

    public override void StoreExperience(Vector<T> observation, Vector<T> action, T reward, Vector<T> nextObservation, bool done)
    {
        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(observation, action, reward, nextObservation, done));
    }

    public override T Train()
    {
        if (_replayBuffer.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        var batch = _replayBuffer.Sample(_options.BatchSize);
        T totalLoss = NumOps.Zero;
        int lossCount = 0;

        foreach (var experience in batch)
        {
            // Step 1: Representation Network - encode initial observation to hidden state
            var stateTensor = Tensor<T>.FromVector(experience.State);
            var representationOutputTensor = _representationNetwork.Predict(stateTensor);
            var hiddenState = representationOutputTensor.ToVector();

            // Step 2: Prediction Network at initial state - predict policy and value
            var predictionTensor = Tensor<T>.FromVector(hiddenState);
            var predictionOutputTensor = _predictionNetwork.Predict(predictionTensor);
            var prediction = predictionOutputTensor.ToVector();
            var predictedValue = ExtractValue(prediction);

            // Compute value loss for initial state
            var valueTarget = experience.Done ? experience.Reward :
                NumOps.Add(experience.Reward, NumOps.Multiply(DiscountFactor, predictedValue));

            var valueDiff = NumOps.Subtract(valueTarget, predictedValue);
            var valueLoss = NumOps.Multiply(valueDiff, valueDiff);
            totalLoss = NumOps.Add(totalLoss, valueLoss);
            lossCount++;

            // Backpropagate prediction loss through prediction network
            var predictionGradient = new Vector<T>(_options.ActionSize + 1);
            predictionGradient[_options.ActionSize] = NumOps.Multiply(NumOps.FromDouble(2.0), valueDiff);
            var predictionGradTensor = Tensor<T>.FromVector(predictionGradient);
            _predictionNetwork.Backpropagate(predictionGradTensor);

            // Step 3: Unroll dynamics for K steps
            for (int k = 0; k < _options.UnrollSteps; k++)
            {
                // Dynamics Network: predict next hidden state and reward
                var actionVec = experience.Action;
                var dynamicsInput = ConcatenateVectors(hiddenState, actionVec);
                var dynamicsInputTensor = Tensor<T>.FromVector(dynamicsInput);
                var dynamicsOutputTensor = _dynamicsNetwork.Predict(dynamicsInputTensor);
                var dynamicsOutput = dynamicsOutputTensor.ToVector();

                // Extract predicted reward and next hidden state
                var predictedReward = dynamicsOutput[_options.LatentStateSize];
                var nextHiddenState = new Vector<T>(_options.LatentStateSize);
                for (int i = 0; i < _options.LatentStateSize; i++)
                {
                    nextHiddenState[i] = dynamicsOutput[i];
                }

                // Compute reward loss
                var rewardDiff = NumOps.Subtract(experience.Reward, predictedReward);
                var rewardLoss = NumOps.Multiply(rewardDiff, rewardDiff);
                totalLoss = NumOps.Add(totalLoss, rewardLoss);
                lossCount++;

                // Backpropagate reward loss through dynamics network
                var dynamicsGradient = new Vector<T>(_options.LatentStateSize + 1);
                dynamicsGradient[_options.LatentStateSize] = NumOps.Multiply(NumOps.FromDouble(2.0), rewardDiff);
                var dynamicsGradTensor = Tensor<T>.FromVector(dynamicsGradient);
                _dynamicsNetwork.Backpropagate(dynamicsGradTensor);

                // Prediction Network at next state
                var nextPredictionTensor = Tensor<T>.FromVector(nextHiddenState);
                var nextPredictionOutputTensor = _predictionNetwork.Predict(nextPredictionTensor);
                var nextPrediction = nextPredictionOutputTensor.ToVector();
                var nextPredictedValue = ExtractValue(nextPrediction);

                // Compute value loss for next state
                var nextValueTarget = experience.Done ? NumOps.Zero : nextPredictedValue;
                var nextValueDiff = NumOps.Subtract(nextValueTarget, nextPredictedValue);
                var nextValueLoss = NumOps.Multiply(nextValueDiff, nextValueDiff);
                totalLoss = NumOps.Add(totalLoss, nextValueLoss);
                lossCount++;

                // Backpropagate next state value loss through prediction network
                var nextPredictionGradient = new Vector<T>(_options.ActionSize + 1);
                nextPredictionGradient[_options.ActionSize] = NumOps.Multiply(NumOps.FromDouble(2.0), nextValueDiff);
                var nextPredictionGradTensor = Tensor<T>.FromVector(nextPredictionGradient);
                _predictionNetwork.Backpropagate(nextPredictionGradTensor);

                // Move to next state
                hiddenState = nextHiddenState;
            }

            // Step 4: Backpropagate through representation network
            // The representation gradient comes from the prediction network loss
            var representationGradient = new Vector<T>(_options.LatentStateSize);
            representationGradient[0] = NumOps.Multiply(NumOps.FromDouble(2.0), valueDiff);
            var representationGradTensor = Tensor<T>.FromVector(representationGradient);
            _representationNetwork.Backpropagate(representationGradTensor);
        }

        _updateCount++;

        return lossCount > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(lossCount)) : NumOps.Zero;
    }


    private Vector<T> ConcatenateVectors(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(a.Length + b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = a[i];
        }
        for (int i = 0; i < b.Length; i++)
        {
            result[a.Length + i] = b[i];
        }
        return result;
    }

    private int ArgMax(Vector<T> values)
    {
        int maxIndex = 0;
        T maxValue = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (NumOps.GreaterThan(values[i], maxValue))
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = NumOps.FromDouble(_updateCount),
            ["buffer_size"] = NumOps.FromDouble(_replayBuffer.Count)
        };
    }

    public override void ResetEpisode()
    {
        // No episode-specific state
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.MuZeroAgent
        };
    }

    public override int FeatureCount => _options.ObservationSize;

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write options
        writer.Write(_options.ObservationSize);
        writer.Write(_options.ActionSize);
        writer.Write(_options.LatentStateSize);
        writer.Write(_options.NumSimulations);
        writer.Write(_options.ReplayBufferSize);
        writer.Write(_options.BatchSize);
        writer.Write(_options.UnrollSteps);
        writer.Write(_options.PUCTConstant);
        writer.Write(NumOps.ToDouble(_options.LearningRate!));
        writer.Write(NumOps.ToDouble(_options.DiscountFactor!));
        writer.Write(_options.Seed ?? 0);
        writer.Write(_options.Seed.HasValue);

        // Write hidden layer configurations
        writer.Write(_options.RepresentationLayers.Count);
        foreach (var size in _options.RepresentationLayers)
            writer.Write(size);

        writer.Write(_options.DynamicsLayers.Count);
        foreach (var size in _options.DynamicsLayers)
            writer.Write(size);

        writer.Write(_options.PredictionLayers.Count);
        foreach (var size in _options.PredictionLayers)
            writer.Write(size);

        // Write update count
        writer.Write(_updateCount);

        // Serialize each network
        var repData = _representationNetwork.Serialize();
        writer.Write(repData.Length);
        writer.Write(repData);

        var dynData = _dynamicsNetwork.Serialize();
        writer.Write(dynData.Length);
        writer.Write(dynData);

        var predData = _predictionNetwork.Serialize();
        writer.Write(predData.Length);
        writer.Write(predData);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read options
        int observationSize = reader.ReadInt32();
        int actionSize = reader.ReadInt32();
        int latentStateSize = reader.ReadInt32();
        int numSimulations = reader.ReadInt32();
        int replayBufferSize = reader.ReadInt32();
        int batchSize = reader.ReadInt32();
        int unrollSteps = reader.ReadInt32();
        double puctConstant = reader.ReadDouble();
        T learningRate = NumOps.FromDouble(reader.ReadDouble());
        T discountFactor = NumOps.FromDouble(reader.ReadDouble());
        int seed = reader.ReadInt32();
        bool hasSeed = reader.ReadBoolean();

        // Read hidden layer configurations
        int repLayerCount = reader.ReadInt32();
        var repLayers = new List<int>();
        for (int i = 0; i < repLayerCount; i++)
            repLayers.Add(reader.ReadInt32());

        int dynLayerCount = reader.ReadInt32();
        var dynLayers = new List<int>();
        for (int i = 0; i < dynLayerCount; i++)
            dynLayers.Add(reader.ReadInt32());

        int predLayerCount = reader.ReadInt32();
        var predLayers = new List<int>();
        for (int i = 0; i < predLayerCount; i++)
            predLayers.Add(reader.ReadInt32());

        _options = new MuZeroOptions<T>
        {
            ObservationSize = observationSize,
            ActionSize = actionSize,
            LatentStateSize = latentStateSize,
            NumSimulations = numSimulations,
            ReplayBufferSize = replayBufferSize,
            BatchSize = batchSize,
            UnrollSteps = unrollSteps,
            PUCTConstant = puctConstant,
            LearningRate = learningRate,
            DiscountFactor = discountFactor,
            Seed = hasSeed ? seed : null,
            RepresentationLayers = repLayers,
            DynamicsLayers = dynLayers,
            PredictionLayers = predLayers
        };

        // Read update count
        _updateCount = reader.ReadInt32();

        // Deserialize each network
        int repLen = reader.ReadInt32();
        byte[] repData = reader.ReadBytes(repLen);
        _representationNetwork.Deserialize(repData);

        int dynLen = reader.ReadInt32();
        byte[] dynData = reader.ReadBytes(dynLen);
        _dynamicsNetwork.Deserialize(dynData);

        int predLen = reader.ReadInt32();
        byte[] predData = reader.ReadBytes(predLen);
        _predictionNetwork.Deserialize(predData);

        // Reinitialize replay buffer (training state not persisted)
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize, _options.Seed);

        // Update Networks list
        Networks = new List<INeuralNetwork<T>>
        {
            _representationNetwork,
            _dynamicsNetwork,
            _predictionNetwork
        };
    }

    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        foreach (var network in Networks)
        {
            var netParams = network.GetParameters();
            for (int i = 0; i < netParams.Length; i++)
            {
                allParams.Add(netParams[i]);
            }
        }

        var paramVector = new Vector<T>(allParams.Count);
        for (int i = 0; i < allParams.Count; i++)
        {
            paramVector[i] = allParams[i];
        }

        return paramVector;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        foreach (var network in Networks)
        {
            int paramCount = network.ParameterCount;
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[offset + i];
            }
            network.UpdateParameters(netParams);
            offset += paramCount;
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        return new MuZeroAgent<T>(_options);
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var gradient = usedLossFunction.CalculateDerivative(prediction, target);
        return gradient;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var currentParams = GetParameters();
        var newParams = new Vector<T>(currentParams.Length);

        for (int i = 0; i < currentParams.Length; i++)
        {
            var update = NumOps.Multiply(learningRate, gradients[i]);
            newParams[i] = NumOps.Subtract(currentParams[i], update);
        }

        SetParameters(newParams);
    }

    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
