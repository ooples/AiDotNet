using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;

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

    // Three core networks
    private NeuralNetwork<T> _representationNetwork;  // h = f(observation)
    private NeuralNetwork<T> _dynamicsNetwork;  // (h', r) = g(h, action)
    private NeuralNetwork<T> _predictionNetwork;  // (p, v) = f(h)

    private ReplayBuffer<T> _replayBuffer;
    private int _updateCount;

    public MuZeroAgent(MuZeroOptions<T> options) : base(options.ObservationSize, options.ActionSize)
    {
        _options = options;
        _updateCount = 0;

        InitializeNetworks();
        InitializeReplayBuffer();
    }

    private void InitializeNetworks()
    {
        // Representation function: observation -> hidden state
        _representationNetwork = CreateNetwork(_options.ObservationSize, _options.LatentStateSize, _options.RepresentationLayers);

        // Dynamics function: (hidden state, action) -> (next hidden state, reward)
        _dynamicsNetwork = CreateNetwork(_options.LatentStateSize + _options.ActionSize, _options.LatentStateSize + 1, _options.DynamicsLayers);

        // Prediction function: hidden state -> (policy, value)
        _predictionNetwork = CreateNetwork(_options.LatentStateSize, _options.ActionSize + 1, _options.PredictionLayers);
    }

    private NeuralNetwork<T> CreateNetwork(int inputSize, int outputSize, List<int> hiddenLayers)
    {
        var network = new NeuralNetwork<T>();
        int previousSize = inputSize;

        foreach (var layerSize in hiddenLayers)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, layerSize));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = layerSize;
        }

        network.AddLayer(new DenseLayer<T>(previousSize, outputSize));

        return network;
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new ReplayBuffer<T>(_options.ReplayBufferSize);
    }

    public override Vector<T> SelectAction(Vector<T> observation, bool training = true)
    {
        // Encode observation to hidden state
        var hiddenState = _representationNetwork.Forward(observation);

        if (!training)
        {
            // Greedy: just use policy network
            var policyValue = _predictionNetwork.Forward(hiddenState);
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
        var rootPrediction = _predictionNetwork.Forward(rootHiddenState);
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

        // Backup: propagate value up the tree
        for (int i = path.Count - 1; i >= 0; i--)
        {
            var (pathNode, pathAction) = path[i];

            if (!pathNode.VisitCounts.ContainsKey(pathAction))
            {
                pathNode.VisitCounts[pathAction] = 0;
                pathNode.QValues[pathAction] = NumOps.Zero;
            }

            pathNode.VisitCounts[pathAction]++;
            pathNode.TotalVisits++;

            // Update Q-value: Q = (Q * n + v) / (n + 1)
            var oldQ = pathNode.QValues[pathAction];
            var visitCount = NumOps.FromDouble(pathNode.VisitCounts[pathAction]);
            var newQ = NumOps.Divide(
                NumOps.Add(NumOps.Multiply(oldQ, visitCount), value),
                NumOps.Add(visitCount, NumOps.One));

            pathNode.QValues[pathAction] = newQ;

            // Discount value for parent
            value = NumOps.Multiply(_options.DiscountFactor, value);
        }
    }

    private int SelectActionPUCT(MCTSNode<T> node)
    {
        // PUCT formula: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        var prediction = _predictionNetwork.Forward(node.HiddenState);
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

    private MCTSNode ExpandNode(MCTSNode<T> parent, int action)
    {
        // Use dynamics network to predict next hidden state and reward
        var actionVec = new Vector<T>(_options.ActionSize);
        actionVec[action] = NumOps.One;

        var dynamicsInput = ConcatenateVectors(parent.HiddenState, actionVec);
        var dynamicsOutput = _dynamicsNetwork.Forward(dynamicsInput);

        // Extract next hidden state and reward
        var nextHiddenState = new Vector<T>(_options.LatentStateSize);
        for (int i = 0; i < _options.LatentStateSize; i++)
        {
            nextHiddenState[i] = dynamicsOutput[i];
        }

        // Get value from prediction network
        var prediction = _predictionNetwork.Forward(nextHiddenState);
        var value = ExtractValue(prediction);

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
        _replayBuffer.Add(observation, action, reward, nextObservation, done);
    }

    public override T Train()
    {
        if (_replayBuffer.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        var batch = _replayBuffer.Sample(_options.BatchSize);
        T totalLoss = NumOps.Zero;

        foreach (var experience in batch)
        {
            // Encode observation
            var hiddenState = _representationNetwork.Forward(experience.observation);

            // Unroll K steps
            for (int k = 0; k < _options.UnrollSteps; k++)
            {
                // Prediction loss
                var prediction = _predictionNetwork.Forward(hiddenState);
                var predictedValue = ExtractValue(prediction);

                // Simplified target: use reward + discounted next value
                var target = experience.done ? experience.reward :
                    NumOps.Add(experience.reward, NumOps.Multiply(_options.DiscountFactor, predictedValue));

                var valueDiff = NumOps.Subtract(target, predictedValue);
                var loss = NumOps.Multiply(valueDiff, valueDiff);
                totalLoss = NumOps.Add(totalLoss, loss);

                // Backprop
                var gradient = new Vector<T>(_options.ActionSize + 1);
                gradient[_options.ActionSize] = valueDiff;

                _predictionNetwork.Backward(gradient);
                _predictionNetwork.UpdateWeights(_options.LearningRate);

                // Dynamics step
                var actionVec = experience.action;
                var dynamicsInput = ConcatenateVectors(hiddenState, actionVec);
                var dynamicsOutput = _dynamicsNetwork.Forward(dynamicsInput);

                // Extract next hidden state
                hiddenState = new Vector<T>(_options.LatentStateSize);
                for (int i = 0; i < _options.LatentStateSize; i++)
                {
                    hiddenState[i] = dynamicsOutput[i];
                }
            }
        }

        _updateCount++;

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count * _options.UnrollSteps));
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
            if (NumOps.Compare(values[i], maxValue) > 0)
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

    public override Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public override Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }
}
