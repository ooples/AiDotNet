using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using System;

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
/// <example>
/// <code>
/// // Create a MuZero agent that learns its own world model
/// var options = new MuZeroOptions&lt;double&gt; { StateSize = 64, ActionSize = 4, NumSimulations = 50 };
/// var agent = new MuZeroAgent&lt;double&gt;(options);
///
/// // Select an action using Monte Carlo Tree Search with learned dynamics
/// var state = new Vector&lt;double&gt;(new double[64]);
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model",
    "https://arxiv.org/abs/1911.08265",
    Year = 2020,
    Authors = "Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., et al.")]
public class MuZeroAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private MuZeroOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    // Three core networks
    private INeuralNetwork<T> _representationNetwork;  // h = f(observation)
    private INeuralNetwork<T> _dynamicsNetwork;  // (h', r) = g(h, action)
    private INeuralNetwork<T> _predictionNetwork;  // (p, v) = f(h)

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private int _updateCount;

    // #1756: Adam optimizer moment state for the K-step unrolled joint training. The representation,
    // dynamics and prediction networks are unrolled on ONE gradient tape and share a single step
    // counter, but each network's flat Adam moments (m, v over its GetParameters() layout) are kept
    // separately keyed by the network instance. Transient optimizer state, not serialized — the same
    // role as PyTorch's optimizer.state.
    private readonly System.Collections.Generic.Dictionary<object, (double[] M, double[] V)> _netAdam = new();
    private int _adamStep;

    /// <summary>
    /// Initializes a new instance with default settings. All hyperparameters
    /// come from <see cref="MuZeroOptions{T}"/>'s field defaults
    /// (ObservationSize=4, ActionSize=2, LatentStateSize=256, etc.) so
    /// callers can override any of them by constructing a populated
    /// <see cref="MuZeroOptions{T}"/> and passing it to the other ctor.
    /// </summary>
    public MuZeroAgent()
        : this(new MuZeroOptions<T>())
    {
    }

    public MuZeroAgent(MuZeroOptions<T> options) : base(new ReinforcementLearningOptions<T>
    {
        LearningRate = (options ?? throw new ArgumentNullException(nameof(options))).LearningRate,
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
            layers.Add(new DenseLayer<T>(layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            previousSize = layerSize;
        }

        layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));

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
            // Inference path: return a one-hot argmax(π) action. The
            // IRLAgent<T> public contract on discrete envs is "give me
            // the action to take" — callers feed the result directly into
            // env.Step(action). Returning a raw policy distribution would
            // change the API from action-selector to policy-diagnostic
            // and break evaluation loops expecting a deterministic action.
            // The full π(·|s) from the prediction network is still
            // observable to callers that need it for diagnostics — they
            // can invoke _predictionNetwork.Predict directly or wire a
            // separate GetPolicy(...) method if needed.
            var policyValueTensor = Tensor<T>.FromVector(hiddenState);
            var policyValueTensorOutput = _predictionNetwork.Predict(policyValueTensor);
            var policyValue = policyValueTensorOutput.ToVector();
            var policy = ExtractPolicy(policyValue);
            int bestIdx = ArgMax(policy);
            var oneHot = new Vector<T>(_options.ActionSize);
            oneHot[bestIdx] = NumOps.One;
            return oneHot;
        }

        // Run MCTS to select action
        int selectedAction = RunMCTS(hiddenState);

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

    private int RunMCTS(Vector<T> rootHiddenState)
    {
        var root = new MCTSNode<T> { HiddenState = rootHiddenState, Value = NumOps.Zero };

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
        if (node.HiddenState is null)
            throw new InvalidOperationException("MCTS node HiddenState must be set before action selection.");
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

        if (parent.HiddenState is null)
            throw new InvalidOperationException("Parent MCTS node HiddenState must be set before expansion.");
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
        // Validate transition shapes at this public ingestion boundary so a malformed experience
        // can't reach Train() (which indexes observations/actions per _options) and crash or build
        // wrong policy/value targets.
        if (observation.Length != _options.ObservationSize)
            throw new ArgumentException($"Observation length must be {_options.ObservationSize}, got {observation.Length}.", nameof(observation));
        if (nextObservation.Length != _options.ObservationSize)
            throw new ArgumentException($"Next observation length must be {_options.ObservationSize}, got {nextObservation.Length}.", nameof(nextObservation));
        if (action.Length != _options.ActionSize)
            throw new ArgumentException($"Action length must be {_options.ActionSize}, got {action.Length}.", nameof(action));

        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(observation, action, reward, nextObservation, done));
    }

    public override T Train()
    {
        // Need at least one real transition (o_0, a_0, r_0, o_1) to unroll from. A single Experience
        // already carries both o_0 and its NextState (o_1), which is enough for a 1-step unroll whose
        // value target bootstraps off NextState (see NStepValueTarget), so training can start as soon
        // as the buffer holds one transition.
        if (_replayBuffer.Count < 1)
        {
            return NumOps.Zero;
        }

        // MuZero joint training (Schrittwieser et al. 2020, "Mastering Atari, Go, Chess and Shogi by
        // Planning with a Learned Model", §Training). The representation (h), dynamics (g) and
        // prediction (f) networks are trained TOGETHER by unrolling the learned model K = UnrollSteps
        // steps from a real trajectory window and backpropagating ONE joint loss through the whole
        // unroll: h_0 = h(o_0); for k: (p_k, v_k) = f(h_k), (h_{k+1}, r_k) = g(h_k, a_k). The dynamics
        // network is applied K times, so gradients flow through the recurrence h_0 -> ... -> h_K. A
        // single GradientTape spans all three networks; their parameters are collected after the
        // forward (lazy layers materialize during it) and updated with one joint Adam step. Value
        // targets bootstrap n-step (TDSteps) off the current network, computed DETACHED (off-tape).
        int unroll = Math.Max(1, _options.UnrollSteps);
        int latent = _options.LatentStateSize;
        int actionDim = _options.ActionSize;
        int nStep = Math.Max(1, _options.TDSteps);
        T gamma = DiscountFactor;
        int batchSize = Math.Min(_replayBuffer.Count, _options.BatchSize);

        var repNet = (NeuralNetworkBase<T>)_representationNetwork;
        var dynNet = (NeuralNetworkBase<T>)_dynamicsNetwork;
        var predNet = (NeuralNetworkBase<T>)_predictionNetwork;

        // ForwardForTraining requires training mode for the whole step (its own contract): in eval mode
        // the forward reads inference-cached weights and does not build the trainable tape graph, so the
        // joint gradient would be empty. Set on all three networks for the unroll, restore after.
        repNet.SetTrainingMode(true);
        dynNet.SetTrainingMode(true);
        predNet.SetTrainingMode(true);
        try
        {

        using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>();

        // Collect every tape-tracked (prediction − target) residual across the batch of unrolled
        // sequences; the joint loss is the mean squared residual (equivalent to summing the per-head
        // value/reward/policy MSEs, and unroll-length-normalized).
        var residuals = new System.Collections.Generic.List<Tensor<T>>();

        for (int b = 0; b < batchSize; b++)
        {
            // Trajectory window of up to K+1 consecutive transitions (truncated at an episode boundary).
            var seq = _replayBuffer.SampleSequence(unroll + 1);
            if (seq.Count < 1) continue;
            // Train one unroll step per transition in the window (capped at K). On a FULL K+1 window this
            // is K steps and seq[K] serves only as the final bootstrap state (unchanged). On a window
            // TRUNCATED at an episode boundary (or a 1-transition buffer) this also trains the last /
            // terminal transition's heads — its value target bootstraps off seq[last].NextState via
            // NStepValueTarget, so no transition near an episode end is silently dropped from the loss.
            int steps = Math.Min(unroll, seq.Count);

            // Cache bootstrap V(s) per unique bootstrap index within THIS sequence: NStepValueTarget is
            // called once per unrolled step k and each call runs a detached representation+prediction pass
            // for its bootstrap state. Different k's often resolve to the same bootstrap index (e.g. every
            // step past the last non-terminal transition bootstraps off the same terminal NextState), so
            // memoizing avoids repeating that inference batchSize×unrollSteps times.
            var bootValueCache = new System.Collections.Generic.Dictionary<int, T>();

            // h_0 = f_representation(o_0). Outputs stay rank-1 (FromVector -> [n], Dense preserves rank),
            // so no Reshape is used in the unroll — Engine.Reshape deliberately does NOT attach an
            // autodiff GradFn (it's sidestepped for view caching), which would sever the tape.
            var h = repNet.ForwardForTraining(Tensor<T>.FromVector(seq[0].State));

            for (int k = 0; k < steps; k++)
            {
                var exp = seq[k];

                // Prediction head at h_k -> [policy (actionDim) | value (1)].
                var pred = predNet.ForwardForTraining(h);
                var policyPred = Engine.TensorSlice(pred, new[] { 0 }, new[] { actionDim });
                var valuePred = Engine.TensorSlice(pred, new[] { actionDim }, new[] { 1 });
                residuals.Add(Engine.TensorSubtract(valuePred, ScalarTensor(NStepValueTarget(seq, k, nStep, gamma, bootValueCache))));
                residuals.Add(Engine.TensorSubtract(policyPred, ActionTensor(exp.Action, actionDim)));

                // Dynamics: (h_{k+1}, r_k) = g(h_k, a_k).
                var dynIn = Engine.TensorConcatenate(new[] { h, ActionTensor(exp.Action, actionDim) }, 0);
                var dynOut = dynNet.ForwardForTraining(dynIn);
                var hNext = Engine.TensorSlice(dynOut, new[] { 0 }, new[] { latent });
                var rewardPred = Engine.TensorSlice(dynOut, new[] { latent }, new[] { 1 });
                residuals.Add(Engine.TensorSubtract(rewardPred, ScalarTensor(exp.Reward)));

                h = hNext;
            }
        }

        if (residuals.Count == 0)
        {
            return NumOps.Zero;
        }

        var allResiduals = residuals.Count == 1
            ? residuals[0]
            : Engine.TensorConcatenate(residuals.ToArray(), 0);
        var loss = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(Engine.TensorMultiply(allResiduals, allResiduals), null),
            NumOps.FromDouble(1.0 / allResiduals.Length));

        // Compute grads for ALL tracked leaves (sources: null) — the pattern NeuralNetworkBase's own
        // tape training uses; passing explicit sources can miss the exact leaf instances recorded.
        var grads = tape.ComputeGradients(loss, sources: null);
        // Apply the joint gradient through each network's canonical GetParameters/UpdateParameters
        // round-trip. The tape's grads are keyed by the tensors the forward used; those are NOT
        // necessarily the same instances GetParameters exposes, so an in-place write to them would not
        // be reflected. Mapping the grads into GetParameters' layer-ordered flat layout and writing
        // back via UpdateParameters is the only path guaranteed to persist.
        _adamStep++;
        ApplyNetworkAdam(repNet, grads);
        ApplyNetworkAdam(dynNet, grads);
        ApplyNetworkAdam(predNet, grads);

        _updateCount++;
        return loss[0];

        }
        finally
        {
            repNet.SetTrainingMode(false);
            dynNet.SetTrainingMode(false);
            predNet.SetTrainingMode(false);
        }
    }

    // ── K-step unroll helpers (#1756, Schrittwieser et al. 2020) ───────────────────────────────
    private Tensor<T> ScalarTensor(T value) => new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { value }));

    private Tensor<T> ActionTensor(Vector<T> action, int actionDim)
    {
        var v = new Vector<T>(actionDim);
        for (int i = 0; i < actionDim; i++) v[i] = i < action.Length ? action[i] : NumOps.Zero;
        return Tensor<T>.FromVector(v);
    }

    /// <summary>
    /// n-step bootstrapped value target for trajectory position <paramref name="k"/>:
    /// Σ_{j=0}^{n-1} γ^j·r_{k+j} + γ^n·V(s_{k+n}), truncated at a terminal. Computed DETACHED (via
    /// Predict, off-tape) — MuZero value targets carry no gradient into the online networks.
    /// </summary>
    private T NStepValueTarget(System.Collections.Generic.List<Experience<T, Vector<T>, Vector<T>>> seq, int k, int nStep, T gamma,
        System.Collections.Generic.Dictionary<int, T> bootValueCache)
    {
        T ret = NumOps.Zero;
        T disc = NumOps.One;
        int j = 0;
        for (; j < nStep && k + j < seq.Count; j++)
        {
            ret = NumOps.Add(ret, NumOps.Multiply(disc, seq[k + j].Reward));
            disc = NumOps.Multiply(disc, gamma);
            if (seq[k + j].Done) return ret; // terminal: no bootstrap beyond the episode end
        }

        // Resolve the bootstrap index; index seq.Count is the sentinel for the tail NextState (window ran
        // off the end without a terminal). Memoize V(s) per resolved index so repeated bootstrap targets
        // within this sequence reuse a single detached representation+prediction pass.
        bool withinSeq = k + j < seq.Count;
        int bootIndex = withinSeq ? k + j : seq.Count;
        if (!bootValueCache.TryGetValue(bootIndex, out var bootValue))
        {
            var bootstrapState = withinSeq ? seq[k + j].State : seq[seq.Count - 1].NextState;
            var hB = _representationNetwork.Predict(Tensor<T>.FromVector(bootstrapState)).ToVector();
            var predB = _predictionNetwork.Predict(Tensor<T>.FromVector(hB)).ToVector();
            bootValue = ExtractValue(predB);
            bootValueCache[bootIndex] = bootValue;
        }
        return NumOps.Add(ret, NumOps.Multiply(disc, bootValue));
    }

    /// <summary>
    /// Applies one Adam step (β1=0.9, β2=0.999, ε=1e-8, lr = the agent's resolved <see cref="LearningRate"/>
    /// — the base-class value with its non-zero fallback, NOT the nullable <c>Options.LearningRate</c>) to a single network,
    /// mapping the joint tape gradients into the network's canonical <c>GetParameters()</c> flat layout
    /// (layer order, the same order <see cref="TapeTrainingStep{T}.CollectParameters"/> walks) and
    /// writing back via <c>UpdateParameters</c> — the only path guaranteed to persist and be read back
    /// by <c>GetParameters</c>. Per-network Adam moment state persists in <see cref="_netAdam"/>.
    /// </summary>
    private void ApplyNetworkAdam(NeuralNetworkBase<T> net, System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var current = net.GetParameters();
        int total = current.Length;
        if (total == 0) return;

        // Build the flat gradient aligned to GetParameters' layout by concatenating each collected
        // parameter tensor's gradient in layer order (advancing the offset for every tensor, so a
        // parameter with no gradient leaves a zero block and is simply not updated).
        var flatGrad = new double[total];
        int offset = 0;
        foreach (var t in Training.TapeTrainingStep<T>.CollectParameters(net.Layers))
        {
            if (grads.TryGetValue(t, out var g) && g is not null)
            {
                var gs = g.AsSpan();
                // Bound by the gradient length too: g is not guaranteed to match t.Length (a partial
                // gradient would otherwise read past gs and throw IndexOutOfRange). Copy the overlapping
                // prefix; any remaining slots in this tensor's block stay zero (unaffected by the update).
                int limit = Math.Min(Math.Min(t.Length, gs.Length), total - offset);
                for (int i = 0; i < limit; i++) flatGrad[offset + i] = Convert.ToDouble(gs[i]);
            }
            offset += t.Length;
        }

        if (!_netAdam.TryGetValue(net, out var mom) || mom.M.Length != total)
        {
            mom = (new double[total], new double[total]);
            _netAdam[net] = mom;
        }
        var (m, v) = mom;

        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        double bc1 = 1.0 - Math.Pow(beta1, _adamStep);
        double bc2 = 1.0 - Math.Pow(beta2, _adamStep);
        // Use the agent's RESOLVED learning rate (base class, with the non-zero fallback), NOT
        // _options.LearningRate — the latter is a nullable T that defaults to null/zero, which would
        // make every Adam step zero and leave the parameters frozen.
        double lr = Convert.ToDouble(LearningRate);
        var updated = new Vector<T>(total);
        for (int i = 0; i < total; i++)
        {
            double g = flatGrad[i];
            double mi = beta1 * m[i] + (1.0 - beta1) * g;
            double vi = beta2 * v[i] + (1.0 - beta2) * g * g;
            m[i] = mi;
            v[i] = vi;
            double step = lr * (mi / bc1) / (Math.Sqrt(vi / bc2) + eps);
            updated[i] = NumOps.Subtract(current[i], NumOps.FromDouble(step));
        }
        net.UpdateParameters(updated);
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

    /// <summary>
    /// IFullModel.Predict surfaces the raw prediction-network output
    /// (policy logits + value) for the input observation rather than the
    /// one-hot committed action. Action commitment is the job of
    /// <see cref="SelectAction"/>; Predict is the policy + value diagnostic
    /// — needed by evaluation harnesses inspecting policy distinguishability
    /// across observations, by MCTS warm-start callers that consume the
    /// raw priors / value estimate, and by off-policy targets.
    /// </summary>
    public override Vector<T> Predict(Vector<T> input)
    {
        var obsTensor = Tensor<T>.FromVector(input);
        var hiddenStateTensorOutput = _representationNetwork.Predict(obsTensor);
        var hiddenState = hiddenStateTensorOutput.ToVector();
        var hiddenStateTensor = Tensor<T>.FromVector(hiddenState);
        var policyValueTensorOutput = _predictionNetwork.Predict(hiddenStateTensor);
        return policyValueTensorOutput.ToVector();
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
            int paramCount = checked((int)network.ParameterCount);
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[offset + i];
            }
            network.UpdateParameters(netParams);
            offset += paramCount;
        }
    }

    /// <summary>
    /// Creates a parameter-identical copy of this agent. The naive
    /// <c>new MuZeroAgent&lt;T&gt;(_options)</c> form only shares the
    /// configuration — the freshly-constructed copy has its three networks
    /// (representation, dynamics, prediction) re-initialized with
    /// independent random weights, so <c>cloned.Predict(state)</c> diverges
    /// from <c>original.Predict(state)</c> on the very first call. Copy the
    /// parameter vector across after construction so the clone observes the
    /// same policy distribution as the source.
    /// </summary>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        // Deep-copy the options before constructing the clone — _options
        // contains mutable collections (RepresentationLayers, DynamicsLayers,
        // PredictionLayers as List<int>) and sharing the same instance
        // would let post-clone edits on one model silently leak into the
        // other.
        var copy = new MuZeroAgent<T>(new MuZeroOptions<T>(_options));
        copy.SetParameters(GetParameters());
        return copy;
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
        if (gradients.Length != currentParams.Length)
        {
            throw new ArgumentException(
                $"Gradient vector length ({gradients.Length}) must match parameter count ({currentParams.Length}).",
                nameof(gradients));
        }

        SetParameters(Engine.Subtract(currentParams, Engine.Multiply(gradients, learningRate)));
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
