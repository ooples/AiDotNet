using AiDotNet.LearningRateSchedulers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using System;

namespace AiDotNet.ReinforcementLearning.Agents.Rainbow;

/// <summary>
/// Rainbow DQN agent combining six extensions to DQN.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Rainbow combines: Double Q-learning, Dueling networks, Prioritized replay,
/// Multi-step learning, Distributional RL (C51), and Noisy networks.
/// </para>
/// <para><b>For Beginners:</b>
/// Rainbow takes the best ideas from six different DQN improvements and combines them.
/// It's currently the strongest DQN variant, achieving state-of-the-art performance.
///
/// Six components:
/// 1. **Double Q-learning**: Reduces overestimation
/// 2. **Dueling Architecture**: Separates value and advantage
/// 3. **Prioritized Replay**: Samples important experiences more
/// 4. **Multi-step Returns**: Better credit assignment
/// 5. **Distributional RL (C51)**: Learns distribution of returns
/// 6. **Noisy Networks**: Parameter noise for exploration
///
/// Famous for: DeepMind's combination achieving human-level Atari performance
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Rainbow DQN agent combining six DQN improvements
/// var options = new RainbowDQNOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, NSteps = 3 };
/// var agent = new RainbowDQNAgent&lt;double&gt;(options);
///
/// // Select an action using distributional RL with noisy exploration
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Rainbow: Combining Improvements in Deep Reinforcement Learning",
    "https://arxiv.org/abs/1710.02298",
    Year = 2018,
    Authors = "Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., & Silver, D.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 6.25e-5, Epsilon = 1.5e-4,
                ReferenceBatchSize = 32,
                Source = "Hessel et al. 2018, Sec. 5: Rainbow uses Adam at a quarter of DQN's 0.00025 "
                        + "learning rate, selected among {a/2, a/4, a/6}, giving 6.25e-5, with 1.5e-4 "
                        + "for Adam's epsilon. The RMSprop the paper mentions belongs to the original "
                        + "DQN setup it builds on, not to Rainbow.")]
public partial class RainbowDQNAgent<T> : DeepReinforcementLearningAgentBase<T>, IActionValueProvider<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The online network is trainable state. The target is a derived periodic copy and
    /// exposing it to the optimizer both doubles the vector and lets its lazy topology drift from
    /// the online path after the first real observation.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent("rainbow/online-network", _onlineNetwork);
    }

    /// <inheritdoc />
    /// <remarks>The target network is a periodically refreshed copy of the online network, not an
    /// independent trainable component. A restore must refresh it immediately so training never
    /// resumes against stale target weights.</remarks>
    protected override void OnParametersRestored()
    {
        CopyNetworkWeights(_onlineNetwork, _targetNetwork);
    }
    private RainbowDQNOptions<T> _options;


    [ParameterAlias("rainbow/online-network")]
    private INeuralNetwork<T> _onlineNetwork;
    [Buffer]
    private INeuralNetwork<T> _targetNetwork;
    private PrioritizedReplayBuffer<T> _replayBuffer;

    private double _epsilon;
    private int _stepCount;
    private int _updateCount;
    private double _beta;

    // N-step buffer
    private List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> _nStepBuffer;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public RainbowDQNAgent()
        : this(new RainbowDQNOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    /// <summary>
    /// Initializes a Rainbow DQN agent from <paramref name="options"/>.
    /// </summary>
    /// <param name="options">The agent's configuration.</param>
    public RainbowDQNAgent(RainbowDQNOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        Options = _options;

        _stepCount = 0;
        _updateCount = 0;
        _epsilon = options.EpsilonStart;
        _beta = options.PriorityBeta;
        _nStepBuffer = new List<(Vector<T>, Vector<T>, T, Vector<T>, bool)>();

        // Initialize networks directly in constructor
        _onlineNetwork = CreateDuelingNetwork();
        _targetNetwork = CreateDuelingNetwork();
        CopyNetworkWeights(_onlineNetwork, _targetNetwork);

        // Register networks with base class
        Networks.Add(_onlineNetwork);
        Networks.Add(_targetNetwork);

        // Initialize replay buffer directly in constructor
        _replayBuffer = new PrioritizedReplayBuffer<T>(_options.ReplayBufferSize, Random.Next());
    }

    private NeuralNetwork<T> CreateDuelingNetwork()
    {
        // For distributional / C51 (Hessel et al. 2018 §3.5), each action
        // gets a categorical distribution over NumAtoms supports. The
        // dueling combination still applies per-atom: Q_{atom}(s, a) =
        // V_{atom}(s) + (A_{atom}(s, a) − mean_a A_{atom}(s, a)). Allocate
        // the advantage-head width as ActionSize * NumAtoms so the head
        // emits one logit per (action, atom) pair; the agent reshapes to
        // [batch, ActionSize, NumAtoms] before applying the per-action
        // softmax in ComputeQValues.
        int duelingActionWidth = _options.UseDistributional
            ? _options.ActionSize * _options.NumAtoms
            : _options.ActionSize;

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: duelingActionWidth
        )
        {
            RandomSeed = NextNetworkInitializationSeed(),
        };

        // Paper-faithful Rainbow dueling head (Wang et al. 2016 + Hessel
        // et al. 2018 §3.4): shared feature trunk → DuelingCombinationLayer.
        // The combination layer holds its own V(s) and A(s, a) projections
        // internally and emits Q(s, a) = V(s) + (A − mean_a A) so the
        // identifiability constraint is enforced without an extra layer.
        // Hessel §3.4 swaps the dense layers inside the heads for noisy
        // variants when UseNoisyNetworks is on; we wrap the same idea by
        // putting a NoisyDenseLayer between the trunk and the dueling head
        // so the action-conditional projection picks up the exploration
        // noise.
        // Shared feature trunk: one ReLU dense layer per SharedLayers width. With noisy nets on, a noisy hidden
        // layer of the trunk's width then feeds both streams, so the action-conditional projection sees the
        // exploration noise. The streams' own hidden layers are deterministic.
        var trunkWidths = _options.SharedLayers is { Count: > 0 } ? _options.SharedLayers : new List<int> { 64 };
        var layerList = new List<ILayer<T>>();
        foreach (var width in trunkWidths)
        {
            layerList.Add(new DenseLayer<T>(width, new ReLUActivation<T>() as IActivationFunction<T>));
        }

        int trunkHidden = trunkWidths[trunkWidths.Count - 1];
        if (_options.UseNoisyNetworks)
        {
            // Factorised Gaussian noise (Fortunato et al. 2018, §3.2): sigma_0 = NoisyNetSigma, initial
            // sigma = sigma_0 / sqrt(p) for p inputs. The noisy layer sits between the trunk and both streams.
            layerList.Add(new NoisyDenseLayer<T>(trunkHidden, trunkHidden,
                new ReLUActivation<T>() as IActivationFunction<T>,
                sigmaInit: _options.NoisyNetSigma / Math.Sqrt(trunkHidden),
                // The noisy layer draws its mu initialisation in its constructor, before any architecture seed
                // reaches it, so it takes this network's seed directly.
                seed: architecture.RandomSeed));
        }

        // The value and advantage streams (Wang et al. 2016, Fig. 1) each run their own hidden layers,
        // ValueStreamLayers and AdvantageStreamLayers. With distributional RL, V has one output per atom and the
        // advantage mean is taken over actions for each atom (Hessel et al. 2018, Sec. 3).
        layerList.Add(new DuelingCombinationLayer<T>(trunkHidden, _options.ActionSize,
            _options.UseDistributional ? _options.NumAtoms : 1,
            (_options.ValueStreamLayers ?? new List<int>()).ToArray(),
            (_options.AdvantageStreamLayers ?? new List<int>()).ToArray(),
            _options.Seed));
        IEnumerable<ILayer<T>> layers = layerList;

        var finalArchitecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: duelingActionWidth,
            layers: layers.ToList()
        )
        {
            RandomSeed = NextNetworkInitializationSeed(),
        };

        return new NeuralNetwork<T>(finalArchitecture,
            optimizer: _options.OptimizerFactory?.Invoke() ?? CreateNetworkOptimizer(LearningRate),
            lossFunction: LossFunction);
    }

    /// <summary>
    /// The initial sigma of every noisy layer in the online network, as each layer recorded it; empty when
    /// noisy networks are off.
    /// </summary>
    internal IReadOnlyList<double> NoisyLayerSigmaInits =>
        ((NeuralNetworkBase<T>)_onlineNetwork).Layers
            .OfType<NoisyDenseLayer<T>>()
            .Select(layer => layer.SigmaInit)
            .ToList();

    /// <summary>
    /// IFullModel.Predict surfaces the raw Q-value vector (one element per
    /// discrete action) rather than the one-hot committed action. Action
    /// commitment for env.Step is the job of <see cref="SelectAction"/>;
    /// Predict is the value-function diagnostic — needed by off-policy
    /// learners that consume Q(s, ·) directly, by evaluation harnesses
    /// inspecting policy distinguishability, and by callers that want to
    /// compare two states' value vectors instead of just their argmaxes.
    /// </summary>
    public override Vector<T> Predict(Vector<T> input) => ComputeQValues(input);

    /// <inheritdoc />
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Fortunato et al. 2017 ("Noisy Networks for Exploration") replaces
        // epsilon-greedy with parametric noise on the linear-layer weights;
        // the Rainbow paper (Hessel et al. 2018 §3.4) adopts that variant.
        // Honor the flag by disabling epsilon-greedy when set so the
        // distributional Q-head's intrinsic exploration takes over.
        double actualEpsilon = _options.UseNoisyNetworks ? 0.0 : _epsilon;

        if (training && Random.NextDouble() < actualEpsilon)
        {
            // ε-greedy exploration branch — returns a one-hot vector
            // representing the randomly-chosen discrete action, since
            // here the agent is committing to a single action choice.
            int randomAction = Random.Next(_options.ActionSize);
            var action = new Vector<T>(_options.ActionSize);
            action[randomAction] = NumOps.One;
            return action;
        }

        // Inference / greedy path: return a one-hot argmax(Q) action.
        // The IRLAgent<T> public contract on discrete envs is action-
        // commitment — callers feed the result into env.Step(action).
        // Returning the raw Q vector would change the API from action-
        // selector to value-diagnostic and would also break the inherited
        // ReinforcementLearningAgentBase.Train(state, target) contract
        // which expects target to be a one-hot action vector.
        var qValues = ComputeQValues(state);
        int bestIdx = qValues.Length == 0 ? 0 : RLTensorOps.ArgMax(qValues);
        return RLTensorOps.OneHot<T>(_options.ActionSize, bestIdx);
    }

    private Vector<T> ComputeQValues(Vector<T> state)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var outputTensor = _onlineNetwork.Predict(stateTensor);
        var output = outputTensor.ToVector();

        if (_options.UseDistributional)
        {
            // Distributional RL (C51, Bellemare et al. 2017): convert the
            // per-action atom distribution to an expected Q-value.
            // Q(s, a) = Σ_i z_i · p_i(s, a), where p_i is the softmax over
            // atom logits for that action. The network head emits raw
            // logits (final layer is IdentityActivation), so we softmax
            // each action's atom slice BEFORE the expectation — taking
            // the dot product of raw logits with z would silently produce
            // mathematically wrong Q-values for both action selection
            // and TD targets.
            return ExpectedAtomValues(output);
        }

        return output;
    }

    /// <summary>
    /// Numerically-stable softmax over <paramref name="logits"/>[<paramref name="offset"/>..<paramref name="offset"/>+<paramref name="count"/>).
    /// Returns a Vector of length <paramref name="count"/> with the per-atom
    /// probabilities for one action's atom slice. Used by both
    /// <see cref="ComputeQValues"/> and the target-network counterpart so
    /// expected-Q math is mathematically correct in C51 mode.
    /// </summary>
    private Vector<T> SoftmaxAtomSlice(Vector<T> logits, int offset, int count)
    {
            // exp(l - max l) / sum, over one action's atom logits; the max skips NaN (a strict greater-than scan from
            // -infinity), and an all-zero sum is left unnormalised.
            var engine = AiDotNetEngine.Current;
            var slice = logits.GetSubVector(offset, count);
            T maxLogit = RLTensorOps.MaxAbove(slice, NumOps.FromDouble(double.NegativeInfinity));
            var probs = engine.TensorExp(engine.TensorSubtractScalar(slice.AsTensor(), maxLogit));
            T sumExp = engine.TensorSum(probs);
            if (NumOps.GreaterThan(sumExp, NumOps.Zero))
            {
                probs = engine.TensorMultiplyScalar(probs, NumOps.Divide(NumOps.One, sumExp));
            }
            return probs.ToVector();
        }

        /// <summary>
        /// C51 expected values Q(s, a) = sum_i z_i p_i(s, a) for every action, with p the per-action softmax over the
        /// atom logits and z_i = VMin + i * deltaZ.
        /// </summary>
        private Vector<T> ExpectedAtomValues(Vector<T> output)
        {
            var engine = AiDotNetEngine.Current;
            int numAtoms = _options.NumAtoms;
            double deltaZ = (_options.VMax - _options.VMin) / (numAtoms - 1);
            var support = engine.TensorAddScalar(
                engine.TensorMultiplyScalar(engine.TensorArange<T>(0, numAtoms, 1), NumOps.FromDouble(deltaZ)),
                NumOps.FromDouble(_options.VMin)).ToVector();
            return Vector<T>.FromEnumerable(Enumerable.Range(0, _options.ActionSize)
                .Select(action => engine.DotProduct(SoftmaxAtomSlice(output, action * numAtoms, numAtoms), support)));
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // N-step learning: accumulate transitions
        _nStepBuffer.Add((state, action, reward, nextState, done));

        if (_nStepBuffer.Count >= _options.NSteps || done)
        {
            // Compute n-step return
            var (nStepState, nStepAction, nStepReturn, nStepNextState, nStepDone) = ComputeNStepReturn();
            _replayBuffer.Add(nStepState, nStepAction, nStepReturn, nStepNextState, nStepDone);

            // Clear n-step buffer on episode end
            if (done)
            {
                _nStepBuffer.Clear();
            }
            else
            {
                _nStepBuffer.RemoveAt(0);
            }
        }

        _stepCount++;

        // Decay epsilon
        if (!_options.UseNoisyNetworks)
        {
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
        }

        // Increase beta for importance sampling
        _beta = Math.Min(1.0, _beta + _options.PriorityBetaIncrement);
    }

    private (Vector<T> state, Vector<T> action, T nStepReturn, Vector<T> nextState, bool done) ComputeNStepReturn()
    {
        var firstState = _nStepBuffer[0].state;
        var firstAction = _nStepBuffer[0].action;

        // sum_i gamma^i r_i up to and including the first terminal transition
        int doneIndex = _nStepBuffer.FindIndex(transition => transition.done);
        int length = doneIndex >= 0 ? doneIndex + 1 : _nStepBuffer.Count;
        var rewards = Vector<T>.FromEnumerable(_nStepBuffer.Take(length).Select(transition => transition.reward));
        T nStepReturn = RLTensorOps.DiscountedSum(rewards, DiscountFactor, out _);

        if (doneIndex >= 0)
        {
            return (firstState, firstAction, nStepReturn, _nStepBuffer[doneIndex].nextState, true);
        }

        var lastTransition = _nStepBuffer[_nStepBuffer.Count - 1];
        return (firstState, firstAction, nStepReturn, lastTransition.nextState, false);
    }

    /// <summary>
    /// Supervised Train(state, target) does one direct regression step on the
    /// online Q-network: minimise ‖Q(state, ·) − target‖² so callers that
    /// provide an explicit Q-value target — offline pretraining from logged
    /// data, distillation, BC warm starts, generated invariant tests — see
    /// the network parameters actually move after a few calls. The base
    /// <see cref="ReinforcementLearningAgentBase{T}.Train(Vector{T}, Vector{T})"/>
    /// synthesises one transition into the replay buffer and calls
    /// <see cref="Train()"/>, but Rainbow's buffer-fill gate keeps that path
    /// no-opping for the entire warmup window. Same paper-adjacent
    /// justification as the SAC supervised BC override (Rajeswaran et al.
    /// 2018 §3.2): supervised pretraining of the value head from
    /// demonstrations is a standard Rainbow trick when offline data exists.
    /// </summary>
    public override void Train(Vector<T> state, Vector<T> target)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (target.Length == 0)
            throw new ArgumentException("target must contain at least one element.", nameof(target));

        // Online-network output width: ActionSize × NumAtoms when
        // distributional (C51 emits per-(action, atom) logits, Bellemare et al.
        // 2017), ActionSize otherwise. Pad / truncate the supervised target
        // to match so the engine forward+backward path runs without a shape
        // mismatch — the trailing-element padding is harmless because the
        // padded slots receive zero gradient when the regression target is
        // zero for them.
        int outputSize = _options.UseDistributional
            ? _options.ActionSize * _options.NumAtoms
            : _options.ActionSize;

        Vector<T> packedTarget;
        if (target.Length == outputSize)
        {
            packedTarget = target;
        }
        else
        {
            packedTarget = new Vector<T>(outputSize);
            int copyLen = Math.Min(outputSize, target.Length);
            target.AsSpan().Slice(0, copyLen).CopyTo(packedTarget.AsWritableSpan());
        }

        var stateTensor = Tensor<T>.FromVector(state);
        var targetTensor = Tensor<T>.FromVector(packedTarget);
        _onlineNetwork.Train(stateTensor, targetTensor);
        TrainingSteps++;
    }

    /// <inheritdoc />
    public override T Train()
    {
        // Rainbow (Hessel et al. 2018) learns only once the replay memory holds WarmupSteps transitions
        // (Dopamine's min_replay_history, 80k frames in the paper). Below that the agent explores without
        // updates. Above it a buffer smaller than BatchSize samples what it has, the usual practical
        // adaptation (Stable Baselines3, RLlib). The gate used to require the buffer to be below BatchSize
        // as well, so any WarmupSteps above BatchSize was silently capped at BatchSize.
        if (_replayBuffer.Count == 0 || _replayBuffer.Count < _options.WarmupSteps)
        {
            return NumOps.Zero;
        }

        int effectiveBatchSize = Math.Min(_replayBuffer.Count, _options.BatchSize);

        // Prioritized experience replay
        var (batch, indices, weights) = _replayBuffer.Sample(
            effectiveBatchSize,
            _options.PriorityAlpha,
            _beta);

        int stateSize = _options.StateSize;
        int n = batch.Count;

        // n-step discount gamma^NSteps, shared by both update paths.
        T nStepDiscount = _options.NSteps <= 0
            ? NumOps.One
            : RLTensorOps.DiscountPowers(DiscountFactor, _options.NSteps + 1)[_options.NSteps];

        // One batched states tensor: the whole minibatch goes through the tape in a single
        // forward/backward pass, exactly as DQNAgent.Train does.
        var states = RLTensorOps.Batch(batch.Select(experience => experience.state), stateSize);

        var priorities = new List<double>(n);
        T totalLoss = _options.UseDistributional
            ? TrainDistributional(states, batch, weights, nStepDiscount, priorities)
            : TrainExpected(states, batch, weights, nStepDiscount, priorities);

        // Update priorities in replay buffer
        _replayBuffer.UpdatePriorities(indices, priorities, _options.PriorityEpsilon);

        // Update target network
        if (_stepCount % _options.TargetUpdateFrequency == 0)
        {
            CopyNetworkWeights(_onlineNetwork, _targetNetwork);
        }

        _updateCount++;

        return totalLoss;
    }

    /// <summary>
    /// Non-distributional Rainbow update: a Double-DQN TD target written into the taken action's
    /// slot, then ONE batched tape-based training step on the online network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The previous implementation built a gradient vector, discarded it, and then subtracted
    /// <c>LearningRate * gradient[j % ActionSize]</c> from every parameter. That is not
    /// backpropagation: it moves only the parameters whose flat index is congruent to the taken
    /// action index modulo ActionSize, so a single-element trainable component -- for example the
    /// dueling head's scalar value bias -- was updated only when its index happened to line up.
    /// <see cref="ApplyGradients"/> already documents and corrects the identical
    /// <c>gradients[i % length]</c> defect.
    /// </para>
    /// </remarks>
    private T TrainExpected(
        Tensor<T> states,
        List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch,
        IReadOnlyList<double> weights,
        T nStepDiscount,
        List<double> priorities)
    {
        int actionSize = _options.ActionSize;
        int n = batch.Count;

        // Targets are built OUTSIDE the tape: the target network must not be tape-tracked.
        var engine = AiDotNetEngine.Current;
        var currentQBatch = _onlineNetwork.Predict(states);
        var takenActions = batch.Select(experience => ArgMax(experience.action)).ToList();
        var rewards = Vector<T>.FromEnumerable(batch.Select(experience => experience.reward));
        var done = batch.Select(experience => experience.done).ToArray();

        // Double Q-learning: the ONLINE network selects the next action, the TARGET network evaluates it
        // (van Hasselt et al. 2016). The next states go through each network as one batch; a terminal row's
        // bootstrap is discarded, so its own state stands in.
        var nextStates = RLTensorOps.Batch(batch.Select(experience => experience.done ? experience.state : experience.nextState), _options.StateSize);
        var bestNextActions = RLTensorOps.Rows(_onlineNetwork.Predict(nextStates), n).Select(ArgMax).ToList();
        var nextQTarget = _targetNetwork.Predict(nextStates);
        int targetWidth = n == 0 ? 0 : nextQTarget.ToVector().Length / n;
        var evaluated = RLTensorOps.PickPerRow(nextQTarget, n, targetWidth, bestNextActions);
        var targets = RLTensorOps.TdTargets(rewards, evaluated, done, nStepDiscount);

        // Importance-sampling weighting belongs on the sample's contribution to the loss, and
        // the priority is the unweighted magnitude of the TD error (Schaul et al. 2016).
        var tdErrors = engine.Subtract(targets, RLTensorOps.PickPerRow(currentQBatch, n, actionSize, takenActions));
        priorities.AddRange(engine.Abs(tdErrors).AsTensor().Cast<double>().ToVector());

        var targetQ = RLTensorOps.ReplaceTakenActionValues(currentQBatch, n, actionSize, takenActions, targets);

        // Single batched training step -- tape-based forward + loss + optimizer update.
        _onlineNetwork.Train(states, targetQ);
        return _onlineNetwork.GetLastLoss();
    }

    /// <summary>
    /// Distributional (C51) Rainbow update: project the n-step return onto the fixed support and
    /// minimise the cross-entropy between the projected target distribution and the online
    /// network's distribution for the action that was actually taken.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Follows Bellemare et al. 2017 (Algorithm 1), with the Double-Q action selection and the
    /// importance-sampling weighting of Hessel et al. 2018 (Rainbow, sections 3.2-3.5). The
    /// projection is ordinary arithmetic on constants, so it runs outside the tape; only the
    /// cross-entropy against the network's own logits is differentiated.
    /// </para>
    /// </remarks>
    private T TrainDistributional(
        Tensor<T> states,
        List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> batch,
        IReadOnlyList<double> weights,
        T nStepDiscount,
        List<double> priorities)
    {
        if (_onlineNetwork is not NeuralNetworkBase<T> trainable)
        {
            throw new NotSupportedException(
                "Distributional Rainbow requires an online network deriving from NeuralNetworkBase<T> "
                + "so the categorical cross-entropy can be differentiated through the gradient tape.");
        }

        int actionSize = _options.ActionSize;
        int numAtoms = _options.NumAtoms;
        int n = batch.Count;
        double vMin = _options.VMin;
        double vMax = _options.VMax;
        double deltaZ = (vMax - vMin) / (numAtoms - 1);
        double discount = NumOps.ToDouble(nStepDiscount);

        // Rows are (sample, action) pairs so the softmax below reduces over each action's atom
        // slice -- the same per-action normalisation ComputeQValues uses. A softmax over the flat
        // ActionSize*NumAtoms row would couple actions that C51 keeps independent.
        var samples = Enumerable.Range(0, n)
            .Select(i => CategoricalTarget(batch[i], ArgMax(batch[i].action), weights[i], discount, n))
            .ToList();

        // Rainbow section 3.5 prioritises transitions by the KL loss, which for a fixed target
        // distribution is this cross-entropy up to a constant.
        var crossEntropies = samples.Select(sample => sample.CrossEntropy).ToList();
        priorities.AddRange(crossEntropies);
        double lossSum = n == 0
            ? 0.0
            : AiDotNetEngine.Current.DotProduct(Vector<double>.FromEnumerable(weights.Take(n)), Vector<double>.FromEnumerable(crossEntropies));
        var coefficients = n == 0
            ? new Tensor<T>([0, numAtoms])
            : AiDotNetEngine.Current.TensorConcatenate(samples.Select(sample => sample.Coefficients).ToArray(), 0);

        trainable.TrainWithCustomLoss(states, logits =>
        {
            var engine = AiDotNetEngine.Current;
            var perAction = engine.Reshape(logits, [n * actionSize, numAtoms]);
            var probs = engine.Softmax(perAction);
            var safeProbs = engine.TensorAddScalar(probs, NumOps.FromDouble(1e-8));
            var logProbs = engine.TensorLog(safeProbs);
            var weighted = engine.TensorMultiply(logProbs, coefficients);
            return engine.ReduceSum(weighted, [0, 1], keepDims: false);
        });

        return NumOps.FromDouble(lossSum / n);
    }

    /// <inheritdoc/>
    Vector<T> IActionValueProvider<T>.GetActionValues(Vector<T> state) => ComputeQValuesFromNetwork(_onlineNetwork, state);

    private Vector<T> ComputeQValuesFromNetwork(INeuralNetwork<T> network, Vector<T> state)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var outputTensor = network.Predict(stateTensor);
        var output = outputTensor.ToVector();

        if (_options.UseDistributional)
        {
            // Same softmax-before-expectation correction as the online
            // ComputeQValues — the C51 head emits raw atom logits and we
            // need true probabilities to compute Q(s,a) = Σ z·p.
            return ExpectedAtomValues(output);
        }

        return output;
    }

    private int ArgMax(Vector<T> values)
    {
        return RLTensorOps.ArgMax(values);
    }

    /// <summary>
    /// One transition's C51 target (Bellemare et al. 2017, Algorithm 1): the target network's distribution for
    /// the online network's next action, projected onto the support shifted by the n-step return; returns the
    /// sample's [ActionSize, NumAtoms] loss coefficients (-w m / n on the taken action's row, zero elsewhere) and
    /// its cross-entropy against the online distribution.
    /// </summary>
    private (Tensor<T> Coefficients, double CrossEntropy) CategoricalTarget(
        (Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done) experience,
        int actionIndex, double weight, double discount, int n)
    {
        var engine = AiDotNetEngine.Current;
        int actionSize = _options.ActionSize;
        int numAtoms = _options.NumAtoms;
        double vMin = _options.VMin;
        double vMax = _options.VMax;
        double deltaZ = (vMax - vMin) / (numAtoms - 1);
        double reward = NumOps.ToDouble(experience.reward);

        // Double Q-learning: the ONLINE network picks the next action, the TARGET network
        // supplies the distribution that gets projected.
        int bestActionIndex = ArgMax(ComputeQValues(experience.nextState));
        var nextLogits = _targetNetwork.Predict(Tensor<T>.FromVector(experience.nextState)).ToVector();
        var mass = SoftmaxAtomSlice(nextLogits, bestActionIndex * numAtoms, numAtoms).AsTensor().Cast<double>();

        // Categorical projection: shift the support by the n-step return, clamp it back into
        // [VMin, VMax], and split each atom's mass between its two neighbouring supports. When
        // the transition is terminal every atom maps to the reward, so the projected
        // distribution collapses to a point mass there -- which is what C51 requires.
        var support = engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorArange<double>(0, numAtoms, 1), deltaZ), vMin);
        var shifted = experience.done
            ? engine.TensorFullLike(support, reward)
            : engine.TensorAddScalar(engine.TensorMultiplyScalar(support, discount), reward);
        var tz = engine.TensorClamp(shifted, vMin, vMax);
        var b = engine.TensorDivideScalar(engine.TensorSubtractScalar(tz, vMin), deltaZ);
        var lower = engine.TensorFloor(b);
        var upper = engine.TensorCeiling(b);
        var onSupport = engine.TensorEquals(lower, upper);
        var zeros = engine.TensorFullLike(b, 0.0);
        var lowerMass = engine.TensorWhere(onSupport, mass, engine.TensorMultiply(mass, engine.TensorSubtract(upper, b)));
        var upperMass = engine.TensorWhere(onSupport, zeros, engine.TensorMultiply(mass, engine.TensorSubtract(b, lower)));
        var projected = engine.TensorScatterAdd(
            engine.TensorScatterAdd(zeros, lower.Cast<int>(), lowerMass, 0), upper.Cast<int>(), upperMass, 0);

        // -sum_k m_k log max(p_k, 1e-8) over the atoms with positive projected mass
        var currentLogits = _onlineNetwork.Predict(Tensor<T>.FromVector(experience.state)).ToVector();
        var currentProbs = SoftmaxAtomSlice(currentLogits, actionIndex * numAtoms, numAtoms).AsTensor().Cast<double>();
        var flooredProbs = engine.TensorWhere(
            engine.TensorLessThan(currentProbs, 1e-8), engine.TensorFullLike(currentProbs, 1e-8), currentProbs);
        var terms = engine.TensorWhere(
            engine.TensorGreaterThan(projected, 0.0), engine.TensorMultiply(projected, engine.TensorLog(flooredProbs)), zeros);
        double crossEntropy = -engine.TensorSum(terms);

        // -w_i * m_k / n on the taken action's row; every other row stays zero, so only the
        // action that was actually taken receives gradient.
        var row = engine.TensorDivideScalar(engine.TensorMultiplyScalar(projected, -weight), (double)n).Cast<T>();
        var rows = engine.TensorTile(engine.Reshape(row, [1, numAtoms]), [actionSize, 1]);
        var taken = engine.TensorTile(engine.Reshape(RLTensorOps.OneHot<T>(actionSize, actionIndex).AsTensor(), [actionSize, 1]), [1, numAtoms]);
        var coefficients = engine.TensorWhere(taken, rows, engine.TensorFullLike(rows, NumOps.Zero));
        return (coefficients, crossEntropy);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetMetrics()
    {
        var baseMetrics = base.GetMetrics();
        baseMetrics["steps"] = NumOps.FromDouble(_stepCount);
        baseMetrics["updates"] = NumOps.FromDouble(_updateCount);
        baseMetrics["buffer_size"] = NumOps.FromDouble(_replayBuffer.Count);
        baseMetrics["epsilon"] = NumOps.FromDouble(_epsilon);
        return baseMetrics;
    }

    /// <inheritdoc />
    public override void ResetEpisode()
    {
        _nStepBuffer.Clear();
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = _options.StateSize,
        };
    }

    /// <inheritdoc />
    public override int FeatureCount => _options.StateSize;

    /// <summary>
    /// Computes gradients of the loss with respect to this agent's parameters, without updating them.
    /// </summary>
    /// <param name="input">The input state.</param>
    /// <param name="target">The target output.</param>
    /// <param name="lossFunction">The loss to differentiate, or null to use the agent's own.</param>
    /// <returns>A gradient vector the same length as <c>GetParameters()</c>.</returns>
    /// <remarks>
    /// Returns PARAMETER-space gradients, as <c>IGradientComputable</c> documents and as
    /// <c>ApplyGradients</c> requires. The gradient comes from the network's own tape pass, so no
    /// derivative is written by hand here.
    /// </remarks>
    public Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        return ComputeGradientsForNetwork(
            _onlineNetwork,
            new[] { _onlineNetwork, _targetNetwork },
            input,
            target,
            lossFunction);
    }

    /// <inheritdoc />
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var currentParams = GetParameters();

        // ComputeGradients returns a parameter-space vector laid out exactly like GetParameters(),
        // so this is a straight element-wise step. It previously indexed gradients[i % length],
        // cycling a short OUTPUT-space vector across the whole parameter vector -- which applied
        // output neuron (i mod n)'s gradient to parameter i, an arbitrary pairing.
        if (gradients.Length != currentParams.Length)
        {
            throw new ArgumentException(
                $"Gradient vector length ({gradients.Length}) must match parameter vector length "
                + $"({currentParams.Length}).",
                nameof(gradients));
        }

        // theta - lr * g
        var engine = AiDotNetEngine.Current;
        var newParams = engine.Subtract(currentParams, engine.Multiply(gradients, learningRate));

        SetParameters(newParams);
    }

    /// <inheritdoc />
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    /// <inheritdoc />
    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
