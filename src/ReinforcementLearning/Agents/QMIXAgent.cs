using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Training;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.QMIX;

/// <summary>
/// QMIX (Rashid et al. 2018): cooperative multi-agent Q-learning whose joint action-value is a monotonic mix of the
/// agents' own values.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Every agent runs one shared recurrent network (a ReLU layer, a GRU cell and a linear layer, fed its observation,
/// its last action and its id) and acts ε-greedily on its own values. In training, the values of the actions taken are
/// mixed into Q_tot by a network whose weights hypernetworks generate from the global state and keep non-negative, so
/// Q_tot rises with every agent's value; the agent and mixing networks train together on the TD error of Q_tot against
/// target networks, over batches of whole episodes unrolled from the episode start.
/// </para>
/// <para>
/// It follows the paper, with the authors' code (pymarl) where the paper is silent; pymarl's own configuration is
/// available through <see cref="QMIXOptions{T}.HypernetLayers"/>, <see cref="QMIXOptions{T}.DoubleQ"/> and
/// <see cref="QMIXOptions{T}.MeanLoss"/>. <c>QmixReferenceTests</c> matches one training step of pymarl's QMixer and
/// RNNAgent to 1e-9 in both configurations.
/// </para>
/// <para>
/// Usage: each step, call <see cref="SelectActionForAgent"/> for every agent, step the environment, and store the
/// joint transition with <see cref="StoreMultiAgentExperience"/>; call <see cref="Train"/> after each step. A transition
/// that ends the episode (<c>done</c>, or <c>truncated</c> at a time limit) completes it, adds it to the buffer and
/// resets the agents' recurrent states; <see cref="Train"/> runs one update per completed episode once the buffer holds
/// a batch.
/// </para>
/// <para><b>For Beginners:</b>
/// Each player estimates how good each of its moves is from what it can see; a coach combines the estimates of the
/// moves actually played into a team score, in a way where a better move for any player can only raise the team's
/// score. So each player picking its own best move also picks the team's best plan.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var agent = new QMIXAgent&lt;double&gt;(new QMIXOptions&lt;double&gt; { NumAgents = 3, StateSize = 8, ActionSize = 4, GlobalStateSize = 12 });
/// var action = agent.SelectActionForAgent(0, observation);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning",
    "https://arxiv.org/abs/1803.11485",
    Year = 2018,
    Authors = "Rashid, T., Samvelyan, M., de Witt, C. S., Farquhar, G., Foerster, J., & Whiteson, S.")]
[PaperOptimizer(OptimizerKind.RmsProp, LearningRate = 5e-4,
                Source = "Rashid et al. 2018, Sec. 5: the learning rate for RMSprop is set to 5e-4.")]
public partial class QMIXAgent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{
    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_agentNetwork);
        RegisterParameterComponent(_mixingNetwork);
    }

    private readonly QMIXOptions<T> _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    // The agent network every agent shares, and the mixing network.
    private INeuralNetwork<T> _agentNetwork;
    [Buffer]
    private INeuralNetwork<T> _targetAgentNetwork;
    private INeuralNetwork<T> _mixingNetwork;
    [Buffer]
    private INeuralNetwork<T> _targetMixingNetwork;

    // pymarl's ReplayBuffer: a ring of whole episodes.
    private readonly Episode?[] _episodes;
    private int _episodesInBuffer;
    private int _episodeSlot;

    // The episode being collected.
    private readonly List<Vector<T>[]> _episodeObservations = new();
    private readonly List<Vector<T>> _episodeStates = new();
    private readonly List<int[]> _episodeActions = new();
    private readonly List<T> _episodeRewards = new();

    // Acting: each agent's recurrent state and last action (-1 = none yet this episode).
    private readonly Tensor<T>[] _hidden;
    private readonly int[] _lastAction;

    private int _stepCount;
    private int _episodeStartStep;
    private int _completedEpisodes;
    private int _trainedAtEpisode;
    private int _lastTargetUpdateEpisode;
    private TapeTrainingStepper<T>? _stepper;

    private sealed class Episode
    {
        public required Vector<T>[][] Observations { get; init; } // [L + 1][agent]
        public required Vector<T>[] States { get; init; }         // [L + 1]
        public required int[][] Actions { get; init; }            // [L][agent]
        public required T[] Rewards { get; init; }                // [L]
        public required bool Terminated { get; init; }
        public int Length => Rewards.Length;
    }

    /// <summary>Initializes a new instance with default settings.</summary>
    public QMIXAgent()
        : this(new QMIXOptions<T> { NumAgents = 2, StateSize = 4, ActionSize = 2, GlobalStateSize = 8 })
    {
    }

    /// <summary>Creates a QMIX agent.</summary>
    /// <param name="options">The configuration; the defaults are the paper's.</param>
    /// <param name="optimizer">Replaces the paper's RMSprop (and <see cref="QMIXOptions{T}.Optimizer"/>).</param>
    public QMIXAgent(QMIXOptions<T> options, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(options)
    {
        Guard.NotNull(options);
        if (options.NumAgents <= 0) throw new ArgumentOutOfRangeException(nameof(options), options.NumAgents, "NumAgents must be positive.");
        if (options.StateSize <= 0) throw new ArgumentOutOfRangeException(nameof(options), options.StateSize, "StateSize must be positive.");
        if (options.ActionSize <= 0) throw new ArgumentOutOfRangeException(nameof(options), options.ActionSize, "ActionSize must be positive.");
        if (options.GlobalStateSize <= 0) throw new ArgumentOutOfRangeException(nameof(options), options.GlobalStateSize, "GlobalStateSize must be positive.");
        if (options.RnnHiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(options), options.RnnHiddenDim, "RnnHiddenDim must be positive.");
        if (options.BatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(options), options.BatchSize, "BatchSize must be positive.");
        if (options.ReplayBufferSize < options.BatchSize)
            throw new ArgumentOutOfRangeException(nameof(options), options.ReplayBufferSize, "The buffer must hold at least a batch of episodes.");
        if (options.TargetUpdateFrequency <= 0) throw new ArgumentOutOfRangeException(nameof(options), options.TargetUpdateFrequency, "TargetUpdateFrequency must be positive.");
        if (options.EpsilonAnnealSteps < 0) throw new ArgumentOutOfRangeException(nameof(options), options.EpsilonAnnealSteps, "EpsilonAnnealSteps cannot be negative.");
        if (!(options.MaxGradientNorm >= 0)) throw new ArgumentOutOfRangeException(nameof(options), options.MaxGradientNorm, "MaxGradientNorm cannot be negative (0 = no clipping).");
        _options = options;
        Options = _options;

        // The paper's learning rate when none is given (the base's generic default is 1e-3).
        if (options.LearningRate is null || NumOps.Equals(options.LearningRate, NumOps.Zero))
            LearningRate = NumOps.FromDouble(5e-4);

        _optimizer = optimizer ?? options.Optimizer ?? new RootMeanSquarePropagationOptimizer<T, Tensor<T>, Tensor<T>>(null,
            new RootMeanSquarePropagationOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = NumOps.ToDouble(LearningRate),
                Decay = options.RmsPropAlpha,
                Epsilon = options.RmsPropEpsilon,
                EnableGradientClipping = false,
            });

        _episodes = new Episode?[options.ReplayBufferSize];
        _hidden = new Tensor<T>[options.NumAgents];
        _lastAction = new int[options.NumAgents];
        ResetActingState();
        InitializeNetworks();
    }

    /// <summary>The width of each agent's network input: observation, last action one-hot, agent id one-hot.</summary>
    private int AgentInputSize =>
        _options.StateSize + (_options.IncludeLastAction ? _options.ActionSize : 0) + (_options.IncludeAgentId ? _options.NumAgents : 0);

    [System.Diagnostics.CodeAnalysis.MemberNotNull(
        nameof(_agentNetwork), nameof(_targetAgentNetwork), nameof(_mixingNetwork), nameof(_targetMixingNetwork))]
    private void InitializeNetworks()
    {
        _agentNetwork = CreateAgentNetwork();
        _targetAgentNetwork = CreateAgentNetwork();
        _mixingNetwork = CreateMixingNetwork();
        _targetMixingNetwork = CreateMixingNetwork();

        AgentLayer(_agentNetwork).InitializeLikeTorch(Random);
        MixingLayer(_mixingNetwork).InitializeLikeTorch(Random);
        Materialize(_targetAgentNetwork, AgentInputSize + _options.RnnHiddenDim);
        Materialize(_targetMixingNetwork, _options.NumAgents + _options.GlobalStateSize);
        CopyNetworkWeights(_agentNetwork, _targetAgentNetwork);
        CopyNetworkWeights(_mixingNetwork, _targetMixingNetwork);

        Networks.Add(_agentNetwork);
        Networks.Add(_targetAgentNetwork);
        Networks.Add(_mixingNetwork);
        Networks.Add(_targetMixingNetwork);
    }

    private static void Materialize(INeuralNetwork<T> network, int inputs)
    {
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            network.Predict(new Tensor<T>(new[] { 1, inputs }));
    }

    private INeuralNetwork<T> CreateAgentNetwork()
    {
        int inputs = AgentInputSize, hidden = _options.RnnHiddenDim;
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputs + hidden,
            outputSize: _options.ActionSize + hidden,
            layers: new List<ILayer<T>> { new QmixAgentLayer<T>(inputs, hidden, _options.ActionSize) });
        return new NeuralNetwork<T>(architecture, lossFunction: LossFunction,
            options: new AiDotNet.NeuralNetworks.Options.NeuralNetworkDefaultOptions { MaxGradNorm = 0.0 });
    }

    private INeuralNetwork<T> CreateMixingNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.NumAgents + _options.GlobalStateSize,
            outputSize: 1,
            layers: new List<ILayer<T>>
            {
                new QmixMixingLayer<T>(_options.NumAgents, _options.GlobalStateSize, _options.MixingEmbedDim,
                    _options.HypernetLayers, _options.HypernetEmbed),
            });
        return new NeuralNetwork<T>(architecture, lossFunction: LossFunction,
            options: new AiDotNet.NeuralNetworks.Options.NeuralNetworkDefaultOptions { MaxGradNorm = 0.0 });
    }

    private static QmixAgentLayer<T> AgentLayer(INeuralNetwork<T> network)
        => (QmixAgentLayer<T>)((NeuralNetworkBase<T>)network).Layers[0];

    private static QmixMixingLayer<T> MixingLayer(INeuralNetwork<T> network)
        => (QmixMixingLayer<T>)((NeuralNetworkBase<T>)network).Layers[0];

    /// <summary>
    /// Loads pymarl's RNNAgent and QMixer state dicts (each tensor flattened row-major, by its PyTorch name) into the
    /// live networks and copies them to the targets.
    /// </summary>
    internal void LoadTorchWeights(IReadOnlyDictionary<string, double[]> agent, IReadOnlyDictionary<string, double[]> mixer)
    {
        AgentLayer(_agentNetwork).LoadTorch(agent);
        MixingLayer(_mixingNetwork).LoadTorch(mixer);
        CopyNetworkWeights(_agentNetwork, _targetAgentNetwork);
        CopyNetworkWeights(_mixingNetwork, _targetMixingNetwork);
    }

    /// <summary>The live agent network's parameters followed by the live mixing network's.</summary>
    internal (Vector<T> Agent, Vector<T> Mixer) LiveNetworkParameters()
        => (_agentNetwork.GetParameters(), _mixingNetwork.GetParameters());

    /// <summary>The current exploration rate: linear from EpsilonStart to EpsilonEnd over EpsilonAnnealSteps, as of the episode's start (pymarl).</summary>
    public double Epsilon
    {
        get
        {
            if (_options.EpsilonAnnealSteps == 0) return _options.EpsilonEnd;
            double delta = (_options.EpsilonStart - _options.EpsilonEnd) / _options.EpsilonAnnealSteps;
            return Math.Max(_options.EpsilonEnd, _options.EpsilonStart - delta * _episodeStartStep);
        }
    }

    #region Acting

    private void ResetActingState()
    {
        for (int i = 0; i < _options.NumAgents; i++)
        {
            _hidden[i] = new Tensor<T>(new[] { 1, _options.RnnHiddenDim });
            _lastAction[i] = -1;
        }
    }

    // One agent's network input at a step: observation, last action one-hot (none = zeros), id one-hot.
    private void WriteAgentInput(T[] row, int offset, Vector<T> observation, int lastAction, int agentId)
    {
        for (int j = 0; j < _options.StateSize; j++) row[offset + j] = observation[j];
        offset += _options.StateSize;
        if (_options.IncludeLastAction)
        {
            for (int a = 0; a < _options.ActionSize; a++) row[offset + a] = NumOps.Zero;
            if (lastAction >= 0) row[offset + lastAction] = NumOps.One;
            offset += _options.ActionSize;
        }
        if (_options.IncludeAgentId)
        {
            for (int k = 0; k < _options.NumAgents; k++) row[offset + k] = k == agentId ? NumOps.One : NumOps.Zero;
        }
    }

    private void ValidateObservation(Vector<T> observation, string name)
    {
        Guard.NotNull(observation, name);
        if (observation.Length != _options.StateSize)
            throw new ArgumentException($"Expected an observation of {_options.StateSize} values, got {observation.Length}.", name);
    }

    // One step of the agent network from a state; returns the action values and the next state.
    private (T[] Values, Tensor<T> Next) AgentStep(int agentId, Vector<T> observation, int lastAction, Tensor<T> hidden)
    {
        int d = AgentInputSize, h = _options.RnnHiddenDim, actions = _options.ActionSize;
        var row = new T[d + h];
        WriteAgentInput(row, 0, observation, lastAction, agentId);
        for (int j = 0; j < h; j++) row[d + j] = hidden.GetFlat(j);
        Tensor<T> output;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            output = _agentNetwork.Predict(new Tensor<T>(new[] { 1, d + h }, new Vector<T>(row)));
        var values = new T[actions];
        for (int a = 0; a < actions; a++) values[a] = output.GetFlat(a);
        var next = new Tensor<T>(new[] { 1, h });
        for (int j = 0; j < h; j++) next.SetFlat(j, output.GetFlat(actions + j));
        return (values, next);
    }

    /// <summary>
    /// Selects one agent's action, ε-greedily when training, advancing that agent's recurrent state.
    /// </summary>
    /// <param name="agentId">The agent, from 0 to NumAgents − 1.</param>
    /// <param name="state">The agent's own observation.</param>
    /// <param name="training">Whether to explore.</param>
    /// <returns>The action, one-hot.</returns>
    public Vector<T> SelectActionForAgent(int agentId, Vector<T> state, bool training = true)
    {
        if (agentId < 0 || agentId >= _options.NumAgents)
            throw new ArgumentOutOfRangeException(nameof(agentId), agentId, $"Agent ids run from 0 to {_options.NumAgents - 1}.");
        ValidateObservation(state, nameof(state));
        var (values, next) = AgentStep(agentId, state, _lastAction[agentId], _hidden[agentId]);
        _hidden[agentId] = next;
        int action = training && Random.NextDouble() < Epsilon ? Random.Next(_options.ActionSize) : ArgMax(values);
        _lastAction[agentId] = action;
        return OneHot(action);
    }

    /// <summary>Selects agent 0's action (QMIX's agents act through <see cref="SelectActionForAgent"/>).</summary>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true) => SelectActionForAgent(0, state, training);

    /// <summary>
    /// Agent 0's greedy action for an observation at the start of an episode (a zero recurrent state and no last
    /// action), without changing any agent's recurrent state.
    /// </summary>
    public override Vector<T> Predict(Vector<T> input)
    {
        ValidateObservation(input, nameof(input));
        var (values, _) = AgentStep(0, input, -1, new Tensor<T>(new[] { 1, _options.RnnHiddenDim }));
        return OneHot(ArgMax(values));
    }

    /// <summary>Starts a new episode: every agent's recurrent state and last action reset, and any partial episode dropped.</summary>
    public override void ResetEpisode()
    {
        ResetActingState();
        ClearEpisode();
        _episodeStartStep = _stepCount;
    }

    #endregion

    #region Experience

    /// <summary>
    /// Stores one joint transition of the episode being collected.
    /// </summary>
    /// <param name="agentStates">Each agent's observation.</param>
    /// <param name="agentActions">Each agent's action, one-hot (the largest entry is taken).</param>
    /// <param name="teamReward">The shared reward.</param>
    /// <param name="nextAgentStates">Each agent's next observation (stored when the transition ends the episode).</param>
    /// <param name="globalState">The global state.</param>
    /// <param name="nextGlobalState">The next global state (stored when the transition ends the episode).</param>
    /// <param name="done">Whether the episode terminated: its last target has no bootstrap.</param>
    /// <param name="truncated">Whether the episode ended at a time limit without terminating: it ends, and its last
    /// target still bootstraps from the next state.</param>
    public void StoreMultiAgentExperience(
        List<Vector<T>> agentStates,
        List<Vector<T>> agentActions,
        T teamReward,
        List<Vector<T>> nextAgentStates,
        Vector<T> globalState,
        Vector<T> nextGlobalState,
        bool done,
        bool truncated = false)
    {
        var observations = AgentVectors(agentStates, nameof(agentStates));
        var next = AgentVectors(nextAgentStates, nameof(nextAgentStates));
        ValidateState(globalState, nameof(globalState));
        ValidateState(nextGlobalState, nameof(nextGlobalState));
        Guard.NotNull(agentActions);
        if (agentActions.Count != _options.NumAgents)
            throw new ArgumentException($"Expected {_options.NumAgents} actions, got {agentActions.Count}.", nameof(agentActions));
        var actions = new int[_options.NumAgents];
        for (int i = 0; i < actions.Length; i++)
        {
            if (agentActions[i] is null || agentActions[i].Length != _options.ActionSize)
                throw new ArgumentException($"Action {i} must be a one-hot vector of {_options.ActionSize} values.", nameof(agentActions));
            actions[i] = ArgMax(agentActions[i]);
        }

        _episodeObservations.Add(observations);
        _episodeStates.Add(globalState.Clone());
        _episodeActions.Add(actions);
        _episodeRewards.Add(teamReward);
        _stepCount++;
        // The agents' next inputs carry the actions actually taken.
        for (int i = 0; i < actions.Length; i++) _lastAction[i] = actions[i];

        if (done || truncated)
        {
            _episodeObservations.Add(next);
            _episodeStates.Add(nextGlobalState.Clone());
            InsertEpisode(new Episode
            {
                Observations = _episodeObservations.ToArray(),
                States = _episodeStates.ToArray(),
                Actions = _episodeActions.ToArray(),
                Rewards = _episodeRewards.ToArray(),
                Terminated = done,
            });
            _completedEpisodes++;
            ResetEpisode();
        }
    }

    /// <summary>QMIX learns from joint transitions: use <see cref="StoreMultiAgentExperience"/>.</summary>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
        => throw new InvalidOperationException(
            "QMIX requires joint transitions stored via StoreMultiAgentExperience: every agent's observation and action, " +
            "the team reward and the global state.");

    private Vector<T>[] AgentVectors(List<Vector<T>> vectors, string name)
    {
        Guard.NotNull(vectors, name);
        if (vectors.Count != _options.NumAgents)
            throw new ArgumentException($"Expected {_options.NumAgents} observations, got {vectors.Count}.", name);
        var copies = new Vector<T>[vectors.Count];
        for (int i = 0; i < copies.Length; i++)
        {
            ValidateObservation(vectors[i], name);
            copies[i] = vectors[i].Clone();
        }
        return copies;
    }

    private void ValidateState(Vector<T> state, string name)
    {
        Guard.NotNull(state, name);
        if (state.Length != _options.GlobalStateSize)
            throw new ArgumentException($"Expected a global state of {_options.GlobalStateSize} values, got {state.Length}.", name);
    }

    private void ClearEpisode()
    {
        _episodeObservations.Clear();
        _episodeStates.Clear();
        _episodeActions.Clear();
        _episodeRewards.Clear();
    }

    private void InsertEpisode(Episode episode)
    {
        _episodes[_episodeSlot] = episode;
        _episodeSlot = (_episodeSlot + 1) % _episodes.Length;
        _episodesInBuffer = Math.Min(_episodesInBuffer + 1, _episodes.Length);
    }

    // pymarl's ReplayBuffer.sample: the buffer in slot order when it holds exactly a batch, else a batch drawn without
    // replacement.
    private Episode[] SampleEpisodes()
    {
        int batch = _options.BatchSize;
        var indices = Enumerable.Range(0, _episodesInBuffer).ToArray();
        if (_episodesInBuffer > batch)
        {
            for (int i = 0; i < batch; i++)
            {
                int j = i + Random.Next(_episodesInBuffer - i);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }
        return indices.Take(batch).Select(i => _episodes[i]!).ToArray();
    }

    #endregion

    #region Training

    /// <summary>
    /// Runs one QMIX update when an episode has completed since the last one and the buffer holds a batch of episodes;
    /// otherwise returns zero.
    /// </summary>
    /// <returns>The loss of the update: the squared TD error summed over the batch's steps (or averaged, with MeanLoss).</returns>
    public override T Train()
    {
        if (_completedEpisodes == _trainedAtEpisode || _episodesInBuffer < _options.BatchSize) return NumOps.Zero;
        _trainedAtEpisode = _completedEpisodes;

        var batch = SampleEpisodes();
        int b = batch.Length, n = _options.NumAgents, actions = _options.ActionSize, s = _options.GlobalStateSize;
        int h = _options.RnnHiddenDim, d = AgentInputSize, steps = batch.Max(e => e.Length);
        int rows = b * n;

        // Agent inputs [B·n, D] for t = 0..T (padding past an episode's end is zero, as an EpisodeBatch holds it), the
        // chosen actions one-hot [B·n, A] and the states [B, S].
        var inputs = new Tensor<T>[steps + 1];
        var states = new Tensor<T>[steps + 1];
        for (int t = 0; t <= steps; t++)
        {
            var row = new T[rows * d];
            var state = new T[b * s];
            for (int e = 0; e < b; e++)
            {
                var episode = batch[e];
                for (int i = 0; i < n; i++)
                {
                    var observation = t <= episode.Length ? episode.Observations[t][i] : new Vector<T>(_options.StateSize);
                    int last = t >= 1 && t - 1 < episode.Length ? episode.Actions[t - 1][i] : -1;
                    WriteAgentInput(row, (e * n + i) * d, observation, last, i);
                }
                if (t <= episode.Length)
                    for (int j = 0; j < s; j++) state[e * s + j] = episode.States[t][j];
            }
            inputs[t] = new Tensor<T>(new[] { rows, d }, new Vector<T>(row));
            states[t] = new Tensor<T>(new[] { b, s }, new Vector<T>(state));
        }

        // The targets, from the target networks (and, with double Q, the live agent network's greedy actions).
        var targetValues = Unroll(_targetAgentNetwork, inputs);
        var liveValues = _options.DoubleQ ? Unroll(_agentNetwork, inputs) : null;
        var y = new T[steps * b];
        var mask = new T[steps * b];
        var chosen = new Tensor<T>[steps];
        double gamma = NumOps.ToDouble(DiscountFactor);
        double filled = 0;
        for (int t = 0; t < steps; t++)
        {
            var next = new T[b * (n + s)];
            var oneHot = Enumerable.Repeat(NumOps.Zero, rows * actions).ToArray();
            for (int e = 0; e < b; e++)
            {
                for (int i = 0; i < n; i++)
                {
                    int r = e * n + i;
                    var target = targetValues[t + 1][r];
                    int pick = liveValues is null ? ArgMax(target) : ArgMax(liveValues[t + 1][r]);
                    next[e * (n + s) + i] = target[pick];
                    int taken = t < batch[e].Length ? batch[e].Actions[t][i] : 0;
                    oneHot[r * actions + taken] = NumOps.One;
                }
                for (int j = 0; j < s; j++) next[e * (n + s) + n + j] = states[t + 1].GetFlat(e * s + j);
            }
            chosen[t] = new Tensor<T>(new[] { rows, actions }, new Vector<T>(oneHot));
            Tensor<T> total;
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                total = _targetMixingNetwork.Predict(new Tensor<T>(new[] { b, n + s }, new Vector<T>(next)));
            for (int e = 0; e < b; e++)
            {
                var episode = batch[e];
                bool live = t < episode.Length;
                bool terminated = episode.Terminated && t == episode.Length - 1;
                double reward = live ? NumOps.ToDouble(episode.Rewards[t]) : 0.0;
                double bootstrap = terminated ? 0.0 : gamma * NumOps.ToDouble(total.GetFlat(e));
                y[t * b + e] = NumOps.FromDouble(reward + bootstrap);
                mask[t * b + e] = live ? NumOps.One : NumOps.Zero;
                if (live) filled++;
            }
        }
        double scale = _options.MeanLoss ? 1.0 / filled : 1.0;

        // The step's data, in the order the loss reads it: per step its agent inputs, chosen actions and state; then
        // the targets, the mask and the loss scale.
        var data = new List<Tensor<T>>(3 * steps + 3);
        for (int t = 0; t < steps; t++)
        {
            data.Add(inputs[t]);
            data.Add(chosen[t]);
            data.Add(states[t]);
        }
        data.Add(new Tensor<T>(new[] { steps * b, 1 }, new Vector<T>(y)));
        data.Add(new Tensor<T>(new[] { steps * b, 1 }, new Vector<T>(mask)));
        var scaleTensor = new Tensor<T>(new[] { 1, 1 });
        scaleTensor.SetFlat(0, NumOps.FromDouble(scale));
        data.Add(scaleTensor);
        var packed = StepInputPacking<T>.Pack(data.ToArray());

        var agentBase = (NeuralNetworkBase<T>)_agentNetwork;
        var mixBase = (NeuralNetworkBase<T>)_mixingNetwork;
        Tensor<T> Loss(Tensor<T> step)
        {
            int offset = 0;
            Tensor<T> Next(int r, int c)
            {
                var slice = StepInputPacking<T>.Slice(step, offset, new[] { r, c });
                offset += r * c;
                return slice;
            }
            var state = new Tensor<T>(new[] { rows, h });
            var mixInputs = new Tensor<T>[steps];
            for (int t = 0; t < steps; t++)
            {
                var x = Next(rows, d);
                var output = agentBase.ForwardForTraining(Engine.TensorConcatenate(new[] { x, state }, 1));
                var values = Engine.TensorSlice(output, new[] { 0, 0 }, new[] { rows, actions });
                state = Engine.TensorSlice(output, new[] { 0, actions }, new[] { rows, h });
                var picked = Engine.ReduceSum(Engine.TensorMultiply(values, Next(rows, actions)), new[] { 1 }, keepDims: true);
                mixInputs[t] = Engine.TensorConcatenate(new[] { Engine.Reshape(picked, new[] { b, n }), Next(b, s) }, 1);
            }
            var qTotal = Engine.Reshape(mixBase.ForwardForTraining(Engine.TensorConcatenate(mixInputs, 0)), new[] { steps * b, 1 });
            var targets = Next(steps * b, 1);
            var weights = Next(steps * b, 1);
            var td = Engine.TensorMultiply(Engine.TensorSubtract(qTotal, targets), weights);
            var sum = Engine.ReduceSum(Engine.TensorMultiply(td, td), new[] { 0, 1 }, keepDims: true);
            return Engine.ReduceSum(Engine.TensorMultiply(sum, Next(1, 1)), new[] { 0, 1 }, keepDims: false);
        }

        var layers = agentBase.Layers.Concat(mixBase.Layers).ToList();
        var stepper = _stepper ??= new TapeTrainingStepper<T>(this);
        T loss = stepper.Step(new FusedTrainingStepRequest<T>
        {
            Layers = layers.OfType<ITrainableLayer<T>>().ToList(),
            Input = packed,
            Target = packed,
            Forward = Loss,
            ComputeLoss = (value, _) => value,
            Optimizer = _optimizer,
            ExtraParameters = TapeTrainingStep<T>.CollectParameters(layers),
            ModelGradientClip = _options.MaxGradientNorm,
        });
        TrainingSteps++;
        LossHistory.Add(loss);

        // pymarl's QLearner: the targets follow every TargetUpdateFrequency episodes, after the step.
        int episodeIndex = _completedEpisodes - 1;
        if (episodeIndex - _lastTargetUpdateEpisode >= _options.TargetUpdateFrequency)
        {
            CopyNetworkWeights(_agentNetwork, _targetAgentNetwork);
            CopyNetworkWeights(_mixingNetwork, _targetMixingNetwork);
            _lastTargetUpdateEpisode = episodeIndex;
        }
        return loss;
    }

    // The network's action values per step [t][row][action], unrolled from a zero state with no gradient.
    private T[][][] Unroll(INeuralNetwork<T> network, Tensor<T>[] inputs)
    {
        int rows = inputs[0].Shape[0], d = inputs[0].Shape[1], h = _options.RnnHiddenDim, actions = _options.ActionSize;
        var values = new T[inputs.Length][][];
        var state = new T[rows * h];
        for (int k = 0; k < state.Length; k++) state[k] = NumOps.Zero;
        for (int t = 0; t < inputs.Length; t++)
        {
            var row = new T[rows * (d + h)];
            for (int r = 0; r < rows; r++)
            {
                for (int j = 0; j < d; j++) row[r * (d + h) + j] = inputs[t].GetFlat(r * d + j);
                for (int j = 0; j < h; j++) row[r * (d + h) + d + j] = state[r * h + j];
            }
            Tensor<T> output;
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                output = network.Predict(new Tensor<T>(new[] { rows, d + h }, new Vector<T>(row)));
            values[t] = new T[rows][];
            for (int r = 0; r < rows; r++)
            {
                values[t][r] = new T[actions];
                for (int a = 0; a < actions; a++) values[t][r][a] = output.GetFlat(r * (actions + h) + a);
                for (int j = 0; j < h; j++) state[r * h + j] = output.GetFlat(r * (actions + h) + actions + j);
            }
        }
        return values;
    }

    #endregion

    private Vector<T> OneHot(int action)
    {
        var v = new Vector<T>(_options.ActionSize);
        v[action] = NumOps.One;
        return v;
    }

    // The first index of the largest value, as torch.max reports it.
    private int ArgMax(IReadOnlyList<T> values)
    {
        int best = 0;
        for (int i = 1; i < values.Count; i++)
            if (NumOps.GreaterThan(values[i], values[best])) best = i;
        return best;
    }

    private int ArgMax(Vector<T> values) => ArgMax(values.ToArray());

    /// <inheritdoc/>
    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["steps"] = NumOps.FromDouble(_stepCount),
            ["episodes"] = NumOps.FromDouble(_completedEpisodes),
            ["buffer_episodes"] = NumOps.FromDouble(_episodesInBuffer),
            ["epsilon"] = NumOps.FromDouble(Epsilon),
        };
    }

    /// <summary>Predicts asynchronously (see <see cref="Predict"/>).</summary>
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));

    /// <summary>Trains asynchronously (see <see cref="Train"/>).</summary>
    public Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["ModelName"] = "QMIX",
                ["NumAgents"] = _options.NumAgents,
                ["StateSize"] = _options.StateSize,
                ["ActionSize"] = _options.ActionSize,
                ["GlobalStateSize"] = _options.GlobalStateSize,
                ["RnnHiddenDim"] = _options.RnnHiddenDim,
                ["MixingEmbedDim"] = _options.MixingEmbedDim,
                ["HypernetLayers"] = _options.HypernetLayers,
            },
        };
    }

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    /// <summary>
    /// Computes the mixing network's parameter gradients of a loss on one input, without updating them.
    /// </summary>
    /// <param name="input">A mixing input: the agents' Q-values, then the global state.</param>
    /// <param name="target">The target Q_tot.</param>
    /// <param name="lossFunction">The loss to differentiate, or null to use the agent's own.</param>
    /// <returns>A gradient vector the length of the mixing network's parameters.</returns>
    public Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
        => ComputeGradientsForNetwork(_mixingNetwork, new[] { _mixingNetwork }, input, target, lossFunction);

    /// <inheritdoc/>
    public override void SaveModel(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filepath));
        System.IO.File.WriteAllBytes(filepath, Serialize());
    }

    /// <inheritdoc/>
    public override void LoadModel(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filepath));
        if (!System.IO.File.Exists(filepath))
            throw new System.IO.FileNotFoundException($"Model file not found: {filepath}", filepath);
        Deserialize(System.IO.File.ReadAllBytes(filepath));
    }
}
