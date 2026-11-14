using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;

using AiDotNet.ReinforcementLearning.Parameters;

namespace AiDotNet.ReinforcementLearning.Agents.NStepQLearning;

/// <summary>
/// N-step Q-Learning agent using multi-step off-policy returns.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> N-step Q-Learning extends regular Q-Learning by looking ahead
/// n steps before bootstrapping. Regular Q-Learning updates based on the very next reward,
/// while n-step looks at n future rewards. This propagates information faster (like seeing
/// further down a road before deciding which way to turn). The trade-off: higher n means
/// faster learning but more variance. Common values are n=3 to n=10.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create an n-step Q-Learning agent for faster reward propagation
/// var options = new NStepQLearningOptions&lt;double&gt; { NSteps = 5, StateSize = 4, ActionSize = 2 };
/// var agent = new NStepQLearningAgent&lt;double&gt;(options);
///
/// // Select an action using epsilon-greedy over Q-values
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Reinforcement Learning: An Introduction",
    "https://incompleteideas.net/book/the-book-2nd.html",
    Year = 2018,
    Authors = "Sutton, R. S. & Barto, A. G.")]
public partial class NStepQLearningAgent<T> : ReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The Q-table, padded to ActionSize per state and clamped to one row -- the same
    /// flattening this agent used to write by hand.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new QTableParameterSource<T>(_qTable, _options.ActionSize));
    }
    private NStepQLearningOptions<T> _options;

    private Dictionary<string, Dictionary<int, T>> _qTable;
    private List<(string state, int action, T reward)> _nStepBuffer;
    private double _epsilon;
    private Random _random;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public NStepQLearningAgent()
        : this(new NStepQLearningOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public NStepQLearningAgent(NStepQLearningOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        Options = _options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _nStepBuffer = new List<(string, int, T)>();
        _epsilon = _options.EpsilonStart;
        _random = Random;
    }

    /// <inheritdoc />
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        string stateKey = VectorToStateKey(state);
        int actionIndex;
        if (training && _random.NextDouble() < _epsilon)
        {
            actionIndex = _random.Next(_options.ActionSize);
        }
        else
        {
            actionIndex = GetBestAction(stateKey);
        }
        var action = new Vector<T>(_options.ActionSize);
        action[actionIndex] = NumOps.One;
        return action;
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        string stateKey = VectorToStateKey(state);
        int actionIndex = GetActionIndex(action);
        _nStepBuffer.Add((stateKey, actionIndex, reward));

        if (done)
        {
            // Episode end: every pending step gets its truncated return G_{t:T} (Sutton & Barto 2018, Section 7.2,
            // tau running on to T - 1), oldest first.
            while (_nStepBuffer.Count > 0)
            {
                UpdateNStep(nextState, true);
                _nStepBuffer.RemoveAt(0);
            }
        }
        else if (_nStepBuffer.Count >= _options.NSteps)
        {
            UpdateNStep(nextState, false);
            _nStepBuffer.RemoveAt(0);
        }
        // Epsilon decay moved to ResetEpisode to decay per episode, not per step
    }

    private void UpdateNStep(Vector<T> finalState, bool done)
    {
        if (_nStepBuffer.Count == 0) return;
        var (firstState, firstAction, firstReward) = _nStepBuffer[0];
        EnsureStateExists(firstState);

        var stepRewards = Vector<T>.FromEnumerable(_nStepBuffer.Select(step => step.reward));
        T G = RLTensorOps.DiscountedSum(stepRewards, DiscountFactor, out T discount);

        if (!done)
        {
            string finalStateKey = VectorToStateKey(finalState);
            EnsureStateExists(finalStateKey);
            T maxQ = GetMaxQValue(finalStateKey);
            G = NumOps.Add(G, NumOps.Multiply(discount, maxQ));
        }

        T currentQ = _qTable[firstState][firstAction];
        T tdError = NumOps.Subtract(G, currentQ);
        T update = NumOps.Multiply(LearningRate, tdError);
        _qTable[firstState][firstAction] = NumOps.Add(currentQ, update);
    }

    private T GetMaxQValue(string stateKey)
    {
        T maxValue = RLTensorOps.Max(RLTensorOps.Row(_qTable[stateKey], _options.ActionSize));
        return maxValue;
    }

    /// <inheritdoc />
    public override T Train() { return NumOps.Zero; }
    /// <inheritdoc />
    public override void ResetEpisode()
    {
        _nStepBuffer.Clear();
        // Decay epsilon per episode, not per step
        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
        base.ResetEpisode();
    }

    private string VectorToStateKey(Vector<T> state)
    {
        var parts = new string[state.Length];
        for (int i = 0; i < state.Length; i++)
        {
            parts[i] = NumOps.ToDouble(state[i]).ToString("F4");
        }
        return string.Join(",", parts);
    }

    private int GetActionIndex(Vector<T> action)
    {
        return RLTensorOps.FirstPositiveIndex(action, NumOps.Zero);
    }

    private void EnsureStateExists(string stateKey)
    {
        if (!_qTable.ContainsKey(stateKey))
        {
            _qTable[stateKey] = new Dictionary<int, T>();
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _qTable[stateKey][a] = NumOps.Zero;
            }
        }
    }

    private int GetBestAction(string stateKey)
    {
        EnsureStateExists(stateKey);
        var qRow = RLTensorOps.Row(_qTable[stateKey], _options.ActionSize);
        int bestAction = RLTensorOps.ArgMax(qRow);
        bool allEqual = RLTensorOps.AllEqual(qRow, NumOps);
        if (allEqual)
            bestAction = HashStateToAction(stateKey, _options.ActionSize);
        return bestAction;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T> { FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    }

    /// <inheritdoc />
    public override int FeatureCount => _options.StateSize;

    /// <inheritdoc />
    public Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Returned GetParameters() -- the WEIGHTS -- where this interface promises
        // "gradients with respect to all model parameters". The vector length matches, so
        // nothing downstream could detect the substitution: ApplyGradients would subtract the
        // weights from themselves, and Elastic Weight Consolidation / Gradient Episodic Memory /
        // Memory Aware Synapses would build Fisher-information estimates out of parameter
        // magnitudes. A distributed trainer averaging these across workers was averaging
        // parameters and calling the result a gradient.
        //
        // This agent is tabular: it owns no network and its update is a value backup, not a
        // differentiable loss, so no parameter gradient exists here to return.
        throw new NotSupportedException(
            "NStepQLearningAgent is a tabular agent whose update is a value backup rather than a "
            + "differentiable loss, so it has no parameter gradients for this interface to "
            + "return. Call Train() instead.");
    }

    /// <inheritdoc />
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // An empty body silently accepted a gradient vector and did nothing with it, so a
        // caller applying gradients believed the update landed. Refusing is the honest
        // contract, and matches ComputeGradients above.
        throw new NotSupportedException(
            "NStepQLearningAgent is a tabular agent and does not apply gradient updates. "
            + "Call Train() instead.");
    }

    /// <inheritdoc />
    public override void SaveModel(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filepath));
        }

        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    /// <inheritdoc />
    public override void LoadModel(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filepath));
        }

        if (!System.IO.File.Exists(filepath))
        {
            throw new System.IO.FileNotFoundException($"Model file not found: {filepath}", filepath);
        }

        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
