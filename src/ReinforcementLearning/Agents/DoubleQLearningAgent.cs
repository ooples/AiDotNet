using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.DoubleQLearning;

/// <summary>
/// Double Q-Learning agent using two Q-tables to reduce overestimation bias.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Double Q-Learning maintains two Q-tables and uses one to select actions
/// and the other to evaluate them, reducing maximization bias.
/// </para>
/// <para><b>For Beginners:</b>
/// Q-Learning tends to overestimate Q-values because it uses max(Q) for both
/// selecting and evaluating actions. Double Q-Learning fixes this by using
/// two separate Q-tables and randomly switching which one is updated.
///
/// Key innovation:
/// - **Two Q-tables**: Q1 and Q2
/// - **Decorrelation**: Use Q1 to select action, Q2 to evaluate (or vice versa)
/// - **Reduced Bias**: Prevents overestimation from max operator
///
/// Famous for: Hado van Hasselt 2010, foundation for Double DQN
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a tabular Double Q-Learning agent to reduce overestimation bias
/// var options = new DoubleQLearningOptions&lt;double&gt; { LearningRate = 0.1, StateSize = 4, ActionSize = 2 };
/// var agent = new DoubleQLearningAgent&lt;double&gt;(options);
///
/// // Select an action for the current state
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Double Q-learning",
    "https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html",
    Year = 2010,
    Authors = "van Hasselt, H.")]
public partial class DoubleQLearningAgent<T> : ReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{
    private DoubleQLearningOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable1;
    private Dictionary<string, Dictionary<int, T>> _qTable2;
    private double _epsilon;
    private Random _random;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public DoubleQLearningAgent()
        : this(new DoubleQLearningOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public DoubleQLearningAgent(DoubleQLearningOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _qTable1 = new Dictionary<string, Dictionary<int, T>>();
        _qTable2 = new Dictionary<string, Dictionary<int, T>>();
        _epsilon = _options.EpsilonStart;
        _random = RandomHelper.CreateSecureRandom();
    }

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
            // Use sum of both Q-tables for action selection
            actionIndex = GetBestAction(stateKey);
        }

        var action = new Vector<T>(_options.ActionSize);
        action[actionIndex] = NumOps.One;
        return action;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        string stateKey = VectorToStateKey(state);
        string nextStateKey = VectorToStateKey(nextState);
        int actionIndex = GetActionIndex(action);

        EnsureStateExists(stateKey);
        EnsureStateExists(nextStateKey);

        // Randomly choose which Q-table to update
        bool updateQ1 = _random.NextDouble() < 0.5;

        if (updateQ1)
        {
            // Update Q1 using Q2 for evaluation
            T currentQ = _qTable1[stateKey][actionIndex];

            if (done)
            {
                T target = reward;
                T tdError = NumOps.Subtract(target, currentQ);
                T update = NumOps.Multiply(LearningRate, tdError);
                _qTable1[stateKey][actionIndex] = NumOps.Add(currentQ, update);
            }
            else
            {
                // Use Q1 to select action, Q2 to evaluate
                int bestAction = GetBestActionFromTable(_qTable1, nextStateKey);
                T nextQ = _qTable2[nextStateKey][bestAction];
                T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextQ));
                T tdError = NumOps.Subtract(target, currentQ);
                T update = NumOps.Multiply(LearningRate, tdError);
                _qTable1[stateKey][actionIndex] = NumOps.Add(currentQ, update);
            }
        }
        else
        {
            // Update Q2 using Q1 for evaluation
            T currentQ = _qTable2[stateKey][actionIndex];

            if (done)
            {
                T target = reward;
                T tdError = NumOps.Subtract(target, currentQ);
                T update = NumOps.Multiply(LearningRate, tdError);
                _qTable2[stateKey][actionIndex] = NumOps.Add(currentQ, update);
            }
            else
            {
                // Use Q2 to select action, Q1 to evaluate
                int bestAction = GetBestActionFromTable(_qTable2, nextStateKey);
                T nextQ = _qTable1[nextStateKey][bestAction];
                T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextQ));
                T tdError = NumOps.Subtract(target, currentQ);
                T update = NumOps.Multiply(LearningRate, tdError);
                _qTable2[stateKey][actionIndex] = NumOps.Add(currentQ, update);
            }
        }

        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
    }

    public override T Train()
    {
        return NumOps.Zero;
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
        for (int i = 0; i < action.Length; i++)
        {
            if (NumOps.GreaterThan(action[i], NumOps.Zero))
            {
                return i;
            }
        }
        return 0;
    }

    private void EnsureStateExists(string stateKey)
    {
        if (!_qTable1.ContainsKey(stateKey))
        {
            _qTable1[stateKey] = new Dictionary<int, T>();
            _qTable2[stateKey] = new Dictionary<int, T>();
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _qTable1[stateKey][a] = NumOps.Zero;
                _qTable2[stateKey][a] = NumOps.Zero;
            }
        }
    }

    private int GetBestAction(string stateKey)
    {
        EnsureStateExists(stateKey);
        int bestAction = 0;
        T bestValue = NumOps.Add(_qTable1[stateKey][0], _qTable2[stateKey][0]);
        bool allEqual = true;

        for (int a = 1; a < _options.ActionSize; a++)
        {
            T sumValue = NumOps.Add(_qTable1[stateKey][a], _qTable2[stateKey][a]);
            if (NumOps.GreaterThan(sumValue, bestValue))
            {
                bestValue = sumValue;
                bestAction = a;
                allEqual = false;
            }
            else if (!NumOps.Equals(sumValue, bestValue))
            {
                allEqual = false;
            }
        }

        // Sutton & Barto §2.3: when all action-values are tied (typical for an
        // unvisited state with zero-initialized Q-tables), default argmax always
        // returns action 0 — a degenerate policy that produces the same action for
        // every unseen state. Break ties deterministically by state hash so the
        // policy stays distinct across states without injecting non-determinism.
        if (allEqual)
            bestAction = HashStateToAction(stateKey, _options.ActionSize);
        return bestAction;
    }

    private int GetBestActionFromTable(Dictionary<string, Dictionary<int, T>> qTable, string stateKey)
    {
        int bestAction = 0;
        T bestValue = qTable[stateKey][0];

        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(qTable[stateKey][a], bestValue))
            {
                bestValue = qTable[stateKey][a];
                bestAction = a;
            }
        }
        return bestAction;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
        };
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent(
            "q-table-1",
            new AiDotNet.Models.Parameters.NestedKeyedScalarCollectionParameterSource<T, string, int>(
                () => _qTable1));
        RegisterParameterComponent(
            "q-table-2",
            new AiDotNet.Models.Parameters.NestedKeyedScalarCollectionParameterSource<T, string, int>(
                () => _qTable2));
    }
    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        var state = new
        {
            QTable1 = _qTable1,
            QTable2 = _qTable2,
            Epsilon = _epsilon,
            Options = _options
        };
        string json = JsonConvert.SerializeObject(state);
        return System.Text.Encoding.UTF8.GetBytes(json);
    }

    public override void Deserialize(byte[] data)
    {
        if (data is null || data.Length == 0)
        {
            throw new ArgumentException("Serialized data cannot be null or empty", nameof(data));
        }

        string json = System.Text.Encoding.UTF8.GetString(data);
        var state = JsonConvert.DeserializeObject<dynamic>(json);
        if (state is null)
        {
            throw new InvalidOperationException("Deserialization returned null");
        }

        _qTable1 = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, T>>>(state.QTable1.ToString()) ?? new Dictionary<string, Dictionary<int, T>>();
        _qTable2 = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, T>>>(state.QTable2.ToString()) ?? new Dictionary<string, Dictionary<int, T>>();
        _epsilon = state.Epsilon;

        // The two tables are validated together, HERE, before anything reads them. GetParameters
        // sizes its vector from _qTable1.Count but then fills from both tables, indexing
        // stateQValues[action] for every action in 0..ActionSize-1. Persisted data with a state
        // missing from one table overruns that vector; a state missing an ACTION throws
        // KeyNotFoundException from inside the flatten. Neither failure says anything about the file
        // that caused it.
        ValidatePairedQTables();
    }

    /// <summary>
    /// Requires the two Q-tables to describe the same states and each state to hold exactly the
    /// actions <c>0 .. ActionSize - 1</c>.
    /// </summary>
    /// <exception cref="InvalidOperationException">The restored tables are not a matched pair.</exception>
    private void ValidatePairedQTables()
    {
        if (_qTable1.Count != _qTable2.Count)
        {
            throw new InvalidOperationException(
                $"Serialized {nameof(DoubleQLearningAgent<T>)} has {_qTable1.Count} states in QTable1 "
                + $"and {_qTable2.Count} in QTable2. Double Q-learning keeps one entry per state in "
                + "both tables; the model data is incomplete or was written by an incompatible version.");
        }

        foreach (var entry in _qTable1)
        {
            if (!_qTable2.ContainsKey(entry.Key))
            {
                throw new InvalidOperationException(
                    $"Serialized {nameof(DoubleQLearningAgent<T>)} has state '{entry.Key}' in QTable1 "
                    + "but not in QTable2.");
            }

            RequireCompleteActionSet(entry.Key, entry.Value, nameof(_qTable1));
            RequireCompleteActionSet(entry.Key, _qTable2[entry.Key], nameof(_qTable2));
        }
    }

    private void RequireCompleteActionSet(string stateKey, Dictionary<int, T> actions, string tableName)
    {
        for (int action = 0; action < _options.ActionSize; action++)
        {
            if (!actions.ContainsKey(action))
            {
                throw new InvalidOperationException(
                    $"Serialized {nameof(DoubleQLearningAgent<T>)} is missing action {action} for state "
                    + $"'{stateKey}' in {tableName}. Every state must hold actions 0 to "
                    + $"{_options.ActionSize - 1}.");
            }
        }
    }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new DoubleQLearningAgent<T>(_options);

        // Deep copy Q-table 1 to avoid shared state
        foreach (var kvp in _qTable1)
        {
            clone._qTable1[kvp.Key] = new Dictionary<int, T>(kvp.Value);
        }

        // Deep copy Q-table 2 to avoid shared state
        foreach (var kvp in _qTable2)
        {
            clone._qTable2[kvp.Key] = new Dictionary<int, T>(kvp.Value);
        }

        clone._epsilon = _epsilon;
        return clone;
    }

    public Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return GetParameters();
    }

    public void ApplyGradients(Vector<T> gradients, T learningRate) { }

    public override void SaveModel(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filepath));
        }

        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

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
