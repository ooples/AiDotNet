using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.AdvancedRL;

/// <summary>
/// Tabular Actor-Critic agent combining policy and value learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Actor-Critic has two components working together:
/// the Actor (decides which action to take) and the Critic (evaluates how good the action was).
/// The Critic provides feedback to help the Actor improve, like a coach watching a player.
/// This tabular version stores both policy preferences and value estimates in tables.
/// It combines the benefits of policy-based methods (can learn stochastic policies) with
/// value-based methods (lower variance updates).</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a tabular Actor-Critic agent with separate policy and value tables
/// var options = new TabularActorCriticOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, ActorLearningRate = 0.01 };
/// var agent = new TabularActorCriticAgent&lt;double&gt;(options);
///
/// // Select an action using the softmax actor policy
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
public class TabularActorCriticAgent<T> : ReinforcementLearningAgentBase<T>
{
    private TabularActorCriticOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _policy;  // Actor: π(a|s)
    private Dictionary<string, T> _valueTable;  // Critic: V(s)

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public TabularActorCriticAgent()
        : this(new TabularActorCriticOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public TabularActorCriticAgent(TabularActorCriticOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _policy = new Dictionary<string, Dictionary<int, T>>();
        _valueTable = new Dictionary<string, T>();
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        EnsureStateExists(state);
        string stateKey = GetStateKey(state);

        var probs = ComputeSoftmax(_policy[stateKey]);
        int selectedAction;
        if (training)
        {
            // The actor is a stochastic softmax policy while collecting training
            // experience (Sutton & Barto, Actor-Critic Methods). Sampling here is
            // what supplies on-policy exploration.
            double r = Random.NextDouble();
            double cumulative = 0.0;
            selectedAction = _options.ActionSize - 1;
            for (int a = 0; a < _options.ActionSize; a++)
            {
                cumulative += NumOps.ToDouble(probs[a]);
                if (r <= cumulative)
                {
                    selectedAction = a;
                    break;
                }
            }
        }
        else
        {
            // Evaluation/Predict must be deterministic. Return the maximum-
            // probability action, using the lowest index as the stable tie-break.
            selectedAction = ArgMax(probs);
        }

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        string stateKey = GetStateKey(state);
        string nextStateKey = GetStateKey(nextState);
        int actionIndex = ArgMax(action);

        EnsureStateExists(state);
        EnsureStateExists(nextState);

        // Compute TD error: δ = r + γV(s') - V(s)
        T currentValue = _valueTable[stateKey];
        T nextValue = done ? NumOps.Zero : _valueTable[nextStateKey];
        T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextValue));
        T tdError = NumOps.Subtract(target, currentValue);

        // Critic update: V(s) ← V(s) + α_c * δ
        T criticUpdate = NumOps.Multiply(NumOps.FromDouble(_options.CriticLearningRate), tdError);
        _valueTable[stateKey] = NumOps.Add(_valueTable[stateKey], criticUpdate);

        // Actor update: θ(s,a) ← θ(s,a) + α_a * δ
        T actorUpdate = NumOps.Multiply(NumOps.FromDouble(_options.ActorLearningRate), tdError);
        _policy[stateKey][actionIndex] = NumOps.Add(_policy[stateKey][actionIndex], actorUpdate);
    }

    public override T Train() => NumOps.Zero;

    private void EnsureStateExists(Vector<T> state)
    {
        string stateKey = GetStateKey(state);
        if (!_policy.ContainsKey(stateKey))
        {
            _policy[stateKey] = new Dictionary<int, T>();
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _policy[stateKey][a] = NumOps.Zero;  // Preferences
            }
            _valueTable[stateKey] = NumOps.Zero;
        }
    }

    private Vector<T> ComputeSoftmax(Dictionary<int, T> preferences)
    {
        T maxPref = preferences[0];
        for (int i = 1; i < preferences.Count; i++)
        {
            if (NumOps.GreaterThan(preferences[i], maxPref))
            {
                maxPref = preferences[i];
            }
        }

        var expValues = new Vector<T>(preferences.Count);
        T sumExp = NumOps.Zero;
        for (int i = 0; i < preferences.Count; i++)
        {
            T expVal = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Subtract(preferences[i], maxPref))));
            expValues[i] = expVal;
            sumExp = NumOps.Add(sumExp, expVal);
        }

        var probs = new Vector<T>(preferences.Count);
        for (int i = 0; i < preferences.Count; i++)
        {
            probs[i] = NumOps.Divide(expValues[i], sumExp);
        }

        return probs;
    }

    private string GetStateKey(Vector<T> state) => string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
    private int ArgMax(Vector<T> values) { int maxIndex = 0; T maxValue = values[0]; for (int i = 1; i < values.Length; i++) if (NumOps.GreaterThan(values[i], maxValue)) { maxValue = values[i]; maxIndex = i; } return maxIndex; }

    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T> { ["states_visited"] = NumOps.FromDouble(_valueTable.Count) };
    public override void ResetEpisode() { }
    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    /// <summary>
    /// Folded from <see cref="GetParameters"/> so the count and the vector cannot disagree.
    /// </summary>
    /// <remarks>
    /// The previous product formula described a DIFFERENT set of tensors than the getter builds,
    /// and the two drifted apart the moment the tables became ragged. Deriving the count from the
    /// vector is the same rule applied to DeepReinforcementLearningAgentBase: one source, the count
    /// is a fold over it, never a second opinion about it.
    /// </remarks>
    public override long ParameterCount => GetParameters().Length;
    public override int FeatureCount => _options.StateSize;
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_valueTable.Count);
        foreach (var kvp in _valueTable)
        {
            writer.Write(kvp.Key);
            writer.Write(NumOps.ToDouble(kvp.Value));
        }

        writer.Write(_policy.Count);
        foreach (var stateEntry in _policy)
        {
            writer.Write(stateEntry.Key);
            writer.Write(stateEntry.Value.Count);
            foreach (var actionEntry in stateEntry.Value)
            {
                writer.Write(actionEntry.Key);
                writer.Write(NumOps.ToDouble(actionEntry.Value));
            }
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        int valueCount = reader.ReadInt32();
        _valueTable.Clear();
        for (int i = 0; i < valueCount; i++)
        {
            string key = reader.ReadString();
            double value = reader.ReadDouble();
            _valueTable[key] = NumOps.FromDouble(value);
        }

        int policyCount = reader.ReadInt32();
        _policy.Clear();
        for (int i = 0; i < policyCount; i++)
        {
            string stateKey = reader.ReadString();
            int actionCount = reader.ReadInt32();
            _policy[stateKey] = new Dictionary<int, T>();
            for (int j = 0; j < actionCount; j++)
            {
                int actionKey = reader.ReadInt32();
                double actionValue = reader.ReadDouble();
                _policy[stateKey][actionKey] = NumOps.FromDouble(actionValue);
            }
        }
    }
    public override Vector<T> GetParameters()
    {
        var valueStates = OrderedValueStates();
        var policyEntries = OrderedPolicyEntries();

        // No synthetic minimum. An untrained agent holds no value estimates and no policy
        // preferences, so its parameter vector is empty; padding it to length 1 reported a
        // parameter that does not exist and that SetParameters had nowhere to put back.
        var vector = new Vector<T>(valueStates.Count + policyEntries.Count);
        int idx = 0;

        foreach (string state in valueStates)
            vector[idx++] = _valueTable[state];

        foreach (var entry in policyEntries)
            vector[idx++] = _policy[entry.State][entry.Action];

        return vector;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        var valueStates = OrderedValueStates();
        var policyEntries = OrderedPolicyEntries();
        int expected = valueStates.Count + policyEntries.Count;

        // Restore walks the SAME ordered entries the export walked. It previously looped
        // 0..ActionSize-1 for every state regardless of which actions that state actually held,
        // so a table with any state missing an action silently shifted every later value onto the
        // wrong (state, action) pair, and the bounds guard hid the mismatch instead of reporting it.
        if (parameters.Length != expected)
        {
            throw new ArgumentException(
                $"Expected {expected} parameters for {valueStates.Count} value estimates and "
                + $"{policyEntries.Count} policy preferences; got {parameters.Length}.",
                nameof(parameters));
        }

        int idx = 0;

        foreach (string state in valueStates)
            _valueTable[state] = parameters[idx++];

        foreach (var entry in policyEntries)
            _policy[entry.State][entry.Action] = parameters[idx++];
    }

    /// <summary>
    /// The value-table states in a fixed order, so export and restore agree.
    /// </summary>
    /// <remarks>
    /// Ordinal by key rather than dictionary order: <see cref="Dictionary{TKey, TValue}"/> makes no
    /// guarantee about enumeration order across insertions and removals, and a parameter vector that
    /// is written in one order and read back in another is silently wrong rather than loudly broken.
    /// </remarks>
    private List<string> OrderedValueStates()
    {
        var states = new List<string>(_valueTable.Keys);
        states.Sort(StringComparer.Ordinal);
        return states;
    }

    /// <summary>
    /// The (state, action) pairs the policy actually holds, in a fixed order.
    /// </summary>
    /// <remarks>
    /// Only the actions present in each state's table, never a 0..ActionSize-1 sweep: a ragged table
    /// is a legitimate state of a tabular agent that has not visited every action.
    /// </remarks>
    private List<(string State, int Action)> OrderedPolicyEntries()
    {
        var entries = new List<(string State, int Action)>();
        var states = new List<string>(_policy.Keys);
        states.Sort(StringComparer.Ordinal);

        foreach (string state in states)
        {
            var actions = new List<int>(_policy[state].Keys);
            actions.Sort();
            foreach (int action in actions) entries.Add((state, action));
        }

        return entries;
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new TabularActorCriticAgent<T>(_options);
        // Copy learned state - the value table and policy preferences
        clone._valueTable = new Dictionary<string, T>(_valueTable);
        clone._policy = new Dictionary<string, Dictionary<int, T>>();
        foreach (var kvp in _policy)
        {
            clone._policy[kvp.Key] = new Dictionary<int, T>(kvp.Value);
        }
        return clone;
    }
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
