using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.MonteCarlo;

/// <summary>
/// Every-Visit Monte Carlo agent that updates all visits to states in an episode.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EveryVisitMonteCarloAgent<T> : ReinforcementLearningAgentBase<T>
{
    private MonteCarloOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, List<T>>> _returns;
    private List<(string state, int action, T reward)> _episode;
    private double _epsilon;
    private Random _random;

    public EveryVisitMonteCarloAgent(MonteCarloOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;

        // Validate EpsilonDecay is in (0, 1] range (1.0 means no decay, which is valid)
        if (_options.EpsilonDecay <= 0.0 || _options.EpsilonDecay > 1.0)
        {
            throw new ArgumentException("EpsilonDecay must be in the range (0, 1] for proper decay behavior.", nameof(options));
        }

        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _returns = new Dictionary<string, Dictionary<int, List<T>>>();
        _episode = new List<(string, int, T)>();
        _epsilon = _options.EpsilonStart;
        _random = Random;
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
            actionIndex = GetBestAction(stateKey);
        }
        var action = new Vector<T>(_options.ActionSize);
        action[actionIndex] = NumOps.One;
        return action;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        string stateKey = VectorToStateKey(state);
        int actionIndex = GetActionIndex(action);
        _episode.Add((stateKey, actionIndex, reward));

        if (done)
        {
            UpdateFromEpisode();
            _episode.Clear();
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
        }
    }

    private void UpdateFromEpisode()
    {
        T G = NumOps.Zero;

        for (int t = _episode.Count - 1; t >= 0; t--)
        {
            var (state, action, reward) = _episode[t];
            G = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, G));

            EnsureStateExists(state);
            if (!_returns.ContainsKey(state))
            {
                _returns[state] = new Dictionary<int, List<T>>();
            }
            if (!_returns[state].ContainsKey(action))
            {
                _returns[state][action] = new List<T>();
            }

            _returns[state][action].Add(G);
            _qTable[state][action] = ComputeAverage(_returns[state][action]);
        }
    }

    public override T Train() { return NumOps.Zero; }

    /// <summary>
    /// Converts a state vector to a string key for the Q-table.
    /// Uses F8 precision (8 decimal places) to minimize state collisions.
    /// Note: States differing only beyond 8 decimal places will be treated as identical.
    /// </summary>
    private string VectorToStateKey(Vector<T> state)
    {
        var parts = new string[state.Length];
        for (int i = 0; i < state.Length; i++)
        {
            parts[i] = NumOps.ToDouble(state[i]).ToString("F8");
        }
        return string.Join(",", parts);
    }

    /// <summary>
    /// Gets the index of the selected action from a one-hot encoded action vector.
    /// </summary>
    /// <param name="action">One-hot encoded action vector.</param>
    /// <returns>Index of the action with value greater than zero.</returns>
    /// <exception cref="ArgumentException">Thrown when action vector is invalid (all elements &lt;= 0).</exception>
    private int GetActionIndex(Vector<T> action)
    {
        if (action == null)
        {
            throw new ArgumentNullException(nameof(action));
        }

        for (int i = 0; i < action.Length; i++)
        {
            if (NumOps.GreaterThan(action[i], NumOps.Zero))
            {
                return i;
            }
        }

        // Invalid action vector - all elements are <= 0
        throw new ArgumentException("Invalid action vector: all elements are <= 0. Expected one-hot encoded vector with exactly one positive element.", nameof(action));
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
        int bestAction = 0;
        T bestValue = _qTable[stateKey][0];
        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(_qTable[stateKey][a], bestValue))
            {
                bestValue = _qTable[stateKey][a];
                bestAction = a;
            }
        }
        return bestAction;
    }

    /// <summary>
    /// Computes the average of a list of returns.
    /// </summary>
    /// <param name="returns">List of return values.</param>
    /// <returns>The average return value.</returns>
    private T ComputeAverage(List<T> returns)
    {
        if (returns == null || returns.Count == 0)
        {
            return NumOps.Zero;
        }

        T sum = NumOps.Zero;
        foreach (T value in returns)
        {
            sum = NumOps.Add(sum, value);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(returns.Count));
    }

    public override void ResetEpisode() { _episode.Clear(); base.ResetEpisode(); }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    }

    public override int ParameterCount => _qTable.Count * _options.ActionSize;
    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        var state = new
        {
            QTable = _qTable,
            Returns = _returns,
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

        _qTable = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, T>>>(state.QTable.ToString()) ?? new Dictionary<string, Dictionary<int, T>>();
        _returns = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, List<T>>>>(state.Returns.ToString()) ?? new Dictionary<string, Dictionary<int, List<T>>>();
        _epsilon = state.Epsilon;
    }

    public override Vector<T> GetParameters()
    {
        var paramsList = new List<T>();
        foreach (var stateEntry in _qTable)
        {
            foreach (var actionValue in stateEntry.Value)
            {
                paramsList.Add(actionValue.Value);
            }
        }

        if (paramsList.Count == 0)
        {
            paramsList.Add(NumOps.Zero);
        }

        var paramsVector = new Vector<T>(paramsList.Count);
        for (int i = 0; i < paramsList.Count; i++)
        {
            paramsVector[i] = paramsList[i];
        }

        return paramsVector;
    }

    /// <summary>
    /// Sets parameters. Note: This method cannot reconstruct the Q-table structure from a flat vector
    /// without additional state mapping information. It only updates existing Q-table entries.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        // Can only update existing Q-table entries since we don't have state mapping
        int index = 0;
        foreach (var stateEntry in _qTable.ToList())
        {
            for (int a = 0; a < _options.ActionSize; a++)
            {
                if (index < parameters.Length)
                {
                    _qTable[stateEntry.Key][a] = parameters[index];
                    index++;
                }
            }
        }

        // Warn if Q-table is empty - parameters cannot be applied
        if (_qTable.Count == 0 && parameters.Length > 0)
        {
            // Parameters will be ignored since Q-table structure doesn't exist yet
            // This is a limitation of the SetParameters design for tabular methods
        }
    }

    /// <summary>
    /// Creates a deep copy of the agent, including all Q-table entries.
    /// </summary>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new EveryVisitMonteCarloAgent<T>(_options);

        // Deep copy Q-table
        foreach (var stateEntry in _qTable)
        {
            clone._qTable[stateEntry.Key] = new Dictionary<int, T>();
            foreach (var actionEntry in stateEntry.Value)
            {
                clone._qTable[stateEntry.Key][actionEntry.Key] = actionEntry.Value;
            }
        }

        // Deep copy returns
        foreach (var kvp in _returns)
        {
            clone._returns[kvp.Key] = new Dictionary<int, List<T>>();
            foreach (var returnKvp in kvp.Value)
            {
                clone._returns[kvp.Key][returnKvp.Key] = new List<T>(returnKvp.Value);
            }
        }

        clone._epsilon = _epsilon;
        return clone;
    }

    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return GetParameters();
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate) { }

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
