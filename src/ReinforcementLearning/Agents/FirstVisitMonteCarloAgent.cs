using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.MonteCarlo;

/// <summary>
/// First-Visit Monte Carlo agent for episodic tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// First-Visit MC estimates value functions by averaging returns following
/// the first visit to each state in an episode.
/// </para>
/// <para><b>For Beginners:</b>
/// Monte Carlo methods learn from complete episodes. They wait until the
/// episode ends, then update Q-values based on the actual returns received.
///
/// Unlike TD methods (Q-Learning, SARSA), MC methods:
/// - **Wait for episode completion**: No bootstrapping
/// - **Use actual returns**: Not estimates
/// - **Model-free**: Don't need environment dynamics
/// - **First-visit**: Only count first occurrence of state-action
///
/// Perfect for: Episodic tasks (games with clear endings)
/// Not good for: Continuing tasks (no episode end)
///
/// Famous for: Foundation of RL, unbiased estimates
/// </para>
/// </remarks>
public class FirstVisitMonteCarloAgent<T> : ReinforcementLearningAgentBase<T>
{
    private MonteCarloOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, List<T>>> _returns;
    private List<(string state, int action, T reward)> _episode;
    private double _epsilon;
    private Random _random;

    public FirstVisitMonteCarloAgent(MonteCarloOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
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
        var visited = new HashSet<string>();

        // Work backwards through episode
        for (int t = _episode.Count - 1; t >= 0; t--)
        {
            var (state, action, reward) = _episode[t];

            G = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, G));

            string stateActionKey = $"{state}:{action}";

            // First-visit: only update if not seen before in this episode
            if (!visited.Contains(stateActionKey))
            {
                visited.Add(stateActionKey);

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

                // Update Q-value as average of returns
                _qTable[state][action] = ComputeAverage(_returns[state][action]);
            }
        }
    }

    private T ComputeAverage(List<T> values)
    {
        if (values.Count == 0)
        {
            return NumOps.Zero;
        }

        T sum = NumOps.Zero;
        foreach (var value in values)
        {
            sum = NumOps.Add(sum, value);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(values.Count));
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
            parts[i] = NumOps.ToDouble(state[i]).ToString("F8");
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

    public override void ResetEpisode()
    {
        _episode.Clear();
        base.ResetEpisode();
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ReinforcementLearning,
        };
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
        // Flatten Q-table into vector
        int stateCount = _qTable.Count;
        var parameters = new Vector<T>(stateCount * _options.ActionSize);

        int idx = 0;
        foreach (var stateQValues in _qTable.Values)
        {
            for (int action = 0; action < _options.ActionSize; action++)
            {
                parameters[idx++] = stateQValues[action];
            }
        }

        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        // Save existing state keys before clearing
        var stateKeys = _qTable.Keys.ToList();

        // If Q-table is empty, cannot reconstruct from parameters alone
        // This method updates existing Q-values but preserves table structure
        if (stateKeys.Count == 0)
        {
            // Cannot set parameters on an uninitialized agent
            // Q-table structure must be built through experience first
            return;
        }

        // Update Q-values while preserving the state keys
        int maxStates = parameters.Length / _options.ActionSize;
        int idx = 0;

        for (int i = 0; i < Math.Min(maxStates, stateKeys.Count); i++)
        {
            var stateKey = stateKeys[i];
            for (int action = 0; action < _options.ActionSize; action++)
            {
                if (idx < parameters.Length)
                {
                    _qTable[stateKey][action] = parameters[idx];
                    idx++;
                }
            }
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new FirstVisitMonteCarloAgent<T>(_options);

        // Deep copy Q-table
        foreach (var kvp in _qTable)
        {
            clone._qTable[kvp.Key] = new Dictionary<int, T>(kvp.Value);
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
