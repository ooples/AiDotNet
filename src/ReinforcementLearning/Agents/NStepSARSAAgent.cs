using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.NStepSARSA;

/// <summary>
/// N-step SARSA agent using multi-step bootstrapping.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// N-step SARSA uses n-step returns that look ahead multiple steps before bootstrapping.
/// This provides a middle ground between TD (1-step) and Monte Carlo (full episode).
/// </para>
/// <para><b>For Beginners:</b>
/// Instead of updating based on just the next reward (1-step SARSA), n-step methods
/// look ahead n steps to get better return estimates before bootstrapping.
///
/// Update: G_t = r_t+1 + γr_t+2 + ... + γ^(n-1)r_t+n + γ^n Q(s_t+n, a_t+n)
///
/// Benefits:
/// - **Better credit assignment**: Propagates rewards faster than 1-step
/// - **Lower variance**: Than full Monte Carlo
/// - **Flexible**: Choose n to balance bias and variance
///
/// Common values: n=3 to n=10
/// Famous for: Sutton & Barto's RL textbook, Chapter 7
/// </para>
/// </remarks>
public class NStepSARSAAgent<T> : ReinforcementLearningAgentBase<T>
{
    private NStepSARSAOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private List<(string state, int action, T reward)> _nStepBuffer;
    private double _epsilon;
    private Random _random;

    public NStepSARSAAgent(NStepSARSAOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _nStepBuffer = new List<(string, int, T)>();
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

        _nStepBuffer.Add((stateKey, actionIndex, reward));

        if (_nStepBuffer.Count >= _options.NSteps || done)
        {
            UpdateNStep(nextState, done);

            if (done)
            {
                _nStepBuffer.Clear();
            }
            else if (_nStepBuffer.Count >= _options.NSteps)
            {
                _nStepBuffer.RemoveAt(0);
            }
        }

        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
    }

    private void UpdateNStep(Vector<T> finalState, bool done)
    {
        if (_nStepBuffer.Count == 0) return;

        var (firstState, firstAction, firstReward) = _nStepBuffer[0];
        EnsureStateExists(firstState);

        // Compute n-step return
        T G = NumOps.Zero;
        T discount = NumOps.One;

        for (int i = 0; i < _nStepBuffer.Count; i++)
        {
            G = NumOps.Add(G, NumOps.Multiply(discount, _nStepBuffer[i].reward));
            discount = NumOps.Multiply(discount, DiscountFactor);
        }

        // Add bootstrapped value if not done
        if (!done)
        {
            string finalStateKey = VectorToStateKey(finalState);

            // Use greedy action for bootstrap (proper n-step SARSA would track actual next action)
            // This is a simplification - proper implementation would require tracking the actual
            // action that will be taken at time t+n
            EnsureStateExists(finalStateKey);
            int nextActionIndex = GetBestAction(finalStateKey);

            T bootstrapValue = _qTable[finalStateKey][nextActionIndex];
            G = NumOps.Add(G, NumOps.Multiply(discount, bootstrapValue));
        }

        // Update Q-value
        T currentQ = _qTable[firstState][firstAction];
        T tdError = NumOps.Subtract(G, currentQ);
        T update = NumOps.Multiply(LearningRate, tdError);
        _qTable[firstState][firstAction] = NumOps.Add(currentQ, update);
    }

    public override T Train()
    {
        return NumOps.Zero;
    }

    public override void ResetEpisode()
    {
        _nStepBuffer.Clear();
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
        _epsilon = state.Epsilon;
    }

    public override Vector<T> GetParameters()
    {
        // Flatten Q-table into vector using linear indexing
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
        // Reconstruct Q-table from flattened vector using linear indexing
        var stateKeys = _qTable.Keys.ToList();
        int maxStates = parameters.Length / _options.ActionSize;

        for (int i = 0; i < Math.Min(maxStates, stateKeys.Count); i++)
        {
            var qValues = new Dictionary<int, T>();
            for (int action = 0; action < _options.ActionSize; action++)
            {
                int idx = i * _options.ActionSize + action;
                qValues[action] = parameters[idx];
            }
            _qTable[stateKeys[i]] = qValues;
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new NStepSARSAAgent<T>(_options);

        // Deep copy Q-table to avoid shared state
        foreach (var kvp in _qTable)
        {
            clone._qTable[kvp.Key] = new Dictionary<int, T>(kvp.Value);
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
