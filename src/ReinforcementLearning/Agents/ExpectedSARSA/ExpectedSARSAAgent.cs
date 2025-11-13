using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.ExpectedSARSA;

/// <summary>
/// Expected SARSA agent using tabular methods.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Expected SARSA is a TD control algorithm that uses the expected value under
/// the current policy instead of sampling the next action.
/// </para>
/// <para><b>For Beginners:</b>
/// Expected SARSA is like SARSA but instead of using the actual next action,
/// it uses the average Q-value weighted by the probability of taking each action.
/// This reduces variance compared to SARSA.
///
/// Update: Q(s,a) ← Q(s,a) + α[r + γ Σ π(a'|s')Q(s',a') - Q(s,a)]
///
/// Benefits over SARSA:
/// - **Lower Variance**: Averages over actions instead of sampling
/// - **Off-Policy Learning**: Can learn optimal policy while exploring
/// - **Better Performance**: Often converges faster than SARSA
///
/// Famous for: Van Seijen et al. 2009, bridging SARSA and Q-Learning
/// </para>
/// </remarks>
public class ExpectedSARSAAgent<T> : ReinforcementLearningAgentBase<T>
{
    private ExpectedSARSAOptions<T> _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private double _epsilon;

    public ExpectedSARSAAgent(ExpectedSARSAOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _epsilon = _options.EpsilonStart;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        string stateKey = VectorToStateKey(state);

        int actionIndex;
        if (training && Random.NextDouble() < _epsilon)
        {
            actionIndex = Random.Next(_options.ActionSize);
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
        string nextStateKey = VectorToStateKey(nextState);
        int actionIndex = GetActionIndex(action);

        EnsureStateExists(stateKey);
        EnsureStateExists(nextStateKey);

        // Expected SARSA: Use expected value under current policy
        T currentQ = _qTable[stateKey][actionIndex];
        T expectedNextQ = done ? NumOps.Zero : ComputeExpectedValue(nextStateKey);

        T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, expectedNextQ));
        T tdError = NumOps.Subtract(target, currentQ);
        T update = NumOps.Multiply(LearningRate, tdError);

        _qTable[stateKey][actionIndex] = NumOps.Add(currentQ, update);

        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
    }

    private T ComputeExpectedValue(string stateKey)
    {
        EnsureStateExists(stateKey);

        // Expected value: Σ π(a|s) Q(s,a)
        // For ε-greedy: (1-ε)Q(a*) + ε * (1/|A|) Σ Q(a)

        int bestAction = GetBestAction(stateKey);
        T bestQ = _qTable[stateKey][bestAction];

        T sumQ = NumOps.Zero;
        for (int a = 0; a < _options.ActionSize; a++)
        {
            sumQ = NumOps.Add(sumQ, _qTable[stateKey][a]);
        }

        // (1 - ε) * Q(best) + ε * mean(Q)
        double prob = 1.0 - _epsilon;
        T greedyPart = NumOps.Multiply(NumOps.FromDouble(prob), bestQ);

        T explorePart = NumOps.Multiply(
            NumOps.FromDouble(_epsilon),
            NumOps.Divide(sumQ, NumOps.FromDouble(_options.ActionSize))
        );

        return NumOps.Add(greedyPart, explorePart);
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
            if (NumOps.Compare(action[i], NumOps.Zero) > 0)
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
            if (NumOps.Compare(_qTable[stateKey][a], bestValue) > 0)
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
            ModelType = "ExpectedSARSA",
            InputSize = _options.StateSize,
            OutputSize = _options.ActionSize,
            ParameterCount = ParameterCount
        };
    }

    public override int ParameterCount => _qTable.Count * _options.ActionSize;
    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        throw new NotImplementedException("ExpectedSARSA serialization not yet implemented");
    }

    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("ExpectedSARSA deserialization not yet implemented");
    }

    public override Matrix<T> GetParameters()
    {
        int stateCount = Math.Max(_qTable.Count, 1);
        var parameters = new Matrix<T>(stateCount, _options.ActionSize);
        int row = 0;
        foreach (var stateQValues in _qTable.Values)
        {
            for (int action = 0; action < _options.ActionSize; action++)
            {
                parameters[row, action] = stateQValues[action];
            }
            row++;
        }
        return parameters;
    }

    public override void SetParameters(Matrix<T> parameters)
    {
        _qTable.Clear();
        var stateKeys = _qTable.Keys.ToList();
        for (int i = 0; i < Math.Min(parameters.Rows, stateKeys.Count); i++)
        {
            var qValues = new Dictionary<int, T>();
            for (int action = 0; action < _options.ActionSize; action++)
            {
                qValues[action] = parameters[i, action];
            }
            _qTable[stateKeys[i]] = qValues;
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new ExpectedSARSAAgent<T>(_options);
        clone._qTable = new Dictionary<string, Dictionary<int, T>>(_qTable);
        clone._epsilon = _epsilon;
        return clone;
    }

    public override (Matrix<T> Gradients, T Loss) ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return (new Matrix<T>(1, 1), NumOps.Zero);
    }

    public override void ApplyGradients(Matrix<T> gradients, T learningRate) { }

    public override void Save(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void Load(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
