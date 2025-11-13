using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

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
public class DoubleQLearningAgent<T> : ReinforcementLearningAgentBase<T>
{
    private DoubleQLearningOptions<T> _options;
    private Dictionary<string, Dictionary<int, T>> _qTable1;
    private Dictionary<string, Dictionary<int, T>> _qTable2;
    private double _epsilon;

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
        bool updateQ1 = Random.NextDouble() < 0.5;

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
            if (NumOps.Compare(action[i], NumOps.Zero) > 0)
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

        for (int a = 1; a < _options.ActionSize; a++)
        {
            T sumValue = NumOps.Add(_qTable1[stateKey][a], _qTable2[stateKey][a]);
            if (NumOps.Compare(sumValue, bestValue) > 0)
            {
                bestValue = sumValue;
                bestAction = a;
            }
        }
        return bestAction;
    }

    private int GetBestActionFromTable(Dictionary<string, Dictionary<int, T>> qTable, string stateKey)
    {
        int bestAction = 0;
        T bestValue = qTable[stateKey][0];

        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.Compare(qTable[stateKey][a], bestValue) > 0)
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
            ModelType = "DoubleQLearning",
            InputSize = _options.StateSize,
            OutputSize = _options.ActionSize,
            ParameterCount = ParameterCount
        };
    }

    public override int ParameterCount => _qTable1.Count * _options.ActionSize * 2;
    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        throw new NotImplementedException("Double Q-Learning serialization not yet implemented");
    }

    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("Double Q-Learning deserialization not yet implemented");
    }

    public override Matrix<T> GetParameters()
    {
        int stateCount = Math.Max(_qTable1.Count, 1);
        var parameters = new Matrix<T>(stateCount * 2, _options.ActionSize);

        int row = 0;
        foreach (var stateQValues in _qTable1.Values)
        {
            for (int action = 0; action < _options.ActionSize; action++)
            {
                parameters[row, action] = stateQValues[action];
            }
            row++;
        }

        foreach (var stateQValues in _qTable2.Values)
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
        _qTable1.Clear();
        _qTable2.Clear();
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new DoubleQLearningAgent<T>(_options);
        clone._qTable1 = new Dictionary<string, Dictionary<int, T>>(_qTable1);
        clone._qTable2 = new Dictionary<string, Dictionary<int, T>>(_qTable2);
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
