using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

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
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private List<(string state, int action, T reward)> _nStepBuffer;
    private double _epsilon;

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
            Vector<T> nextAction = SelectAction(finalState, training: true);
            int nextActionIndex = GetActionIndex(nextAction);

            EnsureStateExists(finalStateKey);
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
            ModelType = "NStepSARSA",
            InputSize = _options.StateSize,
            OutputSize = _options.ActionSize,
            ParameterCount = ParameterCount
        };
    }

    public override int ParameterCount => _qTable.Count * _options.ActionSize;
    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        throw new NotImplementedException("N-step SARSA serialization not yet implemented");
    }

    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("N-step SARSA deserialization not yet implemented");
    }

    public override Vector<T> GetParameters()
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

    public override void SetParameters(Vector<T> parameters)
    {
        _qTable.Clear();
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new NStepSARSAAgent<T>(_options);
        clone._qTable = new Dictionary<string, Dictionary<int, T>>(_qTable);
        clone._epsilon = _epsilon;
        return clone;
    }

    public override (Vector<T> Gradients, T Loss) ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return (new Matrix<T>(1, 1), NumOps.Zero);
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate) { }

    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
