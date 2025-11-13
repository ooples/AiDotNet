using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.MonteCarlo;

/// <summary>
/// Every-Visit Monte Carlo agent that updates all visits to states in an episode.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EveryVisitMonteCarloAgent<T> : ReinforcementLearningAgentBase<T>
{
    private MonteCarloOptions<T> _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, List<T>>> _returns;
    private List<(string state, int action, T reward)> _episode;
    private double _epsilon;

    public EveryVisitMonteCarloAgent(MonteCarloOptions<T> options)
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
            if (NumOps.GreaterThan(action[i], NumOps.Zero)) return i;
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

    public override void ResetEpisode() { _episode.Clear(); base.ResetEpisode(); }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    }

    public override int ParameterCount => _qTable.Count * _options.ActionSize;
    public override int FeatureCount => _options.StateSize;
    public override byte[] Serialize() { throw new NotImplementedException(); }
    public override void Deserialize(byte[] data) { throw new NotImplementedException(); }

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

    public override void SetParameters(Matrix<T> parameters) { _qTable.Clear(); }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new EveryVisitMonteCarloAgent<T>(_options);
        clone._qTable = new Dictionary<string, Dictionary<int, T>>(_qTable);
        clone._epsilon = _epsilon;
        return clone;
    }

    public override (Matrix<T> Gradients, T Loss) ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return (new Matrix<T>(1, 1), NumOps.Zero);
    }

    public override void ApplyGradients(Matrix<T> gradients, T learningRate) { }
    public override void Save(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void Load(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
