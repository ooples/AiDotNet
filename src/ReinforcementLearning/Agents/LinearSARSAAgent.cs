using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.AdvancedRL;

/// <summary>
/// Linear SARSA agent using linear function approximation with on-policy learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LinearSARSAAgent<T> : ReinforcementLearningAgentBase<T>
{
    private LinearSARSAOptions<T> _options;
    private Matrix<T> _weights;  // Weight matrix: [ActionSize x FeatureSize]
    private double _epsilon;
    private int _lastAction = -1;
    private Vector<T>? _lastState = null;

    public LinearSARSAAgent(LinearSARSAOptions<T> options) : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _weights = new Matrix<T>(_options.ActionSize, _options.FeatureSize);

        // Initialize weights to zero
        for (int a = 0; a < _options.ActionSize; a++)
        {
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                _weights[a, f] = NumOps.Zero;
            }
        }

        _epsilon = options.EpsilonStart;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        int selectedAction;
        if (training && Random.NextDouble() < _epsilon)
        {
            selectedAction = Random.Next(_options.ActionSize);
        }
        else
        {
            selectedAction = GetGreedyAction(state);
        }

        _lastState = state;
        _lastAction = selectedAction;

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        if (_lastState == null || _lastAction < 0) return;

        // Compute current Q-value: Q(s,a) = w_a^T * φ(s)
        T currentQ = ComputeQValue(_lastState, _lastAction);

        // Compute next Q-value using the action that will be taken (on-policy)
        T nextQ = NumOps.Zero;
        if (!done)
        {
            // Select next action using current policy
            int nextAction;
            if (Random.NextDouble() < _epsilon)
            {
                nextAction = Random.Next(_options.ActionSize);
            }
            else
            {
                nextAction = GetGreedyAction(nextState);
            }
            nextQ = ComputeQValue(nextState, nextAction);
        }

        // Compute TD target and error
        T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextQ));
        T tdError = NumOps.Subtract(target, currentQ);

        // Update weights: w_a ← w_a + α * δ * φ(s)
        T learningRateT = NumOps.Multiply(LearningRate, tdError);
        for (int f = 0; f < _options.FeatureSize; f++)
        {
            T update = NumOps.Multiply(learningRateT, _lastState[f]);
            _weights[_lastAction, f] = NumOps.Add(_weights[_lastAction, f], update);
        }

        if (done)
        {
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
            _lastAction = -1;
            _lastState = null;
        }
    }

    public override T Train() => NumOps.Zero;

    private T ComputeQValue(Vector<T> features, int actionIndex)
    {
        T qValue = NumOps.Zero;
        for (int f = 0; f < _options.FeatureSize; f++)
        {
            T weightedFeature = NumOps.Multiply(_weights[actionIndex, f], features[f]);
            qValue = NumOps.Add(qValue, weightedFeature);
        }
        return qValue;
    }

    private int GetGreedyAction(Vector<T> state)
    {
        int bestAction = 0;
        T bestValue = ComputeQValue(state, 0);

        for (int a = 1; a < _options.ActionSize; a++)
        {
            T value = ComputeQValue(state, a);
            if (NumOps.GreaterThan(value, bestValue))
            {
                bestValue = value;
                bestAction = a;
            }
        }

        return bestAction;
    }

    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T>
    {
        ["epsilon"] = NumOps.FromDouble(_epsilon),
        ["weight_norm"] = ComputeWeightNorm()
    };

    private T ComputeWeightNorm()
    {
        T sumSquares = NumOps.Zero;
        for (int a = 0; a < _options.ActionSize; a++)
        {
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                T squared = NumOps.Multiply(_weights[a, f], _weights[a, f]);
                sumSquares = NumOps.Add(sumSquares, squared);
            }
        }
        return NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sumSquares)));
    }

    public override void ResetEpisode()
    {
        _lastAction = -1;
        _lastState = null;
    }

    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int ParameterCount => _options.ActionSize * _options.FeatureSize;
    public override int FeatureCount => _options.FeatureSize;
    public override byte[] Serialize()
    {
        var state = new
        {
            Weights = GetParameters(),
            Epsilon = _epsilon,
            LastAction = _lastAction,
            Options = _options
        };
        string json = Newtonsoft.Json.JsonConvert.SerializeObject(state);
        return System.Text.Encoding.UTF8.GetBytes(json);
    }
    public override void Deserialize(byte[] data)
    {
        string json = System.Text.Encoding.UTF8.GetString(data);
        var state = Newtonsoft.Json.JsonConvert.DeserializeObject<dynamic>(json);
        if (state is not null)
        {
            var weightsObj = state.Weights;
            if (weightsObj is Newtonsoft.Json.Linq.JArray jArray)
            {
                var weights = new Vector<T>(jArray.Count);
                for (int i = 0; i < jArray.Count; i++)
                {
                    weights[i] = NumOps.FromDouble((double)jArray[i]);
                }
                SetParameters(weights);
            }
            if (state.Epsilon != null)
            {
                _epsilon = (double)state.Epsilon;
            }
            if (state.LastAction != null)
            {
                _lastAction = (int)state.LastAction;
            }
        }
    }

    public override Vector<T> GetParameters()
    {
        int paramCount = _options.ActionSize * _options.FeatureSize;
        var vector = new Vector<T>(paramCount);
        int idx = 0;

        for (int a = 0; a < _options.ActionSize; a++)
            for (int f = 0; f < _options.FeatureSize; f++)
                vector[idx++] = _weights[a, f];

        return vector;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        for (int a = 0; a < _options.ActionSize; a++)
            for (int f = 0; f < _options.FeatureSize; f++)
                if (idx < parameters.Length)
                    _weights[a, f] = parameters[idx++];
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone() => new LinearSARSAAgent<T>(_options);

    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var pred = Predict(input);
        var lf = lossFunction ?? LossFunction;
        var loss = lf.CalculateLoss(pred, target);
        var grad = lf.CalculateDerivative(pred, target);
        return grad;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate) { }
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
