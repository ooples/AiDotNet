using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.AdvancedRL;

/// <summary>
/// Linear Q-Learning agent using linear function approximation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LinearQLearningAgent<T> : ReinforcementLearningAgentBase<T>
{
    private LinearQLearningOptions<T> _options;
    private Matrix<T> _weights;  // Weight matrix: [ActionSize x FeatureSize]
    private double _epsilon;

    public LinearQLearningAgent(LinearQLearningOptions<T> options) : base(options)
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

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int actionIndex = ArgMax(action);

        // Compute current Q-value: Q(s,a) = w_a^T * φ(s)
        T currentQ = ComputeQValue(state, actionIndex);

        // Compute max Q-value for next state
        T maxNextQ = NumOps.Zero;
        if (!done)
        {
            int bestNextAction = GetGreedyAction(nextState);
            maxNextQ = ComputeQValue(nextState, bestNextAction);
        }

        // Compute TD target and error
        T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, maxNextQ));
        T tdError = NumOps.Subtract(target, currentQ);

        // Update weights: w_a ← w_a + α * δ * φ(s)
        T learningRateT = NumOps.Multiply(LearningRate, tdError);
        for (int f = 0; f < _options.FeatureSize; f++)
        {
            T update = NumOps.Multiply(learningRateT, state[f]);
            _weights[actionIndex, f] = NumOps.Add(_weights[actionIndex, f], update);
        }

        if (done)
        {
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
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

    private int ArgMax(Vector<T> values)
    {
        int maxIndex = 0;
        T maxValue = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (NumOps.GreaterThan(values[i], maxValue))
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }
        return maxIndex;
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

    public override void ResetEpisode() { }
    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public override Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public override Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int ParameterCount => _options.ActionSize * _options.FeatureSize;
    public override int FeatureCount => _options.FeatureSize;
    public override byte[] Serialize() => throw new NotImplementedException();
    public override void Deserialize(byte[] data) => throw new NotImplementedException();

    public override Matrix<T> GetParameters()
    {
        int paramCount = _options.ActionSize * _options.FeatureSize;
        var vector = new Vector<T>(paramCount);
        int idx = 0;

        for (int a = 0; a < _options.ActionSize; a++)
            for (int f = 0; f < _options.FeatureSize; f++)
                vector[idx++] = _weights[a, f];

        return new Matrix<T>(new[] { vector });
    }

    public override void SetParameters(Matrix<T> parameters)
    {
        int idx = 0;
        for (int a = 0; a < _options.ActionSize; a++)
            for (int f = 0; f < _options.FeatureSize; f++)
                if (idx < parameters.Columns)
                    _weights[a, f] = parameters[0, idx++];
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone() => new LinearQLearningAgent<T>(_options);

    public override (Matrix<T> Gradients, T Loss) ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var pred = Predict(input);
        var lf = lossFunction ?? LossFunction;
        var loss = lf.CalculateLoss(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target }));
        var grad = lf.CalculateDerivative(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target }));
        return (grad, loss);
    }

    public override void ApplyGradients(Matrix<T> gradients, T learningRate) { }
    public override void Save(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void Load(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
