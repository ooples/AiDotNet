using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.DynamicProgramming;

/// <summary>
/// Value Iteration agent for reinforcement learning using dynamic programming.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Value Iteration combines policy evaluation and improvement in a single update step,
/// converging to the optimal value function.
/// </remarks>
public class ValueIterationAgent<T> : ReinforcementLearningAgentBase<T>
{
    private ValueIterationOptions<T> _options;
    private Dictionary<string, T> _valueTable;
    private Dictionary<string, Dictionary<int, List<(string nextState, T reward, T probability)>>> _model;

    public ValueIterationAgent(ValueIterationOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _valueTable = new Dictionary<string, T>();
        _model = new Dictionary<string, Dictionary<int, List<(string, T, T)>>>();
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        string stateKey = GetStateKey(state);

        // Initialize value for new states
        if (!_valueTable.ContainsKey(stateKey))
        {
            _valueTable[stateKey] = NumOps.Zero;
        }

        // Select action greedily with respect to value function
        int bestAction = 0;
        T bestValue = NumOps.FromDouble(double.NegativeInfinity);

        for (int a = 0; a < _options.ActionSize; a++)
        {
            T actionValue = ComputeActionValue(stateKey, a);

            if (NumOps.GreaterThan(actionValue, bestValue))
            {
                bestValue = actionValue;
                bestAction = a;
            }
        }

        var result = new Vector<T>(_options.ActionSize);
        result[bestAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // Build model from experience
        string stateKey = GetStateKey(state);
        string nextStateKey = GetStateKey(nextState);
        int actionIndex = ArgMax(action);

        if (!_model.ContainsKey(stateKey))
        {
            _model[stateKey] = new Dictionary<int, List<(string, T, T)>>();
        }

        if (!_model[stateKey].ContainsKey(actionIndex))
        {
            _model[stateKey][actionIndex] = new List<(string, T, T)>();
        }

        // Add transition (assuming deterministic for simplicity)
        _model[stateKey][actionIndex].Add((nextStateKey, reward, NumOps.One));
    }

    public override T Train()
    {
        if (_model.Count == 0)
        {
            return NumOps.Zero;
        }

        T delta;
        int iterations = 0;

        do
        {
            delta = NumOps.Zero;

            foreach (var stateKey in _valueTable.Keys.ToList())
            {
                T oldValue = _valueTable[stateKey];

                // Find max action value (Bellman optimality equation)
                T maxActionValue = NumOps.FromDouble(double.NegativeInfinity);

                for (int a = 0; a < _options.ActionSize; a++)
                {
                    T actionValue = ComputeActionValue(stateKey, a);

                    if (NumOps.GreaterThan(actionValue, maxActionValue))
                    {
                        maxActionValue = actionValue;
                    }
                }

                _valueTable[stateKey] = maxActionValue;

                // Track maximum change
                T diff = NumOps.Subtract(maxActionValue, oldValue);
                T absDiff = NumOps.Compare(diff, NumOps.Zero) >= 0 ? diff : NumOps.Negate(diff);
                if (NumOps.GreaterThan(absDiff, delta))
                {
                    delta = absDiff;
                }
            }

            iterations++;
        }
        while (NumOps.Compare(delta, NumOps.FromDouble(_options.Theta)) >= 0 && iterations < _options.MaxIterations);

        return NumOps.FromDouble(iterations);
    }

    private T ComputeActionValue(string stateKey, int action)
    {
        if (!_model.ContainsKey(stateKey) || !_model[stateKey].ContainsKey(action))
        {
            return NumOps.Zero;
        }

        T expectedValue = NumOps.Zero;

        foreach (var (nextStateKey, reward, probability) in _model[stateKey][action])
        {
            T nextValue = NumOps.Zero;
            if (_valueTable.ContainsKey(nextStateKey))
            {
                nextValue = _valueTable[nextStateKey];
            }

            T transitionValue = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextValue));
            expectedValue = NumOps.Add(expectedValue, NumOps.Multiply(probability, transitionValue));
        }

        return expectedValue;
    }

    private string GetStateKey(Vector<T> state)
    {
        return string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
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

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["states_visited"] = NumOps.FromDouble(_valueTable.Count),
            ["model_transitions"] = NumOps.FromDouble(_model.Count)
        };
    }

    public override void ResetEpisode()
    {
        // No episode-specific state
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = "ValueIteration",
            InputSize = _options.StateSize,
            OutputSize = _options.ActionSize,
            ParameterCount = ParameterCount
        };
    }

    public override int ParameterCount => _valueTable.Count;

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        throw new NotImplementedException("ValueIteration serialization not yet implemented");
    }

    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("ValueIteration deserialization not yet implemented");
    }

    public override Matrix<T> GetParameters()
    {
        var paramsList = new List<T>();
        foreach (var value in _valueTable.Values)
        {
            paramsList.Add(value);
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

        return new Matrix<T>(new[] { paramsVector });
    }

    public override void SetParameters(Matrix<T> parameters)
    {
        int index = 0;
        foreach (var stateKey in _valueTable.Keys.ToList())
        {
            if (index < parameters.Columns)
            {
                _valueTable[stateKey] = parameters[0, index];
                index++;
            }
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        return new ValueIterationAgent<T>(_options);
    }

    public override (Matrix<T> Gradients, T Loss) ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var loss = usedLossFunction.CalculateLoss(new Matrix<T>(new[] { prediction }), new Matrix<T>(new[] { target }));

        var gradient = usedLossFunction.CalculateDerivative(new Matrix<T>(new[] { prediction }), new Matrix<T>(new[] { target }));
        return (gradient, loss);
    }

    public override void ApplyGradients(Matrix<T> gradients, T learningRate)
    {
        // DP methods don't use gradients
    }

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
