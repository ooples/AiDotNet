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

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
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
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int ParameterCount => _options.ActionSize * _options.FeatureSize;
    public override int FeatureCount => _options.FeatureSize;
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write options
        writer.Write(_options.ActionSize);
        writer.Write(_options.FeatureSize);
        writer.Write(_options.EpsilonStart);
        writer.Write(_options.EpsilonEnd);
        writer.Write(_options.EpsilonDecay);

        // Write base class properties that must be serialized
        writer.Write(NumOps.ToDouble(_options.LearningRate ?? NumOps.Zero));
        writer.Write(NumOps.ToDouble(_options.DiscountFactor ?? NumOps.Zero));
        writer.Write(_options.Seed ?? -1); // Use -1 to indicate no seed
        // Serialize loss function type name for reconstruction
        string lossFunctionTypeName = _options.LossFunction?.GetType().AssemblyQualifiedName ?? string.Empty;
        writer.Write(lossFunctionTypeName);

        // Write current epsilon
        writer.Write(_epsilon);

        // Write weights matrix
        writer.Write(_weights.Rows);
        writer.Write(_weights.Columns);
        for (int a = 0; a < _weights.Rows; a++)
        {
            for (int f = 0; f < _weights.Columns; f++)
            {
                writer.Write(NumOps.ToDouble(_weights[a, f]));
            }
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read options
        int actionSize = reader.ReadInt32();
        int featureSize = reader.ReadInt32();
        double epsilonStart = reader.ReadDouble();
        double epsilonEnd = reader.ReadDouble();
        double epsilonDecay = reader.ReadDouble();

        // Read base class properties from serialized data
        double learningRate = reader.ReadDouble();
        double discountFactor = reader.ReadDouble();
        int seedValue = reader.ReadInt32();
        int? seed = seedValue == -1 ? null : seedValue;
        string lossFunctionTypeName = reader.ReadString();

        // Reconstruct loss function from type name
        ILossFunction<T>? lossFunction = null;
        if (!string.IsNullOrEmpty(lossFunctionTypeName))
        {
            Type? lossFunctionType = Type.GetType(lossFunctionTypeName);
            if (lossFunctionType is not null)
            {
                lossFunction = (ILossFunction<T>?)Activator.CreateInstance(lossFunctionType);
            }
        }
        // Fall back to existing loss function if reconstruction failed
        lossFunction ??= _options.LossFunction;

        _options = new LinearQLearningOptions<T>
        {
            ActionSize = actionSize,
            FeatureSize = featureSize,
            EpsilonStart = epsilonStart,
            EpsilonEnd = epsilonEnd,
            EpsilonDecay = epsilonDecay,
            LearningRate = NumOps.FromDouble(learningRate),
            DiscountFactor = NumOps.FromDouble(discountFactor),
            Seed = seed,
            LossFunction = lossFunction
        };

        // Read current epsilon
        _epsilon = reader.ReadDouble();

        // Read weights matrix
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _weights = new Matrix<T>(rows, cols);
        for (int a = 0; a < rows; a++)
        {
            for (int f = 0; f < cols; f++)
            {
                _weights[a, f] = NumOps.FromDouble(reader.ReadDouble());
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

    public override IFullModel<T, Vector<T>, Vector<T>> Clone() => new LinearQLearningAgent<T>(_options);

    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var pred = Predict(input);
        var lf = lossFunction ?? LossFunction;
        var loss = lf.CalculateLoss(pred, target);
        var gradients = lf.CalculateDerivative(pred, target);
        return gradients;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (gradients == null)
        {
            throw new ArgumentNullException(nameof(gradients));
        }

        // Gradients should be flattened from weight matrix [ActionSize x FeatureSize]
        int expectedSize = _options.ActionSize * _options.FeatureSize;
        if (gradients.Length != expectedSize)
        {
            throw new ArgumentException($"Gradient vector length {gradients.Length} does not match expected size {expectedSize}");
        }

        // Apply gradients to weight matrix
        int index = 0;
        for (int a = 0; a < _options.ActionSize; a++)
        {
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                T update = NumOps.Multiply(learningRate, gradients[index]);
                _weights[a, f] = NumOps.Subtract(_weights[a, f], update);  // Gradient descent
                index++;
            }
        }
    }
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
