using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.Bandits;

/// <summary>
/// Gradient Bandit agent using softmax action preferences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GradientBanditAgent<T> : ReinforcementLearningAgentBase<T>
{
    private GradientBanditOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Random _random;
    private Vector<T> _preferences;  // H(a)
    private T _averageReward;
    private int _totalSteps;

    public GradientBanditAgent(GradientBanditOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _random = RandomHelper.CreateSecureRandom();
        _preferences = new Vector<T>(_options.NumArms);
        for (int i = 0; i < _options.NumArms; i++)
        {
            _preferences[i] = NumOps.Zero;
        }
        _averageReward = NumOps.Zero;
        _totalSteps = 0;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Compute softmax probabilities
        var probs = ComputeSoftmax(_preferences);

        // Sample action according to probabilities
        double r = _random.NextDouble();
        double cumulative = 0.0;
        int selectedArm = 0;

        for (int a = 0; a < _options.NumArms; a++)
        {
            cumulative += NumOps.ToDouble(probs[a]);
            if (r <= cumulative)
            {
                selectedArm = a;
                break;
            }
        }

        var result = new Vector<T>(_options.NumArms);
        result[selectedArm] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int armIndex = ArgMax(action);
        _totalSteps++;

        // Update average reward baseline
        if (_options.UseBaseline)
        {
            T alpha = NumOps.Divide(NumOps.One, NumOps.FromDouble(_totalSteps));
            T delta = NumOps.Subtract(reward, _averageReward);
            _averageReward = NumOps.Add(_averageReward, NumOps.Multiply(alpha, delta));
        }

        // Compute softmax probabilities
        var probs = ComputeSoftmax(_preferences);

        // Gradient update: H(a) ← H(a) + α(R - R̄)(1 - π(a)) for selected action
        //                  H(a) ← H(a) - α(R - R̄)π(a) for other actions
        T rewardDiff = NumOps.Subtract(reward, _averageReward);
        T stepSize = NumOps.FromDouble(_options.Alpha);

        for (int a = 0; a < _options.NumArms; a++)
        {
            if (a == armIndex)
            {
                // Selected action
                T update = NumOps.Multiply(stepSize, NumOps.Multiply(rewardDiff, NumOps.Subtract(NumOps.One, probs[a])));
                _preferences[a] = NumOps.Add(_preferences[a], update);
            }
            else
            {
                // Non-selected actions
                T update = NumOps.Multiply(stepSize, NumOps.Multiply(rewardDiff, NumOps.Negate(probs[a])));
                _preferences[a] = NumOps.Add(_preferences[a], update);
            }
        }
    }

    private Vector<T> ComputeSoftmax(Vector<T> preferences)
    {
        // Find max for numerical stability
        T maxPref = preferences[0];
        for (int i = 1; i < preferences.Length; i++)
        {
            if (NumOps.GreaterThan(preferences[i], maxPref))
            {
                maxPref = preferences[i];
            }
        }

        // Compute exp(H(a) - max)
        var expValues = new Vector<T>(preferences.Length);
        T sumExp = NumOps.Zero;
        for (int i = 0; i < preferences.Length; i++)
        {
            T expVal = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Subtract(preferences[i], maxPref))));
            expValues[i] = expVal;
            sumExp = NumOps.Add(sumExp, expVal);
        }

        // Normalize
        var probs = new Vector<T>(preferences.Length);
        for (int i = 0; i < preferences.Length; i++)
        {
            probs[i] = NumOps.Divide(expValues[i], sumExp);
        }

        return probs;
    }

    public override T Train() => NumOps.Zero;

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
        var metrics = new Dictionary<string, T>();
        var probs = ComputeSoftmax(_preferences);
        for (int i = 0; i < _options.NumArms; i++)
        {
            metrics[$"preference_arm_{i}"] = _preferences[i];
            metrics[$"probability_arm_{i}"] = probs[i];
        }
        metrics["average_reward"] = _averageReward;
        return metrics;
    }

    public override void ResetEpisode()
    {
        for (int i = 0; i < _options.NumArms; i++)
        {
            _preferences[i] = NumOps.Zero;
        }
        _averageReward = NumOps.Zero;
        _totalSteps = 0;
    }

    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int ParameterCount => _options.NumArms;
    public override int FeatureCount => 1;
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write options
        writer.Write(_options.NumArms);
        writer.Write(_options.Alpha);
        writer.Write(_options.UseBaseline);

        // Write state
        writer.Write(_totalSteps);
        writer.Write(NumOps.ToDouble(_averageReward));
        for (int i = 0; i < _options.NumArms; i++)
        {
            writer.Write(NumOps.ToDouble(_preferences[i]));
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read and validate options
        var numArms = reader.ReadInt32();
        var alpha = reader.ReadDouble();
        var useBaseline = reader.ReadBoolean();

        if (numArms != _options.NumArms)
            throw new InvalidOperationException($"Serialized NumArms ({numArms}) doesn't match current options ({_options.NumArms})");

        // Read state
        _totalSteps = reader.ReadInt32();
        _averageReward = NumOps.FromDouble(reader.ReadDouble());
        for (int i = 0; i < _options.NumArms; i++)
        {
            _preferences[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
    public override Vector<T> GetParameters() => _preferences;
    public override void SetParameters(Vector<T> parameters) { for (int i = 0; i < _options.NumArms && i < parameters.Length; i++) _preferences[i] = parameters[i]; }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new GradientBanditAgent<T>(_options);
        // Copy preferences and baseline to preserve learned state
        for (int i = 0; i < _options.NumArms; i++)
        {
            clone._preferences[i] = _preferences[i];
        }
        clone._averageReward = _averageReward;
        clone._totalSteps = _totalSteps;
        return clone;
    }
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var predMatrix = new Matrix<T>(new[] { pred }); var targetMatrix = new Matrix<T>(new[] { target }); var loss = lf.CalculateLoss(predMatrix.GetRow(0), targetMatrix.GetRow(0)); var grad = lf.CalculateDerivative(predMatrix.GetRow(0), targetMatrix.GetRow(0)); return grad; }
    public override void ApplyGradients(Vector<T> gradients, T learningRate) { }
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
