using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.Bandits;

/// <summary>
/// Upper Confidence Bound (UCB) Multi-Armed Bandit agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class UCBBanditAgent<T> : ReinforcementLearningAgentBase<T>
{
    private UCBBanditOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Random _random;
    private Vector<T> _qValues;
    private Vector<int> _actionCounts;
    private int _totalSteps;

    public UCBBanditAgent(UCBBanditOptions<T> options) : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _random = RandomHelper.CreateSecureRandom();
        _qValues = new Vector<T>(_options.NumArms);
        _actionCounts = new Vector<int>(_options.NumArms);
        _totalSteps = 0;
        for (int i = 0; i < _options.NumArms; i++)
        {
            _qValues[i] = NumOps.Zero;
            _actionCounts[i] = 0;
        }
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        _totalSteps++;

        // Select arm with highest UCB value
        int selectedArm = 0;
        double maxUCB = double.NegativeInfinity;

        for (int a = 0; a < _options.NumArms; a++)
        {
            double ucb;
            if (_actionCounts[a] == 0)
            {
                ucb = double.PositiveInfinity;  // Explore unvisited arms first
            }
            else
            {
                double exploitation = NumOps.ToDouble(_qValues[a]);
                double exploration = _options.ExplorationParameter * Math.Sqrt(Math.Log(_totalSteps) / _actionCounts[a]);
                ucb = exploitation + exploration;
            }

            if (ucb > maxUCB)
            {
                maxUCB = ucb;
                selectedArm = a;
            }
        }

        var result = new Vector<T>(_options.NumArms);
        result[selectedArm] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int armIndex = ArgMax(action);
        _actionCounts[armIndex]++;

        T currentQ = _qValues[armIndex];
        T alpha = NumOps.Divide(NumOps.One, NumOps.FromDouble(_actionCounts[armIndex]));
        T delta = NumOps.Subtract(reward, currentQ);
        _qValues[armIndex] = NumOps.Add(currentQ, NumOps.Multiply(alpha, delta));
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
        for (int i = 0; i < _options.NumArms; i++)
        {
            metrics[$"q_arm_{i}"] = _qValues[i];
            metrics[$"count_arm_{i}"] = NumOps.FromDouble(_actionCounts[i]);
        }
        metrics["total_steps"] = NumOps.FromDouble(_totalSteps);
        return metrics;
    }

    public override void ResetEpisode()
    {
        _totalSteps = 0;
        for (int i = 0; i < _options.NumArms; i++)
        {
            _qValues[i] = NumOps.Zero;
            _actionCounts[i] = 0;
        }
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
        writer.Write(_options.ExplorationParameter);

        // Write state
        writer.Write(_totalSteps);
        for (int i = 0; i < _options.NumArms; i++)
        {
            writer.Write(NumOps.ToDouble(_qValues[i]));
            writer.Write(_actionCounts[i]);
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read and validate options
        var numArms = reader.ReadInt32();
        var explorationParam = reader.ReadDouble();

        if (numArms != _options.NumArms)
            throw new InvalidOperationException($"Serialized NumArms ({numArms}) doesn't match current options ({_options.NumArms})");

        // Read state
        _totalSteps = reader.ReadInt32();
        for (int i = 0; i < _options.NumArms; i++)
        {
            _qValues[i] = NumOps.FromDouble(reader.ReadDouble());
            _actionCounts[i] = reader.ReadInt32();
        }
    }
    public override Vector<T> GetParameters() => _qValues;
    public override void SetParameters(Vector<T> parameters) { for (int i = 0; i < _options.NumArms && i < parameters.Length; i++) _qValues[i] = parameters[i]; }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new UCBBanditAgent<T>(_options);

        // Deep copy learned state to preserve training
        clone._qValues = new Vector<T>(_options.NumArms);
        clone._actionCounts = new Vector<int>(_options.NumArms);
        for (int i = 0; i < _options.NumArms; i++)
        {
            clone._qValues[i] = _qValues[i];
            clone._actionCounts[i] = _actionCounts[i];
        }
        clone._totalSteps = _totalSteps;

        return clone;
    }
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var predMatrix = new Matrix<T>(new[] { pred }); var targetMatrix = new Matrix<T>(new[] { target }); var loss = lf.CalculateLoss(predMatrix.GetRow(0), targetMatrix.GetRow(0)); var grad = lf.CalculateDerivative(predMatrix.GetRow(0), targetMatrix.GetRow(0)); return grad; }
    public override void ApplyGradients(Vector<T> gradients, T learningRate) { }
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
