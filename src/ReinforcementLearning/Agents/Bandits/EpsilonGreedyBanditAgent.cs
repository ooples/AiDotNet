using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.Bandits;

/// <summary>
/// Epsilon-Greedy Multi-Armed Bandit agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EpsilonGreedyBanditAgent<T> : ReinforcementLearningAgentBase<T>
{
    private EpsilonGreedyBanditOptions<T> _options;
    private Vector<T> _qValues;
    private Vector<int> _actionCounts;

    public EpsilonGreedyBanditAgent(EpsilonGreedyBanditOptions<T> options) : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _qValues = new Vector<T>(_options.NumArms);
        _actionCounts = new Vector<int>(_options.NumArms);
        for (int i = 0; i < _options.NumArms; i++)
        {
            _qValues[i] = NumOps.Zero;
            _actionCounts[i] = 0;
        }
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        int selectedArm;
        if (training && Random.NextDouble() < _options.Epsilon)
        {
            selectedArm = Random.Next(_options.NumArms);
        }
        else
        {
            selectedArm = ArgMax(_qValues);
        }

        var result = new Vector<T>(_options.NumArms);
        result[selectedArm] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int armIndex = ArgMax(action);
        _actionCounts[armIndex]++;

        // Incremental update: Q(a) ← Q(a) + (1/N)(R - Q(a))
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
        return metrics;
    }

    public override void ResetEpisode()
    {
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
    public override byte[] Serialize() => throw new NotImplementedException();
    public override void Deserialize(byte[] data) => throw new NotImplementedException();
    public override Matrix<T> GetParameters() => new Matrix<T>(new[] { _qValues });
    public override void SetParameters(Matrix<T> parameters) { for (int i = 0; i < _options.NumArms && i < parameters.Columns; i++) _qValues[i] = parameters[0, i]; }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone() => new EpsilonGreedyBanditAgent<T>(_options);
    public override (Matrix<T> Gradients, T Loss) ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var loss = lf.CalculateLoss(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target })); var grad = lf.CalculateDerivative(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target })); return (grad, loss); }
    public override void ApplyGradients(Matrix<T> gradients, T learningRate) { }
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
