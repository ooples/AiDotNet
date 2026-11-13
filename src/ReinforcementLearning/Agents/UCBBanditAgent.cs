using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using AiDotNet.Validation;

using AiDotNet.ReinforcementLearning.Parameters;

namespace AiDotNet.ReinforcementLearning.Agents.Bandits;

/// <summary>
/// Upper Confidence Bound (UCB) Multi-Armed Bandit agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> UCB balances exploration and exploitation mathematically.
/// For each action, it calculates an optimistic estimate: average reward + confidence bonus.
/// The confidence bonus is larger for actions tried fewer times, ensuring under-explored
/// actions get a fair chance. Think of it as "optimism in the face of uncertainty" - if you
/// are not sure about an action, assume it could be great until proven otherwise. No epsilon
/// parameter needed; exploration happens automatically.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a UCB bandit agent for optimistic exploration
/// var options = new UCBBanditOptions&lt;double&gt; { NumArms = 10, ExplorationParameter = 2.0 };
/// var agent = new UCBBanditAgent&lt;double&gt;(options);
///
/// // Select an arm using upper confidence bound
/// var state = new Vector&lt;double&gt;(new double[] { 1.0 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Finite-time Analysis of the Multiarmed Bandit Problem",
    "https://doi.org/10.1023/A:1013689704352",
    Year = 2002,
    Authors = "Auer, P., Cesa-Bianchi, N., & Fischer, P.")]
public partial class UCBBanditAgent<T> : ReinforcementLearningAgentBase<T>
{

    /// <inheritdoc />
    /// <remarks>The per-arm action values this bandit learns.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VectorParameterSource<T>(() => _qValues));
    }
    private UCBBanditOptions<T> _options;

    private Random _random;
    private Vector<T> _qValues;
    private Vector<int> _actionCounts;
    private int _totalSteps;

    public UCBBanditAgent() : this(new UCBBanditOptions<T>()) { }

    public UCBBanditAgent(UCBBanditOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        Options = _options;
        _random = Random;
        _qValues = Vector<T>.CreateDefault(_options.NumArms, NumOps.Zero);
        _actionCounts = new Vector<int>(_options.NumArms);
        _totalSteps = 0;
    }

    /// <inheritdoc />
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        _totalSteps++;

        // Select arm with highest UCB value Q(a) + c * sqrt(ln t / N(a)). An unvisited arm scores
        // +infinity, so the first unvisited arm wins outright (explore unvisited arms first).
        var engine = AiDotNetEngine.Current;
        var counts = RLTensorOps.CountsAs<T>(_actionCounts);
        var unvisited = engine.TensorEquals(counts.AsTensor(), NumOps.Zero).ToVector();
        int selectedArm = RLTensorOps.FirstPositiveIndex(unvisited, NumOps.Zero, out bool hasUnvisited);
        if (!hasUnvisited && _options.NumArms > 0)
        {
            // ln(t) / N(a) from the integer counts (in Half, t past 65504 would make ln(t) infinite for every arm).
            var ratio = RLTensorOps.UcbRatio<T>(_totalSteps, _actionCounts);
            var exploration = engine.Multiply(engine.Sqrt(ratio), NumOps.FromDouble(_options.ExplorationParameter));
            selectedArm = RLTensorOps.ArgMax(engine.Add(_qValues, exploration));
        }

        var result = new Vector<T>(_options.NumArms);
        result[selectedArm] = NumOps.One;
        return result;
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int armIndex = ArgMax(action);
        _actionCounts[armIndex]++;

        T currentQ = _qValues[armIndex];
        T alpha = NumOps.Divide(NumOps.One, NumOps.FromDouble(_actionCounts[armIndex]));
        T delta = NumOps.Subtract(reward, currentQ);
        _qValues[armIndex] = NumOps.Add(currentQ, NumOps.Multiply(alpha, delta));
    }

    /// <inheritdoc />
    public override T Train() => NumOps.Zero;

    private int ArgMax(Vector<T> values)
    {
        return RLTensorOps.ArgMax(values);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetMetrics()
    {
        var metrics = new Dictionary<string, T>();
        RLTensorOps.AddIndexedMetrics(metrics, _options.NumArms,
            ("q_arm_", _qValues), ("count_arm_", RLTensorOps.CountsAs<T>(_actionCounts)));
        metrics["total_steps"] = NumOps.FromDouble(_totalSteps);
        return metrics;
    }

    /// <inheritdoc />
    public override void ResetEpisode()
    {
        _totalSteps = 0;
        _qValues.Fill(NumOps.Zero);
        _actionCounts.Fill(0);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    /// <inheritdoc />
    public override int FeatureCount => 1;
    /// <inheritdoc />
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    /// <inheritdoc />
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
