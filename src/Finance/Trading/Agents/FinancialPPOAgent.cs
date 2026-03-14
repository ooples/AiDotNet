using AiDotNet.Attributes;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.LossFunctions;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Financial Proximal Policy Optimization (PPO) agent for robust trading.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The PPO (Proximal Policy Optimization) trading agent is one
/// of the most reliable RL algorithms for trading. It prevents the agent from making too
/// large a policy change in any single update, which keeps learning stable. Think of it
/// as a cautious trader who adjusts their strategy gradually rather than making radical
/// shifts. PPO balances exploration (trying new strategies) with exploitation (sticking
/// with what works), making it robust for financial applications.</para>
/// </remarks>
/// <example>
/// <code>
/// // Define actor and critic architectures for PPO trading (30 state features, 5 continuous actions)
/// var actorArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 30, outputSize: 5);
/// var criticArch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 30, outputSize: 1);
///
/// // Create PPO agent for stable, robust trading policy optimization
/// var options = new TradingAgentOptions&lt;double&gt;();
/// var model = new FinancialPPOAgent&lt;double&gt;(actorArch, criticArch, options);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.ReinforcementLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Proximal Policy Optimization Algorithms", "https://arxiv.org/abs/1707.06347", Year = 2017, Authors = "John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov")]
public class FinancialPPOAgent<T> : TradingAgentBase<T>
{
    #region Fields

    private readonly FinancialPPOAgentOptions<T> _options;
    private readonly INeuralNetwork<T> _actor;
    private readonly INeuralNetwork<T> _critic;
    private readonly ReplayBuffer<T> ReplayBuffer;
    private readonly NeuralNetworkArchitecture<T> _actorArchitecture;
    private readonly NeuralNetworkArchitecture<T> _criticArchitecture;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    /// <inheritdoc/>
    public override int ParameterCount => _actor.ParameterCount + _critic.ParameterCount;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FinancialPPOAgent class.
    /// </summary>
    /// <param name="actorArchitecture">User-provided architecture for the policy (actor).</param>
    /// <param name="criticArchitecture">User-provided architecture for the value (critic).</param>
    /// <param name="options">Configuration options for the trading agent.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, FinancialPPOAgent sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinancialPPOAgent(
        NeuralNetworkArchitecture<T> actorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        TradingAgentOptions<T> options)
        : base(options)
    {
        _options = options as FinancialPPOAgentOptions<T> ?? new FinancialPPOAgentOptions<T>();
        _actorArchitecture = actorArchitecture;
        _criticArchitecture = criticArchitecture;

        EnsureDefaultLayers(actorArchitecture, options.StateSize, options.ActionSize);
        EnsureDefaultLayers(criticArchitecture, options.StateSize, 1);

        _actor = new NeuralNetwork<T>(actorArchitecture, TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _critic = new NeuralNetwork<T>(criticArchitecture, TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        ReplayBuffer = new ReplayBuffer<T>(options.ReplayBufferSize, options.Seed);
    }

    #endregion

    #region Action Selection

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, SelectAction performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var probs = _actor.Predict(Tensor<T>.FromVector(state)).ToVector();
        
        if (training)
        {
            int actionIdx = SampleAction(probs);
            var action = new Vector<T>(TradingOptions.ActionSize);
            action[actionIdx] = NumOps.One;
            return action;
        }

        int bestIdx = 0;
        T maxProb = probs[0];
        for (int i = 1; i < probs.Length; i++)
        {
            if (NumOps.GreaterThan(probs[i], maxProb))
            {
                maxProb = probs[i];
                bestIdx = i;
            }
        }

        var result = new Vector<T>(TradingOptions.ActionSize);
        result[bestIdx] = NumOps.One;
        return result;
    }

    /// <summary>
    /// Executes SampleAction for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, SampleAction performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private int SampleAction(Vector<T> probabilities)
    {
        double r = RandomHelper.CreateSecureRandom().NextDouble();
        double cumulative = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += NumOps.ToDouble(probabilities[i]);
            if (r < cumulative) return i;
        }
        return probabilities.Length - 1;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, Train performs a training step. This updates the FinancialPPOAgent architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override T Train()
    {
        if (ReplayBuffer.Count < TradingOptions.BatchSize) return NumOps.Zero;

        var batch = ReplayBuffer.Sample(TradingOptions.BatchSize);
        T totalLoss = NumOps.Zero;

        foreach (var exp in batch)
        {
            T vCurrent = _critic.Predict(Tensor<T>.FromVector(exp.State)).Data.Span[0];
            T vNext = exp.Done ? NumOps.Zero : _critic.Predict(Tensor<T>.FromVector(exp.NextState)).Data.Span[0];
            T advantage = NumOps.Subtract(NumOps.Add(exp.Reward, NumOps.Multiply(NumOps.FromDouble(Convert.ToDouble(TradingOptions.DiscountFactor)), vNext)), vCurrent);

            // Update Critic
            T targetV = NumOps.Add(exp.Reward, NumOps.Multiply(NumOps.FromDouble(Convert.ToDouble(TradingOptions.DiscountFactor)), vNext));
            var targetVVec = new Vector<T>(new[] { targetV });
            _critic.Train(Tensor<T>.FromVector(exp.State), Tensor<T>.FromVector(targetVVec));

            // Update Actor
            // Calculate loss manually
            T actorLoss = (TradingOptions.LossFunction ?? throw new InvalidOperationException("LossFunction has not been initialized.")).CalculateLoss(
                _actor.Predict(Tensor<T>.FromVector(exp.State)).ToVector(), 
                exp.Action);
            
            _actor.Train(Tensor<T>.FromVector(exp.State), Tensor<T>.FromVector(exp.Action)); 

            totalLoss = NumOps.Add(totalLoss, actorLoss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(TradingOptions.BatchSize));
    }

    #endregion

    #region Base Implementation

    /// <summary>
    /// Executes LoadModel for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, LoadModel performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, SaveModel performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Executes StoreExperience for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, StoreExperience performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        var experience = new Experience<T>(state, action, reward, nextState, done);
        ReplayBuffer.Add(experience);
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Executes Serialize for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, Serialize saves or restores model-specific settings. This lets the FinancialPPOAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        var actorData = _actor.Serialize();
        var criticData = _critic.Serialize();
        writer.Write(actorData.Length);
        writer.Write(actorData);
        writer.Write(criticData.Length);
        writer.Write(criticData);
        return ms.ToArray();
    }

    /// <summary>
    /// Executes Deserialize for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, Deserialize saves or restores model-specific settings. This lets the FinancialPPOAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        int actorLen = reader.ReadInt32();
        _actor.Deserialize(reader.ReadBytes(actorLen));
        int criticLen = reader.ReadInt32();
        _critic.Deserialize(reader.ReadBytes(criticLen));
    }

    /// <summary>
    /// Executes GetParameters for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, GetParameters performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        var actorParams = _actor.GetParameters();
        var criticParams = _critic.GetParameters();
        var combined = new Vector<T>(actorParams.Length + criticParams.Length);
        
        for (int i = 0; i < actorParams.Length; i++)
            combined[i] = actorParams[i];
            
        for (int i = 0; i < criticParams.Length; i++)
            combined[actorParams.Length + i] = criticParams[i];
            
        return combined;
    }

    /// <summary>
    /// Executes SetParameters for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, SetParameters performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int actorCount = _actor.ParameterCount;
        _actor.SetParameters(parameters.Slice(0, actorCount));
        _critic.SetParameters(parameters.Slice(actorCount, _critic.ParameterCount));
    }

    #endregion

    #region Model Metadata

    /// <summary>
    /// Executes GetModelMetadata for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "FinancialPPO" },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <summary>
    /// Executes Clone for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, Clone performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new FinancialPPOAgent<T>(_actorArchitecture, _criticArchitecture, TradingOptions);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <summary>
    /// Executes ComputeGradients for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _actor.ComputeGradients(Tensor<T>.FromVector(input), Tensor<T>.FromVector(target), lossFunction);
    }

    /// <summary>
    /// Executes ApplyGradients for the FinancialPPOAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialPPOAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the FinancialPPOAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _actor.ApplyGradients(gradients, learningRate);
    }

    #endregion
}
