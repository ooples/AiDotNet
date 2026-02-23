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
/// Financial Deep Q-Network (DQN) agent for discrete action trading.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FinancialDQNAgent<T> : TradingAgentBase<T>
{
    #region Fields

    private readonly FinancialDQNAgentOptions<T> _options;
    private readonly NeuralNetwork<T> _qNetwork;
    private readonly NeuralNetwork<T> _targetNetwork;
    private readonly ReplayBuffer<T> ReplayBuffer;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    /// <inheritdoc/>
    public override int ParameterCount => _qNetwork.ParameterCount;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FinancialDQNAgent class.
    /// </summary>
    /// <param name="architecture">The user-provided architecture for the Q-network.</param>
    /// <param name="options">Configuration options for the trading agent.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, FinancialDQNAgent sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinancialDQNAgent(NeuralNetworkArchitecture<T> architecture, TradingAgentOptions<T> options)
        : base(options)
    {
        _options = options as FinancialDQNAgentOptions<T> ?? new FinancialDQNAgentOptions<T>();

        EnsureDefaultLayers(architecture, options.StateSize, options.ActionSize);

        _qNetwork = new NeuralNetwork<T>(architecture, TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _targetNetwork = new NeuralNetwork<T>(architecture, TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        ReplayBuffer = new ReplayBuffer<T>(options.ReplayBufferSize, options.Seed);
        UpdateTargetNetwork();
    }

    #endregion

    #region Action Selection

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, SelectAction performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        if (training && RandomHelper.CreateSecureRandom().NextDouble() < TradingOptions.EpsilonStart)
        {
            var action = new Vector<T>(TradingOptions.ActionSize);
            int randomAction = RandomHelper.CreateSecureRandom().Next(TradingOptions.ActionSize);
            action[randomAction] = NumOps.One;
            return action;
        }

        var qValues = _qNetwork.Predict(Tensor<T>.FromVector(state));
        int bestAction = 0;
        T maxQ = qValues.Data.Span[0];

        for (int i = 1; i < TradingOptions.ActionSize; i++)
        {
            if (NumOps.ToDouble(qValues.Data.Span[i]) > NumOps.ToDouble(maxQ))
            {
                maxQ = qValues.Data.Span[i];
                bestAction = i;
            }
        }

        var result = new Vector<T>(TradingOptions.ActionSize);
        result[bestAction] = NumOps.One;
        return result;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, Train performs a training step. This updates the FinancialDQNAgent architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override T Train()
    {
        if (ReplayBuffer.Count < TradingOptions.BatchSize)
            return NumOps.Zero;

        var batch = ReplayBuffer.Sample(TradingOptions.BatchSize);
        T totalLoss = NumOps.Zero;

        foreach (var exp in batch)
        {
            var currentQ = _qNetwork.Predict(Tensor<T>.FromVector(exp.State));
            var nextQ = _targetNetwork.Predict(Tensor<T>.FromVector(exp.NextState));

            int actionIdx = GetActionIndex(exp.Action);
            T maxNextQ = GetMaxQ(nextQ);
            
            T target = exp.Done 
                ? exp.Reward 
                : NumOps.Add(exp.Reward, NumOps.Multiply(NumOps.FromDouble(Convert.ToDouble(TradingOptions.DiscountFactor)), maxNextQ));

            var expectedOutput = currentQ.ToVector().Clone();
            expectedOutput[actionIdx] = target;

            T loss = TradingOptions.LossFunction!.CalculateLoss(currentQ.ToVector(), expectedOutput);
            totalLoss = NumOps.Add(totalLoss, loss);

            _qNetwork.Train(Tensor<T>.FromVector(exp.State), Tensor<T>.FromVector(expectedOutput));
        }

        if (RandomHelper.CreateSecureRandom().Next(TradingOptions.TargetUpdateFrequency) == 0)
        {
            UpdateTargetNetwork();
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(TradingOptions.BatchSize));
    }

    /// <summary>
    /// Executes UpdateTargetNetwork for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, UpdateTargetNetwork updates internal parameters or state. This keeps the FinancialDQNAgent architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    private void UpdateTargetNetwork()
    {
        _targetNetwork.UpdateParameters(_qNetwork.GetParameters());
    }

    /// <summary>
    /// Executes GetActionIndex for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, GetActionIndex performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private static int GetActionIndex(Vector<T> action)
    {
        for (int i = 0; i < action.Length; i++)
        {
            if (Math.Abs(Convert.ToDouble(action[i]) - 1.0) < 1e-5)
                return i;
        }
        return 0;
    }

    /// <summary>
    /// Executes GetMaxQ for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, GetMaxQ performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private T GetMaxQ(Tensor<T> qValues)
    {
        T max = qValues.Data.Span[0];
        for (int i = 1; i < qValues.Length; i++)
        {
            if (NumOps.ToDouble(qValues.Data.Span[i]) > NumOps.ToDouble(max))
                max = qValues.Data.Span[i];
        }
        return max;
    }

    #endregion

    #region Base Implementation

    /// <summary>
    /// Executes LoadModel for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, LoadModel performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, SaveModel performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Executes StoreExperience for the FinancialDQNAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, StoreExperience performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        var experience = new Experience<T>(state, action, reward, nextState, done);
        ReplayBuffer.Add(experience);
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, Serialize saves or restores model-specific settings. This lets the FinancialDQNAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override byte[] Serialize() => _qNetwork.Serialize();

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, Deserialize saves or restores model-specific settings. This lets the FinancialDQNAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        _qNetwork.Deserialize(data);
        UpdateTargetNetwork();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, GetParameters performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters() => _qNetwork.GetParameters();

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, SetParameters performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        _qNetwork.SetParameters(parameters);
        UpdateTargetNetwork();
    }

    #endregion

    #region Model Metadata

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ReinforcementLearning,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "FinancialDQN" },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, Clone performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new FinancialDQNAgent<T>(_qNetwork.GetArchitecture(), TradingOptions);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _qNetwork.ComputeGradients(Tensor<T>.FromVector(input), Tensor<T>.FromVector(target), lossFunction);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialDQNAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the FinancialDQNAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _qNetwork.ApplyGradients(gradients, learningRate);
        UpdateTargetNetwork();
    }

    #endregion
}
