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
/// Financial Advantage Actor-Critic (A2C) agent for fast trading policy learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FinancialA2CAgent<T> : TradingAgentBase<T>
{
    #region Fields

    private readonly NeuralNetwork<T> _actor;
    private readonly NeuralNetwork<T> _critic;
    private readonly ReplayBuffer<T> ReplayBuffer;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    /// <inheritdoc/>
    public override int ParameterCount => _actor.ParameterCount + _critic.ParameterCount;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FinancialA2CAgent class.
    /// </summary>
    /// <param name="actorArchitecture">User-provided architecture for the policy (actor).</param>
    /// <param name="criticArchitecture">User-provided architecture for the value (critic).</param>
    /// <param name="options">Configuration options for the trading agent.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, FinancialA2CAgent sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinancialA2CAgent(
        NeuralNetworkArchitecture<T> actorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        TradingAgentOptions<T> options)
        : base(options)
    {
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
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, SelectAction performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
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
            if (NumOps.ToDouble(probs[i]) > NumOps.ToDouble(maxProb))
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
    /// Executes SampleAction for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, SampleAction performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, Train performs a training step. This updates the FinancialA2CAgent architecture so it learns from data.
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
            T targetV = NumOps.Add(exp.Reward, NumOps.Multiply(NumOps.FromDouble(Convert.ToDouble(TradingOptions.DiscountFactor)), vNext));
            T advantage = NumOps.Subtract(targetV, vCurrent);

            // Update Critic (MSE Loss)
            var targetVVec = new Vector<T>(new[] { targetV });
            _critic.Train(Tensor<T>.FromVector(exp.State), Tensor<T>.FromVector(targetVVec));

            // Update Actor
            T actorLoss = TradingOptions.LossFunction!.CalculateLoss(
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
    /// Executes LoadModel for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, LoadModel performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, SaveModel performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Executes StoreExperience for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, StoreExperience performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
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
    /// Executes Serialize for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, Serialize saves or restores model-specific settings. This lets the FinancialA2CAgent architecture be reused later.
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
    /// Executes Deserialize for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, Deserialize saves or restores model-specific settings. This lets the FinancialA2CAgent architecture be reused later.
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
    /// Executes GetParameters for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, GetParameters performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
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
    /// Executes SetParameters for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, SetParameters performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
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
    /// Executes GetModelMetadata for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ReinforcementLearning,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "FinancialA2C" },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <summary>
    /// Executes Clone for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, Clone performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new FinancialA2CAgent<T>(_actor.GetArchitecture(), _critic.GetArchitecture(), TradingOptions);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <summary>
    /// Executes ComputeGradients for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _actor.ComputeGradients(Tensor<T>.FromVector(input), Tensor<T>.FromVector(target), lossFunction);
    }

    /// <summary>
    /// Executes ApplyGradients for the FinancialA2CAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialA2CAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the FinancialA2CAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _actor.ApplyGradients(gradients, learningRate);
    }

    #endregion
}
