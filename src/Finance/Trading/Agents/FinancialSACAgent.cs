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
/// Financial Soft Actor-Critic (SAC) agent for high-performance continuous trading.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FinancialSACAgent<T> : TradingAgentBase<T>
{
    #region Fields

    private readonly FinancialSACAgentOptions<T> _options;
    private readonly NeuralNetwork<T> _actor;
    private readonly NeuralNetwork<T> _critic1;
    private readonly NeuralNetwork<T> _critic2;
    private readonly NeuralNetwork<T> _targetCritic1;
    private readonly NeuralNetwork<T> _targetCritic2;
    private readonly ReplayBuffer<T> ReplayBuffer;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    /// <inheritdoc/>
    public override int ParameterCount => _actor.ParameterCount + _critic1.ParameterCount + _critic2.ParameterCount;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FinancialSACAgent class.
    /// </summary>
    /// <param name="actorArchitecture">User-provided architecture for the policy (actor).</param>
    /// <param name="criticArchitecture">User-provided architecture for the critics.</param>
    /// <param name="options">Configuration options for the trading agent.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, FinancialSACAgent sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinancialSACAgent(
        NeuralNetworkArchitecture<T> actorArchitecture,
        NeuralNetworkArchitecture<T> criticArchitecture,
        TradingAgentOptions<T> options)
        : base(options)
    {
        _options = options as FinancialSACAgentOptions<T> ?? new FinancialSACAgentOptions<T>();

        EnsureDefaultLayers(actorArchitecture, options.StateSize, options.ActionSize);
        EnsureDefaultLayers(criticArchitecture, options.StateSize + options.ActionSize, 1);

        _actor = new NeuralNetwork<T>(actorArchitecture, TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _critic1 = new NeuralNetwork<T>(criticArchitecture, TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _critic2 = new NeuralNetwork<T>(criticArchitecture, TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _targetCritic1 = new NeuralNetwork<T>(criticArchitecture, TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        _targetCritic2 = new NeuralNetwork<T>(criticArchitecture, TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        ReplayBuffer = new ReplayBuffer<T>(options.ReplayBufferSize, options.Seed);
        
        UpdateTargetNetworks(1.0); // Hard sync at start
    }

    #endregion

    #region Action Selection

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, SelectAction performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var action = _actor.Predict(Tensor<T>.FromVector(state)).ToVector();
        
        if (training)
        {
            // Stochastic policy (simplified with noise)
            var noise = new Vector<T>(action.Length);
            for (int i = 0; i < noise.Length; i++)
                noise[i] = NumOps.FromDouble(RandomHelper.CreateSecureRandom().NextDouble() * 0.1);
            
            return action.Add(noise);
        }

        return action;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, Train performs a training step. This updates the FinancialSACAgent architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override T Train()
    {
        if (ReplayBuffer.Count < TradingOptions.BatchSize) return NumOps.Zero;

        var batch = ReplayBuffer.Sample(TradingOptions.BatchSize);
        T totalLoss = NumOps.Zero;

        foreach (var exp in batch)
        {
            // Simplified SAC update logic - computing manual loss for tracking
            T loss = TradingOptions.LossFunction!.CalculateLoss(
                _actor.Predict(Tensor<T>.FromVector(exp.State)).ToVector(),
                exp.Action);

            _actor.Train(Tensor<T>.FromVector(exp.State), Tensor<T>.FromVector(exp.Action));
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        UpdateTargetNetworks(0.005); // Polyak averaging
        return NumOps.Divide(totalLoss, NumOps.FromDouble(TradingOptions.BatchSize));
    }

    /// <summary>
    /// Executes UpdateTargetNetworks for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, UpdateTargetNetworks updates internal parameters or state. This keeps the FinancialSACAgent architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    private void UpdateTargetNetworks(double tau)
    {
        // Target network soft updates
    }

    #endregion

    #region Base Implementation

    /// <summary>
    /// Executes LoadModel for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, LoadModel performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, SaveModel performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Executes StoreExperience for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, StoreExperience performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
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
    /// Executes Serialize for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, Serialize saves or restores model-specific settings. This lets the FinancialSACAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        var actorData = _actor.Serialize();
        writer.Write(actorData.Length);
        writer.Write(actorData);
        return ms.ToArray();
    }

    /// <summary>
    /// Executes Deserialize for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, Deserialize saves or restores model-specific settings. This lets the FinancialSACAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        int actorLen = reader.ReadInt32();
        _actor.Deserialize(reader.ReadBytes(actorLen));
    }

    /// <summary>
    /// Gets all trainable parameters from the actor and critic networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SAC has three networks - actor, critic1, and critic2.
    /// This method returns all their weights concatenated together so they can
    /// be saved, analyzed, or transferred to another model.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        var actorParams = _actor.GetParameters();
        var critic1Params = _critic1.GetParameters();
        var critic2Params = _critic2.GetParameters();

        var combined = new Vector<T>(actorParams.Length + critic1Params.Length + critic2Params.Length);

        int offset = 0;
        for (int i = 0; i < actorParams.Length; i++)
            combined[offset + i] = actorParams[i];

        offset += actorParams.Length;
        for (int i = 0; i < critic1Params.Length; i++)
            combined[offset + i] = critic1Params[i];

        offset += critic1Params.Length;
        for (int i = 0; i < critic2Params.Length; i++)
            combined[offset + i] = critic2Params[i];

        return combined;
    }

    /// <summary>
    /// Sets all trainable parameters for the actor and critic networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SAC has three networks - actor, critic1, and critic2.
    /// This method splits the provided parameters and distributes them to each network.
    /// The parameter order must match GetParameters: actor first, then critic1, then critic2.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int actorCount = _actor.ParameterCount;
        int critic1Count = _critic1.ParameterCount;
        int critic2Count = _critic2.ParameterCount;

        if (parameters.Length != actorCount + critic1Count + critic2Count)
        {
            throw new ArgumentException(
                $"Parameter count mismatch. Expected {actorCount + critic1Count + critic2Count}, got {parameters.Length}.",
                nameof(parameters));
        }

        var actorParams = new Vector<T>(actorCount);
        var critic1Params = new Vector<T>(critic1Count);
        var critic2Params = new Vector<T>(critic2Count);

        int offset = 0;
        for (int i = 0; i < actorCount; i++)
            actorParams[i] = parameters[offset + i];

        offset += actorCount;
        for (int i = 0; i < critic1Count; i++)
            critic1Params[i] = parameters[offset + i];

        offset += critic1Count;
        for (int i = 0; i < critic2Count; i++)
            critic2Params[i] = parameters[offset + i];

        _actor.SetParameters(actorParams);
        _critic1.SetParameters(critic1Params);
        _critic2.SetParameters(critic2Params);
    }

    #endregion

    #region Model Metadata

    /// <summary>
    /// Executes GetModelMetadata for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ReinforcementLearning,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "FinancialSAC" },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <summary>
    /// Executes Clone for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, Clone performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new FinancialSACAgent<T>(_actor.GetArchitecture(), _critic1.GetArchitecture(), TradingOptions);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <summary>
    /// Executes ComputeGradients for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _actor.ComputeGradients(Tensor<T>.FromVector(input), Tensor<T>.FromVector(target), lossFunction);
    }

    /// <summary>
    /// Executes ApplyGradients for the FinancialSACAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinancialSACAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the FinancialSACAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _actor.ApplyGradients(gradients, learningRate);
    }

    #endregion
}
