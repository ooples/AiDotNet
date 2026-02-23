using AiDotNet.Finance.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.LossFunctions;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Specialized market making agent using reinforcement learning for optimal quoting.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MarketMakingAgent<T> : TradingAgentBase<T>
{
    #region Fields

    private readonly NeuralNetwork<T> _policyNetwork;
    private readonly MarketMakingOptions<T> _mmOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _mmOptions;

    private readonly ReplayBuffer<T> ReplayBuffer;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    /// <inheritdoc/>
    public override int ParameterCount => _policyNetwork.ParameterCount;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the MarketMakingAgent class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, MarketMakingAgent sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public MarketMakingAgent(NeuralNetworkArchitecture<T> architecture, MarketMakingOptions<T> options)
        : base(options)
    {
        _mmOptions = options;
        EnsureMarketMakingLayers(architecture, options.StateSize, options.ActionSize);
        _policyNetwork = new NeuralNetwork<T>(architecture, TradingOptions.LossFunction ?? new MeanSquaredErrorLoss<T>());
        ReplayBuffer = new ReplayBuffer<T>(options.ReplayBufferSize, options.Seed);
    }

    /// <summary>
    /// Validates the architecture and creates default market-making layers if needed.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This checks that the network matches the input and output sizes
    /// for market-making and fills in a sensible default if no layers were provided.</para>
    /// </remarks>
    private static void EnsureMarketMakingLayers(NeuralNetworkArchitecture<T> architecture, int stateSize, int actionSize)
    {
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));

        if (architecture.CalculatedInputSize != stateSize)
            throw new ArgumentException($"Architecture input size {architecture.CalculatedInputSize} does not match expected {stateSize}.", nameof(architecture));

        if (architecture.OutputSize != actionSize)
            throw new ArgumentException($"Architecture output size {architecture.OutputSize} does not match expected {actionSize}.", nameof(architecture));

        if (architecture.Layers.Count == 0)
        {
            architecture.Layers.AddRange(LayerHelper<T>.CreateDefaultMarketMakingLayers(
                architecture,
                stateSize,
                actionSize));
        }
    }

    #endregion

    #region Action Selection

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, SelectAction performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var action = _policyNetwork.Predict(Tensor<T>.FromVector(state)).ToVector();
        
        if (training)
        {
            // Add exploration noise
            var noise = new Vector<T>(action.Length);
            for (int i = 0; i < noise.Length; i++)
                noise[i] = NumOps.FromDouble(RandomHelper.CreateSecureRandom().NextDouble() * 0.05);
            
            return action.Add(noise);
        }

        return action;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, Train performs a training step. This updates the MarketMakingAgent architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override T Train()
    {
        if (ReplayBuffer.Count < TradingOptions.BatchSize) return NumOps.Zero;

        var batch = ReplayBuffer.Sample(TradingOptions.BatchSize);
        T totalLoss = NumOps.Zero;

        foreach (var exp in batch)
        {
            T loss = TradingOptions.LossFunction!.CalculateLoss(
                _policyNetwork.Predict(Tensor<T>.FromVector(exp.State)).ToVector(),
                exp.Action);

            _policyNetwork.Train(Tensor<T>.FromVector(exp.State), Tensor<T>.FromVector(exp.Action));
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(TradingOptions.BatchSize));
    }

    #endregion

    #region Base Implementation

    /// <summary>
    /// Executes LoadModel for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, LoadModel performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, SaveModel performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Executes StoreExperience for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, StoreExperience performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
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
    /// Executes Serialize for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, Serialize saves or restores model-specific settings. This lets the MarketMakingAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override byte[] Serialize() => _policyNetwork.Serialize();

    /// <summary>
    /// Executes Deserialize for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, Deserialize saves or restores model-specific settings. This lets the MarketMakingAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data) => _policyNetwork.Deserialize(data);

    /// <summary>
    /// Executes GetParameters for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, GetParameters performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters() => _policyNetwork.GetParameters();

    /// <summary>
    /// Executes SetParameters for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, SetParameters performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters) => _policyNetwork.SetParameters(parameters);

    #endregion

    #region Model Metadata

    /// <summary>
    /// Executes GetModelMetadata for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ReinforcementLearning,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AgentType", "MarketMaking" },
                { "MaxInventory", _mmOptions.MaxInventory },
                { "BaseSpread", _mmOptions.BaseSpread },
                { "ParameterCount", ParameterCount }
            }
        };
    }

    /// <summary>
    /// Executes Clone for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, Clone performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new MarketMakingAgent<T>(_policyNetwork.GetArchitecture(), _mmOptions);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <summary>
    /// Executes ComputeGradients for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _policyNetwork.ComputeGradients(Tensor<T>.FromVector(input), Tensor<T>.FromVector(target), lossFunction);
    }

    /// <summary>
    /// Executes ApplyGradients for the MarketMakingAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MarketMakingAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the MarketMakingAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _policyNetwork.ApplyGradients(gradients, learningRate);
    }

    #endregion
}
