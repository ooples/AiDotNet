using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using System.IO;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Unified FinRL-style agent that can switch between multiple RL algorithms.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FinRLAgent<T> : TradingAgentBase<T>
{
    #region Fields

    private readonly TradingAgentBase<T> _innerAgent;
    private readonly FinRLAlgorithm _algorithm;
    private readonly NeuralNetworkArchitecture<T> _primaryArchitecture;
    private readonly NeuralNetworkArchitecture<T>? _secondaryArchitecture;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int FeatureCount => TradingOptions.StateSize;

    /// <inheritdoc/>
    public override int ParameterCount => _innerAgent.ParameterCount;

    /// <summary>
    /// Gets the RL algorithm being used.
    /// </summary>
    public FinRLAlgorithm Algorithm => _algorithm;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FinRLAgent class.
    /// </summary>
    /// <param name="primaryArchitecture">Primary architecture (Q-network or actor).</param>
    /// <param name="options">Configuration options for the trading agent.</param>
    /// <param name="algorithm">The RL algorithm to use.</param>
    /// <param name="secondaryArchitecture">Secondary architecture (critic), required for actor-critic algorithms.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, FinRLAgent sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinRLAgent(
        NeuralNetworkArchitecture<T> primaryArchitecture,
        TradingAgentOptions<T> options,
        FinRLAlgorithm algorithm = FinRLAlgorithm.PPO,
        NeuralNetworkArchitecture<T>? secondaryArchitecture = null)
        : base(options)
    {
        _algorithm = algorithm;
        _primaryArchitecture = primaryArchitecture ?? throw new ArgumentNullException(nameof(primaryArchitecture));
        _secondaryArchitecture = secondaryArchitecture;
        _innerAgent = CreateInnerAgent(primaryArchitecture, secondaryArchitecture, options, algorithm);
    }

    /// <summary>
    /// Creates the concrete agent for the selected algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> FinRL is a wrapper, so it builds the real agent
    /// (DQN, PPO, A2C, or SAC) based on your chosen algorithm.</para>
    /// </remarks>
    private static TradingAgentBase<T> CreateInnerAgent(
        NeuralNetworkArchitecture<T> primaryArchitecture,
        NeuralNetworkArchitecture<T>? secondaryArchitecture,
        TradingAgentOptions<T> options,
        FinRLAlgorithm algorithm)
    {
        return algorithm switch
        {
            FinRLAlgorithm.DQN => new FinancialDQNAgent<T>(primaryArchitecture, options),
            FinRLAlgorithm.PPO => new FinancialPPOAgent<T>(primaryArchitecture, RequireSecondary(secondaryArchitecture, algorithm), options),
            FinRLAlgorithm.A2C => new FinancialA2CAgent<T>(primaryArchitecture, RequireSecondary(secondaryArchitecture, algorithm), options),
            FinRLAlgorithm.SAC => new FinancialSACAgent<T>(primaryArchitecture, RequireSecondary(secondaryArchitecture, algorithm), options),
            _ => throw new ArgumentException($"Unknown algorithm: {algorithm}")
        };
    }

    /// <summary>
    /// Ensures a secondary (critic) architecture is provided when required.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Actor-critic algorithms need two networks:
    /// one for actions (actor) and one for value estimates (critic).</para>
    /// </remarks>
    private static NeuralNetworkArchitecture<T> RequireSecondary(NeuralNetworkArchitecture<T>? secondaryArchitecture, FinRLAlgorithm algorithm)
    {
        if (secondaryArchitecture is null)
            throw new ArgumentNullException(nameof(secondaryArchitecture), $"{algorithm} requires a secondary (critic) architecture.");

        return secondaryArchitecture;
    }

    #endregion

    #region Action Selection

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, SelectAction performs a supporting step in the workflow. It keeps the FinRLAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        return _innerAgent.SelectAction(state, training);
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, Train performs a training step. This updates the FinRLAgent architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override T Train()
    {
        return _innerAgent.Train();
    }

    #endregion

    #region Base Implementation

    /// <summary>
    /// Executes LoadModel for the FinRLAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, LoadModel performs a supporting step in the workflow. It keeps the FinRLAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
    }

    /// <summary>
    /// Executes SaveModel for the FinRLAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, SaveModel performs a supporting step in the workflow. It keeps the FinRLAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
    }

    /// <summary>
    /// Executes StoreExperience for the FinRLAgent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, StoreExperience performs a supporting step in the workflow. It keeps the FinRLAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _innerAgent.StoreExperience(state, action, reward, nextState, done);
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, Serialize saves or restores model-specific settings. This lets the FinRLAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write((int)_algorithm);
        var innerData = _innerAgent.Serialize();
        writer.Write(innerData.Length);
        writer.Write(innerData);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, Deserialize saves or restores model-specific settings. This lets the FinRLAgent architecture be reused later.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        var algorithm = (FinRLAlgorithm)reader.ReadInt32();
        if (algorithm != _algorithm)
        {
            throw new InvalidOperationException($"Cannot deserialize {algorithm} data into {_algorithm} agent.");
        }

        int innerLength = reader.ReadInt32();
        var innerData = reader.ReadBytes(innerLength);
        _innerAgent.Deserialize(innerData);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, GetParameters performs a supporting step in the workflow. It keeps the FinRLAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        return _innerAgent.GetParameters();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, SetParameters performs a supporting step in the workflow. It keeps the FinRLAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        _innerAgent.SetParameters(parameters);
    }

    #endregion

    #region Model Metadata

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, GetModelMetadata performs a supporting step in the workflow. It keeps the FinRLAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var innerMetadata = _innerAgent.GetModelMetadata();
        innerMetadata.AdditionalInfo["WrapperType"] = "FinRL";
        innerMetadata.AdditionalInfo["Algorithm"] = _algorithm.ToString();
        return innerMetadata;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, Clone performs a supporting step in the workflow. It keeps the FinRLAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new FinRLAgent<T>(_primaryArchitecture, TradingOptions, _algorithm, _secondaryArchitecture);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, ComputeGradients performs a supporting step in the workflow. It keeps the FinRLAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _innerAgent.ComputeGradients(input, target, lossFunction);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinRLAgent model, ApplyGradients performs a supporting step in the workflow. It keeps the FinRLAgent architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _innerAgent.ApplyGradients(gradients, learningRate);
    }

    #endregion
}

/// <summary>
/// RL algorithms supported by FinRLAgent.
/// </summary>
public enum FinRLAlgorithm
{
    /// <summary>
    /// Deep Q-Network - good for discrete action spaces.
    /// </summary>
    DQN,

    /// <summary>
    /// Proximal Policy Optimization - robust and stable training.
    /// </summary>
    PPO,

    /// <summary>
    /// Advantage Actor-Critic - fast synchronous training.
    /// </summary>
    A2C,

    /// <summary>
    /// Soft Actor-Critic - best for continuous actions with entropy regularization.
    /// </summary>
    SAC
}
