using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.ReinforcementLearning.Memory;

namespace AiDotNet.ReinforcementLearning.Models;

/// <summary>
/// A multi-agent transformer-based reinforcement learning model for financial market prediction and trading.
/// </summary>
/// <remarks>
/// The Multi-Agent Transformer model uses transformer architecture to model complex interactions between
/// multiple market participants. This approach is especially effective for capturing emergent behaviors
/// in financial markets that arise from the collective actions of various agents with different strategies
/// and information levels.
/// 
/// For beginners: This model views the market as a conversation between different types of traders
/// (like retail investors, institutions, market makers). It uses powerful neural networks called transformers
/// (similar to those in ChatGPT) to understand how these different traders interact and influence market movements,
/// allowing it to make predictions about future market behavior.
/// 
/// This model can be useful for:
/// - Stock market prediction considering multiple market participants
/// - Strategy development that accounts for market microstructure
/// - Risk management that considers various market regimes
/// - Understanding how different types of traders impact price movements
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
public class MultiAgentTransformerModel<T> : ReinforcementLearningModelBase<T>
{
    private readonly MultiAgentTransformerOptions _options = default!;
    private MultiAgentTransformerAgent<T>? _agent;
    private readonly SequentialReplayBuffer<Tensor<T>, Vector<T>, T> _replayBuffer = default!;
    private bool _isInitialized = false;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiAgentTransformerModel{T}"/> class.
    /// </summary>
    /// <param name="options">The configuration options for the model.</param>
    public MultiAgentTransformerModel(MultiAgentTransformerOptions options) : base(options)
    {
        _options = options ?? new MultiAgentTransformerOptions();
        IsContinuous = true; // Financial markets typically use continuous action spaces
        
        // Initialize the replay buffer to store sequences of market interactions
        _replayBuffer = new SequentialReplayBuffer<Tensor<T>, Vector<T>, T>(
            capacity: 10000,
            maxTrajectoryLength: _options.SequenceLength);
    }
    
    /// <summary>
    /// Initializes the agent that will interact with the environment.
    /// </summary>
    protected override void InitializeAgent()
    {
        // Create the multi-agent transformer agent
        _agent = new MultiAgentTransformerAgent<T>(
            numAgents: _options.NumAgents,
            stateDimension: _options.StateDimension,
            actionDimension: _options.ActionDimension,
            hiddenDimension: _options.HiddenDimension,
            numHeads: _options.NumHeads,
            numLayers: _options.NumLayers,
            sequenceLength: _options.SequenceLength,
            posEncodingType: _options.PositionalEncodingType,
            communicationMode: _options.CommunicationMode,
            useCentralizedTraining: _options.UseCentralizedTraining,
            learningRate: _options.TransformerLearningRate,
            gamma: _options.Gamma,
            entropyCoef: _options.EntropyCoefficient,
            useSelfPlay: _options.UseSelfPlay,
            riskAversion: _options.RiskAversionParameter,
            useCausalMask: _options.UseCausalMask,
            modelMarketImpact: _options.ModelMarketImpact);
        
        _isInitialized = true;
    }
    
    /// <summary>
    /// Gets the agent that interacts with the environment.
    /// </summary>
    /// <returns>The agent as an IAgent interface.</returns>
    protected override IAgent<Tensor<T>, Vector<T>, T> GetAgent()
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model is not initialized. Please initialize it first.");
        
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
            
        return _agent;
    }
    
    /// <summary>
    /// Selects an action based on the current state for a specific agent or for all agents.
    /// </summary>
    /// <param name="state">The current market state.</param>
    /// <param name="isTraining">Indicates whether the model is in training mode.</param>
    /// <returns>The selected action vector.</returns>
    /// <remarks>
    /// For beginners: This method decides what trading action to take (like buy, sell, hold)
    /// based on the current market conditions. It uses the transformer network to analyze
    /// how different market participants might react to the current situation.
    /// </remarks>
    public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = false)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model is not initialized. Please initialize it first.");
        
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
        
        return _agent.SelectAction(state, isTraining);
    }
    
    /// <summary>
    /// Selects actions for all agents based on the current state.
    /// </summary>
    /// <param name="state">The current market state.</param>
    /// <param name="isTraining">Indicates whether the model is in training mode.</param>
    /// <returns>A list of action vectors, one for each agent.</returns>
    /// <remarks>
    /// For beginners: Unlike the single-agent version, this method returns actions for all
    /// simulated market participants, showing what each type of trader might do in the current market.
    /// </remarks>
    public List<Vector<T>> SelectActionsForAllAgents(Tensor<T> state, bool isTraining = false)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model is not initialized. Please initialize it first.");
        
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
        
        return _agent.SelectActionsForAllAgents(state, isTraining);
    }
    
    /// <summary>
    /// Updates the model based on the observed transition.
    /// </summary>
    /// <param name="state">The current market state.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The next market state.</param>
    /// <param name="done">Indicates whether the episode is completed.</param>
    /// <returns>The loss value from the update.</returns>
    /// <remarks>
    /// For beginners: This method helps the model learn from experience by updating its understanding
    /// of how different market actions lead to different outcomes. Over time, this helps it develop
    /// better trading strategies that maximize profits.
    /// </remarks>
    public override T Update(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model is not initialized. Please initialize it first.");
        
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
        
        // Add the experience to the replay buffer
        _replayBuffer.Add(state, action, reward, nextState, done);
        
        // Only update the model if we have enough data
        if (_replayBuffer.Size < _options.SequenceLength)
            return NumOps.Zero;
        
        // For single-step updates, we just return zero loss
        // The actual training happens in TrainOnBatch
        return NumOps.Zero;
    }
    
    /// <summary>
    /// Updates the model with multi-agent transitions where each agent has its own action and reward.
    /// </summary>
    /// <param name="state">The shared market state.</param>
    /// <param name="actions">The actions taken by each agent.</param>
    /// <param name="rewards">The rewards received by each agent.</param>
    /// <param name="nextState">The next market state.</param>
    /// <param name="done">Indicates whether the episode is completed.</param>
    /// <returns>The average loss value across all agents.</returns>
    /// <remarks>
    /// For beginners: This specialized method is used when simulating multiple traders
    /// where each has their own strategy and outcomes. It helps the model understand
    /// how different trading approaches perform in the same market conditions.
    /// </remarks>
    public T UpdateMultiAgent(Tensor<T> state, List<Vector<T>> actions, List<T> rewards, Tensor<T> nextState, bool done)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model is not initialized. Please initialize it first.");
        
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
        
        // We don't use the replay buffer for multi-agent updates since it's designed for single-agent
        // Instead, we directly update the multi-agent transformer
        return _agent.UpdateMultiAgent(state, actions, rewards, nextState, done);
    }
    
    /// <summary>
    /// Gets the attention weights between agents, showing how each agent influences the others.
    /// </summary>
    /// <param name="state">The current market state.</param>
    /// <returns>A tensor containing the attention weights between agents.</returns>
    /// <remarks>
    /// For beginners: This method reveals how much each type of market participant
    /// (like retail traders, institutions, etc.) is influencing the others in the current
    /// market conditions. High attention weights suggest strong influence.
    /// </remarks>
    public Tensor<T> GetAgentInteractionAttention(Tensor<T> state)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model is not initialized. Please initialize it first.");
        
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
        
        return _agent.GetAgentInteractionAttention(state);
    }
    
    /// <summary>
    /// Predicts potential future market states based on current conditions and hypothetical actions.
    /// </summary>
    /// <param name="currentState">The current market state.</param>
    /// <param name="numSteps">The number of future steps to predict.</param>
    /// <returns>A list of predicted future market states.</returns>
    /// <remarks>
    /// For beginners: This provides a forecast of how the market might evolve over time,
    /// considering how different traders would react to changing conditions. It's useful
    /// for scenario analysis and planning ahead in trading strategies.
    /// </remarks>
    public List<Tensor<T>> PredictFutureStates(Tensor<T> currentState, int numSteps = 5)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model is not initialized. Please initialize it first.");
        
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
        
        return _agent.PredictFutureStates(currentState, numSteps);
    }
    
    /// <summary>
    /// Analyzes the risk profile of different potential actions in the current market state.
    /// </summary>
    /// <param name="state">The current market state.</param>
    /// <returns>A dictionary mapping actions to their risk-adjusted expected returns.</returns>
    /// <remarks>
    /// For beginners: This method evaluates different trading strategies for their balance
    /// of risk and reward, helping identify the safest or most optimal approaches given
    /// current market conditions.
    /// </remarks>
    public Dictionary<string, T> AnalyzeActionRiskProfile(Tensor<T> state)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model is not initialized. Please initialize it first.");
        
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
        
        return _agent.AnalyzeActionRiskProfile(state);
    }
    
    /// <summary>
    /// Identifies the current market regime based on agent behavior patterns.
    /// </summary>
    /// <param name="state">The current market state.</param>
    /// <returns>A string indicating the detected market regime (e.g., "Bull", "Bear", "Sideways", "Volatile").</returns>
    /// <remarks>
    /// For beginners: This method attempts to categorize the current market environment
    /// (like bullish, bearish, volatile, etc.) based on the patterns of interaction
    /// between different market participants. Different strategies work better in
    /// different market regimes, so this information is valuable for adaptation.
    /// </remarks>
    public string DetectMarketRegime(Tensor<T> state)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model is not initialized. Please initialize it first.");
        
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
        
        return _agent.DetectMarketRegime(state);
    }

    /// <summary>
    /// Saves the model to the specified path.
    /// </summary>
    /// <param name="path">The path to save the model to.</param>
    public override void SaveModel(string path)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Model is not initialized. Please initialize it first.");
        
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
        
        _agent.SaveModel(path);
    }

    /// <summary>
    /// Loads the model from the specified path.
    /// </summary>
    /// <param name="path">The path to load the model from.</param>
    public override void LoadModel(string path)
    {
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
        
        _agent.LoadModel(path);
        _isInitialized = true;
    }

    /// <summary>
    /// Trains the model on a batch of experiences.
    /// </summary>
    protected override T TrainOnBatch(Tensor<T> states, Tensor<T> actions, Vector<T> rewards, Tensor<T> nextStates, Vector<T> dones)
    {
        // Convert Vector<T> actions to List<Vector<T>> for multi-agent
        var actionsList = new List<Vector<T>>();
        var rewardsList = new List<T>();
        
        // For now, treat single batch as single agent
        // In a full implementation, this would handle multiple agents
        var actionVector = new Vector<T>(actions.Shape[1]);
        for (int i = 0; i < actions.Shape[1]; i++)
        {
            actionVector[i] = actions[0, i];
        }
        actionsList.Add(actionVector);
        rewardsList.Add(rewards[0]);
        
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
        
        // Extract first state and next state from batch
        var state = new Tensor<T>(new[] { 1, states.Shape[1] });
        var nextState = new Tensor<T>(new[] { 1, nextStates.Shape[1] });
        
        for (int i = 0; i < states.Shape[1]; i++)
        {
            state[0, i] = states[0, i];
            nextState[0, i] = nextStates[0, i];
        }
        
        return _agent.UpdateMultiAgent(state, actionsList, rewardsList, nextState, NumOps.GreaterThan(dones[0], NumOps.Zero));
    }

    /// <summary>
    /// Gets the model parameters as a vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
            
        return _agent.GetParameters();
    }

    /// <summary>
    /// Sets the model parameters from a vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        if (_agent == null)
            throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
            
        _agent.SetParameters(parameters);
    }

    /// <summary>
    /// Saves the model to a stream.
    /// </summary>
    public override void Save(Stream stream)
    {
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
        {
            writer.Write("MultiAgentTransformer");
            writer.Write(1); // Version
            // Save agent parameters
            if (_agent != null)
            {
                var parameters = _agent.GetParameters();
                writer.Write(parameters.Length);
                for (int i = 0; i < parameters.Length; i++)
                {
                    writer.Write(Convert.ToDouble(parameters[i]));
                }
                
                // Save agent configuration
                writer.Write(_options.NumAgents);
                writer.Write(_options.StateDimension);
                writer.Write(_options.ActionDimension);
                writer.Write(_options.HiddenDimension);
                writer.Write(_options.NumHeads);
                writer.Write(_options.NumLayers);
                writer.Write(_options.SequenceLength);
                writer.Write((int)_options.PositionalEncodingType);
                writer.Write(_options.CommunicationMode);
                writer.Write(_options.UseCentralizedTraining);
                writer.Write(_options.TransformerLearningRate);
                writer.Write(_options.Gamma);
                writer.Write(_options.EntropyCoefficient);
                writer.Write(_options.UseSelfPlay);
                writer.Write(_options.RiskAversionParameter);
                writer.Write(_options.UseCausalMask);
                writer.Write(_options.ModelMarketImpact);
            }
            else
            {
                writer.Write(0); // No parameters
            }
        }
    }

    /// <summary>
    /// Loads the model from a stream.
    /// </summary>
    public override void Load(Stream stream)
    {
        using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, true))
        {
            var modelType = reader.ReadString();
            if (modelType != "MultiAgentTransformer")
                throw new InvalidOperationException($"Invalid model type: {modelType}");
            
            var version = reader.ReadInt32();
            if (version != 1)
                throw new InvalidOperationException($"Unsupported version: {version}");
            
            // Load agent parameters
            int paramCount = reader.ReadInt32();
            if (paramCount > 0)
            {
                var parameters = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++)
                {
                    parameters[i] = NumOps.FromDouble(reader.ReadDouble());
                }
                
                // Load agent configuration
                int numAgents = reader.ReadInt32();
                int stateDim = reader.ReadInt32();
                int actionDim = reader.ReadInt32();
                int hiddenDim = reader.ReadInt32();
                int numHeads = reader.ReadInt32();
                int numLayers = reader.ReadInt32();
                int seqLength = reader.ReadInt32();
                var posEncType = (PositionalEncodingType)reader.ReadInt32();
                int commMode = reader.ReadInt32();
                bool centralized = reader.ReadBoolean();
                double lr = reader.ReadDouble();
                double gamma = reader.ReadDouble();
                double entropy = reader.ReadDouble();
                bool selfPlay = reader.ReadBoolean();
                double risk = reader.ReadDouble();
                bool causal = reader.ReadBoolean();
                bool marketImpact = reader.ReadBoolean();
                
                // Verify configuration matches
                if (numAgents != _options.NumAgents || stateDim != _options.StateDimension || 
                    actionDim != _options.ActionDimension)
                {
                    throw new InvalidOperationException("Loaded model configuration does not match current options");
                }
                
                // Ensure agent is initialized
                if (_agent == null)
                {
                    InitializeAgent();
                }
                
                // Set parameters to the agent
                if (_agent == null)
                    throw new InvalidOperationException("Agent is null. Please ensure InitializeAgent() was called successfully.");
                
                _agent.SetParameters(parameters);
            }
        }
        _isInitialized = true;
    }

    /// <summary>
    /// Creates a new instance of the model.
    /// </summary>
    public override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MultiAgentTransformerModel<T>(_options);
    }
}