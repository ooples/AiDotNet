using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Implementation of the Decision Transformer agent for sequence modeling-based reinforcement learning.
/// </summary>
/// <typeparam name="TState">The type of state observation (typically Tensor<double>).</typeparam>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Decision Transformer reframes reinforcement learning as a sequence modeling problem,
/// using a transformer architecture to autoregressively model the sequence of states,
/// actions, and returns. It can be trained from offline data without requiring active exploration.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// The Decision Transformer works like a language model (similar to ChatGPT) but for predicting
/// actions instead of words. It looks at a sequence of past market states, the actions taken,
/// and the returns that resulted, then predicts what action to take next to achieve a target return.
/// 
/// It's particularly good for financial markets because it:
/// - Can learn from historical data without making real trades
/// - Handles complex temporal patterns well
/// - Can be given a target return to aim for
/// - Adapts to different market conditions
/// </para>
/// </remarks>
public class DecisionTransformerAgent<TState, T> : AgentBase<TState, Vector<T>, T>
    where TState : Tensor<T>
{
    private readonly DecisionTransformerOptions<T> _options = default!;
    
    // Neural network components
    private TransformerEncoderLayer<T> _transformer = default!;
    private FullyConnectedLayer<T> _stateEncoder = default!;
    private FullyConnectedLayer<T> _actionEncoder = default!;
    private FullyConnectedLayer<T> _returnEncoder = default!;
    private FullyConnectedLayer<T> _actionDecoder = default!;
    
    // Memory for storing sequences
    private ISequentialReplayBuffer<TState, Vector<T>, T> _replayBuffer = default!;
    
    // Optimizer for training
    private readonly IGradientBasedOptimizer<T, Vector<T>, T>? _optimizer;
    
    // Tracking variables - _step is now managed by base class as TotalSteps
    private int _step;
    
    // Sequence tracking for context window
    private List<TState> _stateHistory = default!;
    private List<Vector<T>> _actionHistory = default!;
    private List<T> _returnHistory = default!;
    private List<T> _targetReturnHistory = default!;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="DecisionTransformerAgent{TState, T}"/> class.
    /// </summary>
    /// <param name="options">The options for configuring the Decision Transformer.</param>
    public DecisionTransformerAgent(DecisionTransformerOptions<T> options)
        : base(options.Gamma, options.Tau, options.BatchSize, options.Seed)
    {
        _options = options;
        _step = 0;
        
        // Initialize history buffers
        _stateHistory = [];
        _actionHistory = [];
        _returnHistory = [];
        _targetReturnHistory = [];

        // Build the neural network architecture
        // Create state encoder (converts state to embedding space)
        _stateEncoder = new FullyConnectedLayer<T>(
            _options.StateSize,
            _options.EmbeddingDim,
            new ReLUActivation<T>() as IActivationFunction<T>
        );

        // Create action encoder (converts action to embedding space)
        _actionEncoder = new FullyConnectedLayer<T>(
            _options.ActionSize,
            _options.EmbeddingDim,
            new ReLUActivation<T>() as IActivationFunction<T>
        );

        // Create return encoder (converts scalar return to embedding space)
        _returnEncoder = new FullyConnectedLayer<T>(
            1,
            _options.EmbeddingDim,
            new ReLUActivation<T>() as IActivationFunction<T>
        );

        // Create transformer encoder (processes sequence of embeddings)
        var positionalEncodingLayer = PositionalEncodingFactory.Create<T>(
            _options.PositionalEncodingType,
            _options.ContextLength * 3,  // For states, actions, and returns
            _options.EmbeddingDim
        );

        // Cast to IPositionalEncoding<T> if it implements the interface, otherwise use null for default
        IPositionalEncoding<T>? positionalEncoding = positionalEncodingLayer as IPositionalEncoding<T>;

        _transformer = new TransformerEncoderLayer<T>(
            _options.EmbeddingDim,
            _options.NumHeads,
            _options.EmbeddingDim * 4, // Hidden dim is typically 4x embedding dim
            _options.DropoutRate,
            _options.NumTransformerLayers,
            positionalEncoding
        );

        // Create action decoder (converts embeddings back to action space)
        _actionDecoder = new FullyConnectedLayer<T>(
            _options.EmbeddingDim,
            _options.ActionSize,
            _options.IsContinuous ? new TanhActivation<T>() as IActivationFunction<T> : new IdentityActivation<T>() as IActivationFunction<T>
        );

        // Initialize replay buffer
        _replayBuffer = new SequentialReplayBuffer<TState, Vector<T>, T>(
            options.MaxBufferSize,
            options.MaxTrajectoryLength
        );
        
        // Store the optimizer from options or create a default one later
        // We'll create the optimizer after the neural network is fully set up
        if (options.Optimizer != null)
        {
            _optimizer = options.Optimizer as IGradientBasedOptimizer<T, Vector<T>, T> 
                ?? throw new ArgumentException("Provided optimizer must implement IGradientBasedOptimizer<T, Vector<T>, T>");
        }
        else
        {
            // For now, set to null and handle optimization manually in the training loop
            // This is a common pattern in RL where we manually apply gradients to individual components
            _optimizer = null;
        }
    }
    
    /// <summary>
    /// Selects an action for the given state.
    /// </summary>
    /// <param name="state">The current state observation.</param>
    /// <param name="isTraining">Whether the agent should explore (true) or exploit (false).</param>
    /// <returns>The selected action.</returns>
    public override Vector<T> SelectAction(TState state, bool isTraining = true)
    {
        // Update training flag using base class method
        SetTrainingMode(isTraining);
        
        // Add state to history
        _stateHistory.Add(state);
        
        // Limit history to context length
        if (_stateHistory.Count > _options.ContextLength)
        {
            _stateHistory.RemoveAt(0);
            if (_actionHistory.Count > 0) _actionHistory.RemoveAt(0);
            if (_returnHistory.Count > 0) _returnHistory.RemoveAt(0);
            if (_targetReturnHistory.Count > 0) _targetReturnHistory.RemoveAt(0);
        }
        
        // If we don't have enough history, take a random action
        if (_stateHistory.Count < 2)
        {
            return TakeRandomAction();
        }
        
        // Get target return - in evaluation mode, target high returns
        // In training mode, use actual historical returns or a target
        T targetReturn;
        if (!isTraining)
        {
            // During evaluation, aim high
            targetReturn = NumOps.FromDouble(1.0);
        }
        else if (_targetReturnHistory.Count > 0)
        {
            // Use the most recent target return
            targetReturn = _targetReturnHistory[_targetReturnHistory.Count - 1];
        }
        else
        {
            // Default target
            targetReturn = NumOps.FromDouble(0.5);
        }
        
        // Create sequence for transformer
        var sequence = CreateSequenceFromHistory(targetReturn);
        
        // Get action from transformer
        var action = PredictActionFromSequence(sequence);
        
        // Add action to history (for next step prediction)
        _actionHistory.Add(action);
        
        // Increment step counter using base class method
        IncrementStepCounter();
        _step++;
        
        return action;
    }
    
    /// <summary>
    /// Creates a random action as a fallback when history is insufficient.
    /// </summary>
    /// <returns>A random action vector.</returns>
    private Vector<T> TakeRandomAction()
    {
        var action = new Vector<T>(_options.ActionSize);
        
        if (_options.IsContinuous)
        {
            // For continuous actions, sample from -1 to 1
            for (int i = 0; i < action.Length; i++)
            {
                action[i] = NumOps.FromDouble(2 * Random.NextDouble() - 1);
            }
        }
        else
        {
            // For discrete actions, one-hot encode a random action
            int randomAction = Random.Next(_options.ActionSize);
            for (int i = 0; i < action.Length; i++)
            {
                action[i] = i == randomAction ? NumOps.One : NumOps.Zero;
            }
        }
        
        return action;
    }
    
    /// <summary>
    /// Creates a sequence tensor from the history buffers for input to the transformer.
    /// </summary>
    /// <param name="targetReturn">The target return to condition on.</param>
    /// <returns>A tensor containing the encoded sequence.</returns>
    private Tensor<T> CreateSequenceFromHistory(T targetReturn)
    {
        // Calculate effective sequence length
        int seqLength = Math.Min(_stateHistory.Count, _options.ContextLength);
        
        // Create embedding sequence tensors
        var stateEmbeddings = new List<Tensor<T>>();
        var actionEmbeddings = new List<Tensor<T>>();
        var returnEmbeddings = new List<Tensor<T>>();
        
        // Process states, actions, and returns
        for (int i = 0; i < seqLength; i++)
        {
            // Get state
            TState state = _stateHistory[i];
            Tensor<T> stateTensor;
            
            // Convert state to tensor if needed
            if (state is Tensor<T> tensor)
            {
                stateTensor = tensor;
            }
            else if (state is Vector<T> vector)
            {
                stateTensor = Tensor<T>.FromVector(vector);
            }
            else
            {
                throw new InvalidOperationException("State must be convertible to a tensor");
            }
            
            // Encode state
            var stateEncoded = _stateEncoder.Forward(stateTensor);
            stateEmbeddings.Add(stateEncoded);
            
            // Add action embedding if available
            if (i < _actionHistory.Count)
            {
                var actionTensor = Tensor<T>.FromVector(_actionHistory[i]);
                var actionEncoded = _actionEncoder.Forward(actionTensor);
                actionEmbeddings.Add(actionEncoded);
            }
            
            // Add return embedding if available
            if (i < _returnHistory.Count)
            {
                var returnVector = new Vector<T>(1) { [0] = _returnHistory[i] };
                var returnTensor = Tensor<T>.FromVector(returnVector);
                var returnEncoded = _returnEncoder.Forward(returnTensor);
                returnEmbeddings.Add(returnEncoded);
            }
        }
        
        // Add target return embedding for the last position
        var targetReturnVector = new Vector<T>(1) { [0] = targetReturn };
        var targetReturnTensor = Tensor<T>.FromVector(targetReturnVector);
        var targetReturnEncoded = _returnEncoder.Forward(targetReturnTensor);
        returnEmbeddings.Add(targetReturnEncoded);
        
        // Combine all embeddings into a single sequence
        // Format: [r_1, s_1, a_1, r_2, s_2, a_2, ..., r_t, s_t, a_t, r_{target}]
        var embeddingsList = new List<Tensor<T>>();
        
        for (int i = 0; i < seqLength; i++)
        {
            if (i < returnEmbeddings.Count) embeddingsList.Add(returnEmbeddings[i]);
            embeddingsList.Add(stateEmbeddings[i]);
            if (i < actionEmbeddings.Count) embeddingsList.Add(actionEmbeddings[i]);
        }
        
        // Add final target return embedding
        embeddingsList.Add(targetReturnEncoded);
        
        // Stack embeddings to create sequence
        return Tensor<T>.Stack(embeddingsList.ToArray());
    }
    
    /// <summary>
    /// Predicts the next action using the transformer model.
    /// </summary>
    /// <param name="sequence">The encoded sequence tensor.</param>
    /// <returns>The predicted action vector.</returns>
    private Vector<T> PredictActionFromSequence(Tensor<T> sequence)
    {
        // Pass sequence through transformer
        var transformerOutput = _transformer.Forward(sequence);
        
        // Get the embedding at the last position
        var lastEmbedding = transformerOutput.Slice(transformerOutput.Shape[0] - 1);
        
        // Decode to action
        var actionLogits = _actionDecoder.Forward(lastEmbedding);
        
        // Convert to vector
        var actionVector = actionLogits.ToVector();
        
        // For discrete actions, apply softmax and sample
        if (!_options.IsContinuous)
        {
            if (IsTraining)
            {
                // During training, sample based on probabilities
                actionVector = SampleDiscreteAction(actionVector);
            }
            else
            {
                // During evaluation, pick the highest probability action
                actionVector = GetArgmaxAction(actionVector);
            }
        }
        
        return actionVector;
    }
    
    /// <summary>
    /// Samples a discrete action from a probability distribution.
    /// </summary>
    /// <param name="actionLogits">The action logits (pre-softmax probabilities).</param>
    /// <returns>A one-hot encoded action vector.</returns>
    private Vector<T> SampleDiscreteAction(Vector<T> actionLogits)
    {
        // Apply softmax to get probabilities
        var probabilities = VectorHelper<T>.Softmax(actionLogits);
        
        // Sample from distribution
        var result = new Vector<T>(actionLogits.Length);
        
        // Generate a random value between 0 and 1
        double randomValue = Random.NextDouble();
        double cumulativeProbability = 0;
        
        // Select action based on cumulative probabilities
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulativeProbability += Convert.ToDouble(probabilities[i]);
            
            if (randomValue <= cumulativeProbability)
            {
                // One-hot encode the selected action
                result[i] = NumOps.One;
                break;
            }
        }
        
        return result;
    }
    
    /// <summary>
    /// Gets the action with the highest probability (argmax).
    /// </summary>
    /// <param name="actionLogits">The action logits (pre-softmax probabilities).</param>
    /// <returns>A one-hot encoded action vector.</returns>
    private Vector<T> GetArgmaxAction(Vector<T> actionLogits)
    {
        // Create one-hot vector for the highest logit
        var result = new Vector<T>(actionLogits.Length);
        
        // Find max index
        int maxIndex = 0;
        T maxValue = actionLogits[0];
        
        for (int i = 1; i < actionLogits.Length; i++)
        {
            if (NumOps.GreaterThan(actionLogits[i], maxValue))
            {
                maxValue = actionLogits[i];
                maxIndex = i;
            }
        }
        
        // Set the selected action
        result[maxIndex] = NumOps.One;
        
        return result;
    }
    
    /// <summary>
    /// Stores an experience tuple in the replay buffer and updates the agent's history.
    /// </summary>
    /// <param name="state">The state observation.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The next state observation.</param>
    /// <param name="done">Whether the episode is done.</param>
    public override void Learn(TState state, Vector<T> action, T reward, TState nextState, bool done)
    {
        // Skip learning if not in training mode
        if (!IsTraining)
        {
            return;
        }
        
        // Add the experience to the replay buffer
        _replayBuffer.Add(state, action, reward, nextState, done);
        
        // Add to return history
        _returnHistory.Add(reward);
        
        // Update target return history (can be policy dependent)
        _targetReturnHistory.Add(reward); // Simple version: target = most recent reward
        
        // Trim histories if they exceed context length
        if (_returnHistory.Count > _options.ContextLength)
        {
            _returnHistory.RemoveAt(0);
        }
        
        if (_targetReturnHistory.Count > _options.ContextLength)
        {
            _targetReturnHistory.RemoveAt(0);
        }
        
        // Train on replay buffer if we have enough data
        if (_step % _options.EpochsPerUpdate == 0 && _replayBuffer.Size > _options.BatchSize)
        {
            TrainOnBatch();
        }
    }
    
    /// <summary>
    /// Trains the model on a batch of experiences from the replay buffer.
    /// </summary>
    private void TrainOnBatch()
    {
        // Sample a batch of trajectories from the replay buffer
        var batch = _replayBuffer.SampleBatch(_options.BatchSize);
        
        if (batch == null || batch.States.Length == 0)
        {
            return;
        }
        
        // Train the model on the batch
        LastLoss = Train(batch.States, batch.Actions, batch.Rewards, batch.NextStates, batch.Dones);
    }
    
    /// <summary>
    /// Trains the model on historical data offline.
    /// </summary>
    /// <param name="trajectories">A list of trajectory batches containing historical data.</param>
    /// <returns>The final loss value after training.</returns>
    public T TrainOffline(List<Memory.TrajectoryBatch<TState, Vector<T>, T>> trajectories)
    {
        if (trajectories == null || trajectories.Count == 0)
        {
            return NumOps.Zero;
        }
        
        // Train on all trajectories
        T totalLoss = NumOps.Zero;
        int batchCount = 0;
        
        foreach (var batch in trajectories)
        {
            T batchLoss = Train(batch.States, batch.Actions, batch.Rewards, batch.NextStates, batch.Dones);
            totalLoss = NumOps.Add(totalLoss, batchLoss);
            batchCount++;
        }
        
        // Return average loss
        if (batchCount > 0)
        {
            LastLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchCount));
        }
        
        return LastLoss;
    }
    
    /// <summary>
    /// Trains the model on a batch of experiences.
    /// </summary>
    /// <param name="states">The batch of states.</param>
    /// <param name="actions">The batch of actions.</param>
    /// <param name="rewards">The batch of rewards.</param>
    /// <param name="nextStates">The batch of next states.</param>
    /// <param name="dones">The batch of done flags.</param>
    /// <returns>The loss value from the training.</returns>
    public T Train(TState[] states, Vector<T>[] actions, T[] rewards, TState[] nextStates, bool[] dones)
    {
        if (states.Length == 0 || actions.Length == 0 || rewards.Length == 0)
        {
            return NumOps.Zero;
        }
        
        // Convert to tensors
        int batchSize = states.Length;
        
        // Accumulate total loss
        T totalLoss = NumOps.Zero;
        
        // Train on trajectories
        for (int b = 0; b < batchSize; b++)
        {
            // Get trajectory length (up to context length)
            int trajectoryLength = Math.Min(_options.ContextLength, 1);
            
            // Compute returns-to-go
            var returnsToGo = ComputeReturnsToGo(rewards, b, trajectoryLength);
            
            // Create sequence for transformer input
            var sequence = CreateSequenceFromTrajectory(
                states, actions, returnsToGo, b, trajectoryLength);
            
            // Forward pass through transformer
            var transformerOutput = _transformer.Forward(sequence);
            
            // Compute loss
            T loss = ComputeActionPredictionLoss(transformerOutput, actions, b, trajectoryLength);
            
            // Add to total loss
            totalLoss = NumOps.Add(totalLoss, loss);
        }
        
        // Compute average loss
        T avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        
        // Backward pass and optimize (only if optimizer is available)
        if (_optimizer != null)
        {
            // For now, we skip the actual optimization since this requires implementing
            // gradient computation and backpropagation through the transformer
            // In a production system, this would involve:
            // 1. Computing gradients through backpropagation
            // 2. Applying gradients via the optimizer
            // TODO: Implement full gradient computation
        }
        
        // Update last loss using base class field
        LastLoss = avgLoss;
        
        return LastLoss;
    }
    
    /// <summary>
    /// Computes the returns-to-go for a trajectory.
    /// </summary>
    /// <param name="rewards">The batch of rewards.</param>
    /// <param name="batchIndex">The index of the current trajectory in the batch.</param>
    /// <param name="trajectoryLength">The length of the trajectory.</param>
    /// <returns>An array of returns-to-go.</returns>
    private T[] ComputeReturnsToGo(T[] rewards, int batchIndex, int trajectoryLength)
    {
        // Initialize returns-to-go
        var returnsToGo = new T[trajectoryLength];
        
        // Compute discounted returns-to-go
        T futureReturn = NumOps.Zero;
        
        for (int i = trajectoryLength - 1; i >= 0; i--)
        {
            int index = batchIndex * trajectoryLength + i;
            if (index < rewards.Length)
            {
                // R_t = r_t + gamma * R_{t+1}
                futureReturn = NumOps.Add(rewards[index], 
                    NumOps.Multiply(NumOps.FromDouble(_options.Gamma), futureReturn));
            }
            
            returnsToGo[i] = futureReturn;
        }
        
        // Normalize returns if enabled
        if (_options.NormalizeReturns)
        {
            NormalizeArray(returnsToGo);
        }
        
        return returnsToGo;
    }
    
    /// <summary>
    /// Creates a sequence tensor from a trajectory for input to the transformer.
    /// </summary>
    /// <param name="states">The batch of states.</param>
    /// <param name="actions">The batch of actions.</param>
    /// <param name="returnsToGo">The computed returns-to-go.</param>
    /// <param name="batchIndex">The index of the current trajectory in the batch.</param>
    /// <param name="trajectoryLength">The length of the trajectory.</param>
    /// <returns>A tensor containing the encoded sequence.</returns>
    private Tensor<T> CreateSequenceFromTrajectory(
        TState[] states, Vector<T>[] actions, T[] returnsToGo, 
        int batchIndex, int trajectoryLength)
    {
        // Create embedding sequence tensors
        var stateEmbeddings = new List<Tensor<T>>();
        var actionEmbeddings = new List<Tensor<T>>();
        var returnEmbeddings = new List<Tensor<T>>();
        
        // Process states, actions, and returns
        for (int i = 0; i < trajectoryLength; i++)
        {
            // Get state
            int index = batchIndex * trajectoryLength + i;
            if (index >= states.Length) break;
            
            TState state = states[index];
            Tensor<T> stateTensor;
            
            // Convert state to tensor if needed
            if (state is Tensor<T> tensor)
            {
                stateTensor = tensor;
            }
            else if (state is Vector<T> vector)
            {
                stateTensor = Tensor<T>.FromVector(vector);
            }
            else
            {
                throw new InvalidOperationException("State must be convertible to a tensor");
            }
            
            // Encode state
            var stateEncoded = _stateEncoder.Forward(stateTensor);
            stateEmbeddings.Add(stateEncoded);
            
            // Get action
            if (index < actions.Length)
            {
                var actionTensor = Tensor<T>.FromVector(actions[index]);
                var actionEncoded = _actionEncoder.Forward(actionTensor);
                actionEmbeddings.Add(actionEncoded);
            }
            
            // Get return
            if (i < returnsToGo.Length)
            {
                var returnVector = new Vector<T>(1) { [0] = returnsToGo[i] };
                var returnTensor = Tensor<T>.FromVector(returnVector);
                var returnEncoded = _returnEncoder.Forward(returnTensor);
                returnEmbeddings.Add(returnEncoded);
            }
        }
        
        // Combine all embeddings into a single sequence
        // Format: [r_1, s_1, a_1, r_2, s_2, a_2, ..., r_t, s_t, a_t]
        var embeddingsList = new List<Tensor<T>>();
        
        for (int i = 0; i < trajectoryLength; i++)
        {
            if (i < returnEmbeddings.Count) embeddingsList.Add(returnEmbeddings[i]);
            if (i < stateEmbeddings.Count) embeddingsList.Add(stateEmbeddings[i]);
            if (i < actionEmbeddings.Count) embeddingsList.Add(actionEmbeddings[i]);
        }
        
        // Stack embeddings to create sequence
        return Tensor<T>.Stack(embeddingsList.ToArray());
    }
    
    /// <summary>
    /// Computes the action prediction loss for the transformer output.
    /// </summary>
    /// <param name="transformerOutput">The output from the transformer.</param>
    /// <param name="actions">The batch of actions.</param>
    /// <param name="batchIndex">The index of the current trajectory in the batch.</param>
    /// <param name="trajectoryLength">The length of the trajectory.</param>
    /// <returns>The computed loss.</returns>
    private T ComputeActionPredictionLoss(
        Tensor<T> transformerOutput, Vector<T>[] actions, 
        int batchIndex, int trajectoryLength)
    {
        T totalLoss = NumOps.Zero;
        int actionCount = 0;
        
        // Compute loss for each action prediction
        for (int i = 0; i < trajectoryLength; i++)
        {
            // Get actual action
            int index = batchIndex * trajectoryLength + i;
            if (index >= actions.Length) break;
            
            Vector<T> targetAction = actions[index];
            
            // Transformer output pattern is [r_1, s_1, a_1, ...], so action is at position 2, 5, 8, ...
            int actionPosition = i * 3 + 2; 
            if (actionPosition >= transformerOutput.Shape[0]) break;
            
            // Get embedding at action position
            var actionEmbedding = transformerOutput.Slice(actionPosition);
            
            // Decode to action prediction
            var predictedActionTensor = _actionDecoder.Forward(actionEmbedding);
            var predictedAction = predictedActionTensor.ToVector();
            
            // Compute loss based on action space type
            T loss;
            if (_options.IsContinuous)
            {
                // For continuous actions, use MSE loss
                loss = ComputeMSELoss(predictedAction, targetAction);
            }
            else
            {
                // For discrete actions, use cross-entropy loss
                loss = ComputeCrossEntropyLoss(predictedAction, targetAction);
            }
            
            totalLoss = NumOps.Add(totalLoss, loss);
            actionCount++;
        }
        
        // Return average loss
        if (actionCount > 0)
        {
            return NumOps.Divide(totalLoss, NumOps.FromDouble(actionCount));
        }
        
        return NumOps.Zero;
    }
    
    /// <summary>
    /// Computes Mean Squared Error loss between predicted and target actions.
    /// </summary>
    /// <param name="predicted">The predicted action vector.</param>
    /// <param name="target">The target action vector.</param>
    /// <returns>The MSE loss.</returns>
    private T ComputeMSELoss(Vector<T> predicted, Vector<T> target)
    {
        T sumSquaredError = NumOps.Zero;
        
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], target[i]);
            sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Multiply(diff, diff));
        }
        
        return NumOps.Divide(sumSquaredError, NumOps.FromDouble(predicted.Length));
    }
    
    /// <summary>
    /// Computes cross-entropy loss between predicted logits and target one-hot action.
    /// </summary>
    /// <param name="predictedLogits">The predicted action logits.</param>
    /// <param name="targetOneHot">The target one-hot action vector.</param>
    /// <returns>The cross-entropy loss.</returns>
    private T ComputeCrossEntropyLoss(Vector<T> predictedLogits, Vector<T> targetOneHot)
    {
        // Apply softmax to get probabilities
        var probabilities = VectorHelper<T>.Softmax(predictedLogits);
        
        // Compute cross-entropy loss
        T loss = NumOps.Zero;
        
        for (int i = 0; i < probabilities.Length; i++)
        {
            // Only consider non-zero targets
            if (NumOps.GreaterThan(targetOneHot[i], NumOps.Zero))
            {
                // -target * log(prob)
                T logProb = NumOps.Log(NumOps.Add(probabilities[i], NumOps.FromDouble(1e-10))); // Add small constant for numerical stability
                T contribution = NumOps.Multiply(targetOneHot[i], logProb);
                loss = NumOps.Subtract(loss, contribution);
            }
        }
        
        return loss;
    }
    
    /// <summary>
    /// Normalizes an array of values to have zero mean and unit variance.
    /// </summary>
    /// <param name="array">The array to normalize.</param>
    private void NormalizeArray(T[] array)
    {
        if (array.Length <= 1) return;
        
        // Compute mean
        T sum = NumOps.Zero;
        foreach (T value in array)
        {
            sum = NumOps.Add(sum, value);
        }
        T mean = NumOps.Divide(sum, NumOps.FromDouble(array.Length));
        
        // Compute standard deviation
        T sumSquaredDiff = NumOps.Zero;
        foreach (T value in array)
        {
            T diff = NumOps.Subtract(value, mean);
            sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
        }
        T variance = NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(array.Length));
        T stdDev = NumOps.Sqrt(variance);
        
        // Add small constant to avoid division by zero
        stdDev = NumOps.Add(stdDev, NumOps.FromDouble(1e-8));
        
        // Normalize array
        for (int i = 0; i < array.Length; i++)
        {
            array[i] = NumOps.Divide(NumOps.Subtract(array[i], mean), stdDev);
        }
    }
    
    // GetLatestLoss(), IsTraining, and SetTrainingMode() are provided by base class
    
    /// <summary>
    /// Gets the agent's parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters of the agent.</returns>
    public Vector<T> GetParameters()
    {
        // Collect parameters from all network components
        var parameters = new List<Vector<T>>
        {
            _stateEncoder.GetParameters(),
            _actionEncoder.GetParameters(),
            _returnEncoder.GetParameters(),
            _transformer.GetParameters(),
            _actionDecoder.GetParameters()
        };
        
        // Combine into a single vector
        return VectorHelper<T>.Concatenate(parameters);
    }
    
    /// <summary>
    /// Sets the agent's parameters from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public void SetParameters(Vector<T> parameters)
    {
        int currentIndex = 0;
        
        // Set parameters for state encoder
        int stateEncoderParamCount = _stateEncoder.GetParameters().Length;
        _stateEncoder.SetParameters(parameters.Slice(currentIndex, stateEncoderParamCount));
        currentIndex += stateEncoderParamCount;
        
        // Set parameters for action encoder
        int actionEncoderParamCount = _actionEncoder.GetParameters().Length;
        _actionEncoder.SetParameters(parameters.Slice(currentIndex, actionEncoderParamCount));
        currentIndex += actionEncoderParamCount;
        
        // Set parameters for return encoder
        int returnEncoderParamCount = _returnEncoder.GetParameters().Length;
        _returnEncoder.SetParameters(parameters.Slice(currentIndex, returnEncoderParamCount));
        currentIndex += returnEncoderParamCount;
        
        // Set parameters for transformer
        int transformerParamCount = _transformer.GetParameters().Length;
        _transformer.SetParameters(parameters.Slice(currentIndex, transformerParamCount));
        currentIndex += transformerParamCount;
        
        // Set parameters for action decoder
        int actionDecoderParamCount = _actionDecoder.GetParameters().Length;
        _actionDecoder.SetParameters(parameters.Slice(currentIndex, actionDecoderParamCount));
    }
    
    /// <summary>
    /// Saves the agent to the specified path.
    /// </summary>
    /// <param name="filePath">The path to save the agent to.</param>
    public override void Save(string filePath)
    {
        using (var fileStream = new System.IO.FileStream(filePath, System.IO.FileMode.Create))
        {
            using (var writer = new System.IO.BinaryWriter(fileStream))
            {
                // Write model type identifier
                writer.Write("DecisionTransformerAgent");
                
                // Save agent parameters
                var parameters = GetParameters();
                writer.Write(parameters.Length);
                for (int i = 0; i < parameters.Length; i++)
                {
                    writer.Write(Convert.ToDouble(parameters[i]));
                }
                
                // Save options
                writer.Write(_options.StateSize);
                writer.Write(_options.ActionSize);
                writer.Write(_options.IsContinuous);
                writer.Write(_options.ContextLength);
                writer.Write(_options.NumTransformerLayers);
                writer.Write(_options.NumHeads);
                writer.Write(_options.EmbeddingDim);
                writer.Write(_options.ReturnConditioned);
                writer.Write(_options.TransformerLearningRate);
            }
        }
    }
    
    /// <summary>
    /// Loads the agent from the specified path.
    /// </summary>
    /// <param name="filePath">The path to load the agent from.</param>
    public override void Load(string filePath)
    {
        using (var fileStream = new System.IO.FileStream(filePath, System.IO.FileMode.Open))
        {
            using (var reader = new System.IO.BinaryReader(fileStream))
            {
                // Read and verify model type identifier
                string modelType = reader.ReadString();
                if (modelType != "DecisionTransformerAgent")
                {
                    throw new InvalidOperationException($"Expected DecisionTransformerAgent, but got {modelType}");
                }
                
                // Load parameters
                int paramCount = reader.ReadInt32();
                var parameters = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++)
                {
                    parameters[i] = NumOps.FromDouble(reader.ReadDouble());
                }
                
                // Set parameters
                SetParameters(parameters);
                
                // Load and verify options
                int stateSize = reader.ReadInt32();
                int actionSize = reader.ReadInt32();
                bool isContinuous = reader.ReadBoolean();
                int contextLength = reader.ReadInt32();
                int numLayers = reader.ReadInt32();
                int numHeads = reader.ReadInt32();
                int embeddingDim = reader.ReadInt32();
                bool returnConditioned = reader.ReadBoolean();
                double learningRate = reader.ReadDouble();
                
                // Verify that basic options match
                if (stateSize != _options.StateSize || actionSize != _options.ActionSize)
                {
                    throw new InvalidOperationException(
                        $"Model dimensions mismatch. Saved model: State={stateSize}, Action={actionSize}. " +
                        $"Current options: State={_options.StateSize}, Action={_options.ActionSize}");
                }
            }
        }
    }
}