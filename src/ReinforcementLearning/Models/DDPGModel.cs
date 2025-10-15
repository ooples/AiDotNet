using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using System.IO;

namespace AiDotNet.ReinforcementLearning.Models
{
    /// <summary>
    /// Deep Deterministic Policy Gradient (DDPG) model for continuous action spaces.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class DDPGModel<T> : ReinforcementLearningModelBase<T>
    {
        private readonly DDPGOptions _options = default!;
        private readonly DDPGAgent<Tensor<T>, T> _agent = default!;
        private readonly int _batchSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="DDPGModel{T}"/> class.
        /// </summary>
        /// <param name="options">The options for the DDPG algorithm.</param>
        public DDPGModel(DDPGOptions options)
            : base(options)
        {
            _options = options;
            _batchSize = options.BatchSize;
            
            // DDPG is always continuous action space
            IsContinuous = true;
            
            // Initialize agent
            _agent = new DDPGAgent<Tensor<T>, T>(options);
        }
        
        /// <summary>
        /// Initializes the agent that will interact with the environment.
        /// </summary>
        protected override void InitializeAgent()
        {
            // Agent is already initialized in the constructor
        }
        
        /// <summary>
        /// Gets the agent that interacts with the environment.
        /// </summary>
        /// <returns>The DDPG agent as an IAgent interface.</returns>
        protected override IAgent<Tensor<T>, Vector<T>, T> GetAgent()
        {
            return _agent ?? throw new InvalidOperationException("Agent has not been initialized");
        }

        /// <summary>
        /// Makes predictions for the given inputs.
        /// </summary>
        /// <param name="X">The input features.</param>
        /// <returns>The predicted values.</returns>
        public override Vector<T> Predict(Matrix<T> X)
        {
            // Create a vector for predictions
            var predictions = new Vector<T>(X.Rows);

            // Make predictions for each input row
            for (int i = 0; i < X.Rows; i++)
            {
                var input = X.GetRow(i);
                var inputTensor = Tensor<T>.FromVector(input);
                
                // For continuous actions, predict the first dimension's action
                var action = _agent.SelectAction(inputTensor, false);
                predictions[i] = action[0]; // Just return first dimension of action
            }

            return predictions;
        }

        /// <summary>
        /// Performs a forward pass to make predictions for tensor inputs.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            // Check input shape
            if (input.Rank < 1)
            {
                throw new InvalidOperationException("Input tensor must have at least rank 1");
            }

            // If input is a batch (first dimension > 1), process each item
            if (input.Rank > 1 && input.Shape[0] > 1)
            {
                var batchSize = input.Shape[0];
                var outputs = new List<Tensor<T>>();

                // Process each item in the batch
                for (int i = 0; i < batchSize; i++)
                {
                    var item = input.Slice(i);
                    outputs.Add(PredictSingle(item));
                }

                // Stack outputs into a batch
                return Tensor<T>.Stack(outputs.ToArray());
            }
            else
            {
                // Single input
                return PredictSingle(input);
            }
        }

        /// <summary>
        /// Makes a prediction for a single input tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        private Tensor<T> PredictSingle(Tensor<T> input)
        {
            // For continuous actions, return the complete action vector
            var action = _agent.SelectAction(input, false);
            return Tensor<T>.FromVector(action);
        }

        /// <summary>
        /// Trains the agent using an experience tuple.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public void Learn(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done)
        {
            _agent.Learn(state, action, reward, nextState, done);
        }

        /// <summary>
        /// Selects an action based on the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="isTraining">Whether to use exploration during action selection.</param>
        /// <returns>The selected action as a vector.</returns>
        public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = false)
        {
            return _agent.SelectAction(state, isTraining || IsTraining);
        }
        
        /// <summary>
        /// Updates the model based on experience with the environment.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="action">The action taken.</param>
        /// <param name="reward">The reward received.</param>
        /// <param name="nextState">The next state observation.</param>
        /// <param name="done">Whether the episode is done.</param>
        /// <returns>The loss value from the update.</returns>
        public override T Update(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done)
        {
            if (!IsTraining)
            {
                return NumOps.Zero; // No updates during evaluation
            }
            
            // Store experience and update model
            _agent.Learn(state, action, reward, nextState, done);
            
            // Get the latest loss value
            LastLoss = _agent.GetLatestLoss();
            
            return LastLoss;
        }

        /// <summary>
        /// Gets the metadata for this model.
        /// </summary>
        /// <returns>The model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            
            metadata.ModelType = ModelType.NeuralNetwork; // ReinforcementLearning models use neural networks
            metadata.Description = "Actor-critic reinforcement learning algorithm for continuous control";
            metadata.FeatureCount = _options.StateSize;
            metadata.Complexity = (_options.ActorHiddenLayers.Sum() + _options.CriticHiddenLayers.Sum()); // Complexity based on total network size
            metadata.AdditionalInfo = new Dictionary<string, object>
            {
                { "Algorithm", "DDPG" },
                { "StateSize", _options.StateSize },
                { "ActionSize", _options.ActionSize },
                { "ActorNetworkArchitecture", _options.ActorNetworkArchitecture },
                { "CriticNetworkArchitecture", _options.CriticNetworkArchitecture },
                { "UsePrioritizedReplay", _options.UsePrioritizedReplay }
            };

            return metadata;
        }

        /// <summary>
        /// Saves the model to the specified path.
        /// </summary>
        /// <param name="path">The path to save the model to.</param>
        public override void SaveModel(string path)
        {
            // Use the base class implementation which uses our Save(Stream) method
            base.SaveModel(path);
        }

        /// <summary>
        /// Loads the model from the specified path.
        /// </summary>
        /// <param name="path">The path to load the model from.</param>
        public override void LoadModel(string path)
        {
            // Use the base class implementation which uses our Load(Stream) method
            base.LoadModel(path);
        }

        /// <summary>
        /// Sets the model to training mode.
        /// </summary>
        public override void SetTrainingMode()
        {
            base.SetTrainingMode();
            _agent.SetTrainingMode(true);
        }
        
        /// <summary>
        /// Sets the model to evaluation mode.
        /// </summary>
        public override void SetEvaluationMode()
        {
            base.SetEvaluationMode();
            _agent.SetTrainingMode(false);
        }

        /// <summary>
        /// Performs a training step on a batch of data.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="expectedOutput">The expected output tensor.</param>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // This method is not typically used in RL in the same way as supervised learning
            // Instead, the Learn method is used with experience tuples
        }

        /// <summary>
        /// Creates a new instance of the model.
        /// </summary>
        /// <returns>A new instance of the model with the same configuration but no learned parameters.</returns>
        public override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new DDPGModel<T>(_options);
        }
        
        /// <summary>
        /// Creates a deep copy of this model.
        /// </summary>
        /// <returns>A deep copy of this model with the same parameters and state.</returns>
        public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
        {
            // Use the base class implementation which uses serialization/deserialization
            return base.DeepCopy();
        }
        
        /// <summary>
        /// Creates a shallow copy of this model.
        /// </summary>
        /// <returns>A shallow copy of this model.</returns>
        public override IFullModel<T, Tensor<T>, Tensor<T>> Clone()
        {
            // Use the base class implementation which creates a new instance
            return base.Clone();
        }
        
        /// <summary>
        /// Creates a new instance with the specified parameters.
        /// </summary>
        /// <param name="parameters">The parameters to set in the new instance.</param>
        /// <returns>A new instance with the specified parameters.</returns>
        public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
        {
            // Use the base class implementation which uses DeepCopy and SetParameters
            return base.WithParameters(parameters);
        }
        /// <summary>
        /// Gets the parameters of the model as a single vector.
        /// </summary>
        /// <returns>A vector containing all parameters of the model.</returns>
        public override Vector<T> GetParameters()
        {
            return _agent.GetParameters();
        }
        
        /// <summary>
        /// Sets the parameters of the model from a vector.
        /// </summary>
        /// <param name="parameters">A vector containing all parameters to set.</param>
        public override void SetParameters(Vector<T> parameters)
        {
            _agent.SetParameters(parameters);
        }
        
        /// <summary>
        /// Gets the last computed loss value.
        /// </summary>
        /// <returns>The last computed loss value.</returns>
        public override T GetLoss()
        {
            return LastLoss;
        }
        
        /// <summary>
        /// Saves the model to a stream.
        /// </summary>
        /// <param name="stream">The stream to save the model to.</param>
        public override void Save(Stream stream)
        {
            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
            {
                // Write model type identifier
                writer.Write("DDPGModel");
                
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
                writer.Write(_options.ActorLearningRate);
                writer.Write(_options.CriticLearningRate);
                writer.Write(_options.Gamma);
                writer.Write(_options.Tau);
                writer.Write(_options.BatchSize);
                writer.Write(_options.UsePrioritizedReplay);
            }
        }
        
        /// <summary>
        /// Loads the model from a stream.
        /// </summary>
        /// <param name="stream">The stream to load the model from.</param>
        public override void Load(Stream stream)
        {
            using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, true))
            {
                // Read and verify model type identifier
                string modelType = reader.ReadString();
                if (modelType != "DDPGModel")
                {
                    throw new InvalidOperationException($"Expected DDPGModel, but got {modelType}");
                }
                
                // Load parameters
                int paramCount = reader.ReadInt32();
                var parameters = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++)
                {
                    parameters[i] = NumOps.FromDouble(reader.ReadDouble());
                }
                
                // Set parameters to the agent
                SetParameters(parameters);
                
                // Load and verify basic options
                int stateSize = reader.ReadInt32();
                int actionSize = reader.ReadInt32();
                double actorLearningRate = reader.ReadDouble();
                double criticLearningRate = reader.ReadDouble();
                double gamma = reader.ReadDouble();
                double tau = reader.ReadDouble();
                int batchSize = reader.ReadInt32();
                bool usePrioritizedReplay = reader.ReadBoolean();
                
                // Verify that basic options match
                if (stateSize != _options.StateSize || actionSize != _options.ActionSize)
                {
                    throw new InvalidOperationException(
                        $"Model dimensions mismatch. Saved model: State={stateSize}, Action={actionSize}. " +
                        $"Current options: State={_options.StateSize}, Action={_options.ActionSize}");
                }
            }
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
        protected override T TrainOnBatch(
            Tensor<T> states,
            Tensor<T> actions,
            Vector<T> rewards,
            Tensor<T> nextStates,
            Vector<T> dones)
        {
            // Extract continuous actions from the tensor
            var batchSize = actions.Shape[0];
            var actionSize = actions.Shape[1];
            var actionsList = new List<Vector<T>>();
            
            for (int i = 0; i < batchSize; i++)
            {
                var action = new Vector<T>(actionSize);
                for (int j = 0; j < actionSize; j++)
                {
                    action[j] = actions[i, j];
                }
                actionsList.Add(action);
            }
            
            return _agent.Train(states, actionsList.ToArray(), rewards, nextStates, dones);
        }
    }
}