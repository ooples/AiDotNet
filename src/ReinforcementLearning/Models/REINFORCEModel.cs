using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Models.Options;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Models
{
    /// <summary>
    /// Reinforcement Learning model implementing REINFORCE algorithm with policy gradients.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class REINFORCEModel<T> : ReinforcementLearningModelBase<T>
    {
        private readonly PolicyGradientOptions _options = default!;
        private REINFORCEAgent<Tensor<T>, int, T>? _discreteAgent;
        private REINFORCEAgent<Tensor<T>, Vector<T>, T>? _continuousAgent;
        private readonly int _batchSize;
        private readonly int _epochsPerUpdate;

        /// <summary>
        /// Initializes a new instance of the <see cref="REINFORCEModel{T}"/> class.
        /// </summary>
        /// <param name="options">The options for the REINFORCE algorithm.</param>
        public REINFORCEModel(PolicyGradientOptions options)
            : base(options)
        {
            _options = options;
            _batchSize = options.BatchSize;
            _epochsPerUpdate = options.EpochsPerUpdate;
            
            // Set action space type
            IsContinuous = options.IsContinuous;
        }

        /// <summary>
        /// Initializes the appropriate agent based on the action space type.
        /// </summary>
        protected override void InitializeAgent()
        {
            if (IsContinuous)
            {
                _continuousAgent = new REINFORCEAgent<Tensor<T>, Vector<T>, T>(_options);
                _discreteAgent = null;
            }
            else
            {
                _discreteAgent = new REINFORCEAgent<Tensor<T>, int, T>(_options);
                _continuousAgent = null;
            }
        }
        
        /// <summary>
        /// Gets the agent that interacts with the environment.
        /// </summary>
        /// <returns>The agent as an IAgent interface.</returns>
        protected override IAgent<Tensor<T>, Vector<T>, T> GetAgent()
        {
            if (IsContinuous && _continuousAgent != null)
            {
                return _continuousAgent;
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                throw new InvalidOperationException("Discrete agent cannot be returned as IAgent<Tensor<T>, Vector<T>, T>");
            }
            
            throw new InvalidOperationException("No agent initialized");
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
                
                // Get prediction based on action space type
                if (IsContinuous && _continuousAgent != null)
                {
                    // For continuous actions, predict the first dimension's action
                    var action = _continuousAgent.SelectAction(inputTensor, false);
                    predictions[i] = action[0]; // Just return first dimension of action
                }
                else if (!IsContinuous && _discreteAgent != null)
                {
                    // For discrete actions, convert action index to scalar
                    int action = _discreteAgent.SelectAction(inputTensor, false);
                    predictions[i] = NumOps.FromDouble(action);
                }
                else
                {
                    // Fallback
                    predictions[i] = NumOps.Zero;
                }
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
                    var item = input.GetSlice(i);
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
            if (IsContinuous && _continuousAgent != null)
            {
                // For continuous actions, return the complete action vector
                var action = _continuousAgent.SelectAction(input, false);
                return Tensor<T>.FromVector(action);
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                // For discrete actions, create a one-hot tensor
                int action = _discreteAgent.SelectAction(input, false);
                var result = new Tensor<T>(new[] { _options.ActionSize });
                for (int i = 0; i < _options.ActionSize; i++)
                {
                    result[i] = i == action ? NumOps.One : NumOps.Zero;
                }
                return result;
            }
            else
            {
                // Fallback
                return new Tensor<T>(new[] { 1 });
            }
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
            
            if (IsContinuous && _continuousAgent != null)
            {
                // For continuous actions, use the action vector directly
                _continuousAgent.Learn(state, action, reward, nextState, done);
                LastLoss = NumOps.Zero;
                return LastLoss;
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                // For discrete actions, convert the Vector<T> to an int
                int discreteAction = Convert.ToInt32(action[0]);
                _discreteAgent.Learn(state, discreteAction, reward, nextState, done);
                LastLoss = NumOps.Zero;
                return LastLoss;
            }
            
            return NumOps.Zero;
        }
        
        /// <summary>
        /// Trains the agent using an experience tuple (legacy method).
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public void Learn(Tensor<T> state, object action, T reward, Tensor<T> nextState, bool done)
        {
            if (IsContinuous && _continuousAgent != null)
            {
                if (action is Vector<T> continuousAction)
                {
                    _continuousAgent.Learn(state, continuousAction, reward, nextState, done);
                    LastLoss = NumOps.Zero;
                }
                else
                {
                    throw new ArgumentException("Action must be Vector<T> for continuous action spaces");
                }
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                if (action is int discreteAction)
                {
                    _discreteAgent.Learn(state, discreteAction, reward, nextState, done);
                    LastLoss = NumOps.Zero;
                }
                else
                {
                    throw new ArgumentException("Action must be int for discrete action spaces");
                }
            }
        }

        /// <summary>
        /// Selects an action based on the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="isTraining">Whether to use exploration during action selection.</param>
        /// <returns>The selected action as a vector.</returns>
        public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = false)
        {
            if (IsContinuous && _continuousAgent != null)
            {
                return _continuousAgent.SelectAction(state, isTraining || IsTraining);
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                // For discrete actions, convert integer to a single-element vector
                int action = _discreteAgent.SelectAction(state, isTraining || IsTraining);
                var result = new Vector<T>(1);
                result[0] = NumOps.FromDouble(action);
                return result;
            }
            else
            {
                throw new InvalidOperationException("No agent initialized");
            }
        }

        /// <summary>
        /// Gets the metadata for this model.
        /// </summary>
        /// <returns>The model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = new ModelMetadata<T>
            {
                ModelType = Enums.ModelType.REINFORCEModel,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "Algorithm", "REINFORCE" },
                    { "IsContinuous", IsContinuous },
                    { "StateSize", _options.StateSize },
                    { "ActionSize", _options.ActionSize },
                    { "UseBaseline", _options.UseBaseline }
                }
            };

            return metadata;
        }

        /// <summary>
        /// Saves the model to the specified path.
        /// </summary>
        /// <param name="path">The path to save the model to.</param>
        public override void SaveModel(string path)
        {
            if (IsContinuous && _continuousAgent != null)
            {
                _continuousAgent.Save(path);
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                _discreteAgent.Save(path);
            }
        }

        /// <summary>
        /// Loads the model from the specified path.
        /// </summary>
        /// <param name="path">The path to load the model from.</param>
        public override void LoadModel(string path)
        {
            if (IsContinuous && _continuousAgent != null)
            {
                _continuousAgent.Load(path);
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                _discreteAgent.Load(path);
            }
        }

        /// <summary>
        /// Sets the model to training mode.
        /// </summary>
        public override void SetTrainingMode()
        {
            base.SetTrainingMode();
            
            if (IsContinuous && _continuousAgent != null)
            {
                _continuousAgent.SetTrainingMode(true);
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                _discreteAgent.SetTrainingMode(true);
            }
        }
        
        /// <summary>
        /// Sets the model to evaluation mode.
        /// </summary>
        public override void SetEvaluationMode()
        {
            base.SetEvaluationMode();
            
            if (IsContinuous && _continuousAgent != null)
            {
                _continuousAgent.SetTrainingMode(false);
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                _discreteAgent.SetTrainingMode(false);
            }
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
            return new REINFORCEModel<T>(_options);
        }
        /// <summary>
        /// Gets the parameters of the model as a single vector.
        /// </summary>
        /// <returns>A vector containing all parameters of the model.</returns>
        public override Vector<T> GetParameters()
        {
            // Get parameters from the appropriate agent
            if (IsContinuous && _continuousAgent != null)
            {
                return _continuousAgent.GetParameters();
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                return _discreteAgent.GetParameters();
            }
            
            // No agent initialized
            return new Vector<T>(0);
        }
        
        /// <summary>
        /// Sets the parameters of the model from a vector.
        /// </summary>
        /// <param name="parameters">A vector containing all parameters to set.</param>
        public override void SetParameters(Vector<T> parameters)
        {
            // Set parameters to the appropriate agent
            if (IsContinuous && _continuousAgent != null)
            {
                _continuousAgent.SetParameters(parameters);
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                _discreteAgent.SetParameters(parameters);
            }
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
                writer.Write("REINFORCEModel");
                
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
                writer.Write(_options.UseBaseline);
                writer.Write(_options.BatchSize);
                writer.Write(_options.EpochsPerUpdate);
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
                if (modelType != "REINFORCEModel")
                {
                    throw new InvalidOperationException($"Expected REINFORCEModel, but got {modelType}");
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
                bool isContinuous = reader.ReadBoolean();
                bool useBaseline = reader.ReadBoolean();
                int batchSize = reader.ReadInt32();
                int epochsPerUpdate = reader.ReadInt32();
                
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
            if (IsContinuous && _continuousAgent != null)
            {
                // Need to extract continuous actions from the tensor
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
                
                // Convert tensors to arrays
                var statesArray = ConvertTensorToStateArray(states);
                var rewardsArray = rewards.ToArray();
                var nextStatesArray = ConvertTensorToStateArray(nextStates);
                var donesArray = ConvertTensorToBoolArray(dones);
                
                return _continuousAgent.Train(statesArray, actionsList.ToArray(), rewardsArray, nextStatesArray, donesArray);
            }
            else if (!IsContinuous && _discreteAgent != null)
            {
                // Need to extract discrete actions from the tensor
                var batchSize = actions.Shape[0];
                var discreteActions = new int[batchSize];
                
                for (int i = 0; i < batchSize; i++)
                {
                    discreteActions[i] = Convert.ToInt32(actions[i, 0]);
                }
                
                // Convert tensors to arrays
                var statesArray = ConvertTensorToStateArray(states);
                var rewardsArray = rewards.ToArray();
                var nextStatesArray = ConvertTensorToStateArray(nextStates);
                var donesArray = ConvertTensorToBoolArray(dones);
                
                return _discreteAgent.Train(statesArray, discreteActions, rewardsArray, nextStatesArray, donesArray);
            }
            
            return NumOps.Zero;
        }
        
        private Tensor<T>[] ConvertTensorToStateArray(Tensor<T> tensor)
        {
            var batchSize = tensor.Shape[0];
            var stateSize = tensor.Shape[1];
            var states = new Tensor<T>[batchSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                var state = new Tensor<T>(new[] { stateSize });
                for (int j = 0; j < stateSize; j++)
                {
                    state[j] = tensor[i, j];
                }
                states[i] = state;
            }
            
            return states;
        }
        
        private T[] ConvertTensorToArray(Tensor<T> tensor)
        {
            var array = new T[tensor.Shape[0]];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = tensor[i];
            }
            return array;
        }
        
        private bool[] ConvertTensorToBoolArray(Vector<T> vector)
        {
            var array = new bool[vector.Length];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = !NumOps.Equals(vector[i], NumOps.Zero);
            }
            return array;
        }
    }
}