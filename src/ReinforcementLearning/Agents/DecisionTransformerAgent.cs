using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;

/// <summary>
/// Decision Transformer agent for offline reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Decision Transformer treats RL as sequence modeling, using transformer architecture
/// to predict actions conditioned on desired returns-to-go.
/// </para>
/// <para><b>For Beginners:</b>
/// Instead of learning "what's the best action", Decision Transformer learns
/// "what action was taken when the outcome was X". At test time, you specify
/// the desired outcome, and it generates the action sequence.
///
/// Key innovation:
/// - **Return Conditioning**: Specify target return, get actions that achieve it
/// - **Sequence Modeling**: Uses transformers like GPT for temporal dependencies
/// - **No RL Updates**: Just supervised learning on (return, state, action) sequences
/// - **Offline-First**: Designed for learning from fixed datasets
///
/// Think of it as: "Show me examples of successful games, and I'll learn to
/// generate moves that lead to that level of success."
///
/// Famous for: Berkeley/Meta research simplifying RL to sequence modeling
/// </para>
/// </remarks>
public class DecisionTransformerAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private DecisionTransformerOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private IOptimizer<T, Vector<T>, Vector<T>> _optimizer;

    private NeuralNetwork<T> _transformerNetwork;
    private List<(Vector<T> state, Vector<T> action, T reward, T returnToGo, Vector<T> previousAction)> _trajectoryBuffer;
    private int _updateCount;

    private SequenceContext<T> _currentContext;

    public DecisionTransformerAgent(DecisionTransformerOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            InitialLearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _updateCount = 0;
        _trajectoryBuffer = new List<(Vector<T>, Vector<T>, T, T, Vector<T>)>();
        _currentContext = new SequenceContext<T>();

        // Initialize network directly in constructor
        // Input: concatenated [return_to_go, state, previous_action]
        int inputSize = 1 + _options.StateSize + _options.ActionSize;

        // Create initial architecture for layer generation
        var tempArchitecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: _options.ActionSize
        );

        // Use LayerHelper to create production-ready network layers
        // For DecisionTransformer, use feedforward layers to approximate the transformer
        var layers = LayerHelper<T>.CreateDefaultFeedForwardLayers(
            tempArchitecture,
            hiddenLayerCount: _options.NumLayers,
            hiddenLayerSize: _options.EmbeddingDim
        ).ToList();

        // Override final activation to Tanh for continuous actions
        var lastLayer = layers[layers.Count - 1];
        if (lastLayer is DenseLayer<T> denseLayer)
        {
            int layerInputSize = denseLayer.GetInputShape()[0];
            layers[layers.Count - 1] = new DenseLayer<T>(
                layerInputSize,
                _options.ActionSize,
                (IActivationFunction<T>)new TanhActivation<T>()
            );
        }

        // Create final architecture with the modified layers
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: _options.ActionSize,
            layers: layers
        );
        _transformerNetwork = new NeuralNetwork<T>(architecture, _options.LossFunction);

        // Register network with base class
        Networks.Add(_transformerNetwork);
    }

    /// <summary>
    /// Load offline dataset into the trajectory buffer.
    /// Dataset should contain complete trajectories with computed returns-to-go.
    /// </summary>
    public void LoadOfflineData(List<List<(Vector<T> state, Vector<T> action, T reward)>> trajectories)
    {
        foreach (var trajectory in trajectories)
        {
            // Compute returns-to-go for this trajectory
            T returnToGo = NumOps.Zero;
            var returnsToGo = new List<T>();

            for (int i = trajectory.Count - 1; i >= 0; i--)
            {
                returnToGo = NumOps.Add(trajectory[i].reward, returnToGo);
                returnsToGo.Insert(0, returnToGo);
            }

            // Store trajectory with returns-to-go and previous actions
            for (int i = 0; i < trajectory.Count; i++)
            {
                // Previous action is the action from the previous timestep (zero for first step)
                Vector<T> previousAction = i > 0
                    ? trajectory[i - 1].action
                    : new Vector<T>(_options.ActionSize);

                _trajectoryBuffer.Add((
                    trajectory[i].state,
                    trajectory[i].action,
                    trajectory[i].reward,
                    returnsToGo[i],
                    previousAction
                ));
            }
        }
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        return SelectActionWithReturn(state, NumOps.Zero, training);
    }

    /// <summary>
    /// Select action conditioned on desired return-to-go.
    /// </summary>
    public Vector<T> SelectActionWithReturn(Vector<T> state, T targetReturn, bool training = true)
    {
        // Add to context window
        _currentContext.States.Add(state);
        _currentContext.ReturnsToGo.Add(targetReturn);

        // Keep context within window size
        if (_currentContext.Length > _options.ContextLength)
        {
            _currentContext.States.RemoveAt(0);
            _currentContext.ReturnsToGo.RemoveAt(0);
            if (_currentContext.Actions.Count > 0)
            {
                _currentContext.Actions.RemoveAt(0);
            }
        }

        // Prepare input: [return_to_go, state, previous_action]
        var previousAction = _currentContext.Actions.Count > 0
            ? _currentContext.Actions[_currentContext.Actions.Count - 1]
            : new Vector<T>(_options.ActionSize);  // Zero action for first step

        var input = ConcatenateInputs(targetReturn, state, previousAction);

        // Predict action
        var inputTensor = Tensor<T>.FromVector(input);
        var actionOutputTensor = _transformerNetwork.Predict(inputTensor);
        var actionOutput = actionOutputTensor.ToVector();

        // Store action in context
        _currentContext.Actions.Add(actionOutput);

        return actionOutput;
    }

    private Vector<T> ConcatenateInputs(T returnToGo, Vector<T> state, Vector<T> previousAction)
    {
        var input = new Vector<T>(1 + _options.StateSize + _options.ActionSize);
        input[0] = returnToGo;

        for (int i = 0; i < state.Length; i++)
        {
            input[1 + i] = state[i];
        }

        for (int i = 0; i < previousAction.Length; i++)
        {
            input[1 + _options.StateSize + i] = previousAction[i];
        }

        return input;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // Decision Transformer uses offline data loaded via LoadOfflineData()
        // This method is for interface compliance
    }

    public override T Train()
    {
        if (_trajectoryBuffer.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        T totalLoss = NumOps.Zero;

        // Sample a batch
        var batch = SampleBatch(_options.BatchSize);

        foreach (var (state, targetAction, reward, returnToGo, previousAction) in batch)
        {
            // Use actual previous action from trajectory buffer
            var input = ConcatenateInputs(returnToGo, state, previousAction);

            // Forward pass
            var inputTensor = Tensor<T>.FromVector(input);
            var predictedActionTensor = _transformerNetwork.Predict(inputTensor);
            var predictedAction = predictedActionTensor.ToVector();

            // Compute loss (MSE between predicted and target action)
            T loss = NumOps.Zero;
            for (int i = 0; i < _options.ActionSize; i++)
            {
                var diff = NumOps.Subtract(targetAction[i], predictedAction[i]);
                loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
            }

            totalLoss = NumOps.Add(totalLoss, loss);

            // Backward pass
            var gradient = new Vector<T>(_options.ActionSize);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                gradient[i] = NumOps.Subtract(predictedAction[i], targetAction[i]);
            }

            var gradientTensor = Tensor<T>.FromVector(gradient);
            _transformerNetwork.Backpropagate(gradientTensor);

            var parameters = _transformerNetwork.GetParameters();
            for (int i = 0; i < parameters.Length; i++)
            {
                var update = NumOps.Multiply(LearningRate, gradient[i % gradient.Length]);
                parameters[i] = NumOps.Subtract(parameters[i], update);
            }
            _transformerNetwork.UpdateParameters(parameters);
        }

        _updateCount++;

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private List<(Vector<T> state, Vector<T> action, T reward, T returnToGo, Vector<T> previousAction)> SampleBatch(int batchSize)
    {
        var batch = new List<(Vector<T>, Vector<T>, T, T, Vector<T>)>();

        for (int i = 0; i < batchSize && i < _trajectoryBuffer.Count; i++)
        {
            int idx = Random.Next(_trajectoryBuffer.Count);
            batch.Add(_trajectoryBuffer[idx]);
        }

        return batch;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = NumOps.FromDouble(_updateCount),
            ["buffer_size"] = NumOps.FromDouble(_trajectoryBuffer.Count)
        };
    }

    public override void ResetEpisode()
    {
        _currentContext = new SequenceContext<T>();
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.DecisionTransformer,
        };
    }

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write metadata
        writer.Write(_options.StateSize);
        writer.Write(_options.ActionSize);
        writer.Write(_options.ContextLength);
        writer.Write(_options.EmbeddingDim);
        writer.Write(_options.NumHeads);
        writer.Write(_options.NumLayers);

        // Write training state
        writer.Write(_updateCount);

        // Write transformer network
        var networkBytes = _transformerNetwork.Serialize();
        writer.Write(networkBytes.Length);
        writer.Write(networkBytes);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read and validate metadata
        var stateSize = reader.ReadInt32();
        var actionSize = reader.ReadInt32();
        var contextLength = reader.ReadInt32();
        var embeddingDim = reader.ReadInt32();
        var numHeads = reader.ReadInt32();
        var numLayers = reader.ReadInt32();

        if (stateSize != _options.StateSize || actionSize != _options.ActionSize)
            throw new InvalidOperationException("Serialized network dimensions don't match current options");

        // Read training state
        _updateCount = reader.ReadInt32();

        // Read transformer network
        var networkLength = reader.ReadInt32();
        var networkBytes = reader.ReadBytes(networkLength);
        _transformerNetwork.Deserialize(networkBytes);
    }

    public override Vector<T> GetParameters()
    {
        return _transformerNetwork.GetParameters();
    }

    public override void SetParameters(Vector<T> parameters)
    {
        _transformerNetwork.UpdateParameters(parameters);
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        return new DecisionTransformerAgent<T>(_options, _optimizer);
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var loss = usedLossFunction.CalculateLoss(prediction, target);

        var gradient = usedLossFunction.CalculateDerivative(prediction, target);
        return gradient;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var gradientsTensor = Tensor<T>.FromVector(gradients);
        _transformerNetwork.Backpropagate(gradientsTensor);

        // Optimizer weight update happens via backpropagation in the network
        // The gradients have already been applied during Backpropagate()
    }

    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}

