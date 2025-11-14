using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Optimizers;

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
    private IOptimizer<T, Vector<T>, Vector<T>> _optimizer;

    private INeuralNetwork<T> _transformerNetwork;
    private List<(Vector<T> state, Vector<T> action, T reward, T returnToGo)> _trajectoryBuffer;
    private int _updateCount;

    private SequenceContext<T> _currentContext;

    public DecisionTransformerAgent(DecisionTransformerOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            LearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _updateCount = 0;
        _trajectoryBuffer = new List<(Vector<T>, Vector<T>, T, T)>();
        _currentContext = new SequenceContext<T>();

        InitializeNetwork();
    }

    private void InitializeNetwork()
    {
        // Input: concatenated [return_to_go, state, previous_action]
        int inputSize = 1 + _options.StateSize + _options.ActionSize;

        var architecture = new NeuralNetworkArchitecture<T>
        {
            TaskType = TaskType.Regression
        };

        // Use LayerHelper to create production-ready network layers
        // For DecisionTransformer, use feedforward layers to approximate the transformer
        var layers = LayerHelper<T>.CreateDefaultFeedForwardLayers(
            architecture,
            hiddenLayerCount: _options.NumLayers,
            hiddenLayerSize: _options.EmbeddingDim
        ).ToList();

        // Override final activation to Tanh for continuous actions
        var lastLayer = layers[layers.Count - 1];
        if (lastLayer is DenseLayer<T> denseLayer)
        {
            layers[layers.Count - 1] = new DenseLayer<T>(
                denseLayer.GetWeights().Rows,
                _options.ActionSize,
                new TanhActivation<T>()
            );
        }

        architecture.Layers = layers;
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

            // Store trajectory with returns-to-go
            for (int i = 0; i < trajectory.Count; i++)
            {
                _trajectoryBuffer.Add((
                    trajectory[i].state,
                    trajectory[i].action,
                    trajectory[i].reward,
                    returnsToGo[i]
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
        var actionOutput = _transformerNetwork.Predict(input);

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

        foreach (var (state, targetAction, reward, returnToGo) in batch)
        {
            // For simplicity, use zero previous action
            var previousAction = new Vector<T>(_options.ActionSize);
            var input = ConcatenateInputs(returnToGo, state, previousAction);

            // Forward pass
            var predictedAction = _transformerNetwork.Predict(input);

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

            _transformerNetwork.Backward(gradient);
            _transformerNetwork.UpdateWeights(_options.LearningRate);
        }

        _updateCount++;

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private List<(Vector<T> state, Vector<T> action, T reward, T returnToGo)> SampleBatch(int batchSize)
    {
        var batch = new List<(Vector<T>, Vector<T>, T, T)>();

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
            ModelType = "DecisionTransformer",
        };
    }

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        throw new NotImplementedException("DecisionTransformer serialization not yet implemented");
    }

    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("DecisionTransformer deserialization not yet implemented");
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

        var gradient = usedLossFunction.ComputeGradient(prediction, target);
        return gradient;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _transformerNetwork.Backward(gradients);
        _transformerNetwork.UpdateWeights(learningRate);
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

