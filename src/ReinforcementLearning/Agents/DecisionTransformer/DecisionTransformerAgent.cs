using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
using AiDotNet.Helpers;

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
public class DecisionTransformerAgent<T> : ReinforcementLearningAgentBase<T>
{
    private readonly DecisionTransformerOptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    private NeuralNetwork<T> _transformerNetwork;
    private List<(Vector<T> state, Vector<T> action, T reward, T returnToGo)> _trajectoryBuffer;
    private Random _random;
    private int _updateCount;

    // Context window for sequence modeling
    private class SequenceContext
    {
        public List<Vector<T>> States { get; set; } = new();
        public List<Vector<T>> Actions { get; set; } = new();
        public List<T> ReturnsToGo { get; set; } = new();
        public int Length => States.Count;
    }

    private SequenceContext _currentContext;

    public DecisionTransformerAgent(DecisionTransformerOptions<T> options) : base(options.StateSize, options.ActionSize)
    {
        _options = options;
        _numOps = NumericOperations<T>.Instance;
        _random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();
        _updateCount = 0;
        _trajectoryBuffer = new List<(Vector<T>, Vector<T>, T, T)>();
        _currentContext = new SequenceContext();

        InitializeNetwork();
    }

    private void InitializeNetwork()
    {
        // Simplified transformer: state/action/return embeddings + feedforward layers
        _transformerNetwork = new NeuralNetwork<T>();

        // Input: concatenated [return_to_go, state, previous_action]
        int inputSize = 1 + _options.StateSize + _options.ActionSize;
        int previousSize = inputSize;

        // Embedding layer
        _transformerNetwork.AddLayer(new DenseLayer<T>(previousSize, _options.EmbeddingDim));
        _transformerNetwork.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
        previousSize = _options.EmbeddingDim;

        // Simplified transformer blocks (using feedforward layers to approximate attention)
        for (int layer = 0; layer < _options.NumLayers; layer++)
        {
            // Attention approximation via dense layers
            _transformerNetwork.AddLayer(new DenseLayer<T>(previousSize, _options.EmbeddingDim * 2));
            _transformerNetwork.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            _transformerNetwork.AddLayer(new DenseLayer<T>(_options.EmbeddingDim * 2, _options.EmbeddingDim));
            _transformerNetwork.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
        }

        // Output head for action prediction
        _transformerNetwork.AddLayer(new DenseLayer<T>(_options.EmbeddingDim, _options.ActionSize));
        _transformerNetwork.AddLayer(new ActivationLayer<T>(new Tanh<T>()));
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
            T returnToGo = _numOps.Zero;
            var returnsToGo = new List<T>();

            for (int i = trajectory.Count - 1; i >= 0; i--)
            {
                returnToGo = _numOps.Add(trajectory[i].reward, returnToGo);
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
        return SelectActionWithReturn(state, _numOps.Zero, training);
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
        var actionOutput = _transformerNetwork.Forward(input);

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
            return _numOps.Zero;
        }

        T totalLoss = _numOps.Zero;

        // Sample a batch
        var batch = SampleBatch(_options.BatchSize);

        foreach (var (state, targetAction, reward, returnToGo) in batch)
        {
            // For simplicity, use zero previous action
            var previousAction = new Vector<T>(_options.ActionSize);
            var input = ConcatenateInputs(returnToGo, state, previousAction);

            // Forward pass
            var predictedAction = _transformerNetwork.Forward(input);

            // Compute loss (MSE between predicted and target action)
            T loss = _numOps.Zero;
            for (int i = 0; i < _options.ActionSize; i++)
            {
                var diff = _numOps.Subtract(targetAction[i], predictedAction[i]);
                loss = _numOps.Add(loss, _numOps.Multiply(diff, diff));
            }

            totalLoss = _numOps.Add(totalLoss, loss);

            // Backward pass
            var gradient = new Vector<T>(_options.ActionSize);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                gradient[i] = _numOps.Subtract(predictedAction[i], targetAction[i]);
            }

            _transformerNetwork.Backward(gradient);
            _transformerNetwork.UpdateWeights(_options.LearningRate);
        }

        _updateCount++;

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private List<(Vector<T> state, Vector<T> action, T reward, T returnToGo)> SampleBatch(int batchSize)
    {
        var batch = new List<(Vector<T>, Vector<T>, T, T)>();

        for (int i = 0; i < batchSize && i < _trajectoryBuffer.Count; i++)
        {
            int idx = _random.Next(_trajectoryBuffer.Count);
            batch.Add(_trajectoryBuffer[idx]);
        }

        return batch;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = _numOps.FromDouble(_updateCount),
            ["buffer_size"] = _numOps.FromDouble(_trajectoryBuffer.Count)
        };
    }

    public override void ResetEpisode()
    {
        _currentContext = new SequenceContext();
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    public override Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public override Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }
}
