using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Agents.WorldModels;

/// <summary>
/// World Models agent learning compact representations with VAE and RNN.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// World Models learns compact spatial and temporal representations.
/// Agent trains entirely in the "dream" of its learned world model.
/// </para>
/// <para><b>For Beginners:</b>
/// World Models is inspired by how humans learn: we build mental models
/// of the world, then make decisions based on those models rather than
/// raw sensory input.
///
/// Three components (V-M-C):
/// - **V (VAE)**: Compresses visual observations into compact codes
/// - **M (MDN-RNN)**: Learns temporal dynamics (what happens next)
/// - **C (Controller)**: Simple policy acting in latent space
/// - **Learning in Dreams**: Trains entirely in imagined rollouts
///
/// Process: First compress images (VAE), then learn how compressed
/// images change over time (RNN), finally learn to act based on
/// compressed predictions (controller).
///
/// Famous for: Car racing from pixels with limited environment samples
/// </para>
/// </remarks>
public class WorldModelsAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private WorldModelsOptions<T> _options;

    // V: VAE for spatial compression
    private NeuralNetwork<T> _vaeEncoder;
    private NeuralNetwork<T> _vaeDecoder;

    // M: RNN for temporal modeling
    private NeuralNetwork<T> _rnnNetwork;
    private Vector<T> _rnnHiddenState;

    // C: Controller (simple linear policy)
    private Matrix<T> _controllerWeights;

    private ReplayBuffer<T> _replayBuffer;
    private int _updateCount;

    public WorldModelsAgent(WorldModelsOptions<T> options) : base(
        options.ObservationWidth * options.ObservationHeight * options.ObservationChannels,
        options.ActionSize)
    {
        _options = options;
        _updateCount = 0;

        InitializeNetworks();
        InitializeReplayBuffer();
    }

    private void InitializeNetworks()
    {
        int observationSize = _options.ObservationWidth * _options.ObservationHeight * _options.ObservationChannels;

        // VAE Encoder: observation -> latent code
        _vaeEncoder = CreateEncoderNetwork(observationSize, _options.LatentSize * 2);  // mean + logvar

        // VAE Decoder: latent code -> reconstructed observation
        _vaeDecoder = CreateDecoderNetwork(_options.LatentSize, observationSize);

        // RNN: (latent, action, hidden) -> (next_latent_prediction, next_hidden)
        _rnnNetwork = CreateRNNNetwork();

        // Controller: (latent, hidden) -> action (simple linear)
        int controllerInputSize = _options.LatentSize + _options.RNNHiddenSize;
        _controllerWeights = new Matrix<T>(controllerInputSize, _options.ActionSize);

        // Initialize controller weights
        for (int i = 0; i < _controllerWeights.Rows; i++)
        {
            for (int j = 0; j < _controllerWeights.Columns; j++)
            {
                _controllerWeights[i, j] = NumOps.FromDouble((Random.NextDouble() - 0.5) * 0.1);
            }
        }

        // Initialize RNN hidden state
        _rnnHiddenState = new Vector<T>(_options.RNNHiddenSize);
    }

    private NeuralNetwork<T> CreateEncoderNetwork(int inputSize, int outputSize)
    {
        var network = new NeuralNetwork<T>();
        int previousSize = inputSize;

        // Simple feedforward approximation of convolutional VAE
        foreach (var channels in _options.VAEEncoderChannels)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, channels));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = channels;
        }

        network.AddLayer(new DenseLayer<T>(previousSize, outputSize));

        return network;
    }

    private NeuralNetwork<T> CreateDecoderNetwork(int inputSize, int outputSize)
    {
        var network = new NeuralNetwork<T>();
        int previousSize = inputSize;

        // Reverse of encoder
        var reversedChannels = new List<int>(_options.VAEEncoderChannels);
        reversedChannels.Reverse();

        foreach (var channels in reversedChannels)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, channels));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            previousSize = channels;
        }

        network.AddLayer(new DenseLayer<T>(previousSize, outputSize));
        network.AddLayer(new ActivationLayer<T>(new Sigmoid<T>()));

        return network;
    }

    private NeuralNetwork<T> CreateRNNNetwork()
    {
        // Simplified RNN: (latent, action, hidden) -> (next_latent_mean, next_latent_logvar, next_hidden)
        var network = new NeuralNetwork<T>();
        int inputSize = _options.LatentSize + _options.ActionSize + _options.RNNHiddenSize;

        network.AddLayer(new DenseLayer<T>(inputSize, _options.RNNHiddenSize));
        network.AddLayer(new ActivationLayer<T>(new Tanh<T>()));
        network.AddLayer(new DenseLayer<T>(_options.RNNHiddenSize, _options.LatentSize * _options.NumMixtures + _options.RNNHiddenSize));

        return network;
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new ReplayBuffer<T>(_options.ReplayBufferSize);
    }

    public override Vector<T> SelectAction(Vector<T> observation, bool training = true)
    {
        // Encode observation to latent code
        var encoderOutput = _vaeEncoder.Forward(observation);
        var latentMean = ExtractMean(encoderOutput);

        // Concatenate latent and RNN hidden state
        var controllerInput = ConcatenateVectors(latentMean, _rnnHiddenState);

        // Compute action from controller
        var action = new Vector<T>(_options.ActionSize);
        for (int i = 0; i < _options.ActionSize; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < controllerInput.Length; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(controllerInput[j], _controllerWeights[j, i]));
            }
            action[i] = MathHelper.Tanh<T>(sum);
        }

        // Update RNN hidden state for next step
        var rnnInput = ConcatenateVectors(ConcatenateVectors(latentMean, action), _rnnHiddenState);
        var rnnOutput = _rnnNetwork.Predict(rnnInput);

        // Extract new hidden state
        int hiddenOffset = _options.LatentSize * _options.NumMixtures;
        for (int i = 0; i < _options.RNNHiddenSize; i++)
        {
            _rnnHiddenState[i] = rnnOutput[hiddenOffset + i];
        }

        return action;
    }

    public override void StoreExperience(Vector<T> observation, Vector<T> action, T reward, Vector<T> nextObservation, bool done)
    {
        _replayBuffer.Add(observation, action, reward, nextObservation, done);

        if (done)
        {
            // Reset RNN hidden state on episode end
            _rnnHiddenState = new Vector<T>(_options.RNNHiddenSize);
        }
    }

    public override T Train()
    {
        if (_replayBuffer.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        T totalLoss = NumOps.Zero;

        // Train VAE
        T vaeLoss = TrainVAE();
        totalLoss = NumOps.Add(totalLoss, vaeLoss);

        // Train RNN
        T rnnLoss = TrainRNN();
        totalLoss = NumOps.Add(totalLoss, rnnLoss);

        // Train Controller (evolution strategy - simplified to gradient-based)
        T controllerLoss = TrainController();
        totalLoss = NumOps.Add(totalLoss, controllerLoss);

        _updateCount++;

        return NumOps.Divide(totalLoss, NumOps.FromDouble(3));
    }

    private T TrainVAE()
    {
        var batch = _replayBuffer.Sample(_options.BatchSize);
        T totalLoss = NumOps.Zero;

        foreach (var experience in batch)
        {
            // Encode
            var encoderOutput = _vaeEncoder.Forward(experience.observation);
            var latentMean = ExtractMean(encoderOutput);
            var latentLogVar = ExtractLogVar(encoderOutput);

            // Sample latent code
            var latentSample = SampleLatent(latentMean, latentLogVar);

            // Decode
            var reconstruction = _vaeDecoder.Forward(latentSample);

            // Reconstruction loss (MSE)
            T reconLoss = NumOps.Zero;
            for (int i = 0; i < reconstruction.Length; i++)
            {
                var diff = NumOps.Subtract(experience.observation[i], reconstruction[i]);
                reconLoss = NumOps.Add(reconLoss, NumOps.Multiply(diff, diff));
            }

            // KL divergence loss (simplified)
            T klLoss = NumOps.Zero;
            for (int i = 0; i < latentMean.Length; i++)
            {
                var meanSquared = NumOps.Multiply(latentMean[i], latentMean[i]);
                var variance = MathHelper.Exp(latentLogVar[i]);
                klLoss = NumOps.Add(klLoss, NumOps.Add(meanSquared, NumOps.Add(variance, NumOps.Negate(latentLogVar[i]))));
            }
            klLoss = NumOps.Multiply(NumOps.FromDouble(_options.VAEBeta * 0.5), klLoss);

            var loss = NumOps.Add(reconLoss, klLoss);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backprop
            var gradient = new Vector<T>(reconstruction.Length);
            for (int i = 0; i < gradient.Length; i++)
            {
                gradient[i] = NumOps.Subtract(reconstruction[i], experience.observation[i]);
            }

            _vaeDecoder.Backward(gradient);
            _vaeDecoder.UpdateWeights(_options.LearningRate);

            _vaeEncoder.UpdateWeights(_options.LearningRate);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private T TrainRNN()
    {
        var batch = _replayBuffer.Sample(_options.BatchSize);
        T totalLoss = NumOps.Zero;

        foreach (var experience in batch)
        {
            // Encode current and next observation
            var currentLatent = ExtractMean(_vaeEncoder.Forward(experience.observation));
            var nextLatent = ExtractMean(_vaeEncoder.Forward(experience.nextObservation));

            // Predict next latent using RNN
            var rnnInput = ConcatenateVectors(ConcatenateVectors(currentLatent, experience.action), _rnnHiddenState);
            var rnnOutput = _rnnNetwork.Predict(rnnInput);

            // Extract predicted next latent
            var predictedNextLatent = new Vector<T>(_options.LatentSize);
            for (int i = 0; i < _options.LatentSize; i++)
            {
                predictedNextLatent[i] = rnnOutput[i];
            }

            // Prediction loss
            T loss = NumOps.Zero;
            for (int i = 0; i < _options.LatentSize; i++)
            {
                var diff = NumOps.Subtract(nextLatent[i], predictedNextLatent[i]);
                loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
            }

            totalLoss = NumOps.Add(totalLoss, loss);

            // Backprop
            var gradient = new Vector<T>(rnnOutput.Length);
            for (int i = 0; i < _options.LatentSize; i++)
            {
                gradient[i] = NumOps.Subtract(predictedNextLatent[i], nextLatent[i]);
            }

            _rnnNetwork.Backward(gradient);
            _rnnNetwork.UpdateWeights(_options.LearningRate);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private T TrainController()
    {
        // Simplified controller training (in practice, use CMA-ES)
        var batch = _replayBuffer.Sample(Math.Min(10, _replayBuffer.Count));
        T totalReward = NumOps.Zero;

        foreach (var experience in batch)
        {
            totalReward = NumOps.Add(totalReward, experience.reward);
        }

        // Gradient update (simplified)
        T avgReward = NumOps.Divide(totalReward, NumOps.FromDouble(batch.Count));

        // Small random perturbation to controller weights
        for (int i = 0; i < _controllerWeights.Rows; i++)
        {
            for (int j = 0; j < _controllerWeights.Columns; j++)
            {
                var perturbation = NumOps.FromDouble((Random.NextDouble() - 0.5) * 0.01);
                _controllerWeights[i, j] = NumOps.Add(_controllerWeights[i, j], NumOps.Multiply(avgReward, perturbation));
            }
        }

        return avgReward;
    }

    private Vector<T> ExtractMean(Vector<T> encoderOutput)
    {
        var mean = new Vector<T>(_options.LatentSize);
        for (int i = 0; i < _options.LatentSize; i++)
        {
            mean[i] = encoderOutput[i];
        }
        return mean;
    }

    private Vector<T> ExtractLogVar(Vector<T> encoderOutput)
    {
        var logVar = new Vector<T>(_options.LatentSize);
        for (int i = 0; i < _options.LatentSize; i++)
        {
            logVar[i] = encoderOutput[_options.LatentSize + i];
        }
        return logVar;
    }

    private Vector<T> SampleLatent(Vector<T> mean, Vector<T> logVar)
    {
        var sample = new Vector<T>(_options.LatentSize);
        for (int i = 0; i < _options.LatentSize; i++)
        {
            var std = MathHelper.Exp(NumOps.Divide(logVar[i], NumOps.FromDouble(2)));
            var noise = MathHelper.GetNormalRandom<T>(NumOps.Zero, NumOps.One);
            sample[i] = NumOps.Add(mean[i], NumOps.Multiply(std, noise));
        }
        return sample;
    }

    private Vector<T> ConcatenateVectors(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(a.Length + b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = a[i];
        }
        for (int i = 0; i < b.Length; i++)
        {
            result[a.Length + i] = b[i];
        }
        return result;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = NumOps.FromDouble(_updateCount),
            ["buffer_size"] = NumOps.FromDouble(_replayBuffer.Count)
        };
    }

    public override void ResetEpisode()
    {
        _rnnHiddenState = new Vector<T>(_options.RNNHiddenSize);
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
            ModelType = "WorldModels",
            InputSize = _options.ObservationWidth * _options.ObservationHeight * _options.ObservationChannels,
            OutputSize = _options.ActionSize,
            ParameterCount = ParameterCount
        };
    }

    public override int FeatureCount => _options.ObservationWidth * _options.ObservationHeight * _options.ObservationChannels;

    public override byte[] Serialize()
    {
        throw new NotImplementedException("WorldModels serialization not yet implemented");
    }

    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("WorldModels deserialization not yet implemented");
    }

    public override Matrix<T> GetParameters()
    {
        var allParams = new List<T>();

        foreach (var network in Networks)
        {
            var netParams = network.GetParameters();
            for (int i = 0; i < netParams.Length; i++)
            {
                allParams.Add(netParams[i]);
            }
        }

        var paramVector = new Vector<T>(allParams.Count);
        for (int i = 0; i < allParams.Count; i++)
        {
            paramVector[i] = allParams[i];
        }

        return new Matrix<T>(new[] { paramVector });
    }

    public override void SetParameters(Matrix<T> parameters)
    {
        int offset = 0;

        foreach (var network in Networks)
        {
            int paramCount = network.ParameterCount;
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[0, offset + i];
            }
            network.UpdateParameters(netParams);
            offset += paramCount;
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        return new WorldModelsAgent<T>(_options, _optimizer);
    }

    public override (Matrix<T> Gradients, T Loss) ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var loss = usedLossFunction.CalculateLoss(new Matrix<T>(new[] { prediction }), new Matrix<T>(new[] { target }));

        var gradient = usedLossFunction.CalculateDerivative(new Matrix<T>(new[] { prediction }), new Matrix<T>(new[] { target }));
        return (gradient, loss);
    }

    public override void ApplyGradients(Matrix<T> gradients, T learningRate)
    {
        if (Networks.Count > 0)
        {
            Networks[0].Backward(new Vector<T>(gradients.GetRow(0)));
            Networks[0].UpdateWeights(learningRate);
        }
    }

    public override void Save(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void Load(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
