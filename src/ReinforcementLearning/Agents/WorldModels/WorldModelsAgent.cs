using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
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
public class WorldModelsAgent<T> : ReinforcementLearningAgentBase<T>
{
    private readonly WorldModelsOptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    // V: VAE for spatial compression
    private NeuralNetwork<T> _vaeEncoder;
    private NeuralNetwork<T> _vaeDecoder;

    // M: RNN for temporal modeling
    private NeuralNetwork<T> _rnnNetwork;
    private Vector<T> _rnnHiddenState;

    // C: Controller (simple linear policy)
    private Matrix<T> _controllerWeights;

    private ReplayBuffer<T> _replayBuffer;
    private Random _random;
    private int _updateCount;

    public WorldModelsAgent(WorldModelsOptions<T> options) : base(
        options.ObservationWidth * options.ObservationHeight * options.ObservationChannels,
        options.ActionSize)
    {
        _options = options;
        _numOps = NumericOperations<T>.Instance;
        _random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();
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
                _controllerWeights[i, j] = _numOps.FromDouble((_random.NextDouble() - 0.5) * 0.1);
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
            T sum = _numOps.Zero;
            for (int j = 0; j < controllerInput.Length; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(controllerInput[j], _controllerWeights[j, i]));
            }
            action[i] = MathHelper.Tanh<T>(sum);
        }

        // Update RNN hidden state for next step
        var rnnInput = ConcatenateVectors(ConcatenateVectors(latentMean, action), _rnnHiddenState);
        var rnnOutput = _rnnNetwork.Forward(rnnInput);

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
            return _numOps.Zero;
        }

        T totalLoss = _numOps.Zero;

        // Train VAE
        T vaeLoss = TrainVAE();
        totalLoss = _numOps.Add(totalLoss, vaeLoss);

        // Train RNN
        T rnnLoss = TrainRNN();
        totalLoss = _numOps.Add(totalLoss, rnnLoss);

        // Train Controller (evolution strategy - simplified to gradient-based)
        T controllerLoss = TrainController();
        totalLoss = _numOps.Add(totalLoss, controllerLoss);

        _updateCount++;

        return _numOps.Divide(totalLoss, _numOps.FromDouble(3));
    }

    private T TrainVAE()
    {
        var batch = _replayBuffer.Sample(_options.BatchSize);
        T totalLoss = _numOps.Zero;

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
            T reconLoss = _numOps.Zero;
            for (int i = 0; i < reconstruction.Length; i++)
            {
                var diff = _numOps.Subtract(experience.observation[i], reconstruction[i]);
                reconLoss = _numOps.Add(reconLoss, _numOps.Multiply(diff, diff));
            }

            // KL divergence loss (simplified)
            T klLoss = _numOps.Zero;
            for (int i = 0; i < latentMean.Length; i++)
            {
                var meanSquared = _numOps.Multiply(latentMean[i], latentMean[i]);
                var variance = MathHelper.Exp(latentLogVar[i]);
                klLoss = _numOps.Add(klLoss, _numOps.Add(meanSquared, _numOps.Add(variance, _numOps.Negate(latentLogVar[i]))));
            }
            klLoss = _numOps.Multiply(_numOps.FromDouble(_options.VAEBeta * 0.5), klLoss);

            var loss = _numOps.Add(reconLoss, klLoss);
            totalLoss = _numOps.Add(totalLoss, loss);

            // Backprop
            var gradient = new Vector<T>(reconstruction.Length);
            for (int i = 0; i < gradient.Length; i++)
            {
                gradient[i] = _numOps.Subtract(reconstruction[i], experience.observation[i]);
            }

            _vaeDecoder.Backward(gradient);
            _vaeDecoder.UpdateWeights(_options.LearningRate);

            _vaeEncoder.UpdateWeights(_options.LearningRate);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private T TrainRNN()
    {
        var batch = _replayBuffer.Sample(_options.BatchSize);
        T totalLoss = _numOps.Zero;

        foreach (var experience in batch)
        {
            // Encode current and next observation
            var currentLatent = ExtractMean(_vaeEncoder.Forward(experience.observation));
            var nextLatent = ExtractMean(_vaeEncoder.Forward(experience.nextObservation));

            // Predict next latent using RNN
            var rnnInput = ConcatenateVectors(ConcatenateVectors(currentLatent, experience.action), _rnnHiddenState);
            var rnnOutput = _rnnNetwork.Forward(rnnInput);

            // Extract predicted next latent
            var predictedNextLatent = new Vector<T>(_options.LatentSize);
            for (int i = 0; i < _options.LatentSize; i++)
            {
                predictedNextLatent[i] = rnnOutput[i];
            }

            // Prediction loss
            T loss = _numOps.Zero;
            for (int i = 0; i < _options.LatentSize; i++)
            {
                var diff = _numOps.Subtract(nextLatent[i], predictedNextLatent[i]);
                loss = _numOps.Add(loss, _numOps.Multiply(diff, diff));
            }

            totalLoss = _numOps.Add(totalLoss, loss);

            // Backprop
            var gradient = new Vector<T>(rnnOutput.Length);
            for (int i = 0; i < _options.LatentSize; i++)
            {
                gradient[i] = _numOps.Subtract(predictedNextLatent[i], nextLatent[i]);
            }

            _rnnNetwork.Backward(gradient);
            _rnnNetwork.UpdateWeights(_options.LearningRate);
        }

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private T TrainController()
    {
        // Simplified controller training (in practice, use CMA-ES)
        var batch = _replayBuffer.Sample(Math.Min(10, _replayBuffer.Count));
        T totalReward = _numOps.Zero;

        foreach (var experience in batch)
        {
            totalReward = _numOps.Add(totalReward, experience.reward);
        }

        // Gradient update (simplified)
        T avgReward = _numOps.Divide(totalReward, _numOps.FromDouble(batch.Count));

        // Small random perturbation to controller weights
        for (int i = 0; i < _controllerWeights.Rows; i++)
        {
            for (int j = 0; j < _controllerWeights.Columns; j++)
            {
                var perturbation = _numOps.FromDouble((_random.NextDouble() - 0.5) * 0.01);
                _controllerWeights[i, j] = _numOps.Add(_controllerWeights[i, j], _numOps.Multiply(avgReward, perturbation));
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
            var std = MathHelper.Exp(_numOps.Divide(logVar[i], _numOps.FromDouble(2)));
            var noise = MathHelper.GetNormalRandom<T>(_numOps.Zero, _numOps.One);
            sample[i] = _numOps.Add(mean[i], _numOps.Multiply(std, noise));
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
            ["updates"] = _numOps.FromDouble(_updateCount),
            ["buffer_size"] = _numOps.FromDouble(_replayBuffer.Count)
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
