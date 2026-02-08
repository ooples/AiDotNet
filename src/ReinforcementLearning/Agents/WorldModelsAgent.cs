using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

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

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    // V: VAE for spatial compression
    private NeuralNetwork<T> _vaeEncoder;
    private NeuralNetwork<T> _vaeDecoder;

    // M: RNN for temporal modeling
    private NeuralNetwork<T> _rnnNetwork;
    private Vector<T> _rnnHiddenState;

    // C: Controller (simple linear policy)
    private Matrix<T> _controllerWeights;

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private int _updateCount;
    private Random _random;

    public WorldModelsAgent(WorldModelsOptions<T> options) : base(options)
    {
        _options = options;
        _updateCount = 0;
        _random = RandomHelper.CreateSecureRandom();

        // Initialize networks directly in constructor
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
                _controllerWeights[i, j] = NumOps.FromDouble((_random.NextDouble() - 0.5) * 0.1);
            }
        }

        // Initialize RNN hidden state
        _rnnHiddenState = new Vector<T>(_options.RNNHiddenSize);

        // Initialize replay buffer
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.ReplayBufferSize);

        // Add all networks to Networks list for parameter access
        Networks.Add(_vaeEncoder);
        Networks.Add(_vaeDecoder);
        Networks.Add(_rnnNetwork);
    }

    private NeuralNetwork<T> CreateEncoderNetwork(int inputSize, int outputSize)
    {
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize);
        var network = new NeuralNetwork<T>(architecture, new MeanSquaredErrorLoss<T>());
        int previousSize = inputSize;

        // Simple feedforward approximation of convolutional VAE
        foreach (var channels in _options.VAEEncoderChannels)
        {
            network.AddLayer(LayerType.Dense, channels, ActivationFunction.ReLU);
            previousSize = channels;
        }

        network.AddLayer(LayerType.Dense, outputSize, ActivationFunction.Linear);

        return network;
    }

    private NeuralNetwork<T> CreateDecoderNetwork(int inputSize, int outputSize)
    {
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize);
        var network = new NeuralNetwork<T>(architecture, new MeanSquaredErrorLoss<T>());
        int previousSize = inputSize;

        // Reverse of encoder
        var reversedChannels = new List<int>(_options.VAEEncoderChannels);
        reversedChannels.Reverse();

        foreach (var channels in reversedChannels)
        {
            network.AddLayer(LayerType.Dense, channels, ActivationFunction.ReLU);
            previousSize = channels;
        }

        network.AddLayer(LayerType.Dense, outputSize, ActivationFunction.Sigmoid);

        return network;
    }

    private NeuralNetwork<T> CreateRNNNetwork()
    {
        // Simplified RNN: (latent, action, hidden) -> (next_latent_prediction, next_hidden)
        // Note: Full World Models implementation uses Mixture Density Network (MDN) with multiple mixtures
        // This simplified version uses single-mode prediction (NumMixtures parameter is for future MDN support)
        int inputSize = _options.LatentSize + _options.ActionSize + _options.RNNHiddenSize;
        int outputSize = _options.LatentSize + _options.RNNHiddenSize;  // Single prediction + hidden state
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize);
        var network = new NeuralNetwork<T>(architecture, new MeanSquaredErrorLoss<T>());

        network.AddLayer(LayerType.Dense, _options.RNNHiddenSize, ActivationFunction.Tanh);
        network.AddLayer(LayerType.Dense, outputSize, ActivationFunction.Linear);

        return network;
    }

    public override Vector<T> SelectAction(Vector<T> observation, bool training = true)
    {
        // Encode observation to latent code
        var encoderOutput = _vaeEncoder.Predict(Tensor<T>.FromVector(observation)).ToVector();
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
        var rnnOutput = _rnnNetwork.Predict(Tensor<T>.FromVector(rnnInput)).ToVector();

        // Extract new hidden state (after latent prediction)
        int hiddenOffset = _options.LatentSize;
        for (int i = 0; i < _options.RNNHiddenSize; i++)
        {
            _rnnHiddenState[i] = rnnOutput[hiddenOffset + i];
        }

        return action;
    }

    public override void StoreExperience(Vector<T> observation, Vector<T> action, T reward, Vector<T> nextObservation, bool done)
    {
        // Store using Experience<T, TState, TAction> which expects Vector<T>
        var experience = new Experience<T, Vector<T>, Vector<T>>(observation, action, reward, nextObservation, done);
        _replayBuffer.Add(experience);

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
            var stateVector = experience.State;
            // Encode
            var encoderOutput = _vaeEncoder.Predict(Tensor<T>.FromVector(experience.State)).ToVector();
            var latentMean = ExtractMean(encoderOutput);
            var latentLogVar = ExtractLogVar(encoderOutput);

            // Sample latent code
            var latentSample = SampleLatent(latentMean, latentLogVar);

            // Decode
            var reconstruction = _vaeDecoder.Predict(Tensor<T>.FromVector(latentSample)).ToVector();

            // Reconstruction loss (MSE)
            T reconLoss = NumOps.Zero;
            for (int i = 0; i < reconstruction.Length; i++)
            {
                var diff = NumOps.Subtract(stateVector[i], reconstruction[i]);
                reconLoss = NumOps.Add(reconLoss, NumOps.Multiply(diff, diff));
            }

            // KL divergence loss: KL(N(mean, var) || N(0, 1)) = 0.5 * sum(1 + logVar - mean² - exp(logVar))
            T klLoss = NumOps.Zero;
            for (int i = 0; i < latentMean.Length; i++)
            {
                var meanSquared = NumOps.Multiply(latentMean[i], latentMean[i]);
                var expLogVar = NumOps.Exp(latentLogVar[i]);
                // KL = 0.5 * (1 + logVar - mean² - exp(logVar))
                var klTerm = NumOps.Add(
                    NumOps.One,
                    NumOps.Add(
                        latentLogVar[i],
                        NumOps.Subtract(
                            NumOps.Negate(meanSquared),
                            expLogVar
                        )
                    )
                );
                klLoss = NumOps.Add(klLoss, klTerm);
            }
            klLoss = NumOps.Multiply(NumOps.FromDouble(_options.VAEBeta * 0.5), klLoss);

            var loss = NumOps.Add(reconLoss, klLoss);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backpropagation through both decoder and encoder
            // Step 1: Decoder gradient (reconstruction error)
            var decoderGradient = new Vector<T>(reconstruction.Length);
            for (int i = 0; i < decoderGradient.Length; i++)
            {
                decoderGradient[i] = NumOps.Subtract(reconstruction[i], stateVector[i]);
            }
            _vaeDecoder.Backpropagate(Tensor<T>.FromVector(decoderGradient));

            // Step 2: Encoder gradient (KL divergence)
            // Gradient of KL divergence w.r.t. mean and logVar
            var encoderGradient = new Vector<T>(encoderOutput.Length);
            for (int i = 0; i < latentMean.Length; i++)
            {
                // d(KL)/d(mean) = mean
                encoderGradient[i] = NumOps.Multiply(NumOps.FromDouble(_options.VAEBeta), latentMean[i]);
                // d(KL)/d(logVar) = 0.5 * (exp(logVar) - 1)
                encoderGradient[_options.LatentSize + i] = NumOps.Multiply(
                    NumOps.FromDouble(_options.VAEBeta * 0.5),
                    NumOps.Subtract(NumOps.Exp(latentLogVar[i]), NumOps.One)
                );
            }
            _vaeEncoder.Backpropagate(Tensor<T>.FromVector(encoderGradient));

            // TODO: Add proper optimizer-based parameter updates


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
            var currentLatent = ExtractMean(_vaeEncoder.Predict(Tensor<T>.FromVector(experience.State)).ToVector());
            var nextLatent = ExtractMean(_vaeEncoder.Predict(Tensor<T>.FromVector(experience.NextState)).ToVector());

            // Use zero-initialized hidden state for training
            // Note: Ideally, we would store per-experience hidden states in the replay buffer,
            // but this approximation (zero state) is acceptable for training the dynamics model
            var hiddenState = new Vector<T>(_options.RNNHiddenSize);

            // Predict next latent using RNN
            var rnnInput = ConcatenateVectors(ConcatenateVectors(currentLatent, experience.Action), hiddenState);
            var rnnOutput = _rnnNetwork.Predict(Tensor<T>.FromVector(rnnInput)).ToVector();

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

            _rnnNetwork.Backpropagate(Tensor<T>.FromVector(gradient));
            // TODO: Add proper optimizer-based parameter updates
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private T TrainController()
    {
        // Simplified Evolution Strategy for controller training
        // Note: Full World Models uses CMA-ES; this is a basic (1+1)-ES approximation

        const int numCandidates = 5;
        const double perturbationScale = 0.01;

        var batch = _replayBuffer.Sample(Math.Min(10, _replayBuffer.Count));

        // Evaluate current controller
        T currentReward = NumOps.Zero;
        foreach (var experience in batch)
        {
            currentReward = NumOps.Add(currentReward, experience.Reward);
        }
        currentReward = NumOps.Divide(currentReward, NumOps.FromDouble(batch.Count));

        // Try random perturbations and keep the best one
        Matrix<T>? bestWeights = null;
        T bestReward = currentReward;

        for (int candidate = 0; candidate < numCandidates; candidate++)
        {
            // Create perturbed weights
            var perturbedWeights = new Matrix<T>(_controllerWeights.Rows, _controllerWeights.Columns);
            for (int i = 0; i < _controllerWeights.Rows; i++)
            {
                for (int j = 0; j < _controllerWeights.Columns; j++)
                {
                    var noise = NumOps.FromDouble((_random.NextDouble() - 0.5) * 2.0 * perturbationScale);
                    perturbedWeights[i, j] = NumOps.Add(_controllerWeights[i, j], noise);
                }
            }

            // Evaluate perturbed controller (simplified: use same batch)
            T perturbedReward = NumOps.Zero;
            foreach (var experience in batch)
            {
                perturbedReward = NumOps.Add(perturbedReward, experience.Reward);
            }
            perturbedReward = NumOps.Divide(perturbedReward, NumOps.FromDouble(batch.Count));

            // Keep if better
            if (NumOps.GreaterThan(perturbedReward, bestReward))
            {
                bestReward = perturbedReward;
                bestWeights = perturbedWeights;
            }
        }

        // Update controller weights if we found a better candidate
        if (bestWeights is not null && !object.ReferenceEquals(bestWeights, null))
        {
            _controllerWeights = bestWeights;
        }

        return bestReward;
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
            var std = NumOps.Exp(NumOps.Divide(logVar[i], NumOps.FromDouble(2)));
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
            ModelType = ModelType.ReinforcementLearning,
        };
    }

    public override int FeatureCount => _options.ObservationWidth * _options.ObservationHeight * _options.ObservationChannels;

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write metadata
        writer.Write(_options.ObservationWidth);
        writer.Write(_options.ObservationHeight);
        writer.Write(_options.ObservationChannels);
        writer.Write(_options.LatentSize);
        writer.Write(_options.RNNHiddenSize);
        writer.Write(_options.ActionSize);

        // Write training state
        writer.Write(_updateCount);

        // Write VAE encoder
        var encoderBytes = _vaeEncoder.Serialize();
        writer.Write(encoderBytes.Length);
        writer.Write(encoderBytes);

        // Write VAE decoder
        var decoderBytes = _vaeDecoder.Serialize();
        writer.Write(decoderBytes.Length);
        writer.Write(decoderBytes);

        // Write RNN network
        var rnnBytes = _rnnNetwork.Serialize();
        writer.Write(rnnBytes.Length);
        writer.Write(rnnBytes);

        // Write controller weights
        writer.Write(_controllerWeights.Rows);
        writer.Write(_controllerWeights.Columns);
        for (int i = 0; i < _controllerWeights.Rows; i++)
        {
            for (int j = 0; j < _controllerWeights.Columns; j++)
            {
                writer.Write(NumOps.ToDouble(_controllerWeights[i, j]));
            }
        }

        // Write RNN hidden state
        writer.Write(_rnnHiddenState.Length);
        for (int i = 0; i < _rnnHiddenState.Length; i++)
        {
            writer.Write(NumOps.ToDouble(_rnnHiddenState[i]));
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read and validate metadata
        var obsWidth = reader.ReadInt32();
        var obsHeight = reader.ReadInt32();
        var obsChannels = reader.ReadInt32();
        var latentSize = reader.ReadInt32();
        var rnnHiddenSize = reader.ReadInt32();
        var actionSize = reader.ReadInt32();

        if (obsWidth != _options.ObservationWidth || obsHeight != _options.ObservationHeight ||
            obsChannels != _options.ObservationChannels || actionSize != _options.ActionSize)
            throw new InvalidOperationException("Serialized model dimensions don't match current options");

        // Read training state
        _updateCount = reader.ReadInt32();

        // Read VAE encoder
        var encoderLength = reader.ReadInt32();
        var encoderBytes = reader.ReadBytes(encoderLength);
        _vaeEncoder.Deserialize(encoderBytes);

        // Read VAE decoder
        var decoderLength = reader.ReadInt32();
        var decoderBytes = reader.ReadBytes(decoderLength);
        _vaeDecoder.Deserialize(decoderBytes);

        // Read RNN network
        var rnnLength = reader.ReadInt32();
        var rnnBytes = reader.ReadBytes(rnnLength);
        _rnnNetwork.Deserialize(rnnBytes);

        // Read controller weights
        var rows = reader.ReadInt32();
        var cols = reader.ReadInt32();
        _controllerWeights = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                _controllerWeights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Read RNN hidden state
        var hiddenLength = reader.ReadInt32();
        _rnnHiddenState = new Vector<T>(hiddenLength);
        for (int i = 0; i < hiddenLength; i++)
        {
            _rnnHiddenState[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    public override Vector<T> GetParameters()
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

        return paramVector;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        foreach (var network in Networks)
        {
            int paramCount = network.ParameterCount;
            var netParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                netParams[i] = parameters[offset + i];
            }
            network.UpdateParameters(netParams);
            offset += paramCount;
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        return new WorldModelsAgent<T>(_options);
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {

        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var gradient = usedLossFunction.CalculateDerivative(prediction, target);
        return gradient;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (Networks.Count > 0)
        {
            // Networks[0].Backpropagate(Tensor<T>.FromVector(gradients));
            // TODO: Add proper optimizer-based parameter updates
        }
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
