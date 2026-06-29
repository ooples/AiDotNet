using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
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
/// <example>
/// <code>
/// // Create a World Models agent that learns in its own dreams
/// var options = new WorldModelsOptions&lt;double&gt; { StateSize = 64, ActionSize = 3, LatentSize = 32 };
/// var agent = new WorldModelsAgent&lt;double&gt;(options);
///
/// // Select an action from the compressed latent representation
/// var state = new Vector&lt;double&gt;(new double[64]);
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("World Models",
    "https://arxiv.org/abs/1803.10122",
    Year = 2018,
    Authors = "Ha, D. & Schmidhuber, J.")]
public class WorldModelsAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private WorldModelsOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    // V: VAE for spatial compression
    private INeuralNetwork<T> _vaeEncoder;
    private INeuralNetwork<T> _vaeDecoder;

    // M: RNN for temporal modeling
    private INeuralNetwork<T> _rnnNetwork;
    private Vector<T> _rnnHiddenState;

    // C: Controller (simple linear policy)
    private Matrix<T> _controllerWeights;

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private int _updateCount;
    private Random _random;

    // Deterministic base seeds for each sub-network's lazy weight init, so a
    // Clone()'d agent reproduces the original's policy (the networks resolve
    // their weights on first forward and would otherwise draw from the shared
    // non-deterministic RNG). Distinct bases keep the three networks' inits
    // independent.
    private const int EncoderSeedBase = 7001;
    private const int DecoderSeedBase = 7101;
    private const int RNNSeedBase = 7201;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public WorldModelsAgent()
        : this(new WorldModelsOptions<T> { ActionSize = 2 })
    {
    }

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
        // Build the layer list explicitly and hand it to the architecture so
        // NeuralNetwork.InitializeLayers uses EXACTLY these layers. Passing an
        // architecture with no layers makes InitializeLayers fall back to
        // CreateDefaultNeuralNetworkLayers, which sizes a hidden layer to ~2x
        // the (image-flattened) input — for a 64x64x3 = 12,288-wide observation
        // that alone is a single 12,288x24,576 weight (~2.4 GB at FP64). Those
        // unintended default layers then sit in front of the agent's own
        // AddLayer-appended layers, producing a malformed, multi-GB network that
        // trips weight-streaming's per-tensor byte cap on the first forward.
        var layers = new List<ILayer<T>>();
        foreach (var channels in _options.VAEEncoderChannels)
        {
            layers.Add(new DenseLayer<T>(channels, (IActivationFunction<T>)new ReLUActivation<T>()));
        }
        layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        // Pin a deterministic per-layer RandomSeed so the lazy weight
        // initialization (DenseLayer resolves its weights on first forward) is
        // reproducible. NeuralNetwork.InitializeLayers does NOT wire explicitly
        // supplied custom layers, so without this they would fall back to the
        // shared non-deterministic RNG and a Clone()'d agent would initialize a
        // DIFFERENT policy than the original (Clone_ShouldProduceSamePolicy).
        AssignDeterministicSeeds(layers, EncoderSeedBase);

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);
        return new NeuralNetwork<T>(architecture, lossFunction: new MeanSquaredErrorLoss<T>());
    }

    /// <summary>
    /// Assigns a deterministic, reproducible <see cref="LayerBase{T}.RandomSeed"/>
    /// to each layer (base seed + index) so lazy weight initialization is
    /// identical across agent instances built from the same options — required
    /// for Clone() to reproduce the original's policy.
    /// </summary>
    private static void AssignDeterministicSeeds(List<ILayer<T>> layers, int baseSeed)
    {
        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is LayerBase<T> layer)
            {
                layer.RandomSeed = baseSeed + i;
            }
        }
    }

    private NeuralNetwork<T> CreateDecoderNetwork(int inputSize, int outputSize)
    {
        // Explicit layers (mirror of the encoder, reversed) — see
        // CreateEncoderNetwork for why we must not let the architecture
        // auto-generate default layers.
        var reversedChannels = new List<int>(_options.VAEEncoderChannels);
        reversedChannels.Reverse();

        var layers = new List<ILayer<T>>();
        foreach (var channels in reversedChannels)
        {
            layers.Add(new DenseLayer<T>(channels, (IActivationFunction<T>)new ReLUActivation<T>()));
        }
        layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new SigmoidActivation<T>()));

        AssignDeterministicSeeds(layers, DecoderSeedBase);

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);
        return new NeuralNetwork<T>(architecture, lossFunction: new MeanSquaredErrorLoss<T>());
    }

    private NeuralNetwork<T> CreateRNNNetwork()
    {
        // Simplified RNN: (latent, action, hidden) -> (next_latent_prediction, next_hidden)
        // Note: Full World Models implementation uses Mixture Density Network (MDN) with multiple mixtures
        // This simplified version uses single-mode prediction (NumMixtures parameter is for future MDN support)
        int inputSize = _options.LatentSize + _options.ActionSize + _options.RNNHiddenSize;
        int outputSize = _options.LatentSize + _options.RNNHiddenSize;  // Single prediction + hidden state

        // Explicit layers — see CreateEncoderNetwork for why default layers
        // must not be auto-generated.
        var layers = new List<ILayer<T>>
        {
            new DenseLayer<T>(_options.RNNHiddenSize, (IActivationFunction<T>)new TanhActivation<T>()),
            new DenseLayer<T>(outputSize, (IActivationFunction<T>)new IdentityActivation<T>())
        };

        AssignDeterministicSeeds(layers, RNNSeedBase);

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);
        return new NeuralNetwork<T>(architecture, lossFunction: new MeanSquaredErrorLoss<T>());
    }

    public override Vector<T> SelectAction(Vector<T> observation, bool training = true)
    {
        // Encode observation to latent code
        var encoderOutput = _vaeEncoder.Predict(Tensor<T>.FromVector(observation)).ToVector();
        var latentMean = ExtractMean(encoderOutput);

        // Concatenate latent and RNN hidden state
        var controllerInput = ConcatenateVectors(latentMean, _rnnHiddenState);

        // Compute action from controller using Engine.DotProduct
        var action = new Vector<T>(_options.ActionSize);
        for (int i = 0; i < _options.ActionSize; i++)
        {
            // Extract weight column
            var weightCol = new Vector<T>(controllerInput.Length);
            for (int j = 0; j < controllerInput.Length; j++)
            {
                weightCol[j] = _controllerWeights[j, i];
            }
            action[i] = MathHelper.Tanh<T>(Engine.DotProduct(controllerInput, weightCol));
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
        // Validate transition shapes at this public input boundary so a malformed experience can't
        // enter the replay buffer and later overflow rnnInput / encIn / decIn during Train().
        int obsSize = _options.ObservationWidth * _options.ObservationHeight * _options.ObservationChannels;
        if (observation.Length != obsSize)
            throw new ArgumentException($"Observation length must be {obsSize}, got {observation.Length}.", nameof(observation));
        if (nextObservation.Length != obsSize)
            throw new ArgumentException($"Next observation length must be {obsSize}, got {nextObservation.Length}.", nameof(nextObservation));
        if (action.Length != _options.ActionSize)
            throw new ArgumentException($"Action length must be {_options.ActionSize}, got {action.Length}.", nameof(action));

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

        var batch = _replayBuffer.Sample(_options.BatchSize);
        int n = batch.Count;
        T vaeLoss = NumOps.Zero;
        T rnnLoss = NumOps.Zero;

        if (n > 0)
        {
            int obsSize = _options.ObservationWidth * _options.ObservationHeight * _options.ObservationChannels;
            int latentSize = _options.LatentSize;
            int actionDim = _options.ActionSize;
            int hiddenSize = _options.RNNHiddenSize;
            int rnnInSize = latentSize + actionDim + hiddenSize;
            int rnnOutSize = latentSize + hiddenSize;

            var decIn = new Tensor<T>([n, latentSize]);
            var decTgt = new Tensor<T>([n, obsSize]);
            var rnnIn = new Tensor<T>([n, rnnInSize]);
            var rnnTgt = new Tensor<T>([n, rnnOutSize]);
            var encIn = new Tensor<T>([n, obsSize]);
            var encTgt = new Tensor<T>([n, latentSize * 2]);
            var zeroHidden = new Vector<T>(hiddenSize);

            for (int i = 0; i < n; i++)
            {
                var exp = batch[i];
                var mean = ExtractMean(_vaeEncoder.Predict(Tensor<T>.FromVector(exp.State)).ToVector());
                var encNextOut = _vaeEncoder.Predict(Tensor<T>.FromVector(exp.NextState)).ToVector();
                var meanNext = ExtractMean(encNextOut);

                // MDN-RNN next-latent prediction: predict z_{t+1} from (z_t, a_t, h).
                var rnnInput = ConcatenateVectors(ConcatenateVectors(mean, exp.Action), zeroHidden);
                var rnnOut = _rnnNetwork.Predict(Tensor<T>.FromVector(rnnInput)).ToVector();

                // VAE decoder reconstruction: D(z_t) -> o_t.
                for (int j = 0; j < latentSize; j++) decIn[i, j] = mean[j];
                for (int j = 0; j < obsSize && j < exp.State.Length; j++) decTgt[i, j] = exp.State[j];

                // RNN target = its own output with the LATENT half steered toward z_{t+1}.
                for (int j = 0; j < rnnInSize; j++) rnnIn[i, j] = rnnInput[j];
                for (int j = 0; j < rnnOutSize; j++) rnnTgt[i, j] = rnnOut[j];
                for (int j = 0; j < latentSize; j++) rnnTgt[i, j] = meanNext[j];

                // Encoder consistency: steer E(o_{t+1}) mean toward the RNN's predicted next latent
                // (keep its log-variance) so encoder and dynamics agree on the latent space.
                for (int j = 0; j < obsSize && j < exp.NextState.Length; j++) encIn[i, j] = exp.NextState[j];
                for (int j = 0; j < latentSize; j++) encTgt[i, j] = rnnOut[j];
                for (int j = 0; j < latentSize; j++) encTgt[i, latentSize + j] = encNextOut[latentSize + j];
            }

            _vaeDecoder.Train(decIn, decTgt);
            _vaeEncoder.Train(encIn, encTgt);
            _rnnNetwork.Train(rnnIn, rnnTgt);
            vaeLoss = NumOps.Add(_vaeDecoder.GetLastLoss(), _vaeEncoder.GetLastLoss());
            rnnLoss = _rnnNetwork.GetLastLoss();
        }

        T totalLoss = NumOps.Add(vaeLoss, rnnLoss);

        // Train Controller (evolution strategy - (1+1)-ES; full World Models uses CMA-ES)
        T controllerLoss = TrainController();
        totalLoss = NumOps.Add(totalLoss, controllerLoss);

        _updateCount++;

        return NumOps.Divide(totalLoss, NumOps.FromDouble(3));
    }

    private T TrainController()
    {
        // (1+1)-ES controller improvement (full World Models uses CMA-ES). The candidate objective
        // MUST depend on the candidate weights — the previous version scored every candidate from the
        // same replay rewards (independent of perturbedWeights), so the search never moved the
        // controller and merely added a reward as "loss". Here each candidate is scored by
        // REWARD-WEIGHTED behavior agreement: a controller that reproduces the actions taken on
        // HIGH-reward transitions scores higher. This is a legitimate offline policy-improvement
        // signal (advantage/reward-weighted behavior) and genuinely depends on the candidate policy.

        const int numCandidates = 5;
        const double perturbationScale = 0.01;

        var batch = _replayBuffer.Sample(Math.Min(10, _replayBuffer.Count));
        if (batch.Count == 0) return NumOps.Zero;

        // Pre-encode each batch state to its latent code once (the controller maps
        // latent ⊕ zero-hidden -> action, mirroring Act()).
        var zeroHidden = new Vector<T>(_options.RNNHiddenSize);
        var controllerInputs = new Vector<T>[batch.Count];
        for (int b = 0; b < batch.Count; b++)
        {
            var z = ExtractMean(_vaeEncoder.Predict(Tensor<T>.FromVector(batch[b].State)).ToVector());
            controllerInputs[b] = ConcatenateVectors(z, zeroHidden);
        }

        // Score(weights) = mean over the batch of -reward · ||controller(z) - storedAction||²
        // (higher = better: low action error on high-reward transitions). Depends on `weights`.
        T ScoreCandidate(Matrix<T> weights)
        {
            T total = NumOps.Zero;
            for (int b = 0; b < batch.Count; b++)
            {
                var input = controllerInputs[b];
                T err = NumOps.Zero;
                for (int i = 0; i < _options.ActionSize; i++)
                {
                    var col = new Vector<T>(input.Length);
                    for (int j = 0; j < input.Length; j++) col[j] = weights[j, i];
                    T act = MathHelper.Tanh<T>(Engine.DotProduct(input, col));
                    T d = NumOps.Subtract(act, batch[b].Action[i]);
                    err = NumOps.Add(err, NumOps.Multiply(d, d));
                }
                total = NumOps.Subtract(total, NumOps.Multiply(batch[b].Reward, err));
            }
            return NumOps.Divide(total, NumOps.FromDouble(batch.Count));
        }

        T bestScore = ScoreCandidate(_controllerWeights);
        Matrix<T>? bestWeights = null;
        for (int candidate = 0; candidate < numCandidates; candidate++)
        {
            var perturbedWeights = new Matrix<T>(_controllerWeights.Rows, _controllerWeights.Columns);
            for (int i = 0; i < _controllerWeights.Rows; i++)
            {
                for (int j = 0; j < _controllerWeights.Columns; j++)
                {
                    var noise = NumOps.FromDouble((_random.NextDouble() - 0.5) * 2.0 * perturbationScale);
                    perturbedWeights[i, j] = NumOps.Add(_controllerWeights[i, j], noise);
                }
            }

            T candidateScore = ScoreCandidate(perturbedWeights);
            if (NumOps.GreaterThan(candidateScore, bestScore))
            {
                bestScore = candidateScore;
                bestWeights = perturbedWeights;
            }
        }

        if (bestWeights is not null)
        {
            _controllerWeights = bestWeights;
        }

        // Return a clean, non-negative behavior LOSS for the chosen controller (mean unweighted
        // action MSE over the batch) for the totalLoss aggregation — lower is better.
        T lossTotal = NumOps.Zero;
        for (int b = 0; b < batch.Count; b++)
        {
            var input = controllerInputs[b];
            for (int i = 0; i < _options.ActionSize; i++)
            {
                var col = new Vector<T>(input.Length);
                for (int j = 0; j < input.Length; j++) col[j] = _controllerWeights[j, i];
                T act = MathHelper.Tanh<T>(Engine.DotProduct(input, col));
                T d = NumOps.Subtract(act, batch[b].Action[i]);
                lossTotal = NumOps.Add(lossTotal, NumOps.Multiply(d, d));
            }
        }
        return NumOps.Divide(lossTotal, NumOps.FromDouble(batch.Count));
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
        // Predict is the pure inference contract and MUST be side-effect free:
        // SelectAction advances the RNN hidden state for sequential rollout, so
        // snapshot it before and restore it after, keeping repeated Predict
        // calls deterministic (Policy_ShouldBeDeterministic) and independent of
        // prior inference calls.
        var savedHidden = new Vector<T>(_rnnHiddenState.Length);
        for (int i = 0; i < _rnnHiddenState.Length; i++)
        {
            savedHidden[i] = _rnnHiddenState[i];
        }

        var action = SelectAction(input, training: false);

        _rnnHiddenState = savedHidden;
        return action;
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
            int paramCount = checked((int)network.ParameterCount);
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
        // The fresh constructor reproduces the VAE/RNN network ARCHITECTURE exactly (deterministic
        // per-layer seeds), but Train() now UPDATES those networks' weights (the VAE encoder/decoder
        // and the MDN-RNN are trained, not just the controller). So we must copy the learned network
        // parameters onto the clone — otherwise it would keep the seed-initial weights and produce a
        // different world model / policy than the trained original (Clone_ShouldProduceSamePolicy).
        // GetParameters/SetParameters round-trip the network weights in-place WITHOUT rebuilding the
        // layer graph, so the RandomSeed pins (and thus tensor shapes) are preserved — unlike a full
        // serialization round-trip. The trained controller weights are copied separately below.
        var clone = new WorldModelsAgent<T>(_options);
        clone.SetParameters(GetParameters());

        var controllerCopy = new Matrix<T>(_controllerWeights.Rows, _controllerWeights.Columns);
        for (int i = 0; i < _controllerWeights.Rows; i++)
        {
            for (int j = 0; j < _controllerWeights.Columns; j++)
            {
                controllerCopy[i, j] = _controllerWeights[i, j];
            }
        }
        clone._controllerWeights = controllerCopy;

        var hiddenCopy = new Vector<T>(_rnnHiddenState.Length);
        for (int i = 0; i < _rnnHiddenState.Length; i++)
        {
            hiddenCopy[i] = _rnnHiddenState[i];
        }
        clone._rnnHiddenState = hiddenCopy;

        clone._updateCount = _updateCount;
        return clone;
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
