using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Graph Generation Model using Variational Autoencoder (VAE) architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Graph generation models learn to generate new graph structures from latent representations.
/// This implementation uses a Variational Graph Autoencoder (VGAE) approach that learns
/// a latent distribution of graph structures and can sample new valid graphs.
/// </para>
/// <para><b>For Beginners:</b> Graph generation creates new graphs similar to training data.
///
/// **How it works:**
/// - Encoder: Compress graph structure into latent space using GNN
/// - Latent space: Learn probabilistic representation (mean and variance)
/// - Decoder: Reconstruct graph from latent representation
/// - Sampling: Generate new graphs by sampling from latent space
///
/// **Example - Drug Discovery:**
/// - Train on known drug molecules
/// - Learn latent representation of valid molecular structures
/// - Generate new candidate molecules by sampling
/// - Filter candidates by predicted properties
///
/// **Key Components:**
/// - **GNN Encoder**: Maps node features to latent space
/// - **Variational Layer**: Learns mean and log-variance for each node
/// - **Inner Product Decoder**: Reconstructs adjacency matrix
/// - **Reparameterization**: Enables gradient flow through sampling
///
/// **Loss Function:**
/// - **Reconstruction Loss**: How well we reconstruct the adjacency matrix
/// - **KL Divergence**: Regularization to keep latent space well-structured
///
/// **Applications:**
/// - Molecular design and drug discovery
/// - Social network generation
/// - Circuit design
/// - Protein structure generation
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new GraphGenerationModelOptions { NodeFeatureSize = 9, MaxNodes = 50, HiddenSize = 128 };
/// var model = new GraphGenerationModel&lt;float&gt;(options);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 128 });
/// var graph = model.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.GraphAnalysis)]
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class GraphGenerationModel<T> : NeuralNetworkBase<T>
{
    private readonly GraphGenerationModelOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The loss function used to calculate the reconstruction error.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimization algorithm used to update the network's parameters during training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Gets the latent dimension size.
    /// </summary>
    public int LatentDim { get; }

    /// <summary>
    /// Gets the hidden dimension for encoder layers.
    /// </summary>
    public int HiddenDim { get; }

    /// <summary>
    /// Gets the number of encoder layers.
    /// </summary>
    public int NumEncoderLayers { get; }

    /// <summary>
    /// Gets the maximum number of nodes for graph generation.
    /// </summary>
    public int MaxNodes { get; }

    /// <summary>
    /// Gets the number of layers in the model.
    /// </summary>
    public int NumLayers { get; }

    /// <summary>
    /// Gets the type of graph generation.
    /// </summary>
    public GraphGenerationType GenerationType { get; }

    /// <summary>
    /// KL divergence weight for balancing reconstruction and regularization.
    /// </summary>
    public double KLWeight { get; set; } = 1.0;

    /// <summary>
    /// Encoder weights for mean projection.
    /// </summary>
    private Tensor<T> _meanWeights;

    /// <summary>
    /// Encoder weights for log-variance projection.
    /// </summary>
    private Tensor<T> _logVarWeights;

    /// <summary>
    /// Cached latent mean from last forward pass.
    /// </summary>
    private Tensor<T>? _lastMean;

    /// <summary>
    /// Cached latent log-variance from last forward pass.
    /// </summary>
    private Tensor<T>? _lastLogVar;

    /// <summary>
    /// Cached sampled latent representation.
    /// </summary>
    private Tensor<T>? _lastLatent;

    /// <summary>
    /// Cached encoder output before variational layer.
    /// </summary>
    private Tensor<T>? _lastEncoderOutput;

    /// <summary>
    /// Cached input adjacency matrix.
    /// </summary>
    private Tensor<T>? _cachedAdjacencyMatrix;

    /// <summary>
    /// Random number generator for sampling.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Gradient for mean weights.
    /// </summary>
    private Tensor<T>? _meanWeightsGradient;

    /// <summary>
    /// Gradient for log-variance weights.
    /// </summary>
    private Tensor<T>? _logVarWeightsGradient;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphGenerationModel{T}"/> class.
    /// </summary>
    /// <param name="inputFeatures">Number of input features per node (default: 16).</param>
    /// <param name="hiddenDim">Hidden dimension for encoder layers (default: 32).</param>
    /// <param name="latentDim">Dimension of latent space (default: 16).</param>
    /// <param name="numEncoderLayers">Number of GNN encoder layers (default: 2).</param>
    /// <param name="maxNodes">Maximum number of nodes for graph generation (default: 100).</param>
    /// <param name="generationType">Type of graph generation approach (default: VariationalAutoencoder).</param>
    /// <param name="klWeight">Weight for KL divergence term (default: 1.0).</param>
    /// <param name="optimizer">Optional optimizer for training (default: AdamOptimizer).</param>
    /// <param name="lossFunction">Optional loss function for training (default: BinaryCrossEntropyLoss).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creating a graph generation model:
    ///
    /// ```csharp
    /// // Create model with all defaults
    /// var model = new GraphGenerationModel&lt;double&gt;();
    ///
    /// // Create model for molecular generation with custom settings
    /// var model = new GraphGenerationModel&lt;double&gt;(
    ///     inputFeatures: 9,        // Atom features
    ///     hiddenDim: 32,           // Hidden layer size
    ///     latentDim: 16,           // Latent space dimension
    ///     numEncoderLayers: 2,     // 2 GNN encoder layers
    ///     maxNodes: 50,            // Maximum 50 atoms per molecule
    ///     klWeight: 0.5);          // KL divergence weight
    ///
    /// // Train on molecular graphs
    /// model.Train(molecules, adjacencyMatrices, epochs: 100);
    ///
    /// // Generate new molecules
    /// var newMolecules = model.Generate(numSamples: 10, numNodes: 20);
    /// ```
    /// </para>
    /// </remarks>
    public GraphGenerationModel(
        int inputFeatures = 16,
        int hiddenDim = 32,
        int latentDim = 16,
        int numEncoderLayers = 2,
        int maxNodes = 100,
        GraphGenerationType generationType = GraphGenerationType.VariationalAutoencoder,
        double klWeight = 1.0,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0,
        ILearningRateScheduler? learningRateScheduler = null,
        GraphGenerationModelOptions? options = null)
        : base(CreateArchitecture(inputFeatures, hiddenDim, latentDim, numEncoderLayers),
               lossFunction ?? new BinaryCrossEntropyLoss<T>(),
               maxGradNorm)
    {
        _options = options ?? new GraphGenerationModelOptions();
        Options = _options;
        LatentDim = latentDim;
        HiddenDim = hiddenDim;
        NumEncoderLayers = numEncoderLayers;
        MaxNodes = maxNodes;
        NumLayers = numEncoderLayers;
        GenerationType = generationType;
        KLWeight = klWeight;

        _lossFunction = lossFunction ?? new BinaryCrossEntropyLoss<T>();
        var adamOpts = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = 0.001,
            LearningRateScheduler = learningRateScheduler ?? new ExponentialLRScheduler(
                baseLearningRate: 0.001, gamma: 0.99),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        };
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, adamOpts);
        _random = RandomHelper.CreateSeededRandom(42);

        // Initialize variational layer weights
        _meanWeights = new Tensor<T>([hiddenDim, latentDim]);
        _logVarWeights = new Tensor<T>([hiddenDim, latentDim]);
        InitializeVariationalWeights();

        InitializeLayers();
    }

    /// <summary>
    /// Creates the encoder architecture without layers (layers are created in InitializeLayers).
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateArchitecture(
        int inputFeatures,
        int hiddenDim,
        int latentDim,
        int numEncoderLayers)
    {
        // Create architecture without layers - layers will be created in InitializeLayers
        // using LayerHelper for consistency with other models
        return new NeuralNetworkArchitecture<T>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: inputFeatures,
            outputSize: hiddenDim); // Output is hiddenDim since variational layer projects to latentDim
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Create default graph generation encoder layers using LayerHelper
            Layers.AddRange(LayerHelper<T>.CreateDefaultGraphGenerationLayers(
                Architecture, HiddenDim, NumEncoderLayers));
        }
    }

    /// <summary>
    /// Initializes the variational layer weights using Xavier initialization.
    /// </summary>
    private void InitializeVariationalWeights()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (HiddenDim + LatentDim)));
        var randomTensor = Tensor<T>.CreateRandom(_meanWeights.Shape._dims);
        var halfTensor = new Tensor<T>(_meanWeights.Shape._dims);
        Engine.TensorFill(halfTensor, NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);
        _meanWeights = Engine.TensorMultiplyScalar(shifted, scale);

        randomTensor = Tensor<T>.CreateRandom(_logVarWeights.Shape._dims);
        shifted = Engine.TensorSubtract(randomTensor, halfTensor);
        _logVarWeights = Engine.TensorMultiplyScalar(shifted, scale);
    }

    /// <summary>
    /// Encodes node features into latent space representations.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor of shape [numNodes, inputFeatures].</param>
    /// <param name="adjacencyMatrix">Adjacency matrix of shape [numNodes, numNodes].</param>
    /// <returns>Tuple of (mean, log_variance) tensors for the latent distribution.</returns>
    public (Tensor<T> mean, Tensor<T> logVar) Encode(Tensor<T> nodeFeatures, Tensor<T> adjacencyMatrix)
    {
        _cachedAdjacencyMatrix = adjacencyMatrix;

        // Set adjacency matrix on all graph layers
        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
            {
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
            }
        }

        // Forward through encoder layers
        Tensor<T> hidden = nodeFeatures;
        foreach (var layer in Layers)
        {
            hidden = layer.Forward(hidden);
        }

        _lastEncoderOutput = hidden;

        // Project to mean and log-variance
        int numNodes = hidden.Shape[0];
        _lastMean = Engine.TensorMatMul(hidden, _meanWeights);
        _lastLogVar = Engine.TensorMatMul(hidden, _logVarWeights);

        return (_lastMean, _lastLogVar);
    }

    /// <summary>
    /// Samples from the latent distribution using the reparameterization trick.
    /// </summary>
    /// <param name="mean">Mean of the latent distribution.</param>
    /// <param name="logVar">Log-variance of the latent distribution.</param>
    /// <returns>Sampled latent representation.</returns>
    public Tensor<T> Reparameterize(Tensor<T> mean, Tensor<T> logVar)
    {
        // z = mean + std * epsilon, where epsilon ~ N(0, 1)
        // Clamp logVar to prevent exp() overflow (exp(20) ≈ 5e8, exp(40) ≈ 2e17)
        var clampedLogVar = new Tensor<T>(logVar.Shape._dims);
        for (int i = 0; i < logVar.Length; i++)
        {
            double v = NumOps.ToDouble(logVar.GetFlat(i));
            v = Math.Max(-20.0, Math.Min(20.0, v));
            clampedLogVar.SetFlat(i, NumOps.FromDouble(v));
        }
        var std = Engine.TensorSqrt(Engine.TensorExp(clampedLogVar));

        // Generate standard normal samples
        var epsilon = new Tensor<T>(mean.Shape._dims);
        for (int i = 0; i < epsilon.Length; i++)
        {
            epsilon.SetFlat(i, NumOps.FromDouble(_random.NextGaussian()));
        }

        _lastLatent = Engine.TensorAdd(mean, Engine.TensorMultiply(std, epsilon));
        return _lastLatent;
    }

    /// <summary>
    /// Decodes latent representations to reconstruct the adjacency matrix.
    /// </summary>
    /// <param name="latent">Latent representation tensor of shape [numNodes, latentDim].</param>
    /// <returns>Reconstructed adjacency matrix of shape [numNodes, numNodes].</returns>
    public Tensor<T> Decode(Tensor<T> latent)
    {
        // Inner product decoder: A_ij = sigma(z_i^T * z_j)
        int numNodes = latent.Shape[0];
        var latentT = Engine.TensorTranspose(latent);
        var logits = Engine.TensorMatMul(latent, latentT);

        // === Vectorized sigmoid using IEngine (Phase B: US-GPU-015) ===
        return Engine.Sigmoid(logits);
    }

    /// <summary>
    /// Performs a complete forward pass: encode, sample, decode.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor.</param>
    /// <param name="adjacencyMatrix">Adjacency matrix.</param>
    /// <returns>Reconstructed adjacency matrix.</returns>
    public Tensor<T> Forward(Tensor<T> nodeFeatures, Tensor<T> adjacencyMatrix)
    {
        var (mean, logVar) = Encode(nodeFeatures, adjacencyMatrix);
        var latent = Reparameterize(mean, logVar);
        return Decode(latent);
    }

    /// <summary>
    /// Computes the ELBO loss (reconstruction + KL divergence).
    /// </summary>
    /// <param name="reconstructed">Reconstructed adjacency matrix.</param>
    /// <param name="original">Original adjacency matrix.</param>
    /// <returns>Total ELBO loss value.</returns>
    public T ComputeLoss(Tensor<T> reconstructed, Tensor<T> original)
    {
        // Reconstruction loss: Binary cross-entropy
        var reconLoss = ComputeReconstructionLoss(reconstructed, original);

        // KL divergence: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        var klLoss = ComputeKLDivergence();

        // Total loss = reconstruction + KL_weight * KL
        var scaledKL = NumOps.Multiply(NumOps.FromDouble(KLWeight), klLoss);
        return NumOps.Add(reconLoss, scaledKL);
    }

    /// <summary>
    /// Computes binary cross-entropy reconstruction loss.
    /// </summary>
    private T ComputeReconstructionLoss(Tensor<T> reconstructed, Tensor<T> original)
    {
        int numNodes = original.Shape[0];
        var loss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                T p = reconstructed[i, j];
                T y = original[i, j];

                // Clamp p to [epsilon, 1-epsilon] for numerical stability
                var upperBound = NumOps.Subtract(NumOps.One, epsilon);
                p = NumOps.LessThan(p, upperBound) ? p : upperBound;
                p = NumOps.GreaterThan(p, epsilon) ? p : epsilon;

                // BCE: -y * log(p) - (1-y) * log(1-p)
                T term1 = NumOps.Multiply(y, NumOps.Log(p));
                T term2 = NumOps.Multiply(NumOps.Subtract(NumOps.One, y),
                    NumOps.Log(NumOps.Subtract(NumOps.One, p)));
                loss = NumOps.Subtract(loss, NumOps.Add(term1, term2));
            }
        }

        // Average over all edges
        return NumOps.Divide(loss, NumOps.FromDouble(numNodes * numNodes));
    }

    /// <summary>
    /// Computes KL divergence from standard normal.
    /// </summary>
    private T ComputeKLDivergence()
    {
        if (_lastMean == null || _lastLogVar == null)
        {
            return NumOps.Zero;
        }

        // === Vectorized KL divergence using IEngine (Phase B: US-GPU-015) ===
        // KL = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        //    = 0.5 * sum(exp(log_var) + mean^2 - 1 - log_var)
        var expLogVar = Engine.TensorExp(_lastLogVar);
        var meanSquared = Engine.TensorMultiply(_lastMean, _lastMean);

        T sumExpLogVar = Engine.TensorSum(expLogVar);
        T sumMeanSquared = Engine.TensorSum(meanSquared);
        T sumLogVar = Engine.TensorSum(_lastLogVar);
        T n = NumOps.FromDouble(_lastMean.Length);

        // sum(exp(logVar) + mean^2 - 1 - logVar)
        T sum = NumOps.Subtract(
            NumOps.Add(sumExpLogVar, sumMeanSquared),
            NumOps.Add(n, sumLogVar));

        T kl = NumOps.Multiply(NumOps.FromDouble(0.5), sum);
        return NumOps.Divide(kl, n);
    }

    /// <summary>
    /// Performs backward pass through the model.
    /// </summary>
    /// <param name="outputGradient">Gradient of the loss with respect to reconstructed adjacency.</param>
    /// <returns>Gradient with respect to input features.</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastLatent == null || _lastMean == null || _lastLogVar == null ||
            _lastEncoderOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        int numNodes = _lastLatent.Shape[0];

        // Gradient through decoder (inner product)
        // d(z_i^T * z_j)/d(z) = sum_j(grad_ij * z_j) for each i
        var latentGrad = new Tensor<T>(_lastLatent.Shape._dims);
        for (int i = 0; i < numNodes; i++)
        {
            for (int k = 0; k < LatentDim; k++)
            {
                var grad = NumOps.Zero;
                for (int j = 0; j < numNodes; j++)
                {
                    // Gradient through sigmoid: grad * p * (1-p)
                    T logit = NumOps.Zero;
                    for (int l = 0; l < LatentDim; l++)
                    {
                        logit = NumOps.Add(logit, NumOps.Multiply(_lastLatent[i, l], _lastLatent[j, l]));
                    }
                    T negLogit = NumOps.Negate(logit);
                    T expNeg = NumOps.Exp(negLogit);
                    T onePlusExp = NumOps.Add(NumOps.One, expNeg);
                    T p = NumOps.Divide(NumOps.One, onePlusExp);
                    T sigmoidDeriv = NumOps.Multiply(p, NumOps.Subtract(NumOps.One, p));

                    T gradFromJ = NumOps.Multiply(NumOps.Multiply(outputGradient[i, j], sigmoidDeriv), _lastLatent[j, k]);
                    grad = NumOps.Add(grad, gradFromJ);

                    // Symmetric contribution
                    if (i != j)
                    {
                        T gradFromI = NumOps.Multiply(NumOps.Multiply(outputGradient[j, i], sigmoidDeriv), _lastLatent[i, k]);
                        grad = NumOps.Add(grad, gradFromI);
                    }
                }
                latentGrad[i, k] = grad;
            }
        }

        // Gradient through reparameterization
        // z = mean + std * epsilon
        // d(loss)/d(mean) = d(loss)/d(z)
        // d(loss)/d(logvar) = d(loss)/d(z) * epsilon * 0.5 * exp(0.5 * logvar)
        var meanGrad = latentGrad;

        // Gradient through mean projection: encoder_output @ mean_weights = mean
        // d(loss)/d(mean_weights) = encoder_output^T @ mean_grad
        var encoderT = Engine.TensorTranspose(_lastEncoderOutput);
        _meanWeightsGradient = Engine.TensorMatMul(encoderT, meanGrad);

        // Add KL gradient for mean: d(KL)/d(mean) = mean
        var klMeanGrad = _lastMean;
        var totalMeanGrad = Engine.TensorAdd(meanGrad, Engine.TensorMultiplyScalar(klMeanGrad, NumOps.FromDouble(KLWeight)));

        // === Vectorized KL log-variance gradient using IEngine (Phase B: US-GPU-015) ===
        // d(KL)/d(logvar) = 0.5 * KLWeight * (exp(logvar) - 1)
        var expLogVar = Engine.TensorExp(_lastLogVar);
        var expMinusOne = Engine.TensorSubtractScalar(expLogVar, NumOps.One);
        var klLogVarGrad = Engine.TensorMultiplyScalar(expMinusOne, NumOps.FromDouble(0.5 * KLWeight));

        _logVarWeightsGradient = Engine.TensorMatMul(encoderT, klLogVarGrad);

        // Gradient to encoder output
        var encoderGrad = Engine.TensorMatMul(totalMeanGrad, Engine.TensorTranspose(_meanWeights));
        var encoderGradFromLogVar = Engine.TensorMatMul(klLogVarGrad, Engine.TensorTranspose(_logVarWeights));
        encoderGrad = Engine.TensorAdd(encoderGrad, encoderGradFromLogVar);

        // Backward through encoder layers
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            encoderGrad = Layers[i].Backward(encoderGrad);
        }

        return encoderGrad;
    }

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParamCount = layer.ParameterCount;
            if (layerParamCount > 0)
            {
                var layerParams = parameters.SubVector(index, layerParamCount);
                layer.SetParameters(layerParams);
                index += layerParamCount;
            }
        }

        // Update variational layer parameters
        int meanCount = _meanWeights.Length;
        int logVarCount = _logVarWeights.Length;

        if (index + meanCount + logVarCount <= parameters.Length)
        {
            _meanWeights = Tensor<T>.FromVector(parameters.SubVector(index, meanCount))
                .Reshape(_meanWeights.Shape._dims);
            index += meanCount;

            _logVarWeights = Tensor<T>.FromVector(parameters.SubVector(index, logVarCount))
                .Reshape(_logVarWeights.Shape._dims);
        }
    }

    /// <summary>
    /// Trains the model on graph data.
    /// </summary>
    /// <param name="nodeFeatures">Node feature tensor.</param>
    /// <param name="adjacencyMatrix">Adjacency matrix (target for reconstruction).</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate.</param>
    public void Train(
        Tensor<T> nodeFeatures,
        Tensor<T> adjacencyMatrix,
        int epochs = 200,
        double learningRate = 0.01)
    {
        var lr = NumOps.FromDouble(learningRate);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Set to training mode
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(true);
            }

            // Forward pass
            var reconstructed = Forward(nodeFeatures, adjacencyMatrix);

            // Compute reconstruction gradient
            var reconGrad = ComputeReconstructionGradient(reconstructed, adjacencyMatrix);

            // Backward pass
            Backward(reconGrad);

            // Update encoder parameters
            foreach (var layer in Layers)
            {
                layer.UpdateParameters(lr);
            }

            // Update variational layer parameters
            if (_meanWeightsGradient != null)
            {
                _meanWeights = Engine.TensorSubtract(_meanWeights,
                    Engine.TensorMultiplyScalar(_meanWeightsGradient, lr));
            }
            if (_logVarWeightsGradient != null)
            {
                _logVarWeights = Engine.TensorSubtract(_logVarWeights,
                    Engine.TensorMultiplyScalar(_logVarWeightsGradient, lr));
            }
        }

        // Set to inference mode
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Computes gradient of reconstruction loss.
    /// </summary>
    private Tensor<T> ComputeReconstructionGradient(Tensor<T> reconstructed, Tensor<T> original)
    {
        int numNodes = original.Shape[0];
        var gradient = new Tensor<T>([numNodes, numNodes]);
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                T p = reconstructed[i, j];
                T y = original[i, j];

                // Clamp p for numerical stability
                var upperBound = NumOps.Subtract(NumOps.One, epsilon);
                p = NumOps.LessThan(p, upperBound) ? p : upperBound;
                p = NumOps.GreaterThan(p, epsilon) ? p : epsilon;

                // d(BCE)/d(p) = -y/p + (1-y)/(1-p)
                T term1 = NumOps.Divide(y, p);
                T term2 = NumOps.Divide(NumOps.Subtract(NumOps.One, y),
                    NumOps.Subtract(NumOps.One, p));
                gradient[i, j] = NumOps.Subtract(term2, term1);
            }
        }

        // Scale by number of elements
        return Engine.TensorMultiplyScalar(gradient, NumOps.FromDouble(1.0 / (numNodes * numNodes)));
    }

    /// <summary>
    /// Generates new graphs by sampling from the latent space.
    /// </summary>
    /// <param name="numNodes">Number of nodes in generated graphs.</param>
    /// <param name="numSamples">Number of graphs to generate.</param>
    /// <param name="threshold">Edge probability threshold (default: 0.5).</param>
    /// <returns>List of generated adjacency matrices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generating new graphs:
    ///
    /// ```csharp
    /// // Generate 10 new molecular graphs with 20 atoms each
    /// var newGraphs = model.Generate(numNodes: 20, numSamples: 10, threshold: 0.5);
    ///
    /// // Each graph is an adjacency matrix where 1 indicates an edge
    /// foreach (var adj in newGraphs)
    /// {
    ///     // Process generated molecule structure
    /// }
    /// ```
    /// </para>
    /// </remarks>
    public List<Tensor<T>> Generate(int numNodes, int numSamples = 1, double threshold = 0.5)
    {
        var generatedGraphs = new List<Tensor<T>>();
        for (int i = 0; i < numSamples; i++)
        {
            generatedGraphs.Add(Generate(1, numNodes, null, threshold));
        }
        return generatedGraphs;
    }

    /// <summary>
    /// Generates new graphs by sampling from the latent space with optional conditioning.
    /// </summary>
    /// <param name="numSamples">Number of graphs to generate.</param>
    /// <param name="numNodes">Number of nodes in generated graphs.</param>
    /// <param name="conditioningInput">Optional conditioning input tensor.</param>
    /// <param name="threshold">Edge probability threshold (default: 0.5).</param>
    /// <returns>Generated adjacency matrix tensor.</returns>
    public Tensor<T> Generate(int numSamples, int numNodes, Tensor<T>? conditioningInput = null, double threshold = 0.5)
    {
        // For simplicity, generate a single graph and return it
        // Sample from standard normal for latent space
        var latent = new Tensor<T>([numNodes, LatentDim]);
        for (int i = 0; i < latent.Length; i++)
        {
            latent.SetFlat(i, NumOps.FromDouble(_random.NextGaussian()));
        }

        // Decode to get edge probabilities
        var edgeProbs = Decode(latent);

        // Threshold to get binary adjacency matrix
        var adjacency = new Tensor<T>([numNodes, numNodes]);
        T thresholdT = NumOps.FromDouble(threshold);
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                if (NumOps.GreaterThan(edgeProbs[i, j], thresholdT))
                {
                    adjacency[i, j] = NumOps.One;
                }
                else
                {
                    adjacency[i, j] = NumOps.Zero;
                }
            }
        }

        // Make symmetric (undirected graph)
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = i + 1; j < numNodes; j++)
            {
                if (NumOps.GreaterThan(adjacency[i, j], NumOps.Zero) ||
                    NumOps.GreaterThan(adjacency[j, i], NumOps.Zero))
                {
                    adjacency[i, j] = NumOps.One;
                    adjacency[j, i] = NumOps.One;
                }
            }
            // No self-loops
            adjacency[i, i] = NumOps.Zero;
        }

        return adjacency;
    }

    /// <summary>
    /// Interpolates between two graphs in latent space.
    /// </summary>
    /// <param name="graph1Features">Node features of first graph.</param>
    /// <param name="graph1Adj">Adjacency matrix of first graph.</param>
    /// <param name="graph2Features">Node features of second graph.</param>
    /// <param name="graph2Adj">Adjacency matrix of second graph.</param>
    /// <param name="numSteps">Number of interpolation steps.</param>
    /// <returns>List of interpolated adjacency matrices.</returns>
    public List<Tensor<T>> Interpolate(
        Tensor<T> graph1Features,
        Tensor<T> graph1Adj,
        Tensor<T> graph2Features,
        Tensor<T> graph2Adj,
        int numSteps = 5)
    {
        // Encode both graphs
        var (mean1, _) = Encode(graph1Features, graph1Adj);
        var latent1 = mean1; // Use mean for deterministic interpolation

        var (mean2, _) = Encode(graph2Features, graph2Adj);
        var latent2 = mean2;

        var interpolatedGraphs = new List<Tensor<T>>();

        for (int step = 0; step <= numSteps; step++)
        {
            double alpha = (double)step / numSteps;

            // Linear interpolation in latent space
            var interpolated = new Tensor<T>(latent1.Shape._dims);
            for (int i = 0; i < latent1.Length; i++)
            {
                T val1 = latent1.GetFlat(i);
                T val2 = latent2.GetFlat(i);
                T interp = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(1 - alpha), val1),
                    NumOps.Multiply(NumOps.FromDouble(alpha), val2));
                interpolated.SetFlat(i, interp);
            }

            // Decode
            var reconstructed = Decode(interpolated);

            // Threshold
            T threshold = NumOps.FromDouble(0.5);
            var adjacency = new Tensor<T>(reconstructed.Shape._dims);
            for (int i = 0; i < reconstructed.Length; i++)
            {
                adjacency.SetFlat(i, NumOps.GreaterThan(reconstructed.GetFlat(i), threshold)
                    ? NumOps.One : NumOps.Zero);
            }

            interpolatedGraphs.Add(adjacency);
        }

        return interpolatedGraphs;
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the network.
    /// </summary>
    public new int GetParameterCount()
    {
        int count = 0;
        foreach (var layer in Layers)
        {
            count += layer.ParameterCount;
        }
        count += _meanWeights.Length;
        count += _logVarWeights.Length;
        return count;
    }

    /// <summary>
    /// Gets all parameter gradients as a vector (encoder layers + variational weights).
    /// </summary>
    public override Vector<T> GetParameterGradients()
    {
        // Encoder layer gradients
        var baseGradients = base.GetParameterGradients();
        var allGrads = new List<T>();
        for (int i = 0; i < baseGradients.Length; i++)
            allGrads.Add(baseGradients[i]);

        // Variational layer gradients
        if (_meanWeightsGradient != null)
        {
            for (int i = 0; i < _meanWeightsGradient.Length; i++)
                allGrads.Add(_meanWeightsGradient.GetFlat(i));
        }
        else
        {
            for (int i = 0; i < _meanWeights.Length; i++)
                allGrads.Add(NumOps.Zero);
        }

        if (_logVarWeightsGradient != null)
        {
            for (int i = 0; i < _logVarWeightsGradient.Length; i++)
                allGrads.Add(_logVarWeightsGradient.GetFlat(i));
        }
        else
        {
            for (int i = 0; i < _logVarWeights.Length; i++)
                allGrads.Add(NumOps.Zero);
        }

        return new Vector<T>(allGrads.ToArray());
    }

    /// <summary>
    /// Gets all parameters as a vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        // Encoder parameters
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                allParams.Add(layerParams[i]);
            }
        }

        // Variational layer parameters
        for (int i = 0; i < _meanWeights.Length; i++)
        {
            allParams.Add(_meanWeights.GetFlat(i));
        }
        for (int i = 0; i < _logVarWeights.Length; i++)
        {
            allParams.Add(_logVarWeights.GetFlat(i));
        }

        return new Vector<T>([.. allParams]);
    }

    #region Abstract Method Implementations

    /// <summary>
    /// Makes a prediction by encoding input features and decoding to adjacency matrix.
    /// Uses the mean of the latent distribution (no sampling) for deterministic inference.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Ensure 2D input [numNodes, features]
        if (input.Rank == 1)
            input = input.Reshape([1, input.Shape[0]]);

        int numNodes = input.Shape[0];
        var adjacencyMatrix = EnsureAdjacencyMatrix(numNodes);

        // Set to inference mode
        foreach (var layer in Layers)
            layer.SetTrainingMode(false);

        // Encode → use mean (deterministic) → decode
        var (mean, _) = Encode(input, adjacencyMatrix);
        return Decode(mean);
    }

    private Tensor<T>? _autoAdjacencyMatrix;

    private Tensor<T> EnsureAdjacencyMatrix(int numNodes)
    {
        if (_autoAdjacencyMatrix != null && _autoAdjacencyMatrix.Shape[0] == numNodes)
            return _autoAdjacencyMatrix;

        var adj = new Tensor<T>([numNodes, numNodes]);
        for (int i = 0; i < numNodes; i++)
            for (int j = 0; j < numNodes; j++)
                adj[i, j] = NumOps.One;

        _autoAdjacencyMatrix = adj;
        return adj;
    }

    /// <summary>
    /// Trains the model on a single batch of data using VAE training (encode → sample → decode).
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Ensure 2D input [numNodes, features]
        if (input.Rank == 1)
            input = input.Reshape([1, input.Shape[0]]);

        int numNodes = input.Shape[0];
        var adjacencyMatrix = EnsureAdjacencyMatrix(numNodes);

        // Set all layers to training mode
        foreach (var layer in Layers)
            layer.SetTrainingMode(true);

        // Forward pass: encode → sample → decode
        var reconstructed = Forward(input, adjacencyMatrix);

        // Compute ELBO loss
        LastLoss = ComputeLoss(reconstructed, adjacencyMatrix);

        // Compute reconstruction gradient
        var reconGrad = ComputeReconstructionGradient(reconstructed, adjacencyMatrix);

        // Backward pass
        Backward(reconGrad);

        // Collect all parameter gradients (encoder layers + variational weights)
        Vector<T> parameterGradients = GetParameterGradients();

        // Fresh optimizer per step for stable convergence (no momentum oscillation).
        var stepOptimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Get current parameters
        Vector<T> currentParameters = GetParameters();

        // Update all parameters using the optimizer
        Vector<T> updatedParameters = stepOptimizer.UpdateParameters(currentParameters, parameterGradients);

        // Apply updated parameters (includes encoder layers + variational weights)
        UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Gets the intermediate activations from each layer, ensuring adjacency is set.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (input.Rank == 1)
            input = input.Reshape([1, input.Shape[0]]);

        int numNodes = input.Shape[0];
        var adjacencyMatrix = EnsureAdjacencyMatrix(numNodes);

        foreach (var layer in Layers)
        {
            if (layer is IGraphConvolutionLayer<T> graphLayer)
                graphLayer.SetAdjacencyMatrix(adjacencyMatrix);
        }

        var activations = new Dictionary<string, Tensor<T>>();
        var current = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    /// <summary>
    /// Gets metadata about this model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["NetworkType"] = "GraphGenerationModel",
                ["LatentDim"] = LatentDim,
                ["HiddenDim"] = HiddenDim,
                ["MaxNodes"] = MaxNodes,
                ["NumLayers"] = NumLayers,
                ["GenerationType"] = GenerationType.ToString()
            }
        };
    }

    /// <summary>
    /// Serializes network-specific data to a binary writer.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(LatentDim);
        writer.Write(HiddenDim);
        writer.Write(MaxNodes);
        writer.Write(NumLayers);
        writer.Write((int)GenerationType);
        SerializationHelper<T>.SerializeInterface(writer, _lossFunction);
        SerializationHelper<T>.SerializeInterface(writer, _optimizer);

        // Serialize variational layer weights
        writer.Write(_meanWeights.Length);
        for (int i = 0; i < _meanWeights.Length; i++)
            writer.Write(Convert.ToDouble(_meanWeights.GetFlat(i)));
        writer.Write(_logVarWeights.Length);
        for (int i = 0; i < _logVarWeights.Length; i++)
            writer.Write(Convert.ToDouble(_logVarWeights.GetFlat(i)));
    }

    /// <summary>
    /// Deserializes network-specific data from a binary reader.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // LatentDim
        _ = reader.ReadInt32(); // HiddenDim
        _ = reader.ReadInt32(); // MaxNodes
        _ = reader.ReadInt32(); // NumLayers
        _ = (GraphGenerationType)reader.ReadInt32();
        _ = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader);
        _ = DeserializationHelper.DeserializeInterface<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>(reader);

        // Restore variational layer weights
        int meanCount = reader.ReadInt32();
        var meanData = new T[meanCount];
        for (int i = 0; i < meanCount; i++)
            meanData[i] = NumOps.FromDouble(reader.ReadDouble());
        _meanWeights = Tensor<T>.FromVector(new Vector<T>(meanData)).Reshape(_meanWeights.Shape._dims);

        int logVarCount = reader.ReadInt32();
        var logVarData = new T[logVarCount];
        for (int i = 0; i < logVarCount; i++)
            logVarData[i] = NumOps.FromDouble(reader.ReadDouble());
        _logVarWeights = Tensor<T>.FromVector(new Vector<T>(logVarData)).Reshape(_logVarWeights.Shape._dims);
    }

    /// <summary>
    /// Creates a new instance of this model type.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new GraphGenerationModel<T>(
            inputFeatures: Architecture.InputSize,
            hiddenDim: HiddenDim,
            latentDim: LatentDim,
            numEncoderLayers: NumLayers,
            maxNodes: MaxNodes,
            generationType: GenerationType);
    }

    #endregion
}
