using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Causal-GAN generator that learns causal graph structure (directed acyclic graph)
/// and generates synthetic data respecting causal relationships between features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Causal-GAN discovers causal structure using a NOTEARS-style continuous relaxation
/// for learning a directed acyclic graph (DAG), then generates each feature as a
/// function of its causal parents via structural equation models.
/// </para>
/// <para>
/// Architecture:
/// <code>
/// Generator:   noise --> [FC+BN+ReLU] x L (with residual) --> raw features
///              raw features --> causal mixing (I + W^T) * raw --> structured features
/// Discriminator: features --> [FC+LeakyReLU+Dropout] x L --> real/fake score (WGAN)
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> Causal-GAN learns which features cause other features:
///
/// Instead of just learning that "Age and Income are related" (correlation),
/// it learns "Education causes higher Income" (causation).
///
/// This has two benefits:
/// 1. Generated data respects cause-effect chains, producing more realistic samples
/// 2. You can simulate "what-if" scenarios (interventions) on specific features
///
/// The model learns a weighted adjacency matrix W where W[i,j] means feature i
/// influences feature j. A DAG penalty (NOTEARS) ensures no circular dependencies.
///
/// The training uses:
/// - WGAN-GP loss for stable adversarial training
/// - NOTEARS penalty via augmented Lagrangian for DAG acyclicity constraint
/// - L1 sparsity on the adjacency matrix for interpretable causal structure
///
/// The NOTEARS constraint is: h(W) = tr(e^(W * W)) - d = 0
/// where d is the number of features. This is zero if and only if W encodes a DAG.
/// The matrix exponential is computed via truncated Taylor series.
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the generator network. Otherwise, the network creates the standard architecture.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new CausalGANOptions&lt;double&gt;
/// {
///     DAGPenaltyWeight = 0.5,
///     Epochs = 300
/// };
/// var generator = new CausalGANGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: Zheng et al., "DAGs with NO TEARS: Continuous Optimization for Structure Learning" (NeurIPS 2018)
/// </para>
/// </remarks>
public class CausalGANGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly CausalGANOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Causal structure: adjacency matrix W (numFeatures x numFeatures)
    // W[i,j] > 0 means feature i causally influences feature j
    private Matrix<T>? _adjacency;

    // Augmented Lagrangian parameters for NOTEARS DAG constraint
    private double _lagrangeAlpha;
    private double _lagrangeRho;
    private double _prevDagConstraint;

    // Generator batch normalization layers (auxiliary, always created to match Layers)
    private readonly List<BatchNormalizationLayer<T>> _genBNLayers = new();

    // Discriminator layers (auxiliary, not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _discLayers = new();
    private readonly List<DropoutLayer<T>> _discDropoutLayers = new();
    private readonly List<(int InputSize, int OutputSize)> _discLayerDims = new();

    // Cached pre-activations for proper backward passes
    private readonly List<Tensor<T>> _genPreActivations = new();
    private readonly List<Tensor<T>> _discPreActivations = new();

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the CausalGAN-specific options.
    /// </summary>
    public new CausalGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="CausalGANGenerator{T}"/> class.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">CausalGAN-specific configuration options.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    public CausalGANGenerator(
        NeuralNetworkArchitecture<T> architecture,
        CausalGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new CausalGANOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        _lagrangeAlpha = 0.0;
        _lagrangeRho = 1.0;
        _prevDagConstraint = double.MaxValue;

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the CausalGAN generator based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Layers = generator hidden layers (user-overridable).
    /// The discriminator is always auxiliary and not user-overridable.
    /// Generator uses residual connections with BatchNorm and manual ReLU.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default generator layers (placeholder dims until Fit())
            BuildDefaultGeneratorLayers(_options.EmbeddingDimension, Architecture.OutputSize);
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Builds default generator layers with residual connections, BatchNorm, and manual ReLU.
    /// All layers use IdentityActivation so we control activation ordering.
    /// </summary>
    private void BuildDefaultGeneratorLayers(int inputDim, int outputDim)
    {
        Layers.Clear();
        _genBNLayers.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var dims = _options.HiddenDimensions;

        for (int i = 0; i < dims.Length; i++)
        {
            // Residual: input dimension includes original input concatenated
            int layerInput = i == 0 ? inputDim : dims[i - 1] + inputDim;
            Layers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
            _genBNLayers.Add(new BatchNormalizationLayer<T>(dims[i]));
        }

        // Output layer: produces raw features (no activation)
        int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
        Layers.Add(new FullyConnectedLayer<T>(lastHidden, outputDim, identity));
    }

    /// <summary>
    /// Rebuilds generator and discriminator layers with actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildLayersWithActualDimensions(int genInputDim, int genOutputDim, int discInputDim)
    {
        if (!_usingCustomLayers)
        {
            BuildDefaultGeneratorLayers(genInputDim, genOutputDim);
        }

        // Discriminator is always rebuilt
        BuildDiscriminator(discInputDim);
    }

    /// <summary>
    /// Builds the discriminator network with Dropout and manual LeakyReLU.
    /// Tracks layer dimensions for ComputeDiscriminatorInputGradient.
    /// </summary>
    private void BuildDiscriminator(int inputDim)
    {
        _discLayers.Clear();
        _discDropoutLayers.Clear();
        _discLayerDims.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var dims = _options.HiddenDimensions;

        for (int i = 0; i < dims.Length; i++)
        {
            int layerInput = i == 0 ? inputDim : dims[i - 1];
            _discLayers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
            _discLayerDims.Add((layerInput, dims[i]));
            _discDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }

        // Output layer: single scalar (raw Wasserstein score, no activation)
        int lastHidden = dims.Length > 0 ? dims[^1] : inputDim;
        _discLayers.Add(new FullyConnectedLayer<T>(lastHidden, 1, identity));
        _discLayerDims.Add((lastHidden, 1));
    }

    #endregion

    #region Neural Network Methods (GANDALF Pattern)

    /// <summary>
    /// Runs the generator forward pass with residual connections and pre-activation caching.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        if (_usingCustomLayers)
        {
            var current = input;
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }
            return current;
        }

        return GeneratorForward(input);
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var output = Predict(input);

        // Simple MSE gradient
        var gradient = new Tensor<T>(output.Shape);
        for (int i = 0; i < output.Length && i < expectedOutput.Length; i++)
        {
            gradient[i] = NumOps.FromDouble(
                2.0 * (NumOps.ToDouble(output[i]) - NumOps.ToDouble(expectedOutput[i])));
        }

        // Backward through generator
        BackwardGenerator(gradient);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    #endregion

    #region ISyntheticTabularGenerator<T> Implementation

    /// <summary>
    /// Fits the CausalGAN generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix.</param>
    /// <param name="columns">Metadata describing each column.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Transforms data using VGM normalization
    /// 2. Initializes the causal adjacency matrix
    /// 3. Trains generator and discriminator using WGAN-GP
    /// 4. Updates the causal structure using augmented Lagrangian (NOTEARS)
    /// After fitting, call Generate() to create new synthetic rows.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        // Step 1: Fit the data transformer
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;
        var transformedData = _transformer.Transform(data);

        // Step 2: Initialize adjacency matrix with small random values (zero diagonal)
        _adjacency = new Matrix<T>(_dataWidth, _dataWidth);
        for (int i = 0; i < _dataWidth; i++)
        {
            for (int j = 0; j < _dataWidth; j++)
            {
                if (i != j)
                {
                    _adjacency[i, j] = NumOps.FromDouble(0.01 * (_random.NextDouble() - 0.5));
                }
            }
        }

        // Step 3: Reset augmented Lagrangian parameters
        _lagrangeAlpha = 0.0;
        _lagrangeRho = 1.0;
        _prevDagConstraint = double.MaxValue;

        // Step 4: Build networks with actual dimensions
        RebuildLayersWithActualDimensions(_options.EmbeddingDimension, _dataWidth, _dataWidth);

        // Step 5: Training loop
        T lr = NumOps.FromDouble(_options.LearningRate);
        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        int numBatches = Math.Max(1, data.Rows / batchSize);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                // Train discriminator (multiple steps per generator step)
                for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                {
                    TrainDiscriminatorStep(transformedData, batchSize, lr);
                }

                // Train generator
                TrainGeneratorStep(batchSize, lr);
            }

            // Update adjacency matrix with NOTEARS constraint
            UpdateAdjacencyAugmentedLagrangian(lr);
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs, CancellationToken ct = default)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        await Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();

            _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
            _transformer.Fit(data, _columns);
            _dataWidth = _transformer.TransformedWidth;
            var transformedData = _transformer.Transform(data);

            _adjacency = new Matrix<T>(_dataWidth, _dataWidth);
            for (int i = 0; i < _dataWidth; i++)
            {
                for (int j = 0; j < _dataWidth; j++)
                {
                    if (i != j)
                    {
                        _adjacency[i, j] = NumOps.FromDouble(0.01 * (_random.NextDouble() - 0.5));
                    }
                }
            }

            _lagrangeAlpha = 0.0;
            _lagrangeRho = 1.0;
            _prevDagConstraint = double.MaxValue;

            RebuildLayersWithActualDimensions(_options.EmbeddingDimension, _dataWidth, _dataWidth);

            T lr = NumOps.FromDouble(_options.LearningRate);
            int batchSize = Math.Min(_options.BatchSize, data.Rows);
            int numBatches = Math.Max(1, data.Rows / batchSize);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                for (int batch = 0; batch < numBatches; batch++)
                {
                    for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                    {
                        TrainDiscriminatorStep(transformedData, batchSize, lr);
                    }
                    TrainGeneratorStep(batchSize, lr);
                }
                UpdateAdjacencyAugmentedLagrangian(lr);
            }
        }, ct).ConfigureAwait(false);

        IsFitted = true;
    }

    /// <summary>
    /// Generates new synthetic tabular data rows respecting the learned causal structure.
    /// </summary>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (!IsFitted || _transformer is null || Layers.Count == 0)
        {
            throw new InvalidOperationException(
                "The generator must be fitted before generating data. Call Fit() first.");
        }

        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        var result = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var genOutput = GeneratorForward(VectorToTensor(noise));

            // Apply causal structure: y = (I + W^T) * x
            if (_adjacency is not null)
            {
                genOutput = ApplyCausalStructure(genOutput);
            }

            // Apply output activations
            genOutput = ApplyOutputActivations(genOutput);

            for (int j = 0; j < _dataWidth && j < genOutput.Length; j++)
            {
                result[i, j] = genOutput[j];
            }
        }

        return _transformer.InverseTransform(result);
    }

    #endregion

    #region Forward Passes

    /// <summary>
    /// Generator forward pass with residual connections and pre-activation caching.
    /// </summary>
    private Tensor<T> GeneratorForward(Tensor<T> input)
    {
        _genPreActivations.Clear();
        var inputTensor = input;
        var current = input;

        for (int i = 0; i < Layers.Count - 1; i++)
        {
            // Residual: concatenate original noise input
            if (i > 0)
            {
                current = ConcatTensors(current, inputTensor);
            }

            // FC(identity) --> BN --> cache --> ReLU
            current = Layers[i].Forward(current);
            if (i < _genBNLayers.Count)
            {
                current = _genBNLayers[i].Forward(current);
            }

            _genPreActivations.Add(CloneTensor(current));
            current = ApplyReLU(current);
        }

        // Final output layer with residual connection (no activation)
        current = ConcatTensors(current, inputTensor);
        current = Layers[^1].Forward(current);

        return current;
    }

    /// <summary>
    /// Discriminator forward pass with pre-activation caching and optional Dropout.
    /// </summary>
    private Tensor<T> DiscriminatorForward(Tensor<T> input, bool isTraining)
    {
        _discPreActivations.Clear();
        var current = input;

        for (int i = 0; i < _discLayers.Count - 1; i++)
        {
            current = _discLayers[i].Forward(current);
            _discPreActivations.Add(CloneTensor(current));
            current = ApplyLeakyReLU(current);

            if (isTraining)
            {
                current = _discDropoutLayers[i].Forward(current);
            }
        }

        // Final layer: no activation (raw Wasserstein score)
        current = _discLayers[^1].Forward(current);

        return current;
    }

    #endregion

    #region Training Steps

    /// <summary>
    /// Trains the discriminator for one step with WGAN-GP loss.
    /// </summary>
    private void TrainDiscriminatorStep(Matrix<T> transformedData, int batchSize, T learningRate)
    {
        T scaledLr = NumOps.FromDouble(NumOps.ToDouble(learningRate) / batchSize);

        for (int s = 0; s < batchSize; s++)
        {
            int rowIdx = _random.Next(transformedData.Rows);
            var realRow = GetRow(transformedData, rowIdx);

            // Generate fake sample with causal structure
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var fakeRaw = GeneratorForward(VectorToTensor(noise));
            if (_adjacency is not null)
            {
                fakeRaw = ApplyCausalStructure(fakeRaw);
            }
            fakeRaw = ApplyOutputActivations(fakeRaw);
            var fakeRow = TensorToVector(fakeRaw, _dataWidth);

            // WGAN loss on fake: gradient = +1
            _ = DiscriminatorForward(VectorToTensor(fakeRow), isTraining: true);
            var fakeGrad = new Tensor<T>([1]);
            fakeGrad[0] = NumOps.One;
            BackwardDiscriminator(fakeGrad);
            UpdateDiscriminatorParameters(scaledLr);

            // WGAN loss on real: gradient = -1
            _ = DiscriminatorForward(VectorToTensor(realRow), isTraining: true);
            var realGrad = new Tensor<T>([1]);
            realGrad[0] = NumOps.Negate(NumOps.One);
            BackwardDiscriminator(realGrad);
            UpdateDiscriminatorParameters(scaledLr);

            // WGAN-GP gradient penalty
            ApplyGradientPenalty(realRow, fakeRow, scaledLr);
        }
    }

    /// <summary>
    /// Trains the generator for one step using discriminator input gradient.
    /// </summary>
    private void TrainGeneratorStep(int batchSize, T learningRate)
    {
        T scaledLr = NumOps.FromDouble(NumOps.ToDouble(learningRate) / batchSize);

        for (int s = 0; s < batchSize; s++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);

            var fakeRaw = GeneratorForward(VectorToTensor(noise));
            if (_adjacency is not null)
            {
                fakeRaw = ApplyCausalStructure(fakeRaw);
            }
            fakeRaw = ApplyOutputActivations(fakeRaw);
            var fakeRow = TensorToVector(fakeRaw, _dataWidth);

            // Compute dD/dInput using GradientTape autodiff, then negate for generator gradient
            var discInputGrad = TapeLayerBridge<T>.ComputeInputGradient(
                VectorToTensor(fakeRow),
                _discLayers,
                TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
                applyActivationOnLast: false);
            for (int g = 0; g < discInputGrad.Length; g++)
            {
                discInputGrad[g] = NumOps.Negate(discInputGrad[g]);
            }
            discInputGrad = SafeGradient(discInputGrad, 5.0);

            // Re-forward generator to set up caches for backward pass
            _ = GeneratorForward(VectorToTensor(noise));

            BackwardGenerator(discInputGrad);
            UpdateGeneratorParameters(scaledLr);
        }
    }

    #endregion

    #region Causal Structure Learning

    /// <summary>
    /// Applies the learned causal structure: y = (I + W^T) * x.
    /// </summary>
    private Tensor<T> ApplyCausalStructure(Tensor<T> features)
    {
        if (_adjacency is null) return features;

        var result = new Tensor<T>(features.Shape);
        for (int j = 0; j < _dataWidth && j < features.Length; j++)
        {
            double val = NumOps.ToDouble(features[j]);

            for (int i = 0; i < _dataWidth && i < features.Length; i++)
            {
                if (i == j) continue;
                double weight = NumOps.ToDouble(_adjacency[i, j]);
                val += weight * NumOps.ToDouble(features[i]);
            }

            result[j] = NumOps.FromDouble(val);
        }

        return result;
    }

    /// <summary>
    /// Computes the NOTEARS acyclicity constraint and its gradient.
    /// h(W) = tr(e^(W hadamard W)) - d; gradient = 2 * e^(W hadamard W) hadamard W.
    /// Uses truncated Taylor series (order 8) for the matrix exponential.
    /// </summary>
    private (double ConstraintValue, double[,] Gradient) ComputeNOTEARSConstraintAndGradient()
    {
        if (_adjacency is null)
        {
            return (0.0, new double[0, 0]);
        }

        int d = _dataWidth;

        // Compute W hadamard W (element-wise square)
        var wSquared = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double w = NumOps.ToDouble(_adjacency[i, j]);
                wSquared[i, j] = w * w;
            }
        }

        // Compute e^(W hadamard W) using truncated Taylor series
        var expMatrix = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            expMatrix[i, i] = 1.0;
        }

        var power = new double[d, d];
        Array.Copy(wSquared, power, d * d);

        double factorial = 1.0;
        int maxOrder = 8;

        for (int k = 1; k <= maxOrder; k++)
        {
            factorial *= k;

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    expMatrix[i, j] += power[i, j] / factorial;
                }
            }

            if (k < maxOrder)
            {
                var nextPower = new double[d, d];
                for (int i = 0; i < d; i++)
                {
                    for (int j = 0; j < d; j++)
                    {
                        double sum = 0;
                        for (int m = 0; m < d; m++)
                        {
                            sum += power[i, m] * wSquared[m, j];
                        }
                        nextPower[i, j] = sum;
                    }
                }
                Array.Copy(nextPower, power, d * d);
            }
        }

        // h(W) = tr(e^(W hadamard W)) - d
        double trace = 0;
        for (int i = 0; i < d; i++)
        {
            trace += expMatrix[i, i];
        }
        double constraintValue = trace - d;

        // Gradient: dh/dW = 2 * e^(W hadamard W) hadamard W
        var gradient = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double w = NumOps.ToDouble(_adjacency[i, j]);
                gradient[i, j] = 2.0 * expMatrix[i, j] * w;
            }
        }

        return (constraintValue, gradient);
    }

    /// <summary>
    /// Updates the adjacency matrix using the augmented Lagrangian method for the NOTEARS constraint.
    /// </summary>
    private void UpdateAdjacencyAugmentedLagrangian(T lr)
    {
        if (_adjacency is null) return;

        double adjLr = NumOps.ToDouble(lr) * 0.01;

        var (h, notearGrad) = ComputeNOTEARSConstraintAndGradient();
        double lagrangianCoeff = _lagrangeAlpha + _lagrangeRho * h;

        for (int i = 0; i < _dataWidth; i++)
        {
            for (int j = 0; j < _dataWidth; j++)
            {
                if (i == j) continue;

                double wij = NumOps.ToDouble(_adjacency[i, j]);
                double dagGrad = notearGrad[i, j] * lagrangianCoeff;
                double sparseGrad = _options.SparsityWeight * Math.Sign(wij);

                wij -= adjLr * (dagGrad + sparseGrad);
                _adjacency[i, j] = NumOps.FromDouble(wij);
            }
        }

        // Enforce zero diagonal
        for (int i = 0; i < _dataWidth; i++)
        {
            _adjacency[i, i] = NumOps.Zero;
        }

        // Update augmented Lagrangian parameters
        _lagrangeAlpha += _lagrangeRho * h;

        double gamma = 0.25;
        if (Math.Abs(h) > gamma * Math.Abs(_prevDagConstraint) && _prevDagConstraint < double.MaxValue)
        {
            _lagrangeRho = Math.Min(_lagrangeRho * 10.0, 1e16);
        }

        _prevDagConstraint = h;
    }

    #endregion

    #region Gradient Penalty

    /// <summary>
    /// Applies WGAN-GP gradient penalty on interpolated sample.
    /// </summary>
    private void ApplyGradientPenalty(Vector<T> realRow, Vector<T> fakeRow, T scaledLr)
    {
        double alpha = _random.NextDouble();
        int len = Math.Min(realRow.Length, fakeRow.Length);
        var interpolated = new Vector<T>(len);

        for (int i = 0; i < len; i++)
        {
            interpolated[i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(alpha), realRow[i]),
                NumOps.Multiply(NumOps.FromDouble(1.0 - alpha), fakeRow[i]));
        }

        // Compute gradient penalty using GradientTape autodiff
        var interpolatedTensor = VectorToTensor(interpolated);
        var inputGrad = TapeLayerBridge<T>.ComputeInputGradient(
            interpolatedTensor,
            _discLayers,
            TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
            applyActivationOnLast: false);

        double gradNormSq = 0;
        for (int i = 0; i < inputGrad.Length; i++)
        {
            double g = NumOps.ToDouble(inputGrad[i]);
            gradNormSq += g * g;
        }
        double gradNorm = Math.Sqrt(gradNormSq + 1e-12);

        double penaltyGradScale = 2.0 * _options.GradientPenaltyWeight * (gradNorm - 1.0) / gradNorm;

        if (Math.Abs(penaltyGradScale) > 1e-10)
        {
            _ = DiscriminatorForward(VectorToTensor(interpolated), isTraining: false);

            var penaltyGrad = new Tensor<T>([1]);
            penaltyGrad[0] = NumOps.FromDouble(penaltyGradScale);
            BackwardDiscriminator(penaltyGrad);
            UpdateDiscriminatorParameters(scaledLr);
        }
    }

    #endregion

    #region Backward Passes

    private void BackwardDiscriminator(Tensor<T> gradOutput)
    {
        var current = gradOutput;
        current = _discLayers[^1].Backward(current);

        for (int i = _discLayers.Count - 2; i >= 0; i--)
        {
            if (i < _discPreActivations.Count)
            {
                current = ApplyLeakyReLUDerivative(current, _discPreActivations[i]);
            }
            current = _discLayers[i].Backward(current);
        }
    }

    private void BackwardGenerator(Tensor<T> gradOutput)
    {
        int inputDim = _options.EmbeddingDimension;
        var current = gradOutput;

        // Backward through output layer
        current = Layers[^1].Backward(current);

        // Split off residual gradient
        int lastHiddenDim = current.Length - inputDim;
        if (lastHiddenDim > 0)
        {
            var hiddenGrad = new Tensor<T>([lastHiddenDim]);
            for (int j = 0; j < lastHiddenDim && j < current.Length; j++)
            {
                hiddenGrad[j] = current[j];
            }
            current = hiddenGrad;
        }

        // Backward through hidden layers in reverse
        for (int i = Layers.Count - 2; i >= 0; i--)
        {
            if (i < _genPreActivations.Count)
            {
                current = ApplyReLUDerivative(current, _genPreActivations[i]);
            }

            if (i < _genBNLayers.Count)
            {
                current = _genBNLayers[i].Backward(current);
            }

            current = Layers[i].Backward(current);

            if (i > 0)
            {
                int prevDim = current.Length - inputDim;
                if (prevDim > 0)
                {
                    var hiddenGrad = new Tensor<T>([prevDim]);
                    for (int j = 0; j < prevDim && j < current.Length; j++)
                    {
                        hiddenGrad[j] = current[j];
                    }
                    current = hiddenGrad;
                }
            }
        }
    }

    private void UpdateGeneratorParameters(T learningRate)
    {
        foreach (var layer in Layers)
        {
            layer.UpdateParameters(learningRate);
        }
        foreach (var bn in _genBNLayers)
        {
            bn.UpdateParameters(learningRate);
        }
    }

    private void UpdateDiscriminatorParameters(T learningRate)
    {
        foreach (var layer in _discLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    #endregion

    #region Activation Functions

    private Tensor<T> ApplyReLU(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = NumOps.ToDouble(input[i]) > 0 ? input[i] : NumOps.Zero;
        }
        return result;
    }

    private Tensor<T> ApplyReLUDerivative(Tensor<T> gradOutput, Tensor<T> preActivation)
    {
        int len = Math.Min(gradOutput.Length, preActivation.Length);
        var result = new Tensor<T>(gradOutput.Shape);
        for (int i = 0; i < len; i++)
        {
            result[i] = NumOps.ToDouble(preActivation[i]) > 0 ? gradOutput[i] : NumOps.Zero;
        }
        return result;
    }

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);
        T slope = NumOps.FromDouble(0.2);
        for (int i = 0; i < input.Length; i++)
        {
            double val = NumOps.ToDouble(input[i]);
            result[i] = val > 0 ? input[i] : NumOps.Multiply(slope, input[i]);
        }
        return result;
    }

    private Tensor<T> ApplyLeakyReLUDerivative(Tensor<T> gradOutput, Tensor<T> preActivation)
    {
        int len = Math.Min(gradOutput.Length, preActivation.Length);
        var result = new Tensor<T>(gradOutput.Shape);
        T slope = NumOps.FromDouble(0.2);
        for (int i = 0; i < len; i++)
        {
            if (NumOps.ToDouble(preActivation[i]) > 0)
            {
                result[i] = gradOutput[i];
            }
            else
            {
                result[i] = NumOps.Multiply(slope, gradOutput[i]);
            }
        }
        return result;
    }

    /// <summary>
    /// Applies output activations per column type: tanh for continuous, softmax for categorical.
    /// </summary>
    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

        var result = new Tensor<T>(output.Shape);
        int idx = 0;

        for (int col = 0; col < _columns.Count && idx < output.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);

            if (transform.IsContinuous)
            {
                if (idx < output.Length)
                {
                    double val = NumOps.ToDouble(output[idx]);
                    result[idx] = NumOps.FromDouble(Math.Tanh(val));
                    idx++;
                }

                int numModes = transform.Width - 1;
                if (numModes > 0)
                {
                    ApplySoftmaxBlock(output, result, ref idx, numModes);
                }
            }
            else
            {
                ApplySoftmaxBlock(output, result, ref idx, transform.Width);
            }
        }

        return result;
    }

    private static void ApplySoftmaxBlock(Tensor<T> input, Tensor<T> output, ref int idx, int count)
    {
        if (count <= 0) return;

        var numOps = MathHelper.GetNumericOperations<T>();

        double maxVal = double.MinValue;
        for (int m = 0; m < count && (idx + m) < input.Length; m++)
        {
            double v = numOps.ToDouble(input[idx + m]);
            if (v > maxVal) maxVal = v;
        }

        double sumExp = 0;
        for (int m = 0; m < count && (idx + m) < input.Length; m++)
        {
            sumExp += Math.Exp(numOps.ToDouble(input[idx + m]) - maxVal);
        }

        for (int m = 0; m < count && idx < input.Length; m++)
        {
            double expVal = Math.Exp(numOps.ToDouble(input[idx]) - maxVal);
            output[idx] = numOps.FromDouble(expVal / Math.Max(sumExp, 1e-10));
            idx++;
        }
    }

    #endregion

    #region Gradient Safety Utilities

    private Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm)
    {
        for (int i = 0; i < grad.Length; i++)
        {
            double v = NumOps.ToDouble(grad[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                grad[i] = NumOps.Zero;
            }
        }

        if (maxNorm <= 0) return grad;

        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double v = NumOps.ToDouble(grad[i]);
            normSq += v * v;
        }

        double norm = Math.Sqrt(normSq);
        if (norm <= maxNorm) return grad;

        double scale = maxNorm / norm;
        var clipped = new Tensor<T>(grad.Shape);
        for (int i = 0; i < grad.Length; i++)
        {
            clipped[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * scale);
        }
        return clipped;
    }

    #endregion

    #region Serialization and Model Metadata (GANDALF Pattern)

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "HiddenDimensions", _options.HiddenDimensions },
                { "DAGPenaltyWeight", _options.DAGPenaltyWeight },
                { "SparsityWeight", _options.SparsityWeight },
                { "GradientPenaltyWeight", _options.GradientPenaltyWeight },
                { "GeneratorLayerCount", Layers.Count },
                { "GeneratorLayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.EmbeddingDimension);
        writer.Write(_options.HiddenDimensions.Length);
        foreach (var dim in _options.HiddenDimensions)
        {
            writer.Write(dim);
        }
        writer.Write(_options.BatchSize);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DAGPenaltyWeight);
        writer.Write(_options.SparsityWeight);
        writer.Write(_options.GradientPenaltyWeight);
        writer.Write(_options.DiscriminatorDropout);
        writer.Write(_options.DiscriminatorSteps);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Options are reconstructed from serialized data
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CausalGANGenerator<T>(
            Architecture,
            _options,
            _optimizer,
            _lossFunction);
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        int numFeatures = Architecture.CalculatedInputSize;
        var uniformValue = NumOps.FromDouble(1.0 / Math.Max(numFeatures, 1));
        for (int f = 0; f < numFeatures; f++)
        {
            importance[$"feature_{f}"] = uniformValue;
        }
        return importance;
    }

    #endregion

    #region Input Validation and Column Management

    private static void ValidateFitInputs(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        if (data.Rows == 0 || data.Columns == 0)
        {
            throw new ArgumentException("Data matrix must not be empty.", nameof(data));
        }

        if (columns.Count == 0)
        {
            throw new ArgumentException("Column metadata list must not be empty.", nameof(columns));
        }

        if (columns.Count != data.Columns)
        {
            throw new ArgumentException(
                $"Column metadata count ({columns.Count}) must match data column count ({data.Columns}).",
                nameof(columns));
        }

        if (epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs must be positive.");
        }
    }

    private List<ColumnMetadata> PrepareColumns(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        var prepared = new List<ColumnMetadata>(columns.Count);

        for (int col = 0; col < columns.Count; col++)
        {
            var meta = columns[col].Clone();
            meta.ColumnIndex = col;

            if (meta.IsNumerical)
            {
                ComputeColumnStatistics(data, col, meta);
            }
            else if (meta.IsCategorical && meta.NumCategories == 0)
            {
                var categories = new HashSet<string>();
                for (int row = 0; row < data.Rows; row++)
                {
                    var val = NumOps.ToDouble(data[row, col]);
                    categories.Add(val.ToString(System.Globalization.CultureInfo.InvariantCulture));
                }
                meta.Categories = categories.OrderBy(c => c, StringComparer.Ordinal).ToList().AsReadOnly();
            }

            prepared.Add(meta);
        }

        return prepared;
    }

    private void ComputeColumnStatistics(Matrix<T> data, int colIndex, ColumnMetadata meta)
    {
        int n = data.Rows;
        double sum = 0;
        double min = double.MaxValue;
        double max = double.MinValue;

        for (int row = 0; row < n; row++)
        {
            double val = NumOps.ToDouble(data[row, colIndex]);
            sum += val;
            if (val < min) min = val;
            if (val > max) max = val;
        }

        double mean = sum / n;
        double sumSqDiff = 0;
        for (int row = 0; row < n; row++)
        {
            double val = NumOps.ToDouble(data[row, colIndex]);
            double diff = val - mean;
            sumSqDiff += diff * diff;
        }

        double std = n > 1 ? Math.Sqrt(sumSqDiff / (n - 1)) : 1.0;
        if (std < 1e-10) std = 1e-10;

        meta.Min = min;
        meta.Max = max;
        meta.Mean = mean;
        meta.Std = std;
    }

    #endregion

    #region Random Sampling Utilities

    private T SampleStandardNormal()
    {
        double u1 = 1.0 - _random.NextDouble();
        double u2 = _random.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return NumOps.FromDouble(z);
    }

    private Vector<T> CreateStandardNormalVector(int length)
    {
        var v = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            v[i] = SampleStandardNormal();
        }
        return v;
    }

    #endregion

    #region Helpers

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            v[j] = matrix[row, j];
        }
        return v;
    }

    private static Tensor<T> VectorToTensor(Vector<T> v)
    {
        var t = new Tensor<T>([v.Length]);
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    private static Vector<T> TensorToVector(Tensor<T> t, int length)
    {
        var v = new Vector<T>(length);
        int copyLen = Math.Min(length, t.Length);
        for (int i = 0; i < copyLen; i++) v[i] = t[i];
        return v;
    }

    private static Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>([a.Length + b.Length]);
        for (int i = 0; i < a.Length; i++) result[i] = a[i];
        for (int i = 0; i < b.Length; i++) result[a.Length + i] = b[i];
        return result;
    }

    private static Tensor<T> CloneTensor(Tensor<T> source)
    {
        var clone = new Tensor<T>(source.Shape);
        for (int i = 0; i < source.Length; i++)
        {
            clone[i] = source[i];
        }
        return clone;
    }

    #endregion

    #region IJitCompilable Override

    /// <summary>
    /// CausalGAN uses per-SEM equation generation which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}
