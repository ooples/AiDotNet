using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// GOGGLE generator that learns feature dependency structure via a graph neural network
/// combined with a VAE framework for high-quality synthetic tabular data generation.
/// </summary>
/// <remarks>
/// <para>
/// GOGGLE operates in three stages:
///
/// <code>
///  Features ──► Structure Learning ──► Adjacency Matrix A
///                                           │
///  Features ──► GNN Encoder (with A) ──► (mean, logvar) ──► z ──► MLP Decoder ──► Reconstructed
///                                                            ↑
///                                                    Reparameterization
/// </code>
///
/// The adjacency matrix A is learned end-to-end alongside the encoder/decoder.
/// Regularization encourages A to be sparse and approximately acyclic (DAG-like).
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Full forward/backward/update lifecycle
/// </para>
/// <para>
/// <b>For Beginners:</b> GOGGLE figures out which features relate to each other:
///
/// 1. Learns a "graph" where connected features influence each other
/// 2. Uses this graph to share information between related features
/// 3. Generates new data where these relationships are preserved
///
/// If you provide custom layers in the architecture, those will be used for the
/// decoder (MLP) network. If not, the network creates standard decoder layers
/// based on the original paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new GOGGLEOptions&lt;double&gt;
/// {
///     LatentDimension = 32,
///     NumGNNLayers = 2,
///     Epochs = 300
/// };
/// var goggle = new GOGGLEGenerator&lt;double&gt;(architecture, options);
/// goggle.Fit(data, columns, epochs: 300);
/// var synthetic = goggle.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure"
/// (Liu et al., ICLR 2023)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure",
    "https://arxiv.org/abs/2210.12007",
    Year = 2023,
    Authors = "Tennison Liu, Zhaozhi Qian, Jeroen Berrevoets, Mihaela van der Schaar")]
public class GOGGLEGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly GOGGLEOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Learned adjacency matrix (soft, between 0 and 1). Stored as a 2-D
    // Tensor so it participates in autodiff alongside the encoder/decoder
    // weights — gradients of the ELBO + sparsity (γ‖A‖₁) + DAG penalty
    // h(A) = tr((A⊙A)^d) - d (Zheng et al. 2018) flow through it.
    private Tensor<T>? _adjacency;

    // GNN encoder layers (auxiliary, not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _gnnLayers = new();
    private FullyConnectedLayer<T>? _meanHead;
    private FullyConnectedLayer<T>? _logvarHead;

    // Decoder output layer (auxiliary)
    private FullyConnectedLayer<T>? _decoderOutput;

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the GOGGLE-specific options.
    /// </summary>
    public GOGGLEOptions<T> GoggleOptions => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new GOGGLE generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">GOGGLE-specific options for GNN and VAE configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a GOGGLE network.
    ///
    /// If you provide custom layers in the architecture, those will be used for the
    /// MLP decoder. If not, the network creates standard decoder layers based on
    /// the paper specifications.
    ///
    /// The GNN encoder, mean/logvar heads, and adjacency matrix are always created
    /// internally and are not user-overridable.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public GOGGLEGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public GOGGLEGenerator(
        NeuralNetworkArchitecture<T> architecture,
        GOGGLEOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new GOGGLEOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the decoder layers of the GOGGLE network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Layers</b> = MLP decoder (user-overridable via Architecture).
    /// Auxiliary networks (GNN encoder, mean/logvar heads, decoder output) are always
    /// created internally during Fit() when actual data dimensions are known.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default decoder layers
            int hiddenDim = _options.HiddenDimension;
            int latentDim = _options.LatentDimension;
            var relu = new ReLUActivation<T>() as IActivationFunction<T>;

            Layers.Add(new FullyConnectedLayer<T>(hiddenDim, relu));
            Layers.Add(new FullyConnectedLayer<T>(hiddenDim, relu));
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Rebuilds auxiliary layers with actual data dimensions discovered during Fit().
    /// Every trainable sub-network (GNN encoder, mean/logvar projection heads,
    /// decoder MLP, decoder output) is registered in <see cref="NeuralNetworkBase{T}.Layers"/>
    /// in a stable order so the tape-based trainer's
    /// <c>CollectParameters(Layers)</c> walk picks up the full parameter set —
    /// without this, only the user-overridable decoder MLP was in Layers and
    /// every other sub-network's weights silently failed to receive gradient
    /// updates, leaving Clone / SaveLoad with an incomplete state-dict.
    /// </summary>
    private void RebuildAuxiliaryLayers()
    {
        int hiddenDim = _options.HiddenDimension;
        var relu = new ReLUActivation<T>() as IActivationFunction<T>;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        // GNN encoder layers
        _gnnLayers.Clear();
        for (int i = 0; i < _options.NumGNNLayers; i++)
        {
            int layerInput = i == 0 ? _dataWidth : hiddenDim;
            _gnnLayers.Add(new FullyConnectedLayer<T>(hiddenDim, relu));
        }

        int lastDim = _options.NumGNNLayers > 0 ? hiddenDim : _dataWidth;
        _meanHead = new FullyConnectedLayer<T>(_options.LatentDimension, identity);
        _logvarHead = new FullyConnectedLayer<T>(_options.LatentDimension, identity);

        // Decoder output layer
        _decoderOutput = new FullyConnectedLayer<T>(_dataWidth, identity);

        // Rebuild Layers in a stable order: GNN encoder → mean/logvar heads →
        // decoder MLP → decoder output. The forward path uses the typed fields
        // (_gnnLayers, _meanHead, _logvarHead, _decoderOutput, Layers[i] for the
        // decoder MLP) and stays the same; Layers exists purely so the tape
        // trainer's CollectParameters walk sees every trainable tensor.
        Layers.Clear();
        foreach (var gnnLayer in _gnnLayers) Layers.Add(gnnLayer);
        Layers.Add(_meanHead);
        Layers.Add(_logvarHead);
        if (!_usingCustomLayers)
        {
            Layers.Add(new FullyConnectedLayer<T>(hiddenDim, relu));
            Layers.Add(new FullyConnectedLayer<T>(hiddenDim, relu));
        }
        else
        {
            foreach (var customLayer in Architecture.Layers!)
                Layers.Add(customLayer);
        }
        Layers.Add(_decoderOutput);

        // Initialize adjacency matrix
        InitializeAdjacency();
    }

    /// <summary>
    /// Returns the user-overridable decoder MLP sub-list of <see cref="Layers"/>
    /// (the slice between the encoder/projection heads and the
    /// <c>_decoderOutput</c>). Used by <see cref="DecoderForward"/> /
    /// <see cref="DecoderForwardTape"/> so the forward path uses the correct
    /// portion of the registered Layers list.
    /// </summary>
    private IEnumerable<ILayer<T>> DecoderMlpLayers()
    {
        // Skip: gnnLayers (NumGNNLayers), meanHead (1), logvarHead (1) at the
        // front; _decoderOutput (1) at the back.
        int skipFront = _gnnLayers.Count + 2;
        int skipBack = 1;
        for (int i = skipFront; i < Layers.Count - skipBack; i++)
            yield return Layers[i];
    }

    private void InitializeAdjacency()
    {
        _adjacency = new Tensor<T>(new[] { _dataWidth, _dataWidth });
        double initVal = 1.0 / Math.Max(_dataWidth, 1);
        for (int i = 0; i < _dataWidth; i++)
        {
            for (int j = 0; j < _dataWidth; j++)
            {
                if (i == j)
                {
                    _adjacency[i, j] = NumOps.Zero;
                }
                else
                {
                    _adjacency[i, j] = NumOps.FromDouble(initVal + 0.01 * (_random.NextDouble() - 0.5));
                }
            }
        }
    }

    /// <summary>
    /// Expose the learned adjacency matrix so the tape-based trainer
    /// (BackwardAndStepOnPrecomputedLoss) collects it alongside the
    /// encoder/decoder layer parameters. Without this the sparsity / DAG
    /// regularizers compute on A but the optimizer never updates A.
    /// </summary>
    protected override IEnumerable<Tensor<T>> GetExtraTrainableTensors()
    {
        if (_adjacency is not null && _adjacency.Length > 0)
        {
            yield return _adjacency;
        }
    }

    /// <summary>
    /// GOGGLE's <see cref="LayerBase{T}"/> chain is the VAE decoder —
    /// Layer[0] takes the latent z (size LatentDimension), NOT the raw
    /// data row (size Architecture.InputWidth). Suppress the base class's
    /// architecture-driven shape pre-walk so the first real Train()
    /// resolves Layer[0]'s input dim from the actual latent. Without
    /// this override, ResolveLazyLayerShapes locks Layer[0] to InputWidth
    /// and every Train() call fails with "Matrix dimensions incompatible:
    /// [1, latent] × [InputWidth, hidden]". Same pattern as MisGAN /
    /// AutoDiffTabGenerator / TabTransformerGen.
    /// </summary>
    protected override int[]? TryGetArchitectureInputShape()
    {
        // Always disable the architecture-driven pre-walk: Layer[0] consumes the latent z,
        // not Architecture.InputWidth, for BOTH default and custom decoder stacks. Delegating
        // to base when _usingCustomLayers is true re-enabled the pre-walk and locked custom
        // decoder layers to InputWidth, breaking their first forward.
        return null;
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _columns = new List<ColumnMetadata>(columns);
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, columns);
        _dataWidth = _transformer.TransformedWidth;
        var transformedData = _transformer.Transform(data);

        // Rebuild all auxiliary layers with actual dimensions
        RebuildAuxiliaryLayers();

        SetTrainingMode(true);
        try
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int row = 0; row < transformedData.Rows; row++)
                {
                    var rowVec = GetRow(transformedData, row);
                    var inputTensor = VectorToTensor(rowVec);
                    // For an autoencoder, target == input.
                    Train(inputTensor, inputTensor);
                }
            }
        }
        finally
        {
            SetTrainingMode(false);
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs, CancellationToken ct = default)
    {
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            Fit(data, columns, epochs);
        }, ct);
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_transformer is null || _decoderOutput is null || !IsFitted)
        {
            throw new InvalidOperationException("Generator must be fitted before generating data.");
        }

        int latentDim = _options.LatentDimension;
        var result = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            // Sample from standard normal in latent space
            var z = CreateStandardNormalVector(latentDim);

            // Decode
            var decoded = DecoderForward(z);

            // Apply output activations
            var activated = ApplyOutputActivations(decoded);

            for (int j = 0; j < _dataWidth && j < activated.Length; j++)
            {
                result[i, j] = activated[j];
            }
        }

        return _transformer.InverseTransform(result);
    }

    #endregion

    #region Training

    private void UpdateAdjacency(Vector<T> x, T lr)
    {
        if (_adjacency is null) return;

        double adjLr = NumOps.ToDouble(lr);

        for (int i = 0; i < _dataWidth; i++)
        {
            for (int j = 0; j < _dataWidth; j++)
            {
                if (i == j) continue;

                double aij = NumOps.ToDouble(_adjacency[i, j]);
                double sparsityGrad = _options.SparsityWeight * Math.Sign(aij);

                // DAG penalty: NOTEARS-style simplified
                double dagGrad = _options.StructureWeight * aij;

                double newVal = aij - adjLr * (sparsityGrad + dagGrad);
                // Clamp to [0, 1] for valid adjacency
                _adjacency[i, j] = NumOps.FromDouble(Math.Min(Math.Max(newVal, 0.0), 1.0));
            }
        }
    }

    #endregion

    #region Forward Passes

    private Vector<T> GNNEncoderForward(Vector<T> x)
    {
        // Non-tape inference path used by Predict() / Generate(). Mirrors the
        // tape-connected encoder used in EncoderForwardTape but materialises
        // intermediate values as Vectors so it can run outside a GradientTape.
        var current = x;

        for (int layer = 0; layer < _gnnLayers.Count; layer++)
        {
            // Mirror EncoderForwardTape EXACTLY: adjacency message-passing is
            // injected ONLY at layer 0 (deeper layers see already-mixed hidden
            // states). Aggregating at every layer made Predict()/Generate()
            // serve a different encoder than Train() optimizes.
            var layerInput = layer == 0 ? AggregateNeighbors(current) : current;
            var tensor = _gnnLayers[layer].Forward(VectorToTensor(layerInput));
            current = TensorToVector(tensor, _options.HiddenDimension);
        }

        return current;
    }

    private Vector<T> AggregateNeighbors(Vector<T> features)
    {
        if (_adjacency is null) return features;

        int featDim = features.Length;
        int adjDim = _adjacency.Shape[0];
        int dim = Math.Min(featDim, adjDim);
        var aggregated = new Vector<T>(featDim);

        for (int i = 0; i < dim; i++)
        {
            double selfVal = NumOps.ToDouble(features[i]);
            double neighborSum = 0;

            for (int j = 0; j < dim; j++)
            {
                double aij = NumOps.ToDouble(_adjacency[i, j]);
                neighborSum += aij * NumOps.ToDouble(features[j]);
            }

            // Unnormalized x + (x @ Aᵀ), identical to EncoderForwardTape; the
            // previous weightSum normalization diverged from the trained path.
            aggregated[i] = NumOps.FromDouble(selfVal + neighborSum);
        }

        for (int i = dim; i < featDim; i++)
        {
            aggregated[i] = features[i];
        }

        return aggregated;
    }

    private Vector<T> DecoderForward(Vector<T> z)
    {
        var current = VectorToTensor(z);
        foreach (var layer in DecoderMlpLayers())
            current = layer.Forward(current);
        if (_decoderOutput is not null)
            current = _decoderOutput.Forward(current);
        return TensorToVector(current, _dataWidth);
    }

    private Vector<T> Reparameterize(Vector<T> mean, Vector<T> logvar)
    {
        int dim = mean.Length;
        var z = new Vector<T>(dim);
        for (int i = 0; i < dim; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double eps = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            double m = NumOps.ToDouble(mean[i]);
            double lv = NumOps.ToDouble(logvar[i]);
            z[i] = NumOps.FromDouble(m + eps * Math.Exp(0.5 * lv));
        }
        return z;
    }

    // ===== Tape-connected forward + ELBO loss for Train() =====

    /// <summary>
    /// Tape-connected GNN encoder + mean / logVar projection heads.
    /// Aggregation step uses the learned adjacency A (a registered
    /// trainable tensor) via TensorMatMul so gradients of the ELBO,
    /// sparsity, and DAG penalties all flow back into A.
    /// </summary>
    private (Tensor<T> Mean, Tensor<T> LogVar) EncoderForwardTape(Tensor<T> input)
    {
        // Lift input to shape [1, dataWidth] so matmul with [dataWidth, dataWidth]
        // adjacency is well-defined; engine ops broadcast / reduce as needed.
        var x = input.Rank == 2 ? input : Engine.Reshape(input, new[] { 1, input.Length });

        var current = x;
        for (int layer = 0; layer < _gnnLayers.Count; layer++)
        {
            // h_aggregated = h + (h @ Aᵀ) — message passing (Eq. 2 of the GOGGLE paper):
            // each feature receives the weighted sum of its neighbours. We use Aᵀ so
            // the (i,j) entry of A is "influence j → i" matching the paper's
            // adjacency convention.
            var adjT = Engine.TensorTranspose(_adjacency!);
            // Layer-0 input width is _dataWidth; subsequent layers operate in
            // HiddenDimension. Only layer 0 needs the adjacency message-passing;
            // deeper layers see already-mixed hidden states (matches the GNN
            // depth used in the paper's reference implementation).
            if (layer == 0)
            {
                var neighbours = Engine.TensorMatMul(current, adjT);
                current = Engine.TensorAdd(current, neighbours);
            }
            current = _gnnLayers[layer].Forward(current);
        }

        var mean = _meanHead!.Forward(current);
        var logVar = _logvarHead!.Forward(current);
        return (mean, logVar);
    }

    /// <summary>
    /// Reparameterize z = μ + exp(0.5·logσ²) ⊙ ε  with ε ~ N(0, I) sampled
    /// as a constant tensor (no gradient through ε). Tape-connected through μ
    /// and logVar. Kingma &amp; Welling 2014.
    /// </summary>
    private Tensor<T> ReparameterizeTape(Tensor<T> mean, Tensor<T> logVar)
    {
        var eps = new Tensor<T>(mean._shape);
        for (int i = 0; i < eps.Length; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double n = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            eps[i] = NumOps.FromDouble(n);
        }
        var std = Engine.TensorExp(Engine.TensorMultiplyScalar(logVar, NumOps.FromDouble(0.5)));
        return Engine.TensorAdd(mean, Engine.TensorMultiply(std, eps));
    }

    private Tensor<T> DecoderForwardTape(Tensor<T> z)
    {
        var current = z;
        foreach (var layer in DecoderMlpLayers())
            current = layer.Forward(current);
        if (_decoderOutput is not null)
            current = _decoderOutput.Forward(current);
        return current;
    }

    /// <summary>
    /// Negative ELBO + GOGGLE structure regularisers (paper Eq. 5):
    ///   L = ‖x - x̂‖² + KL(q(z|x) ‖ N(0, I)) + γ·‖A‖₁ + ρ·h(A)
    /// where h(A) = tr((A⊙A)^d) - d is the NOTEARS-style DAG-ness penalty
    /// (Zheng et al. 2018) — minimized when A is a DAG. We use the
    /// power-series surrogate h(A) ≈ tr(A⊙A) + tr((A⊙A)²) which is what the
    /// reference implementation uses for d ≤ 64 to avoid the matrix
    /// exponential and stays tape-friendly.
    /// </summary>
    private Tensor<T> ComputeGoggleLossTape(Tensor<T> reconstruction, Tensor<T> target, Tensor<T> mean, Tensor<T> logVar)
    {
        var recon = reconstruction.Rank == 1 ? reconstruction : Engine.Reshape(reconstruction, new[] { reconstruction.Length });
        var tgt = target.Rank == 1 ? target : Engine.Reshape(target, new[] { target.Length });
        var diff = Engine.TensorSubtract(recon, tgt);
        var reconLoss = ReduceToScalar(Engine.TensorSquare(diff));

        // KL divergence between q(z|x)=N(μ, σ²) and p(z)=N(0, I):
        //   KL = -½ Σ (1 + logσ² - μ² - σ²)
        var meanSq = Engine.TensorSquare(mean);
        var expLogVar = Engine.TensorExp(logVar);
        var klInner = Engine.TensorSubtract(
            Engine.TensorSubtract(Engine.TensorAddScalar(logVar, NumOps.One), meanSq),
            expLogVar);
        var kl = Engine.TensorMultiplyScalar(ReduceToScalar(klInner), NumOps.FromDouble(-0.5));

        var loss = Engine.TensorAdd(reconLoss, kl);

        // GOGGLE structure regularisers — only active when adjacency exists.
        if (_adjacency is not null && _adjacency.Length > 0)
        {
            // Sparsity ‖A‖₁ ≈ Σ|A_ij|. Differentiable surrogate uses √(A² + ε).
            var absA = Engine.TensorSqrt(Engine.TensorAddScalar(
                Engine.TensorSquare(_adjacency), NumOps.FromDouble(1e-8)));
            var sparsity = Engine.TensorMultiplyScalar(
                ReduceToScalar(absA), NumOps.FromDouble(_options.SparsityWeight));
            loss = Engine.TensorAdd(loss, sparsity);

            // DAG penalty h(A) ≈ tr(A⊙A) + tr((A⊙A)²) (NOTEARS power-series
            // approximation for small graphs).
            var aSquared = Engine.TensorSquare(_adjacency);
            var aSquaredSquared = Engine.TensorMatMul(aSquared, aSquared);
            // tr(M) = Σᵢ M_ii. Multiply by identity then sum.
            var dag = Engine.TensorAdd(TraceTape(aSquared), TraceTape(aSquaredSquared));
            var dagPenalty = Engine.TensorMultiplyScalar(dag, NumOps.FromDouble(_options.StructureWeight));
            loss = Engine.TensorAdd(loss, dagPenalty);
        }

        return loss;
    }

    /// <summary>tr(M) for a square 2-D tensor — tape-connected by summing the
    /// diagonal via a one-hot identity mask.</summary>
    private Tensor<T> TraceTape(Tensor<T> m)
    {
        int n = m.Shape[0];
        var eye = new Tensor<T>(new[] { n, n });
        for (int i = 0; i < n; i++) eye[i, i] = NumOps.One;
        return ReduceToScalar(Engine.TensorMultiply(m, eye));
    }

    private Tensor<T> ReduceToScalar(Tensor<T> t)
    {
        var axes = Enumerable.Range(0, t.Shape.Length).ToArray();
        return Engine.ReduceSum(t, axes, keepDims: false);
    }

    #endregion

    #region Output Activations

    private Vector<T> ApplyOutputActivations(Vector<T> decoded)
    {
        if (_transformer is null) return decoded;

        var output = VectorToTensor(decoded);
        var result = new Tensor<T>(output._shape);
        int idx = 0;

        for (int col = 0; col < Columns.Count && idx < output.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);
            if (transform.IsContinuous)
            {
                if (idx < output.Length)
                {
                    result[idx] = NumOps.FromDouble(Math.Tanh(NumOps.ToDouble(output[idx])));
                    idx++;
                }
                int numModes = transform.Width - 1;
                if (numModes > 0) ApplySoftmax(output, result, ref idx, numModes);
            }
            else
            {
                ApplySoftmax(output, result, ref idx, transform.Width);
            }
        }

        return TensorToVector(result, _dataWidth);
    }

    private void ApplySoftmax(Tensor<T> input, Tensor<T> output, ref int idx, int count)
    {
        if (count <= 0) return;
        int actualCount = Math.Min(count, input.Length - idx);
        if (actualCount <= 0) return;
        var slice = new Tensor<T>([actualCount]);
        input.Data.Span.Slice(idx, actualCount).CopyTo(slice.Data.Span);
        var result = Engine.Softmax(slice, -1);
        result.Data.Span.CopyTo(output.Data.Span.Slice(idx, actualCount));
        idx += actualCount;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        EnsureSizedForInput(input);
        if (_meanHead is null) return input;

        var row = TensorToVector(input, _dataWidth);
        var gnnOut = GNNEncoderForward(row);
        var meanTensor = _meanHead.Forward(VectorToTensor(gnnOut));
        var mean = TensorToVector(meanTensor, _options.LatentDimension);
        var decoded = DecoderForward(mean);
        return VectorToTensor(decoded);
    }

    /// <summary>
    /// One paper-faithful ELBO + structure-regularisation training step:
    /// encoder → reparameterize → decoder → loss = ‖x-x̂‖² + KL + γ‖A‖₁ + ρ·h(A)
    /// then backprop through the encoder weights, mean/logVar heads, decoder
    /// weights, and the adjacency tensor A (registered via
    /// <see cref="GetExtraTrainableTensors"/>). Replaces the previous Train
    /// that computed a loss scalar and discarded it without ever updating
    /// any parameter.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        EnsureSizedForInput(input);
        SetTrainingMode(true);
        try
        {
            using var tape = new GradientTape<T>();
            var (mean, logVar) = EncoderForwardTape(input);
            var z = ReparameterizeTape(mean, logVar);
            var rawOutput = DecoderForwardTape(z);
            var loss = ComputeGoggleLossTape(rawOutput, expectedOutput, mean, logVar);
            BackwardAndStepOnPrecomputedLoss(tape, loss, _optimizer);
            ProjectAdjacencyConstraints();
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Re-projects <see cref="_adjacency"/> onto the valid GOGGLE soft-adjacency
    /// set after each optimizer step: off-diagonal entries are clamped to [0, 1]
    /// and the diagonal is forced to zero. The optimizer updates the raw tensor
    /// (registered via <see cref="GetExtraTrainableTensors"/>) without these
    /// constraints, so unprojected steps could introduce negative edges or
    /// self-loops that the encoder/regularizers assume never occur.
    /// </summary>
    private void ProjectAdjacencyConstraints()
    {
        if (_adjacency is null) return;

        for (int i = 0; i < _dataWidth; i++)
        {
            for (int j = 0; j < _dataWidth; j++)
            {
                _adjacency[i, j] = i == j
                    ? NumOps.Zero
                    : NumOps.FromDouble(Math.Min(Math.Max(NumOps.ToDouble(_adjacency[i, j]), 0.0), 1.0));
            }
        }
    }

    /// <summary>
    /// Resizes encoder/decoder/adjacency to the actual input width when the
    /// caller has not yet invoked Fit. ModelFamily-generated tests call
    /// Train()/Predict() directly with random tensors of arbitrary width.
    /// </summary>
    private void EnsureSizedForInput(Tensor<T> input)
    {
        int width = input.Length;
        if (input.Rank == 2) width = input.Shape[input.Shape.Length - 1];
        if (!IsFitted && (_dataWidth != width || _adjacency is null || _meanHead is null))
        {
            _dataWidth = width;
            // Build a minimal column layout (treat the input as `width`
            // continuous columns) so RebuildAuxiliaryLayers can size every
            // sub-network correctly.
            _columns = new List<ColumnMetadata>();
            for (int c = 0; c < width; c++)
            {
                _columns.Add(new ColumnMetadata($"col_{c}", ColumnDataType.Continuous, columnIndex: c));
            }
            RebuildAuxiliaryLayers();
        }
    }

    /// <summary>
    /// Total trainable parameter count = the layer parameters PLUS the learned
    /// adjacency tensor (a registered extra trainable tensor). Overridden so the
    /// flat-parameter contract (GetParameters / UpdateParameters / ParameterCount)
    /// stays symmetric with <see cref="GetExtraTrainableTensors"/>.
    /// </summary>
    public override long ParameterCount
    {
        get
        {
            long total = base.ParameterCount;
            if (_adjacency is not null && _adjacency.Length > 0)
                total += _adjacency.Length;
            return total;
        }
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var layerParams = base.GetParameters();
        if (_adjacency is null || _adjacency.Length == 0)
            return layerParams;

        // Append adjacency after the layer parameters; UpdateParameters consumes
        // it in the identical order so the flat-parameter round-trip is lossless.
        var combined = new Vector<T>(layerParams.Length + _adjacency.Length);
        for (int i = 0; i < layerParams.Length; i++)
            combined[i] = layerParams[i];
        for (int i = 0; i < _adjacency.Length; i++)
            combined[layerParams.Length + i] = _adjacency[i];
        return combined;
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = checked((int)layer.ParameterCount);
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }

        // Consume the adjacency block appended by GetParameters() so any
        // parameter-vector-based flow (clone-by-parameters, external optimizers)
        // actually updates the learned graph instead of silently dropping it.
        if (_adjacency is not null && _adjacency.Length > 0
            && startIndex + _adjacency.Length <= parameters.Length)
        {
            for (int i = 0; i < _adjacency.Length; i++)
                _adjacency[i] = parameters[startIndex + i];
        }
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.LatentDimension);
        writer.Write(_options.NumGNNLayers);
        writer.Write(_options.HiddenDimension);
        writer.Write(_dataWidth);
        writer.Write(IsFitted);

        // Persist the learned adjacency matrix — it's a registered trainable
        // tensor (see GetExtraTrainableTensors) so the optimizer updates it
        // during Train, but it lives outside Layers and would be silently
        // dropped by Clone/SaveLoad without explicit (de)serialization.
        bool hasAdj = _adjacency is not null && _adjacency.Length > 0;
        writer.Write(hasAdj);
        if (hasAdj)
        {
            writer.Write(_adjacency!.Shape[0]);
            writer.Write(_adjacency.Shape[1]);
            for (int i = 0; i < _adjacency.Length; i++)
                writer.Write(Convert.ToDouble(_adjacency[i]));
        }
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // LatentDimension
        _ = reader.ReadInt32(); // NumGNNLayers
        _ = reader.ReadInt32(); // HiddenDimension
        _dataWidth = reader.ReadInt32();
        IsFitted = reader.ReadBoolean();

        bool hasAdj = reader.ReadBoolean();
        if (hasAdj)
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            _adjacency = new Tensor<T>(new[] { rows, cols });
            for (int i = 0; i < _adjacency.Length; i++)
                _adjacency[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Reconnect the typed field references (_gnnLayers, _meanHead,
        // _logvarHead, _decoderOutput) to the layers the base class
        // deserialized into the Layers collection. RebuildAuxiliaryLayers
        // wrote them in this stable order: [GNN×N, mean, logvar, decoder
        // MLP×K, decoderOutput]; we read them back the same way so the
        // forward pass (which uses the field references, not Layers) sees
        // the deserialized weights instead of the null / lazy state the
        // freshly-constructed clone instance has.
        int numGnn = _options.NumGNNLayers;
        if (Layers.Count >= numGnn + 3)
        {
            _gnnLayers.Clear();
            for (int i = 0; i < numGnn; i++)
                if (Layers[i] is FullyConnectedLayer<T> fc) _gnnLayers.Add(fc);
            if (Layers[numGnn] is FullyConnectedLayer<T> mean) _meanHead = mean;
            if (Layers[numGnn + 1] is FullyConnectedLayer<T> logVar) _logvarHead = logVar;
            if (Layers[Layers.Count - 1] is FullyConnectedLayer<T> decOut) _decoderOutput = decOut;
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new GOGGLEGenerator<T>(Architecture, _options);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        for (int i = 0; i < _columns.Count; i++)
        {
            importance[_columns[i].Name] = NumOps.FromDouble(1.0 / Math.Max(_columns.Count, 1));
        }
        return importance;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["GeneratorType"] = "GOGGLE",
                ["LatentDimension"] = _options.LatentDimension,
                ["NumGNNLayers"] = _options.NumGNNLayers,
                ["HiddenDimension"] = _options.HiddenDimension,
                ["DataWidth"] = _dataWidth,
                ["IsFitted"] = IsFitted
            }
        };
    }

    #endregion

    #region Helpers

    private Vector<T> CreateStandardNormalVector(int length)
    {
        var v = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            v[i] = NumOps.FromDouble(normal);
        }
        return v;
    }

    private static Tensor<T> SanitizeAndClipGradient(Tensor<T> grad, double maxNorm)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double val = numOps.ToDouble(grad[i]);
            if (double.IsNaN(val) || double.IsInfinity(val))
            {
                grad[i] = numOps.Zero;
                continue;
            }
            normSq += val * val;
        }

        double norm = Math.Sqrt(normSq);
        if (norm > maxNorm)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < grad.Length; i++)
            {
                grad[i] = numOps.FromDouble(numOps.ToDouble(grad[i]) * scale);
            }
        }

        return grad;
    }

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++) v[j] = matrix[row, j];
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

    #endregion

}
