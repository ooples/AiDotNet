using AiDotNet.LearningRateSchedulers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Graph;

/// <summary>
/// RelationalGCN (Relational Graph Convolutional Network) for multi-relational graph learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// RelationalGCN extends Graph Convolutional Networks to handle multi-relational data
/// where different types of edges (relations) exist between nodes, making it ideal for
/// knowledge graphs and heterogeneous networks.
/// </para>
/// <para><b>For Beginners:</b> RelationalGCN is designed for knowledge graphs:
///
/// <b>The Key Insight:</b>
/// Standard GCN treats all edges equally, but in knowledge graphs and financial networks,
/// different types of relationships matter differently. A "supplies-to" relationship is
/// fundamentally different from a "competes-with" relationship. R-GCN learns separate
/// transformations for each relation type.
///
/// <b>What Problems Does RelationalGCN Solve?</b>
/// - Entity classification in knowledge graphs (company type, sector classification)
/// - Link prediction in multi-relational networks (predicting missing relationships)
/// - Financial network analysis with multiple relationship types
/// - Supply chain modeling with different connection types
///
/// <b>How RelationalGCN Works:</b>
/// 1. <b>Relation-Specific Weights:</b> Learns different weights W_r for each relation type r
/// 2. <b>Basis Decomposition:</b> W_r = sum_b (a_rb * B_b) shares parameters via learned bases
/// 3. <b>Block Decomposition:</b> Alternative using block-diagonal weight matrices
/// 4. <b>Self-Connections:</b> Special weight W_0 for a node's own features
///
/// <b>RelationalGCN Architecture:</b>
/// - Message Passing: h_i^(l+1) = sigma(sum_r sum_j A_r[i,j] * h_j^(l) * W_r + h_i^(l) * W_0)
/// - Basis Decomposition: W_r = sum_b (a_rb * B_b) with shared bases B
/// - Block Decomposition: W_r = diag(W_r1, ..., W_rB) with block-diagonal structure
///
/// <b>Key Benefits:</b>
/// - Handles heterogeneous graphs with multiple edge types
/// - Parameter efficient through basis or block decomposition
/// - Captures relation-specific patterns in the data
/// - Effective for both entity classification and link prediction
/// </para>
/// <para>
/// <b>Reference:</b> Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks", ESWC 2018.
/// https://arxiv.org/abs/1703.06103
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for multi-relational knowledge graph node classification (50 entities, 5 classes)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 50, inputWidth: 16, inputDepth: 1, outputSize: 5);
///
/// // Training mode: learns relation-specific transformations with basis decomposition
/// var model = new RelationalGCN&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained Relational GCN model
/// var onnxModel = new RelationalGCN&lt;double&gt;(architecture, "rgcn_knowledge_graph.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.GraphAnalysis)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Modeling Relational Data with Graph Convolutional Networks", "https://arxiv.org/abs/1703.06103", Year = 2018, Authors = "Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.01,
                Source = "Schlichtkrull et al. 2018, Sec. 4: trained with Adam for 50 epochs using a "
                        + "learning rate of 0.01.")]
public partial class RelationalGCN<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    
    #region Relational Graph Fields
    /// <summary>
    /// The adjacency of every relation type, shaped [numRelations, numNodes, numNodes]. Entry [r, i, j] is the
    /// weight of the edge of type r along which node i receives a message from node j. A buffer: saved with the
    /// model (the default graph is drawn at random) but never trained.
    /// </summary>
    [Buffer]
    private Tensor<T>? _relationAdjacencies;

    /// <summary>
    /// Per R-GCN layer: the shared bases V_b [numBases, hidden, hidden] under basis decomposition, otherwise the
    /// relation weights W_r [numRelations, hidden, hidden] (masked to blocks under block decomposition).
    /// </summary>
    private readonly List<Tensor<T>> _relationBases = new List<Tensor<T>>();

    /// <summary>Per R-GCN layer under basis decomposition: the coefficients a_rb, [numRelations, numBases].</summary>
    private readonly List<Tensor<T>> _relationCoefficients = new List<Tensor<T>>();

    /// <summary>Per R-GCN layer when self-loops are on: the self-connection weight W_0, [hidden, hidden].</summary>
    private readonly List<Tensor<T>> _selfLoopWeights = new List<Tensor<T>>();

    /// <summary>Per R-GCN layer: the dropout applied to its output.</summary>
    private readonly List<DropoutLayer<T>> _relationDropouts = new List<DropoutLayer<T>>();

    /// <summary>The block-diagonal mask [hidden, hidden] of block decomposition, or null.</summary>
    private Tensor<T>? _blockMask;

    /// <summary>The L2 penalty of the most recent training forward, handed to the objective once.</summary>
    private Tensor<T>? _pendingRegularization;

    private readonly bool _meanAggregation;

    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly RelationalGCNOptions<T> _options;

    private int _sequenceLength;
    private int _forecastHorizon;
    private int _numNodes;
    private int _numFeatures;
    private int _numRelations;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numBases;
    private int _numBlocks;
    private double _regularization;
    private double _dropoutRate;
    private bool _useBasisDecomposition;
    private bool _useBlockDecomposition;
    private bool _useSelfLoop;
    private string _aggregation;
    private int _numSamples;
    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => 1;

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => false;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the number of nodes (entities) in the knowledge graph.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many entities exist in the graph.
    /// For example, number of companies, securities, or geographic regions.
    /// </para>
    /// </remarks>
    public int NumNodes => _numNodes;

    /// <summary>
    /// Gets the number of relation types.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different types of relationships exist.
    /// Examples: "supplies-to", "competes-with", "same-sector", "co-owned-by".
    /// </para>
    /// </remarks>
    public int NumRelations => _numRelations;

    /// <summary>
    /// Gets the forecast horizon.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future time steps to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon => _forecastHorizon;

    /// <summary>
    /// Gets whether the model supports training (native mode only).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX mode is inference-only.
    /// Native mode supports both training and inference.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the number of basis matrices for basis decomposition.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For basis decomposition, relation weights are
    /// linear combinations of shared basis matrices. Fewer bases = more parameter sharing.
    /// If NumBases == NumRelations, there's no sharing (full weights per relation).
    /// </para>
    /// </remarks>
    public int NumBases => _numBases;

    /// <summary>
    /// Gets the number of samples for uncertainty estimation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For probabilistic forecasting with MC Dropout.
    /// </para>
    /// </remarks>
    public int NumSamples => _numSamples;

    /// <summary>
    /// Gets whether basis decomposition is used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Basis decomposition reduces parameters by sharing
    /// basis matrices across relations. Recommended when you have many relation types.
    /// </para>
    /// </remarks>
    public bool UseBasisDecomposition => _useBasisDecomposition;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the RelationalGCN model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">RelationalGCN-specific options.</param>
    /// <param name="relationAdjacencies">The adjacency of every relation type, shaped [numRelations, numNodes, numNodes];
    /// entry [r, i, j] weights the edge of type r along which node i receives from node j.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained RelationalGCN model.
    /// The ONNX model includes learned basis matrices and relation coefficients.
    /// You must provide the relation adjacency matrices for the knowledge graph.
    /// </para>
    /// </remarks>
    public RelationalGCN(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        RelationalGCNOptions<T>? options = null,
        Tensor<T>? relationAdjacencies = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);
        _options = options ?? new RelationalGCNOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _numRelations = _options.NumRelations;
        _hiddenDimension = _options.HiddenDimension;
        _numLayers = _options.NumLayers;
        _numBases = _options.NumBases;
        _numBlocks = _options.NumBlocks;
        _regularization = _options.Regularization;
        _dropoutRate = _options.DropoutRate;
        _useBasisDecomposition = _options.UseBasisDecomposition;
        _useBlockDecomposition = _options.UseBlockDecomposition;
        _useSelfLoop = _options.UseSelfLoop;
        _aggregation = _options.Aggregation;
        _numSamples = _options.NumSamples;

        _meanAggregation = UsesMeanAggregation(_aggregation);
        _relationAdjacencies = relationAdjacencies is null ? null : ValidatedAdjacency(relationAdjacencies);
        _random = LayerInitializationSeedScope.CreateComponentRandom(_options.Seed);
    }

    /// <summary>
    /// Initializes a new instance of the RelationalGCN model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">RelationalGCN-specific options.</param>
    /// <param name="relationAdjacencies">The adjacency of every relation type, shaped [numRelations, numNodes, numNodes];
    /// entry [r, i, j] weights the edge of type r along which node i receives from node j.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create a RelationalGCN model for training.
    /// Provide the relation adjacency matrices that define the knowledge graph structure.
    /// Each adjacency matrix represents one type of relationship between nodes.
    /// </para>
    /// </remarks>
    public RelationalGCN(
        NeuralNetworkArchitecture<T> architecture,
        RelationalGCNOptions<T>? options = null,
        Tensor<T>? relationAdjacencies = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new RelationalGCNOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numNodes = _options.NumNodes;
        _numFeatures = _options.NumFeatures;
        _numRelations = _options.NumRelations;
        _hiddenDimension = _options.HiddenDimension;
        _numLayers = _options.NumLayers;
        _numBases = _options.NumBases;
        _numBlocks = _options.NumBlocks;
        _regularization = _options.Regularization;
        _dropoutRate = _options.DropoutRate;
        _useBasisDecomposition = _options.UseBasisDecomposition;
        _useBlockDecomposition = _options.UseBlockDecomposition;
        _useSelfLoop = _options.UseSelfLoop;
        _aggregation = _options.Aggregation;
        _numSamples = _options.NumSamples;

        ValidateRelationalOptions();
        _meanAggregation = UsesMeanAggregation(_aggregation);

        // Initialize random before CreateDefaultRelationAdjacencies which uses it
        _random = LayerInitializationSeedScope.CreateComponentRandom(_options.Seed);
        _relationAdjacencies = relationAdjacencies is null
            ? CreateDefaultRelationAdjacencies()
            : ValidatedAdjacency(relationAdjacencies);

        InitializeRelationalWeights();
        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the per-node layers around the R-GCN propagation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Every layer but the last is a per-node encoder that turns a node's history
    /// window into its hidden state h_i^(0); the last layer is the per-node head that turns the final hidden
    /// state into the forecast. The relational graph convolutions run between the two (see
    /// <see cref="ApplyRelationalLayer"/>). By default the encoder is one dense layer to the hidden dimension
    /// and the head one dense layer to the forecast horizon.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayersWithOwnershipRollback(Layers);
        }
        else if (_useNativeMode)
        {
            Layers.Add(new DenseLayer<T>(outputSize: _hiddenDimension, activationFunction: new ReLUActivation<T>()));
            Layers.Add(new DenseLayer<T>(outputSize: _forecastHorizon, activationFunction: null));
        }
    }

    /// <summary>
    /// Rejects option combinations the R-GCN layers cannot be built from.
    /// </summary>
    private void ValidateRelationalOptions()
    {
        if (_numNodes < 1)
            throw new ArgumentException("RelationalGCN needs at least one node.", "options");
        if (_numRelations < 1)
            throw new ArgumentException("RelationalGCN needs at least one relation type.", "options");
        if (_hiddenDimension < 1)
            throw new ArgumentException("RelationalGCN needs a positive hidden dimension.", "options");
        if (_numLayers < 1)
            throw new ArgumentException("RelationalGCN needs at least one R-GCN layer.", "options");
        if (_useBlockDecomposition && (_numBlocks < 1 || _hiddenDimension % _numBlocks != 0))
            throw new ArgumentException(
                $"Block decomposition splits the hidden dimension ({_hiddenDimension}) into NumBlocks ({_numBlocks}) equal blocks, so NumBlocks must divide it.",
                "options");
        if (!_useBlockDecomposition && _useBasisDecomposition && _numBases < 1)
            throw new ArgumentException("Basis decomposition needs at least one basis.", "options");
    }

    /// <summary>
    /// Reads the aggregation option: "sum" keeps c_{i,r} = 1, "mean" sets c_{i,r} to the node's in-degree under r.
    /// </summary>
    /// <param name="aggregation">The configured aggregation.</param>
    /// <returns>True for mean aggregation, false for sum.</returns>
    private static bool UsesMeanAggregation(string aggregation)
    {
        if (string.Equals(aggregation, "mean", StringComparison.OrdinalIgnoreCase))
            return true;
        if (string.Equals(aggregation, "sum", StringComparison.OrdinalIgnoreCase))
            return false;
        throw new ArgumentException(
            $"RelationalGCN aggregates each relation's messages as a normalized sum (Schlichtkrull et al. 2018, Eq. 2), so Aggregation must be \"sum\" or \"mean\"; got \"{aggregation}\".",
            "options");
    }

    /// <summary>
    /// Initializes the trainable R-GCN weights of every layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Under basis decomposition (Schlichtkrull et al. 2018, Eq. 3) each layer holds
    /// shared bases V_b and per-relation coefficients a_rb, and the relation weight is W_r = sum_b a_rb V_b.
    /// Under block decomposition (Eq. 4) each W_r is block-diagonal with NumBlocks blocks, and block decomposition
    /// takes precedence when both are switched on. With neither, each relation has its own full W_r. Self-loops
    /// add the self-connection weight W_0. All of them are trained by backpropagation through
    /// <see cref="ApplyRelationalLayer"/>.
    /// </para>
    /// </remarks>
    private void InitializeRelationalWeights()
    {
        bool useBlocks = _useBlockDecomposition;
        bool useBases = !useBlocks && _useBasisDecomposition;
        double scale = Math.Sqrt(2.0 / _hiddenDimension);
        int weightCount = useBases ? _numBases : _numRelations;
        if (useBlocks)
            _blockMask = BlockDiagonalMask(_hiddenDimension, _numBlocks);

        for (int layer = 0; layer < _numLayers; layer++)
        {
            _relationBases.Add(UniformTensor(new[] { weightCount, _hiddenDimension, _hiddenDimension }, scale));
            if (useBases)
                _relationCoefficients.Add(UniformTensor(new[] { _numRelations, _numBases }, 0.05));
            if (_useSelfLoop)
                _selfLoopWeights.Add(UniformTensor(new[] { _hiddenDimension, _hiddenDimension }, scale));
            _relationDropouts.Add(new DropoutLayer<T>(_dropoutRate));
        }
    }

    /// <summary>
    /// Draws a tensor of independent uniform values in [-limit, limit] from the model's random stream.
    /// </summary>
    private Tensor<T> UniformTensor(int[] shape, double limit)
    {
        int count = shape.Aggregate(1, (product, dimension) => product * dimension);
        var values = Enumerable.Range(0, count)
            .Select(_ => NumOps.FromDouble((_random.NextDouble() - 0.5) * 2.0 * limit))
            .ToArray();
        return new Tensor<T>(shape, new Vector<T>(values));
    }

    /// <summary>
    /// The [size, size] mask that keeps the <paramref name="blocks"/> equal diagonal blocks of a weight matrix.
    /// </summary>
    private static Tensor<T> BlockDiagonalMask(int size, int blocks)
    {
        int blockSize = size / blocks;
        var values = Enumerable.Range(0, size * size)
            .Select(index => index / size / blockSize == index % size / blockSize ? NumOps.One : NumOps.Zero)
            .ToArray();
        return new Tensor<T>(new[] { size, size }, new Vector<T>(values));
    }

    /// <summary>
    /// Creates default relation adjacencies when none provided.
    /// </summary>
    /// <returns>A [numRelations, numNodes, numNodes] tensor of row-normalized random adjacencies.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> If no relation adjacencies are provided, we create
    /// sparse random adjacencies (10% density, each row scaled to sum to 1) as defaults. In practice, you should
    /// always provide actual relation adjacencies from your knowledge graph.
    /// </para>
    /// </remarks>
    private Tensor<T> CreateDefaultRelationAdjacencies()
    {
        var edges = Enumerable.Range(0, _numRelations * _numNodes * _numNodes)
            .Select(_ => _random.NextDouble() < 0.1 ? NumOps.One : NumOps.Zero)
            .ToArray();
        var adjacency = new Tensor<T>(new[] { _numRelations, _numNodes, _numNodes }, new Vector<T>(edges));
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return DivideByRowSums(adjacency);
    }

    /// <summary>
    /// Divides every row of a [numRelations, numNodes, numNodes] adjacency by its sum; empty rows stay zero.
    /// </summary>
    private Tensor<T> DivideByRowSums(Tensor<T> adjacency)
    {
        var rowSums = Engine.ReduceSum(adjacency, new[] { 2 }, true);
        var divisors = Engine.TensorWhere(
            Engine.TensorEquals(rowSums, NumOps.Zero), Engine.TensorFullLike(rowSums, NumOps.One), rowSums);
        return Engine.TensorDivide(adjacency, Engine.TensorBroadcastTo(divisors, adjacency._shape));
    }

    /// <summary>
    /// Checks the relation adjacencies' shape and takes a private copy, so restoring the model's saved state never
    /// writes into the caller's tensor.
    /// </summary>
    /// <param name="adjacency">The [numRelations, numNodes, numNodes] adjacencies.</param>
    /// <returns>The model's own copy.</returns>
    private Tensor<T> ValidatedAdjacency(Tensor<T> adjacency)
    {
        if (adjacency.Rank != 3 || adjacency._shape[0] != _numRelations
            || adjacency._shape[1] != _numNodes || adjacency._shape[2] != _numNodes)
        {
            throw new ArgumentException(
                $"Relation adjacencies must be shaped [{_numRelations}, {_numNodes}, {_numNodes}] (relations, nodes, nodes); got [{string.Join(", ", adjacency._shape)}].",
                "relationAdjacencies");
        }

        return adjacency.Clone();
    }

    /// <summary>
    /// The relation adjacencies with the 1/c_{i,r} normalization applied, stacked for propagation.
    /// </summary>
    /// <returns>The normalized adjacencies as [numRelations * numNodes, numNodes].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Row (r, i) of the result holds A_r[i, j] / c_{i,r}, so multiplying it by the node
    /// states gives sum_{j in N_i^r} h_j / c_{i,r}. Under "sum" aggregation c_{i,r} = 1 and the adjacency weights
    /// are used as given; under "mean" c_{i,r} is the node's weighted in-degree under relation r, which for a 0/1
    /// adjacency is the |N_i^r| the paper uses.
    /// </para>
    /// </remarks>
    private Tensor<T> Propagation()
    {
        var adjacency = _relationAdjacencies
            ?? throw new InvalidOperationException("RelationalGCN has no relation adjacencies to propagate over.");
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var normalized = _meanAggregation ? DivideByRowSums(adjacency) : adjacency.Contiguous();
        return Engine.Reshape(normalized, new[] { _numRelations * _numNodes, _numNodes });
    }

    /// <summary>
    /// Validates custom layers provided by the user.
    /// </summary>
    /// <param name="layers">The list of custom layers.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Custom layers must hold at least a per-node encoder and a per-node head: every
    /// layer but the last encodes a node's history into its hidden state, and the last maps the final hidden
    /// state to the forecast.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 2)
            throw new ArgumentException("RelationalGCN requires at least 2 layers (a per-node encoder and a per-node head).");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Performs forward prediction on the input tensor.
    /// </summary>
    /// <param name="input">Input tensor with shape [nodes, sequence, features].</param>
    /// <returns>Output tensor with forecasts for all nodes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Processes input through:
    /// 1. Embed node features to hidden dimension
    /// 2. For each R-GCN layer: aggregate neighbor messages per relation, apply relation-specific transforms
    /// 3. Add self-loop transformation if enabled
    /// 4. Project to forecast dimension
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// The R-GCN weights are trainable: the bases V_b and coefficients a_rb (or the full or block relation
    /// weights) and the self-connection weights W_0 of every layer. The optimizer updates them from the gradients
    /// that reach them through <see cref="ApplyRelationalLayer"/> on the training tape.
    /// </summary>
    protected override IEnumerable<Tensor<T>> GetExtraTrainableTensors()
    {
        for (int layer = 0; layer < _relationBases.Count; layer++)
        {
            yield return _relationBases[layer];
            if (layer < _relationCoefficients.Count)
                yield return _relationCoefficients[layer];
            if (layer < _selfLoopWeights.Count)
                yield return _selfLoopWeights[layer];
        }
    }

    /// <summary>
    /// The trainable R-GCN weight tensors, in the order <see cref="GetExtraTrainableTensors"/> yields them.
    /// </summary>
    internal IReadOnlyList<Tensor<T>> RelationalWeightTensors => GetExtraTrainableTensors().ToList();

    /// <inheritdoc/>
    /// <remarks>The R-GCN dropout layers sit outside <c>Layers</c>, so they are switched here too.</remarks>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var dropout in _relationDropouts)
            dropout.SetTrainingMode(isTraining);
    }

    /// <summary>
    /// Training-mode forward: the same computation as <see cref="Forward"/>, kept in training mode so dropout is
    /// active, which also records the L2 penalty on the R-GCN weights for the objective.
    /// </summary>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        var output = Forward(input);
        _pendingRegularization = _useNativeMode ? RegularizationPenalty() : null;
        return output;
    }

    /// <inheritdoc/>
    protected override Tensor<T>? ConsumeAuxiliaryTapeLoss()
    {
        var penalty = _pendingRegularization;
        _pendingRegularization = null;
        return penalty;
    }

    /// <summary>
    /// Regularization * (sum of the squared R-GCN weights), as a tape-connected scalar, or null when the
    /// regularization strength is zero.
    /// </summary>
    private Tensor<T>? RegularizationPenalty()
    {
        if (_regularization <= 0)
            return null;

        Tensor<T>? total = null;
        foreach (var weight in GetExtraTrainableTensors())
        {
            var squares = Engine.Reshape(
                Engine.ReduceSum(Engine.TensorSquare(weight), Enumerable.Range(0, weight.Rank).ToArray(), false),
                new[] { 1 });
            total = total is null ? squares : Engine.TensorAdd(total, squares);
        }

        return total is null ? null : Engine.TensorMultiplyScalar(total, NumOps.FromDouble(_regularization));
    }

    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
    /// <summary>
    /// Gets metadata about the RelationalGCN model.
    /// </summary>
    /// <returns>ModelMetadata containing model information.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns a dictionary of model configuration values
    /// for logging, debugging, and serialization purposes.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "RelationalGCN" },
                { "SequenceLength", _sequenceLength },
                { "ForecastHorizon", _forecastHorizon },
                { "NumNodes", _numNodes },
                { "NumFeatures", _numFeatures },
                { "NumRelations", _numRelations },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumBases", _numBases },
                { "NumBlocks", _numBlocks },
                { "UseBasisDecomposition", _useBasisDecomposition },
                { "UseBlockDecomposition", _useBlockDecomposition },
                { "UseSelfLoop", _useSelfLoop },
                { "Aggregation", _aggregation },
                { "Regularization", _regularization },
                { "DropoutRate", _dropoutRate },
                { "NumSamples", _numSamples },
                { "UseNativeMode", _useNativeMode }
            }
        };
    }

    /// <summary>
    /// Serializes RelationalGCN-specific data.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves model configuration and learned parameters
    /// (basis matrices, relation coefficients) to disk for later loading.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Deserializes RelationalGCN-specific data.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Loads model configuration and learned parameters
    /// from disk, restoring the model to its saved state.
    /// </para>
    /// </remarks>


    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates forecasts for all nodes.
    /// </summary>
    /// <param name="historicalData">Input tensor with node features.</param>
    /// <param name="quantiles">Optional quantile levels for probabilistic forecasting.</param>
    /// <returns>Forecast tensor with predictions for all nodes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uses the learned relation-specific transformations
    /// to aggregate information across the knowledge graph, then projects to forecasts.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastCore(Tensor<T> historicalData, double[]? quantiles)
    {
        if (quantiles is not null && quantiles.Length > 0)
        {
            var samples = GenerateSamples(historicalData, _numSamples);
            return ComputeQuantiles(samples, quantiles);
        }

        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <summary>
    /// Generates forecasts with prediction intervals.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="confidenceLevel">Confidence level (e.g., 0.95).</param>
    /// <returns>Tuple of (forecast, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uses MC Dropout to generate multiple forecasts,
    /// then computes prediction intervals from the sample distribution.
    /// </para>
    /// </remarks>
    public (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ForecastWithIntervals(
        Tensor<T> input,
        double confidenceLevel = 0.95)
    {
        var samples = GenerateSamples(input, _numSamples);
        return ComputePredictionIntervals(samples, confidenceLevel);
    }

    /// <summary>
    /// Performs autoregressive forecasting for extended horizons.
    /// </summary>
    /// <param name="input">Initial input.</param>
    /// <param name="steps">Number of forecast steps.</param>
    /// <returns>Extended forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For forecasting beyond the model's horizon,
    /// feeds predictions back as input to generate longer forecasts.
    /// </para>
    /// </remarks>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        var predictions = new List<Tensor<T>>();
        var currentInput = input;

        for (int i = 0; i < steps; i++)
        {
            var prediction = Forecast(currentInput, null);
            predictions.Add(prediction);
            currentInput = ShiftInputWindow(currentInput, prediction);
        }

        return ConcatenatePredictions(predictions);
    }

    /// <summary>
    /// Evaluates forecast quality against actual values.
    /// </summary>
    /// <param name="predictions">Predicted values for all nodes.</param>
    /// <param name="actuals">Actual values for all nodes.</param>
    /// <returns>Dictionary of evaluation metrics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Computes standard forecasting metrics (MSE, MAE, RMSE)
    /// to assess how well the model's predictions match reality.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();

        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = Math.Min(predictions.Data.Length, actuals.Data.Length);

        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], actuals.Data.Span[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
        }

        if (count > 0)
        {
            mse = NumOps.Divide(mse, NumOps.FromDouble(count));
            mae = NumOps.Divide(mae, NumOps.FromDouble(count));
        }

        T rmse = NumOps.Sqrt(mse);

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = rmse;

        return metrics;
    }

    /// <summary>
    /// Applies instance normalization (identity for RelationalGCN).
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>The input tensor unchanged.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instance normalization is not typically used
    /// in R-GCN. This method returns the input unchanged.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <summary>
    /// Gets financial-specific metrics.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns metrics relevant to financial forecasting,
    /// including model configuration values.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["LastLoss"] = lastLoss,
            ["NumNodes"] = NumOps.FromDouble(_numNodes),
            ["NumRelations"] = NumOps.FromDouble(_numRelations),
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["NumBases"] = NumOps.FromDouble(_numBases)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass: per-node encoder, R-GCN layers, per-node head.
    /// </summary>
    /// <param name="input">Input tensor holding numNodes x sequenceLength x numFeatures values, node-major
    /// (for example [numNodes, sequenceLength, numFeatures]).</param>
    /// <returns>The forecast, shaped [numNodes, forecastHorizon].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass:
    /// 1. Encodes each node's history window into its hidden state h_i^(0)
    /// 2. Applies each R-GCN layer, h_i' = ReLU(W_0 h_i + sum_r sum_{j in N_i^r} (1/c_{i,r}) W_r h_j), then dropout
    /// 3. Projects each node's final hidden state to its forecast
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        if (!_useNativeMode)
            return ForecastOnnx(input);

        var current = NodeFeatures(input);
        int headIndex = Layers.Count - 1;
        for (int i = 0; i < headIndex; i++)
            current = Layers[i].Forward(current);

        if (current.Rank != 2 || current._shape[0] != _numNodes || current._shape[1] != _hiddenDimension)
        {
            throw new InvalidOperationException(
                $"The RelationalGCN encoder layers must map the [{_numNodes}, {_sequenceLength * _numFeatures}] node histories to [{_numNodes}, {_hiddenDimension}] hidden states; they produced [{string.Join(", ", current._shape)}].");
        }

        for (int layer = 0; layer < _relationBases.Count; layer++)
            current = _relationDropouts[layer].Forward(ApplyRelationalLayer(current, layer));

        var output = Layers[headIndex].Forward(current);
        if (output.Length != _numNodes * _forecastHorizon)
        {
            throw new InvalidOperationException(
                $"The RelationalGCN head layer must produce {_forecastHorizon} values per node ({_numNodes * _forecastHorizon} in all); it produced [{string.Join(", ", output._shape)}].");
        }

        return Engine.Reshape(output, new[] { _numNodes, _forecastHorizon });
    }

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Runs the full R-GCN forward pass using
    /// the learned parameters to generate predictions for all nodes.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> input)
    {
        SetTrainingMode(false);
        return Forward(input);
    }

    /// <summary>
    /// Performs ONNX mode forecasting.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uses the ONNX runtime to run inference
    /// with a pretrained model, enabling deployment without the training framework.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var flatInput = FlattenInput(input);
        var inputData = new float[flatInput.Data.Length];
        for (int i = 0; i < flatInput.Data.Length; i++)
        {
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(flatInput.Data.Span[i]));
        }

        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData,
            new[] { 1, _numNodes, _sequenceLength, _numFeatures });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(new[] { _numNodes * _forecastHorizon }, new Vector<T>(outputData));
    }

    #endregion

    #region Relational Graph Convolution

    /// <summary>
    /// Applies one R-GCN layer (Schlichtkrull et al. 2018, Eq. 2).
    /// </summary>
    /// <param name="nodeStates">The node states h^(l), shaped [numNodes, hidden].</param>
    /// <param name="layer">The index of the R-GCN layer.</param>
    /// <returns>h^(l+1) with h_i^(l+1) = ReLU(W_0 h_i + sum_r sum_{j in N_i^r} (1/c_{i,r}) W_r h_j).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> One matrix product gathers, for every relation r and node i, the normalized
    /// neighbour sum sum_j A_r[i, j] / c_{i,r} h_j. Laying those sums side by side per node and multiplying by the
    /// relation weights stacked vertically sums (A_r H) W_r over all relations at once. Everything is an engine
    /// operation, so the gradient reaches the relation weights, the bases and coefficients they are built from,
    /// W_0, and the encoder below.
    /// </para>
    /// </remarks>
    internal Tensor<T> ApplyRelationalLayer(Tensor<T> nodeStates, int layer)
    {
        var neighbourSums = Engine.TensorMatMul(Propagation(), nodeStates);
        var perNode = Engine.Reshape(
            Engine.TensorPermute(
                Engine.Reshape(neighbourSums, new[] { _numRelations, _numNodes, _hiddenDimension }),
                new[] { 1, 0, 2 }),
            new[] { _numNodes, _numRelations * _hiddenDimension });
        var messages = Engine.TensorMatMul(perNode, RelationWeights(layer));
        if (layer < _selfLoopWeights.Count)
            messages = Engine.TensorAdd(messages, Engine.TensorMatMul(nodeStates, _selfLoopWeights[layer]));

        return Engine.TensorReLU(messages);
    }

    /// <summary>
    /// The relation weights W_r of one layer, stacked vertically as [numRelations * hidden, hidden].
    /// </summary>
    /// <param name="layer">The index of the R-GCN layer.</param>
    /// <returns>The stacked relation weights.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Under basis decomposition W_r = sum_b a_rb V_b, computed for every relation as
    /// one product of the coefficients with the flattened bases. Under block decomposition W_r is its full matrix
    /// with everything off the diagonal blocks masked to zero.
    /// </para>
    /// </remarks>
    internal Tensor<T> RelationWeights(int layer)
    {
        var weights = _relationBases[layer];
        int squared = _hiddenDimension * _hiddenDimension;
        if (layer < _relationCoefficients.Count)
        {
            weights = Engine.TensorMatMul(
                _relationCoefficients[layer], Engine.Reshape(weights, new[] { weights._shape[0], squared }));
        }
        else if (_blockMask is not null)
        {
            weights = Engine.TensorMultiply(weights, Engine.TensorBroadcastTo(_blockMask, weights._shape));
        }

        return Engine.Reshape(weights, new[] { _numRelations * _hiddenDimension, _hiddenDimension });
    }

    /// <summary>
    /// Views the input as one history row per node, [numNodes, sequenceLength * numFeatures].
    /// </summary>
    /// <param name="input">The node-major input.</param>
    /// <returns>The per-node history rows.</returns>
    private Tensor<T> NodeFeatures(Tensor<T> input)
    {
        int width = _sequenceLength * _numFeatures;
        if (input.Length != _numNodes * width)
        {
            throw new ArgumentException(
                $"RelationalGCN expects {_numNodes} nodes x {_sequenceLength} steps x {_numFeatures} features ({_numNodes * width} values, node-major); got shape [{string.Join(", ", input._shape)}].",
                nameof(input));
        }

        return Engine.Reshape(input.Contiguous(), new[] { _numNodes, width });
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Flattens input tensor for layer processing.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Flattened tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Converts multi-dimensional input into a format
    /// suitable for dense layer processing.
    /// </para>
    /// </remarks>
    private Tensor<T> FlattenInput(Tensor<T> input)
    {
        int totalSize = 1;
        foreach (var dim in input._shape)
        {
            totalSize *= dim;
        }

        var flattened = new Tensor<T>(new[] { totalSize });
        for (int i = 0; i < totalSize; i++)
        {
            flattened.Data.Span[i] = input.Data.Span[i];
        }

        return flattened;
    }

    /// <summary>
    /// Generates MC Dropout samples for uncertainty estimation.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="numSamples">Number of samples.</param>
    /// <returns>List of sample forecasts.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> MC Dropout keeps dropout active during inference,
    /// generating multiple different predictions. The variation between predictions
    /// indicates model uncertainty.
    /// </para>
    /// </remarks>
    private List<Tensor<T>> GenerateSamples(Tensor<T> input, int numSamples)
    {
        var samples = new List<Tensor<T>>();

        for (int i = 0; i < numSamples; i++)
        {
            SetTrainingMode(true);
            var sample = Forward(input);
            samples.Add(sample);
        }

        SetTrainingMode(false);
        return samples;
    }

    /// <summary>
    /// Computes quantiles from samples.
    /// </summary>
    /// <param name="samples">List of sample forecasts.</param>
    /// <param name="quantiles">Quantile levels to compute.</param>
    /// <returns>Tensor with quantile forecasts.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sorts the samples at each position and extracts
    /// values at the requested quantile positions.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeQuantiles(List<Tensor<T>> samples, double[] quantiles)
    {
        if (samples.Count == 0)
            return new Tensor<T>(new[] { _numNodes, _forecastHorizon });

        int dataLength = samples[0].Data.Length;
        var result = new T[dataLength * quantiles.Length];

        for (int i = 0; i < dataLength; i++)
        {
            var values = samples.Select(s => NumOps.ToDouble(s.Data.Span[i])).OrderBy(v => v).ToArray();

            for (int q = 0; q < quantiles.Length; q++)
            {
                int idx = (int)(quantiles[q] * (values.Length - 1));
                idx = Math.Max(0, Math.Min(values.Length - 1, idx));
                result[q * dataLength + i] = NumOps.FromDouble(values[idx]);
            }
        }

        return new Tensor<T>(new[] { quantiles.Length, _numNodes, _forecastHorizon },
            new Vector<T>(result));
    }

    /// <summary>
    /// Computes prediction intervals from samples.
    /// </summary>
    /// <param name="samples">List of sample forecasts.</param>
    /// <param name="confidenceLevel">Confidence level (e.g., 0.95).</param>
    /// <returns>Tuple of (mean forecast, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Computes the mean prediction and bounds that
    /// contain the specified percentage of samples.
    /// </para>
    /// </remarks>
    private (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ComputePredictionIntervals(
        List<Tensor<T>> samples,
        double confidenceLevel)
    {
        double alpha = (1 - confidenceLevel) / 2;
        var quantiles = new[] { alpha, 0.5, 1 - alpha };
        var quantileTensor = ComputeQuantiles(samples, quantiles);

        int dataLength = samples[0].Data.Length;
        var lower = new T[dataLength];
        var median = new T[dataLength];
        var upper = new T[dataLength];

        for (int i = 0; i < dataLength; i++)
        {
            lower[i] = quantileTensor.Data.Span[i];
            median[i] = quantileTensor.Data.Span[dataLength + i];
            upper[i] = quantileTensor.Data.Span[2 * dataLength + i];
        }

        var shape = new[] { _numNodes, _forecastHorizon };
        return (
            new Tensor<T>(shape, new Vector<T>(median)),
            new Tensor<T>(shape, new Vector<T>(lower)),
            new Tensor<T>(shape, new Vector<T>(upper))
        );
    }

    /// <summary>
    /// Shifts the input window by appending a prediction.
    /// </summary>
    /// <param name="input">Current input.</param>
    /// <param name="prediction">Prediction to append, [numNodes, forecastHorizon].</param>
    /// <returns>Shifted input tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For autoregressive forecasting, drops each node's oldest forecastHorizon
    /// steps and appends its newest predictions. The input is node-major, so the shift runs along each node's
    /// time axis. Only a single-feature graph can be rolled forward this way, because a forecast supplies no
    /// future values for any other input feature.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> prediction)
    {
        if (_numFeatures != 1)
        {
            throw new NotSupportedException(
                $"Autoregressive RelationalGCN forecasting feeds forecasts back as inputs, which needs future values of every input feature; it is only defined for NumFeatures = 1 (got {_numFeatures}).");
        }

        var history = NodeFeatures(input);
        var predicted = Engine.Reshape(prediction.Contiguous(), new[] { _numNodes, _forecastHorizon });
        var extended = Engine.Concat(new[] { history, predicted }, 1);
        var window = Engine.TensorNarrow(extended, 1, _forecastHorizon, _sequenceLength);
        return Engine.Reshape(window.Contiguous(), input._shape);
    }

    /// <summary>
    /// Concatenates multiple predictions into one tensor.
    /// </summary>
    /// <param name="predictions">List of predictions.</param>
    /// <returns>Concatenated tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Combines multiple forecast steps into a single
    /// extended forecast tensor, joining each node's steps along its time axis.
    /// </para>
    /// </remarks>
    protected Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { _numNodes, 0 });

        var perNode = predictions
            .Select(prediction => Engine.Reshape(prediction.Contiguous(), new[] { _numNodes, _forecastHorizon }))
            .ToArray();
        return Engine.Concat(perNode, 1);
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes managed resources.
    /// </summary>
    /// <param name="disposing">Whether to dispose managed resources.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Cleans up resources when the model is no longer needed,
    /// particularly the ONNX session which holds native resources.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxSession?.Dispose();
        }

        base.Dispose(disposing);
    }

    #endregion
}



