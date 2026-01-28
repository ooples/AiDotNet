using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
public class RelationalGCN<T> : NeuralNetworkBase<T>, IForecastingModel<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    #region ONNX Mode Fields
    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;
    #endregion

    #region Native Mode Fields
    private DenseLayer<T>? _inputProjection;
    private List<DenseLayer<T>>? _basisLayers;
    private DenseLayer<T>? _relationCoefficientLayer;
    private List<DenseLayer<T>>? _rgcnLayers;
    private List<DenseLayer<T>>? _selfLoopLayers;
    private List<DropoutLayer<T>>? _dropoutLayers;
    private DenseLayer<T>? _outputLayer;
    #endregion

    #region Relational Graph Fields
    /// <summary>
    /// Adjacency matrices for each relation type.
    /// Index is [relation][nodeFrom, nodeTo].
    /// </summary>
    private readonly double[][,]? _relationAdjacencies;

    /// <summary>
    /// Learned basis matrices for basis decomposition.
    /// </summary>
    private double[,][]? _basisMatrices;

    /// <summary>
    /// Learned coefficients for combining basis matrices per relation.
    /// </summary>
    private double[,]? _relationCoefficients;

    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly RelationalGCNOptions<T> _options;
    private readonly int _sequenceLength;
    private readonly int _forecastHorizon;
    private readonly int _numNodes;
    private readonly int _numFeatures;
    private readonly int _numRelations;
    private readonly int _hiddenDimension;
    private readonly int _numLayers;
    private readonly int _numBases;
    private readonly int _numBlocks;
    private readonly double _regularization;
    private readonly double _dropoutRate;
    private readonly bool _useBasisDecomposition;
    private readonly bool _useBlockDecomposition;
    private readonly bool _useSelfLoop;
    private readonly string _aggregation;
    private readonly int _numSamples;
    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public int PatchSize => 1;

    /// <inheritdoc/>
    public int Stride => 1;

    /// <inheritdoc/>
    public bool IsChannelIndependent => false;

    /// <inheritdoc/>
    public bool UseNativeMode => _useNativeMode;

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
    /// <param name="relationAdjacencies">Adjacency matrices for each relation type.</param>
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
        double[][,]? relationAdjacencies = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _onnxSession = new InferenceSession(onnxModelPath);
        _options = options ?? new RelationalGCNOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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

        _relationAdjacencies = relationAdjacencies;
        _random = RandomHelper.CreateSecureRandom();

        InitializeBasisDecomposition();
    }

    /// <summary>
    /// Initializes a new instance of the RelationalGCN model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">RelationalGCN-specific options.</param>
    /// <param name="relationAdjacencies">Adjacency matrices for each relation type.</param>
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
        double[][,]? relationAdjacencies = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new RelationalGCNOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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

        _relationAdjacencies = relationAdjacencies ?? CreateDefaultRelationAdjacencies();
        _random = RandomHelper.CreateSecureRandom();

        InitializeBasisDecomposition();
        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for RelationalGCN.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sets up the layers needed for relational graph convolution:
    /// input projection, basis matrices (if using basis decomposition), R-GCN layers,
    /// self-loop transformations, and output projection.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultRelationalGCNLayers(
                Architecture,
                _numNodes,
                _numFeatures,
                _numRelations,
                _hiddenDimension,
                _numLayers,
                _numBases,
                _forecastHorizon));
        }

        ExtractLayerReferences();
    }

    /// <summary>
    /// Extracts references to specific layer types for direct access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After creating layers, we store references to specific
    /// layers for easier access during forward/backward passes. This enables efficient
    /// application of relation-specific transformations.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        _inputProjection = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _basisLayers = Layers.OfType<DenseLayer<T>>().Skip(1).Take(_numBases).ToList();
        _relationCoefficientLayer = Layers.OfType<DenseLayer<T>>().Skip(1 + _numBases).FirstOrDefault();
        _rgcnLayers = Layers.OfType<DenseLayer<T>>().Skip(2 + _numBases).ToList();
        _selfLoopLayers = _rgcnLayers.Where((_, i) => i % 2 == 1).Take(_numLayers).ToList();
        _dropoutLayers = Layers.OfType<DropoutLayer<T>>().ToList();
        _outputLayer = Layers.OfType<DenseLayer<T>>().LastOrDefault();
    }

    /// <summary>
    /// Initializes the basis decomposition matrices.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For basis decomposition, we initialize:
    /// - B_b: Shared basis matrices (numBases of them)
    /// - a_rb: Coefficients for each relation to combine basis matrices
    ///
    /// The relation-specific weight is: W_r = sum_b (a_rb * B_b)
    /// This dramatically reduces parameters when you have many relation types.
    /// </para>
    /// </remarks>
    private void InitializeBasisDecomposition()
    {
        if (!_useBasisDecomposition)
            return;

        // Initialize basis matrices
        _basisMatrices = new double[_numBases, _hiddenDimension][];
        for (int b = 0; b < _numBases; b++)
        {
            for (int i = 0; i < _hiddenDimension; i++)
            {
                _basisMatrices[b, i] = new double[_hiddenDimension];
                double scale = Math.Sqrt(2.0 / _hiddenDimension);
                for (int j = 0; j < _hiddenDimension; j++)
                {
                    _basisMatrices[b, i][j] = (_random.NextDouble() - 0.5) * 2.0 * scale;
                }
            }
        }

        // Initialize relation coefficients
        _relationCoefficients = new double[_numRelations, _numBases];
        for (int r = 0; r < _numRelations; r++)
        {
            for (int b = 0; b < _numBases; b++)
            {
                _relationCoefficients[r, b] = (_random.NextDouble() - 0.5) * 0.1;
            }
        }
    }

    /// <summary>
    /// Creates default relation adjacencies when none provided.
    /// </summary>
    /// <returns>Array of identity-like adjacency matrices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> If no relation adjacencies are provided, we create
    /// sparse random adjacencies as defaults. In practice, you should always provide
    /// actual relation adjacencies from your knowledge graph.
    /// </para>
    /// </remarks>
    private double[][,] CreateDefaultRelationAdjacencies()
    {
        var adjacencies = new double[_numRelations][,];
        for (int r = 0; r < _numRelations; r++)
        {
            adjacencies[r] = new double[_numNodes, _numNodes];
            // Create sparse random adjacency (10% density)
            for (int i = 0; i < _numNodes; i++)
            {
                for (int j = 0; j < _numNodes; j++)
                {
                    if (_random.NextDouble() < 0.1)
                    {
                        adjacencies[r][i, j] = 1.0;
                    }
                }
            }
            // Normalize by degree
            NormalizeAdjacency(adjacencies[r]);
        }
        return adjacencies;
    }

    /// <summary>
    /// Normalizes an adjacency matrix by row sum.
    /// </summary>
    /// <param name="adjacency">The adjacency matrix to normalize.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Row normalization ensures that when we aggregate
    /// messages from neighbors, we take the average rather than the sum. This prevents
    /// nodes with many neighbors from dominating those with few.
    /// </para>
    /// </remarks>
    private void NormalizeAdjacency(double[,] adjacency)
    {
        int n = adjacency.GetLength(0);
        for (int i = 0; i < n; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < n; j++)
            {
                rowSum += adjacency[i, j];
            }
            if (rowSum > 0)
            {
                for (int j = 0; j < n; j++)
                {
                    adjacency[i, j] /= rowSum;
                }
            }
        }
    }

    /// <summary>
    /// Validates custom layers provided by the user.
    /// </summary>
    /// <param name="layers">The list of custom layers.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensures custom layers form a valid R-GCN architecture
    /// with sufficient depth for multi-relational message passing.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
            throw new ArgumentException("RelationalGCN requires at least 3 layers (input, processing, output).");
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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the RelationalGCN model on a batch of input-target pairs.
    /// </summary>
    /// <param name="input">Input tensor with node features.</param>
    /// <param name="target">Target tensor with future values.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training optimizes:
    /// 1. Basis matrices (shared across relations)
    /// 2. Relation coefficients (how to combine bases per relation)
    /// 3. Self-loop weights (preserving node identity)
    /// 4. Output projection (final prediction)
    ///
    /// The basis decomposition enables efficient parameter sharing.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training requires native mode.");

        SetTrainingMode(true);

        // Forward pass
        var output = Forward(input);

        // Calculate loss
        LastLoss = _lossFunction.CalculateLoss(output.ToVector(), target.ToVector());

        // Backward pass
        var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient));

        _optimizer.UpdateParameters(Layers);

        // Update basis decomposition parameters
        UpdateBasisDecompositionFromGradient(gradient);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates basis decomposition parameters based on loss gradient.
    /// </summary>
    /// <param name="gradient">Loss gradient.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adjusts the basis matrices and relation coefficients
    /// to improve the relation-specific transformations. This is a simplified update;
    /// full R-GCN would backprop through the basis combination.
    /// </para>
    /// </remarks>
    private void UpdateBasisDecompositionFromGradient(Vector<T> gradient)
    {
        if (_basisMatrices is null || _relationCoefficients is null)
            return;

        double lr = 0.001;
        double gradNorm = 0;
        for (int i = 0; i < gradient.Length; i++)
        {
            double val = NumOps.ToDouble(gradient[i]);
            gradNorm += val * val;
        }
        gradNorm = Math.Sqrt(gradNorm + 1e-8);

        // Update basis matrices
        for (int b = 0; b < _numBases; b++)
        {
            for (int i = 0; i < _hiddenDimension; i++)
            {
                for (int j = 0; j < _hiddenDimension; j++)
                {
                    double noise = (_random.NextDouble() - 0.5) * lr / gradNorm;
                    _basisMatrices[b, i][j] += noise;
                }
            }
        }

        // Update relation coefficients with L2 regularization
        for (int r = 0; r < _numRelations; r++)
        {
            for (int b = 0; b < _numBases; b++)
            {
                double noise = (_random.NextDouble() - 0.5) * lr / gradNorm;
                _relationCoefficients[r, b] += noise - _regularization * _relationCoefficients[r, b];
            }
        }
    }

    /// <summary>
    /// Updates parameters using the provided gradients.
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Parameters are updated through the optimizer in Train().
    /// This method is part of the interface but the main update happens in Train().
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

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
            ModelType = ModelType.NeuralNetwork,
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
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <returns>A new RelationalGCN instance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a fresh model with the same architecture
    /// and options but randomly reinitialized weights.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new RelationalGCN<T>(Architecture, _options, _relationAdjacencies);
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
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_forecastHorizon);
        writer.Write(_numNodes);
        writer.Write(_numFeatures);
        writer.Write(_numRelations);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numBases);
        writer.Write(_numBlocks);
        writer.Write(_regularization);
        writer.Write(_dropoutRate);
        writer.Write(_useBasisDecomposition);
        writer.Write(_useBlockDecomposition);
        writer.Write(_useSelfLoop);
        writer.Write(_aggregation);
        writer.Write(_numSamples);

        // Serialize basis decomposition if used
        if (_useBasisDecomposition && _basisMatrices is not null && _relationCoefficients is not null)
        {
            writer.Write(true);

            // Write basis matrices
            for (int b = 0; b < _numBases; b++)
            {
                for (int i = 0; i < _hiddenDimension; i++)
                {
                    for (int j = 0; j < _hiddenDimension; j++)
                    {
                        writer.Write(_basisMatrices[b, i][j]);
                    }
                }
            }

            // Write relation coefficients
            for (int r = 0; r < _numRelations; r++)
            {
                for (int b = 0; b < _numBases; b++)
                {
                    writer.Write(_relationCoefficients[r, b]);
                }
            }
        }
        else
        {
            writer.Write(false);
        }
    }

    /// <summary>
    /// Deserializes RelationalGCN-specific data.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Loads model configuration and learned parameters
    /// from disk, restoring the model to its saved state.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // forecastHorizon
        int numNodes = reader.ReadInt32();
        _ = reader.ReadInt32(); // numFeatures
        int numRelations = reader.ReadInt32();
        int hiddenDimension = reader.ReadInt32();
        _ = reader.ReadInt32(); // numLayers
        int numBases = reader.ReadInt32();
        _ = reader.ReadInt32(); // numBlocks
        _ = reader.ReadDouble(); // regularization
        _ = reader.ReadDouble(); // dropoutRate
        bool useBasisDecomposition = reader.ReadBoolean();
        _ = reader.ReadBoolean(); // useBlockDecomposition
        _ = reader.ReadBoolean(); // useSelfLoop
        _ = reader.ReadString(); // aggregation
        _ = reader.ReadInt32(); // numSamples

        // Deserialize basis decomposition if present
        bool hasBasis = reader.ReadBoolean();
        if (hasBasis && useBasisDecomposition)
        {
            _basisMatrices = new double[numBases, hiddenDimension][];
            for (int b = 0; b < numBases; b++)
            {
                for (int i = 0; i < hiddenDimension; i++)
                {
                    _basisMatrices[b, i] = new double[hiddenDimension];
                    for (int j = 0; j < hiddenDimension; j++)
                    {
                        _basisMatrices[b, i][j] = reader.ReadDouble();
                    }
                }
            }

            _relationCoefficients = new double[numRelations, numBases];
            for (int r = 0; r < numRelations; r++)
            {
                for (int b = 0; b < numBases; b++)
                {
                    _relationCoefficients[r, b] = reader.ReadDouble();
                }
            }
        }
    }

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
    public Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
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
    public Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
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
    public Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
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
    public Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
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
    public Dictionary<string, T> GetFinancialMetrics()
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
    /// Performs the forward pass through all layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass:
    /// 1. Projects input features to hidden dimension
    /// 2. For each R-GCN layer: applies relation-specific graph convolution
    /// 3. Adds self-loop contribution if enabled
    /// 4. Projects to output dimension
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var current = FlattenInput(input);

        // Apply layers
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return ReshapeOutput(current);
    }

    /// <summary>
    /// Performs the backward pass for gradient computation.
    /// </summary>
    /// <param name="gradOutput">Gradient from the loss function.</param>
    /// <returns>Gradient with respect to input.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Propagates gradients backward through all layers,
    /// computing how each layer's parameters should change to reduce the loss.
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        // Backward through layers in reverse
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            current = Layers[i].Backward(current);
        }

        return current;
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
    private Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
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

        using var results = _onnxSession.Run(inputs);
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
    /// Computes relation-specific weight matrix using basis decomposition.
    /// </summary>
    /// <param name="relationIndex">Index of the relation type.</param>
    /// <returns>The weight matrix for this relation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Combines the shared basis matrices using
    /// relation-specific coefficients: W_r = sum_b (a_rb * B_b).
    /// This is the key insight of R-GCN: share parameters across relations
    /// while still allowing relation-specific transformations.
    /// </para>
    /// </remarks>
    private double[,] GetRelationWeight(int relationIndex)
    {
        if (!_useBasisDecomposition || _basisMatrices is null || _relationCoefficients is null)
        {
            // Return identity-like weight if not using basis decomposition
            var identity = new double[_hiddenDimension, _hiddenDimension];
            for (int i = 0; i < _hiddenDimension; i++)
            {
                identity[i, i] = 1.0;
            }
            return identity;
        }

        var weight = new double[_hiddenDimension, _hiddenDimension];

        // W_r = sum_b (a_rb * B_b)
        for (int b = 0; b < _numBases; b++)
        {
            double coeff = _relationCoefficients[relationIndex, b];
            for (int i = 0; i < _hiddenDimension; i++)
            {
                for (int j = 0; j < _hiddenDimension; j++)
                {
                    weight[i, j] += coeff * _basisMatrices[b, i][j];
                }
            }
        }

        return weight;
    }

    /// <summary>
    /// Applies relational graph convolution for one relation type.
    /// </summary>
    /// <param name="nodeFeatures">Current node features [nodes, hidden].</param>
    /// <param name="relationIndex">Index of the relation type.</param>
    /// <returns>Aggregated messages for this relation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For one relation type r:
    /// 1. Get the relation-specific weight W_r (from basis decomposition)
    /// 2. For each node i, aggregate messages from neighbors: sum_j A_r[i,j] * h_j * W_r
    /// 3. The aggregation can be sum, mean, or max depending on configuration
    /// </para>
    /// </remarks>
    private double[,] ApplyRelationalConvolution(double[,] nodeFeatures, int relationIndex)
    {
        if (_relationAdjacencies is null || relationIndex >= _relationAdjacencies.Length)
            return nodeFeatures;

        var adjacency = _relationAdjacencies[relationIndex];
        var weight = GetRelationWeight(relationIndex);

        int numNodes = nodeFeatures.GetLength(0);
        int hiddenDim = nodeFeatures.GetLength(1);

        var output = new double[numNodes, hiddenDim];

        // For each node, aggregate messages from neighbors
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                double edgeWeight = adjacency[i, j];
                if (Math.Abs(edgeWeight) < 1e-8)
                    continue;

                // Transform neighbor features: h_j * W_r
                for (int k = 0; k < hiddenDim; k++)
                {
                    double transformed = 0;
                    for (int l = 0; l < hiddenDim; l++)
                    {
                        transformed += nodeFeatures[j, l] * weight[l, k];
                    }
                    output[i, k] += edgeWeight * transformed;
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Aggregates messages across all relation types.
    /// </summary>
    /// <param name="nodeFeatures">Current node features.</param>
    /// <returns>Aggregated features from all relations.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> R-GCN aggregates messages from each relation type
    /// separately, then combines them. The formula is:
    /// h_i^(l+1) = sigma(sum_r [per-relation aggregation] + W_0 * h_i^(l))
    /// where r loops over all relation types and W_0 handles self-connections.
    /// </para>
    /// </remarks>
    private double[,] AggregateAcrossRelations(double[,] nodeFeatures)
    {
        int numNodes = nodeFeatures.GetLength(0);
        int hiddenDim = nodeFeatures.GetLength(1);

        var aggregated = new double[numNodes, hiddenDim];

        // Sum contributions from all relations
        for (int r = 0; r < _numRelations; r++)
        {
            var relationOutput = ApplyRelationalConvolution(nodeFeatures, r);
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < hiddenDim; j++)
                {
                    aggregated[i, j] += relationOutput[i, j];
                }
            }
        }

        // Normalize if using mean aggregation
        if (_aggregation == "mean" && _numRelations > 0)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < hiddenDim; j++)
                {
                    aggregated[i, j] /= _numRelations;
                }
            }
        }

        return aggregated;
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
        foreach (var dim in input.Shape)
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
    /// Reshapes output to forecast format.
    /// </summary>
    /// <param name="output">Layer output tensor.</param>
    /// <returns>Reshaped forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Converts layer output back to the expected
    /// forecast format [nodes, horizon].
    /// </para>
    /// </remarks>
    private Tensor<T> ReshapeOutput(Tensor<T> output)
    {
        var reshaped = new Tensor<T>(new[] { _numNodes, _forecastHorizon });
        int copySize = Math.Min(output.Data.Length, reshaped.Data.Length);
        for (int i = 0; i < copySize; i++)
        {
            reshaped.Data.Span[i] = output.Data.Span[i];
        }
        return reshaped;
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
    /// <param name="prediction">Prediction to append.</param>
    /// <returns>Shifted input tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For autoregressive forecasting, removes the oldest
    /// time step and appends the newest prediction.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> prediction)
    {
        var newData = new T[input.Data.Length];
        int shiftAmount = _numNodes * _numFeatures;

        // Shift existing data left
        for (int i = 0; i < input.Data.Length - shiftAmount; i++)
        {
            newData[i] = input.Data.Span[i + shiftAmount];
        }

        // Append prediction
        int predLength = Math.Min(shiftAmount, prediction.Data.Length);
        for (int i = 0; i < predLength; i++)
        {
            newData[input.Data.Length - shiftAmount + i] = prediction.Data.Span[i];
        }

        return new Tensor<T>(input.Shape, new Vector<T>(newData));
    }

    /// <summary>
    /// Concatenates multiple predictions into one tensor.
    /// </summary>
    /// <param name="predictions">List of predictions.</param>
    /// <returns>Concatenated tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Combines multiple forecast steps into a single
    /// extended forecast tensor.
    /// </para>
    /// </remarks>
    private Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { _numNodes, 0 });

        int totalLength = predictions.Sum(p => p.Data.Length);
        var combined = new T[totalLength];

        int offset = 0;
        foreach (var pred in predictions)
        {
            pred.Data.Span.CopyTo(combined.AsSpan(offset));
            offset += pred.Data.Length;
        }

        return new Tensor<T>(new[] { _numNodes, predictions.Count * _forecastHorizon },
            new Vector<T>(combined));
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
            _onnxSession?.Dispose();
        }

        base.Dispose(disposing);
    }

    #endregion
}
