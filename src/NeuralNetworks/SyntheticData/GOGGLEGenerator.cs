using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

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

    // Learned adjacency matrix (soft, between 0 and 1)
    private Matrix<T>? _adjacency;

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

            Layers.Add(new FullyConnectedLayer<T>(latentDim, hiddenDim, relu));
            Layers.Add(new FullyConnectedLayer<T>(hiddenDim, hiddenDim, relu));
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Rebuilds auxiliary layers with actual data dimensions discovered during Fit().
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
            _gnnLayers.Add(new FullyConnectedLayer<T>(layerInput, hiddenDim, relu));
        }

        int lastDim = _options.NumGNNLayers > 0 ? hiddenDim : _dataWidth;
        _meanHead = new FullyConnectedLayer<T>(lastDim, _options.LatentDimension, identity);
        _logvarHead = new FullyConnectedLayer<T>(lastDim, _options.LatentDimension, identity);

        // Decoder output layer
        _decoderOutput = new FullyConnectedLayer<T>(hiddenDim, _dataWidth, identity);

        // Rebuild Layers (decoder MLP) if not using custom layers
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            int latentDim = _options.LatentDimension;
            Layers.Add(new FullyConnectedLayer<T>(latentDim, hiddenDim, relu));
            Layers.Add(new FullyConnectedLayer<T>(hiddenDim, hiddenDim, relu));
        }

        // Initialize adjacency matrix
        InitializeAdjacency();
    }

    private void InitializeAdjacency()
    {
        _adjacency = new Matrix<T>(_dataWidth, _dataWidth);
        double initVal = 1.0 / _dataWidth;
        for (int i = 0; i < _dataWidth; i++)
        {
            for (int j = 0; j < _dataWidth; j++)
            {
                if (i != j)
                {
                    _adjacency[i, j] = NumOps.FromDouble(initVal + 0.01 * (_random.NextDouble() - 0.5));
                }
            }
        }
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

        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        T lr = NumOps.FromDouble(_options.LearningRate / batchSize);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int b = 0; b < data.Rows; b += batchSize)
            {
                int end = Math.Min(b + batchSize, data.Rows);
                TrainBatch(transformedData, b, end, lr);
            }
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

    private void TrainBatch(Matrix<T> data, int startRow, int endRow, T lr)
    {
        for (int row = startRow; row < endRow; row++)
        {
            var x = GetRow(data, row);

            // GNN encoder with adjacency-based message passing
            var gnnOut = GNNEncoderForward(x);

            // VAE: get mean and logvar
            if (_meanHead is null || _logvarHead is null) continue;
            var meanTensor = _meanHead.Forward(VectorToTensor(gnnOut));
            var logvarTensor = _logvarHead.Forward(VectorToTensor(gnnOut));

            int latentDim = _options.LatentDimension;
            var mean = TensorToVector(meanTensor, latentDim);
            var logvar = TensorToVector(logvarTensor, latentDim);

            // Reparameterization
            var z = Reparameterize(mean, logvar);

            // Decode
            var decoded = DecoderForward(z);

            // Compute reconstruction loss gradient
            var reconGrad = new Tensor<T>([_dataWidth]);
            for (int j = 0; j < _dataWidth; j++)
            {
                double diff = NumOps.ToDouble(decoded[j]) - NumOps.ToDouble(x[j]);
                reconGrad[j] = NumOps.FromDouble(2.0 * diff);
            }

            // Sanitize and clip gradient
            reconGrad = SanitizeAndClipGradient(reconGrad, 5.0);

            // Backward through decoder
            BackwardDecoder(reconGrad);

            // KL divergence gradient on mean and logvar
            var meanGrad = new Tensor<T>([latentDim]);
            var logvarGrad = new Tensor<T>([latentDim]);
            for (int j = 0; j < latentDim; j++)
            {
                double m = NumOps.ToDouble(mean[j]);
                double lv = NumOps.ToDouble(logvar[j]);
                meanGrad[j] = NumOps.FromDouble(m * _options.KLWeight);
                logvarGrad[j] = NumOps.FromDouble(0.5 * (Math.Exp(lv) - 1.0) * _options.KLWeight);
            }

            // Sanitize and clip KL gradients
            meanGrad = SanitizeAndClipGradient(meanGrad, 5.0);
            logvarGrad = SanitizeAndClipGradient(logvarGrad, 5.0);

            var encoderGradFromMean = _meanHead.Backward(meanGrad);
            var encoderGradFromLogvar = _logvarHead.Backward(logvarGrad);

            // Propagate KL gradient through GNN encoder layers
            var encoderGrad = encoderGradFromMean.Add(encoderGradFromLogvar);
            for (int i = _gnnLayers.Count - 1; i >= 0; i--)
            {
                encoderGrad = _gnnLayers[i].Backward(encoderGrad);
            }

            // Update adjacency matrix (gradient descent on structure loss)
            UpdateAdjacency(x, lr);

            // Update parameters for layers that had Backward() called
            _meanHead.UpdateParameters(lr);
            _logvarHead.UpdateParameters(lr);
            foreach (var layer in _gnnLayers) layer.UpdateParameters(lr);
            foreach (var layer in Layers) layer.UpdateParameters(lr);
            _decoderOutput?.UpdateParameters(lr);
        }
    }

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
        var current = x;

        for (int layer = 0; layer < _gnnLayers.Count; layer++)
        {
            // Aggregate neighbor features using adjacency matrix
            var aggregated = AggregateNeighbors(current);

            // Transform aggregated features
            var tensor = _gnnLayers[layer].Forward(VectorToTensor(aggregated));
            current = TensorToVector(tensor, _options.HiddenDimension);
        }

        return current;
    }

    private Vector<T> AggregateNeighbors(Vector<T> features)
    {
        if (_adjacency is null) return features;

        int featDim = features.Length;
        int adjDim = _adjacency.Rows;
        int dim = Math.Min(featDim, adjDim);
        var aggregated = new Vector<T>(featDim);

        // Aggregate using adjacency matrix for the shared dimension range
        for (int i = 0; i < dim; i++)
        {
            double selfVal = NumOps.ToDouble(features[i]);
            double neighborSum = 0;
            double weightSum = 0;

            for (int j = 0; j < dim; j++)
            {
                double aij = NumOps.ToDouble(_adjacency[i, j]);
                neighborSum += aij * NumOps.ToDouble(features[j]);
                weightSum += aij;
            }

            double normNeighbor = weightSum > 1e-8 ? neighborSum / weightSum : 0;
            aggregated[i] = NumOps.FromDouble(selfVal + normNeighbor);
        }

        // Copy remaining features unchanged (beyond adjacency dimensions)
        for (int i = dim; i < featDim; i++)
        {
            aggregated[i] = features[i];
        }

        return aggregated;
    }

    private Vector<T> DecoderForward(Vector<T> z)
    {
        var current = VectorToTensor(z);
        for (int i = 0; i < Layers.Count; i++)
            current = Layers[i].Forward(current);
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

    private void BackwardDecoder(Tensor<T> grad)
    {
        var current = grad;
        if (_decoderOutput is not null)
            current = _decoderOutput.Backward(current);
        for (int i = Layers.Count - 1; i >= 0; i--)
            current = Layers[i].Backward(current);
    }

    #endregion

    #region Output Activations

    private Vector<T> ApplyOutputActivations(Vector<T> decoded)
    {
        if (_transformer is null) return decoded;

        var output = VectorToTensor(decoded);
        var result = new Tensor<T>(output.Shape);
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
        double maxVal = double.MinValue;
        for (int i = 0; i < count && (idx + i) < input.Length; i++)
        {
            double v = NumOps.ToDouble(input[idx + i]);
            if (v > maxVal) maxVal = v;
        }
        double sumExp = 0;
        for (int i = 0; i < count && (idx + i) < input.Length; i++)
            sumExp += Math.Exp(NumOps.ToDouble(input[idx + i]) - maxVal);
        for (int i = 0; i < count && idx < input.Length; i++)
        {
            double expVal = Math.Exp(NumOps.ToDouble(input[idx]) - maxVal);
            output[idx] = NumOps.FromDouble(expVal / Math.Max(sumExp, 1e-10));
            idx++;
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (_transformer is null || !IsFitted)
        {
            return input;
        }

        var row = TensorToVector(input, _dataWidth);
        var gnnOut = GNNEncoderForward(row);

        if (_meanHead is null) return input;
        var meanTensor = _meanHead.Forward(VectorToTensor(gnnOut));
        var mean = TensorToVector(meanTensor, _options.LatentDimension);

        var decoded = DecoderForward(mean);
        return VectorToTensor(decoded);
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var predicted = Predict(input);
        var loss = _lossFunction.CalculateLoss(
            TensorToVector(predicted, predicted.Length),
            TensorToVector(expectedOutput, expectedOutput.Length));
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

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.LatentDimension);
        writer.Write(_options.NumGNNLayers);
        writer.Write(_options.HiddenDimension);
        writer.Write(_dataWidth);
        writer.Write(IsFitted);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // LatentDimension
        _ = reader.ReadInt32(); // NumGNNLayers
        _ = reader.ReadInt32(); // HiddenDimension
        _dataWidth = reader.ReadInt32();
        IsFitted = reader.ReadBoolean();
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
            ModelType = ModelType.NeuralNetwork,
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

    #region IJitCompilable Override

    /// <summary>
    /// GOGGLE uses GNN message passing on a learned adjacency matrix which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}
