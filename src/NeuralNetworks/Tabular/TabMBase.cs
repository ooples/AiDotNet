using AiDotNet.Helpers;
using AiDotNet.Engines;
using AiDotNet.Attributes;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base implementation of TabM, a parameter-efficient ensemble model for tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabM uses BatchEnsemble-style parameter sharing to create multiple ensemble members
/// with minimal parameter overhead. The architecture consists of:
/// 1. Optional feature embedding layer
/// 2. Multiple BatchEnsemble hidden layers with activation and normalization
/// 3. Prediction head
/// </para>
/// <para>
/// <b>For Beginners:</b> TabM is a neural network that combines the power of ensemble
/// methods (multiple models voting) with the efficiency of parameter sharing.
///
/// Architecture overview:
/// 1. **Input Layer**: Raw features or embedded features
/// 2. **Hidden Layers**: BatchEnsemble layers with shared weights + per-member modulation
/// 3. **Output Layer**: Predictions for each ensemble member
/// 4. **Aggregation**: Average (or other) of member predictions
///
/// Why TabM works well:
/// - Ensembles reduce variance and improve generalization
/// - Parameter sharing keeps the model small and efficient
/// - Diverse members (via rank vectors) capture different aspects of the data
/// - Training is stable and convergence is fast
///
/// Comparison to other methods:
/// - vs Gradient Boosting: Often similar performance, TabM is more parallelizable
/// - vs Random Forests: TabM learns feature interactions, RF doesn't
/// - vs Deep Ensembles: TabM is 4-10x more parameter efficient
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract partial class TabMBase<T>
{
    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Hardware-accelerated engine for tensor operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// The model configuration options.
    /// </summary>
    protected readonly TabMOptions<T> Options;

    /// <summary>
    /// Number of input features.
    /// </summary>
    protected readonly int NumFeatures;

    // Feature embedding (optional)
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T>? _featureEmbeddings;  // [numFeatures, embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _featureEmbeddingsGrad;

    // BatchEnsemble hidden layers
    private readonly List<BatchEnsembleLayer<T>> _hiddenLayers;

    // Cache for backward pass
    [Scratch]
    private readonly List<Tensor<T>> _hiddenOutputsCache;

    /// <summary>
    /// Gets the number of ensemble members.
    /// </summary>
    public int NumMembers => Options.NumEnsembleMembers;

    /// <summary>
    /// Gets the hidden layer dimensions.
    /// </summary>
    public int[] HiddenDimensions => Options.HiddenDimensions;

    /// <summary>
    /// Trainable layers a subclass adds on top of the backbone, walked after it by every
    /// parameter surface below.
    /// </summary>
    /// <remarks>
    /// The task head. A classifier and a regressor share this whole backbone and differ only by a
    /// final projection, and each re-implemented the surfaces just to append that one layer. The
    /// subclass now states WHAT it adds and the folds below decide where it goes, so the count and
    /// the vector cannot place it differently. Mirrors NeuralNetworkBase.GetExtraTrainableLayers,
    /// which solves the same problem for models that derive from it -- this base has no base at
    /// all, so it cannot inherit that one.
    /// </remarks>
    protected virtual IEnumerable<ILayer<T>> GetExtraTrainableLayers()
        => AiDotNet.Models.Parameters.GeneratedParameterDiscovery.EnumerateDerivedLayers<T>(this, typeof(TabMBase<T>));

    /// <summary>
    /// Gets the total number of trainable parameters in the base model.
    /// </summary>
    public virtual long ParameterCount
    {
        get
        {
            int count = 0;
            if (_featureEmbeddings != null) count += _featureEmbeddings.Length;
            foreach (var layer in _hiddenLayers)
            {
                count += (int)layer.ParameterCount;
            }

            foreach (var extra in GetExtraTrainableLayers())
            {
                if (extra is not null) count += (int)extra.ParameterCount;
            }
            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the TabMBase class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="options">Model configuration options.</param>
    protected TabMBase(int numFeatures, TabMOptions<T>? options = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new TabMOptions<T>();
        NumFeatures = numFeatures;

        // Initialize feature embeddings if enabled
        if (Options.UseFeatureEmbeddings)
        {
            _featureEmbeddings = new Tensor<T>([numFeatures, Options.FeatureEmbeddingDimension]);
            InitializeXavier(_featureEmbeddings);
        }

        // Determine input dimension to first hidden layer
        int inputDim = Options.UseFeatureEmbeddings
            ? numFeatures * Options.FeatureEmbeddingDimension
            : numFeatures;

        // Initialize BatchEnsemble hidden layers
        _hiddenLayers = new List<BatchEnsembleLayer<T>>();
        int prevDim = inputDim;

        foreach (int hiddenDim in Options.HiddenDimensions)
        {
            var layer = new BatchEnsembleLayer<T>(
                prevDim,
                hiddenDim,
                Options.NumEnsembleMembers,
                Options.UseBias,
                Options.RankInitScale);

            _hiddenLayers.Add(layer);
            prevDim = hiddenDim;
        }

        _hiddenOutputsCache = new List<Tensor<T>>();
    }

    /// <summary>
    /// Initializes a tensor with Xavier/Glorot initialization.
    /// </summary>
    private void InitializeXavier(Tensor<T> tensor)
    {
        var random = RandomHelper.CreateSecureRandom();
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape.Length > 1 ? tensor.Shape[1] : 1;
        double stdDev = Math.Sqrt(2.0 / (fanIn + fanOut));

        for (int i = 0; i < tensor.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            tensor[i] = NumOps.FromDouble(normal * stdDev);
        }
    }

    /// <summary>
    /// Gets the output dimension of the last hidden layer.
    /// </summary>
    protected int GetLastHiddenDim()
    {
        return Options.HiddenDimensions.Length > 0
            ? Options.HiddenDimensions[^1]
            : (Options.UseFeatureEmbeddings ? NumFeatures * Options.FeatureEmbeddingDimension : NumFeatures);
    }

    /// <summary>
    /// Performs the forward pass through the TabM backbone.
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <returns>Hidden representation [batch_size * num_members, last_hidden_dim].</returns>
    protected Tensor<T> ForwardBackbone(Tensor<T> features)
    {
        _hiddenOutputsCache.Clear();
        int batchSize = features.Shape[0];

        // Step 1: Apply feature embeddings if enabled
        Tensor<T> x;
        if (_featureEmbeddings != null)
        {
            // Embed each feature: [batch, features] -> [batch, features * embed_dim]
            int embedDim = Options.FeatureEmbeddingDimension;
            var values = TabularMetrics.Flat(features, batchSize * NumFeatures).Reshape(batchSize, NumFeatures, 1);
            x = Engine.TensorMultiply(values, _featureEmbeddings.Reshape(1, NumFeatures, embedDim))
                .Reshape(batchSize, NumFeatures * embedDim);
        }
        else
        {
            x = features;
        }

        // Step 2: Pass through BatchEnsemble hidden layers
        foreach (var layer in _hiddenLayers)
        {
            _hiddenOutputsCache.Add(x);
            var output = layer.Forward(x);

            // ReLU as the scalar test wrote it: anything that compares below zero (negative or NaN) becomes 0
            var zeros = Engine.TensorZerosLike(output);
            output = Engine.TensorWhere(Engine.TensorGreaterEqual(output, zeros), output, zeros);

            x = output;
        }

        return x;
    }

    /// <summary>
    /// Averages predictions across ensemble members.
    /// </summary>
    /// <param name="memberOutputs">Per-member outputs [batch_size * num_members, output_dim].</param>
    /// <param name="outputDim">Output dimension.</param>
    /// <returns>Averaged outputs [batch_size, output_dim].</returns>
    protected Tensor<T> AverageMemberOutputs(Tensor<T> memberOutputs, int outputDim)
    {
        int expandedBatchSize = memberOutputs.Shape[0];
        int batchSize = expandedBatchSize / Options.NumEnsembleMembers;

        var scale = NumOps.FromDouble(1.0 / Options.NumEnsembleMembers);
        var members = TabularMetrics.Flat(memberOutputs, batchSize * Options.NumEnsembleMembers * outputDim)
            .Reshape(batchSize, Options.NumEnsembleMembers, outputDim);
        var averaged = TensorTapeOps.TapeMultiplyScalar(Engine, Engine.ReduceSum(members, new[] { 1 }, false), scale);

        return averaged;
    }

    /// <summary>
    /// Updates all parameters using the calculated gradients.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        // Update feature embeddings if used
        if (_featureEmbeddings != null && _featureEmbeddingsGrad != null)
        {
            var updated = Engine.TensorSubtract(_featureEmbeddings,
                Engine.TensorMultiplyScalar(_featureEmbeddingsGrad, learningRate));
            updated.AsSpan().CopyTo(_featureEmbeddings.AsWritableSpan());
        }

        // Update hidden layers
        foreach (var layer in _hiddenLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public virtual Vector<T> GetParameters()
    {
        var parts = new List<Vector<T>>();

        // Feature embeddings
        if (_featureEmbeddings != null)
        {
            parts.Add(_featureEmbeddings.ToVector());
        }

        // Hidden layers
        foreach (var layer in _hiddenLayers)
        {
            parts.Add(layer.GetParameters());
        }

        // Subclass head, last, in the same position ParameterCount puts it.
        foreach (var extra in GetExtraTrainableLayers())
        {
            if (extra is null) continue;
            parts.Add(extra.GetParameters());
        }

        return ConcatenateVectors(parts);
    }

    /// <summary>
    /// Joins parameter or gradient vectors end to end with bulk copies.
    /// </summary>
    private static Vector<T> ConcatenateVectors(List<Vector<T>> parts)
    {
        int total = 0;
        foreach (var part in parts)
        {
            total += part.Length;
        }

        var result = new Vector<T>(total);
        var destination = result.AsWritableSpan();
        int offset = 0;
        foreach (var part in parts)
        {
            part.AsSpan().CopyTo(destination.Slice(offset, part.Length));
            offset += part.Length;
        }

        return result;
    }

    /// <summary>
    /// Sets all trainable parameters from a vector.
    /// </summary>
    public virtual void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Feature embeddings
        if (_featureEmbeddings != null)
        {
            parameters.AsSpan().Slice(offset, _featureEmbeddings.Length).CopyTo(_featureEmbeddings.AsWritableSpan());
            offset += _featureEmbeddings.Length;
        }

        // Hidden layers
        foreach (var layer in _hiddenLayers)
        {
            int layerCount = checked((int)layer.ParameterCount);
            var layerParams = parameters.GetSubVector(offset, layerCount);
            offset += layerCount;
            layer.SetParameters(layerParams);
        }
    }

    /// <summary>
    /// Gets parameter gradients as a single vector.
    /// </summary>
    public virtual Vector<T> GetParameterGradients()
    {
        var parts = new List<Vector<T>>();

        // Feature embeddings (zeros until a gradient exists)
        if (_featureEmbeddings != null)
        {
            parts.Add(_featureEmbeddingsGrad != null
                ? _featureEmbeddingsGrad.ToVector()
                : new Vector<T>(_featureEmbeddings.Length));
        }

        // Hidden layers
        foreach (var layer in _hiddenLayers)
        {
            parts.Add(layer.GetParameterGradients());
        }

        return ConcatenateVectors(parts);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _featureEmbeddingsGrad = null;
        _hiddenOutputsCache.Clear();

        foreach (var layer in _hiddenLayers)
        {
            layer.ResetGradients();
        }
    }

    /// <summary>
    /// Gets the diversity of ensemble predictions (standard deviation across members).
    /// </summary>
    /// <param name="memberOutputs">Per-member outputs [batch_size * num_members, output_dim].</param>
    /// <param name="outputDim">Output dimension.</param>
    /// <returns>Per-sample diversity scores [batch_size].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensemble diversity measures how different the predictions are
    /// across ensemble members. Higher diversity often indicates:
    /// - Better uncertainty estimation
    /// - More robust final predictions
    /// - Each member has learned something different
    ///
    /// Low diversity might mean:
    /// - Members have collapsed to similar predictions
    /// - Consider increasing rank_init_scale or training longer
    /// </para>
    /// </remarks>
    public Vector<T> GetEnsembleDiversity(Tensor<T> memberOutputs, int outputDim)
    {
        int expandedBatchSize = memberOutputs.Shape[0];
        int batchSize = expandedBatchSize / Options.NumEnsembleMembers;

        int members = Options.NumEnsembleMembers;
        var outputs = TabularMetrics.Flat(memberOutputs, batchSize * members * outputDim)
            .Reshape(batchSize, members, outputDim);

        // Mean across members, then the population variance over every (member, output) pair of a sample
        var mean = Engine.TensorDivideScalar(Engine.ReduceSum(outputs, new[] { 1 }, true), NumOps.FromDouble(members));
        var diff = Engine.TensorSubtract(outputs, mean);
        var totalVariance = Engine.TensorDivideScalar(
            Engine.ReduceSum(Engine.TensorMultiply(diff, diff), new[] { 1, 2 }, false),
            NumOps.FromDouble(members * outputDim));

        var diversity = Engine.TensorSqrt(totalVariance).ToVector();

        return diversity;
    }
}
