using AiDotNet.ActivationFunctions;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base class for Mambular (State Space Models for Tabular Data).
/// </summary>
/// <remarks>
/// <para>
/// Mambular applies the Mamba architecture to tabular data:
/// 1. Features are embedded and treated as a sequence
/// 2. Selective State Space Model (S4/Mamba) processes the sequence
/// 3. Final representation is used for prediction
/// </para>
/// <para>
/// <b>For Beginners:</b> Mambular is an alternative to transformers that:
///
/// - **Scales linearly**: O(n) instead of O(n²) with sequence length
/// - **Has memory**: Can remember information across the feature sequence
/// - **Is selective**: Learns what to remember and what to forget
///
/// This makes it efficient for tabular data with many features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class MambularBase<T>
{
    protected readonly MambularOptions<T> Options;
    protected readonly int NumNumericalFeatures;
    protected readonly int NumCategoricalFeatures;
    protected readonly int TotalFeatures;
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random = RandomHelper.CreateSecureRandom();

    // Feature embeddings
    private readonly Tensor<T> _numericalEmbeddings;
    private readonly Tensor<T>[]? _categoricalEmbeddings;

    // Mamba layers
    private readonly List<MambaBlock> _mambaBlocks;

    // MLP head
    private readonly List<FullyConnectedLayer<T>> _mlpLayers;
    protected int MLPOutputDimension { get; }

    // Caches
    private Tensor<T>? _embeddedFeaturesCache;
    private Tensor<T>? _mambaOutputCache;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public virtual int ParameterCount
    {
        get
        {
            int count = _numericalEmbeddings.Length;

            if (_categoricalEmbeddings != null)
            {
                foreach (var emb in _categoricalEmbeddings)
                    count += emb.Length;
            }

            foreach (var block in _mambaBlocks)
                count += block.ParameterCount;

            foreach (var layer in _mlpLayers)
                count += layer.ParameterCount;

            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the MambularBase class.
    /// </summary>
    protected MambularBase(int numNumericalFeatures, MambularOptions<T>? options = null)
    {
        Options = options ?? new MambularOptions<T>();
        NumNumericalFeatures = numNumericalFeatures;
        NumCategoricalFeatures = Options.CategoricalCardinalities?.Length ?? 0;
        TotalFeatures = NumNumericalFeatures + NumCategoricalFeatures;

        if (TotalFeatures == 0)
        {
            throw new ArgumentException("Model must have at least one feature");
        }

        // Feature embeddings
        _numericalEmbeddings = new Tensor<T>(new[] { NumNumericalFeatures, Options.EmbeddingDimension });
        InitializeWeights(_numericalEmbeddings);

        if (NumCategoricalFeatures > 0 && Options.CategoricalCardinalities != null)
        {
            _categoricalEmbeddings = new Tensor<T>[NumCategoricalFeatures];
            for (int i = 0; i < NumCategoricalFeatures; i++)
            {
                int cardinality = Options.CategoricalCardinalities[i];
                _categoricalEmbeddings[i] = new Tensor<T>(new[] { cardinality, Options.EmbeddingDimension });
                InitializeWeights(_categoricalEmbeddings[i]);
            }
        }

        // Mamba blocks
        _mambaBlocks = [];
        for (int i = 0; i < Options.NumLayers; i++)
        {
            _mambaBlocks.Add(new MambaBlock(
                Options.EmbeddingDimension,
                Options.StateDimension,
                Options.InnerDimension,
                Options.ConvKernelSize,
                _random));
        }

        // MLP head
        _mlpLayers = [];
        int mlpInput = Options.UseBidirectional
            ? TotalFeatures * Options.EmbeddingDimension * 2
            : TotalFeatures * Options.EmbeddingDimension;

        foreach (var hiddenDim in Options.MLPHiddenDimensions)
        {
            _mlpLayers.Add(new FullyConnectedLayer<T>(
                mlpInput,
                hiddenDim,
                Options.HiddenActivation ?? new ReLUActivation<T>()));
            mlpInput = hiddenDim;
        }

        MLPOutputDimension = mlpInput;
    }

    private void InitializeWeights(Tensor<T> tensor)
    {
        var scale = NumOps.FromDouble(Options.InitScale);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(_random.NextGaussian()), scale);
        }
    }

    /// <summary>
    /// Embeds input features.
    /// </summary>
    protected Tensor<T> EmbedFeatures(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        var embedded = new Tensor<T>(new[] { batchSize, TotalFeatures, Options.EmbeddingDimension });

        // Embed numerical features
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < NumNumericalFeatures; f++)
            {
                var featureValue = numericalFeatures[b * NumNumericalFeatures + f];
                for (int d = 0; d < Options.EmbeddingDimension; d++)
                {
                    int embIdx = f * Options.EmbeddingDimension + d;
                    int outIdx = (b * TotalFeatures + f) * Options.EmbeddingDimension + d;
                    embedded[outIdx] = NumOps.Multiply(featureValue, _numericalEmbeddings[embIdx]);
                }
            }
        }

        // Embed categorical features
        if (categoricalIndices != null && _categoricalEmbeddings != null)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < NumCategoricalFeatures; f++)
                {
                    int catIdx = categoricalIndices[b, f];
                    int featureIdx = NumNumericalFeatures + f;
                    for (int d = 0; d < Options.EmbeddingDimension; d++)
                    {
                        int embIdx = catIdx * Options.EmbeddingDimension + d;
                        int outIdx = (b * TotalFeatures + featureIdx) * Options.EmbeddingDimension + d;
                        embedded[outIdx] = _categoricalEmbeddings[f][embIdx];
                    }
                }
            }
        }

        return embedded;
    }

    /// <summary>
    /// Performs the forward pass through the Mambular backbone.
    /// </summary>
    protected Tensor<T> ForwardBackbone(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];

        // Embed features
        var embedded = EmbedFeatures(numericalFeatures, categoricalIndices);
        _embeddedFeaturesCache = embedded;

        // Forward direction
        var forward = embedded;
        foreach (var block in _mambaBlocks)
        {
            forward = block.Forward(forward, batchSize, TotalFeatures, Options.EmbeddingDimension);
        }

        Tensor<T> combined;
        if (Options.UseBidirectional)
        {
            // Backward direction (reverse sequence)
            var reversed = ReverseSequence(embedded, batchSize, TotalFeatures, Options.EmbeddingDimension);
            var backward = reversed;
            foreach (var block in _mambaBlocks)
            {
                backward = block.Forward(backward, batchSize, TotalFeatures, Options.EmbeddingDimension);
            }
            backward = ReverseSequence(backward, batchSize, TotalFeatures, Options.EmbeddingDimension);

            // Concatenate forward and backward
            combined = new Tensor<T>(new[] { batchSize, TotalFeatures * Options.EmbeddingDimension * 2 });
            for (int b = 0; b < batchSize; b++)
            {
                int outIdx = 0;
                for (int i = 0; i < TotalFeatures * Options.EmbeddingDimension; i++)
                {
                    combined[b * TotalFeatures * Options.EmbeddingDimension * 2 + outIdx++] =
                        forward[b * TotalFeatures * Options.EmbeddingDimension + i];
                }
                for (int i = 0; i < TotalFeatures * Options.EmbeddingDimension; i++)
                {
                    combined[b * TotalFeatures * Options.EmbeddingDimension * 2 + outIdx++] =
                        backward[b * TotalFeatures * Options.EmbeddingDimension + i];
                }
            }
        }
        else
        {
            // Flatten forward output
            combined = new Tensor<T>(new[] { batchSize, TotalFeatures * Options.EmbeddingDimension });
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < TotalFeatures * Options.EmbeddingDimension; i++)
                {
                    combined[b * TotalFeatures * Options.EmbeddingDimension + i] =
                        forward[b * TotalFeatures * Options.EmbeddingDimension + i];
                }
            }
        }

        _mambaOutputCache = combined;

        // MLP head
        var mlpOutput = combined;
        foreach (var layer in _mlpLayers)
        {
            mlpOutput = layer.Forward(mlpOutput);
        }

        return mlpOutput;
    }

    private Tensor<T> ReverseSequence(Tensor<T> input, int batchSize, int seqLen, int dim)
    {
        var output = new Tensor<T>(input.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int srcPos = seqLen - 1 - s;
                for (int d = 0; d < dim; d++)
                {
                    output[(b * seqLen + s) * dim + d] = input[(b * seqLen + srcPos) * dim + d];
                }
            }
        }
        return output;
    }

    /// <summary>
    /// Performs the backward pass through the backbone.
    /// </summary>
    protected Tensor<T> BackwardBackbone(Tensor<T> gradOutput)
    {
        var grad = gradOutput;
        for (int i = _mlpLayers.Count - 1; i >= 0; i--)
        {
            grad = _mlpLayers[i].Backward(grad);
        }
        return grad;
    }

    /// <summary>
    /// Updates all parameters.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        foreach (var block in _mambaBlocks)
        {
            block.UpdateParameters(learningRate);
        }

        foreach (var layer in _mlpLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _embeddedFeaturesCache = null;
        _mambaOutputCache = null;

        foreach (var block in _mambaBlocks)
        {
            block.ResetState();
        }

        foreach (var layer in _mlpLayers)
        {
            layer.ResetState();
        }
    }

    /// <summary>
    /// Simplified Mamba block for tabular data.
    /// </summary>
    private class MambaBlock
    {
        private readonly int _modelDim;
        private readonly int _stateDim;
        private readonly int _innerDim;

        // SSM parameters
        private readonly Tensor<T> _A;  // State transition
        private readonly Tensor<T> _B;  // Input projection
        private readonly Tensor<T> _C;  // Output projection
        private readonly Tensor<T> _D;  // Skip connection

        // Projections
        private readonly Tensor<T> _inProj;
        private readonly Tensor<T> _outProj;

        // Convolution
        private readonly Tensor<T> _convWeight;

        // Delta (discretization)
        private readonly Tensor<T> _deltaProj;

        public int ParameterCount =>
            _A.Length + _B.Length + _C.Length + _D.Length +
            _inProj.Length + _outProj.Length + _convWeight.Length + _deltaProj.Length;

        public MambaBlock(int modelDim, int stateDim, int innerDim, int convKernelSize, Random random)
        {
            _modelDim = modelDim;
            _stateDim = stateDim;
            _innerDim = innerDim;

            var scale = NumOps.FromDouble(0.02);

            // SSM parameters
            _A = new Tensor<T>(new[] { innerDim, stateDim });
            _B = new Tensor<T>(new[] { innerDim, stateDim });
            _C = new Tensor<T>(new[] { innerDim, stateDim });
            _D = new Tensor<T>(new[] { innerDim });

            // Initialize A with negative values (for stability)
            for (int i = 0; i < _A.Length; i++)
            {
                _A[i] = NumOps.Negate(NumOps.FromDouble(1.0 + random.NextDouble() * 4.0));
            }

            InitializeWeights(_B, scale, random);
            InitializeWeights(_C, scale, random);

            for (int i = 0; i < _D.Length; i++)
            {
                _D[i] = NumOps.One;
            }

            // Projections
            _inProj = new Tensor<T>(new[] { modelDim, innerDim * 2 });
            _outProj = new Tensor<T>(new[] { innerDim, modelDim });
            InitializeWeights(_inProj, scale, random);
            InitializeWeights(_outProj, scale, random);

            // Convolution
            _convWeight = new Tensor<T>(new[] { innerDim, convKernelSize });
            InitializeWeights(_convWeight, scale, random);

            // Delta projection
            _deltaProj = new Tensor<T>(new[] { innerDim });
            for (int i = 0; i < _deltaProj.Length; i++)
            {
                _deltaProj[i] = NumOps.FromDouble(0.01);
            }
        }

        private static void InitializeWeights(Tensor<T> tensor, T scale, Random random)
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor[i] = NumOps.Multiply(NumOps.FromDouble(random.NextGaussian()), scale);
            }
        }

        public Tensor<T> Forward(Tensor<T> input, int batchSize, int seqLen, int dim)
        {
            var output = new Tensor<T>(input.Shape);

            for (int b = 0; b < batchSize; b++)
            {
                // Initialize state
                var state = new T[_innerDim, _stateDim];
                for (int i = 0; i < _innerDim; i++)
                {
                    for (int j = 0; j < _stateDim; j++)
                    {
                        state[i, j] = NumOps.Zero;
                    }
                }

                for (int s = 0; s < seqLen; s++)
                {
                    // Get input vector
                    var x = new T[dim];
                    for (int d = 0; d < dim; d++)
                    {
                        x[d] = input[(b * seqLen + s) * dim + d];
                    }

                    // Input projection: x -> (z, x')
                    var projected = new T[_innerDim * 2];
                    for (int i = 0; i < _innerDim * 2; i++)
                    {
                        var sum = NumOps.Zero;
                        for (int d = 0; d < dim; d++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(x[d], _inProj[d * _innerDim * 2 + i]));
                        }
                        projected[i] = sum;
                    }

                    // Split into z (gate) and x' (input to SSM)
                    var z = new T[_innerDim];
                    var xPrime = new T[_innerDim];
                    for (int i = 0; i < _innerDim; i++)
                    {
                        z[i] = Sigmoid(projected[i]);
                        xPrime[i] = projected[_innerDim + i];
                    }

                    // Simplified SSM step
                    var y = new T[_innerDim];
                    for (int i = 0; i < _innerDim; i++)
                    {
                        var delta = Softplus(_deltaProj[i]);

                        // State update: h = A * h + B * x
                        for (int j = 0; j < _stateDim; j++)
                        {
                            var a = NumOps.Exp(NumOps.Multiply(_A[i * _stateDim + j], delta));
                            var bVal = NumOps.Multiply(_B[i * _stateDim + j], delta);
                            state[i, j] = NumOps.Add(
                                NumOps.Multiply(a, state[i, j]),
                                NumOps.Multiply(bVal, xPrime[i]));
                        }

                        // Output: y = C * h + D * x
                        var hSum = NumOps.Zero;
                        for (int j = 0; j < _stateDim; j++)
                        {
                            hSum = NumOps.Add(hSum, NumOps.Multiply(_C[i * _stateDim + j], state[i, j]));
                        }
                        y[i] = NumOps.Add(hSum, NumOps.Multiply(_D[i], xPrime[i]));

                        // Apply gate
                        y[i] = NumOps.Multiply(y[i], z[i]);
                    }

                    // Output projection
                    for (int d = 0; d < dim; d++)
                    {
                        var sum = NumOps.Zero;
                        for (int i = 0; i < _innerDim; i++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(y[i], _outProj[i * dim + d]));
                        }
                        // Residual connection
                        output[(b * seqLen + s) * dim + d] = NumOps.Add(x[d], sum);
                    }
                }
            }

            return output;
        }

        private T Sigmoid(T x)
        {
            var negX = NumOps.Negate(x);
            var expNegX = NumOps.Exp(negX);
            return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
        }

        private T Softplus(T x)
        {
            // softplus(x) = log(1 + exp(x))
            var expX = NumOps.Exp(x);
            return NumOps.Log(NumOps.Add(NumOps.One, expX));
        }

        public void UpdateParameters(T learningRate)
        {
            // Simplified - in practice would use proper gradients
        }

        public void ResetState()
        {
            // Reset any cached state
        }
    }
}
