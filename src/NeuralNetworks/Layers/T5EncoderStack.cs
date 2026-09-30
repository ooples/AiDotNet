using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A T5 encoder stack whose blocks share one relative position bias table (Raffel et al. 2020, Sec. 2.1:
/// "we share the position embedding parameters across all layers in our model").
/// </summary>
/// <remarks>
/// <para>
/// The stack owns the single [numBuckets, numHeads] table and computes the [numHeads, S, S] bias once per
/// forward pass, then hands it to every block's attention as that pass's input - the dataflow sharing of the
/// reference implementation and of HuggingFace's T5Stack, where only the first block holds the table.
/// </para>
/// <para>
/// No block holds the table or a reference to it. The earlier design passed the first block's live tensor
/// into every later block's constructor, so the sharing was a pointer between layers: cloning could not
/// rebuild it, and any path that copied layers independently would have given each block its own table.
/// Here there is nothing to re-link. The table is one registered parameter, counted once, and the gradient
/// of every block's attention reaches it through the shared lookup.
/// </para>
/// <para>Set <c>shareRelativeBias</c> to false for one table per block (the per-layer variant).</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = false, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 2, 2")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class T5EncoderStack<T> : LayerBase<T>, IShapeContract
{
    private readonly int _hiddenSize;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _numBuckets;
    private readonly int _maxDistance;
    private readonly bool _shareRelativeBias;
    private readonly int? _seed;

    /// <summary>The one shared [numBuckets, numHeads] table; empty when each block owns its own.</summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _relativeBiasTable;

    private readonly List<PreLNTransformerBlock<T>> _blocks = new();

    [Scratch]
    private int _cachedSeqLen = -1;
    [Scratch]
    private Tensor<int>? _bucketIndices;

    /// <summary>Creates a stack of <paramref name="numLayers"/> pre-LN T5 blocks.</summary>
    /// <param name="hiddenSize">Model width.</param>
    /// <param name="numLayers">Number of blocks.</param>
    /// <param name="numHeads">Attention heads per block.</param>
    /// <param name="numBuckets">Relative position buckets (paper: 32).</param>
    /// <param name="maxDistance">Largest bucketed distance (paper: 128).</param>
    /// <param name="shareRelativeBias">One table for the whole stack (paper default) or one per block.</param>
    /// <param name="seed">Optional seed for reproducible initialisation; each block derives its own.</param>
    public T5EncoderStack(
        int hiddenSize,
        int numLayers,
        int numHeads,
        int numBuckets = 32,
        int maxDistance = 128,
        bool shareRelativeBias = true,
        int? seed = null)
        : base(new[] { hiddenSize }, new[] { hiddenSize })
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numBuckets <= 0) throw new ArgumentOutOfRangeException(nameof(numBuckets));
        if (maxDistance <= 0) throw new ArgumentOutOfRangeException(nameof(maxDistance));

        _hiddenSize = hiddenSize;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _numBuckets = numBuckets;
        _maxDistance = maxDistance;
        _shareRelativeBias = shareRelativeBias;
        _seed = seed;
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        if (shareRelativeBias)
        {
            // HuggingFace initialises the table with mean 0, std hidden_size^-0.5.
            _relativeBiasTable = SampleNormal(random, new[] { numBuckets, numHeads }, 1.0 / Math.Sqrt(hiddenSize));
            RegisterTrainableParameter(_relativeBiasTable, PersistentTensorRole.Weights);
        }
        else
        {
            _relativeBiasTable = new Tensor<T>([0, 0]);
        }

        for (int i = 0; i < numLayers; i++)
        {
            var attention = new T5RelativeBiasAttentionLayer<T>(
                hiddenSize: hiddenSize,
                numHeads: numHeads,
                numBuckets: numBuckets,
                maxDistance: maxDistance,
                bidirectional: true,
                seed: seed.HasValue ? seed.Value + i + 1 : null,
                usesExternalPositionBias: shareRelativeBias);
            var block = new PreLNTransformerBlock<T>(
                hiddenSize: hiddenSize,
                ffnDim: hiddenSize * 4,
                attention: attention,
                ffnActivation: new AiDotNet.ActivationFunctions.GELUActivation<T>());
            _blocks.Add(block);
            RegisterSubLayer(block);
        }
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>The shared table, or an empty tensor when each block owns its own.</summary>
    public Tensor<T> GetRelativeBiasTable() => _relativeBiasTable;

    /// <summary>Whether the blocks share this stack's single table.</summary>
    public bool SharesRelativeBias => _shareRelativeBias;

    /// <summary>The stack's blocks, in forward order.</summary>
    public IReadOnlyList<PreLNTransformerBlock<T>> Blocks => _blocks;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        if (!_shareRelativeBias)
        {
            var y = input;
            foreach (var block in _blocks) y = block.Forward(y);
            return y;
        }

        int seqLen = input.Shape.Length >= 2 ? input.Shape[input.Shape.Length - 2] : 1;
        var bias = PositionBias(seqLen);
        var x = input;
        foreach (var block in _blocks)
        {
            var attention = (T5RelativeBiasAttentionLayer<T>)block.AttentionLayer;
            attention.SetExternalPositionBias(bias);
            try
            {
                x = block.Forward(x);
            }
            finally
            {
                attention.SetExternalPositionBias(null);
            }
        }

        return x;
    }

    /// <summary>The shared bias [numHeads, S, S], looked up through the tape so every block trains the table.</summary>
    private Tensor<T> PositionBias(int seqLen)
    {
        if (_cachedSeqLen != seqLen || _bucketIndices is null)
        {
            var idx = new Tensor<int>(new[] { seqLen, seqLen });
            for (int q = 0; q < seqLen; q++)
                for (int k = 0; k < seqLen; k++)
                    idx[q, k] = T5RelativeBiasAttentionLayer<T>.RelativePositionBucket(k - q, true, _numBuckets, _maxDistance);
            _bucketIndices = idx;
            _cachedSeqLen = seqLen;
        }

        var looked = Engine.TensorEmbeddingLookup<T, int>(_relativeBiasTable, _bucketIndices);
        return Engine.TensorPermute(looked, new[] { 2, 0, 1 });
    }

    /// <inheritdoc/>
    protected internal override ShapeRelationKind OutputShapeRelation => ShapeRelationKind.FeatureOnly;

    /// <inheritdoc/>
    protected override void OnFirstForward(Tensor<T> input)
    {
        ResolveShapes(new[] { -1, _hiddenSize }, new[] { -1, _hiddenSize });
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var block in _blocks) block.ResetState();
        _cachedSeqLen = -1;
        _bucketIndices = null;
    }

    private static Tensor<T> SampleNormal(Random random, int[] shape, double std)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            tensor[i] = numOps.FromDouble(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        return tensor;
    }
}
