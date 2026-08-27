// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// A stack of <see cref="RWKV7Block{T}"/> layers that threads the RWKV-7 value residual between them.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <remarks>
/// <para>
/// RWKV-7 (arXiv:2503.14456) blends every layer's value projection back toward the FIRST layer's:
/// <c>v = v + (v_first - v) * sigmoid(v0 + (x_v @ v1) @ v2)</c>. That is a cross-layer term, and a
/// layer whose contract is <c>Forward(Tensor) -&gt; Tensor</c> cannot express it — it sees only its
/// own input. The reference implementation solves this by having the PARENT own the loop:
/// </para>
/// <code>
///   v_first = torch.empty_like(x)
///   for block in self.blocks:
///       x, v_first = block(x, v_first)
/// </code>
/// <para>
/// This type is that parent. It owns its blocks and threads v_first as an ordinary local, so the
/// residual is a normal set of tape edges rather than hidden shared state — the gradient flows back
/// through the blend into the producing layer's value projection like any other data dependency.
/// </para>
/// <para>
/// This is also the established idiom here rather than a new invention: <c>DenseBlock</c> owns its
/// layers to give each one all previous feature maps, and <c>UNetDiscriminator</c> owns its encoder
/// and decoder to carry skip connections across. Twenty-one composite layers in this codebase follow
/// the same shape.
/// </para>
/// <para><b>For Beginners:</b> Most layers are a simple chain — each takes the previous one's output
/// and nothing else. RWKV-7 needs something extra: every layer wants to see what the FIRST layer
/// thought, not just the one below it. A plain chain has nowhere to put that, so instead one object
/// holds the whole run of layers and hands the first layer's answer along as it goes.</para>
/// </remarks>
// Stacks RWKV7 blocks and threads vFirst through them; each block is shape-preserving, so the stack is
// too. Same [Time, Features] convention as the rest of this folder.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class Rwkv7Stack<T> : LayerBase<T>, IShapeContract
{
    private readonly List<RWKV7Block<T>> _blocks;

    /// <summary>Construction state: the 'numLayers' the layer was built with.</summary>
    private readonly int _numLayers;

    /// <summary>Construction state: the 'sequenceLength' the layer was built with.</summary>
    private readonly int _sequenceLength;

    /// <summary>Construction state: the 'ffnMultiplier' the layer was built with.</summary>
    private readonly double _ffnMultiplier;

    /// <summary>Construction state: the 'globalIclrMultiplier' the layer was built with.</summary>
    private readonly double _globalIclrMultiplier;

    /// <summary>Construction state: the 'modelDimension' the layer was built with.</summary>
    private readonly int _modelDimension;

    /// <summary>Construction state: the 'numHeads' the layer was built with.</summary>
    private readonly int _numHeads;

    /// <summary>Creates a stack of RWKV-7 blocks sharing one value-residual chain.</summary>
    /// <param name="numLayers">Number of blocks. Must be at least one.</param>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">Model width.</param>
    /// <param name="numHeads">Attention heads; <paramref name="modelDimension"/> must divide by it.</param>
    /// <param name="ffnMultiplier">Channel-mix hidden-width multiplier.</param>
    /// <param name="globalIclrMultiplier">
    /// RWKV-7's global ICLR multiplier c in the state transition. Forwarded unchanged to every
    /// block, so the whole stack shares one setting.
    /// </param>
    public Rwkv7Stack(
        int numLayers,
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 4,
        double ffnMultiplier = 3.5,
        double globalIclrMultiplier = 1.0)
        : base([sequenceLength, modelDimension], [sequenceLength, modelDimension],
               (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _numHeads = numHeads;
        _modelDimension = modelDimension;
        _globalIclrMultiplier = globalIclrMultiplier;
        _ffnMultiplier = ffnMultiplier;
        _sequenceLength = sequenceLength;
        _numLayers = numLayers;
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), "An RWKV-7 stack needs at least one block.");

        _blocks = new List<RWKV7Block<T>>(numLayers);
        for (int i = 0; i < numLayers; i++)
        {
            var block = new RWKV7Block<T>(
                sequenceLength, modelDimension, numHeads, ffnMultiplier,
                globalIclrMultiplier: globalIclrMultiplier);
            _blocks.Add(block);
            // Equivalent to PyTorch's nn.Module child registration, and NOT optional: the tape
            // training step discovers parameters by walking GetSubLayers() recursively from the
            // network's top-level Layers list. These blocks are nested inside this composite rather
            // than siblings in that list, so without registering them the walk never reaches them
            // and all 21k block parameters silently sit at their initial values while the
            // surrounding dense layers train normally.
            RegisterSubLayer(block);
        }
    }

    /// <summary>The blocks in this stack, in execution order.</summary>
    /// <remarks>
    /// Exposed because callers used to reach the blocks via <c>Layers.OfType&lt;RWKV7Block&lt;T&gt;&gt;()</c>
    /// when they were siblings in a flat layer list. They are nested now, so that search finds nothing.
    /// </remarks>
    public IReadOnlyList<RWKV7Block<T>> Blocks => _blocks;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var current = input;
        // null marks "you are the first layer"; the first block answers with its own value
        // projection and every later block receives and passes it along.
        Tensor<T>? vFirst = null;
        foreach (var block in _blocks)
        {
            (current, vFirst) = block.ForwardWithValueResidual(current, vFirst);
        }
        return current;
    }

    // No Backward override: LayerBase dropped manual backward in favour of tape-based autodiff, and
    // Forward above is built from IEngine ops, so the gradient — including the value-residual blend
    // between blocks — is recorded and replayed by the tape.

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        foreach (var block in _blocks) block.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameterGradients()
        => new Vector<T>(_blocks.SelectMany(b => b.GetParameterGradients().ToArray()).ToArray());

    /// <inheritdoc />
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var block in _blocks) block.ClearGradients();
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var block in _blocks) block.ResetState();
    }
}
