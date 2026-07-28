using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A Branchformer encoder block per Peng et al., ICML 2022, "Branchformer: Parallel
/// MLP-Attention Architectures to Capture Local and Global Context for Speech Recognition and
/// Understanding" (arXiv:2207.02971).
/// </summary>
/// <remarks>
/// <para>Two branches run in PARALLEL over the same input: self-attention captures long-range
/// dependencies, and a convolutional gating MLP (cgMLP) captures local ones. Their outputs are
/// concatenated and projected back to the model width.</para>
/// <para>This is the distinction from Conformer, which composes attention and convolution
/// SEQUENTIALLY inside one branch. The parallel arrangement is what lets the model weigh the two
/// context types against each other per layer, and it is the paper's entire contribution — a
/// Branchformer built out of Conformer blocks is just a Conformer.</para>
/// <para><b>cgMLP and the CSGU:</b> a channel projection with GeLU expands to the hidden width,
/// then the Convolutional Spatial Gating Unit splits that in half, applies layer norm and a
/// depth-wise convolution to one half, and multiplies the two halves element-wise. The paper
/// notes the gating is LINEAR — no extra nonlinearity between the projection and the
/// multiplication — so the depth-wise path acts as a learned spatial gate rather than another
/// activation.</para>
/// <para><b>Why a composite layer rather than loose layers:</b> a parallel structure flattened
/// into a sequential layer list would be executed as attention-then-cgMLP by any default forward
/// pass, silently turning it back into a sequential block. Keeping both branches inside one
/// layer makes that impossible.</para>
/// <para><b>Gradient tracking:</b> every operation goes through <c>IEngine</c>, so the tape
/// records both branches and the merge without a manual backward.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 8, 16", TestConstructorArgs = "16, 4, 64, 31")]
public class BranchformerBlock<T> : LayerBase<T>
{
    private readonly int _modelDim;
    private readonly int _numHeads;
    private readonly int _cgmlpHiddenDim;
    private readonly int _kernelSize;

    private readonly MultiHeadAttentionLayer<T> _attention;
    private readonly LayerNormalizationLayer<T> _attentionNorm;

    private readonly LayerNormalizationLayer<T> _cgmlpNorm;
    private readonly DenseLayer<T> _cgmlpExpand;
    private readonly LayerNormalizationLayer<T> _csguNorm;
    private readonly DepthwiseConv1DLayer<T> _csguConv;
    private readonly DenseLayer<T> _cgmlpProject;

    private readonly DenseLayer<T> _merge;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount =>
        _attention.ParameterCount + _attentionNorm.ParameterCount +
        _cgmlpNorm.ParameterCount + _cgmlpExpand.ParameterCount +
        _csguNorm.ParameterCount + _csguConv.ParameterCount +
        _cgmlpProject.ParameterCount + _merge.ParameterCount;

    /// <summary>Initializes a new Branchformer block.</summary>
    /// <param name="modelDim">Model width; input and output are both this wide.</param>
    /// <param name="numHeads">Attention heads for the global branch.</param>
    /// <param name="cgmlpHiddenDim">
    /// Hidden width of the cgMLP branch before the CSGU halves it. The paper uses 2048-3072
    /// against a 256-512 model width.
    /// </param>
    /// <param name="kernelSize">Depth-wise convolution kernel in the CSGU. The paper uses 31.</param>
    public BranchformerBlock(int modelDim, int numHeads, int cgmlpHiddenDim, int kernelSize = 31)
        : base(new[] { -1, -1, modelDim }, new[] { -1, -1, modelDim })
    {
        if (modelDim <= 0) throw new ArgumentOutOfRangeException(nameof(modelDim));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (cgmlpHiddenDim <= 1) throw new ArgumentOutOfRangeException(nameof(cgmlpHiddenDim), "cgMLP hidden width must exceed 1 so the CSGU can split it.");
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));

        _modelDim = modelDim;
        _cgmlpHiddenDim = cgmlpHiddenDim;
        _kernelSize = kernelSize;

        (int heads, _) = ChooseHeads(modelDim, numHeads);
        _numHeads = heads;

        var gelu = (IActivationFunction<T>)new GELUActivation<T>();
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();

        _attention = new MultiHeadAttentionLayer<T>(_numHeads, modelDim / _numHeads, identity);
        _attentionNorm = new LayerNormalizationLayer<T>();

        _cgmlpNorm = new LayerNormalizationLayer<T>();
        _cgmlpExpand = new DenseLayer<T>(cgmlpHiddenDim, gelu);
        _csguNorm = new LayerNormalizationLayer<T>();
        _csguConv = new DepthwiseConv1DLayer<T>(cgmlpHiddenDim / 2, kernelSize);
        _cgmlpProject = new DenseLayer<T>(modelDim, identity);

        // Concatenation merge (the paper's default): [attention | cgMLP] projected back to width.
        _merge = new DenseLayer<T>(modelDim, identity);

        RegisterSubLayer(_attention);
        RegisterSubLayer(_attentionNorm);
        RegisterSubLayer(_cgmlpNorm);
        RegisterSubLayer(_cgmlpExpand);
        RegisterSubLayer(_csguNorm);
        RegisterSubLayer(_csguConv);
        RegisterSubLayer(_cgmlpProject);
        RegisterSubLayer(_merge);
    }

    private static (int heads, int headDim) ChooseHeads(int modelDim, int requested)
    {
        // Snap to a divisor so headDim * heads reconstructs modelDim exactly.
        for (int h = Math.Min(requested, modelDim); h >= 1; h--)
        {
            if (modelDim % h == 0) return (h, modelDim / h);
        }

        return (1, modelDim);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        bool unbatched = input.Shape.Length == 2;
        if (unbatched)
            input = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]]);

        if (input.Shape.Length != 3)
            throw new ArgumentException($"BranchformerBlock expects rank-2 [S, D] or rank-3 [B, S, D], got rank {input.Shape.Length}.", nameof(input));

        int B = input.Shape[0];
        int S = input.Shape[1];
        int D = input.Shape[2];

        if (D != _modelDim)
            throw new ArgumentException($"BranchformerBlock was configured for modelDim={_modelDim} but got D={D}.", nameof(input));

        // Global branch.
        var attentionOut = _attention.Forward(_attentionNorm.Forward(input));

        // Local branch: cgMLP with the Convolutional Spatial Gating Unit.
        var cgmlpOut = ForwardCgMlp(input, B, S);

        // Merge: concatenate the two branches and project back to the model width, then residual.
        var merged = _merge.Forward(Engine.TensorConcatenate(new[] { attentionOut, cgmlpOut }, axis: 2));
        var result = Engine.TensorAdd(merged, input);

        return unbatched ? Engine.Reshape(result, [S, D]) : result;
    }

    /// <summary>
    /// The cgMLP branch: expand with GeLU, gate through the CSGU, project back.
    /// </summary>
    private Tensor<T> ForwardCgMlp(Tensor<T> input, int B, int S)
    {
        int half = _cgmlpHiddenDim / 2;

        var x = _cgmlpExpand.Forward(_cgmlpNorm.Forward(input));   // [B, S, H]

        // CSGU: split the hidden width in half; one half is normalized and passed through a
        // depth-wise convolution, then multiplied element-wise into the other. The paper applies
        // NO activation between the convolution and the multiplication — the gating is linear.
        var gateInput = Engine.TensorNarrow(x, dim: 2, start: 0, length: half);
        var gateSignal = Engine.TensorNarrow(x, dim: 2, start: half, length: half);

        var normed = _csguNorm.Forward(gateSignal);

        // Depth-wise convolution runs channels-first.
        var chFirst = Engine.TensorPermute(normed, new[] { 0, 2, 1 });
        var convolved = _csguConv.Forward(chFirst);
        var gate = Engine.TensorPermute(convolved, new[] { 0, 2, 1 });   // [B, S, H/2]

        var gated = Engine.TensorMultiply(gateInput, gate);

        return _cgmlpProject.Forward(gated);                             // [B, S, D]
    }

    /// <summary>
    /// Materializes the lazily-allocated children from this block's known geometry, without
    /// executing them.
    /// </summary>
    private void ResolveChildShapes()
    {
        int half = _cgmlpHiddenDim / 2;

        if (!_attentionNorm.IsShapeResolved) _attentionNorm.ResolveFromShape(new[] { 1, 1, _modelDim });
        if (!_attention.IsShapeResolved) _attention.ResolveFromShape(new[] { 1, 1, _modelDim });
        if (!_cgmlpNorm.IsShapeResolved) _cgmlpNorm.ResolveFromShape(new[] { 1, 1, _modelDim });
        if (!_cgmlpExpand.IsShapeResolved) _cgmlpExpand.ResolveFromShape(new[] { 1, 1, _modelDim });
        if (!_csguNorm.IsShapeResolved) _csguNorm.ResolveFromShape(new[] { 1, 1, half });
        if (!_csguConv.IsShapeResolved) _csguConv.ResolveFromShape(new[] { 1, half, 1 });
        if (!_cgmlpProject.IsShapeResolved) _cgmlpProject.ResolveFromShape(new[] { 1, 1, half });
        if (!_merge.IsShapeResolved) _merge.ResolveFromShape(new[] { 1, 1, 2 * _modelDim });
    }

    private LayerBase<T>[] Children => new LayerBase<T>[]
    {
        _attention, _attentionNorm, _cgmlpNorm, _cgmlpExpand,
        _csguNorm, _csguConv, _cgmlpProject, _merge
    };

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var parts = Children.Select(c => c.GetParameters()).ToArray();
        int total = parts.Sum(p => p.Length);

        var flat = new Vector<T>(total);
        int at = 0;
        foreach (var p in parts)
            for (int i = 0; i < p.Length; i++) flat[at++] = p[i];

        return flat;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        ResolveChildShapes();

        var children = Children;
        var sizes = children.Select(c => c.GetParameters().Length).ToArray();

        if (parameters.Length != sizes.Sum())
            throw new ArgumentException($"Expected {sizes.Sum()} parameters, got {parameters.Length}.", nameof(parameters));

        int at = 0;
        for (int c = 0; c < children.Length; c++)
        {
            var slice = new Vector<T>(sizes[c]);
            for (int i = 0; i < sizes[c]; i++) slice[i] = parameters[at++];
            children[c].SetParameters(slice);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Enumerates the children explicitly, since <c>LayerBase</c> does not recurse into
    /// registered sub-layers.
    /// </remarks>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters()
    {
        var result = new List<Tensor<T>>();
        foreach (var c in Children) result.AddRange(c.GetTrainableParameters());
        return result;
    }

    /// <inheritdoc/>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        var children = Children;
        var counts = children.Select(c => c.GetTrainableParameters().Count).ToArray();

        if (parameters.Count != counts.Sum())
            throw new ArgumentException($"Expected {counts.Sum()} trainable tensors, got {parameters.Count}.", nameof(parameters));

        int at = 0;
        for (int c = 0; c < children.Length; c++)
        {
            children[c].SetTrainableParameters(parameters.Skip(at).Take(counts[c]).ToList());
            at += counts[c];
        }
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["ModelDim"] = _modelDim.ToString(inv);
        metadata["NumHeads"] = _numHeads.ToString(inv);
        metadata["CgmlpHiddenDim"] = _cgmlpHiddenDim.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        return metadata;
    }

    /// <inheritdoc/>
    /// <remarks>Tape-based autodiff drives the update; no manual gradient step here.</remarks>
    public override void UpdateParameters(T learningRate)
    {
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var c in Children) c.ResetState();
    }
}
