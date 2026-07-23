using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Prepends a learnable <c>[CLS]</c> token to a sequence-of-embeddings
/// input, as introduced by BERT (Devlin et al. 2018 §3.1) and adopted by
/// ViT / AST (Dosovitskiy et al. 2020 §3.1; Gong et al. 2021 §2.2). The
/// CLS token starts as a learnable parameter <c>[1, embedDim]</c>; the
/// layer broadcasts it across the batch and concatenates it at sequence
/// position 0 so the transformer's first output position becomes the
/// classification representation.
/// </summary>
/// <typeparam name="T">Numeric type for tensor data.</typeparam>
/// <remarks>
/// <para>
/// Pairs with <see cref="SequenceTokenSliceLayer{T}"/> using
/// <see cref="SequenceTokenSliceLayer{T}.Position.First"/> after the
/// transformer stack to extract the trained classification embedding —
/// the canonical AST / ViT classification head.
/// </para>
/// <para><b>For Beginners:</b> Most transformer classifiers prepend a
/// special learnable token to the input sequence; the network learns to
/// use that one token as a "summary slot" for the whole sequence. After
/// the transformer runs, you read just that one position to get the
/// classification feature — no mean-pooling required, and gradient flow
/// during training teaches the CLS token to aggregate task-relevant
/// information from the rest of the sequence.</para>
/// </remarks>
public class PrependCLSTokenLayer<T> : LayerBase<T>
{
    private readonly int _embedDim;

    // Trainable CLS token — shape [1, embedDim]. Held by reference so the
    // gradient tape can track parameter identity.
    private readonly Tensor<T> _cls;

    /// <summary>Creates a CLS-token prepender for embedDim-wide inputs.</summary>
    /// <param name="embedDim">Embedding dimension (must match the input's last axis).</param>
    /// <param name="initScale">Gaussian init std. ViT / BERT both use 0.02.</param>
    /// <param name="seed">Optional RNG seed for reproducibility.</param>
    public PrependCLSTokenLayer(int embedDim, double initScale = 0.02, int? seed = null)
        : base(
            inputShape: [-1, -1, -1],
            outputShape: [-1, -1, -1],
            scalarActivation: new IdentityActivation<T>())
    {
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        _embedDim = embedDim;
        _cls = new Tensor<T>(new[] { 1, embedDim });

        var rng = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        for (int i = 0; i < embedDim; i++)
            _cls[0, i] = NumOps.FromDouble(rng.NextGaussian() * initScale);
    }

    /// <inheritdoc/>
    public override long ParameterCount => _embedDim;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters() => new[] { _cls };

    /// <inheritdoc/>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        if (parameters.Count != 1)
            throw new ArgumentException("Expected exactly 1 parameter tensor (CLS).", nameof(parameters));
        var src = parameters[0];
        if (src.Length != _cls.Length)
            throw new ArgumentException(
                $"CLS shape mismatch: source length={src.Length}, expected {_cls.Length}.");
        for (int i = 0; i < src.Length; i++) _cls[i] = src[i];
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Shape.Length != 3)
            throw new ArgumentException(
                $"PrependCLSTokenLayer expects rank-3 [batch, seq, embedDim]; got rank {input.Shape.Length}.",
                nameof(input));
        if (input.Shape[2] != _embedDim)
            throw new ArgumentException(
                $"PrependCLSTokenLayer embedDim mismatch: layer={_embedDim}, input[2]={input.Shape[2]}.",
                nameof(input));

        int batch = input.Shape[0];
        int seq = input.Shape[1];

        // Tile the [1, embedDim] CLS to [batch, 1, embedDim], then concat
        // along axis 1 (sequence) with input.
        var clsRow = Engine.Reshape(_cls, new[] { 1, 1, _embedDim });
        var clsTiled = Engine.TensorTile(clsRow, new[] { batch, 1, 1 });
        return Engine.TensorConcatenate(new[] { clsTiled, input }, axis: 1);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var p = new Vector<T>(_embedDim);
        for (int i = 0; i < _embedDim; i++) p[i] = _cls[0, i];
        return p;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _embedDim)
            throw new ArgumentException($"Expected {_embedDim} params, got {parameters.Length}.");
        for (int i = 0; i < _embedDim; i++) _cls[0, i] = parameters[i];
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (ParameterGradients is null) return;
        if (ParameterGradients.Length != ParameterCount)
            throw new InvalidOperationException(
                $"PrependCLSTokenLayer.UpdateParameters: gradient buffer length " +
                $"{ParameterGradients.Length} does not match ParameterCount {ParameterCount}.");
        for (int i = 0; i < _embedDim; i++)
            _cls[0, i] = NumOps.Subtract(_cls[0, i],
                NumOps.Multiply(learningRate, ParameterGradients[i]));
    }

    /// <inheritdoc/>
    public override void ResetState() { }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["EmbedDim"] = _embedDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
