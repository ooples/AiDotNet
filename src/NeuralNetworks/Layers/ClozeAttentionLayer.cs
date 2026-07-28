using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Bidirectional Cloze Network (BCN) attention per Fang et al., CVPR 2021,
/// "Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text
/// Recognition" (arXiv:2103.06495).
/// </summary>
/// <remarks>
/// <para>This is ABINet's <b>Bidirectional</b> principle. Each position attends to every OTHER
/// position — both left and right — but is forbidden from attending to itself:</para>
/// <code>
///   M[i, j] = 0     when i != j
///   M[i, j] = -inf  when i == j
///   attention = softmax(QK^T / sqrt(d) + M) . V
/// </code>
/// <para>Blocking the diagonal is what makes it a <i>cloze</i>: the representation at position
/// <c>i</c> is built purely from its surrounding context, so predicting the character there
/// cannot trivially copy the character itself. That is precisely the information leak an
/// unmasked bidirectional attention would introduce, and why this cannot be expressed with the
/// causal (triangular) mask an ordinary attention layer offers — a causal mask would also remove
/// all right-hand context, collapsing the model back to unidirectional.</para>
/// <para><b>Gradient tracking:</b> the mask is a constant tensor added to the scores, and every
/// other step is an <c>IEngine</c> op, so the tape records the whole block.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 3, Cost = ComputeCost.Medium, TestInputShape = "1, 4, 8", TestConstructorArgs = "8")]
public class ClozeAttentionLayer<T> : LayerBase<T>
{
    private readonly int _modelDim;

    private readonly DenseLayer<T> _query;
    private readonly DenseLayer<T> _key;
    private readonly DenseLayer<T> _value;
    private readonly DenseLayer<T> _output;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount =>
        _query.ParameterCount + _key.ParameterCount + _value.ParameterCount + _output.ParameterCount;

    /// <summary>Initializes a new bidirectional cloze attention block.</summary>
    /// <param name="modelDim">Model width; input and output are both this wide.</param>
    public ClozeAttentionLayer(int modelDim)
        : base(new[] { -1, -1, modelDim }, new[] { -1, -1, modelDim })
    {
        if (modelDim <= 0) throw new ArgumentOutOfRangeException(nameof(modelDim));

        _modelDim = modelDim;
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();

        _query = new DenseLayer<T>(modelDim, identity);
        _key = new DenseLayer<T>(modelDim, identity);
        _value = new DenseLayer<T>(modelDim, identity);
        _output = new DenseLayer<T>(modelDim, identity);

        RegisterSubLayer(_query);
        RegisterSubLayer(_key);
        RegisterSubLayer(_value);
        RegisterSubLayer(_output);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        bool unbatched = input.Shape.Length == 2;
        if (unbatched)
            input = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]]);

        if (input.Shape.Length != 3)
            throw new ArgumentException($"ClozeAttentionLayer expects rank-2 [S, D] or rank-3 [B, S, D], got rank {input.Shape.Length}.", nameof(input));

        int B = input.Shape[0];
        int S = input.Shape[1];
        int D = input.Shape[2];

        if (D != _modelDim)
            throw new ArgumentException($"ClozeAttentionLayer was configured for modelDim={_modelDim} but got D={D}.", nameof(input));

        var q = _query.Forward(input);
        var k = _key.Forward(input);
        var v = _value.Forward(input);

        // scores = Q . K^T / sqrt(d)
        var kT = Engine.TensorPermute(k, new[] { 0, 2, 1 });
        var scores = Engine.TensorBatchMatMul<T>(q, kT);
        scores = Engine.TensorDivideScalar(scores, NumOps.FromDouble(Math.Sqrt(D)));

        // Cloze mask: a large negative on the diagonal only, so softmax drives self-attention to
        // zero while every other position stays available in both directions. Built as a
        // constant, so it contributes no gradient of its own.
        var mask = new Tensor<T>(new[] { B, S, S });
        T blocked = NumOps.FromDouble(-1e9);
        for (int b = 0; b < B; b++)
            for (int i = 0; i < S; i++)
                mask[b, i, i] = blocked;

        scores = Engine.TensorAdd(scores, mask);

        var weights = Engine.TensorSoftmax(scores, axis: scores.Shape.Length - 1);
        var attended = Engine.TensorBatchMatMul<T>(weights, v);
        var result = _output.Forward(attended);

        return unbatched ? Engine.Reshape(result, [S, D]) : result;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var parts = new[]
        {
            _query.GetParameters(), _key.GetParameters(),
            _value.GetParameters(), _output.GetParameters()
        };

        int total = 0;
        foreach (var p in parts) total += p.Length;

        var flat = new Vector<T>(total);
        int at = 0;
        foreach (var p in parts)
            for (int i = 0; i < p.Length; i++) flat[at++] = p[i];

        return flat;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var targets = new[] { _query, _key, _value, _output };
        var sizes = targets.Select(t => t.GetParameters().Length).ToArray();

        // Sub-layers allocate lazily on first Forward; materialize before validating so a
        // restore into a fresh instance lines up with the source layout.
        if (sizes.Sum() == 0 && parameters.Length > 0)
        {
            _ = Forward(new Tensor<T>(new[] { 1, 1, _modelDim }));
            sizes = targets.Select(t => t.GetParameters().Length).ToArray();
        }

        if (parameters.Length != sizes.Sum())
            throw new ArgumentException($"Expected {sizes.Sum()} parameters, got {parameters.Length}.", nameof(parameters));

        int at = 0;
        for (int t = 0; t < targets.Length; t++)
        {
            var slice = new Vector<T>(sizes[t]);
            for (int i = 0; i < sizes[t]; i++) slice[i] = parameters[at++];
            targets[t].SetParameters(slice);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Explicitly includes the projections' tensors: <c>LayerBase</c> does not recurse into
    /// registered sub-layers, and a composite that omits them reports an empty trainable set
    /// while still advertising a parameter count, which corrupts training silently.
    /// </remarks>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters()
    {
        var result = new List<Tensor<T>>();
        result.AddRange(_query.GetTrainableParameters());
        result.AddRange(_key.GetTrainableParameters());
        result.AddRange(_value.GetTrainableParameters());
        result.AddRange(_output.GetTrainableParameters());
        return result;
    }

    /// <inheritdoc/>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        var targets = new[] { _query, _key, _value, _output };
        var counts = targets.Select(t => t.GetTrainableParameters().Count).ToArray();

        if (parameters.Count != counts.Sum())
            throw new ArgumentException($"Expected {counts.Sum()} trainable tensors, got {parameters.Count}.", nameof(parameters));

        int at = 0;
        for (int t = 0; t < targets.Length; t++)
        {
            targets[t].SetTrainableParameters(parameters.Skip(at).Take(counts[t]).ToList());
            at += counts[t];
        }
    }

    /// <inheritdoc/>
    /// <remarks>Tape-based autodiff drives the update; no manual gradient step here.</remarks>
    public override void UpdateParameters(T learningRate)
    {
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _query.ResetState();
        _key.ResetState();
        _value.ResetState();
        _output.ResetState();
    }
}
