using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Adds sinusoidal positional encodings scaled by a trainable weight: <c>x_i + α · PE(i)</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Transformer TTS (Li et al. 2019, §3.2, Eq. 8): text embeddings and mel spectrograms live at different scales, so fixed
/// positional embeddings "may impose heavy constraints on both the encoder and decoder pre-nets"; the triangle positional
/// embeddings of Vaswani et al. (<c>PE(pos, 2i) = sin(pos / 10000^(2i/d))</c>, <c>PE(pos, 2i+1) = cos(...)</c>) are added
/// with a trainable weight α, one for the encoder and one for the decoder. α starts at 1.
/// </para>
/// <para>Input and output are <c>[time, d]</c> or <c>[batch, time, d]</c>.</para>
/// <para><b>For Beginners:</b> Tells each position where it is in the sequence, with a learned volume knob for how loudly.</para>
/// </remarks>
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 5, 8", TestConstructorArgs = "8")]
[ElementWiseShape(Note = "Adds a positional term; every dimension is carried through.")]
[AutoParameters]
public sealed partial class ScaledPositionalEncodingLayer<T> : LayerBase<T>
{
    private readonly int _dimension;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _alpha;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer for features of width <paramref name="dimension"/>.</summary>
    public ScaledPositionalEncodingLayer([LayerState] int dimension)
        : base(new[] { dimension }, new[] { dimension })
    {
        if (dimension <= 0) throw new ArgumentOutOfRangeException(nameof(dimension));
        _dimension = dimension;
        _alpha = Tensor<T>.CreateDefault(new[] { 1 }, NumOps.One);
        RegisterTrainableParameter(_alpha, PersistentTensorRole.Weights);
    }

    /// <summary>The current scale α.</summary>
    public T Alpha => _alpha[0];

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank is not (2 or 3) || input.Shape[input.Rank - 1] != _dimension)
            throw new ArgumentException(
                $"Expected [time, {_dimension}] or [batch, time, {_dimension}], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int time = input.Shape[input.Rank - 2];
        var encoding = new Tensor<T>(new[] { time * _dimension });
        for (int pos = 0; pos < time; pos++)
            for (int i = 0; i < _dimension; i++)
            {
                double angle = pos / Math.Pow(10000.0, 2.0 * (i / 2) / _dimension);
                encoding[pos * _dimension + i] = NumOps.FromDouble(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
            }
        var scaled = Engine.TensorMultiply(encoding, Engine.TensorTile(_alpha, new[] { time * _dimension }));
        var shaped = Engine.Reshape(scaled, new[] { time, _dimension });
        if (input.Rank == 3)
            shaped = Engine.TensorTile(Engine.Reshape(shaped, new[] { 1, time, _dimension }), new[] { input.Shape[0], 1, 1 });
        return Engine.TensorAdd(input, shaped);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != 1) return;
        _alpha[0] = NumOps.Subtract(_alpha[0], NumOps.Multiply(learningRate, gradients[0]));
        Engine.InvalidatePersistentTensor(_alpha);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor argument.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Dimension"] = _dimension.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
