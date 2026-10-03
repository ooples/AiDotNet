using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Activation normalization (Kingma &amp; Dhariwal 2018): a per-channel affine map <c>z = b + e^s ⊙ x</c> whose scale
/// and bias are initialized from the first batch so that its output has zero mean and unit variance per channel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Glow-TTS's decoder starts every flow block with one (Kim et al. 2020, §3.3, Fig. 8; reference
/// implementation <c>modules.ActNorm</c> with data-dependent initialization). The log-determinant is
/// <c>T · Σ s</c> per sequence. Whether the data-dependent initialization has run is part of the layer's persisted
/// state, so a restored model does not re-initialize its trained values.</para>
/// <para><b>For Beginners:</b> Rescales each channel to a standard range, learning the scale after setting it once from
/// real data.</para>
/// </remarks>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.ActivationNormalization)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, TestInputShape = "1, 4, 6", TestConstructorArgs = "4, true")]
[ElementWiseShape(Note = "Per-channel affine map; the shape is carried through.")]
[AutoParameters]
public partial class ActNormFlowLayer<T> : LayerBase<T>, IInvertibleFlowStep<T>
{
    private readonly int _channels;
    private bool _initialized;

    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _logScale;
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _bias;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer.</summary>
    /// <param name="channels">Channels of the sequence.</param>
    /// <param name="initialized">False to initialize from the first training batch (data-dependent initialization).</param>
    public ActNormFlowLayer([LayerState] int channels, [LayerState] bool initialized = false)
        : base(new[] { channels }, new[] { channels })
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        _channels = channels;
        _initialized = initialized;
        _logScale = new Tensor<T>(new[] { channels });
        _bias = new Tensor<T>(new[] { channels });
        RegisterTrainableParameter(_logScale, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_bias, PersistentTensorRole.NormalizationParams);
    }

    /// <summary>Whether the data-dependent initialization has run.</summary>
    public bool IsDataInitialized => _initialized;

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => Transform(input, false).Output;

    /// <inheritdoc />
    public (Tensor<T> Output, Tensor<T>? LogDeterminant) Transform(Tensor<T> input, bool reverse)
    {
        if (input.Rank != 3 || input.Shape[1] != _channels)
            throw new ArgumentException($"Expected [batch, {_channels}, time], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], time = input.Shape[2];
        if (!reverse && !_initialized && IsTrainingMode)
            InitializeFromData(input);

        var scale = Engine.TensorTile(Engine.Reshape(Engine.TensorExp(_logScale), new[] { 1, _channels, 1 }), new[] { batch, 1, time });
        var bias = Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, _channels, 1 }), new[] { batch, 1, time });
        if (reverse)
            return (Engine.TensorDivide(Engine.TensorSubtract(input, bias), scale), null);
        var output = Engine.TensorAdd(bias, Engine.TensorMultiply(scale, input));
        var logDet = Engine.TensorMultiplyScalar(Engine.ReduceSum(_logScale, new[] { 0 }, keepDims: false),
            NumOps.FromDouble(batch * time));
        return (output, logDet);
    }

    // Data-dependent initialization: per-channel mean m and variance v over batch and time; s = -0.5 log(max(v, 1e-6)),
    // b = -m e^{-s}, so the first batch comes out with zero mean and unit variance.
    private void InitializeFromData(Tensor<T> x)
    {
        int batch = x.Shape[0], time = x.Shape[2];
        double count = batch * time;
        for (int c = 0; c < _channels; c++)
        {
            double sum = 0, sumSq = 0;
            for (int b = 0; b < batch; b++)
                for (int t = 0; t < time; t++)
                {
                    double v = NumOps.ToDouble(x[b, c, t]);
                    sum += v;
                    sumSq += v * v;
                }
            double mean = sum / count, variance = sumSq / count - mean * mean;
            double logs = 0.5 * Math.Log(Math.Max(variance, 1e-6));
            _bias[c] = NumOps.FromDouble(-mean * Math.Exp(-logs));
            _logScale[c] = NumOps.FromDouble(-logs);
        }
        Engine.InvalidatePersistentTensor(_logScale);
        Engine.InvalidatePersistentTensor(_bias);
        _initialized = true;
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != 2 * _channels) return;
        for (int i = 0; i < _channels; i++)
        {
            _logScale[i] = NumOps.Subtract(_logScale[i], NumOps.Multiply(learningRate, gradients[i]));
            _bias[i] = NumOps.Subtract(_bias[i], NumOps.Multiply(learningRate, gradients[_channels + i]));
        }
        Engine.InvalidatePersistentTensor(_logScale);
        Engine.InvalidatePersistentTensor(_bias);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments, including whether the data-dependent initialization ran.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Initialized"] = _initialized.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
