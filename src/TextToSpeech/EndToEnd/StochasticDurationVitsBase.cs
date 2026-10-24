using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// A VITS-family model whose durations come from VITS's stochastic duration predictor (Kim et al. 2021 §2.2.2), trained
/// with the rest of the generator on the alignment's durations and sampled through its reversed flows at synthesis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>VITS, Piper and YourTTS share this duration model; the predictor reads the stop-gradient text encoding and,
/// when present, the stop-gradient speaker and language embeddings (reference <c>StochasticDurationPredictor</c>, Coqui
/// <c>stochastic_duration_predictor.py</c>). Its convolutions are <see cref="VitsModelOptions"/>' hidden width wide.</para>
/// <para><b>For Beginners:</b> This model learns a distribution of how long each sound lasts, so the same text can be
/// spoken with natural variations in rhythm.</para>
/// </remarks>
public abstract partial class StochasticDurationVitsBase<T> : VitsTtsModelBase<T>
{
    private VitsStochasticDurationPredictor<T>? _durationPredictor;

    /// <summary>Creates a native (trainable) model.</summary>
    protected StochasticDurationVitsBase(NeuralNetworkArchitecture<T> architecture, VITSOptions options,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
        : base(architecture, options, optimizer)
    {
    }

    /// <summary>Creates a model that runs an exported ONNX graph.</summary>
    protected StochasticDurationVitsBase(NeuralNetworkArchitecture<T> architecture, string modelPath, VITSOptions options)
        : base(architecture, modelPath, options)
    {
    }

    private VITSOptions DurationOptions => (VITSOptions)VitsOptions;

    /// <inheritdoc />
    protected override IEnumerable<LayerBase<T>> CreateDurationModel(int hidden, int speakerChannels, int languageChannels)
    {
        var o = DurationOptions;
        _durationPredictor = new VitsStochasticDurationPredictor<T>(Engine, hidden, o.HiddenDim, o.DurationPredictorKernelSize,
            o.DurationPredictorDropout, o.DurationPredictorFlows, speakerChannels, languageChannels);
        return _durationPredictor.Layers;
    }

    /// <inheritdoc />
    protected override IEnumerable<LayerBase<T>> JointDurationLayers => _durationPredictor?.Layers ?? (IEnumerable<LayerBase<T>>)Array.Empty<LayerBase<T>>();

    /// <inheritdoc />
    /// <remarks>The stochastic duration predictor's negative lower bound on log p(d | c), per token (reference
    /// <c>l_length / sum(x_mask)</c>).</remarks>
    protected override Tensor<T>? JointDurationLoss(Tensor<T> hidden, Tensor<T>? speaker, Tensor<T>? language, int[] durations, Random random)
    {
        int tokens = durations.Length;
        var w = new Tensor<T>(new[] { 1, 1, tokens });
        for (int i = 0; i < tokens; i++) w[0, 0, i] = NumOps.FromDouble(durations[i]);
        return Engine.TensorMultiplyScalar(_durationPredictor!.NegativeLogLikelihood(hidden, speaker, language, w, random), NumOps.FromDouble(1.0 / tokens));
    }

    /// <inheritdoc />
    protected override double[] PredictDurations(Tensor<T> hidden, Tensor<T>? speaker, Tensor<T>? language, Random random)
    {
        var logDurations = _durationPredictor!.SampleLogDurations(hidden, speaker, language, random, DurationOptions.DurationNoiseScale);
        var durations = new double[logDurations.Length];
        for (int i = 0; i < durations.Length; i++) durations[i] = Math.Exp(NumOps.ToDouble(logDurations[i]));
        return durations;
    }
}
