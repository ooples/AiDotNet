using AiDotNet.Audio.Pitch;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// Base class for non-autoregressive acoustic models of the FastSpeech family: a phoneme encoder, a variance adaptor
/// (duration and length regulation, optionally pitch and energy), and a mel decoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The layer stack is <c>encoder layers, <see cref="VarianceAdaptorLayer{T}"/>, decoder layers</c>, registered with
/// <see cref="TtsModelBase{T}.AddEncoderDecoderLayers"/> so the adaptor sits at <c>Layers[EncoderLayerCount]</c>.
/// Inference is an ordinary sequential forward (the adaptor uses its predictions). Training is the papers'
/// teacher-forced objective: forced-alignment durations, and pitch and energy when the adaptor has those branches,
/// drive the adaptor, and the loss is the mel reconstruction error plus the mean squared error of each predictor
/// (FastSpeech, Ren et al. 2019 §3.3; FastSpeech 2, Ren et al. 2021 §2.3).
/// </para>
/// <para>A model supplies its layers and its paper's mel loss; the training and inference plumbing is shared.</para>
/// </remarks>
public abstract class VarianceAdaptorTtsModelBase<T> : TtsModelBase<T>
{
    /// <summary>Creates the model.</summary>
    protected VarianceAdaptorTtsModelBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction) { }

    /// <summary>The variance adaptor between the phoneme encoder and the mel decoder.</summary>
    public VarianceAdaptorLayer<T> VarianceAdaptor =>
        EncoderLayerCount < Layers.Count && Layers[EncoderLayerCount] is VarianceAdaptorLayer<T> adaptor
            ? adaptor
            : throw new InvalidOperationException(
                $"{GetType().Name} has no variance adaptor after its encoder; build it with AddEncoderDecoderLayers.");

    /// <inheritdoc />
    /// <remarks>Durations come from a forced alignment outside the recording (an attention teacher for FastSpeech,
    /// the Montreal Forced Aligner for FastSpeech 2).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Durations;

    /// <summary>Whether the paper scores the mel spectrogram with mean absolute error (true) or squared error.</summary>
    protected abstract bool UsesL1MelLoss { get; }

    /// <summary>The optimizer training steps use; null for the base default.</summary>
    protected abstract IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer { get; }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        ThrowIfTokenMelTrainingUnsupported();
    }

    private bool _adaptorModelDisposed;

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        _adaptorModelDisposed = true;
        base.Dispose(disposing);
    }

    private void ThrowIfDisposed()
    {
        if (_adaptorModelDisposed)
            throw new ObjectDisposedException(GetType().FullName ?? GetType().Name);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        var x = input;
        foreach (var layer in Layers)
            x = layer.Forward(x);
        return x;
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var (tokens, mel, objective) = BuildObjective(sample);
        return TrainWithCustomObjective(tokens, mel, objective, TrainingOptimizer);
    }

    /// <inheritdoc />
    /// <remarks>The teacher-forced objective that <see cref="TtsModelBase{T}.Train(TtsTrainingSample{T})"/> minimizes,
    /// evaluated with dropout off and without updating any weight.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var (tokens, mel, objective) = BuildObjective(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            return objective(tokens, mel)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// Builds the paper objective for one utterance: targets derived from the sample, and a function of the tokens and
    /// mel that runs the teacher-forced forward and returns the summed loss.
    /// </summary>
    private (Tensor<T> Tokens, Tensor<T> Mel, Func<Tensor<T>, Tensor<T>, Tensor<T>> Objective) BuildObjective(
        TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var durations = sample.Durations ?? throw new ArgumentException(
            $"{GetType().Name} trains on per-token durations from a forced alignment; set Durations.", nameof(sample));
        var adaptor = VarianceAdaptor;
        var targets = DeriveAcousticTargets(sample);
        int frames = targets.MelFrames;
        if (durations.Sum() != frames)
            throw new ArgumentException(
                $"The durations sum to {durations.Sum()} frames but the mel spectrogram has {frames}.", nameof(sample));

        double[]? continuousF0 = null;
        Tensor<T>? pitchTarget = null, statisticsTarget = null, energyTarget = null;
        if (adaptor.UsePitch)
        {
            var pitch = targets.Pitch ?? throw new ArgumentException(
                $"{GetType().Name} trains on frame pitch: supply the recording or Pitch.", nameof(sample));
            // FastSpeech 2 App. C.2: fill unvoiced frames, log, normalize per utterance, 10-scale CWT.
            var (_, contLogF0) = PitchWaveletTransform.ContinuousLogF0(pitch);
            double mean = contLogF0.Average();
            double std = Math.Sqrt(contLogF0.Select(v => (v - mean) * (v - mean)).Average());
            var normalized = contLogF0.Select(v => std > 0 ? (v - mean) / std : 0.0).ToArray();
            var spectrogram = PitchWaveletTransform.Forward(normalized);
            continuousF0 = contLogF0.Select(Math.Exp).ToArray();
            pitchTarget = new Tensor<T>(new[] { frames, PitchWaveletTransform.ScaleCount });
            for (int t = 0; t < frames; t++)
                for (int s = 0; s < PitchWaveletTransform.ScaleCount; s++)
                    pitchTarget[t, s] = NumOps.FromDouble(spectrogram[t, s]);
            statisticsTarget = ToTensor(new[] { mean, std });
        }
        if (adaptor.UseEnergy)
        {
            var energy = targets.Energy ?? throw new ArgumentException(
                $"{GetType().Name} trains on frame energy: supply the recording or Energy.", nameof(sample));
            energyTarget = ToTensor(energy);
        }

        var adaptorTargets = new VarianceTargets { Durations = durations, Pitch = continuousF0, Energy = targets.Energy };
        var logDurationTarget = ToTensor(durations.Select(d => Math.Log(d + 1.0)).ToArray());

        Tensor<T> Objective(Tensor<T> tokens, Tensor<T> mel)
        {
            var hidden = RunEncoder(tokens);
            var adapted = VarianceAdaptor.Adapt(hidden, adaptorTargets);
            var predictedMel = RunLayers(adapted.Expanded, EncoderLayerCount + 1, Layers.Count);

            var loss = UsesL1MelLoss ? MeanAbsoluteError(predictedMel, mel) : MeanSquaredError(predictedMel, mel);
            loss = Engine.TensorAdd(loss, MeanSquaredError(adapted.LogDuration, logDurationTarget));
            if (adapted.PitchSpectrogram is not null && adapted.PitchStatistics is not null)
            {
                loss = Engine.TensorAdd(loss, MeanSquaredError(adapted.PitchSpectrogram, pitchTarget!));
                loss = Engine.TensorAdd(loss, MeanSquaredError(adapted.PitchStatistics, statisticsTarget!));
            }
            if (adapted.Energy is not null)
                loss = Engine.TensorAdd(loss, MeanSquaredError(adapted.Energy, energyTarget!));
            return loss;
        }

        return (sample.Tokens, targets.Mel, Objective);
    }

    private Tensor<T> ToTensor(double[] values)
    {
        var t = new Tensor<T>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++) t[i] = NumOps.FromDouble(values[i]);
        return t;
    }

    private Tensor<T> MeanAbsoluteError(Tensor<T> prediction, Tensor<T> target)
    {
        var diff = Engine.TensorAbs(Engine.TensorSubtract(prediction, Engine.Reshape(target, prediction._shape)));
        return Engine.ReduceMean(diff, Enumerable.Range(0, diff.Rank).ToArray(), keepDims: false);
    }

    private Tensor<T> MeanSquaredError(Tensor<T> prediction, Tensor<T> target)
    {
        var diff = Engine.TensorSubtract(prediction, Engine.Reshape(target, prediction._shape));
        var sq = Engine.TensorMultiply(diff, diff);
        return Engine.ReduceMean(sq, Enumerable.Range(0, sq.Rank).ToArray(), keepDims: false);
    }
}
