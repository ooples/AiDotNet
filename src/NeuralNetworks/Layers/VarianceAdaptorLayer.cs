using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Audio.Pitch;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The variance adaptor of FastSpeech 2: duration prediction and length regulation, then pitch and energy
/// prediction, each added to the expanded sequence as a quantized embedding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Ren et al. 2021, §2.3 and Fig. 1b. In order:
/// </para>
/// <list type="number">
/// <item>The duration predictor reads the phoneme hidden sequence and predicts log(duration + 1) per phoneme.</item>
/// <item>The length regulator (FastSpeech, Ren et al. 2019, §3.2) repeats each phoneme's hidden state for its
/// duration in frames: ground-truth durations in training, rounded predictions at inference.</item>
/// <item>The pitch predictor reads the frame sequence and predicts the 10-scale CWT pitch spectrogram, plus the
/// utterance's log-F0 mean and deviation from the time-averaged convolution features (App. C.2). The F0 contour
/// (ground truth in training, recomposed from the prediction at inference) is quantized into 256 log-spaced bins
/// and its embedding added.</item>
/// <item>The energy predictor reads the result and predicts frame energy; the energy (ground truth in training,
/// predicted at inference) is quantized into 256 uniform bins and its embedding added.</item>
/// </list>
/// <para>
/// Bin ranges, which the paper leaves to the implementation: pitch bins span WORLD DIO's search range (71 to 800 Hz
/// by default) and energy bins span the range the STFT energy can take for audio in [-1, 1]
/// (<see cref="MaxStftEnergy"/>), so every value the extractors produce has a bin. Reference implementations fit
/// both ranges from the training set instead; <see cref="SetEnergyRange"/> does that for energy.
/// </para>
/// <para>As a layer, the forward pass runs the inference path, so a model whose layer stack is
/// encoder → adaptor → decoder synthesizes with an ordinary sequential forward. Training uses
/// <see cref="Adapt"/>, which takes the ground-truth targets and returns the predictions for the losses.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, Cost = ComputeCost.Medium, TestInputShape = "1, 5, 16", TestConstructorArgs = "16, 16, 3, 0.0, 256, 71.0, 800.0, 256, 0.0, 100.0, true, true")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class VarianceAdaptorLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _hiddenSize;
    private readonly int _filterSize;
    private readonly int _kernelSize;
    private readonly double _dropoutRate;
    private readonly int _pitchBins;
    private readonly double _pitchMinHz;
    private readonly double _pitchMaxHz;
    private readonly int _energyBins;
    private double _energyMin;
    private double _energyMax;
    private double[] _pitchBoundaries;
    private double[] _energyBoundaries;
    private readonly bool _usePitch;
    private readonly bool _useEnergy;

    [SubLayerInput("1, 1, _hiddenSize")]
    private readonly VariancePredictorLayer<T> _durationPredictor;
    // The pitch and energy branches exist only when enabled, so a model without them (FastSpeech) carries no
    // parameters that never receive a gradient.
    [SubLayerInput("1, 1, _hiddenSize")]
    private readonly VariancePredictorLayer<T>? _pitchPredictor;
    [SubLayerInput("1, _filterSize")]
    private readonly DenseLayer<T>? _pitchStatistics;
    private readonly EmbeddingLayer<T>? _pitchEmbedding;
    [SubLayerInput("1, 1, _hiddenSize")]
    private readonly VariancePredictorLayer<T>? _energyPredictor;
    private readonly EmbeddingLayer<T>? _energyEmbedding;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates a variance adaptor (defaults: FastSpeech 2, App. A).
    /// </summary>
    /// <param name="hiddenSize">Width of the phoneme hidden sequence (256).</param>
    /// <param name="filterSize">Convolution channels of each variance predictor (256).</param>
    /// <param name="kernelSize">Convolution kernel of each variance predictor (3).</param>
    /// <param name="dropoutRate">Dropout of each variance predictor (0.5).</param>
    /// <param name="pitchBins">Number of log-spaced pitch bins (256).</param>
    /// <param name="pitchMinHz">Lowest pitch bin edge, in Hz.</param>
    /// <param name="pitchMaxHz">Highest pitch bin edge, in Hz.</param>
    /// <param name="energyBins">Number of uniform energy bins (256).</param>
    /// <param name="energyMin">Lowest energy bin edge.</param>
    /// <param name="energyMax">Highest energy bin edge (default: <see cref="MaxStftEnergy"/> for a 1024-point STFT).</param>
    /// <param name="usePitch">Whether the pitch branch (predictor and embedding) is present; the paper's ablation removes it.</param>
    /// <param name="useEnergy">Whether the energy branch (predictor and embedding) is present; the paper's ablation removes it.</param>
    public VarianceAdaptorLayer(
        [LayerState] int hiddenSize,
        [LayerState] int filterSize = 256,
        [LayerState] int kernelSize = 3,
        [LayerState] double dropoutRate = 0.5,
        [LayerState] int pitchBins = 256,
        [LayerState] double pitchMinHz = 71.0,
        [LayerState] double pitchMaxHz = 800.0,
        [LayerState] int energyBins = 256,
        [LayerState] double energyMin = 0.0,
        [LayerState] double energyMax = 677.3123356325352,
        [LayerState] bool usePitch = true,
        [LayerState] bool useEnergy = true)
        : base(new[] { hiddenSize }, new[] { hiddenSize })
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (pitchBins < 2) throw new ArgumentOutOfRangeException(nameof(pitchBins));
        if (energyBins < 2) throw new ArgumentOutOfRangeException(nameof(energyBins));
        if (pitchMinHz <= 0 || pitchMaxHz <= pitchMinHz) throw new ArgumentOutOfRangeException(nameof(pitchMaxHz));
        if (energyMax <= energyMin) throw new ArgumentOutOfRangeException(nameof(energyMax));

        _hiddenSize = hiddenSize;
        _filterSize = filterSize;
        _kernelSize = kernelSize;
        _dropoutRate = dropoutRate;
        _pitchBins = pitchBins;
        _pitchMinHz = pitchMinHz;
        _pitchMaxHz = pitchMaxHz;
        _energyBins = energyBins;
        _energyMin = energyMin;
        _energyMax = energyMax;
        _pitchBoundaries = LogSpacedBoundaries(pitchMinHz, pitchMaxHz, pitchBins);
        _energyBoundaries = LinearBoundaries(energyMin, energyMax, energyBins);
        _usePitch = usePitch;
        _useEnergy = useEnergy;

        _durationPredictor = new VariancePredictorLayer<T>(hiddenSize, filterSize, 1, kernelSize, dropoutRate);
        RegisterSubLayer(_durationPredictor);
        if (usePitch)
        {
            _pitchPredictor = new VariancePredictorLayer<T>(hiddenSize, filterSize, PitchWaveletTransform.ScaleCount, kernelSize, dropoutRate);
            _pitchStatistics = new DenseLayer<T>(2, new IdentityActivation<T>() as IActivationFunction<T>);
            _pitchEmbedding = new EmbeddingLayer<T>(pitchBins, hiddenSize);
            RegisterSubLayer(_pitchPredictor);
            RegisterSubLayer(_pitchStatistics);
            RegisterSubLayer(_pitchEmbedding);
        }
        if (useEnergy)
        {
            _energyPredictor = new VariancePredictorLayer<T>(hiddenSize, filterSize, 1, kernelSize, dropoutRate);
            _energyEmbedding = new EmbeddingLayer<T>(energyBins, hiddenSize);
            RegisterSubLayer(_energyPredictor);
            RegisterSubLayer(_energyEmbedding);
        }
    }

    /// <summary>Width of the hidden sequence.</summary>
    public int HiddenSize => _hiddenSize;
    /// <summary>Convolution channels of each predictor.</summary>
    public int FilterSize => _filterSize;
    /// <summary>Convolution kernel of each predictor.</summary>
    public int KernelSize => _kernelSize;
    /// <summary>Dropout of each predictor.</summary>
    public double DropoutRate => _dropoutRate;
    /// <summary>Number of pitch bins.</summary>
    public int PitchBins => _pitchBins;
    /// <summary>Lowest pitch bin edge, in Hz.</summary>
    public double PitchMinHz => _pitchMinHz;
    /// <summary>Highest pitch bin edge, in Hz.</summary>
    public double PitchMaxHz => _pitchMaxHz;
    /// <summary>Number of energy bins.</summary>
    public int EnergyBins => _energyBins;
    /// <summary>Lowest energy bin edge.</summary>
    public double EnergyMin => _energyMin;
    /// <summary>Highest energy bin edge.</summary>
    public double EnergyMax => _energyMax;
    /// <summary>
    /// Speed control for inference: predicted durations are multiplied by this before rounding (FastSpeech's α,
    /// Ren et al. 2019 §3.2; 1 is the trained speed, larger is slower).
    /// </summary>
    public double DurationScale { get; set; } = 1.0;
    /// <summary>Whether the pitch branch is present.</summary>
    public bool UsePitch => _usePitch;
    /// <summary>Whether the energy branch is present.</summary>
    public bool UseEnergy => _useEnergy;

    /// <summary>
    /// The largest L2 norm a frame of a periodic-Hann STFT of size <paramref name="fftSize"/> can have for audio in
    /// [-1, 1]: by Parseval the half spectrum carries at most N·Σw²/2 + (Σw)² = N²·7/16 of energy.
    /// </summary>
    public static double MaxStftEnergy(int fftSize) => fftSize * Math.Sqrt(7.0 / 16.0);

    /// <summary>Sets the energy bin range, normally the minimum and maximum frame energy of the training data.</summary>
    public void SetEnergyRange(double min, double max)
    {
        if (!(max > min)) throw new ArgumentOutOfRangeException(nameof(max), "The energy range must be non-empty.");
        _energyMin = min;
        _energyMax = max;
        _energyBoundaries = LinearBoundaries(min, max, _energyBins);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The inference path. Each utterance of a batch expands by its own predicted durations, so a batch of more than
    /// one is zero-padded to its longest expansion, <c>[batch, maxFrames, hidden]</c>, as FastSpeech's reference
    /// implementations pad their length-regulator output.
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[0] == 1)
            return Adapt(input, targets: null, DurationScale).Expanded;

        int batch = input.Shape[0], tokens = input.Shape[1], hidden = input.Shape[2];
        var expanded = new Tensor<T>[batch];
        int maxFrames = 0;
        for (int b = 0; b < batch; b++)
        {
            var utterance = Engine.TensorSlice(input, new[] { b, 0, 0 }, new[] { 1, tokens, hidden });
            expanded[b] = Adapt(utterance, targets: null, DurationScale).Expanded; // [1, frames, hidden]
            maxFrames = Math.Max(maxFrames, expanded[b].Shape[1]);
        }
        for (int b = 0; b < batch; b++)
        {
            int frames = expanded[b].Shape[1];
            if (frames < maxFrames)
                expanded[b] = Engine.TensorConcatenate(
                    new[] { expanded[b], new Tensor<T>(new[] { 1, maxFrames - frames, hidden }) }, axis: 1);
        }
        return Engine.TensorConcatenate(expanded, axis: 0);
    }

    /// <summary>
    /// Runs the adaptor. With <paramref name="targets"/> the ground-truth duration, pitch and energy drive the
    /// expansion and embeddings (training); without, the predictions do (inference).
    /// </summary>
    /// <param name="hidden">Phoneme hidden sequence, <c>[tokens, hidden]</c> or <c>[1, tokens, hidden]</c>.</param>
    /// <param name="targets">Ground-truth variance for training, or null at inference.</param>
    /// <param name="durationScale">Speaking-rate control applied to predicted durations (FastSpeech §3.2).</param>
    public VarianceAdaptation<T> Adapt(Tensor<T> hidden, VarianceTargets? targets, double durationScale = 1.0)
    {
        if (hidden is null) throw new ArgumentNullException(nameof(hidden));
        bool batched = hidden.Rank == 3;
        if (batched && hidden.Shape[0] != 1)
            throw new ArgumentException(
                $"Adapt takes one utterance, [tokens, hidden] or [1, tokens, hidden], because each utterance expands to its own length; got batch {hidden.Shape[0]}.",
                nameof(hidden));
        var phonemes = batched ? Engine.Reshape(hidden, new[] { hidden.Shape[1], hidden.Shape[2] }) : hidden;
        int tokenCount = phonemes.Shape[0];

        var logDuration = Engine.Reshape(_durationPredictor.Forward(phonemes), new[] { tokenCount });
        int[] durations = targets?.Durations ?? PredictedDurations(logDuration, durationScale);
        if (durations.Length != tokenCount)
            throw new ArgumentException($"Got {durations.Length} durations for {tokenCount} tokens.", nameof(targets));

        var frames = LengthRegulate(phonemes, durations);
        var (adapted, pitchSpectrogram, pitchStatistics, energy) = AddVariance(frames, targets?.Pitch, targets?.Energy);

        int frameCount = adapted.Shape[0];
        var expanded = batched ? Engine.Reshape(adapted, new[] { 1, frameCount, _hiddenSize }) : adapted;
        return new VarianceAdaptation<T>(expanded, logDuration, durations, pitchSpectrogram, pitchStatistics, energy);
    }

    /// <summary>
    /// Adds the pitch and energy embeddings to a frame-level hidden sequence, from the given frame pitch and energy,
    /// or from the adaptor's predictions where those are null.
    /// </summary>
    /// <param name="frames">Frame-level hidden sequence, <c>[frames, hidden]</c>.</param>
    /// <param name="pitch">F0 per frame in Hz (0 unvoiced), or null to use the prediction.</param>
    /// <param name="energy">Energy per frame, or null to use the prediction.</param>
    /// <remarks>The second half of <see cref="Adapt"/>, for a hidden sequence that is already frame-level: AdaSpeech 2
    /// reconstructs untranscribed speech from a mel encoder's frame sequence through the same variance information
    /// and decoder (Yan et al. 2021, §2.3).</remarks>
    public Tensor<T> AddFrameVariance(Tensor<T> frames, double[]? pitch, double[]? energy)
    {
        if (frames is null) throw new ArgumentNullException(nameof(frames));
        if (frames.Rank != 2 || frames.Shape[1] != _hiddenSize)
            throw new ArgumentException($"Expected frames [frames, {_hiddenSize}], got [{string.Join(", ", frames.Shape)}].", nameof(frames));
        return AddVariance(frames, pitch, energy).Frames;
    }

    private (Tensor<T> Frames, Tensor<T>? PitchSpectrogram, Tensor<T>? PitchStatistics, Tensor<T>? Energy) AddVariance(
        Tensor<T> frames, double[]? pitchTarget, double[]? energyTarget)
    {
        int frameCount = frames.Shape[0];
        Tensor<T>? pitchSpectrogram = null, pitchStatistics = null;
        if (_pitchPredictor is not null && _pitchStatistics is not null && _pitchEmbedding is not null)
        {
            pitchSpectrogram = _pitchPredictor.ForwardWithHidden(frames, out var pitchHidden);
            var pooled = Engine.ReduceMean(pitchHidden, new[] { 0 }, keepDims: true); // [1, filter]
            pitchStatistics = Engine.Reshape(_pitchStatistics.Forward(pooled), new[] { 2 });

            double[] f0 = pitchTarget ?? PredictedF0(pitchSpectrogram, pitchStatistics);
            if (f0.Length != frameCount)
                throw new ArgumentException($"Got {f0.Length} pitch values for {frameCount} frames.", nameof(pitchTarget));
            frames = Engine.TensorAdd(frames, _pitchEmbedding.Forward(Bucketize(f0, _pitchBoundaries)));
        }

        Tensor<T>? energy = null;
        if (_energyPredictor is not null && _energyEmbedding is not null)
        {
            energy = Engine.Reshape(_energyPredictor.Forward(frames), new[] { frameCount });
            double[] energyValues = energyTarget ?? ToDoubles(energy);
            if (energyValues.Length != frameCount)
                throw new ArgumentException($"Got {energyValues.Length} energy values for {frameCount} frames.", nameof(energyTarget));
            frames = Engine.TensorAdd(frames, _energyEmbedding.Forward(Bucketize(energyValues, _energyBoundaries)));
        }

        return (frames, pitchSpectrogram, pitchStatistics, energy);
    }

    /// <summary>
    /// FastSpeech's length regulator: repeats row <c>i</c> of <paramref name="phonemes"/> <c>durations[i]</c> times
    /// (<see cref="AiDotNet.TextToSpeech.LengthRegulator.Expand"/>).
    /// </summary>
    public Tensor<T> LengthRegulate(Tensor<T> phonemes, int[] durations)
        => AiDotNet.TextToSpeech.LengthRegulator.Expand(phonemes, durations);

    private int[] PredictedDurations(Tensor<T> logDuration, double scale)
    {
        var durations = new int[logDuration.Length];
        int total = 0;
        for (int i = 0; i < durations.Length; i++)
        {
            double d = Math.Round((Math.Exp(NumOps.ToDouble(logDuration[i])) - 1.0) * scale);
            durations[i] = (int)Math.Max(0, d);
            total += durations[i];
        }
        if (total == 0)
        {
            // Every phoneme predicted to vanish: keep one frame per phoneme rather than synthesize nothing.
            for (int i = 0; i < durations.Length; i++) durations[i] = 1;
        }
        return durations;
    }

    private double[] PredictedF0(Tensor<T> pitchSpectrogram, Tensor<T> pitchStatistics)
    {
        int frames = pitchSpectrogram.Shape[0], scales = pitchSpectrogram.Shape[1];
        var spectrogram = new double[frames, scales];
        for (int t = 0; t < frames; t++)
            for (int s = 0; s < scales; s++)
                spectrogram[t, s] = NumOps.ToDouble(pitchSpectrogram[t, s]);
        double mean = NumOps.ToDouble(pitchStatistics[0]);
        double std = Math.Abs(NumOps.ToDouble(pitchStatistics[1]));
        return PitchWaveletTransform.ToF0(spectrogram, mean, std);
    }

    private double[] ToDoubles(Tensor<T> tensor)
    {
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.ToDouble(tensor[i]);
        return values;
    }

    /// <summary>torch.bucketize(values, boundaries) with right=False: the index of the first boundary >= value.</summary>
    private Tensor<T> Bucketize(double[] values, double[] boundaries)
    {
        var indices = new Tensor<T>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++)
        {
            int lo = 0, hi = boundaries.Length;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (boundaries[mid] < values[i]) lo = mid + 1; else hi = mid;
            }
            indices[i] = NumOps.FromDouble(lo);
        }
        return indices;
    }

    private static double[] LogSpacedBoundaries(double min, double max, int bins)
    {
        var edges = new double[bins - 1];
        double lmin = Math.Log(min), lmax = Math.Log(max);
        for (int i = 0; i < edges.Length; i++)
            edges[i] = Math.Exp(lmin + (lmax - lmin) * i / Math.Max(1, edges.Length - 1));
        return edges;
    }

    private static double[] LinearBoundaries(double min, double max, int bins)
    {
        var edges = new double[bins - 1];
        for (int i = 0; i < edges.Length; i++) edges[i] = min + (max - min) * i / Math.Max(1, edges.Length - 1);
        return edges;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child.ResetState();
    }

    /// <summary>Persists the constructor arguments, including the fitted energy range.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["HiddenSize"] = _hiddenSize.ToString(inv);
        metadata["FilterSize"] = _filterSize.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        metadata["DropoutRate"] = _dropoutRate.ToString("R", inv);
        metadata["PitchBins"] = _pitchBins.ToString(inv);
        metadata["PitchMinHz"] = _pitchMinHz.ToString("R", inv);
        metadata["PitchMaxHz"] = _pitchMaxHz.ToString("R", inv);
        metadata["EnergyBins"] = _energyBins.ToString(inv);
        metadata["EnergyMin"] = _energyMin.ToString("R", inv);
        metadata["EnergyMax"] = _energyMax.ToString("R", inv);
        metadata["UsePitch"] = _usePitch.ToString(inv);
        metadata["UseEnergy"] = _useEnergy.ToString(inv);
        return metadata;
    }
}

/// <summary>Ground-truth variance for one utterance (FastSpeech 2 teacher forcing).</summary>
public sealed class VarianceTargets
{
    /// <summary>Frames per token.</summary>
    public int[]? Durations { get; init; }
    /// <summary>F0 per frame in Hz, with unvoiced frames filled (continuous contour).</summary>
    public double[]? Pitch { get; init; }
    /// <summary>Energy per frame.</summary>
    public double[]? Energy { get; init; }
}

/// <summary>What the variance adaptor produced for one utterance.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class VarianceAdaptation<T>
{
    internal VarianceAdaptation(Tensor<T> expanded, Tensor<T> logDuration, int[] durations,
        Tensor<T>? pitchSpectrogram, Tensor<T>? pitchStatistics, Tensor<T>? energy)
    {
        Expanded = expanded;
        LogDuration = logDuration;
        Durations = durations;
        PitchSpectrogram = pitchSpectrogram;
        PitchStatistics = pitchStatistics;
        Energy = energy;
    }

    /// <summary>The frame-level hidden sequence with pitch and energy embeddings added.</summary>
    public Tensor<T> Expanded { get; }
    /// <summary>Predicted log(duration + 1) per token.</summary>
    public Tensor<T> LogDuration { get; }
    /// <summary>The durations used for the expansion.</summary>
    public int[] Durations { get; }
    /// <summary>Predicted CWT pitch spectrogram, <c>[frames, scales]</c>; null without the pitch branch.</summary>
    public Tensor<T>? PitchSpectrogram { get; }
    /// <summary>Predicted utterance log-F0 mean and deviation, <c>[2]</c>; null without the pitch branch.</summary>
    public Tensor<T>? PitchStatistics { get; }
    /// <summary>Predicted energy per frame; null without the energy branch.</summary>
    public Tensor<T>? Energy { get; }
}
