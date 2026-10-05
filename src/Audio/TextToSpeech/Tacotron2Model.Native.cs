using AiDotNet.Helpers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>Tacotron 2's spectrogram prediction network (Shen et al. 2018, §2.2), the native path of
/// <see cref="Tacotron2Model{T}"/>.</summary>
public partial class Tacotron2Model<T>
{
    // Layers, in the order CreateTacotron2Layers publishes them.
    private EmbeddingLayer<T>? _embedding;
    private ConvBatchNormStackLayer<T>? _encoderConvolutions;
    private BidirectionalRecurrentLayer<T>? _encoderLstm;
    private BiasFreeLinearLayer<T>? _prenet1;
    private BiasFreeLinearLayer<T>? _prenet2;
    private LSTMCellLayer<T>? _attentionRnn;
    private LocationSensitiveAttentionLayer<T>? _attention;
    private LSTMCellLayer<T>? _decoderRnn;
    private DenseLayer<T>? _melProjection;
    private DenseLayer<T>? _stopProjection;
    private ConvBatchNormStackLayer<T>? _postnet;

    private Random _regularizationRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0);

    [Scratch]
    private Tensor<T>? _teacherForcingTarget;

    private bool HasPaperLayers => _attention is not null;

    private void InitializeNativeLayers()
    {
        bool builtDefaultLayers = Architecture.Layers is not { Count: > 0 };
        Layers.AddRange(builtDefaultLayers
            ? LayerHelper<T>.CreateTacotron2Layers(_vocabSize, _embeddingDim, _encoderDim, _decoderDim, _attentionDim,
                _attentionFilters, _prenetDim, NumMels, _numMelsPerFrame, _numEncoderConvLayers, _numPostnetConvLayers,
                _postnetEmbeddingDim, _options.ConvolutionDropout, _options.AttentionKernelSize)
            : Architecture.Layers!);
        BindNativeLayersFromPublishedList();
        if (builtDefaultLayers)
        {
            int contextIn = _decoderDim + _encoderDim;
            _melProjection?.ResolveShapesOnly(new[] { contextIn });
            _stopProjection?.ResolveShapesOnly(new[] { contextIn });
        }
    }

    // Binds the published layers by position; a caller-supplied stack that does not follow CreateTacotron2Layers' order
    // runs front to back instead.
    private void BindNativeLayersFromPublishedList()
    {
        var expected = Layers.Count == 11
            && Layers[0] is EmbeddingLayer<T> && Layers[1] is ConvBatchNormStackLayer<T>
            && Layers[2] is BidirectionalRecurrentLayer<T> && Layers[3] is BiasFreeLinearLayer<T>
            && Layers[4] is BiasFreeLinearLayer<T> && Layers[5] is LSTMCellLayer<T>
            && Layers[6] is LocationSensitiveAttentionLayer<T> && Layers[7] is LSTMCellLayer<T>
            && Layers[8] is DenseLayer<T> && Layers[9] is DenseLayer<T> && Layers[10] is ConvBatchNormStackLayer<T>;
        if (!expected)
            return;
        _embedding = (EmbeddingLayer<T>)Layers[0];
        _encoderConvolutions = (ConvBatchNormStackLayer<T>)Layers[1];
        _encoderLstm = (BidirectionalRecurrentLayer<T>)Layers[2];
        _prenet1 = (BiasFreeLinearLayer<T>)Layers[3];
        _prenet2 = (BiasFreeLinearLayer<T>)Layers[4];
        _attentionRnn = (LSTMCellLayer<T>)Layers[5];
        _attention = (LocationSensitiveAttentionLayer<T>)Layers[6];
        _decoderRnn = (LSTMCellLayer<T>)Layers[7];
        _melProjection = (DenseLayer<T>)Layers[8];
        _stopProjection = (DenseLayer<T>)Layers[9];
        _postnet = (ConvBatchNormStackLayer<T>)Layers[10];
    }

    private Tensor<T> Tokens(Tensor<T> phonemes)
        => phonemes.Rank == 2 && phonemes.Shape[0] == 1 ? Engine.Reshape(phonemes, new[] { phonemes.Shape[1] }) : phonemes;

    /// <summary>Encoder (§2.2): character embedding, 3 convolutions with batch normalization, ReLU and dropout, and a
    /// bidirectional LSTM; <c>[tokens, encoderDim]</c>.</summary>
    private Tensor<T> Encode(Tensor<T> phonemes)
    {
        var x = _embedding!.Forward(Tokens(phonemes));
        x = _encoderConvolutions!.Forward(x);
        return _encoderLstm!.Forward(x);
    }

    /// <summary>The pre-net (§2.2): 2 bias-free fully connected ReLU layers with dropout 0.5 that stays on at inference,
    /// "to introduce output variation".</summary>
    private Tensor<T> Prenet(Tensor<T> previousFrames)
    {
        var x = AlwaysOnDropout(Engine.ReLU(_prenet1!.Forward(previousFrames)));
        return AlwaysOnDropout(Engine.ReLU(_prenet2!.Forward(x)));
    }

    private Tensor<T> AlwaysOnDropout(Tensor<T> x)
    {
        double p = _options.PrenetDropout;
        if (p <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - p));
        for (int i = 0; i < mask.Length; i++) mask[i] = _regularizationRandom.NextDouble() < p ? NumOps.Zero : keep;
        return Engine.TensorMultiply(x, mask);
    }

    // Zoneout (Krueger et al. 2017), probability 0.1 (§2.2): in training each unit of the LSTM state keeps its previous
    // value with probability p; at inference the expectation p · previous + (1 − p) · new.
    private Tensor<T> Zoneout(Tensor<T> previous, Tensor<T> next)
    {
        double p = _options.ZoneoutProbability;
        if (p <= 0) return next;
        if (!IsTrainingMode)
            return Engine.TensorAdd(Engine.TensorMultiplyScalar(previous, NumOps.FromDouble(p)), Engine.TensorMultiplyScalar(next, NumOps.FromDouble(1 - p)));
        var mask = new Tensor<T>(next._shape);
        for (int i = 0; i < mask.Length; i++) mask[i] = _regularizationRandom.NextDouble() < p ? NumOps.One : NumOps.Zero;
        var change = Engine.TensorAddScalar(Engine.TensorNegate(mask), NumOps.One);
        return Engine.TensorAdd(Engine.TensorMultiply(mask, previous), Engine.TensorMultiply(change, next));
    }

    /// <summary>
    /// The autoregressive decoder (§2.2): pre-net on the previous frame group, attention RNN, location-sensitive
    /// attention over [previous; cumulative] weights, decoder RNN, then the linear mel projection and the stop token from
    /// [decoder output; context]. Teacher-forced when <paramref name="target"/> is given (the previous ground-truth group,
    /// all zeros first); otherwise it feeds back its own output and stops at the first stop probability above the
    /// threshold. Returns the frames before the post-net <c>[frames, mel]</c> and the stop logits <c>[steps]</c>.
    /// </summary>
    private (Tensor<T> Frames, Tensor<T> StopLogits) Decode(Tensor<T> memory, Tensor<T>? target)
    {
        int tokens = memory.Shape[0], groupWidth = NumMels * _numMelsPerFrame;
        int steps = target is null ? _maxDecoderSteps : (target.Shape[0] + _numMelsPerFrame - 1) / _numMelsPerFrame;
        var projectedMemory = _attention!.ProjectMemory(memory);
        var attentionState = new Tensor<T>(new[] { 1, 2 * _decoderDim });
        var decoderState = new Tensor<T>(new[] { 1, 2 * _decoderDim });
        var context = new Tensor<T>(new[] { 1, _encoderDim });
        var weights = new Tensor<T>(new[] { 1, tokens });
        var cumulative = new Tensor<T>(new[] { 1, tokens });
        var previous = new Tensor<T>(new[] { 1, groupWidth });
        var groups = new List<Tensor<T>>();
        var stops = new List<Tensor<T>>();
        for (int step = 0; step < steps; step++)
        {
            var prenet = Prenet(previous);
            attentionState = Zoneout(attentionState, _attentionRnn!.Forward(Engine.TensorConcatenate(new[] { prenet, context }, 1), attentionState));
            var attentionHidden = _attentionRnn.SplitState(attentionState).Hidden;
            (context, weights) = _attention.Attend(attentionHidden, memory, projectedMemory, weights, cumulative);
            cumulative = Engine.TensorAdd(cumulative, weights);
            decoderState = Zoneout(decoderState, _decoderRnn!.Forward(Engine.TensorConcatenate(new[] { attentionHidden, context }, 1), decoderState));
            var decoderOutput = Engine.TensorConcatenate(new[] { _decoderRnn.SplitState(decoderState).Hidden, context }, 1);
            var group = _melProjection!.Forward(decoderOutput);                                  // [1, mel · r]
            var stop = _stopProjection!.Forward(decoderOutput);                                  // [1, 1]
            groups.Add(group);
            stops.Add(Engine.Reshape(stop, new[] { 1 }));
            if (target is null)
            {
                previous = group;
                if (1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(stop[0, 0]))) > _stopThreshold)
                    break;
            }
            else
            {
                previous = TargetGroup(target, step);
            }
        }
        var frames = Engine.Reshape(groups.Count == 1 ? groups[0] : Engine.TensorConcatenate(groups.ToArray(), 0),
            new[] { groups.Count * _numMelsPerFrame, NumMels });
        var stopLogits = stops.Count == 1 ? stops[0] : Engine.TensorConcatenate(stops.ToArray(), 0);
        return (frames, stopLogits);
    }

    // The ground-truth frame group of step `step`, zero-padded past the end.
    private Tensor<T> TargetGroup(Tensor<T> target, int step)
    {
        var group = new Tensor<T>(new[] { 1, NumMels * _numMelsPerFrame });
        for (int f = 0; f < _numMelsPerFrame; f++)
        {
            int frame = step * _numMelsPerFrame + f;
            if (frame >= target.Shape[0]) break;
            for (int m = 0; m < NumMels; m++) group[0, f * NumMels + m] = target[frame, m];
        }
        return group;
    }

    // [1, frames, mel] or [1, frames · mel] or [frames, mel] -> [frames, mel].
    private Tensor<T> MelRows(Tensor<T> mel)
    {
        if (mel.Rank == 3) return Engine.Reshape(mel, new[] { mel.Shape[1], mel.Shape[2] });
        if (mel.Rank == 2 && mel.Shape[1] == NumMels && mel.Shape[0] != 1) return mel;
        if (mel.Rank == 2) return Engine.Reshape(mel, new[] { mel.Shape[1] / NumMels, NumMels });
        throw new ArgumentException($"Expected a mel spectrogram [1, frames, {NumMels}], got [{string.Join(", ", mel.Shape)}].", nameof(mel));
    }

    /// <summary>Synthesis (§2.2): encode, decode autoregressively until the stop token fires, add the post-net residual;
    /// <c>[1, frames, mel]</c>. The pre-net dropout is drawn from the sampling seed, so a model synthesizes the same
    /// output for the same text.</summary>
    private Tensor<T> ForwardNative(Tensor<T> phonemes)
    {
        if (!HasPaperLayers)
        {
            var x = phonemes;
            foreach (var layer in Layers) x = layer.Forward(x);
            return x;
        }
        using var _ = new NoGradScope<T>();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        _regularizationRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        try
        {
            var (frames, _) = Decode(Encode(phonemes), null);
            var refined = Engine.TensorAdd(frames, _postnet!.Forward(frames));
            return Engine.Reshape(refined, new[] { 1, refined.Shape[0], NumMels });
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (_teacherForcingTarget is not null && HasPaperLayers)
        {
            var target = MelRows(_teacherForcingTarget);
            var (frames, _) = Decode(Encode(input), target);
            frames = Engine.TensorSlice(frames, new[] { 0, 0 }, new[] { target.Shape[0], NumMels });
            var refined = Engine.TensorAdd(frames, _postnet!.Forward(frames));
            return Engine.Reshape(refined, new[] { 1, refined.Shape[0], NumMels });
        }
        return base.ForwardForTraining(input);
    }

    /// <summary>
    /// The training objective (§2.2, §3.1; reference <c>Tacotron2Loss</c>): teacher-forced decoding, the summed MSE of
    /// the mel before and after the post-net, plus the binary cross-entropy of the stop logits against a target that is
    /// 1 from the last frame's step on.
    /// </summary>
    private Tensor<T> Objective(Tensor<T> phonemes, Tensor<T> expected)
    {
        var target = MelRows(expected);
        int frames = target.Shape[0];
        var (decoded, stopLogits) = Decode(Encode(phonemes), target);
        var before = Engine.TensorSlice(decoded, new[] { 0, 0 }, new[] { frames, NumMels });
        var after = Engine.TensorAdd(before, _postnet!.Forward(before));
        var loss = Engine.TensorAdd(MeanSquaredDifference(before, target), MeanSquaredDifference(after, target));

        int steps = stopLogits.Length;
        var stopTarget = new Tensor<T>(new[] { steps });
        stopTarget[steps - 1] = NumOps.One;
        // BCE with logits: softplus(z) − z · y, averaged over steps.
        var bce = Engine.TensorSubtract(Engine.Softplus(stopLogits), Engine.TensorMultiply(stopLogits, stopTarget));
        return Engine.TensorAdd(loss, Engine.ReduceMean(bce, new[] { 0 }, keepDims: false));
    }

    private Tensor<T> MeanSquaredDifference(Tensor<T> predicted, Tensor<T> target)
    {
        var diff = Engine.TensorSubtract(predicted, target);
        var squared = Engine.TensorMultiply(diff, diff);
        return Engine.ReduceMean(squared, Enumerable.Range(0, squared.Shape.Length).ToArray(), keepDims: false);
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot train in ONNX inference mode.");
        if (expectedOutput is null)
            throw new ArgumentException("expectedOutput cannot be null for teacher-forced training.", nameof(expectedOutput));
        if (!HasPaperLayers)
        {
            TrainWithTape(input, expectedOutput, _optimizer);
            return;
        }
        var target = MelRows(expectedOutput);
        if (target.Shape[0] == 0)
            throw new ArgumentException("expectedOutput has no mel frames.", nameof(expectedOutput));
        _teacherForcingTarget = expectedOutput;
        try
        {
            TrainWithCustomObjective(input, expectedOutput, Objective, _optimizer);
        }
        finally
        {
            _teacherForcingTarget = null;
        }
    }
}
