using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Shared parallel duration/alignment path for native token-to-mel models with a registered
/// encoder and decoder. It does not change the inherited frame-level Predict/Train contract.
/// </summary>
/// <remarks>
/// Equal-length examples share a forward pass. Different lengths are bucketed rather than
/// exposing padding to an attention layer without an external padding-mask contract. Hidden
/// features, expansion, and losses stay on the selected tensor engine; only integer lengths,
/// durations, and discrete monotonic-alignment scores are read on the CPU.
/// </remarks>
public abstract partial class AlignedTextToMelModelBase<T> : AudioNeuralNetworkBase<T>
{
    private EmbeddingLayer<T>? _tokenEmbedding;
    private TokenDurationPredictorLayer<T>? _durationPredictor;
    private FullyConnectedLayer<T>? _melPrior;
    private int _tokenVocabularySize;
    private int _encoderWidth;
    private int _melChannels;
    private int _melHopSize;
    private int _encoderStart;
    private int _decoderStart;
    [Scratch] private bool _alignedDisposed;

    /// <summary>Creates a model that may configure a native aligned path after building its layers.</summary>
    protected AlignedTextToMelModelBase(NeuralNetworkArchitecture<T> architecture) : base(architecture) { }

    /// <summary>Whether this model has a registered native token encoder and mel decoder split.</summary>
    public bool SupportsAlignedMelSynthesis => _tokenEmbedding is not null && _durationPredictor is not null && _melPrior is not null;

    /// <summary>
    /// Registers the token embedding, duration branch, and Gaussian mel prior. Layer boundaries
    /// index the authoritative Layers collection, including after deserialization replaces it.
    /// </summary>
    protected void ConfigureAlignedMelPath(int vocabularySize, int encoderWidth, int melChannels,
        int melHopSize, int durationWidth, int durationDepth, double dropoutRate,
        int encoderStart, int decoderStart)
    {
        if (vocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (encoderWidth <= 0) throw new ArgumentOutOfRangeException(nameof(encoderWidth));
        if (melChannels <= 0) throw new ArgumentOutOfRangeException(nameof(melChannels));
        if (melHopSize <= 0) throw new ArgumentOutOfRangeException(nameof(melHopSize));
        if (durationWidth <= 0) throw new ArgumentOutOfRangeException(nameof(durationWidth));
        if (durationDepth <= 0) throw new ArgumentOutOfRangeException(nameof(durationDepth));
        if (double.IsNaN(dropoutRate) || double.IsInfinity(dropoutRate) || dropoutRate < 0 || dropoutRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        if (encoderStart < 0 || decoderStart <= encoderStart || decoderStart >= Layers.Count)
            throw new ArgumentException("The registered encoder and decoder ranges must both be nonempty.");
        if (SupportsAlignedMelSynthesis)
            throw new InvalidOperationException("The aligned path is already configured.");

        _tokenVocabularySize = vocabularySize;
        _encoderWidth = encoderWidth;
        _melChannels = melChannels;
        _melHopSize = melHopSize;
        _encoderStart = encoderStart;
        _decoderStart = decoderStart;
        _tokenEmbedding = new EmbeddingLayer<T>(vocabularySize, encoderWidth);
        _durationPredictor = new TokenDurationPredictorLayer<T>(encoderWidth, durationWidth, durationDepth, dropoutRate);
        _melPrior = new FullyConnectedLayer<T>(encoderWidth, melChannels, new IdentityActivation<T>());
    }

    /// <summary>
    /// Evaluates the decoded-mel, aligned-prior, and log-duration losses. If durations are
    /// absent, monotonic maximum-path alignment is derived from the learned Gaussian prior.
    /// A caller-owned gradient tape can differentiate the returned losses.
    /// </summary>
    public AlignedMelObjective<T> EvaluateAlignedMel(AlignedMelBatch<T> batch)
    {
        ValidateBatch(batch);
        var tokens = ConvertActiveTokens(batch.Tokens, batch.TokenLengths);
        var result = ComputeObjective(tokens, batch.TargetMels, batch, includeOutput: true);
        return result.Output is { } output
            ? new AlignedMelObjective<T>(result.MelLoss, result.PriorLoss, result.DurationLoss, result.TotalLoss, output)
            : throw new InvalidOperationException("The aligned evaluation did not produce its requested output.");
    }

    /// <summary>Trains all reached encoder, duration, prior, and decoder parameters in one tape step.</summary>
    protected T TrainAlignedMel(AlignedMelBatch<T> batch,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
    {
        ValidateBatch(batch);
        var tokens = ConvertActiveTokens(batch.Tokens, batch.TokenLengths);
        return TrainWithCustomObjective(tokens, batch.TargetMels,
            (currentTokens, currentMels) => ComputeObjective(currentTokens, currentMels, batch, includeOutput: false).TotalLoss,
            optimizer);
    }

    /// <summary>
    /// Predicts learned durations and expands actual token features into a mel spectrogram.
    /// A larger speaking rate produces shorter durations. This result is not a waveform.
    /// </summary>
    /// <remarks>Runs without gradient recording and restores the caller's training mode even if synthesis fails.</remarks>
    public AlignedMelOutput<T> SynthesizeMel(Tensor<int> tokens, IReadOnlyList<int> tokenLengths,
        double speakingRate = 1.0, int maximumMelFrames = 100000)
    {
        ValidateTokens(tokens, tokenLengths);
        if (double.IsNaN(speakingRate) || double.IsInfinity(speakingRate) || speakingRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(speakingRate));
        if (maximumMelFrames <= 0) throw new ArgumentOutOfRangeException(nameof(maximumMelFrames));
        bool wasTraining = IsTrainingMode;
        try
        {
            SetTrainingMode(false);
            using var noGrad = new NoGradScope<T>();
            return SynthesizeMelCore(tokens, tokenLengths, speakingRate, maximumMelFrames);
        }
        finally
        {
            if (IsTrainingMode != wasTraining) SetTrainingMode(wasTraining);
        }
    }

    private AlignedMelOutput<T> SynthesizeMelCore(Tensor<int> tokens, IReadOnlyList<int> tokenLengths,
        double speakingRate, int maximumMelFrames)
    {
        double logSpeakingRate = Math.Log(speakingRate);
        var numericTokens = ConvertActiveTokens(tokens, tokenLengths);
        var outputs = new List<OutputBucket>();
        var durations = new Tensor<int>(tokens.Shape.ToArray());
        var melLengths = new int[tokens.Shape[0]];

        foreach (var group in MakeBuckets(tokenLengths, null))
        {
            int count = group.Value.Count;
            int tokenCount = group.Key.Tokens;
            var encoded = EncodeTokens(SelectBucket(numericTokens, group.Value, tokenCount));
            var predicted = PredictLogDurations(encoded);
            var predictedValues = predicted.AsSpan();
            var byMelLength = new Dictionary<int, List<int>>();
            for (int row = 0; row < count; row++)
            {
                long total = 0;
                int originalRow = group.Value[row];
                for (int token = 0; token < tokenCount; token++)
                {
                    double logDuration = NumOps.ToDouble(predictedValues[row * tokenCount + token]);
                    if (double.IsNaN(logDuration) || double.IsInfinity(logDuration))
                        throw new InvalidOperationException("The duration predictor returned a non-finite log-duration.");
                    double raw = Math.Exp(logDuration - logSpeakingRate);
                    if (double.IsNaN(raw) || double.IsInfinity(raw) || raw > maximumMelFrames)
                        throw new InvalidOperationException("Predicted durations exceed the requested finite mel-frame budget.");
                    int duration = Math.Max(1, (int)Math.Ceiling(raw));
                    total += duration;
                    if (total > maximumMelFrames)
                        throw new InvalidOperationException("Predicted durations exceed the requested mel-frame budget.");
                    durations[originalRow, token] = duration;
                }
                int frames = checked((int)total);
                melLengths[originalRow] = frames;
                if (!byMelLength.TryGetValue(frames, out var members))
                    byMelLength.Add(frames, members = new List<int>());
                members.Add(row);
            }

            foreach (var subgroup in byMelLength)
            {
                var originalRows = subgroup.Value.Select(row => group.Value[row]).ToArray();
                var branchEncoded = SelectRows(encoded, subgroup.Value);
                var branchLogs = SelectRows(predicted, subgroup.Value);
                var gatherIndices = BuildExpansionIndices(originalRows, tokenCount, subgroup.Key, durations);
                var mel = DecodeMel(Expand(branchEncoded, gatherIndices, originalRows.Length, subgroup.Key));
                outputs.Add(new OutputBucket(originalRows, mel, branchLogs));
            }
        }

        return AssembleOutput(outputs, durations, melLengths, tokens.Shape[1]);
    }

    /// <summary>Converts the duration-expanded mels with a caller-owned compatible vocoder.</summary>
    /// <remarks>The model does not dispose the supplied vocoder or invent untrained default waveform weights.</remarks>
    public IReadOnlyList<Tensor<T>> SynthesizeWithVocoder(Tensor<int> tokens, IReadOnlyList<int> tokenLengths,
        IVocoder<T> vocoder, double speakingRate = 1.0, int maximumMelFrames = 100000)
    {
        if (vocoder is null) throw new ArgumentNullException(nameof(vocoder));
        EnsureAlignedPath();
        if (vocoder.MelChannels != _melChannels || vocoder.SampleRate != SampleRate || vocoder.UpsampleFactor != _melHopSize)
            throw new ArgumentException("Vocoder mel channels, sample rate, and upsample factor must match this mel model.", nameof(vocoder));
        var result = SynthesizeMel(tokens, tokenLengths, speakingRate, maximumMelFrames);
        var waveforms = new Tensor<T>[result.MelLengths.Count];
        for (int row = 0; row < waveforms.Length; row++)
        {
            int frames = result.MelLengths[row];
            var mel = Engine.TensorSlice(result.MelSpectrogram, new[] { row, 0, 0 }, new[] { 1, frames, _melChannels });
            mel = Engine.TensorTranspose(Engine.Reshape(mel, new[] { frames, _melChannels }));
            var waveform = vocoder.MelToWaveform(mel);
            if (waveform.Length != checked(frames * _melHopSize))
                throw new InvalidOperationException("Vocoder output length does not match its declared upsample factor.");
            waveforms[row] = Engine.Reshape(waveform, new[] { waveform.Length });
        }
        return waveforms;
    }

    private ObjectiveResult ComputeObjective(Tensor<T> tokens, Tensor<T> mels, AlignedMelBatch<T> batch, bool includeOutput)
    {
        var melLoss = new Tensor<T>(new[] { 1 });
        var priorLoss = new Tensor<T>(new[] { 1 });
        var durationLoss = new Tensor<T>(new[] { 1 });
        var durations = new Tensor<int>(tokens.Shape.ToArray());
        var outputs = new List<OutputBucket>();
        long totalTokens = 0, totalFrames = 0;

        foreach (var group in MakeBuckets(batch.TokenLengths, batch.MelLengths))
        {
            int tokenCount = group.Key.Tokens;
            int frameCount = group.Key.Frames;
            int count = group.Value.Count;
            var encoded = EncodeTokens(SelectBucket(tokens, group.Value, tokenCount));
            var predictedLogs = PredictLogDurations(encoded);
            var prior = RequireMelPrior().Forward(encoded);
            var targets = SelectBucket(mels, group.Value, frameCount);
            int[][] aligned = batch.Durations is { } supplied
                ? group.Value.Select(row => supplied[row]).ToArray()
                : FindMonotonicDurations(prior, targets);

            var logTargets = new Tensor<T>(new[] { count, tokenCount, 1 });
            for (int row = 0; row < count; row++)
                for (int token = 0; token < tokenCount; token++)
                {
                    durations[group.Value[row], token] = aligned[row][token];
                    logTargets[row, token, 0] = NumOps.FromDouble(Math.Log(aligned[row][token]));
                }
            var expansion = BuildExpansionIndices(group.Value, tokenCount, frameCount, durations);
            var predictedMel = DecodeMel(Expand(encoded, expansion, count, frameCount));
            var expandedPrior = Expand(prior, expansion, count, frameCount);
            melLoss = Engine.TensorAdd(melLoss, SquaredErrorSum(predictedMel, targets));
            priorLoss = Engine.TensorAdd(priorLoss, SquaredErrorSum(expandedPrior, targets));
            durationLoss = Engine.TensorAdd(durationLoss, SquaredErrorSum(predictedLogs, logTargets));
            totalTokens += (long)count * tokenCount;
            totalFrames += (long)count * frameCount;
            if (includeOutput) outputs.Add(new OutputBucket(group.Value.ToArray(), predictedMel, predictedLogs));
        }

        melLoss = Engine.TensorMultiplyScalar(melLoss, NumOps.FromDouble(1.0 / (totalFrames * _melChannels)));
        priorLoss = Engine.TensorMultiplyScalar(priorLoss, NumOps.FromDouble(0.5 / (totalFrames * _melChannels)));
        durationLoss = Engine.TensorMultiplyScalar(durationLoss, NumOps.FromDouble(1.0 / totalTokens));
        var total = Engine.TensorAdd(Engine.TensorAdd(melLoss, priorLoss), durationLoss);
        var output = includeOutput ? AssembleOutput(outputs, durations, batch.MelLengths.ToArray(), tokens.Shape[1]) : null;
        return new ObjectiveResult(melLoss, priorLoss, durationLoss, total, output);
    }

    private Tensor<T> EncodeTokens(Tensor<T> tokens)
    {
        var embedding = _tokenEmbedding ?? throw new InvalidOperationException("No native token embedding is configured.");
        var current = embedding.Forward(tokens);
        for (int i = _encoderStart; i < _decoderStart; i++) current = Layers[i].Forward(current);
        if (current.Rank != 3 || current.Shape[2] != _encoderWidth)
            throw new InvalidOperationException("The registered text encoder did not preserve [batch, tokens, encoder width].");
        return current;
    }

    private Tensor<T> PredictLogDurations(Tensor<T> encoded)
    {
        var predictor = _durationPredictor ?? throw new InvalidOperationException("No native duration predictor is configured.");
        // Train duration weights without moving the encoder to satisfy a separate duration objective.
        return predictor.Forward(Engine.StopGradient(encoded));
    }

    private FullyConnectedLayer<T> RequireMelPrior() =>
        _melPrior ?? throw new InvalidOperationException("No native Gaussian mel prior is configured.");

    private Tensor<T> DecodeMel(Tensor<T> expanded)
    {
        var current = expanded;
        for (int i = _decoderStart; i < Layers.Count; i++) current = Layers[i].Forward(current);
        if (current.Rank != 3 || current.Shape[0] != expanded.Shape[0] || current.Shape[1] != expanded.Shape[1] || current.Shape[2] != _melChannels)
            throw new InvalidOperationException("The registered decoder did not preserve the aligned mel-frame length.");
        return current;
    }

    private Tensor<T> Expand(Tensor<T> source, Tensor<int> indices, int batchSize, int frames)
    {
        int width = source.Shape[2];
        var flat = Engine.Reshape(source, new[] { checked(source.Shape[0] * source.Shape[1]), width });
        return Engine.Reshape(Engine.TensorGather(flat, indices, axis: 0), new[] { batchSize, frames, width });
    }

    private static Tensor<int> BuildExpansionIndices(IReadOnlyList<int> rows, int tokens, int frames, Tensor<int> durations)
    {
        var indices = new Tensor<int>(new[] { checked(rows.Count * frames) });
        int position = 0;
        for (int row = 0; row < rows.Count; row++)
            for (int token = 0; token < tokens; token++)
                for (int repeat = 0; repeat < durations[rows[row], token]; repeat++)
                    indices[position++] = row * tokens + token;
        if (position != indices.Length) throw new InvalidOperationException("Alignment did not consume exactly the requested mel frames.");
        return indices;
    }

    private int[][] FindMonotonicDurations(Tensor<T> prior, Tensor<T> targets)
    {
        // Alignment is a discrete argmax; recording this score-only graph retains a quadratic
        // activation graph that has no derivative. Prior and decoder losses remain tape-tracked.
        using var noGrad = new NoGradScope<T>();
        int batch = prior.Shape[0], tokens = prior.Shape[1], frames = targets.Shape[1];
        var priorNorm = Engine.Reshape(Engine.ReduceSum(Engine.TensorMultiply(prior, prior), new[] { 2 }, false), new[] { batch, tokens, 1 });
        var targetNorm = Engine.Reshape(Engine.ReduceSum(Engine.TensorMultiply(targets, targets), new[] { 2 }, false), new[] { batch, 1, frames });
        var dot = Engine.BatchMatMul(prior, Engine.TensorPermute(targets, new[] { 0, 2, 1 }));
        var scores = Engine.TensorMultiplyScalar(Engine.TensorSubtract(Engine.TensorAdd(priorNorm, targetNorm),
            Engine.TensorMultiplyScalar(dot, NumOps.FromDouble(2))), NumOps.FromDouble(-0.5));
        var values = scores.AsSpan();
        var result = new int[batch][];
        for (int row = 0; row < batch; row++)
        {
            var previous = new double[tokens];
            var current = new double[tokens];
            var advance = new bool[checked(tokens * frames)];
            for (int token = 0; token < tokens; token++) previous[token] = double.NegativeInfinity;
            for (int frame = 0; frame < frames; frame++)
            {
                for (int token = 0; token < tokens; token++) current[token] = double.NegativeInfinity;
                int first = Math.Max(0, tokens - frames + frame);
                int last = Math.Min(tokens - 1, frame);
                for (int token = first; token <= last; token++)
                {
                    double score = NumOps.ToDouble(values[(row * tokens + token) * frames + frame]);
                    if (double.IsNaN(score) || double.IsInfinity(score))
                        throw new ArgumentException("Active mel targets and prior alignment scores must be finite.", nameof(targets));
                    double stay = previous[token];
                    double step = token > 0 ? previous[token - 1] : double.NegativeInfinity;
                    bool moves = step > stay;
                    current[token] = score + (frame == 0 && token == 0 ? 0 : moves ? step : stay);
                    advance[frame * tokens + token] = moves;
                }
                var swap = previous;
                previous = current;
                current = swap;
            }
            var durations = new int[tokens];
            int activeToken = tokens - 1;
            for (int frame = frames - 1; frame >= 0; frame--)
            {
                durations[activeToken]++;
                if (advance[frame * tokens + activeToken]) activeToken--;
            }
            if (activeToken != 0 || durations.Any(duration => duration <= 0))
                throw new InvalidOperationException("Monotonic alignment failed to cover every active token.");
            result[row] = durations;
        }
        return result;
    }

    private Tensor<T> SquaredErrorSum(Tensor<T> predicted, Tensor<T> expected)
    {
        var error = Engine.TensorSubtract(predicted, expected);
        return Engine.Reshape(Engine.ReduceSum(Engine.TensorMultiply(error, error), Enumerable.Range(0, error.Rank).ToArray(), false), new[] { 1 });
    }

    private Tensor<T> SelectRows(Tensor<T> input, IReadOnlyList<int> rows)
    {
        bool identity = rows.Count == input.Shape[0];
        for (int i = 0; identity && i < rows.Count; i++) identity = rows[i] == i;
        if (identity) return input;
        var indices = new Tensor<int>(new[] { rows.Count });
        for (int i = 0; i < rows.Count; i++) indices[i] = rows[i];
        return Engine.TensorGather(input, indices, axis: 0);
    }

    private Tensor<T> SelectBucket(Tensor<T> input, IReadOnlyList<int> rows, int length)
    {
        var selected = SelectRows(input, rows);
        if (selected.Shape[1] == length) return selected;
        var shape = selected.Shape.ToArray();
        shape[1] = length;
        return Engine.TensorSlice(selected, new int[selected.Rank], shape);
    }

    private AlignedMelOutput<T> AssembleOutput(List<OutputBucket> buckets, Tensor<int> durations, int[] melLengths, int paddedTokens)
    {
        int maxFrames = melLengths.Max();
        var mels = new List<Tensor<T>>();
        var logs = new List<Tensor<T>>();
        var inverse = new int[melLengths.Length];
        int offset = 0;
        foreach (var bucket in buckets)
        {
            var mel = bucket.Mels;
            if (mel.Shape[1] < maxFrames)
                mel = Engine.TensorConcatenate(new[] { mel, new Tensor<T>(new[] { mel.Shape[0], maxFrames - mel.Shape[1], _melChannels }) }, axis: 1);
            var log = Engine.Reshape(bucket.LogDurations, new[] { bucket.Rows.Length, bucket.LogDurations.Shape[1] });
            if (log.Shape[1] < paddedTokens)
                log = Engine.TensorConcatenate(new[] { log, new Tensor<T>(new[] { log.Shape[0], paddedTokens - log.Shape[1] }) }, axis: 1);
            mels.Add(mel);
            logs.Add(log);
            foreach (int originalRow in bucket.Rows) inverse[originalRow] = offset++;
        }
        var allMels = mels.Count == 1 ? mels[0] : Engine.TensorConcatenate(mels.ToArray(), axis: 0);
        var allLogs = logs.Count == 1 ? logs[0] : Engine.TensorConcatenate(logs.ToArray(), axis: 0);
        return new AlignedMelOutput<T>(SelectRows(allMels, inverse), SelectRows(allLogs, inverse), durations, melLengths);
    }

    private static Dictionary<(int Tokens, int Frames), List<int>> MakeBuckets(IReadOnlyList<int> tokenLengths, IReadOnlyList<int>? melLengths)
    {
        var groups = new Dictionary<(int Tokens, int Frames), List<int>>();
        for (int row = 0; row < tokenLengths.Count; row++)
        {
            var key = (tokenLengths[row], melLengths is null ? 0 : melLengths[row]);
            if (!groups.TryGetValue(key, out var rows)) groups.Add(key, rows = new List<int>());
            rows.Add(row);
        }
        return groups;
    }

    private Tensor<T> ConvertActiveTokens(Tensor<int> tokens, IReadOnlyList<int> lengths)
    {
        var numeric = new Tensor<T>(tokens.Shape.ToArray());
        for (int row = 0; row < lengths.Count; row++)
            for (int token = 0; token < lengths[row]; token++) numeric[row, token] = NumOps.FromDouble(tokens[row, token]);
        return numeric;
    }

    private void ValidateTokens(Tensor<int> tokens, IReadOnlyList<int> lengths)
    {
        EnsureAlignedPath();
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (lengths is null) throw new ArgumentNullException(nameof(lengths));
        if (tokens.Rank != 2 || tokens.Shape[0] <= 0 || lengths.Count != tokens.Shape[0])
            throw new ArgumentException("Tokens must have shape [batch, padded tokens] with one true length per row.", nameof(tokens));
        for (int row = 0; row < lengths.Count; row++)
        {
            if (lengths[row] <= 0 || lengths[row] > tokens.Shape[1]) throw new ArgumentOutOfRangeException(nameof(lengths));
            for (int token = 0; token < lengths[row]; token++)
                if (tokens[row, token] < 0 || tokens[row, token] >= _tokenVocabularySize)
                    throw new ArgumentOutOfRangeException(nameof(tokens), "Active token identifiers must be inside the configured vocabulary.");
        }
    }

    private void ValidateBatch(AlignedMelBatch<T> batch)
    {
        if (batch is null) throw new ArgumentNullException(nameof(batch));
        ValidateTokens(batch.Tokens, batch.TokenLengths);
        if (batch.TargetMels.Rank != 3 || batch.TargetMels.Shape[0] != batch.Tokens.Shape[0] || batch.TargetMels.Shape[2] != _melChannels
            || batch.MelLengths.Count != batch.Tokens.Shape[0])
            throw new ArgumentException("Mels must be [batch, padded frames, configured mel channels] with one length per row.", nameof(batch));
        if (batch.Durations is { } rows && rows.Length != batch.TokenLengths.Count)
            throw new ArgumentException("One explicit duration row is required per example.", nameof(batch));
        for (int row = 0; row < batch.TokenLengths.Count; row++)
        {
            int frames = batch.MelLengths[row], tokens = batch.TokenLengths[row];
            if (frames < tokens || frames > batch.TargetMels.Shape[1])
                throw new ArgumentException("Mel lengths must fit their tensor and provide at least one frame per active token.", nameof(batch));
            if (batch.Durations is not { } supplied) continue;
            if (supplied[row].Length != tokens || supplied[row].Any(duration => duration <= 0)
                || supplied[row].Sum(duration => (long)duration) != frames)
                throw new ArgumentException("Positive explicit durations must cover active tokens and sum exactly to the mel length.", nameof(batch));
        }
    }

    private void EnsureAlignedPath()
    {
        if (_alignedDisposed) throw new ObjectDisposedException(GetType().FullName);
        if (!SupportsAlignedMelSynthesis)
            throw new NotSupportedException("Aligned mel APIs require a native model with a registered encoder/decoder split; the frame API is unchanged.");
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_alignedDisposed) return;
        _alignedDisposed = true;
        try
        {
            if (disposing)
            {
                if (_tokenEmbedding is not null) AiDotNet.Helpers.DisposeOnceGuard.TryDispose(_tokenEmbedding);
                if (_durationPredictor is not null) AiDotNet.Helpers.DisposeOnceGuard.TryDispose(_durationPredictor);
                if (_melPrior is not null) AiDotNet.Helpers.DisposeOnceGuard.TryDispose(_melPrior);
            }
        }
        finally { base.Dispose(disposing); }
    }

    private sealed class OutputBucket
    {
        internal int[] Rows { get; }
        internal Tensor<T> Mels { get; }
        internal Tensor<T> LogDurations { get; }
        internal OutputBucket(int[] rows, Tensor<T> mels, Tensor<T> logs) { Rows = rows; Mels = mels; LogDurations = logs; }
    }

    private sealed class ObjectiveResult
    {
        internal Tensor<T> MelLoss { get; }
        internal Tensor<T> PriorLoss { get; }
        internal Tensor<T> DurationLoss { get; }
        internal Tensor<T> TotalLoss { get; }
        internal AlignedMelOutput<T>? Output { get; }
        internal ObjectiveResult(Tensor<T> melLoss, Tensor<T> priorLoss, Tensor<T> durationLoss, Tensor<T> totalLoss, AlignedMelOutput<T>? output)
        { MelLoss = melLoss; PriorLoss = priorLoss; DurationLoss = durationLoss; TotalLoss = totalLoss; Output = output; }
    }
}
