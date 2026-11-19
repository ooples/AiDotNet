using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.Audio.Whisper;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.SpeechRecognition.Foundation;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio;

public abstract partial class AudioNeuralNetworkBase<T>
{
    /// <summary>
    /// When overridden to return a non-null duration in seconds, <see cref="PreprocessAudio"/> truncates
    /// incoming raw waveforms to <c>SampleRate * maxSeconds</c> samples using recorded engine operations
    /// instead of transforming through <see cref="MelSpec"/>.
    /// </summary>
    private protected virtual int? NativeWaveformMaxAudioLengthSeconds => null;

    /// <summary>
    /// Truncates a raw waveform tensor to at most <c>SampleRate * maxAudioLengthSeconds</c> samples
    /// via engine reshape/slice operations so GraphMode and device residency are preserved.
    /// </summary>
    private protected Tensor<T> TruncateAudioWaveform(Tensor<T> rawAudio, int maxAudioLengthSeconds)
    {
        if (rawAudio is null) throw new ArgumentNullException(nameof(rawAudio));
        int length = Math.Min(rawAudio.Length, Math.Max(0, SampleRate * maxAudioLengthSeconds));
        var flat = Engine.Reshape(rawAudio, new[] { rawAudio.Length });
        return rawAudio.Length <= length
            ? flat
            : Engine.TensorSlice(flat, new[] { 0 }, new[] { length });
    }

    /// <summary>
    /// Extracts utterance <paramref name="rowIndex"/> as a 1D <c>[samples]</c> tensor from a single or batched waveform.
    /// </summary>
    private protected Tensor<T> SliceBatchAudioRow(Tensor<T> input, int rowIndex, int rows, int samples)
    {
        if (rows == 1)
        {
            return input.Rank == 1 && input.Shape[0] == samples
                ? input
                : Engine.Reshape(input, new[] { samples });
        }

        return Engine.Reshape(
            Engine.TensorSlice(Engine.Reshape(input, new[] { rows, samples }), new[] { rowIndex, 0 }, new[] { 1, samples }),
            new[] { samples });
    }

    /// <summary>
    /// Extracts utterance <paramref name="rowIndex"/> from a single or batched waveform.
    /// </summary>
    private protected Tensor<T> SliceBatchAudioRow(Tensor<T> input, int rowIndex, int rows)
    {
        if (rows == 1) return input;
        int samples = input.Length / rows;
        return SliceBatchAudioRow(input, rowIndex, rows, samples);
    }

    /// <summary>
    /// Extracts target row <paramref name="rowIndex"/> from a single or batched target tensor, preserving trailing axes.
    /// </summary>
    private protected Tensor<T> SliceBatchTargetRow(
        Tensor<T> target,
        int rowIndex,
        int rows,
        Func<int, bool>? isDistributionWidth = null)
    {
        if (rows == 1) return target;
        int perRow = target.Length / rows;
        int width = target.Rank >= 2 ? target.Shape[^1] : 0;
        int[] rowShape;
        if (target.Rank >= 3)
        {
            rowShape = target.Shape.ToArray().Skip(1).ToArray();
        }
        else if (width > 0 && isDistributionWidth is not null && isDistributionWidth(width) && perRow % width == 0)
        {
            rowShape = new[] { perRow / width, width };
        }
        else
        {
            rowShape = new[] { perRow };
        }

        return Engine.Reshape(
            Engine.TensorSlice(Engine.Reshape(target, new[] { rows, perRow }), new[] { rowIndex, 0 }, new[] { 1, perRow }),
            rowShape);
    }

    /// <summary>
    /// Converts a host waveform tensor (or slice) into <c>double[]</c> outside compiled graph execution
    /// (for offline data preparation or decoding routines).
    /// </summary>
    private protected double[] ExtractHostAudioSamples(Tensor<T> waveform)
    {
        if (waveform is null) throw new ArgumentNullException(nameof(waveform));
        return waveform.Cast<double>().ToArray();
    }

    /// <summary>
    /// Converts a contiguous slice of a host waveform tensor into <c>double[]</c> outside compiled graph execution.
    /// </summary>
    private protected double[] ExtractHostAudioSamples(Tensor<T> waveform, int offset, int count)
    {
        if (waveform is null) throw new ArgumentNullException(nameof(waveform));
        if (offset == 0 && count == waveform.Length)
            return waveform.Cast<double>().ToArray();

        var slice = Engine.TensorSlice(
            Engine.Reshape(waveform, new[] { waveform.Length }),
            new[] { offset },
            new[] { count });
        return slice.Cast<double>().ToArray();
    }

    /// <summary>
    /// Runs ONNX inference, sequential <see cref="NeuralNetworkBase{T}.Layers"/> fallback, or per-utterance native
    /// waveform forward stacking (<c>[frames, ...]</c> for 1D input or <c>[batch, frames, ...]</c> for batched input).
    /// </summary>
    private protected Tensor<T> ForwardNativeWaveformBatch<TNet>(
        Tensor<T> input,
        TNet? network,
        Func<TNet, Tensor<T>, Tensor<T>> forwardSingleWaveform)
        where TNet : class
    {
        if (IsOnnxMode && OnnxEncoder is not null)
            return OnnxEncoder.Run(input);

        if (network is null)
        {
            var current = input;
            foreach (var layer in Layers)
                current = layer.Forward(current);
            return current;
        }

        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        int samples = input.Length / rows;
        if (rows == 1 && input.Rank == 1)
            return forwardSingleWaveform(network, SliceBatchAudioRow(input, 0, 1, samples));

        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
            outputs[r] = forwardSingleWaveform(network, SliceBatchAudioRow(input, r, rows, samples));
        return Engine.TensorStack(outputs, 0);
    }

    /// <summary>
    /// Parses a CTC/transducer target row as either collapsed greedy CTC frame distributions <c>[frames, vocabSize]</c>
    /// or strictly positive token IDs in <c>(0, vocabSize)</c>.
    /// </summary>
    private protected int[] ParseCtcOrTokenTargetLabels(
        Tensor<T> targetRow,
        int vocabSize,
        Func<Tensor<T>, IReadOnlyList<int>> greedyCtcDecode)
    {
        if (targetRow.Rank >= 2 && targetRow.Shape[^1] == vocabSize)
        {
            var frames = Engine.Reshape(targetRow, new[] { targetRow.Length / vocabSize, vocabSize });
            var decoded = greedyCtcDecode(frames);
            return decoded is int[] array ? array : decoded.ToArray();
        }

        double[] values = targetRow.Cast<double>().ToArray();
        var labels = new int[values.Length];
        for (int i = 0; i < labels.Length; i++)
        {
            double value = Math.Round(values[i]);
            if (!(value > 0 && value < vocabSize))
                throw new ArgumentException($"Label {value} is the blank or outside the vocabulary.", nameof(targetRow));
            labels[i] = (int)value;
        }
        return labels;
    }

    /// <summary>
    /// Parses a sequence-to-sequence / LLM-ASR target row as either per-step token distributions <c>[length, vocabSize]</c>
    /// or non-negative token IDs excluding <paramref name="forbiddenEndToken"/>.
    /// </summary>
    private protected int[] ParseVocabularyTargetTokens(
        Tensor<T> targetRow,
        int vocabSize,
        int forbiddenEndToken)
    {
        double[] values = targetRow.Cast<double>().ToArray();
        if (targetRow.Rank >= 2 && targetRow.Shape[^1] == vocabSize)
        {
            var tokens = new int[targetRow.Length / vocabSize];
            for (int r = 0; r < tokens.Length; r++)
            {
                int best = -1;
                int rowOffset = r * vocabSize;
                for (int v = 0; v < vocabSize; v++)
                {
                    if (v != forbiddenEndToken && (best < 0 || values[rowOffset + v] > values[rowOffset + best]))
                        best = v;
                }
                tokens[r] = best;
            }
            return tokens;
        }

        var ids = new int[values.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            double value = Math.Round(values[i]);
            if (!(value >= 0 && value < vocabSize) || (int)value == forbiddenEndToken)
                throw new ArgumentException($"Token {value} is outside the vocabulary or the end token.", nameof(targetRow));
            ids[i] = (int)value;
        }
        return ids;
    }

    /// <summary>
    /// Parses one integer token/label sequence per utterance in <paramref name="target"/> before entering a training trace.
    /// </summary>
    private protected int[][] ParseBatchTargetRows(
        Tensor<T> input,
        Tensor<T> target,
        Func<Tensor<T>, int[]> parseRow,
        Func<int, bool>? isDistributionWidth = null)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        var parsed = new int[rows][];
        for (int r = 0; r < rows; r++)
            parsed[r] = parseRow(SliceBatchTargetRow(target, r, rows, isDistributionWidth));
        return parsed;
    }

    /// <summary>
    /// Computes the batch mean of a per-utterance scalar loss tensor over <paramref name="batchAudio"/>.
    /// </summary>
    private protected Tensor<T> AverageWaveformBatchLoss(
        Tensor<T> batchAudio,
        Func<int, Tensor<T>, Tensor<T>> computeRowLoss)
    {
        int rows = batchAudio.Rank == 1 ? 1 : batchAudio.Shape[0];
        if (rows == 0)
            throw new ArgumentException("No audio.", nameof(batchAudio));

        int samples = batchAudio.Length / rows;
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var wave = SliceBatchAudioRow(batchAudio, r, rows, samples);
            var loss = computeRowLoss(r, wave);
            if (loss.Shape.Length != 1 || loss.Length != 1)
                loss = Engine.Reshape(loss, new[] { 1 });
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }

        if (total is null)
            throw new ArgumentException("No audio.", nameof(batchAudio));

        return rows == 1
            ? total
            : Engine.TensorMultiplyScalar(total, NumOps.FromDouble(1.0 / rows));
    }

    /// <summary>
    /// Executes one tape/compiled training step over single or batched waveforms with pre-parsed per-row integer targets.
    /// </summary>
    private protected bool TrainNativeWaveformBatch<TNet>(
        Tensor<T> input,
        Tensor<T> target,
        TNet? network,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        Func<TNet, Tensor<T>, int[]> parseTargetRow,
        Func<TNet, Tensor<T>, int[], Tensor<T>> computeRowLoss,
        Action<TNet>? beforeTrain = null,
        Action<TNet>? afterTrain = null,
        Func<int, bool>? isDistributionWidth = null)
        where TNet : class
    {
        if (network is null) return false;
        var labels = ParseBatchTargetRows(input, target, row => parseTargetRow(network, row), isDistributionWidth);
        beforeTrain?.Invoke(network);
        try
        {
            TrainWithCustomObjective(
                input,
                target,
                (stepInput, _) => AverageWaveformBatchLoss(stepInput, (r, wave) => computeRowLoss(network, wave, labels[r])),
                optimizer);
        }
        finally
        {
            afterTrain?.Invoke(network);
        }

        return true;
    }

    /// <summary>
    /// Evaluates the batch-averaged scalar training objective without gradient recording.
    /// </summary>
    private protected T EvaluateNativeWaveformBatchObjective<TNet>(
        Tensor<T> input,
        Tensor<T> target,
        TNet? network,
        Func<TNet, Tensor<T>, int[]> parseTargetRow,
        Func<TNet, Tensor<T>, int[], Tensor<T>> computeRowEvalLoss,
        Func<int, bool>? isDistributionWidth = null)
        where TNet : class
    {
        if (network is null)
            throw new NotSupportedException("The objective is defined on the native network.");

        using var _ = new NoGradScope<T>();
        var labels = ParseBatchTargetRows(input, target, row => parseTargetRow(network, row), isDistributionWidth);
        var averageLoss = AverageWaveformBatchLoss(input, (r, wave) => computeRowEvalLoss(network, wave, labels[r]));
        return averageLoss.AsSpan()[0];
    }

    /// <summary>
    /// Runs ONNX inference, sequential layer fallback, or batched Whisper start-sequence prompt logits.
    /// </summary>
    private protected Tensor<T> ForwardNativeWhisperBatch(
        Tensor<T> input,
        WhisperRecognizer<T>? whisper,
        string? language)
    {
        string resolvedLanguage = language ?? "en";
        return ForwardNativeWaveformBatch(
            input,
            whisper,
            (w, wave) => w.PromptLogits(w.WindowTensor(wave), resolvedLanguage));
    }

    /// <summary>
    /// Trains a native <see cref="WhisperRecognizer{T}"/> over single or batched waveforms with teacher-forced transcripts.
    /// </summary>
    private protected bool TrainNativeWhisperBatch(
        Tensor<T> input,
        Tensor<T> target,
        WhisperRecognizer<T>? whisper,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        string? language)
    {
        string resolvedLanguage = language ?? "en";
        return TrainNativeWaveformBatch(
            input,
            target,
            whisper,
            optimizer,
            (w, row) => w.TranscriptTokens(row),
            (w, wave, transcript) => w.TranscriptLoss(w.WindowTensor(wave), transcript, resolvedLanguage));
    }

    /// <summary>
    /// Evaluates the batch-averaged teacher-forced transcript cross-entropy on a native <see cref="WhisperRecognizer{T}"/>.
    /// </summary>
    private protected T EvaluateNativeWhisperBatchObjective(
        Tensor<T> input,
        Tensor<T> target,
        WhisperRecognizer<T>? whisper,
        string? language)
    {
        string resolvedLanguage = language ?? "en";
        return EvaluateNativeWaveformBatchObjective(
            input,
            target,
            whisper,
            (w, row) => w.TranscriptTokens(row),
            (w, wave, transcript) => w.TranscriptLoss(w.WindowTensor(wave), transcript, resolvedLanguage));
    }

    /// <summary>
    /// Runs native Whisper multi-window transcription with temperature fallback and optional word timestamps.
    /// </summary>
    private protected TranscriptionResult<T> TranscribeNativeWhisper(
        Tensor<T> audio,
        WhisperRecognizer<T>? whisper,
        string? language,
        bool wordTimestamps,
        IReadOnlyList<double> temperatures)
    {
        if (whisper is null)
            throw new NotSupportedException("Transcription runs on the native Whisper network.");

        return whisper.Transcribe(ExtractHostAudioSamples(audio), language, wordTimestamps, temperatures);
    }

    /// <summary>
    /// Detects the highest-probability Whisper language code from the first 30-second window.
    /// </summary>
    private protected static string DetectNativeWhisperLanguage(
        Tensor<T> audio,
        WhisperRecognizer<T>? whisper)
    {
        if (whisper is null)
            throw new NotSupportedException("Language detection runs on the native Whisper network.");

        var probs = whisper.DetectLanguage(audio);
        string bestKey = string.Empty;
        double bestProb = double.NegativeInfinity;
        foreach (var pair in probs)
        {
            if (pair.Value > bestProb)
            {
                bestProb = pair.Value;
                bestKey = pair.Key;
            }
        }
        return bestKey;
    }

    /// <summary>
    /// Computes typed <typeparamref name="T"/> Whisper language probabilities using bulk tensor casting.
    /// </summary>
    private protected static IReadOnlyDictionary<string, T> DetectNativeWhisperLanguageProbabilities(
        Tensor<T> audio,
        WhisperRecognizer<T>? whisper)
    {
        if (whisper is null)
            throw new NotSupportedException("Language detection runs on the native Whisper network.");

        var raw = whisper.DetectLanguage(audio);
        var keys = new string[raw.Count];
        var probs = new double[raw.Count];
        int index = 0;
        foreach (var pair in raw)
        {
            keys[index] = pair.Key;
            probs[index] = pair.Value;
            index++;
        }

        var cast = probs.Length == 0
            ? Array.Empty<T>()
            : new Tensor<double>(new[] { probs.Length }, new Vector<double>(probs)).Cast<T>().ToArray();
        var result = new Dictionary<string, T>(keys.Length, StringComparer.Ordinal);
        for (int i = 0; i < keys.Length; i++)
            result[keys[i]] = cast[i];
        return result;
    }

    /// <summary>
    /// Decodes Unicode code-point tokens into trimmed text, skipping control/null tokens (&lt;= 0).
    /// </summary>
    private protected static string DecodeUnicodeTokens(IReadOnlyList<int> tokens)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < tokens.Count; i++)
        {
            int t = tokens[i];
            if (t > 0 && t <= char.MaxValue)
            {
                if (t < 0xD800 || t > 0xDFFF)
                    sb.Append((char)t);
            }
            else if (t > char.MaxValue && t <= 0x10FFFF)
            {
                sb.Append(char.ConvertFromUtf32(t));
            }
        }
        return sb.ToString().Trim();
    }

    /// <summary>
    /// Builds a single utterance-spanning <see cref="TranscriptionSegment{T}"/> when <paramref name="text"/> is non-whitespace.
    /// </summary>
    private protected IReadOnlyList<TranscriptionSegment<T>> BuildSingleUtteranceSegment(
        string text,
        double durationSeconds,
        double confidence)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<TranscriptionSegment<T>>();

        return new[]
        {
            new TranscriptionSegment<T>
            {
                Text = text,
                StartTime = 0.0,
                EndTime = durationSeconds,
                Confidence = NumOps.FromDouble(confidence),
            },
        };
    }

    /// <summary>
    /// Simple CJK-vs-Latin heuristic language fallback shared by CTC recognizers without dedicated language heads.
    /// </summary>
    private protected static string ClassifyCjkOrDefaultLanguage(
        IReadOnlyList<int> tokens,
        IReadOnlyCollection<string> supportedLanguages,
        string defaultLanguage)
    {
        if (tokens.Count == 0) return defaultLanguage;
        int cjkCount = 0;
        int latinCount = 0;
        for (int i = 0; i < tokens.Count; i++)
        {
            int t = tokens[i];
            if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++;
            else if (t >= 0x41 && t <= 0x7A) latinCount++;
        }
        if (cjkCount > latinCount && supportedLanguages.Contains("zh"))
            return "zh";
        return defaultLanguage;
    }

    /// <summary>
    /// Evaluates the batch-averaged BEST-RQ / USM masked-prediction pre-training loss without gradient recording.
    /// </summary>
    private protected T EvaluateBestRqPretrainingLoss(
        Tensor<T> audio,
        BestRqNetwork<T>? network,
        Random random,
        string modelName)
    {
        if (network is null)
            throw new InvalidOperationException($"Pre-training runs on the native {modelName} network.");

        using var _ = new NoGradScope<T>();
        int rows = audio.Rank == 1 ? 1 : audio.Shape[0];
        if (rows == 0)
            throw new ArgumentException("No audio.", nameof(audio));
        int samples = audio.Length / rows;
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var loss = network.PretrainingLoss(ExtractHostAudioSamples(audio, r * samples, samples), training: false, random);
            if (loss.Shape.Length != 1 || loss.Length != 1)
                loss = Engine.Reshape(loss, new[] { 1 });
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }

        var mean = rows == 1
            ? (total ?? throw new ArgumentException("No audio.", nameof(audio)))
            : Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(audio)), NumOps.FromDouble(1.0 / rows));
        return mean.AsSpan()[0];
    }

    /// <summary>
    /// Runs one BEST-RQ / USM masked-prediction pre-training step over single or batched waveforms.
    /// </summary>
    private protected T PretrainBestRqBatch(
        Tensor<T> audio,
        BestRqNetwork<T>? network,
        Random random,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        string modelName)
    {
        if (network is null)
            throw new NotSupportedException($"Pre-training runs on the native {modelName} network.");

        int rows = audio.Rank == 1 ? 1 : audio.Shape[0];
        if (rows == 0)
            throw new ArgumentException("No audio.", nameof(audio));
        int samples = audio.Length / rows;
        var raw = new double[rows][];
        for (int r = 0; r < rows; r++)
            raw[r] = ExtractHostAudioSamples(audio, r * samples, samples);
        var prepared = network.PreparePretrainingBatch(raw, random);
        return TrainWithCustomObjective(
            prepared[0].MaskedFeatures,
            prepared[0].Scale,
            (_, _) => network.PretrainingLossFromPrepared(prepared, training: true),
            optimizer);
    }
}
