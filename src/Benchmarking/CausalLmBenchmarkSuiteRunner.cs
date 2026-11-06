using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Benchmarking;

/// <summary>
/// Executes full-context causal language-model benchmarks (<see cref="BenchmarkSuite.LAMBADA"/> and
/// <see cref="BenchmarkSuite.Perplexity"/>) behind <see cref="AiModelResult{T, TInput, TOutput}.EvaluateBenchmarksAsync(BenchmarkingOptions?, CancellationToken)"/>
/// with zero target visibility during inference, strict calibration/eval disjointness checks, unclamped metrics,
/// and raw <see cref="Stopwatch.GetTimestamp"/> latency reporting.
/// </summary>
internal static class CausalLmBenchmarkSuiteRunner
{
    public static Task<BenchmarkSuiteReport> RunAsync<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> model,
        BenchmarkSuite suite,
        BenchmarkingOptions options,
        CausalLmBenchmarkOptions lmOptions,
        CancellationToken cancellationToken)
    {
        return suite switch
        {
            BenchmarkSuite.LAMBADA => Task.Run(() => RunLambada(model, options, lmOptions, cancellationToken), cancellationToken),
            BenchmarkSuite.Perplexity => Task.Run(() => RunPerplexity(model, options, lmOptions, cancellationToken), cancellationToken),
            _ => throw new NotSupportedException($"Unsupported causal LM suite '{suite}'.")
        };
    }

    private static BenchmarkSuiteReport RunLambada<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> model,
        BenchmarkingOptions options,
        CausalLmBenchmarkOptions lmOptions,
        CancellationToken cancellationToken)
    {
        var tokenizer = model.RequireConfiguredTokenizer(nameof(BenchmarkSuite.LAMBADA));
        var allPassages = LoadLambadaPassages(lmOptions);
        if (allPassages.Count == 0)
        {
            throw new InvalidOperationException(
                "No valid LAMBADA passages found. Supply CausalLmBenchmarkOptions.LambadaDatasetPath or LambadaPassages.");
        }

        int targetCount = options.SampleSize ?? (options.CiMode ? Math.Min(25, allPassages.Count) : allPassages.Count);
        targetCount = Math.Max(1, Math.Min(targetCount, allPassages.Count));

        IReadOnlyList<string> selectedPassages;
        if (targetCount == allPassages.Count)
        {
            selectedPassages = allPassages;
        }
        else if (options.Seed.HasValue)
        {
            var rng = RandomHelper.CreateSeededRandom(options.Seed.Value);
            var indices = Enumerable.Range(0, allPassages.Count).ToArray();
            for (int i = 0; i < targetCount; i++)
            {
                int j = i + rng.Next(indices.Length - i);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            selectedPassages = indices.Take(targetCount).Select(idx => allPassages[idx]).ToArray();
        }
        else
        {
            selectedPassages = allPassages.Take(targetCount).ToArray();
        }

        int maxSubwords = Math.Max(1, lmOptions.MaxSubwordTokensPerWord);
        int warmupCount = Math.Max(0, Math.Min(lmOptions.WarmupItems, selectedPassages.Count));
        for (int w = 0; w < warmupCount; w++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string warmFull = selectedPassages[w];
            int warmSpace = warmFull.LastIndexOf(' ');
            if (warmSpace > 0)
            {
                _ = model.PredictNextWord(warmFull.Substring(0, warmSpace), maxSubwords);
            }
        }

        int evaluated = 0;
        int wholeWordCorrect = 0;
        int firstSubwordCorrect = 0;
        var itemLatenciesMs = new List<double>(selectedPassages.Count);
        long suiteStartTicks = Stopwatch.GetTimestamp();

        for (int i = 0; i < selectedPassages.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string fullText = selectedPassages[i].Trim();
            int lastSpace = fullText.LastIndexOf(' ');
            if (lastSpace <= 0 || lastSpace >= fullText.Length - 1)
                continue;

            // Strictly separate input prefix from held-out target word; only contextPrefix enters the model.
            string contextPrefix = fullText.Substring(0, lastSpace);
            string groundTruthRaw = fullText.Substring(lastSpace + 1);
            string groundTruthNorm = NormalizeWord(groundTruthRaw);
            if (groundTruthNorm.Length == 0)
                continue;

            long t0 = Stopwatch.GetTimestamp();
            var (predictedRaw, firstPredTokenId) = model.PredictNextWordWithFirstToken(contextPrefix, maxSubwords);
            long t1 = Stopwatch.GetTimestamp();

            double elapsedMs = (t1 - t0) * 1000.0 / Stopwatch.Frequency;
            itemLatenciesMs.Add(elapsedMs);
            evaluated++;

            string predictedNorm = NormalizeWord(predictedRaw);
            if (string.Equals(predictedNorm, groundTruthNorm, StringComparison.OrdinalIgnoreCase))
            {
                wholeWordCorrect++;
            }

            // First-subword top-1 check computed strictly outside inference.
            var fullTokenIds = model.EncodeCausalPrompt(tokenizer, fullText);
            var prefixTokenIds = model.EncodeCausalPrompt(tokenizer, contextPrefix.TrimEnd());
            if (prefixTokenIds.Count > 0 && fullTokenIds.Count > prefixTokenIds.Count)
            {
                int expectedFirstTargetToken = fullTokenIds[prefixTokenIds.Count];
                if (firstPredTokenId == expectedFirstTargetToken)
                    firstSubwordCorrect++;
            }
        }

        long suiteEndTicks = Stopwatch.GetTimestamp();
        double totalMs = (suiteEndTicks - suiteStartTicks) * 1000.0 / Stopwatch.Frequency;
        if (evaluated == 0)
            throw new InvalidOperationException("Zero valid multi-word passages could be evaluated in LAMBADA suite.");

        double accuracy = (double)wholeWordCorrect / evaluated;
        double subwordTop1 = (double)firstSubwordCorrect / evaluated;
        var (medianMs, p95Ms) = ComputePercentiles(itemLatenciesMs);

        var metrics = new List<BenchmarkMetricValue>
        {
            new() { Metric = BenchmarkMetric.TotalEvaluated, Value = evaluated },
            new() { Metric = BenchmarkMetric.CorrectCount, Value = wholeWordCorrect },
            new() { Metric = BenchmarkMetric.Accuracy, Value = accuracy },
            new() { Metric = BenchmarkMetric.SubwordTop1Accuracy, Value = subwordTop1 },
            new() { Metric = BenchmarkMetric.TotalDurationMilliseconds, Value = totalMs },
            new() { Metric = BenchmarkMetric.AverageTimePerItemMilliseconds, Value = itemLatenciesMs.Average() },
            new() { Metric = BenchmarkMetric.MedianTimePerItemMilliseconds, Value = medianMs },
            new() { Metric = BenchmarkMetric.P95TimePerItemMilliseconds, Value = p95Ms }
        };

        return new BenchmarkSuiteReport
        {
            Suite = BenchmarkSuite.LAMBADA,
            Kind = BenchmarkSuiteKind.DatasetSuite,
            Name = BenchmarkSuiteRegistry.GetDisplayName(BenchmarkSuite.LAMBADA),
            Status = BenchmarkExecutionStatus.Succeeded,
            Metrics = metrics,
            DataSelection = new BenchmarkDataSelectionSummary
            {
                Seed = options.Seed ?? 0,
                CiMode = options.CiMode,
                TestSamplesUsed = evaluated
            }
        };
    }

    private static BenchmarkSuiteReport RunPerplexity<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> model,
        BenchmarkingOptions options,
        CausalLmBenchmarkOptions lmOptions,
        CancellationToken cancellationToken)
    {
        var tokenizer = model.RequireConfiguredTokenizer(nameof(BenchmarkSuite.Perplexity));
        string corpusText = LoadPerplexityCorpus(lmOptions);
        if (string.IsNullOrEmpty(corpusText))
        {
            throw new InvalidOperationException(
                "Perplexity corpus is empty. Supply CausalLmBenchmarkOptions.PerplexityCorpusPath or PerplexityCorpusText.");
        }

        var allTokens = model.EncodeCausalPrompt(tokenizer, corpusText);
        if (allTokens.Count < 2)
            throw new InvalidOperationException("Perplexity evaluation requires at least 2 tokens in the corpus.");

        int evalStart = Math.Max(0, lmOptions.EvaluationTokenStart);
        if (evalStart >= allTokens.Count - 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(lmOptions),
                $"EvaluationTokenStart ({evalStart}) leaves fewer than 2 tokens in corpus of length {allTokens.Count}.");
        }

        int evalCount = lmOptions.EvaluationTokenCount.HasValue
            ? Math.Min(lmOptions.EvaluationTokenCount.Value, allTokens.Count - evalStart)
            : (allTokens.Count - evalStart);
        int evalEnd = evalStart + evalCount;
        if (evalCount < 2)
            throw new InvalidOperationException("Evaluation slice must contain at least 2 tokens.");

        // Enforce strict disjointness between calibration/training slice and held-out evaluation slice.
        if (lmOptions.CalibrationTokenRange is var (calStart, calCount) && calCount > 0)
        {
            int calEnd = calStart + calCount;
            if (evalStart < calEnd && calStart < evalEnd)
            {
                int overlapStart = Math.Max(evalStart, calStart);
                int overlapEnd = Math.Min(evalEnd, calEnd);
                throw new InvalidOperationException(
                    $"Calibration/eval token-slice overlap detected ([{overlapStart}..{overlapEnd}), {overlapEnd - overlapStart} overlapping tokens). " +
                    "Evaluation slices must be strictly disjoint from calibration/training ranges.");
            }
        }

        int windowLen = Math.Max(2, lmOptions.ContextWindowLength);
        var windows = new List<int[]>();
        for (int pos = evalStart; pos + 1 < evalEnd; pos += windowLen)
        {
            int len = Math.Min(windowLen, evalEnd - pos);
            if (len < 2)
                break;

            var win = new int[len];
            for (int i = 0; i < len; i++)
                win[i] = allTokens[pos + i];
            windows.Add(win);
        }

        if (options.SampleSize.HasValue && options.SampleSize.Value > 0 && options.SampleSize.Value < windows.Count)
        {
            windows = windows.Take(options.SampleSize.Value).ToList();
        }

        int warmupCount = Math.Max(0, Math.Min(lmOptions.WarmupItems, windows.Count));
        for (int w = 0; w < warmupCount; w++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _ = ForwardWindowLogits(model, windows[w]);
        }

        double totalNats = 0.0;
        long totalScoredTokens = 0;
        long top1Correct = 0;
        long totalUtf8Bytes = 0;
        var windowLatenciesMs = new List<double>(windows.Count);
        long suiteStartTicks = Stopwatch.GetTimestamp();

        for (int w = 0; w < windows.Count; w++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var win = windows[w];

            long t0 = Stopwatch.GetTimestamp();
            double[,] logits = ForwardWindowLogits(model, win);
            long t1 = Stopwatch.GetTimestamp();
            windowLatenciesMs.Add((t1 - t0) * 1000.0 / Stopwatch.Frequency);

            int seqLen = logits.GetLength(0);
            int vocab = logits.GetLength(1);

            for (int t = 0; t < seqLen - 1; t++)
            {
                int targetId = win[t + 1];
                if (targetId < 0 || targetId >= vocab)
                {
                    throw new InvalidOperationException(
                        $"Target token ID {targetId} is outside model logits vocabulary [0, {vocab}).");
                }

                double maxL = double.NegativeInfinity;
                int bestIdx = 0;
                for (int v = 0; v < vocab; v++)
                {
                    double l = logits[t, v];
                    if (l > maxL)
                    {
                        maxL = l;
                        bestIdx = v;
                    }
                }

                double sumExp = 0.0;
                for (int v = 0; v < vocab; v++)
                    sumExp += Math.Exp(logits[t, v] - maxL);

                double logZ = maxL + Math.Log(sumExp);
                double nll = logZ - logits[t, targetId];
                totalNats += nll;
                totalScoredTokens++;
                if (bestIdx == targetId)
                    top1Correct++;
            }

            // Count UTF-8 bytes of target tokens in this window for bits-per-byte.
            var targetSlice = new List<int>(win.Length - 1);
            for (int t = 1; t < win.Length; t++)
                targetSlice.Add(win[t]);
            string decodedTargets = tokenizer.Decode(targetSlice, skipSpecialTokens: false);
            totalUtf8Bytes += Encoding.UTF8.GetByteCount(decodedTargets);
        }

        long suiteEndTicks = Stopwatch.GetTimestamp();
        double totalMs = (suiteEndTicks - suiteStartTicks) * 1000.0 / Stopwatch.Frequency;
        if (totalScoredTokens == 0)
            throw new InvalidOperationException("Zero target tokens were scored in Perplexity suite.");

        double crossEntropy = totalNats / totalScoredTokens;
        double perplexity = Math.Exp(crossEntropy); // Unclamped exact exp(NLL)
        double bpb = totalUtf8Bytes > 0 ? totalNats / (Math.Log(2.0) * totalUtf8Bytes) : double.NaN;
        double top1Acc = (double)top1Correct / totalScoredTokens;
        var (medianMs, p95Ms) = ComputePercentiles(windowLatenciesMs);

        var metrics = new List<BenchmarkMetricValue>
        {
            new() { Metric = BenchmarkMetric.TotalEvaluated, Value = windows.Count },
            new() { Metric = BenchmarkMetric.EvaluatedTokens, Value = totalScoredTokens },
            new() { Metric = BenchmarkMetric.CrossEntropyLoss, Value = crossEntropy },
            new() { Metric = BenchmarkMetric.Perplexity, Value = perplexity },
            new() { Metric = BenchmarkMetric.BitsPerByte, Value = bpb },
            new() { Metric = BenchmarkMetric.SubwordTop1Accuracy, Value = top1Acc },
            new() { Metric = BenchmarkMetric.Accuracy, Value = top1Acc },
            new() { Metric = BenchmarkMetric.TotalDurationMilliseconds, Value = totalMs },
            new() { Metric = BenchmarkMetric.AverageTimePerItemMilliseconds, Value = windowLatenciesMs.Average() },
            new() { Metric = BenchmarkMetric.MedianTimePerItemMilliseconds, Value = medianMs },
            new() { Metric = BenchmarkMetric.P95TimePerItemMilliseconds, Value = p95Ms }
        };

        return new BenchmarkSuiteReport
        {
            Suite = BenchmarkSuite.Perplexity,
            Kind = BenchmarkSuiteKind.DatasetSuite,
            Name = BenchmarkSuiteRegistry.GetDisplayName(BenchmarkSuite.Perplexity),
            Status = BenchmarkExecutionStatus.Succeeded,
            Metrics = metrics,
            DataSelection = new BenchmarkDataSelectionSummary
            {
                Seed = options.Seed ?? 0,
                CiMode = options.CiMode,
                TrainSamplesUsed = lmOptions.CalibrationTokenRange?.Count ?? 0,
                TestSamplesUsed = windows.Count
            }
        };
    }

    private static double[,] ForwardWindowLogits<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> result,
        int[] windowTokens)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int seqLen = windowTokens.Length;

        if (result.Model is ILanguageModel<T> lm)
        {
            var input = new Tensor<T>(new[] { 1, seqLen });
            for (int i = 0; i < seqLen; i++)
                input[0, i] = numOps.FromDouble(windowTokens[i]);
            return Extract2DLogits(lm.ForwardLogits(input), numOps);
        }

        if (result.Model is NeuralNetworkBase<T> network)
        {
            var input = new Tensor<T>(new[] { 1, seqLen });
            for (int i = 0; i < seqLen; i++)
                input[0, i] = numOps.FromDouble(windowTokens[i]);
            network.ResetState();
            return Extract2DLogits(network.Predict(input), numOps);
        }

        if (result.Model is IIncrementalCausalLanguageModel<T> incLm)
        {
            try
            {
                incLm.ResetCache();
                var first = incLm.StartSequence(new[] { windowTokens[0] });
                int vocab = first.Length;
                var mat = new double[seqLen, vocab];
                for (int v = 0; v < vocab; v++)
                    mat[0, v] = numOps.ToDouble(first[v]);

                for (int t = 1; t < seqLen; t++)
                {
                    var next = incLm.AppendToken(windowTokens[t]);
                    for (int v = 0; v < vocab; v++)
                        mat[t, v] = numOps.ToDouble(next[v]);
                }

                return mat;
            }
            finally
            {
                incLm.ResetCache();
            }
        }

        if (result.Model is ICausalLanguageModel<T> causalLm)
        {
            int vocab = causalLm.VocabularySize;
            var mat = new double[seqLen, vocab];
            var prefix = new List<int>(seqLen);
            for (int t = 0; t < seqLen; t++)
            {
                prefix.Add(windowTokens[t]);
                var vec = causalLm.NextTokenLogits(prefix);
                for (int v = 0; v < vocab; v++)
                    mat[t, v] = numOps.ToDouble(vec[v]);
            }

            return mat;
        }

        throw new InvalidOperationException(
            $"Model '{result.Model?.GetType().Name}' does not implement ILanguageModel<T>, NeuralNetworkBase<T>, or ICausalLanguageModel<T>.");
    }

    private static double[,] Extract2DLogits<T>(Tensor<T> tensor, AiDotNet.Tensors.Interfaces.INumericOperations<T> numOps)
    {
        var shape = tensor.Shape;
        if (shape.Length == 3)
        {
            int seqLen = shape[1];
            int vocab = shape[2];
            var mat = new double[seqLen, vocab];
            for (int t = 0; t < seqLen; t++)
                for (int v = 0; v < vocab; v++)
                    mat[t, v] = numOps.ToDouble(tensor[0, t, v]);
            return mat;
        }

        if (shape.Length == 2)
        {
            int seqLen = shape[0];
            int vocab = shape[1];
            var mat = new double[seqLen, vocab];
            for (int t = 0; t < seqLen; t++)
                for (int v = 0; v < vocab; v++)
                    mat[t, v] = numOps.ToDouble(tensor[t, v]);
            return mat;
        }

        throw new InvalidOperationException($"Expected rank-2 or rank-3 logits tensor; got [{string.Join(",", shape)}].");
    }

    private static List<string> LoadLambadaPassages(CausalLmBenchmarkOptions lmOptions)
    {
        var passages = new List<string>();
        if (lmOptions.LambadaPassages is { Count: > 0 })
        {
            foreach (var p in lmOptions.LambadaPassages.Where(p => !string.IsNullOrWhiteSpace(p) && p.Trim().IndexOf(' ') > 0))
            {
                passages.Add(p.Trim());
            }

            return passages;
        }

        if (string.IsNullOrWhiteSpace(lmOptions.LambadaDatasetPath))
            return passages;

        if (!File.Exists(lmOptions.LambadaDatasetPath))
            throw new FileNotFoundException($"LAMBADA dataset file not found: {lmOptions.LambadaDatasetPath}", lmOptions.LambadaDatasetPath);

        foreach (var trimmed in File.ReadLines(lmOptions.LambadaDatasetPath)
            .Where(line => !string.IsNullOrWhiteSpace(line)).Select(line => line.Trim()))
        {
            if (trimmed.StartsWith("{", StringComparison.Ordinal))
            {
                var obj = JObject.Parse(trimmed);
                var text = obj["text"]?.Value<string>();
                if (!string.IsNullOrWhiteSpace(text) && text!.Trim().IndexOf(' ') > 0)
                    passages.Add(text.Trim());
            }
            else if (trimmed.IndexOf(' ') > 0)
            {
                passages.Add(trimmed);
            }
        }

        return passages;
    }

    private static string LoadPerplexityCorpus(CausalLmBenchmarkOptions lmOptions)
    {
        if (!string.IsNullOrEmpty(lmOptions.PerplexityCorpusText))
            return lmOptions.PerplexityCorpusText!;

        if (!string.IsNullOrWhiteSpace(lmOptions.PerplexityCorpusPath))
        {
            if (!File.Exists(lmOptions.PerplexityCorpusPath))
                throw new FileNotFoundException($"Perplexity corpus file not found: {lmOptions.PerplexityCorpusPath}", lmOptions.PerplexityCorpusPath);
            return File.ReadAllText(lmOptions.PerplexityCorpusPath);
        }

        return string.Empty;
    }

    private static string NormalizeWord(string word)
    {
        if (string.IsNullOrWhiteSpace(word))
            return string.Empty;

        int start = 0;
        int end = word.Length - 1;
        while (start <= end && IsPunctOrSpace(word[start]))
            start++;
        while (end >= start && IsPunctOrSpace(word[end]))
            end--;

        return start <= end ? word.Substring(start, end - start + 1).ToLowerInvariant() : string.Empty;
    }

    private static bool IsPunctOrSpace(char c) =>
        char.IsWhiteSpace(c)
        || c is '.' or ',' or '!' or '?' or ';' or ':' or '"' or '\'' or '\u201c' or '\u201d' or '\u2018' or '\u2019'
            or '(' or ')' or '[' or ']' or '{' or '}';

    private static (double Median, double P95) ComputePercentiles(List<double> values)
    {
        if (values.Count == 0)
            return (0.0, 0.0);

        var sorted = values.OrderBy(v => v).ToArray();
        int n = sorted.Length;
        double median = (n & 1) == 1
            ? sorted[n / 2]
            : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
        int p95Idx = Math.Min(n - 1, (int)Math.Ceiling(0.95 * n) - 1);
        return (median, sorted[Math.Max(0, p95Idx)]);
    }
}
