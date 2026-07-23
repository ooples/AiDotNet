using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace AiDotNet.Serving.Benchmarks;

/// <summary>A single request to issue: either prompt text (openai) or prompt token IDs (native).</summary>
public sealed class RequestSpec
{
    public required int Index { get; init; }

    /// <summary>Prompt text for the openai backend (null for native).</summary>
    public string? PromptText { get; init; }

    /// <summary>Prompt token IDs for the aidotnet-native backend (null for openai).</summary>
    public int[]? PromptTokenIds { get; init; }

    /// <summary>Max new tokens to generate.</summary>
    public required int MaxTokens { get; init; }

    /// <summary>Approximate prompt token count (used for throughput accounting when the server omits usage).</summary>
    public required int ApproxPromptTokens { get; init; }

    /// <summary>When to dispatch this request, relative to run start, in ms (arrival schedule).</summary>
    public double ArrivalOffsetMs { get; set; }
}

/// <summary>Builds reproducible synthetic (or dataset-backed) workloads and arrival schedules.</summary>
public static class Workload
{
    // A small fixed word pool so that, for typical subword tokenizers, one word ≈ one token
    // (a documented approximation; use --dataset for real text distributions).
    private static readonly string[] Words =
    {
        "the","model","serves","tokens","quickly","under","load","while","memory","stays",
        "bounded","and","latency","remains","low","across","many","concurrent","requests","today",
        "system","cache","paged","attention","batch","stream","kernel","tensor","vector","matrix",
        "compute","device","throughput","decode","prefill","context","window","sequence","prompt","reply",
    };

    /// <summary>Builds <paramref name="count"/> request specs plus their arrival schedule.</summary>
    public static List<RequestSpec> Build(BenchmarkOptions o, int count, int seedOffset = 0)
    {
        var rng = new Random(o.Seed + seedOffset);
        string[]? dataset = LoadDataset(o);

        var specs = new List<RequestSpec>(count);
        double arrival = 0.0;

        for (int i = 0; i < count; i++)
        {
            int inTok = o.InputTokens;
            int outTok = o.OutputTokens;
            if (o.RandomizeLengths)
            {
                inTok = Math.Max(1, (int)(o.InputTokens * (0.5 + rng.NextDouble())));
                outTok = Math.Max(1, (int)(o.OutputTokens * (0.5 + rng.NextDouble())));
            }

            RequestSpec spec;
            if (o.Backend == "aidotnet-native")
            {
                var ids = new int[inTok];
                for (int k = 0; k < inTok; k++) ids[k] = 1 + rng.Next(o.Vocab - 1); // avoid id 0 (often padding)
                spec = new RequestSpec
                {
                    Index = i,
                    PromptTokenIds = ids,
                    MaxTokens = outTok,
                    ApproxPromptTokens = inTok,
                };
            }
            else
            {
                string prompt = dataset is { Length: > 0 }
                    ? dataset[i % dataset.Length]
                    : SynthPrompt(rng, inTok);
                spec = new RequestSpec
                {
                    Index = i,
                    PromptText = prompt,
                    MaxTokens = outTok,
                    ApproxPromptTokens = inTok,
                };
            }

            // Poisson arrivals: exponential inter-arrival gaps when a finite rate is set.
            if (!double.IsPositiveInfinity(o.RequestRate))
            {
                double u = 1.0 - rng.NextDouble(); // (0,1]
                double gapMs = -Math.Log(u) / o.RequestRate * 1000.0;
                arrival += gapMs;
                spec.ArrivalOffsetMs = arrival;
            }
            else
            {
                spec.ArrivalOffsetMs = 0.0; // dispatch-bound: fire immediately, bounded by concurrency
            }

            specs.Add(spec);
        }

        return specs;
    }

    private static string SynthPrompt(Random rng, int approxTokens)
    {
        var sb = new StringBuilder(approxTokens * 6);
        for (int i = 0; i < approxTokens; i++)
        {
            if (i > 0) sb.Append(' ');
            sb.Append(Words[rng.Next(Words.Length)]);
        }
        return sb.ToString();
    }

    private static string[]? LoadDataset(BenchmarkOptions o)
    {
        if (string.IsNullOrEmpty(o.DatasetPath)) return null;
        if (!File.Exists(o.DatasetPath))
            throw new FileNotFoundException($"Dataset file not found: {o.DatasetPath}");
        var lines = new List<string>();
        foreach (var line in File.ReadLines(o.DatasetPath))
        {
            string t = line.Trim();
            if (t.Length > 0) lines.Add(t);
        }
        return lines.Count > 0 ? lines.ToArray() : null;
    }
}
