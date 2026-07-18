using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Serving.Benchmarks;

/// <summary>
/// Configuration for a single serving benchmark run. Parsed from <c>--key value</c> CLI arguments.
/// </summary>
/// <remarks>
/// The harness is backend-agnostic on purpose: the same workload, pacing, and metric definitions
/// drive AiDotNet, vLLM, and TGI so the numbers are genuinely comparable. That is the whole point
/// of measuring before claiming to "exceed" anything.
/// </remarks>
public sealed class BenchmarkOptions
{
    /// <summary>Backend adapter: <c>openai</c> (vLLM / TGI / AiDotNet OpenAI route) or <c>aidotnet-native</c>.</summary>
    public string Backend { get; set; } = "openai";

    /// <summary>Base URL of the server under test, e.g. <c>http://localhost:8000</c>.</summary>
    public string BaseUrl { get; set; } = "http://localhost:8000";

    /// <summary>Model name/id to request.</summary>
    public string Model { get; set; } = "default";

    /// <summary>Optional bearer token (sent as <c>Authorization: Bearer ...</c>).</summary>
    public string? ApiKey { get; set; }

    /// <summary>OpenAI route: <c>chat</c> (/v1/chat/completions) or <c>completions</c> (/v1/completions).</summary>
    public string Mode { get; set; } = "chat";

    /// <summary>Total number of measured requests.</summary>
    public int NumPrompts { get; set; } = 200;

    /// <summary>Maximum number of in-flight requests.</summary>
    public int Concurrency { get; set; } = 32;

    /// <summary>Target arrival rate in requests/second. Infinity = dispatch as fast as concurrency allows.</summary>
    public double RequestRate { get; set; } = double.PositiveInfinity;

    /// <summary>Approximate prompt length in tokens (synthetic workload).</summary>
    public int InputTokens { get; set; } = 256;

    /// <summary>Maximum new tokens to request per generation.</summary>
    public int OutputTokens { get; set; } = 128;

    /// <summary>Randomize input/output lengths (0.5x–1.5x) around the targets.</summary>
    public bool RandomizeLengths { get; set; }

    /// <summary>Number of unmeasured warmup requests before the timed run.</summary>
    public int Warmup { get; set; } = 4;

    /// <summary>RNG seed for reproducible workloads and arrival schedules.</summary>
    public int Seed { get; set; } = 12345;

    /// <summary>Sampling temperature (0 = greedy).</summary>
    public double Temperature { get; set; }

    /// <summary>Goodput SLA: max acceptable TTFT in ms.</summary>
    public double SlaTtftMs { get; set; } = 1000;

    /// <summary>Goodput SLA: max acceptable mean inter-token latency (TPOT) in ms.</summary>
    public double SlaTpotMs { get; set; } = 50;

    /// <summary>Optional path to a newline-delimited prompt file (one prompt per line) for the openai backend.</summary>
    public string? DatasetPath { get; set; }

    /// <summary>Optional path to write the full report as JSON.</summary>
    public string? OutputJson { get; set; }

    /// <summary>Per-request HTTP timeout in seconds.</summary>
    public int TimeoutSeconds { get; set; } = 600;

    /// <summary>Vocabulary size used to synthesize token IDs for the aidotnet-native backend.</summary>
    public int Vocab { get; set; } = 32000;

    /// <summary>Parses CLI arguments into options. Throws <see cref="ArgumentException"/> on malformed input.</summary>
    public static BenchmarkOptions Parse(string[] args)
    {
        var o = new BenchmarkOptions();
        for (int i = 0; i < args.Length; i++)
        {
            string key = args[i];
            string Next()
            {
                if (i + 1 >= args.Length) throw new ArgumentException($"Missing value for {key}");
                return args[++i];
            }
            double D(string s) => double.Parse(s, CultureInfo.InvariantCulture);
            int I(string s) => int.Parse(s, CultureInfo.InvariantCulture);

            switch (key)
            {
                case "--backend": o.Backend = Next(); break;
                case "--base-url": o.BaseUrl = Next(); break;
                case "--model": o.Model = Next(); break;
                case "--api-key": o.ApiKey = Next(); break;
                case "--mode": o.Mode = Next(); break;
                case "--num-prompts": o.NumPrompts = I(Next()); break;
                case "--concurrency": o.Concurrency = I(Next()); break;
                case "--request-rate":
                    string rr = Next();
                    o.RequestRate = rr.Equals("inf", StringComparison.OrdinalIgnoreCase) ? double.PositiveInfinity : D(rr);
                    break;
                case "--input-tokens": o.InputTokens = I(Next()); break;
                case "--output-tokens": o.OutputTokens = I(Next()); break;
                case "--randomize-lengths": o.RandomizeLengths = true; break;
                case "--warmup": o.Warmup = I(Next()); break;
                case "--seed": o.Seed = I(Next()); break;
                case "--temperature": o.Temperature = D(Next()); break;
                case "--sla-ttft-ms": o.SlaTtftMs = D(Next()); break;
                case "--sla-tpot-ms": o.SlaTpotMs = D(Next()); break;
                case "--dataset": o.DatasetPath = Next(); break;
                case "--output-json": o.OutputJson = Next(); break;
                case "--timeout": o.TimeoutSeconds = I(Next()); break;
                case "--vocab": o.Vocab = I(Next()); break;
                default:
                    throw new ArgumentException($"Unknown argument: {key}");
            }
        }

        if (o.NumPrompts <= 0) throw new ArgumentException("--num-prompts must be > 0");
        if (o.Concurrency <= 0) throw new ArgumentException("--concurrency must be > 0");
        if (o.RequestRate <= 0) throw new ArgumentException("--request-rate must be > 0 (or 'inf')");
        return o;
    }

    /// <summary>Human-readable summary of the run configuration.</summary>
    public string Describe()
    {
        var sb = new StringBuilder();
        string rate = double.IsPositiveInfinity(RequestRate) ? "inf (concurrency-bound)" : RequestRate.ToString("0.##", CultureInfo.InvariantCulture) + " req/s";
        sb.AppendLine($"  backend        : {Backend}" + (Backend == "openai" ? $" ({Mode})" : ""));
        sb.AppendLine($"  base-url       : {BaseUrl}");
        sb.AppendLine($"  model          : {Model}");
        sb.AppendLine($"  num-prompts    : {NumPrompts}");
        sb.AppendLine($"  concurrency    : {Concurrency}");
        sb.AppendLine($"  request-rate   : {rate}");
        sb.AppendLine($"  input-tokens   : {InputTokens}{(RandomizeLengths ? " (randomized)" : "")}");
        sb.AppendLine($"  output-tokens  : {OutputTokens}{(RandomizeLengths ? " (randomized)" : "")}");
        sb.AppendLine($"  temperature    : {Temperature.ToString(CultureInfo.InvariantCulture)}");
        sb.AppendLine($"  warmup         : {Warmup}");
        sb.AppendLine($"  seed           : {Seed}");
        sb.Append($"  goodput SLA    : TTFT<={SlaTtftMs}ms, TPOT<={SlaTpotMs}ms");
        return sb.ToString();
    }

    /// <summary>Usage text for <c>--help</c>.</summary>
    public static string Usage() =>
        """
        adnbench - AiDotNet serving benchmark harness

        Measures throughput, TTFT, inter-token latency, end-to-end latency, and goodput
        for an LLM serving endpoint. Backend-agnostic so AiDotNet, vLLM, and TGI can be
        compared under an identical workload.

        USAGE:
          adnbench [options]

        BACKENDS:
          --backend openai            OpenAI-compatible SSE streaming (vLLM, TGI, AiDotNet /v1). Default.
          --backend aidotnet-native   AiDotNet current token-ID endpoint (api/inference/generate, non-streaming).

        COMMON:
          --base-url URL      Server base URL (default http://localhost:8000)
          --model NAME        Model id (default "default")
          --api-key KEY       Bearer token (optional)
          --mode chat|completions   OpenAI route (default chat)
          --num-prompts N     Measured requests (default 200)
          --concurrency N     Max in-flight requests (default 32)
          --request-rate R    Arrival rate req/s, or 'inf' (default inf)
          --input-tokens N    Approx prompt length (default 256)
          --output-tokens N   Max new tokens (default 128)
          --randomize-lengths Vary in/out lengths 0.5x-1.5x
          --temperature F     Sampling temperature (default 0)
          --warmup N          Unmeasured warmup requests (default 4)
          --seed N            RNG seed (default 12345)
          --sla-ttft-ms F     Goodput TTFT SLA (default 1000)
          --sla-tpot-ms F     Goodput TPOT SLA (default 50)
          --dataset PATH      Newline-delimited prompt file (openai backend)
          --output-json PATH  Write full report JSON
          --timeout SEC       Per-request HTTP timeout (default 600)
          --vocab N           Vocab size for native token synthesis (default 32000)

        EXAMPLES:
          # AiDotNet engine throughput today (native token-ID endpoint):
          adnbench --backend aidotnet-native --base-url http://localhost:5000 --model llama --num-prompts 500

          # Compare against vLLM (OpenAI-compatible):
          adnbench --base-url http://localhost:8000 --model meta-llama/Llama-3.1-8B --request-rate 10

          # Compare against TGI (OpenAI-compatible route):
          adnbench --base-url http://localhost:8080 --model tgi --mode chat --concurrency 64
        """;
}
