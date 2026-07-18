using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Serving.Benchmarks.Backends;

namespace AiDotNet.Serving.Benchmarks;

/// <summary>Entry point: parse options, build the workload, drive the backend, report the metrics.</summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        if (Array.Exists(args, a => a is "--help" or "-h" or "-?"))
        {
            Console.WriteLine(BenchmarkOptions.Usage());
            return 0;
        }

        // Comparison mode: `adnbench compare label=report.json ...` renders a side-by-side table across backends.
        if (args.Length >= 1 && args[0] == "compare")
        {
            var entries = new List<(string, BenchmarkReport)>();
            foreach (var arg in args.Skip(1))
            {
                int eq = arg.IndexOf('=');
                string label = eq > 0 ? arg.Substring(0, eq) : System.IO.Path.GetFileNameWithoutExtension(arg);
                string path = eq > 0 ? arg.Substring(eq + 1) : arg;
                if (!System.IO.File.Exists(path)) { Console.Error.WriteLine($"report not found: {path}"); return 2; }
                try { entries.Add((label, BenchmarkReport.FromJson(await System.IO.File.ReadAllTextAsync(path)))); }
                catch (Exception ex) { Console.Error.WriteLine($"bad report {path}: {ex.Message}"); return 2; }
            }
            if (entries.Count == 0) { Console.Error.WriteLine("compare: pass one or more [label=]report.json files"); return 2; }
            Console.WriteLine(BenchmarkReport.CompareConsole(entries));
            return 0;
        }

        BenchmarkOptions o;
        try
        {
            o = BenchmarkOptions.Parse(args);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("Argument error: " + ex.Message);
            Console.Error.WriteLine("Run with --help for usage.");
            return 2;
        }

        Console.WriteLine("adnbench - AiDotNet serving benchmark");
        Console.WriteLine(o.Describe());
        Console.WriteLine();

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

        var handler = new SocketsHttpHandler
        {
            MaxConnectionsPerServer = Math.Max(o.Concurrency, 16),
            PooledConnectionLifetime = TimeSpan.FromMinutes(10),
        };
        using var http = new HttpClient(handler)
        {
            BaseAddress = new Uri(o.BaseUrl),
            Timeout = TimeSpan.FromSeconds(o.TimeoutSeconds),
        };
        if (!string.IsNullOrEmpty(o.ApiKey))
            http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", o.ApiKey);

        IServingBackend backend;
        try
        {
            backend = o.Backend switch
            {
                "openai" => new OpenAiBackend(http, o),
                "aidotnet-native" => new AiDotNetNativeBackend(http, o),
                _ => throw new ArgumentException($"Unknown backend '{o.Backend}' (expected 'openai' or 'aidotnet-native')"),
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("Argument error: " + ex.Message);
            return 2;
        }

        try
        {
            if (o.Warmup > 0)
            {
                Console.WriteLine($"Warming up ({o.Warmup} requests)...");
                var warmSpecs = Workload.Build(o, o.Warmup, seedOffset: 9999);
                foreach (var s in warmSpecs) s.ArrivalOffsetMs = 0; // warmup ignores arrival pacing
                await Runner.RunAsync(backend, warmSpecs, o, cts.Token);
            }

            Console.WriteLine($"Running {o.NumPrompts} measured requests against {backend.Name}...");
            var specs = Workload.Build(o, o.NumPrompts);
            var (results, durationSec) = await Runner.RunAsync(backend, specs, o, cts.Token);

            var report = BenchmarkReport.Compute(o, results, durationSec);
            Console.WriteLine();
            Console.WriteLine(report.ToConsole());

            int shown = 0;
            foreach (var r in results)
            {
                if (!r.Success && shown < 3)
                {
                    Console.WriteLine($"  [sample error, req #{r.Index}] {r.Error}");
                    shown++;
                }
            }
            if (report.Failed > 3)
                Console.WriteLine($"  ... and {report.Failed - 3} more failures");

            if (!string.IsNullOrEmpty(o.OutputJson))
            {
                await System.IO.File.WriteAllTextAsync(o.OutputJson, report.ToJson(), cts.Token);
                Console.WriteLine($"\nReport written to {o.OutputJson}");
            }

            // Non-zero exit only if everything failed (useful in CI gates).
            return report.Completed == 0 ? 1 : 0;
        }
        catch (OperationCanceledException)
        {
            Console.Error.WriteLine("Cancelled.");
            return 130;
        }
    }
}
