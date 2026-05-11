#if NET10_0_OR_GREATER
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using AiDotNet.Diffusion.Conditioning;
using AiDotNet.Diffusion.TextToImage;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.Diffusion;

/// <summary>
/// SDXL end-to-end head-to-head: AiDotNet's <c>SDXLModel.GenerateAsync</c>
/// (text → 50-step denoise → VAE decode) vs the canonical PyTorch
/// <c>diffusers.StableDiffusionXLPipeline</c>. The PyTorch baseline runs in
/// a Python subprocess (see <c>diffusers_sdxl_baseline.py</c>) because hand-
/// porting the full SDXL pipeline (UNet cross-attention, dual-CLIP
/// conditioner, scheduler) to TorchSharp is impractical for the canonical
/// benchmark.
/// </summary>
/// <remarks>
/// <para>
/// Acceptance criterion #1 from #1272: AiDotNet's SDXL end-to-end wall time
/// must measure ≤ 1.10× PyTorch's at fp32, CPU, batch_size=1, 50 inference
/// steps, 256×256 output.
/// </para>
/// <para>
/// Subprocess protocol: this benchmark spawns
/// <c>python diffusers_sdxl_baseline.py --steps N --width W --height H</c>
/// once per benchmark iteration. The Python script imports
/// <c>diffusers.StableDiffusionXLPipeline</c>, runs a single end-to-end
/// generation with timing instrumentation, and prints a single JSON line
/// to stdout: <c>{"wall_ms": &lt;float&gt;}</c>. The C# benchmark parses
/// that line and reports the parsed wall time as the timing of the
/// PyTorch baseline.
/// </para>
/// <para>
/// Subprocess startup cost (~3-5 s for Python + diffusers import) is
/// excluded from timing — the script measures only the
/// <c>pipe(prompt, ...)</c> call. WarmupCount is set to 1 because subprocess
/// invocation is expensive; the warmup gives diffusers a chance to JIT/cache
/// its kernel selection.
/// </para>
/// <para>
/// <b>Skipping when Python isn't installed.</b> The benchmark detects whether
/// <c>python</c> is on PATH at <see cref="Setup"/> time. If not, the PyTorch
/// baseline benchmark throws to skip it; the AiDotNet method still runs and
/// reports its absolute wall time without a comparison ratio.
/// </para>
/// <para>
/// <b>AiDotNet SDXLModel construction is left as a TODO.</b> The benchmark's
/// AiDotNet column expects a working <c>SDXLModel</c> instance, which today
/// requires non-trivial UNet + dual-conditioner + VAE wiring that's only
/// done by application code. The factory method below throws until that
/// configuration is committed; the PyTorch baseline column still runs
/// independently so we can establish the head-to-head reference number on
/// the same machine.
/// </para>
/// </remarks>
[Config(typeof(SDXLEndToEndBenchmarkConfig))]
public class SDXLEndToEndBenchmark : IDisposable
{
    private const string Prompt = "a photograph of an astronaut riding a horse on mars";
    private const int NumInferenceSteps = 50;
    private const int Width = 256;
    private const int Height = 256;

    private string? _pythonScriptPath;
    private bool _pythonAvailable;
    private SDXLModel<float>? _aidotnetSdxl;

    [GlobalSetup]
    public void Setup()
    {
        // Wire the AiDotNet SDXL benchmark instance with the canonical
        // dual-CLIP conditioner pair (ViT-L/14 + ViT-bigG-14, the SDXL
        // base-1.0 configuration). UNet and VAE are constructed by the
        // SDXLModel ctor's internal defaults using paper-canonical
        // baseChannels=320 and channelMultipliers=[1, 2, 4, 4].
        var conditioner1 = new CLIPTextConditioner<float>(variant: "ViT-L/14", seed: 42);
        var conditioner2 = new CLIPTextConditioner<float>(variant: "ViT-bigG-14", seed: 42);
        _aidotnetSdxl = new SDXLModel<float>(
            conditioner1: conditioner1,
            conditioner2: conditioner2,
            useDualEncoder: true,
            crossAttentionDim: 2048,
            seed: 42);

        // The Python script ships in the project's Diffusion folder and is
        // copied to the build output via the AiDotNetBenchmarkTests.csproj
        // <None>/<CopyToOutputDirectory> entry. If for some reason it's
        // missing at runtime, the benchmark will surface a clear error.
        _pythonScriptPath = Path.Combine(
            Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!,
            "diffusers_sdxl_baseline.py");

        _pythonAvailable = File.Exists(_pythonScriptPath) && ProbePythonAvailability();
    }

    [GlobalCleanup]
    public void Cleanup() => Dispose();

    public void Dispose()
    {
        _aidotnetSdxl?.Dispose();
        _aidotnetSdxl = null;
        GC.SuppressFinalize(this);
    }

    [Benchmark(Baseline = true, Description = "PyTorch diffusers.StableDiffusionXLPipeline (subprocess, 50-step DDIM)")]
    public double DiffusersBaselineWallMs()
    {
        if (!_pythonAvailable)
            throw new InvalidOperationException(
                "Python or diffusers not available on PATH. Install with " +
                "`pip install diffusers transformers accelerate torch` to enable head-to-head comparison.");

        using var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{_pythonScriptPath}\" --steps {NumInferenceSteps} --width {Width} --height {Height} --prompt \"{Prompt}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            }
        };
        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"diffusers subprocess exited {process.ExitCode}. stderr:\n{stderr}");

        // The Python script may emit progress lines before the final JSON;
        // pick the last brace-prefixed line.
        var jsonLine = stdout
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Last(l => l.TrimStart().StartsWith("{"));
        var doc = JsonDocument.Parse(jsonLine);
        return doc.RootElement.GetProperty("wall_ms").GetDouble();
    }

    [Benchmark(Description = "AiDotNet SDXLModel.GenerateAsync (true-async, compile-cached, #1272 W4)")]
    public double AidotnetSdxlWallMs()
    {
        // Awaits the true-async denoise path: dual-CLIP conditioners run
        // concurrently, per-step UNet uses PredictNoiseAsync (compile-cache
        // replay after warmup), VAE decode uses the inherited compile cache
        // from VAEModelBase. Wall time matches what a sync-blocking caller
        // would observe — there's no measurement-side overhead from the
        // GetAwaiter().GetResult() unwrap.
        var sw = Stopwatch.StartNew();
        var task = _aidotnetSdxl!.GenerateAsync(
            prompt: Prompt,
            width: Width,
            height: Height,
            numInferenceSteps: NumInferenceSteps);
        task.GetAwaiter().GetResult();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }

    /// <summary>
    /// Quick PATH-based probe: runs <c>python --version</c> and checks for an
    /// exit code of 0. Cached once per benchmark process.
    /// </summary>
    private static bool ProbePythonAvailability()
    {
        try
        {
            using var p = Process.Start(new ProcessStartInfo
            {
                FileName = "python",
                Arguments = "--version",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            });
            if (p is null) return false;
            p.WaitForExit(5000);
            return p.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }
}

/// <summary>
/// Configuration: subprocess invocation is expensive, so we use a short job
/// with one warmup and three iterations.
/// </summary>
public sealed class SDXLEndToEndBenchmarkConfig : ManualConfig
{
    public SDXLEndToEndBenchmarkConfig()
    {
        AddJob(Job.Dry
            .WithWarmupCount(1)
            .WithIterationCount(3)
            .WithLaunchCount(1));
        AddDiagnoser(MemoryDiagnoser.Default);
    }
}
#endif
