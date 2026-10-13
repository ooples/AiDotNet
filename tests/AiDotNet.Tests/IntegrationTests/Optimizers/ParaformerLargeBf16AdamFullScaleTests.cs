using System.Diagnostics;
using System.Reflection;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.AlibabaASR;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// #1767: ParaformerLarge at its default (paper) scale, ~661M parameters, trained through the eager per-step
/// tape path with an explicit BF16-moment Adam8Bit optimizer. That is the path the issue reported "Source
/// array was not long enough" from. <see cref="Adam8BitBf16TrainingTests"/> covers the BF16 recurrence on a
/// small network; only the full-scale model reproduces the arena state the failure was attributed to, so
/// this runs only on the nightly HeavyTimeout lane.
/// </summary>
/// <remarks>
/// <para>
/// Compilation is switched off so the fused compiled-training kernel cannot stand in for the eager
/// <c>Adam8BitOptimizer.Step</c>, and AIDOTNET_BF16_ADAM=1 is set so the base ladder agrees with the
/// explicit optimizer. Nothing is scaled down: options are the ParaformerLargeOptions defaults.
/// </para>
/// <para>
/// A failure here records every exception thrown on any thread during training, at the moment it is
/// thrown, with the full stack of the throwing thread and the throwing thread's arena state. The issue's
/// stack trace had been edited by hand, so the frame that actually throws is unknown; a first-chance
/// record is taken before any catch or rethrow can lose it. With AIDOTNET_DIAG_DIR set, the record is also
/// written to a file there, because the console log of this lane is too large to search reliably.
/// </para>
/// </remarks>
[Trait("Category", "HeavyTimeout")]
[Collection("FoundationScaleSerial")]
public class ParaformerLargeBf16AdamFullScaleTests : IDisposable
{
    private const int Frames = 32;
    private const int MaxSteps = 20;
    private const int MinSteps = 2;
    private const int MaxRecordedExceptions = 8;

    // The 120 s attribute is multiplied by AIDOTNET_HEAVY_TEST_TIMEOUT_SCALE (4 on the lane: 8 min), and
    // --blame-hang stops a test at 10 min. Training stops starting new steps after this budget so the
    // test finishes inside both, whatever the runner's speed.
    private static readonly TimeSpan TrainingBudget = TimeSpan.FromMinutes(5.5);

    private readonly ITestOutputHelper _output;
    private readonly object _recordLock = new();
    private readonly List<string> _firstChance = new();
    private volatile bool _recording;

    [ThreadStatic]
    private static bool t_inHandler;

    public ParaformerLargeBf16AdamFullScaleTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => ModelFamilyTestGcGate.ReclaimBetweenTests();

    [Fact(Timeout = 120000)]
    public async Task FullScale_Bf16Adam8Bit_EagerSteps_DoNotThrow()
    {
        await Task.Yield();
        var started = Stopwatch.StartNew();
        var originalCodec = TensorCodecOptions.Current;
        string? originalBf16 = Environment.GetEnvironmentVariable("AIDOTNET_BF16_ADAM");
        var log = new StringBuilder();
        AppDomain.CurrentDomain.FirstChanceException += OnFirstChanceException;
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_BF16_ADAM", "1");
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });

            var options = new ParaformerLargeOptions();
            var architecture = new NeuralNetworkArchitecture<float>(
                inputType: InputType.TwoDimensional,
                taskType: NeuralNetworkTaskType.SequenceToSequence,
                inputHeight: Frames, inputWidth: options.NumMels, inputDepth: 1,
                outputSize: options.VocabSize);
            var optimizer = new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    UseBFloat16MomentStorage = true,
                    UseAMSGrad = false,
                });
            using var model = new ParaformerLarge<float>(architecture, options, optimizer);

            var input = Random(1767, 1, Frames, options.NumMels);
            var predicted = model.Predict(input);
            var expected = Random(1768, predicted.Shape.ToArray());
            // Predict's derived weight packs would otherwise sit beside the training state for the whole test.
            AiDotNet.Tensors.Engines.InferenceWeightCache.InvalidateAll();

            // One full parameter vector is ~2.6 GB at this scale, so keep a strided sample rather than a copy.
            Vector<float>? initialParameters = model.GetParameters();
            long parameterCount = initialParameters.Length;
            var before = Sample(initialParameters);
            initialParameters = null;
            Line(log, $"ParaformerLarge<float>: {parameterCount:N0} parameters, input [{string.Join(",", input.Shape.ToArray())}], " +
                $"output [{string.Join(",", predicted.Shape.ToArray())}], streaming={model.StreamingTraining}, " +
                $"setup {started.Elapsed.TotalSeconds:F1}s, managed heap {GC.GetTotalMemory(false) / (1024 * 1024):N0} MB");
            Assert.True(parameterCount > 600_000_000, $"Expected the paper-scale model (~661M parameters), got {parameterCount:N0}.");

            int steps = 0;
            _recording = true;
            while (steps < MaxSteps && (steps < MinSteps || started.Elapsed < TrainingBudget))
            {
                var stepClock = Stopwatch.StartNew();
                try
                {
                    model.Train(input, expected);
                }
                catch (Exception ex)
                {
                    _recording = false;
                    Line(log, $"step {steps + 1} threw after {stepClock.Elapsed.TotalSeconds:F1}s:");
                    Line(log, ex.ToString());
                    Line(log, "arena at catch: " + DescribeArena());
                    throw;
                }
                steps++;
                Line(log, $"step {steps}: {stepClock.Elapsed.TotalSeconds:F1}s, loss {model.GetLastLoss()}, " +
                    $"managed heap {GC.GetTotalMemory(false) / (1024 * 1024):N0} MB");
            }
            _recording = false;

            Assert.True(steps >= MinSteps, $"Only {steps} step(s) ran.");
            var trained = model.GetParameters();
            for (int i = 0; i < trained.Length; i++)
                if (!float.IsFinite(trained[i])) Assert.Fail($"Parameter {i} became non-finite ({trained[i]}) after {steps} steps.");
            Assert.NotEqual(before, Sample(trained));
        }
        finally
        {
            _recording = false;
            AppDomain.CurrentDomain.FirstChanceException -= OnFirstChanceException;
            TensorCodecOptions.SetCurrent(originalCodec);
            Environment.SetEnvironmentVariable("AIDOTNET_BF16_ADAM", originalBf16);
            lock (_recordLock)
            {
                Line(log, $"first-chance exceptions recorded during training: {_firstChance.Count}");
                foreach (var record in _firstChance) Line(log, record);
            }
            Flush(log);
        }
    }

    private void OnFirstChanceException(object? sender, System.Runtime.ExceptionServices.FirstChanceExceptionEventArgs e)
    {
        if (!_recording || t_inHandler) return;
        if (e.Exception is OperationCanceledException) return;
        t_inHandler = true;
        try
        {
            lock (_recordLock)
            {
                if (_firstChance.Count >= MaxRecordedExceptions) return;
            }
            // Taken on the throwing thread, inside the first pass, so it still holds the throwing frame and
            // every caller, and the arena is the one that was live at the throw.
            string record =
                $"--- first-chance {e.Exception.GetType().FullName} on thread {Environment.CurrentManagedThreadId}: {e.Exception.Message}" +
                Environment.NewLine + new StackTrace(1, true) +
                "arena at throw: " + DescribeArena();
            lock (_recordLock)
            {
                if (_firstChance.Count < MaxRecordedExceptions) _firstChance.Add(record);
            }
        }
        catch (Exception)
        {
            // A diagnostic must never turn into a second failure.
        }
        finally
        {
            t_inHandler = false;
        }
    }

    /// <summary>
    /// The diagnostic AiDotNet.Tensors branch adds TensorArena.DescribeCurrentStateForDiagnostics; a
    /// released package does not have it, so it is looked up rather than called.
    /// </summary>
    private static string DescribeArena()
    {
        try
        {
            var method = typeof(TensorArena).GetMethod("DescribeCurrentStateForDiagnostics",
                BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Static);
            return method?.Invoke(null, null) as string ?? "(this AiDotNet.Tensors build has no arena diagnostic)";
        }
        catch (Exception ex)
        {
            return "(arena diagnostic failed: " + ex.GetType().Name + ")";
        }
    }

    private void Line(StringBuilder log, string text)
    {
        log.AppendLine(text);
        try { _output.WriteLine(text); } catch (InvalidOperationException) { }
    }

    private static void Flush(StringBuilder log)
    {
        string? dir = Environment.GetEnvironmentVariable("AIDOTNET_DIAG_DIR");
        if (string.IsNullOrWhiteSpace(dir)) return;
        try
        {
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, "paraformerlarge-1767-diagnostic.txt"), log.ToString());
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static float[] Sample(Vector<float> parameters)
    {
        const int SampleSize = 65_536;
        int stride = Math.Max(1, parameters.Length / SampleSize);
        var sample = new float[Math.Min(SampleSize, parameters.Length)];
        for (int i = 0; i < sample.Length; i++) sample[i] = parameters[i * stride];
        return sample;
    }

    private static Tensor<float> Random(int seed, params int[] shape)
    {
        int total = 1;
        foreach (int axis in shape) total *= axis;
        var data = new float[total];
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < total; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(shape, new Vector<float>(data));
    }
}
