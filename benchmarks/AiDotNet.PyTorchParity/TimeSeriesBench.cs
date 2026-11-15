using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;

namespace AiDotNet.PyTorchParity;

/// <summary>
/// The time-series forecaster rows (#2561): AiDotNet's forecasters trained through their own <c>Train</c> against a
/// PyTorch twin with the same architecture, loss, optimizer, window and batch (<c>pytorch/timeseries.py</c>, run with
/// <c>--compile</c> for <c>torch.compile</c>). This side writes the series as <c>series.csv</c> so both sides train on
/// exactly the same numbers. Each row records the wall time of every epoch (read from the training callback, so it is
/// the model's own loop, batch building and all) and the epoch losses, so a faster epoch that learned less shows up.
/// </summary>
internal static class TimeSeriesBench
{
    /// <summary>
    /// Writes the shared series, trains and times every forecaster row on the current engine (the global
    /// <c>--device</c> handling has already chosen CPU or CUDA), and writes the AiDotNet report beside the series.
    /// </summary>
    /// <param name="outputDirectory">Where the series and the report go.</param>
    /// <param name="epochs">Epochs per row. Epoch 0 carries compilation and warm-up and is excluded from the median.</param>
    /// <param name="seriesLength">Length of the synthetic series.</param>
    /// <param name="seed">The data and model seed.</param>
    /// <param name="only">Row names to run; empty runs every row.</param>
    /// <returns>0 on success.</returns>
    public static int Run(string outputDirectory, int epochs, int seriesLength, int seed, IReadOnlyCollection<string> only)
    {
        if (epochs < 2)
            throw new ArgumentOutOfRangeException(nameof(epochs), epochs, "Need at least 2 epochs: epoch 0 is warm-up.");

        Directory.CreateDirectory(outputDirectory);
        var series = WriteSeries(outputDirectory, seriesLength, seed);
        string device = global::BenchStats.DeviceName;

        var all = new (string Name, Func<Dictionary<string, object>> Measure)[]
        {
            ("nbeats", () => NBeatsRow(series, epochs, seed, device)),
        };

        var rows = new List<Dictionary<string, object>>();
        foreach (var (name, measure) in all)
        {
            if (only.Count > 0 && !only.Contains(name))
                continue;
            rows.Add(measure());
        }

        var report = new Dictionary<string, object>
        {
            ["framework"] = "AiDotNet",
            ["device"] = device,
            ["cpu_threads"] = Environment.GetEnvironmentVariable("AIDOTNET_CPU_THREADS") ?? "default",
            ["series_length"] = seriesLength,
            ["epochs"] = epochs,
            ["rows"] = rows,
        };
        string path = Path.Combine(outputDirectory, $"aidotnet_timeseries_{device}.json");
        File.WriteAllText(path, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"[timeseries] report written to {path}");
        return 0;
    }

    /// <summary>
    /// A daily-plus-weekly seasonal series with a slow trend and Gaussian noise. The values are written with "R"
    /// round-trip formatting so the Python side reads bit-identical doubles.
    /// </summary>
    private static double[] WriteSeries(string directory, int length, int seed)
    {
        var random = new Random(seed);
        var values = new double[length];
        for (int t = 0; t < length; t++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double noise = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            values[t] = Math.Sin(2.0 * Math.PI * t / 24.0) + 0.5 * Math.Sin(2.0 * Math.PI * t / 168.0)
                + 0.0005 * t + 0.1 * noise;
        }

        var csv = new StringBuilder("y\n");
        foreach (double v in values)
            csv.Append(v.ToString("R", CultureInfo.InvariantCulture)).Append('\n');
        File.WriteAllText(Path.Combine(directory, "series.csv"), csv.ToString());
        return values;
    }

    /// <summary>
    /// N-BEATS at its library defaults (Oreshkin et al. 2020 interpretable configuration as AiDotNet ships it):
    /// 30 single-block stacks, 4 hidden layers of 512, polynomial basis of degree 2, lookback 10, horizon 5,
    /// batch 1024, Adam lr 1e-3 with global-norm clip 1.0, MSE loss. pytorch/timeseries.py builds the same network.
    /// </summary>
    private static Dictionary<string, object> NBeatsRow(double[] series, int epochs, int seed, string device)
    {
        var options = new NBEATSModelOptions<float>
        {
            Epochs = epochs,
            Seed = seed,
            UseEarlyStopping = false,
        };
        var model = new NBEATSModel<float>(options);
        int windows = series.Length - options.LookbackWindow - options.ForecastHorizon + 1;
        int stepsPerEpoch = (windows + options.BatchSize - 1) / options.BatchSize;
        return TrainAndTime("nbeats", device, model, series, epochs, stepsPerEpoch,
            new Dictionary<string, object>
            {
                ["stacks"] = options.NumStacks * options.NumBlocksPerStack,
                ["hidden"] = options.HiddenLayerSize,
                ["hidden_layers"] = options.NumHiddenLayers,
                ["polynomial_degree"] = options.PolynomialDegree,
                ["lookback"] = options.LookbackWindow,
                ["horizon"] = options.ForecastHorizon,
                ["batch_size"] = options.BatchSize,
            });
    }

    /// <summary>
    /// Trains <paramref name="model"/> once for <paramref name="epochs"/> epochs and times each epoch from the
    /// training callback's elapsed clock. The input matrix only supplies the row count; the forecasters window the
    /// target series themselves.
    /// </summary>
    private static Dictionary<string, object> TrainAndTime(string name, string device, TimeSeriesModelBase<float> model,
        double[] series, int epochs, int stepsPerEpoch, Dictionary<string, object> config)
    {
        Console.WriteLine($"[timeseries] {name}: training {epochs} epoch(s) on {device}, {stepsPerEpoch} step(s) per epoch...");
        var y = new Vector<float>(series.Length);
        var x = new Matrix<float>(series.Length, 1);
        for (int t = 0; t < series.Length; t++)
        {
            y[t] = (float)series[t];
            x[t, 0] = t;
        }

        var elapsed = new List<double>();
        var losses = new List<double>();
        model.TrainingEpochCallback = progress =>
        {
            elapsed.Add(progress.Elapsed.TotalSeconds);
            losses.Add(Convert.ToDouble(progress.Loss, CultureInfo.InvariantCulture));
            return true;
        };

        // N-BEATS reads each batch's loss back to the host inside its epoch, so every epoch callback already follows a
        // device synchronization; the final barrier makes the total honest for any model that does not.
        var clock = Stopwatch.StartNew();
        model.Train(x, y);
        global::BenchStats.SynchronizeDevice();
        double total = clock.Elapsed.TotalSeconds;

        // Epoch i's duration is the gap between consecutive callbacks; epoch 0 also includes the setup before it.
        var epochSeconds = new List<double>();
        for (int i = 0; i < elapsed.Count; i++)
            epochSeconds.Add(i == 0 ? elapsed[0] : elapsed[i] - elapsed[i - 1]);
        var steady = epochSeconds.Skip(1).OrderBy(v => v).ToArray();
        if (steady.Length == 0)
            throw new InvalidOperationException($"{name}: the training callback reported {elapsed.Count} epoch(s); need at least 2.");

        Console.WriteLine($"[timeseries] {name}: epochs {string.Join(", ", epochSeconds.Select(s => s.ToString("F3", CultureInfo.InvariantCulture)))} s; " +
            $"loss {losses.First():F4} -> {losses.Last():F4}");
        return new Dictionary<string, object>
        {
            ["model"] = name,
            ["parameters"] = model.GetParameters().Length,
            ["config"] = config,
            ["steps_per_epoch"] = stepsPerEpoch,
            ["epoch_seconds"] = epochSeconds,
            ["total_seconds"] = total,
            ["steady_state_epoch_seconds_median"] = Quantile(steady, 0.5),
            ["steady_state_epoch_seconds_p25"] = Quantile(steady, 0.25),
            ["steady_state_epoch_seconds_p75"] = Quantile(steady, 0.75),
            ["first_loss"] = losses.First(),
            ["last_loss"] = losses.Last(),
        };
    }

    private static double Quantile(double[] sorted, double q)
    {
        double position = q * (sorted.Length - 1);
        int lower = (int)Math.Floor(position);
        int upper = (int)Math.Ceiling(position);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }
}
