using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Models.Options;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

if (args.Length != 0 && (args.Length != 2 || args[0] != "--output"))
    throw new ArgumentException("Usage: EvolutionNoise [--output new-report.json]");
// A dedicated stream keeps native-library stdout diagnostics out of machine-readable evidence.
// CreateNew refuses an existing report before any observations run.
using var reportStream = args.Length == 2 ? new FileStream(args[1], FileMode.CreateNew, FileAccess.Write) : null;

// Frozen before execution: two screening presets, six roots, no fitted thresholds or optional stopping.
// These are trusted built-in configurations, never dynamically executed program text.
ulong[] seeds = { 1103, 2207, 3301, 4409, 5519, 6607 };
double[] thresholds = { .5, .98 };
var rows = new List<object>();
bool valid = true;
foreach (double threshold in thresholds)
foreach (ulong seed in seeds)
{
    int checks = 0, fits = 0;
    var observations = new List<object>();
    var ledger = new EvolutionResourceLedger($"ridge-{threshold.ToString(CultureInfo.InvariantCulture)}-{seed}", EvolutionResources.Of("cost_units", 10000));
    var correctness = new DelegateProgramFitnessEvaluator((g, _, _) =>
    {
        checks++;
        return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(
            g.Source is "0" or "4" or "10000" ? 1 : 0, new Dictionary<string, double>(), costUnits: 1));
    }, versionHash: "trusted-ridge-configuration-v1");
    DelegateProgramFitnessEvaluator Fit(int samples, string partition) => new((g, context, token) =>
    {
        token.ThrowIfCancellationRequested();
        var random = context.CreateRandom();
        var x = new Matrix<double>(samples, 1); var y = new Vector<double>(samples);
        for (int i = 0; i < samples; i++)
        {
            x[i, 0] = random.NextDouble() * 2 - 1;
            y[i] = 2 * x[i, 0] + (random.NextDouble() * 2 - 1) * .2;
        }
        // Every observation owns a new model and independent train/test draws; no weight/cache reuse.
        var model = new RidgeRegression<double>(new RidgeRegressionOptions<double>
        { Alpha = double.Parse(g.Source, CultureInfo.InvariantCulture), UseIntercept = false });
        fits++; model.Train(x, y);
        var test = new Matrix<double>(64, 1); var target = new double[64];
        for (int i = 0; i < 64; i++)
        {
            test[i, 0] = random.NextDouble() * 2 - 1;
            target[i] = 2 * test[i, 0] + (random.NextDouble() * 2 - 1) * .2;
        }
        var prediction = model.Predict(test);
        double mse = Enumerable.Range(0, 64).Average(i => Math.Pow(prediction[i] - target[i], 2));
        if (!double.IsFinite(mse)) throw new InvalidOperationException("Nonfinite model prediction.");
        double quality = 1 / (1 + mse); // a defined bounded metric, not clipping observed timing
        observations.Add(new { Partition = partition, g.Id, Alpha = g.Source, context.SeedStream, TrainingRows = samples, Mse = mse, Quality = quality });
        return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(quality, new Dictionary<string, double>(), costUnits: 1));
    }, versionHash: $"fresh-ridge-v1-{partition}-{samples}");
    var options = new ProgramNoiseEvaluationOptions(auditCandidates: 3, maximumChallenges: 2, screenThreshold: threshold,
        usefulThreshold: .7, minimumImprovement: .05);
    var session = new ProgramNoiseEvaluationSession(options, ledger, correctness, Fit(8, "screen"), Fit(64, "search"),
        correctness, Fit(64, "confirmation"));
    ProgramGenome Genome(string alpha) => new(alpha, ProgramLanguage.Generic);
    var population = new[] { Genome("0"), Genome("4"), Genome("10000") };
    var screen = await session.ScreenAndAuditAsync("preset", population, seed, seed + 100000);
    var challenge = await session.ChallengeAsync(0, population[0], population[2], seed + 200000);
    var reverse = await session.ChallengeAsync(1, population[2], population[0], seed + 300000);
    decimal charged = screen.ChargedCostUnits + challenge.ChargedCostUnits + reverse.ChargedCostUnits;
    bool runValid = screen.IsComplete && challenge.IsConfirmed && !reverse.IsConfirmed &&
        checks == fits && charged == checks + fits && ledger.Snapshot().Spent["cost_units"] == charged &&
        challenge.CandidateConfirmation!.Samples.All(sample => sample.Context.Purpose == EvolutionReplicationPurpose.Confirmation);
    valid &= runValid;
    rows.Add(new { Workload = "ridge-regression", Seed = seed, ScreenThreshold = threshold, Valid = runValid,
        PresetApproved = screen.IsComplete && screen.Entries.Any(row => row.Passed) && screen.Audit is not null && screen.Audit.FalseRejectionRateUpper <= .1,
        session.VersionHash, Screen = screen, Challenge = challenge, ReverseChallenge = reverse,
        Checks = checks, Fits = fits, Ledger = ledger.Snapshot(), Observations = observations });
}

// Real runtime observations: correctness-checked trusted sort kernels, cold state per invocation, warmups explicit.
foreach (ulong seed in seeds)
{
    int checks = 0, invocations = 0;
    var observations = new List<EvolutionTaskResult>();
    var ledger = new EvolutionResourceLedger("sorting-" + seed, EvolutionResources.Of("cost_units", 10000));
    var timing = new EvolutionTimingProtocol(1, 1000);
    var correctness = new DelegateProgramFitnessEvaluator((g, _, _) =>
    {
        checks++;
        return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(g.Source is "array-sort" or "insertion-sort" ? 1 : 0,
            new Dictionary<string, double>(), costUnits: 1));
    }, versionHash: "trusted-sort-allowlist-v1");
    DelegateProgramFitnessEvaluator Measure(int size, string stage) => new(async (g, context, token) =>
    {
        var random = context.CreateRandom();
        int[] input = Enumerable.Range(0, size).Select(_ => random.NextInt(100000)).ToArray();
        int[] expected = input.OrderBy(value => value).ToArray();
        var result = await timing.MeasureAsync(_ =>
        {
            invocations++;
            int[] copy = input.ToArray(); // reset mutated input for every warmup and measured call
            if (g.Source == "array-sort") Array.Sort(copy);
            else for (int i = 1; i < copy.Length; i++)
            {
                int value = copy[i], j = i - 1;
                while (j >= 0 && copy[j] > value) { copy[j + 1] = copy[j]; j--; }
                copy[j + 1] = value;
            }
            if (!copy.SequenceEqual(expected)) throw new InvalidOperationException("Sort correctness failed.");
            return default;
        }, token);
        observations.Add(result); return result;
    }, versionHash: $"trusted-sort-v1-{stage}-{size}-{timing.VersionHash}");
    var options = new ProgramNoiseEvaluationOptions(confirmationSamples: 32, auditCandidates: 2, maximumChallenges: 1,
        maximumQuality: 1000, screenThreshold: 0, usefulThreshold: 500, minimumImprovement: 0,
        maximumCostPerSample: 3, direction: EvolutionOptimizationDirection.Minimize);
    var session = new ProgramNoiseEvaluationSession(options, ledger, correctness, Measure(128, "screen"), Measure(2048, "search"),
        correctness, Measure(2048, "confirmation"));
    var fast = new ProgramGenome("array-sort", ProgramLanguage.Generic);
    var slow = new ProgramGenome("insertion-sort", ProgramLanguage.Generic);
    // Deliberately aggressive zero-ms screen is a negative-control preset, not a speed recommendation.
    var screen = await session.ScreenAndAuditAsync("timing-preset", new[] { fast, slow }, seed, seed + 100000);
    var challenge = await session.ChallengeAsync(0, fast, slow, seed + 200000);
    decimal charged = screen.ChargedCostUnits + challenge.ChargedCostUnits;
    bool runValid = screen.IsComplete && challenge.CandidateSearch.IsComplete && challenge.IncumbentSearch?.IsComplete == true &&
        charged == checks + invocations && ledger.Snapshot().Spent["cost_units"] == charged;
    valid &= runValid;
    rows.Add(new { Workload = "trusted-sorting", Seed = seed, Valid = runValid, Screen = screen, Challenge = challenge,
        PresetApproved = screen.IsComplete && screen.Entries.Any(row => row.Passed) && screen.Audit is not null && screen.Audit.FalseRejectionRateUpper <= .1,
        Checks = checks, Invocations = invocations, Ledger = ledger.Snapshot(), Observations = observations });
}
var reportJson = JsonSerializer.Serialize(new
{
    Schema = "us06-consumer-noise-study-v1", Valid = valid, Seeds = seeds, ScreeningThresholds = thresholds, Runs = rows,
    Assemblies = new[] { typeof(ProgramNoiseEvaluationSession).Assembly.Location, typeof(EvolutionIncumbentChallenge<>).Assembly.Location,
        typeof(Matrix<>).Assembly.Location }.ToDictionary(path => Path.GetFileName(path)!, path => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)))),
    Interpretation = "Fresh AiDotNet RidgeRegression training and actual trusted sorting on fixed roots/presets. Report failures and wide intervals. No fitted thresholds, arbitrary code execution, external data or paid calls. Evidence applies to these workloads/environment, not universal production superiority."
}, new JsonSerializerOptions { WriteIndented = true });
if (reportStream is null) Console.WriteLine(reportJson);
else
{
    using var writer = new StreamWriter(reportStream, new System.Text.UTF8Encoding(false));
    writer.Write(reportJson);
}
return valid ? 0 : 1;
