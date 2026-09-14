using System.Diagnostics;
using AiDotNet.AutoML;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Deployment;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Evolution;

public sealed partial class EvolutionDeploymentLifecycleTests
{
    [Fact]
    public async Task ProgramAdapter_UsesFreshCorrectnessBeforeFitnessAndPreservesBothCosts()
    {
        int fitnessCalls = 0;
        var check = new DelegateProgramFitnessEvaluator((genome, _, _) => new ValueTask<EvolutionTaskResult>(
            EvolutionTaskResult.Completed(genome.Source == "wrong" ? 0 : 1, costUnits: 2)));
        var fitness = new DelegateProgramFitnessEvaluator((_, _, _) =>
        { fitnessCalls++; return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(3, costUnits: 4)); });
        var evaluator = EvolutionDeploymentEvaluators.Program(check, fitness);
        var invalid = await evaluator(Program("wrong"), 0, CancellationToken.None);
        Assert.False(invalid.CorrectnessPassed);
        Assert.Equal(0, fitnessCalls);
        Assert.Equal(2, invalid.CostUnits);
        var valid = await evaluator(Program("winner"), 0, CancellationToken.None);
        Assert.True(valid.CorrectnessPassed);
        Assert.True(valid.IsFresh);
        Assert.Equal(6, valid.CostUnits);
        Assert.Equal(1, fitnessCalls);
    }

    [Fact]
    public async Task ProgramRetuner_UsesRealEngineAndOverridesInflatedCallerBudgets()
    {
        int calls = 0;
        var variation = new DeploymentVariation();
        var retuner = EvolutionDeploymentRetuners.Program(_ => new ProgramEvolutionOptions
        {
            SeedPrograms = new List<string> { "base" }, MaxProgramChars = 512,
            CustomVariation = variation,
            CustomFitnessEvaluator = new DelegateProgramFitnessEvaluator(genome =>
            { calls++; return genome.Source == "winner" ? 2 : 1; }),
            Engine = new EvolutionEngineOptions { MaxEvaluationAttempts = 9999, MaxProposals = 9999 }
        }, _ => new DelegateProgramFitnessEvaluator(_ => 1), EvolutionOptimizationDirection.Maximize);
        var artifact = await retuner(new EvolutionDeploymentRetuneRequest(Envelope(0), Policy()), CancellationToken.None);
        Assert.Equal("winner", artifact.ReadProgram().Source);
        Assert.Equal(Envelope(0).Key, artifact.Envelope.Key);
        Assert.InRange(calls, 1, 3);
        Assert.InRange(variation.Proposals, 1, 6);
    }

    [Fact]
    public async Task ModelAdapter_RestoresTheFrozenStateAndDisposesEachEvaluationInstance()
    {
        using var original = new TinyModel { Weight = 7 };
        var artifact = EvolutionDeployableArtifact.FromModel(original, "tiny-v1", Envelope());
        TinyModel? restored = null;
        var evaluator = EvolutionDeploymentEvaluators.Model("tiny-v1", _ => restored = new TinyModel(),
            (TinyModel model, int _, CancellationToken _) => new ValueTask<EvolutionDeploymentMeasurement>(Measurement(model.Weight)));
        original.Weight = 99;
        var measurement = await evaluator(artifact, 0, CancellationToken.None);
        Assert.Equal(7, measurement.Quality);
        Assert.True(restored!.Disposed);
    }

    [Fact(Timeout = 120000)]
    public async Task AutoMLRetuning_PromotesReloadableTrainedWeightsUnderIndependentHoldoutValidation()
    {
        var trainX = new Matrix<double>(24, 2);
        var trainY = new Vector<double>(24);
        for (int i = 0; i < 24; i++)
        {
            trainX[i, 0] = (i - 12) / 6d; trainX[i, 1] = (i % 5) / 5d;
            trainY[i] = 1 + 2 * trainX[i, 0] - 3 * trainX[i, 1];
        }
        var validationX = new Matrix<double>(8, 2);
        var validationY = new Vector<double>(8);
        for (int i = 0; i < 8; i++)
        {
            validationX[i, 0] = (i - 3) / 5d; validationX[i, 1] = (i % 3) / 3d;
            validationY[i] = 1 + 2 * validationX[i, 0] - 3 * validationX[i, 1];
        }
        using var baseline = new MultipleLinearRegression<double>();
        baseline.Train(trainX, new Vector<double>(24));
        const string format = "linear-serialized-state-v1";
        var evaluator = EvolutionDeploymentEvaluators.Model(format, _ => new MultipleLinearRegression<double>(),
            (MultipleLinearRegression<double> model, int pair, CancellationToken token) =>
            {
                token.ThrowIfCancellationRequested();
                // Deployment holdout points are not supplied to AutoML's training/search-validation factory.
                var holdout = new Matrix<double>(2, 2);
                for (int row = 0; row < 2; row++)
                { holdout[row, 0] = 2.1 + pair * 0.17 + row * 0.03; holdout[row, 1] = 0.13 + pair * 0.07; }
                var timer = Stopwatch.StartNew();
                var prediction = model.Predict(holdout);
                timer.Stop();
                double mse = Enumerable.Range(0, 2).Average(row => Math.Pow(prediction[row] - (1 + 2 * holdout[row, 0] - 3 * holdout[row, 1]), 2));
                return new ValueTask<EvolutionDeploymentMeasurement>(new EvolutionDeploymentMeasurement(true, mse,
                    EvolutionOptimizationDirection.Minimize, TimeSpan.FromTicks(Math.Max(1, timer.Elapsed.Ticks)), 2));
            });
        var policy = new EvolutionDeploymentPolicy(true, EvolutionOptimizationDirection.Minimize, pairedSamples: 5,
            maximumP95LatencyRatio: 100000, maximumRetunes: 1, searchEvaluations: 3, searchProposals: 6,
            timeout: TimeSpan.FromSeconds(45), cooldown: TimeSpan.Zero);
        var lifecycle = new EvolutionDeploymentLifecycle(Registry(), Envelope(), policy,
            env => EvolutionDeployableArtifact.FromModel(baseline, format, env), evaluator);
        int created = 0;
        var retuner = EvolutionDeploymentRetuners.AutoML<double, Matrix<double>, Vector<double>>(
            _ => (trainX, trainY, validationX, validationY), (search, _) =>
            {
                search.SetCandidateModels(new List<Type> { typeof(MultipleLinearRegression<>) });
                search.EnsembleOptions.Enabled = false;
                search.TrialLimit = 9999; // The admitted driver must overwrite this before the search starts.
                search.OnCandidateCreated += _ => created++;
            }, format, new MapElitesAutoMLOptions { InitialPopulationSize = 1, MaxProposalMultiplier = 10000 });
        Assert.True(lifecycle.Select(Envelope(0)).IsFallback);
        var result = await lifecycle.RetunePendingAsync(retuner, _ => Task.CompletedTask);
        Assert.True(result.Activated, result.Outcome);
        Assert.InRange(created, 1, 3);
        var selected = lifecycle.Select(Envelope(0));
        Assert.False(selected.IsFallback);
        var restoredArtifact = Registry().Load(selected.Artifact.Id, Envelope(0));
        using var restored = restoredArtifact.RestoreModel(() => new MultipleLinearRegression<double>(), format);
        var heldout = new Matrix<double>(1, 2); heldout[0, 0] = 4; heldout[0, 1] = 0.5;
        Assert.Equal(7.5, restored.Predict(heldout)[0], precision: 8);
        Assert.NotEqual(baseline.Serialize(), restoredArtifact.CopyPayload());
    }

    private sealed class DeploymentVariation : IProgramVariationOperator
    {
        internal int Proposals { get; private set; }
        public string Id => "deployment-test-variation";
        public string VersionHash => "deployment-test-variation-v1";
        public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default)
        { cancellationToken.ThrowIfCancellationRequested(); Proposals++; return new(new ProgramGenome("winner")); }
        public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) { }
        public ProgramEvolutionLlmUsage GetUsage() => new();
    }
}
