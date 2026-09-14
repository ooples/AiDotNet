using System.Diagnostics;
using AiDotNet.Evolution.Programs;

namespace AiDotNet.Evolution.Deployment;

/// <summary>First-party bridges from exact deployable artifacts to independent consumer validation.</summary>
public static class EvolutionDeploymentEvaluators
{
    /// <summary>Builds a fresh correctness-first program evaluator; do not supply the hidden deployment checks to search/proposal generation.</summary>
    public static Func<EvolutionDeployableArtifact, int, CancellationToken, ValueTask<EvolutionDeploymentMeasurement>> Program(
        IProgramFitnessEvaluator correctness, IProgramFitnessEvaluator fitness, ulong rootSeed = 1234)
    {
        var evaluator = new CorrectnessGatedProgramFitnessEvaluator(correctness, fitness);
        return async (artifact, pairIndex, token) =>
        {
            if (pairIndex < 0) throw new ArgumentOutOfRangeException(nameof(pairIndex));
            ProgramGenome program = artifact.ReadProgram();
            // The same pair index supplies the same random stream to candidate and incumbent.
            var context = new EvolutionEvaluationContext(pairIndex + 1L, rootSeed, (ulong)pairIndex + 1, 1);
            var clock = Stopwatch.StartNew();
            var result = await evaluator.EvaluateAsync(program, context, token).ConfigureAwait(false);
            clock.Stop();
            bool fresh = result.MeasurementOrigin is not { Kind: not EvolutionMeasurementOriginKind.Measured };
            bool passed = result.Status == EvolutionEvaluationStatus.Completed && result.Quality.HasValue &&
                result.ConstraintViolations.All(value => value == 0) && fresh;
            return new EvolutionDeploymentMeasurement(passed, result.Quality ?? 0, result.Direction,
                TimeSpan.FromTicks(Math.Max(1, clock.Elapsed.Ticks)), result.CostUnits, fresh, result.Quality.HasValue);
        };
    }

    /// <summary>Restores exact trained bytes into an explicitly allowlisted factory before each independent model validation.</summary>
    /// <remarks>The factory selects trusted types; artifact metadata is never passed to reflection activation. The model is disposed after evaluation.</remarks>
    public static Func<EvolutionDeployableArtifact, int, CancellationToken, ValueTask<EvolutionDeploymentMeasurement>> Model<TModel>(
        string serializationVersion, Func<string, TModel> modelFactory,
        Func<TModel, int, CancellationToken, ValueTask<EvolutionDeploymentMeasurement>> evaluate)
        where TModel : class, IModelSerializer, IDisposable
    {
        DeploymentEncoding.RequireLabel(serializationVersion, 256);
        if (modelFactory is null) throw new ArgumentNullException(nameof(modelFactory));
        if (evaluate is null) throw new ArgumentNullException(nameof(evaluate));
        return async (artifact, pairIndex, token) =>
        {
            token.ThrowIfCancellationRequested();
            var model = artifact.RestoreModel(() => modelFactory(artifact.TypeIdentity), serializationVersion);
            EvolutionDeploymentMeasurement result;
            try { result = await evaluate(model, pairIndex, token).ConfigureAwait(false); }
            catch (Exception evaluationError)
            {
                try { model.Dispose(); }
                catch (Exception disposalError) { throw new AggregateException(evaluationError, disposalError); }
                throw;
            }
            model.Dispose();
            return result;
        };
    }
}
