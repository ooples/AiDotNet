using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet;

/// <content>
/// Training stages that run after the main fit on data of their own, shared by the in-memory and the streaming
/// supervised paths: one-shot fine-tuning, the staged training pipeline and the curriculum pass.
/// </content>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Gets whether any stage that trains after the main fit is configured.
    /// </summary>
    private bool HasConfiguredPostTrainingStage =>
        _fineTuningConfiguration is { Enabled: true }
        || _trainingPipelineConfiguration?.Stages is { Count: > 0 }
        || _curriculumLearningOptions is { Dataset: not null };

    /// <summary>
    /// Runs fine-tuning, then the training pipeline, then curriculum learning on the trained model, and returns
    /// the model the last configured stage produced.
    /// </summary>
    /// <remarks>
    /// None of the stages reads the training set of the main fit: fine-tuning and every pipeline stage carry their
    /// own <c>TrainingData</c>, and the curriculum pass trains on <c>CurriculumLearningOptions.Dataset</c>.
    /// </remarks>
    private async Task<IFullModel<T, TInput, TOutput>?> RunConfiguredPostTrainingStagesAsync(
        IFullModel<T, TInput, TOutput>? trainedModel,
        CancellationToken cancellationToken)
    {
        trainedModel = await RunConfiguredFineTuningAsync(trainedModel, cancellationToken).ConfigureAwait(false);
        trainedModel = await RunConfiguredTrainingPipelineAsync(trainedModel, cancellationToken).ConfigureAwait(false);
        return RunConfiguredCurriculumLearning(trainedModel);
    }

    /// <summary>
    /// Applies preference learning, RLHF, SFT and the like to the trained model, before metric finalization, so
    /// any checkpoint or result returned reflects the fine-tuned weights (#1357 / #1361).
    /// </summary>
    private async Task<IFullModel<T, TInput, TOutput>?> RunConfiguredFineTuningAsync(
        IFullModel<T, TInput, TOutput>? trainedModel,
        CancellationToken cancellationToken)
    {
        if (_fineTuningConfiguration?.Enabled != true)
        {
            return trainedModel;
        }

        var ftImpl = _fineTuningConfiguration.Implementation
            ?? throw new InvalidOperationException(
                "ConfigureFineTuning was enabled but no Implementation was provided. " +
                "Set FineTuningConfiguration.Implementation to a concrete IFineTuning<T, TInput, TOutput> instance " +
                "(e.g. new SupervisedFineTuning<...>(options), new DirectPreferenceOptimization<...>(options)).");
        if (_fineTuningConfiguration.TrainingData is null)
            throw new InvalidOperationException(
                "ConfigureFineTuning was enabled but no TrainingData was supplied. " +
                "Set FineTuningConfiguration.TrainingData to a FineTuningData<T, TInput, TOutput> appropriate for the chosen method " +
                "(SFT needs Inputs+Outputs; DPO/SimPO need preference pairs; RLHF/GRPO need rewards).");
        if (trainedModel is null)
            throw new InvalidOperationException(
                "ConfigureFineTuning was enabled but the optimizer did not produce a BestSolution to fine-tune. " +
                "This usually means main training failed silently — check earlier logs.");

        // Honor the BuildAsync caller's cancellation token across the fine-tune await (review #1361 fix #5).
        var fineTunedModel = await ftImpl.FineTuneAsync(
            trainedModel,
            _fineTuningConfiguration.TrainingData,
            cancellationToken).ConfigureAwait(false);

        // Rebind so downstream metric/checkpoint code sees the fine-tuned model.
        _model = fineTunedModel;
        return fineTunedModel;
    }

    /// <summary>
    /// Executes the user-defined sequence of training stages (#1361 #2). Each stage takes the previous stage's
    /// model and its own training data. Stages with <c>Enabled = false</c> or whose <c>RunCondition</c> returns
    /// false are skipped.
    /// </summary>
    private async Task<IFullModel<T, TInput, TOutput>?> RunConfiguredTrainingPipelineAsync(
        IFullModel<T, TInput, TOutput>? trainedModel,
        CancellationToken cancellationToken)
    {
        if (_trainingPipelineConfiguration?.Stages is not { Count: > 0 } stages)
        {
            return trainedModel;
        }

        if (trainedModel is null)
            throw new InvalidOperationException(
                "ConfigureTrainingPipeline was provided but main training did not produce a BestSolution. " +
                "Check earlier logs for an upstream training failure.");

        var currentModel = trainedModel;
        TrainingStageResult<T, TInput, TOutput>? previousStageResult = null;

        for (int stageIndex = 0; stageIndex < stages.Count; stageIndex++)
        {
            var stage = stages[stageIndex];
            if (!stage.Enabled) continue;
            if (stage.RunCondition is { } cond && !cond(previousStageResult)) continue;

            if (stage.CustomTrainingFunction is null)
                throw new InvalidOperationException(
                    $"TrainingPipeline stage '{stage.Name}' (index {stageIndex}) has Enabled=true but no " +
                    $"CustomTrainingFunction. The current wire-up requires each enabled stage to provide " +
                    $"its own training delegate; the StageType={stage.StageType} / FineTuningMethod=" +
                    $"{stage.FineTuningMethod} auto-dispatch path is not yet implemented. " +
                    $"Set stage.CustomTrainingFunction to an async (model, data, ct) => trainedModel delegate, " +
                    $"or remove this stage from the pipeline.");
            if (stage.TrainingData is null && !stage.IsEvaluationOnly)
                throw new InvalidOperationException(
                    $"TrainingPipeline stage '{stage.Name}' (index {stageIndex}) has no TrainingData and is " +
                    $"not marked IsEvaluationOnly. Each training stage needs a FineTuningData<T, TInput, " +
                    $"TOutput> appropriate for its FineTuningMethod.");

            var stageStart = DateTime.UtcNow;
            var stageResult = new TrainingStageResult<T, TInput, TOutput>
            {
                StageName = stage.Name,
                StageIndex = stageIndex,
            };
            try
            {
                if (!stage.IsEvaluationOnly)
                {
                    // Pass through the BuildAsync caller's cancellation token so user-supplied stage delegates
                    // can honor cancellation (review #1361 fix #5).
                    currentModel = await stage.CustomTrainingFunction(
                        currentModel,
                        (stage.TrainingData ?? throw new InvalidOperationException("stage.TrainingData is null where it is required.")),
                        cancellationToken).ConfigureAwait(false);
                    if (currentModel is null)
                        throw new InvalidOperationException(
                            $"TrainingPipeline stage '{stage.Name}' returned null from " +
                            $"CustomTrainingFunction. Stages must return a non-null model.");
                }
                stageResult.Model = currentModel;
                stageResult.Success = true;
            }
            catch (Exception ex)
            {
                stageResult.Success = false;
                stageResult.ErrorMessage = ex.Message;
                throw;
            }
            finally
            {
                stageResult.Duration = DateTime.UtcNow - stageStart;
            }
            previousStageResult = stageResult;
        }

        // Rebind _model so weight-streaming reports, checkpoint writes and any other consumer that reads _model
        // directly sees the model the pipeline produced.
        _model = currentModel;
        return currentModel;
    }

    /// <summary>
    /// Runs a curriculum-scheduled refinement pass over <c>CurriculumLearningOptions.Dataset</c> (#1361 #3). The
    /// learner ranks the samples by difficulty, using the custom estimator or a loss-based one tied to the trained
    /// model, and trains in phases from easy to hard. Skipped when no dataset was supplied.
    /// </summary>
    private IFullModel<T, TInput, TOutput>? RunConfiguredCurriculumLearning(IFullModel<T, TInput, TOutput>? trainedModel)
    {
        if (_curriculumLearningOptions is null || _curriculumLearningOptions.Dataset is null)
        {
            return trainedModel;
        }

        if (trainedModel is null)
            throw new InvalidOperationException(
                "ConfigureCurriculumLearning was provided with a Dataset but main training did not " +
                "produce a BestSolution to curriculum-train. Check earlier logs for an upstream " +
                "training failure.");

        var curriculumConfig = new AiDotNet.CurriculumLearning.CurriculumLearnerConfig<T>
        {
            TotalEpochs = _curriculumLearningOptions.TotalEpochs ?? 100,
            NumPhases = _curriculumLearningOptions.NumPhases ?? 5,
            InitialDataFraction = MathHelper.GetNumericOperations<T>().FromDouble(
                _curriculumLearningOptions.InitialDataFraction ?? 0.2),
            FinalDataFraction = MathHelper.GetNumericOperations<T>().FromDouble(
                _curriculumLearningOptions.FinalDataFraction ?? 1.0),
            ScheduleType = _curriculumLearningOptions.ScheduleType,
            RecalculateDifficulties = _curriculumLearningOptions.RecalculateDifficulties ?? false,
            DifficultyRecalculationFrequency = _curriculumLearningOptions.DifficultyRecalculationFrequency ?? 10,
            NormalizeDifficulties = _curriculumLearningOptions.NormalizeDifficulties ?? true,
            EarlyStoppingPatience = _curriculumLearningOptions.EarlyStopping?.Patience ?? 10,
            EarlyStoppingMinDelta = MathHelper.GetNumericOperations<T>().FromDouble(
                _curriculumLearningOptions.EarlyStopping?.MinDelta ?? 0.001),
            UseEarlyStopping = _curriculumLearningOptions.EarlyStopping?.Enabled ?? true,
            BatchSize = _curriculumLearningOptions.BatchSize ?? 32,
            LearningRate = MathHelper.GetNumericOperations<T>().FromDouble(0.001),
            ShuffleWithinPhase = _curriculumLearningOptions.ShuffleWithinPhase ?? true,
            UseDifficultyWeighting = _curriculumLearningOptions.UseDifficultyWeighting ?? false,
            RandomSeed = _curriculumLearningOptions.RandomSeed,
            Verbosity = _curriculumLearningOptions.Verbosity,
        };

        var difficultyEstimator = _curriculumLearningOptions.CustomDifficultyEstimator
            ?? new AiDotNet.CurriculumLearning.DifficultyEstimators
                   .LossBasedDifficultyEstimator<T, TInput, TOutput>(
                       lossFunction: null,
                       normalize: curriculumConfig.NormalizeDifficulties);

        var curriculumLearner = new AiDotNet.CurriculumLearning.CurriculumLearner<T, TInput, TOutput>(
            baseModel: trainedModel,
            config: curriculumConfig,
            difficultyEstimator: difficultyEstimator,
            scheduler: _curriculumLearningOptions.CustomScheduler);

        curriculumLearner.Train(_curriculumLearningOptions.Dataset);

        // The CurriculumLearner mutates BaseModel in place; reassign to make the post-curriculum weights explicit.
        _model = curriculumLearner.BaseModel;
        return curriculumLearner.BaseModel;
    }
}
