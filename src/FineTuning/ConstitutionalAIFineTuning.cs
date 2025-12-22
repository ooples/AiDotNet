using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Constitutional AI (CAI) fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// Constitutional AI uses a set of principles (a "constitution") to guide model behavior.
/// The model learns to critique and revise its own outputs based on these principles.
/// </para>
/// <para><b>For Beginners:</b> CAI is like giving the model a rulebook to follow.
/// The model learns to check its own answers against rules like "be helpful" or
/// "don't cause harm" and revise answers that break the rules.</para>
/// <para>
/// CAI training has two phases:
/// 1. Supervised Learning from AI Feedback (SLAIF): Generate, critique, revise
/// 2. Reinforcement Learning from AI Feedback (RLAIF): Use revised outputs for preference training
/// </para>
/// <para>
/// Original paper: "Constitutional AI: Harmlessness from AI Feedback"
/// by Bai et al. (2022)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class ConstitutionalAIFineTuning<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    private IFullModel<T, TInput, TOutput>? _referenceModel;

    /// <summary>
    /// Initializes a new instance of Constitutional AI fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public ConstitutionalAIFineTuning(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.ConstitutionalAI)
        {
            options.MethodType = FineTuningMethodType.ConstitutionalAI;
        }
    }

    /// <inheritdoc/>
    public override string MethodName => "CAI";

    /// <inheritdoc/>
    public override FineTuningCategory Category => FineTuningCategory.Constitutional;

    /// <inheritdoc/>
    public override bool RequiresRewardModel => false;

    /// <inheritdoc/>
    public override bool RequiresReferenceModel => true;

    /// <summary>
    /// Gets the constitutional principles used for training.
    /// </summary>
    public string[] Principles => Options.ConstitutionalPrinciples;

    /// <inheritdoc/>
    protected override void ValidateTrainingData(FineTuningData<T, TInput, TOutput> data)
    {
        base.ValidateTrainingData(data);

        // CAI can work with either critique-revision data or generate it from SFT data
        if (!data.HasSFTData && data.CritiqueRevisions.Length == 0)
        {
            throw new ArgumentException(
                "Constitutional AI requires either SFT data (Inputs/Outputs) or CritiqueRevisions data.",
                nameof(data));
        }
    }

    /// <inheritdoc/>
    public override async Task<IFullModel<T, TInput, TOutput>> FineTuneAsync(
        IFullModel<T, TInput, TOutput> baseModel,
        FineTuningData<T, TInput, TOutput> trainingData,
        CancellationToken cancellationToken = default)
    {
        if (baseModel == null)
        {
            throw new ArgumentNullException(nameof(baseModel));
        }

        ValidateTrainingData(trainingData);

        _referenceModel = baseModel.Clone();
        var policyModel = baseModel.Clone();

        CurrentMetrics = new FineTuningMetrics<T>
        {
            MethodName = MethodName,
            TrainingStartTime = DateTime.UtcNow
        };

        var beta = Options.Beta;
        var critiqueIterations = Options.CritiqueIterations;
        var totalSteps = Options.Epochs * (trainingData.Count / Options.BatchSize);
        var currentStep = 0;

        for (int epoch = 0; epoch < Options.Epochs; epoch++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            foreach (var batch in CreateBatches(trainingData, Options.BatchSize))
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                var batchLoss = await ComputeCAILossAndUpdateAsync(
                    policyModel, batch, beta, critiqueIterations, Options.LearningRate, cancellationToken);
                currentStep++;

                UpdateMetrics(batchLoss, currentStep);
                LogProgress(currentStep, totalSteps, batchLoss, $"principles={Principles.Length}");
            }
        }

        CurrentMetrics.TrainingEndTime = DateTime.UtcNow;
        CurrentMetrics.TrainingTimeSeconds = (CurrentMetrics.TrainingEndTime - CurrentMetrics.TrainingStartTime).TotalSeconds;

        return policyModel;
    }

    /// <inheritdoc/>
    public override async Task<FineTuningMetrics<T>> EvaluateAsync(
        IFullModel<T, TInput, TOutput> model,
        FineTuningData<T, TInput, TOutput> evaluationData,
        CancellationToken cancellationToken = default)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (evaluationData == null)
        {
            throw new ArgumentNullException(nameof(evaluationData));
        }

        var metrics = new FineTuningMetrics<T>
        {
            MethodName = MethodName
        };

        double totalHarmlessnessScore = 0.0;
        double totalHelpfulnessScore = 0.0;
        int count = 0;

        for (int i = 0; i < evaluationData.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            var input = evaluationData.Inputs[i];
            var output = model.Predict(input);

            // Evaluate against constitutional principles
            var (harmlessness, helpfulness) = EvaluateConstitutionalCompliance(input, output);
            totalHarmlessnessScore += harmlessness;
            totalHelpfulnessScore += helpfulness;
            count++;
        }

        metrics.HarmlessnessScore = count > 0 ? totalHarmlessnessScore / count : 0.0;
        metrics.HelpfulnessScore = count > 0 ? totalHelpfulnessScore / count : 0.0;

        return await Task.FromResult(metrics);
    }

    /// <summary>
    /// Computes the CAI loss for a batch and updates model parameters.
    /// </summary>
    private async Task<double> ComputeCAILossAndUpdateAsync(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch,
        double beta,
        int critiqueIterations,
        double learningRate,
        CancellationToken cancellationToken)
    {
        if (_referenceModel == null)
        {
            throw new InvalidOperationException("Reference model not initialized.");
        }

        double totalLoss = 0.0;

        // If we have pre-computed critique-revision data, use it
        if (batch.CritiqueRevisions.Length > 0)
        {
            for (int i = 0; i < batch.CritiqueRevisions.Length; i++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                if (i >= batch.Inputs.Length)
                {
                    break; // No corresponding input for remaining revisions
                }

                var (original, _, revised) = batch.CritiqueRevisions[i];
                var input = batch.Inputs[i];

                // Train to prefer revised over original (like DPO)
                var piRevisedLogProb = ComputeLogProbability(policyModel, input, revised);
                var piOriginalLogProb = ComputeLogProbability(policyModel, input, original);
                var refRevisedLogProb = ComputeLogProbability(_referenceModel, input, revised);
                var refOriginalLogProb = ComputeLogProbability(_referenceModel, input, original);

                var revisedLogRatio = piRevisedLogProb - refRevisedLogProb;
                var originalLogRatio = piOriginalLogProb - refOriginalLogProb;

                var margin = beta * (revisedLogRatio - originalLogRatio);
                var loss = -LogSigmoid(margin);

                // Compute and apply gradients to update model parameters
                var gradients = policyModel.ComputeGradients(input, revised);
                policyModel.ApplyGradients(gradients, NumOps.FromDouble(learningRate));

                totalLoss += loss;
            }

            return await Task.FromResult(batch.CritiqueRevisions.Length > 0
                ? totalLoss / batch.CritiqueRevisions.Length
                : 0.0);
        }

        // Otherwise, use SFT data and simulate critique-revision process
        for (int i = 0; i < batch.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            var input = batch.Inputs[i];
            var targetOutput = batch.Outputs[i];

            // SFT-style loss on target outputs (which should be constitutionally aligned)
            var logProb = ComputeLogProbability(policyModel, input, targetOutput);
            var loss = -logProb;

            // Compute and apply gradients to update model parameters
            var gradients = policyModel.ComputeGradients(input, targetOutput);
            policyModel.ApplyGradients(gradients, NumOps.FromDouble(learningRate));

            totalLoss += loss;
        }

        return await Task.FromResult(batch.Count > 0 ? totalLoss / batch.Count : 0.0);
    }

    /// <summary>
    /// Evaluates how well an output complies with constitutional principles.
    /// </summary>
    /// <returns>Tuple of (harmlessness score, helpfulness score) between 0 and 1.</returns>
    private (double Harmlessness, double Helpfulness) EvaluateConstitutionalCompliance(
        TInput input, TOutput output)
    {
        // This is a placeholder - in a real implementation, this would use
        // another model or rule-based system to evaluate compliance
        // For now, return default scores
        return (0.8, 0.8);
    }
}
