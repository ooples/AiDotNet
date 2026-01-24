using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Supervised Fine-Tuning (SFT) - the foundational fine-tuning method.
/// </summary>
/// <remarks>
/// <para>
/// SFT trains a model on labeled input-output pairs using standard supervised learning.
/// This is the most straightforward fine-tuning approach and serves as the foundation
/// for more advanced methods like RLHF or DPO.
/// </para>
/// <para><b>For Beginners:</b> SFT is like teaching by example. You show the model many
/// examples of correct input-output pairs, and it learns to produce similar outputs.
/// For instance, you might train a model on high-quality question-answer pairs to make
/// it better at answering questions.</para>
/// <para><b>Use Cases:</b></para>
/// <list type="bullet">
/// <item><term>Instruction Tuning</term><description>Teaching models to follow instructions</description></item>
/// <item><term>Task Adaptation</term><description>Specializing models for specific tasks</description></item>
/// <item><term>Format Learning</term><description>Teaching models specific output formats</description></item>
/// <item><term>Pre-alignment</term><description>Initial training before preference optimization</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class SupervisedFineTuning<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    private readonly ILossFunction<T>? _customLossFunction;

    /// <summary>
    /// Initializes a new instance of SFT with default loss function.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    /// <remarks>
    /// This constructor is provided for compatibility with Activator.CreateInstance
    /// and other reflection-based instantiation that requires a single-parameter constructor.
    /// The model's default loss function will be used during training.
    /// </remarks>
    public SupervisedFineTuning(FineTuningOptions<T> options)
        : this(options, null)
    {
    }

    /// <summary>
    /// Initializes a new instance of SFT with a custom loss function.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    /// <param name="lossFunction">Custom loss function. Uses model default if null.</param>
    public SupervisedFineTuning(FineTuningOptions<T> options, ILossFunction<T>? lossFunction)
        : base(options)
    {
        _customLossFunction = lossFunction;
    }

    /// <inheritdoc/>
    public override string MethodName => "SFT";

    /// <inheritdoc/>
    public override FineTuningCategory Category => FineTuningCategory.SupervisedFineTuning;

    /// <inheritdoc/>
    public override bool RequiresRewardModel => false;

    /// <inheritdoc/>
    public override bool RequiresReferenceModel => false;

    /// <inheritdoc/>
    protected override void ValidateTrainingData(FineTuningData<T, TInput, TOutput> data)
    {
        base.ValidateTrainingData(data);

        if (!data.HasSFTData)
        {
            throw new ArgumentException("SFT requires Inputs and Outputs arrays with matching lengths.", nameof(data));
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

        // Clone the model for fine-tuning
        var model = baseModel.DeepCopy();
        var lossFunction = _customLossFunction ?? model.DefaultLossFunction;
        var learningRate = NumOps.FromDouble(Options.LearningRate);

        var totalSteps = Options.Epochs * (int)Math.Ceiling((double)trainingData.Count / Options.BatchSize);

        int step = 0;
        for (int epoch = 0; epoch < Options.Epochs; epoch++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            foreach (var batch in CreateBatches(trainingData, Options.BatchSize, shuffle: true))
            {
                cancellationToken.ThrowIfCancellationRequested();

                double batchLoss = await ProcessBatchAsync(model, batch, lossFunction, learningRate);
                step++;

                UpdateMetrics(batchLoss, step);
                LogProgress(step, totalSteps, batchLoss, $"Epoch {epoch + 1}/{Options.Epochs}");
            }
        }

        return model;
    }

    /// <summary>
    /// Processes a single batch of training data.
    /// </summary>
    private Task<double> ProcessBatchAsync(
        IFullModel<T, TInput, TOutput> model,
        FineTuningData<T, TInput, TOutput> batch,
        ILossFunction<T> lossFunction,
        T learningRate)
    {
        double totalLoss = 0.0;
        int count = batch.Count;

        for (int i = 0; i < count; i++)
        {
            var input = batch.Inputs[i];
            var output = batch.Outputs[i];

            // Forward pass
            var prediction = model.Predict(input);

            // Compute gradients
            var gradients = model.ComputeGradients(input, output, lossFunction);

            // Apply gradients
            model.ApplyGradients(gradients, learningRate);

            // Accumulate loss for logging
            var predVector = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
            var targetVector = ConversionsHelper.ConvertToVector<T, TOutput>(output);
            totalLoss += NumOps.ToDouble(lossFunction.CalculateLoss(predVector, targetVector));
        }

        var avgLoss = count > 0 ? totalLoss / count : 0.0;
        return Task.FromResult(avgLoss);
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

        ValidateTrainingData(evaluationData);

        var lossFunction = _customLossFunction ?? model.DefaultLossFunction;
        var metrics = new FineTuningMetrics<T>
        {
            MethodName = MethodName,
            TrainingStartTime = DateTime.UtcNow
        };

        double totalLoss = 0.0;
        int correctPredictions = 0;
        int total = evaluationData.Count;

        for (int i = 0; i < total; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var input = evaluationData.Inputs[i];
            var output = evaluationData.Outputs[i];

            var prediction = model.Predict(input);
            var predVector = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
            var targetVector = ConversionsHelper.ConvertToVector<T, TOutput>(output);

            totalLoss += NumOps.ToDouble(lossFunction.CalculateLoss(predVector, targetVector));

            // Simple accuracy check for classification
            if (IsCorrectPrediction(predVector, targetVector))
            {
                correctPredictions++;
            }
        }

        metrics.ValidationLoss = total > 0 ? totalLoss / total : 0.0;
        metrics.CustomMetrics["accuracy"] = total > 0 ? (double)correctPredictions / total : 0.0;
        metrics.TrainingEndTime = DateTime.UtcNow;

        return await Task.FromResult(metrics);
    }

    /// <summary>
    /// Checks if a prediction matches the target (for accuracy calculation).
    /// </summary>
    private static bool IsCorrectPrediction(Tensors.LinearAlgebra.Vector<T> prediction, Tensors.LinearAlgebra.Vector<T> target)
    {
        if (prediction.Length != target.Length || prediction.Length == 0)
        {
            return false;
        }

        // For classification, check if argmax matches
        int predArgmax = 0;
        int targetArgmax = 0;
        double predMax = NumOps.ToDouble(prediction[0]);
        double targetMax = NumOps.ToDouble(target[0]);

        for (int i = 1; i < prediction.Length; i++)
        {
            var predVal = NumOps.ToDouble(prediction[i]);
            var targetVal = NumOps.ToDouble(target[i]);

            if (predVal > predMax)
            {
                predMax = predVal;
                predArgmax = i;
            }

            if (targetVal > targetMax)
            {
                targetMax = targetVal;
                targetArgmax = i;
            }
        }

        return predArgmax == targetArgmax;
    }
}
