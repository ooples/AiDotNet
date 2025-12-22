using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.CurriculumLearning.DifficultyEstimators;

/// <summary>
/// Difficulty estimator based on transfer learning from a simpler "teacher" model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This estimator uses the performance gap between a simple
/// "teacher" model and the main "student" model to estimate difficulty. Samples that
/// are easy for the simple model but hard for the main model are considered easier,
/// while samples that are hard for both are considered more difficult.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>A simple teacher model is trained on the data first</description></item>
/// <item><description>For each sample, both teacher and student predictions are compared</description></item>
/// <item><description>The gap indicates sample difficulty relative to model complexity</description></item>
/// </list>
///
/// <para><b>Difficulty Calculation:</b></para>
/// <list type="bullet">
/// <item><description>Teacher correct, Student wrong → Medium difficulty (learnable)</description></item>
/// <item><description>Both wrong → High difficulty (hard sample)</description></item>
/// <item><description>Both correct → Low difficulty (easy sample)</description></item>
/// </list>
///
/// <para><b>References:</b></para>
/// <list type="bullet">
/// <item><description>Weinshall et al. "Curriculum Learning by Transfer Learning" (2018)</description></item>
/// </list>
/// </remarks>
public class TransferBasedDifficultyEstimator<T, TInput, TOutput> : DifficultyEstimatorBase<T, TInput, TOutput>
{
    private readonly IFullModel<T, TInput, TOutput>? _teacherModel;
    private readonly TransferDifficultyMode _mode;
    private readonly bool _normalize;

    /// <summary>
    /// Gets the name of this estimator.
    /// </summary>
    public override string Name => "TransferBased";

    /// <summary>
    /// Gets whether this estimator requires the main model.
    /// </summary>
    public override bool RequiresModel => true;

    /// <summary>
    /// Gets the teacher model used for comparison.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? TeacherModel => _teacherModel;

    /// <summary>
    /// Initializes a new instance of the <see cref="TransferBasedDifficultyEstimator{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="teacherModel">The simpler teacher model. If null, teacher must be trained separately.</param>
    /// <param name="mode">The transfer difficulty calculation mode.</param>
    /// <param name="normalize">Whether to normalize difficulties to [0, 1].</param>
    public TransferBasedDifficultyEstimator(
        IFullModel<T, TInput, TOutput>? teacherModel = null,
        TransferDifficultyMode mode = TransferDifficultyMode.LossGap,
        bool normalize = true)
    {
        _teacherModel = teacherModel;
        _mode = mode;
        _normalize = normalize;
    }

    /// <summary>
    /// Estimates the difficulty of a single sample using transfer comparison.
    /// </summary>
    public override T EstimateDifficulty(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model),
                "TransferBasedDifficultyEstimator requires a student model.");
        }

        if (_teacherModel == null)
        {
            throw new InvalidOperationException(
                "TransferBasedDifficultyEstimator requires a teacher model. " +
                "Either provide one in the constructor or use TrainTeacher().");
        }

        return _mode switch
        {
            TransferDifficultyMode.LossGap => CalculateLossGapDifficulty(input, expectedOutput, model),
            TransferDifficultyMode.TeacherLoss => CalculateTeacherLossDifficulty(input, expectedOutput),
            TransferDifficultyMode.ConfidenceGap => CalculateConfidenceGapDifficulty(input, expectedOutput, model),
            TransferDifficultyMode.Combined => CalculateCombinedDifficulty(input, expectedOutput, model),
            _ => CalculateLossGapDifficulty(input, expectedOutput, model)
        };
    }

    /// <summary>
    /// Estimates difficulty scores for all samples.
    /// </summary>
    public override Vector<T> EstimateDifficulties(
        IDataset<T, TInput, TOutput> dataset,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        var difficulties = base.EstimateDifficulties(dataset, model);

        if (_normalize)
        {
            difficulties = NormalizeDifficulties(difficulties);
        }

        return difficulties;
    }

    /// <summary>
    /// Calculates difficulty based on loss gap between teacher and student.
    /// </summary>
    private T CalculateLossGapDifficulty(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput> studentModel)
    {
        var teacherPrediction = _teacherModel!.Predict(input);
        var studentPrediction = studentModel.Predict(input);

        // Convert to vectors for loss calculation
        var expectedVector = ConversionsHelper.ConvertToVector<T, TOutput>(expectedOutput);
        var teacherPredVector = ConversionsHelper.ConvertToVector<T, TOutput>(teacherPrediction);
        var studentPredVector = ConversionsHelper.ConvertToVector<T, TOutput>(studentPrediction);

        var teacherLoss = _teacherModel.DefaultLossFunction.CalculateLoss(teacherPredVector, expectedVector);
        var studentLoss = studentModel.DefaultLossFunction.CalculateLoss(studentPredVector, expectedVector);

        // Difficulty = student_loss - teacher_loss
        // Positive gap means student struggles more than teacher (harder)
        // We also add teacher loss to account for inherently hard samples
        var gap = NumOps.Subtract(studentLoss, teacherLoss);

        // Weight: more emphasis on samples where teacher is also wrong
        var teacherWeight = NumOps.FromDouble(0.3);
        var weighted = NumOps.Add(gap, NumOps.Multiply(teacherWeight, teacherLoss));

        return weighted;
    }

    /// <summary>
    /// Calculates difficulty based only on teacher loss.
    /// </summary>
    private T CalculateTeacherLossDifficulty(TInput input, TOutput expectedOutput)
    {
        var teacherPrediction = _teacherModel!.Predict(input);
        var expectedVector = ConversionsHelper.ConvertToVector<T, TOutput>(expectedOutput);
        var predictionVector = ConversionsHelper.ConvertToVector<T, TOutput>(teacherPrediction);
        return _teacherModel.DefaultLossFunction.CalculateLoss(predictionVector, expectedVector);
    }

    /// <summary>
    /// Calculates difficulty based on confidence gap.
    /// </summary>
    private T CalculateConfidenceGapDifficulty(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput> studentModel)
    {
        var teacherConfidence = GetModelConfidence(_teacherModel!, input);
        var studentConfidence = GetModelConfidence(studentModel, input);

        // Higher teacher confidence + lower student confidence = learnable sample
        // Both low = very hard sample
        var gap = NumOps.Subtract(teacherConfidence, studentConfidence);
        var avgConfidence = NumOps.Divide(
            NumOps.Add(teacherConfidence, studentConfidence),
            NumOps.FromDouble(2.0));

        // Combine: low average confidence indicates hard sample
        // Large gap indicates learning opportunity
        return NumOps.Subtract(NumOps.One, avgConfidence);
    }

    /// <summary>
    /// Calculates combined difficulty using multiple metrics.
    /// </summary>
    private T CalculateCombinedDifficulty(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput> studentModel)
    {
        var lossGap = CalculateLossGapDifficulty(input, expectedOutput, studentModel);
        var teacherLoss = CalculateTeacherLossDifficulty(input, expectedOutput);

        // Weighted combination
        var weight = NumOps.FromDouble(0.6);
        var oneMinusWeight = NumOps.FromDouble(0.4);

        return NumOps.Add(
            NumOps.Multiply(weight, lossGap),
            NumOps.Multiply(oneMinusWeight, teacherLoss));
    }

    /// <summary>
    /// Gets model confidence for a prediction.
    /// </summary>
    private T GetModelConfidence(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        var prediction = model.Predict(input);

        // Try to get probabilities
        if (model is IProbabilisticModel<T, TInput, TOutput> probModel)
        {
            var probs = probModel.PredictProbabilities(input);
            return probs.Max();
        }

        // Fallback: use inverse of prediction magnitude as uncertainty
        if (prediction is Vector<T> vector)
        {
            var max = vector.Max();
            return NumOps.Compare(max, NumOps.Zero) > 0
                ? NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, max))
                : NumOps.FromDouble(0.5);
        }

        return NumOps.FromDouble(0.5);
    }
}

/// <summary>
/// Mode for transfer-based difficulty calculation.
/// </summary>
public enum TransferDifficultyMode
{
    /// <summary>
    /// Uses the loss gap between teacher and student models.
    /// </summary>
    LossGap,

    /// <summary>
    /// Uses only the teacher model's loss as difficulty.
    /// </summary>
    TeacherLoss,

    /// <summary>
    /// Uses the confidence gap between models.
    /// </summary>
    ConfidenceGap,

    /// <summary>
    /// Combines multiple metrics for robust estimation.
    /// </summary>
    Combined
}
