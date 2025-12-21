using AiDotNet.Enums;
using AiDotNet.Evaluation;

namespace AiDotNet.AutoML.Policies;

/// <summary>
/// Infers an <see cref="AutoMLTaskFamily"/> from training targets when possible.
/// </summary>
/// <remarks>
/// <para>
/// AutoML uses task-family inference to choose default metrics and candidate model sets when the user does not
/// explicitly provide overrides.
/// </para>
/// <para>
/// <b>For Beginners:</b> If your labels look like 0/1, AutoML treats it as classification. If your labels look like
/// real numbers, AutoML treats it as regression.
/// </para>
/// </remarks>
internal static class AutoMLTaskFamilyInference
{
    public static AutoMLTaskFamily InferFromTargets<T, TOutput>(TOutput targets)
    {
        var predictionType = PredictionTypeInference.InferFromTargets<T, TOutput>(targets);
        return predictionType switch
        {
            PredictionType.Binary => AutoMLTaskFamily.BinaryClassification,
            PredictionType.MultiClass => AutoMLTaskFamily.MultiClassClassification,
            PredictionType.MultiLabel => AutoMLTaskFamily.MultiLabelClassification,
            _ => AutoMLTaskFamily.Regression
        };
    }
}
