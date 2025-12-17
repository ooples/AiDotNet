using System;
using AiDotNet.Enums;

namespace AiDotNet.AutoML.Policies;

/// <summary>
/// Provides default candidate model types for AutoML runs, based on the inferred or overridden task family.
/// </summary>
/// <remarks>
/// <para>
/// This policy is intended for the built-in AutoML strategies (random search, Bayesian optimization, etc.) and is
/// designed to be conservative: it prefers models that are known to work with the current evaluation pipeline.
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoML tries multiple model "families" to find a strong result. This list controls which
/// model families AutoML will try by default for a given problem type.
/// </para>
/// </remarks>
internal static class AutoMLDefaultCandidateModelsPolicy
{
    public static IReadOnlyList<ModelType> GetDefaultCandidates(AutoMLTaskFamily taskFamily, int featureCount)
    {
        return taskFamily switch
        {
            AutoMLTaskFamily.BinaryClassification => new[]
            {
                ModelType.LogisticRegression,
                ModelType.RandomForest,
                ModelType.GradientBoosting,
                ModelType.KNearestNeighbors,
                ModelType.SupportVectorRegression
            },

            AutoMLTaskFamily.MultiClassClassification => new[]
            {
                ModelType.MultinomialLogisticRegression,
                ModelType.RandomForest,
                ModelType.GradientBoosting,
                ModelType.KNearestNeighbors
            },

            AutoMLTaskFamily.TimeSeriesForecasting => new[]
            {
                ModelType.TimeSeriesRegression
            },

            AutoMLTaskFamily.Regression => featureCount == 1
                ? new[]
                {
                    ModelType.SimpleRegression,
                    ModelType.MultipleRegression,
                    ModelType.PolynomialRegression,
                    ModelType.RandomForest,
                    ModelType.GradientBoosting,
                    ModelType.KNearestNeighbors,
                    ModelType.SupportVectorRegression,
                    ModelType.NeuralNetworkRegression
                }
                : new[]
                {
                    ModelType.MultipleRegression,
                    ModelType.PolynomialRegression,
                    ModelType.RandomForest,
                    ModelType.GradientBoosting,
                    ModelType.KNearestNeighbors,
                    ModelType.SupportVectorRegression,
                    ModelType.NeuralNetworkRegression
                },

            _ => Array.Empty<ModelType>()
        };
    }
}
