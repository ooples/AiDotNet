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
        return GetDefaultCandidates(taskFamily, featureCount, AutoMLBudgetPreset.Standard);
    }

    public static IReadOnlyList<ModelType> GetDefaultCandidates(AutoMLTaskFamily taskFamily, int featureCount, AutoMLBudgetPreset preset)
    {
        switch (taskFamily)
        {
            case AutoMLTaskFamily.BinaryClassification:
                return preset switch
                {
                    AutoMLBudgetPreset.CI => new[]
                    {
                        ModelType.LogisticRegression,
                        ModelType.RandomForest
                    },
                    AutoMLBudgetPreset.Fast => new[]
                    {
                        ModelType.LogisticRegression,
                        ModelType.RandomForest,
                        ModelType.GradientBoosting
                    },
                    AutoMLBudgetPreset.Thorough => new[]
                    {
                        ModelType.LogisticRegression,
                        ModelType.RandomForest,
                        ModelType.GradientBoosting,
                        ModelType.DecisionTree,
                        ModelType.ExtremelyRandomizedTrees,
                        ModelType.KNearestNeighbors
                    },
                    _ => new[]
                    {
                        ModelType.LogisticRegression,
                        ModelType.RandomForest,
                        ModelType.GradientBoosting,
                        ModelType.KNearestNeighbors
                    }
                };

            case AutoMLTaskFamily.MultiClassClassification:
                return preset switch
                {
                    AutoMLBudgetPreset.CI => new[]
                    {
                        ModelType.MultinomialLogisticRegression,
                        ModelType.RandomForest
                    },
                    AutoMLBudgetPreset.Fast => new[]
                    {
                        ModelType.MultinomialLogisticRegression,
                        ModelType.RandomForest,
                        ModelType.GradientBoosting
                    },
                    AutoMLBudgetPreset.Thorough => new[]
                    {
                        ModelType.MultinomialLogisticRegression,
                        ModelType.RandomForest,
                        ModelType.GradientBoosting,
                        ModelType.KNearestNeighbors
                    },
                    _ => new[]
                    {
                        ModelType.MultinomialLogisticRegression,
                        ModelType.RandomForest,
                        ModelType.GradientBoosting,
                        ModelType.KNearestNeighbors
                    }
                };

            case AutoMLTaskFamily.TimeSeriesAnomalyDetection:
                // Default to the same conservative candidate set as binary classification.
                // Many anomaly detection setups are supervised binary problems (anomaly vs normal),
                // and we keep defaults focused on models supported by the current evaluator pipeline.
                return GetDefaultCandidates(AutoMLTaskFamily.BinaryClassification, featureCount, preset);

            case AutoMLTaskFamily.TimeSeriesForecasting:
                return new[]
                {
                    ModelType.TimeSeriesRegression
                };

            case AutoMLTaskFamily.Ranking:
            case AutoMLTaskFamily.Recommendation:
                // Ranking/recommendation are often framed as learning-to-rank with scalar relevance targets.
                // The built-in defaults treat this as a supervised regression-style objective and rely on
                // ranking metrics (NDCG/MAP/MRR) for scoring.
                return GetRegressionCandidates(featureCount, preset);

            case AutoMLTaskFamily.Regression:
                return GetRegressionCandidates(featureCount, preset);

            default:
                return Array.Empty<ModelType>();
        }
    }

    private static IReadOnlyList<ModelType> GetRegressionCandidates(int featureCount, AutoMLBudgetPreset preset)
    {
        if (preset == AutoMLBudgetPreset.CI)
        {
            return featureCount == 1
                ? new[] { ModelType.SimpleRegression, ModelType.RandomForest }
                : new[] { ModelType.MultipleRegression, ModelType.RandomForest };
        }

        if (preset == AutoMLBudgetPreset.Fast)
        {
            return featureCount == 1
                ? new[]
                {
                    ModelType.SimpleRegression,
                    ModelType.MultipleRegression,
                    ModelType.PolynomialRegression,
                    ModelType.RandomForest,
                    ModelType.GradientBoosting,
                    ModelType.KNearestNeighbors,
                    ModelType.SupportVectorRegression
                }
                : new[]
                {
                    ModelType.MultipleRegression,
                    ModelType.PolynomialRegression,
                    ModelType.RandomForest,
                    ModelType.GradientBoosting,
                    ModelType.KNearestNeighbors,
                    ModelType.SupportVectorRegression
                };
        }

        if (preset == AutoMLBudgetPreset.Thorough)
        {
            return featureCount == 1
                ? new[]
                {
                    ModelType.SimpleRegression,
                    ModelType.MultipleRegression,
                    ModelType.PolynomialRegression,
                    ModelType.BayesianRegression,
                    ModelType.GaussianProcessRegression,
                    ModelType.KernelRidgeRegression,
                    ModelType.RandomForest,
                    ModelType.ExtremelyRandomizedTrees,
                    ModelType.DecisionTree,
                    ModelType.ConditionalInferenceTree,
                    ModelType.M5ModelTree,
                    ModelType.AdaBoostR2,
                    ModelType.QuantileRegressionForests,
                    ModelType.GradientBoosting,
                    ModelType.KNearestNeighbors,
                    ModelType.SupportVectorRegression,
                    ModelType.RadialBasisFunctionRegression,
                    ModelType.GeneralizedAdditiveModelRegression,
                    ModelType.MultilayerPerceptronRegression,
                    ModelType.QuantileRegression,
                    ModelType.RobustRegression,
                    ModelType.NeuralNetworkRegression
                }
                : new[]
                {
                    ModelType.MultipleRegression,
                    ModelType.PolynomialRegression,
                    ModelType.BayesianRegression,
                    ModelType.GaussianProcessRegression,
                    ModelType.KernelRidgeRegression,
                    ModelType.RandomForest,
                    ModelType.ExtremelyRandomizedTrees,
                    ModelType.DecisionTree,
                    ModelType.ConditionalInferenceTree,
                    ModelType.M5ModelTree,
                    ModelType.AdaBoostR2,
                    ModelType.QuantileRegressionForests,
                    ModelType.GradientBoosting,
                    ModelType.KNearestNeighbors,
                    ModelType.SupportVectorRegression,
                    ModelType.RadialBasisFunctionRegression,
                    ModelType.GeneralizedAdditiveModelRegression,
                    ModelType.MultilayerPerceptronRegression,
                    ModelType.QuantileRegression,
                    ModelType.RobustRegression,
                    ModelType.NeuralNetworkRegression
                };
        }

        return featureCount == 1
            ? new[]
            {
                ModelType.SimpleRegression,
                ModelType.MultipleRegression,
                ModelType.PolynomialRegression,
                ModelType.RandomForest,
                ModelType.GradientBoosting,
                ModelType.KNearestNeighbors,
                ModelType.SupportVectorRegression,
                ModelType.BayesianRegression,
                ModelType.KernelRidgeRegression,
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
                ModelType.BayesianRegression,
                ModelType.KernelRidgeRegression,
                ModelType.NeuralNetworkRegression
            };
    }
}
