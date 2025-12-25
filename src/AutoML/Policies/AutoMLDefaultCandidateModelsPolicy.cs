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
            // 3D Point Cloud Tasks
            case AutoMLTaskFamily.PointCloudClassification:
                return Get3DPointCloudClassificationCandidates(preset);

            case AutoMLTaskFamily.PointCloudSegmentation:
                return Get3DPointCloudSegmentationCandidates(preset);

            case AutoMLTaskFamily.PointCloudCompletion:
                return Get3DPointCloudCompletionCandidates(preset);

            // 3D Volumetric Tasks
            case AutoMLTaskFamily.VolumetricClassification:
                return Get3DVolumetricClassificationCandidates(preset);

            case AutoMLTaskFamily.VolumetricSegmentation:
                return Get3DVolumetricSegmentationCandidates(preset);

            // 3D Mesh Tasks
            case AutoMLTaskFamily.MeshClassification:
                return Get3DMeshClassificationCandidates(preset);

            case AutoMLTaskFamily.MeshSegmentation:
                return Get3DMeshSegmentationCandidates(preset);

            // Neural Radiance Fields
            case AutoMLTaskFamily.RadianceFieldReconstruction:
                return GetRadianceFieldCandidates(preset);

            // 3D Detection and Depth
            case AutoMLTaskFamily.ThreeDObjectDetection:
                return Get3DObjectDetectionCandidates(preset);

            case AutoMLTaskFamily.DepthEstimation:
                return GetDepthEstimationCandidates(preset);


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
    /// <summary>
    /// Gets candidate models for 3D point cloud classification tasks.
    /// </summary>
    private static IReadOnlyList<ModelType> Get3DPointCloudClassificationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new[]
            {
                ModelType.PointNet
            },
            AutoMLBudgetPreset.Fast => new[]
            {
                ModelType.PointNet,
                ModelType.DGCNN
            },
            AutoMLBudgetPreset.Thorough => new[]
            {
                ModelType.PointNet,
                ModelType.PointNetPlusPlus,
                ModelType.DGCNN
            },
            _ => new[]
            {
                ModelType.PointNet,
                ModelType.PointNetPlusPlus,
                ModelType.DGCNN
            }
        };
    }

    /// <summary>
    /// Gets candidate models for 3D point cloud segmentation tasks.
    /// </summary>
    private static IReadOnlyList<ModelType> Get3DPointCloudSegmentationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new[]
            {
                ModelType.PointNet
            },
            AutoMLBudgetPreset.Fast => new[]
            {
                ModelType.PointNet,
                ModelType.PointNetPlusPlus
            },
            AutoMLBudgetPreset.Thorough => new[]
            {
                ModelType.PointNet,
                ModelType.PointNetPlusPlus,
                ModelType.DGCNN
            },
            _ => new[]
            {
                ModelType.PointNetPlusPlus,
                ModelType.DGCNN
            }
        };
    }

    /// <summary>
    /// Gets candidate models for 3D point cloud completion tasks.
    /// </summary>
    private static IReadOnlyList<ModelType> Get3DPointCloudCompletionCandidates(AutoMLBudgetPreset preset)
    {
        // Point cloud completion typically uses encoder-decoder architectures
        // PointNet++ with feature propagation is well-suited for this
        return preset switch
        {
            AutoMLBudgetPreset.CI => new[]
            {
                ModelType.PointNet
            },
            _ => new[]
            {
                ModelType.PointNetPlusPlus,
                ModelType.DGCNN
            }
        };
    }

    /// <summary>
    /// Gets candidate models for 3D volumetric classification tasks.
    /// </summary>
    private static IReadOnlyList<ModelType> Get3DVolumetricClassificationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new[]
            {
                ModelType.VoxelCNN
            },
            AutoMLBudgetPreset.Fast => new[]
            {
                ModelType.VoxelCNN
            },
            AutoMLBudgetPreset.Thorough => new[]
            {
                ModelType.VoxelCNN,
                ModelType.UNet3D
            },
            _ => new[]
            {
                ModelType.VoxelCNN
            }
        };
    }

    /// <summary>
    /// Gets candidate models for 3D volumetric segmentation tasks.
    /// </summary>
    private static IReadOnlyList<ModelType> Get3DVolumetricSegmentationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new[]
            {
                ModelType.UNet3D
            },
            _ => new[]
            {
                ModelType.UNet3D,
                ModelType.VoxelCNN
            }
        };
    }

    /// <summary>
    /// Gets candidate models for 3D mesh classification tasks.
    /// </summary>
    private static IReadOnlyList<ModelType> Get3DMeshClassificationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new[]
            {
                ModelType.MeshCNN
            },
            AutoMLBudgetPreset.Fast => new[]
            {
                ModelType.MeshCNN,
                ModelType.SpiralNetPlusPlus
            },
            AutoMLBudgetPreset.Thorough => new[]
            {
                ModelType.MeshCNN,
                ModelType.SpiralNetPlusPlus,
                ModelType.DiffusionNet
            },
            _ => new[]
            {
                ModelType.MeshCNN,
                ModelType.DiffusionNet
            }
        };
    }

    /// <summary>
    /// Gets candidate models for 3D mesh segmentation tasks.
    /// </summary>
    private static IReadOnlyList<ModelType> Get3DMeshSegmentationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new[]
            {
                ModelType.MeshCNN
            },
            _ => new[]
            {
                ModelType.MeshCNN,
                ModelType.DiffusionNet
            }
        };
    }

    /// <summary>
    /// Gets candidate models for neural radiance field reconstruction tasks.
    /// </summary>
    private static IReadOnlyList<ModelType> GetRadianceFieldCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new[]
            {
                ModelType.InstantNGP  // Fastest for CI
            },
            AutoMLBudgetPreset.Fast => new[]
            {
                ModelType.InstantNGP,
                ModelType.GaussianSplatting
            },
            AutoMLBudgetPreset.Thorough => new[]
            {
                ModelType.NeRF,
                ModelType.InstantNGP,
                ModelType.GaussianSplatting
            },
            _ => new[]
            {
                ModelType.InstantNGP,
                ModelType.GaussianSplatting
            }
        };
    }

    /// <summary>
    /// Gets candidate models for 3D object detection tasks.
    /// </summary>
    private static IReadOnlyList<ModelType> Get3DObjectDetectionCandidates(AutoMLBudgetPreset preset)
    {
        // 3D object detection often uses point cloud backbones with detection heads
        return preset switch
        {
            AutoMLBudgetPreset.CI => new[]
            {
                ModelType.PointNet
            },
            _ => new[]
            {
                ModelType.PointNetPlusPlus,
                ModelType.DGCNN,
                ModelType.VoxelCNN
            }
        };
    }

    /// <summary>
    /// Gets candidate models for depth estimation tasks.
    /// </summary>
    private static IReadOnlyList<ModelType> GetDepthEstimationCandidates(AutoMLBudgetPreset preset)
    {
        // Depth estimation typically uses encoder-decoder CNNs
        // For now, map to general neural network types
        return preset switch
        {
            AutoMLBudgetPreset.CI => new[]
            {
                ModelType.NeuralNetworkRegression
            },
            _ => new[]
            {
                ModelType.NeuralNetworkRegression,
                ModelType.UNet3D
            }
        };
    }
}

