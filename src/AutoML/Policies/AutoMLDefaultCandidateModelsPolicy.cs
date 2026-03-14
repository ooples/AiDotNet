using System;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Classification.Neighbors;
using AiDotNet.Classification.Trees;
using AiDotNet.Regression;
using AiDotNet.NeuralNetworks;
using AiDotNet.PointCloud.Models;
using AiDotNet.NeuralRadianceFields.Models;

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
    public static IReadOnlyList<Type> GetDefaultCandidates(AutoMLTaskFamily taskFamily, int featureCount)
    {
        return GetDefaultCandidates(taskFamily, featureCount, AutoMLBudgetPreset.Standard);
    }

    public static IReadOnlyList<Type> GetDefaultCandidates(AutoMLTaskFamily taskFamily, int featureCount, AutoMLBudgetPreset preset)
    {
        switch (taskFamily)
        {
            case AutoMLTaskFamily.BinaryClassification:
                return preset switch
                {
                    AutoMLBudgetPreset.CI => new Type[]
                    {
                        typeof(LogisticRegression<>),
                        typeof(RandomForestClassifier<>)
                    },
                    AutoMLBudgetPreset.Fast => new Type[]
                    {
                        typeof(LogisticRegression<>),
                        typeof(RandomForestClassifier<>),
                        typeof(GradientBoostingClassifier<>)
                    },
                    AutoMLBudgetPreset.Thorough => new Type[]
                    {
                        typeof(LogisticRegression<>),
                        typeof(RandomForestClassifier<>),
                        typeof(GradientBoostingClassifier<>),
                        typeof(DecisionTreeClassifier<>),
                        typeof(ExtraTreesClassifier<>),
                        typeof(KNeighborsClassifier<>)
                    },
                    _ => new Type[]
                    {
                        typeof(LogisticRegression<>),
                        typeof(RandomForestClassifier<>),
                        typeof(GradientBoostingClassifier<>),
                        typeof(KNeighborsClassifier<>)
                    }
                };

            case AutoMLTaskFamily.MultiClassClassification:
                return preset switch
                {
                    AutoMLBudgetPreset.CI => new Type[]
                    {
                        typeof(MultinomialLogisticRegression<>),
                        typeof(RandomForestClassifier<>)
                    },
                    AutoMLBudgetPreset.Fast => new Type[]
                    {
                        typeof(MultinomialLogisticRegression<>),
                        typeof(RandomForestClassifier<>),
                        typeof(GradientBoostingClassifier<>)
                    },
                    AutoMLBudgetPreset.Thorough => new Type[]
                    {
                        typeof(MultinomialLogisticRegression<>),
                        typeof(RandomForestClassifier<>),
                        typeof(GradientBoostingClassifier<>),
                        typeof(KNeighborsClassifier<>)
                    },
                    _ => new Type[]
                    {
                        typeof(MultinomialLogisticRegression<>),
                        typeof(RandomForestClassifier<>),
                        typeof(GradientBoostingClassifier<>),
                        typeof(KNeighborsClassifier<>)
                    }
                };

            case AutoMLTaskFamily.TimeSeriesAnomalyDetection:
                return GetDefaultCandidates(AutoMLTaskFamily.BinaryClassification, featureCount, preset);

            case AutoMLTaskFamily.TimeSeriesForecasting:
                return new Type[]
                {
                    typeof(TimeSeriesRegression<>)
                };

            case AutoMLTaskFamily.Ranking:
            case AutoMLTaskFamily.Recommendation:
                return GetRegressionCandidates(featureCount, preset);

            case AutoMLTaskFamily.Regression:
                return GetRegressionCandidates(featureCount, preset);

            case AutoMLTaskFamily.PointCloudClassification:
                return Get3DPointCloudClassificationCandidates(preset);

            case AutoMLTaskFamily.PointCloudSegmentation:
                return Get3DPointCloudSegmentationCandidates(preset);

            case AutoMLTaskFamily.PointCloudCompletion:
                return Get3DPointCloudCompletionCandidates(preset);

            case AutoMLTaskFamily.VolumetricClassification:
                return Get3DVolumetricClassificationCandidates(preset);

            case AutoMLTaskFamily.VolumetricSegmentation:
                return Get3DVolumetricSegmentationCandidates(preset);

            case AutoMLTaskFamily.MeshClassification:
                return Get3DMeshClassificationCandidates(preset);

            case AutoMLTaskFamily.MeshSegmentation:
                return Get3DMeshSegmentationCandidates(preset);

            case AutoMLTaskFamily.RadianceFieldReconstruction:
                return GetRadianceFieldCandidates(preset);

            case AutoMLTaskFamily.ThreeDObjectDetection:
                return Get3DObjectDetectionCandidates(preset);

            case AutoMLTaskFamily.DepthEstimation:
                return GetDepthEstimationCandidates(preset);

            default:
                return Array.Empty<Type>();
        }
    }

    private static IReadOnlyList<Type> GetRegressionCandidates(int featureCount, AutoMLBudgetPreset preset)
    {
        if (preset == AutoMLBudgetPreset.CI)
        {
            return featureCount == 1
                ? new Type[] { typeof(SimpleRegression<>), typeof(RandomForestRegression<>) }
                : new Type[] { typeof(MultipleRegression<>), typeof(RandomForestRegression<>) };
        }

        if (preset == AutoMLBudgetPreset.Fast)
        {
            return featureCount == 1
                ? new Type[]
                {
                    typeof(SimpleRegression<>),
                    typeof(MultipleRegression<>),
                    typeof(PolynomialRegression<>),
                    typeof(RandomForestRegression<>),
                    typeof(GradientBoostingRegression<>),
                    typeof(KNearestNeighborsRegression<>),
                    typeof(SupportVectorRegression<>)
                }
                : new Type[]
                {
                    typeof(MultipleRegression<>),
                    typeof(PolynomialRegression<>),
                    typeof(RandomForestRegression<>),
                    typeof(GradientBoostingRegression<>),
                    typeof(KNearestNeighborsRegression<>),
                    typeof(SupportVectorRegression<>)
                };
        }

        if (preset == AutoMLBudgetPreset.Thorough)
        {
            return featureCount == 1
                ? new Type[]
                {
                    typeof(SimpleRegression<>),
                    typeof(MultipleRegression<>),
                    typeof(PolynomialRegression<>),
                    typeof(BayesianRegression<>),
                    typeof(GaussianProcessRegression<>),
                    typeof(KernelRidgeRegression<>),
                    typeof(RandomForestRegression<>),
                    typeof(ExtremelyRandomizedTreesRegression<>),
                    typeof(DecisionTreeRegression<>),
                    typeof(ConditionalInferenceTreeRegression<>),
                    typeof(M5ModelTree<>),
                    typeof(AdaBoostR2Regression<>),
                    typeof(QuantileRegressionForests<>),
                    typeof(GradientBoostingRegression<>),
                    typeof(KNearestNeighborsRegression<>),
                    typeof(SupportVectorRegression<>),
                    typeof(RadialBasisFunctionRegression<>),
                    typeof(GeneralizedAdditiveModel<>),
                    typeof(MultilayerPerceptronRegression<>),
                    typeof(QuantileRegression<>),
                    typeof(RobustRegression<>),
                    typeof(NeuralNetworkRegression<>)
                }
                : new Type[]
                {
                    typeof(MultipleRegression<>),
                    typeof(PolynomialRegression<>),
                    typeof(BayesianRegression<>),
                    typeof(GaussianProcessRegression<>),
                    typeof(KernelRidgeRegression<>),
                    typeof(RandomForestRegression<>),
                    typeof(ExtremelyRandomizedTreesRegression<>),
                    typeof(DecisionTreeRegression<>),
                    typeof(ConditionalInferenceTreeRegression<>),
                    typeof(M5ModelTree<>),
                    typeof(AdaBoostR2Regression<>),
                    typeof(QuantileRegressionForests<>),
                    typeof(GradientBoostingRegression<>),
                    typeof(KNearestNeighborsRegression<>),
                    typeof(SupportVectorRegression<>),
                    typeof(RadialBasisFunctionRegression<>),
                    typeof(GeneralizedAdditiveModel<>),
                    typeof(MultilayerPerceptronRegression<>),
                    typeof(QuantileRegression<>),
                    typeof(RobustRegression<>),
                    typeof(NeuralNetworkRegression<>)
                };
        }

        return featureCount == 1
            ? new Type[]
            {
                typeof(SimpleRegression<>),
                typeof(MultipleRegression<>),
                typeof(PolynomialRegression<>),
                typeof(RandomForestRegression<>),
                typeof(GradientBoostingRegression<>),
                typeof(KNearestNeighborsRegression<>),
                typeof(SupportVectorRegression<>),
                typeof(BayesianRegression<>),
                typeof(KernelRidgeRegression<>),
                typeof(NeuralNetworkRegression<>)
            }
            : new Type[]
            {
                typeof(MultipleRegression<>),
                typeof(PolynomialRegression<>),
                typeof(RandomForestRegression<>),
                typeof(GradientBoostingRegression<>),
                typeof(KNearestNeighborsRegression<>),
                typeof(SupportVectorRegression<>),
                typeof(BayesianRegression<>),
                typeof(KernelRidgeRegression<>),
                typeof(NeuralNetworkRegression<>)
            };
    }

    private static IReadOnlyList<Type> Get3DPointCloudClassificationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new Type[] { typeof(PointNet<>) },
            AutoMLBudgetPreset.Fast => new Type[] { typeof(PointNet<>), typeof(DGCNN<>) },
            AutoMLBudgetPreset.Thorough => new Type[] { typeof(PointNet<>), typeof(PointNetPlusPlus<>), typeof(DGCNN<>) },
            _ => new Type[] { typeof(PointNet<>), typeof(PointNetPlusPlus<>), typeof(DGCNN<>) }
        };
    }

    private static IReadOnlyList<Type> Get3DPointCloudSegmentationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new Type[] { typeof(PointNet<>) },
            AutoMLBudgetPreset.Fast => new Type[] { typeof(PointNet<>), typeof(PointNetPlusPlus<>) },
            AutoMLBudgetPreset.Thorough => new Type[] { typeof(PointNet<>), typeof(PointNetPlusPlus<>), typeof(DGCNN<>) },
            _ => new Type[] { typeof(PointNetPlusPlus<>), typeof(DGCNN<>) }
        };
    }

    private static IReadOnlyList<Type> Get3DPointCloudCompletionCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new Type[] { typeof(PointNet<>) },
            _ => new Type[] { typeof(PointNetPlusPlus<>), typeof(DGCNN<>) }
        };
    }

    private static IReadOnlyList<Type> Get3DVolumetricClassificationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new Type[] { typeof(VoxelCNN<>) },
            AutoMLBudgetPreset.Fast => new Type[] { typeof(VoxelCNN<>) },
            AutoMLBudgetPreset.Thorough => new Type[] { typeof(VoxelCNN<>), typeof(UNet3D<>) },
            _ => new Type[] { typeof(VoxelCNN<>) }
        };
    }

    private static IReadOnlyList<Type> Get3DVolumetricSegmentationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new Type[] { typeof(UNet3D<>) },
            _ => new Type[] { typeof(UNet3D<>), typeof(VoxelCNN<>) }
        };
    }

    private static IReadOnlyList<Type> Get3DMeshClassificationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new Type[] { typeof(MeshCNN<>) },
            AutoMLBudgetPreset.Fast => new Type[] { typeof(MeshCNN<>), typeof(SpiralNet<>) },
            AutoMLBudgetPreset.Thorough => new Type[] { typeof(MeshCNN<>), typeof(SpiralNet<>) },
            _ => new Type[] { typeof(MeshCNN<>), typeof(SpiralNet<>) }
        };
    }

    private static IReadOnlyList<Type> Get3DMeshSegmentationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new Type[] { typeof(MeshCNN<>) },
            _ => new Type[] { typeof(MeshCNN<>), typeof(SpiralNet<>) }
        };
    }

    private static IReadOnlyList<Type> GetRadianceFieldCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new Type[] { typeof(InstantNGP<>) },
            AutoMLBudgetPreset.Fast => new Type[] { typeof(InstantNGP<>), typeof(GaussianSplatting<>) },
            AutoMLBudgetPreset.Thorough => new Type[] { typeof(NeRF<>), typeof(InstantNGP<>), typeof(GaussianSplatting<>) },
            _ => new Type[] { typeof(InstantNGP<>), typeof(GaussianSplatting<>) }
        };
    }

    private static IReadOnlyList<Type> Get3DObjectDetectionCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new Type[] { typeof(PointNet<>) },
            _ => new Type[] { typeof(PointNetPlusPlus<>), typeof(DGCNN<>), typeof(VoxelCNN<>) }
        };
    }

    private static IReadOnlyList<Type> GetDepthEstimationCandidates(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => new Type[] { typeof(NeuralNetworkRegression<>) },
            _ => new Type[] { typeof(NeuralNetworkRegression<>), typeof(UNet3D<>) }
        };
    }
}
