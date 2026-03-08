using AiDotNet.Classification.DiscriminantAnalysis;
using AiDotNet.Classification.Linear;
using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Classification.SVM;
using AiDotNet.Classification.Trees;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Tests that verify Serialize → Deserialize round-trip for ALL classifier families.
/// This is the #1 bug category: classifiers that don't override Serialize/Deserialize
/// lose their trained state (weights, biases, tree structures, etc.) during DeepCopy,
/// which the optimizer uses internally.
///
/// BUG PATTERN: ClassifierBase.Serialize() only saves ClassLabels, NumClasses, NumFeatures, TaskType.
/// Any classifier with additional state (Weights, Intercept, support vectors, feature probs, etc.)
/// MUST override Serialize/Deserialize or predictions will silently break after DeepCopy.
/// </summary>
public class ClassifierSerializationRoundTripTests
{
    #region Linear Classifiers — Weights + Intercept (Fixed in LinearClassifierBase)

    [Fact]
    public void RidgeClassifier_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateBinaryData(60, 3, separation: 5.0, seed: 42);
        var classifier = new RidgeClassifier<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 100);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new RidgeClassifier<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "RidgeClassifier");
    }

    [Fact]
    public void SGDClassifier_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 5.0, seed: 77);
        var classifier = new SGDClassifier<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 200);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new SGDClassifier<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "SGDClassifier");
    }

    [Fact]
    public void PerceptronClassifier_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 6.0, seed: 55);
        var classifier = new PerceptronClassifier<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 300);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new PerceptronClassifier<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "PerceptronClassifier");
    }

    [Fact]
    public void PassiveAggressiveClassifier_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 5.0, seed: 33);
        var classifier = new PassiveAggressiveClassifier<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 400);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new PassiveAggressiveClassifier<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "PassiveAggressiveClassifier");
    }

    #endregion

    #region Naive Bayes Classifiers — LogPriors + class-specific params

    [Fact]
    public void GaussianNaiveBayes_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 4.0, seed: 42);
        var classifier = new GaussianNaiveBayes<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 500);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new GaussianNaiveBayes<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "GaussianNaiveBayes");
    }

    [Fact]
    public void MultinomialNaiveBayes_SerializeRoundTrip_PredictionsMatch()
    {
        // MultinomialNB needs non-negative features (counts/frequencies)
        var (trainX, trainY) = CreateNonNegativeBinaryData(80, 4, seed: 42);
        var classifier = new MultinomialNaiveBayes<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateNonNegativeMatrix(10, 4, seed: 600);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new MultinomialNaiveBayes<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "MultinomialNaiveBayes");
    }

    [Fact]
    public void BernoulliNaiveBayes_SerializeRoundTrip_PredictionsMatch()
    {
        // BernoulliNB expects binary features
        var (trainX, trainY) = CreateBinaryFeatureBinaryData(80, 5, seed: 42);
        var classifier = new BernoulliNaiveBayes<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateBinaryFeatureMatrix(10, 5, seed: 700);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new BernoulliNaiveBayes<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "BernoulliNaiveBayes");
    }

    #endregion

    #region SVM Classifiers — support vectors, weights, bias

    [Fact]
    public void LinearSVC_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 5.0, seed: 42);
        var classifier = new LinearSupportVectorClassifier<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 800);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new LinearSupportVectorClassifier<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "LinearSupportVectorClassifier");
    }

    #endregion

    #region Tree Classifiers — tree structure

    [Fact]
    public void DecisionTreeClassifier_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 4.0, seed: 42);
        var classifier = new DecisionTreeClassifier<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 900);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new DecisionTreeClassifier<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "DecisionTreeClassifier");
    }

    #endregion

    #region Accuracy Validation — classifiers must learn, not just serialize

    [Fact]
    public void RidgeClassifier_TrainAndPredict_AchievesReasonableAccuracy()
    {
        var (trainX, trainY) = CreateBinaryData(100, 3, separation: 5.0, seed: 42);
        var classifier = new RidgeClassifier<double>();
        classifier.Train(trainX, trainY);

        var (testX, testY) = CreateBinaryData(40, 3, separation: 5.0, seed: 999);
        var predictions = classifier.Predict(testX);

        double accuracy = ComputeAccuracy(predictions, testY);
        Assert.True(accuracy > 0.70,
            $"RidgeClassifier accuracy {accuracy:P1} too low for linearly separable data (expected > 70%)");
    }

    [Fact]
    public void DecisionTreeClassifier_TrainAndPredict_AchievesReasonableAccuracy()
    {
        var (trainX, trainY) = CreateBinaryData(100, 3, separation: 4.0, seed: 42);
        var classifier = new DecisionTreeClassifier<double>();
        classifier.Train(trainX, trainY);

        var (testX, testY) = CreateBinaryData(40, 3, separation: 4.0, seed: 999);
        var predictions = classifier.Predict(testX);

        double accuracy = ComputeAccuracy(predictions, testY);
        Assert.True(accuracy > 0.70,
            $"DecisionTreeClassifier accuracy {accuracy:P1} too low for separable data (expected > 70%)");
    }

    [Fact]
    public void GaussianNaiveBayes_TrainAndPredict_AchievesReasonableAccuracy()
    {
        var (trainX, trainY) = CreateBinaryData(120, 3, separation: 4.0, seed: 42);
        var classifier = new GaussianNaiveBayes<double>();
        classifier.Train(trainX, trainY);

        var (testX, testY) = CreateBinaryData(40, 3, separation: 4.0, seed: 999);
        var predictions = classifier.Predict(testX);

        double accuracy = ComputeAccuracy(predictions, testY);
        Assert.True(accuracy > 0.60,
            $"GaussianNaiveBayes accuracy {accuracy:P1} too low for Gaussian data (expected > 60%)");
    }

    [Fact]
    public void SGDClassifier_TrainAndPredict_AchievesReasonableAccuracy()
    {
        var (trainX, trainY) = CreateBinaryData(100, 3, separation: 5.0, seed: 42);
        var classifier = new SGDClassifier<double>();
        classifier.Train(trainX, trainY);

        var (testX, testY) = CreateBinaryData(40, 3, separation: 5.0, seed: 999);
        var predictions = classifier.Predict(testX);

        double accuracy = ComputeAccuracy(predictions, testY);
        Assert.True(accuracy > 0.60,
            $"SGDClassifier accuracy {accuracy:P1} too low for linearly separable data (expected > 60%)");
    }

    #endregion

    #region DeepCopy round-trip — tests the exact path the optimizer uses

    [Fact]
    public void RidgeClassifier_DeepCopy_PreservesTrainedState()
    {
        var (trainX, trainY) = CreateBinaryData(60, 3, separation: 5.0, seed: 42);
        var classifier = new RidgeClassifier<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 100);
        var original = classifier.Predict(testX);

        // DeepCopy is what the optimizer calls internally
        var copy = classifier.DeepCopy();
        var copyPreds = copy.Predict(testX);

        AssertPredictionsMatch(original, (Vector<double>)copyPreds, "RidgeClassifier DeepCopy");
    }

    [Fact]
    public void GaussianNaiveBayes_DeepCopy_PreservesTrainedState()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 4.0, seed: 42);
        var classifier = new GaussianNaiveBayes<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 100);
        var original = classifier.Predict(testX);

        var copy = classifier.DeepCopy();
        var copyPreds = copy.Predict(testX);

        AssertPredictionsMatch(original, (Vector<double>)copyPreds, "GaussianNaiveBayes DeepCopy");
    }

    #endregion

    #region Discriminant Analysis — ClassMeans + Covariance + Priors

    [Fact]
    public void LDA_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 6.0, seed: 42);
        var classifier = new LinearDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 1000);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new LinearDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "LinearDiscriminantAnalysis");
    }

    [Fact]
    public void QDA_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 6.0, seed: 42);
        var classifier = new QuadraticDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 1100);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new QuadraticDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "QuadraticDiscriminantAnalysis");
    }

    [Fact]
    public void LDA_DeepCopy_PreservesTrainedState()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 6.0, seed: 42);
        var classifier = new LinearDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 1000);
        var original = classifier.Predict(testX);

        var copy = classifier.DeepCopy();
        var copyPreds = copy.Predict(testX);

        AssertPredictionsMatch(original, (Vector<double>)copyPreds, "LDA DeepCopy");
    }

    [Fact]
    public void QDA_DeepCopy_PreservesTrainedState()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 6.0, seed: 42);
        var classifier = new QuadraticDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 1100);
        var original = classifier.Predict(testX);

        var copy = classifier.DeepCopy();
        var copyPreds = copy.Predict(testX);

        AssertPredictionsMatch(original, (Vector<double>)copyPreds, "QDA DeepCopy");
    }

    [Fact]
    public void LDA_TrainAndPredict_AchievesReasonableAccuracy()
    {
        var (trainX, trainY) = CreateBinaryData(100, 3, separation: 6.0, seed: 42);
        var classifier = new LinearDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        classifier.Train(trainX, trainY);

        var (testX, testY) = CreateBinaryData(40, 3, separation: 6.0, seed: 999);
        var predictions = classifier.Predict(testX);

        double accuracy = ComputeAccuracy(predictions, testY);
        Assert.True(accuracy > 0.85,
            $"LDA accuracy {accuracy:P1} too low for well-separated data (expected > 85%)");
    }

    [Fact]
    public void QDA_TrainAndPredict_AchievesReasonableAccuracy()
    {
        var (trainX, trainY) = CreateBinaryData(100, 3, separation: 6.0, seed: 42);
        var classifier = new QuadraticDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        classifier.Train(trainX, trainY);

        var (testX, testY) = CreateBinaryData(40, 3, separation: 6.0, seed: 999);
        var predictions = classifier.Predict(testX);

        double accuracy = ComputeAccuracy(predictions, testY);
        Assert.True(accuracy > 0.85,
            $"QDA accuracy {accuracy:P1} too low for well-separated data (expected > 85%)");
    }

    [Fact]
    public void LDA_PredictProbabilities_SumsToOne()
    {
        var (trainX, trainY) = CreateBinaryData(80, 3, separation: 6.0, seed: 42);
        var classifier = new LinearDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        classifier.Train(trainX, trainY);

        var probs = classifier.PredictProbabilities(trainX);

        for (int i = 0; i < trainX.Rows; i++)
        {
            double sum = 0;
            for (int c = 0; c < probs.Columns; c++)
            {
                sum += probs[i, c];
                Assert.True(probs[i, c] >= -1e-10 && probs[i, c] <= 1.0 + 1e-10,
                    $"LDA probability at [{i},{c}] = {probs[i, c]} should be in [0,1]");
            }
            Assert.True(Math.Abs(sum - 1.0) < 1e-6,
                $"LDA probabilities for sample {i} sum to {sum}, should be 1.0");
        }
    }

    #endregion

    #region Complement & Categorical Naive Bayes — complementLogProbs + categoryLogProbs

    [Fact]
    public void ComplementNaiveBayes_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateNonNegativeBinaryData(80, 4, seed: 42);
        var classifier = new ComplementNaiveBayes<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateNonNegativeMatrix(10, 4, seed: 1200);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new ComplementNaiveBayes<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "ComplementNaiveBayes");
    }

    [Fact]
    public void ComplementNaiveBayes_DeepCopy_PreservesTrainedState()
    {
        var (trainX, trainY) = CreateNonNegativeBinaryData(80, 4, seed: 42);
        var classifier = new ComplementNaiveBayes<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateNonNegativeMatrix(10, 4, seed: 1200);
        var original = classifier.Predict(testX);

        var copy = classifier.DeepCopy();
        var copyPreds = copy.Predict(testX);

        AssertPredictionsMatch(original, (Vector<double>)copyPreds, "ComplementNB DeepCopy");
    }

    [Fact]
    public void CategoricalNaiveBayes_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateCategoricalBinaryData(80, 3, seed: 42);
        var classifier = new CategoricalNaiveBayes<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateCategoricalMatrix(10, 3, seed: 1300);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new CategoricalNaiveBayes<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "CategoricalNaiveBayes");
    }

    [Fact]
    public void CategoricalNaiveBayes_DeepCopy_PreservesTrainedState()
    {
        var (trainX, trainY) = CreateCategoricalBinaryData(80, 3, seed: 42);
        var classifier = new CategoricalNaiveBayes<double>();
        classifier.Train(trainX, trainY);

        var testX = CreateCategoricalMatrix(10, 3, seed: 1300);
        var original = classifier.Predict(testX);

        var copy = classifier.DeepCopy();
        var copyPreds = copy.Predict(testX);

        AssertPredictionsMatch(original, (Vector<double>)copyPreds, "CategoricalNB DeepCopy");
    }

    #endregion

    #region Full SVM Serialize Round-trips

    [Fact]
    public void SVC_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateBinaryData(60, 2, separation: 5.0, seed: 42);
        var classifier = new SupportVectorClassifier<double>(new SVMOptions<double>
        {
            C = 1.0,
            Kernel = KernelType.RBF,
            MaxIterations = 200,
            Tolerance = 1e-3,
            Seed = 42
        });
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 2, seed: 1600);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new SupportVectorClassifier<double>(new SVMOptions<double>
        {
            C = 1.0,
            Kernel = KernelType.RBF,
            MaxIterations = 200,
            Tolerance = 1e-3,
            Seed = 42
        });
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "SupportVectorClassifier");
    }

    [Fact]
    public void NuSVC_SerializeRoundTrip_PredictionsMatch()
    {
        var (trainX, trainY) = CreateBinaryData(60, 2, separation: 5.0, seed: 42);
        var classifier = new NuSupportVectorClassifier<double>(new SVMOptions<double>
        {
            C = 1.0,
            Kernel = KernelType.RBF,
            MaxIterations = 200,
            Tolerance = 1e-3,
            Seed = 42
        });
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 2, seed: 1700);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new NuSupportVectorClassifier<double>(new SVMOptions<double>
        {
            C = 1.0,
            Kernel = KernelType.RBF,
            MaxIterations = 200,
            Tolerance = 1e-3,
            Seed = 42
        });
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "NuSupportVectorClassifier");
    }

    [Fact]
    public void SVC_DeepCopy_PreservesTrainedState()
    {
        var (trainX, trainY) = CreateBinaryData(60, 2, separation: 5.0, seed: 42);
        var classifier = new SupportVectorClassifier<double>(new SVMOptions<double>
        {
            C = 1.0,
            Kernel = KernelType.RBF,
            MaxIterations = 200,
            Tolerance = 1e-3,
            Seed = 42
        });
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 2, seed: 1600);
        var original = classifier.Predict(testX);

        var copy = classifier.DeepCopy();
        var copyPreds = copy.Predict(testX);

        AssertPredictionsMatch(original, (Vector<double>)copyPreds, "SVC DeepCopy");
    }

    #endregion

    #region SVM Accuracy Validation

    [Fact]
    public void SVC_TrainAndPredict_AchievesReasonableAccuracy()
    {
        var (trainX, trainY) = CreateBinaryData(80, 2, separation: 5.0, seed: 42);
        var classifier = new SupportVectorClassifier<double>(new SVMOptions<double>
        {
            C = 1.0,
            Kernel = KernelType.RBF,
            MaxIterations = 200,
            Tolerance = 1e-3,
            Seed = 42
        });
        classifier.Train(trainX, trainY);

        var (testX, testY) = CreateBinaryData(40, 2, separation: 5.0, seed: 999);
        var predictions = classifier.Predict(testX);

        double accuracy = ComputeAccuracy(predictions, testY);
        Assert.True(accuracy > 0.70,
            $"SVC accuracy {accuracy:P1} too low for well-separated data (expected > 70%)");
    }

    [Fact]
    public void NuSVC_TrainAndPredict_AchievesReasonableAccuracy()
    {
        var (trainX, trainY) = CreateBinaryData(80, 2, separation: 5.0, seed: 42);
        var classifier = new NuSupportVectorClassifier<double>(new SVMOptions<double>
        {
            C = 1.0,
            Kernel = KernelType.RBF,
            MaxIterations = 200,
            Tolerance = 1e-3,
            Seed = 42
        });
        classifier.Train(trainX, trainY);

        var (testX, testY) = CreateBinaryData(40, 2, separation: 5.0, seed: 999);
        var predictions = classifier.Predict(testX);

        double accuracy = ComputeAccuracy(predictions, testY);
        Assert.True(accuracy > 0.60,
            $"NuSVC accuracy {accuracy:P1} too low for well-separated data (expected > 60%)");
    }

    [Fact]
    public void LinearSVC_TrainAndPredict_AchievesReasonableAccuracy()
    {
        var (trainX, trainY) = CreateBinaryData(100, 3, separation: 5.0, seed: 42);
        var classifier = new LinearSupportVectorClassifier<double>();
        classifier.Train(trainX, trainY);

        var (testX, testY) = CreateBinaryData(40, 3, separation: 5.0, seed: 999);
        var predictions = classifier.Predict(testX);

        double accuracy = ComputeAccuracy(predictions, testY);
        Assert.True(accuracy > 0.70,
            $"LinearSVC accuracy {accuracy:P1} too low for linearly separable data (expected > 70%)");
    }

    #endregion

    #region Multi-class Serialize Round-trips

    [Fact]
    public void LDA_MultiClass_SerializeRoundTrip()
    {
        var (trainX, trainY) = CreateMultiClassData(90, 3, numClasses: 3, separation: 6.0, seed: 42);
        var classifier = new LinearDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 1400);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new LinearDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "LDA MultiClass");
    }

    [Fact]
    public void QDA_MultiClass_SerializeRoundTrip()
    {
        var (trainX, trainY) = CreateMultiClassData(90, 3, numClasses: 3, separation: 6.0, seed: 42);
        var classifier = new QuadraticDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        classifier.Train(trainX, trainY);

        var testX = CreateRandomMatrix(10, 3, seed: 1500);
        var original = classifier.Predict(testX);

        var bytes = classifier.Serialize();
        var restored = new QuadraticDiscriminantAnalysis<double>(new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        });
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "QDA MultiClass");
    }

    #endregion

    #region Helper Methods

    private static (Matrix<double> x, Vector<double> y) CreateBinaryData(
        int totalSamples, int features, double separation, int seed)
    {
        var random = new Random(seed);
        int samplesPerClass = totalSamples / 2;
        var x = new Matrix<double>(totalSamples, features);
        var y = new Vector<double>(totalSamples);

        for (int i = 0; i < totalSamples; i++)
        {
            int classLabel = i < samplesPerClass ? 0 : 1;
            y[i] = classLabel;
            for (int j = 0; j < features; j++)
            {
                double noise = NextGaussian(random);
                x[i, j] = classLabel * separation + noise;
            }
        }
        return (x, y);
    }

    private static (Matrix<double> x, Vector<double> y) CreateNonNegativeBinaryData(
        int totalSamples, int features, int seed)
    {
        var random = new Random(seed);
        int samplesPerClass = totalSamples / 2;
        var x = new Matrix<double>(totalSamples, features);
        var y = new Vector<double>(totalSamples);

        for (int i = 0; i < totalSamples; i++)
        {
            int classLabel = i < samplesPerClass ? 0 : 1;
            y[i] = classLabel;
            for (int j = 0; j < features; j++)
            {
                // Class 0: low counts, Class 1: high counts
                x[i, j] = classLabel == 0
                    ? random.Next(0, 5)
                    : random.Next(5, 15);
            }
        }
        return (x, y);
    }

    private static (Matrix<double> x, Vector<double> y) CreateBinaryFeatureBinaryData(
        int totalSamples, int features, int seed)
    {
        var random = new Random(seed);
        int samplesPerClass = totalSamples / 2;
        var x = new Matrix<double>(totalSamples, features);
        var y = new Vector<double>(totalSamples);

        for (int i = 0; i < totalSamples; i++)
        {
            int classLabel = i < samplesPerClass ? 0 : 1;
            y[i] = classLabel;
            for (int j = 0; j < features; j++)
            {
                // Binary features correlated with class
                double prob = classLabel == 0 ? 0.2 + j * 0.05 : 0.7 + j * 0.05;
                x[i, j] = random.NextDouble() < Math.Min(prob, 0.95) ? 1.0 : 0.0;
            }
        }
        return (x, y);
    }

    private static Matrix<double> CreateRandomMatrix(int rows, int cols, int seed)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = random.NextDouble() * 10 - 5;
        return matrix;
    }

    private static Matrix<double> CreateNonNegativeMatrix(int rows, int cols, int seed)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = random.Next(0, 10);
        return matrix;
    }

    private static Matrix<double> CreateBinaryFeatureMatrix(int rows, int cols, int seed)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = random.NextDouble() < 0.5 ? 1.0 : 0.0;
        return matrix;
    }

    private static (Matrix<double> x, Vector<double> y) CreateCategoricalBinaryData(
        int totalSamples, int features, int seed)
    {
        var random = new Random(seed);
        int samplesPerClass = totalSamples / 2;
        var x = new Matrix<double>(totalSamples, features);
        var y = new Vector<double>(totalSamples);

        for (int i = 0; i < totalSamples; i++)
        {
            int classLabel = i < samplesPerClass ? 0 : 1;
            y[i] = classLabel;
            for (int j = 0; j < features; j++)
            {
                // Class 0 leans toward category 0, class 1 toward category 2
                x[i, j] = classLabel == 0
                    ? (random.NextDouble() < 0.8 ? 0 : random.Next(0, 3))
                    : (random.NextDouble() < 0.8 ? 2 : random.Next(0, 3));
            }
        }
        return (x, y);
    }

    private static Matrix<double> CreateCategoricalMatrix(int rows, int cols, int seed)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = random.Next(0, 3);
        return matrix;
    }

    private static (Matrix<double> x, Vector<double> y) CreateMultiClassData(
        int samplesPerClass, int features, int numClasses, double separation, int seed)
    {
        var random = new Random(seed);
        int totalSamples = samplesPerClass * numClasses;
        var x = new Matrix<double>(totalSamples, features);
        var y = new Vector<double>(totalSamples);

        for (int c = 0; c < numClasses; c++)
        {
            for (int i = 0; i < samplesPerClass; i++)
            {
                int idx = c * samplesPerClass + i;
                y[idx] = c;
                for (int j = 0; j < features; j++)
                {
                    x[idx, j] = c * separation + NextGaussian(random);
                }
            }
        }
        return (x, y);
    }

    private static double NextGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static void AssertPredictionsMatch(Vector<double> original, Vector<double> restored, string modelName)
    {
        Assert.Equal(original.Length, restored.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.True(Math.Abs(original[i] - restored[i]) < 1e-10,
                $"{modelName}: Prediction mismatch at index {i}: original={original[i]}, restored={restored[i]}. " +
                $"This indicates the Serialize/Deserialize round-trip is losing trained model state.");
        }
    }

    private static double ComputeAccuracy(Vector<double> predictions, Vector<double> actual)
    {
        int correct = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            double rounded = Math.Round(predictions[i]);
            if (Math.Abs(rounded - actual[i]) < 0.5)
                correct++;
        }
        return (double)correct / predictions.Length;
    }

    #endregion
}
