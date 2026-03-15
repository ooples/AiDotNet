using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for classification models. Tests deep mathematical invariants
/// that any correctly implemented classifier must satisfy.
/// </summary>
public abstract class ClassificationModelTestBase
{
    protected abstract IFullModel<double, Matrix<double>, Vector<double>> CreateModel();

    protected virtual int TrainSamples => 120;
    protected virtual int TestSamples => 30;
    protected virtual int Features => 3;
    protected virtual int NumClasses => 2;

    // =====================================================
    // MATHEMATICAL INVARIANT: Predictions Are Valid Class Labels
    // Every prediction must be in {0, 1, ..., K-1}. No floats, no negatives,
    // no out-of-range. This catches silent type coercion bugs.
    // =====================================================

    [Fact]
    public void Predictions_ShouldBeValidClassLabels()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);
        var (testX, _) = ModelTestHelpers.GenerateClassificationData(TestSamples, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        var predictions = model.Predict(testX);

        for (int i = 0; i < predictions.Length; i++)
        {
            double p = predictions[i];
            Assert.False(double.IsNaN(p), $"Prediction[{i}] is NaN.");
            Assert.False(double.IsInfinity(p), $"Prediction[{i}] is Infinity.");

            double rounded = Math.Round(p);
            Assert.True(rounded >= 0 && rounded < NumClasses,
                $"Prediction[{i}] = {p:F4} (rounded={rounded}) is not a valid class in [0, {NumClasses}).");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Better Than Random on Separable Data
    // On data with well-separated Gaussian clusters (σ=0.5, center spacing=4),
    // ANY classifier should beat uniform random (1/K).
    // Failing this means the model isn't learning at all.
    // =====================================================

    [Fact]
    public void Accuracy_ShouldBeatChance_OnSeparableData()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);
        var (testX, testY) = ModelTestHelpers.GenerateClassificationData(TestSamples, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        var predictions = model.Predict(testX);

        double accuracy = ModelTestHelpers.CalculateAccuracy(testY, predictions);
        double chanceLevel = 1.0 / NumClasses;
        Assert.True(accuracy > chanceLevel,
            $"Accuracy = {accuracy:F4}, chance = {chanceLevel:F4}. " +
            "Classifier is not learning from separable Gaussian data.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: High Accuracy on Perfectly Separable Data
    // With center spacing >> cluster std (4.0 vs 0.5), accuracy should be > 80%.
    // =====================================================

    [Fact]
    public void Accuracy_ShouldBeHigh_OnPerfectlySeparableData()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);
        var (testX, testY) = ModelTestHelpers.GenerateClassificationData(TestSamples, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        var predictions = model.Predict(testX);

        double accuracy = ModelTestHelpers.CalculateAccuracy(testY, predictions);
        Assert.True(accuracy > 0.8,
            $"Accuracy = {accuracy:F4} on perfectly separable data (should be >80%). " +
            "Classifier may have a decision boundary bug.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Accuracy ≥ Test Accuracy
    // The model should fit training data at least as well as test data.
    // =====================================================

    [Fact]
    public void TrainingAccuracy_ShouldBeAtLeastAsGood_AsTestAccuracy()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);
        var (testX, testY) = ModelTestHelpers.GenerateClassificationData(TestSamples, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        var trainPred = model.Predict(trainX);
        var testPred = model.Predict(testX);

        double trainAcc = ModelTestHelpers.CalculateAccuracy(trainY, trainPred);
        double testAcc = ModelTestHelpers.CalculateAccuracy(testY, testPred);

        // Training accuracy should be ≥ test accuracy (allow small margin for stochastic models)
        Assert.True(trainAcc >= testAcc - 0.15,
            $"Training accuracy ({trainAcc:F4}) is much worse than test accuracy ({testAcc:F4}). " +
            "Model may not be fitting training data correctly.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: More Data → Better or Equal Accuracy
    // =====================================================

    [Fact]
    public void MoreData_ShouldNotDegrade_Accuracy()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var (trainX1, trainY1) = ModelTestHelpers.GenerateClassificationData(30, Features, NumClasses, rng1);

        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model2 = CreateModel();
        var (trainX2, trainY2) = ModelTestHelpers.GenerateClassificationData(150, Features, NumClasses, rng2);

        var rngTest = ModelTestHelpers.CreateSeededRandom(99);
        var (testX, testY) = ModelTestHelpers.GenerateClassificationData(50, Features, NumClasses, rngTest);

        model1.Train(trainX1, trainY1);
        model2.Train(trainX2, trainY2);

        var pred1 = model1.Predict(testX);
        var pred2 = model2.Predict(testX);

        double acc1 = ModelTestHelpers.CalculateAccuracy(testY, pred1);
        double acc2 = ModelTestHelpers.CalculateAccuracy(testY, pred2);

        Assert.True(acc2 >= acc1 - 0.15,
            $"5x more data made accuracy worse: acc(30)={acc1:F4}, acc(150)={acc2:F4}. " +
            "Model may not be correctly learning from additional data.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Irrelevant Feature Should Not Help
    // =====================================================

    [Fact]
    public void IrrelevantFeature_ShouldNotImprove_Accuracy()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX_real, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, 2, NumClasses, rng1);
        var (testX_real, testY) = ModelTestHelpers.GenerateClassificationData(TestSamples, 2, NumClasses, rng2);

        var rngNoise = ModelTestHelpers.CreateSeededRandom(77);
        var trainX_noisy = ModelTestHelpers.AddNoiseFeature(trainX_real, rngNoise);
        var testX_noisy = ModelTestHelpers.AddNoiseFeature(testX_real, rngNoise);

        model1.Train(trainX_real, trainY);
        model2.Train(trainX_noisy, trainY);

        var pred1 = model1.Predict(testX_real);
        var pred2 = model2.Predict(testX_noisy);

        double accClean = ModelTestHelpers.CalculateAccuracy(testY, pred1);
        double accNoisy = ModelTestHelpers.CalculateAccuracy(testY, pred2);

        Assert.True(accNoisy <= accClean + 0.15,
            $"Adding noise feature improved accuracy: clean={accClean:F4}, noisy={accNoisy:F4}. " +
            "Model may be overfitting to noise.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Each Class Predicted At Least Once
    // On balanced data with well-separated clusters, the model should
    // predict every class at least once. If it doesn't, it has collapsed
    // to always predicting one class (a common bug).
    // =====================================================

    [Fact]
    public void AllClasses_ShouldBePredicted_OnBalancedData()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);
        var (testX, _) = ModelTestHelpers.GenerateClassificationData(60, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        var predictions = model.Predict(testX);

        var predictedClasses = new HashSet<int>();
        for (int i = 0; i < predictions.Length; i++)
            predictedClasses.Add((int)Math.Round(predictions[i]));

        Assert.True(predictedClasses.Count >= NumClasses,
            $"Only predicted {predictedClasses.Count}/{NumClasses} classes: {{{string.Join(",", predictedClasses)}}}. " +
            "Model may have collapsed to predicting a single class.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Confusion Matrix Diagonal Dominance
    // For separable data, the confusion matrix diagonal should dominate.
    // Most predictions for class c should actually be class c.
    // =====================================================

    [Fact]
    public void ConfusionMatrix_ShouldBeDiagonalDominant()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);
        var (testX, testY) = ModelTestHelpers.GenerateClassificationData(60, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        var predictions = model.Predict(testX);

        // Build confusion matrix
        var cm = new int[NumClasses, NumClasses];
        for (int i = 0; i < predictions.Length; i++)
        {
            int actual = (int)Math.Round(testY[i]);
            int predicted = (int)Math.Round(predictions[i]);
            if (actual >= 0 && actual < NumClasses && predicted >= 0 && predicted < NumClasses)
                cm[actual, predicted]++;
        }

        // Check diagonal dominance: for each class, correct > total errors for that class
        for (int c = 0; c < NumClasses; c++)
        {
            int rowTotal = 0;
            for (int j = 0; j < NumClasses; j++)
                rowTotal += cm[c, j];

            if (rowTotal > 0)
            {
                double classPrecision = (double)cm[c, c] / rowTotal;
                Assert.True(classPrecision > 0.5,
                    $"Class {c}: precision = {classPrecision:F4} (correct={cm[c, c]}, total={rowTotal}). " +
                    "Confusion matrix is not diagonal-dominant for separable data.");
            }
        }
    }

    // =====================================================
    // DETERMINISM + OUTPUT SHAPE + CLONE + METADATA (basic contracts)
    // =====================================================

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);
        var (testX, _) = ModelTestHelpers.GenerateClassificationData(TestSamples, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        var pred1 = model.Predict(testX);
        var pred2 = model.Predict(testX);

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact]
    public void OutputDimension_ShouldMatchInputRows()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);
        var (testX, _) = ModelTestHelpers.GenerateClassificationData(TestSamples, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        Assert.Equal(TestSamples, model.Predict(testX).Length);
    }

    [Fact]
    public void Clone_ShouldProduceIdenticalPredictions()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);
        var (testX, _) = ModelTestHelpers.GenerateClassificationData(TestSamples, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        var cloned = model.Clone();
        var pred1 = model.Predict(testX);
        var pred2 = cloned.Predict(testX);

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact]
    public void Metadata_ShouldExistAfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact]
    public void Parameters_ShouldBeNonEmpty_AfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);

        model.Train(trainX, trainY);
        Assert.True(model.GetParameters().Length > 0, "Trained classifier should have learnable parameters.");
    }

    // =====================================================
    // INTEGRATION: Builder Pipeline
    // =====================================================

    [Fact]
    public void Builder_ShouldProduceResult()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiDotNet.AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        Assert.NotNull(result);
    }

    [Fact]
    public void Builder_AccuracyShouldBeatChance()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = ModelTestHelpers.GenerateClassificationData(TrainSamples, Features, NumClasses, rng);
        var (testX, testY) = ModelTestHelpers.GenerateClassificationData(TestSamples, Features, NumClasses, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiDotNet.AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var predictions = result.Predict(testX);
        double accuracy = ModelTestHelpers.CalculateAccuracy(testY, predictions);
        Assert.True(accuracy > 1.0 / NumClasses,
            $"Builder pipeline accuracy = {accuracy:F4}, chance = {1.0 / NumClasses:F4}.");
    }
}
