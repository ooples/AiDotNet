using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for classification models. Tests deep mathematical invariants
/// that any correctly implemented classifier must satisfy.
/// </summary>
/// <remarks>
/// Fixtures and assertion math remain in double precision so class-label checks and accuracy
/// thresholds keep their original meaning. Only the model boundary is converted to <typeparamref name="T"/>.
/// </remarks>
public abstract class ClassificationModelTestBase<T> : System.IDisposable
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    protected static Matrix<T> ToT(Matrix<double> matrix)
    {
        if (typeof(T) == typeof(double)) return (Matrix<T>)(object)matrix;

        var converted = new Matrix<T>(matrix.Rows, matrix.Columns);
        for (int row = 0; row < matrix.Rows; row++)
            for (int column = 0; column < matrix.Columns; column++)
                converted[row, column] = NumOps.FromDouble(matrix[row, column]);
        return converted;
    }

    protected static Vector<T> ToT(Vector<double> vector)
    {
        if (typeof(T) == typeof(double)) return (Vector<T>)(object)vector;

        var converted = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
            converted[i] = NumOps.FromDouble(vector[i]);
        return converted;
    }

    protected static Vector<double> ToD(Vector<T> vector)
    {
        if (typeof(T) == typeof(double)) return (Vector<double>)(object)vector;

        var converted = new Vector<double>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
            converted[i] = NumOps.ToDouble(vector[i]);
        return converted;
    }

    /// <summary>
    /// Reclaim memory between tests (shared model-family teardown). xUnit constructs a fresh
    /// test-class instance per test and calls Dispose() afterward, so this clears the
    /// InferenceWeightCache and compacts the LOH between model classes — keeping committed memory
    /// from accumulating across a shard. Pure hygiene; no test-observable behavior change.
    /// </summary>
    public virtual void Dispose()
    {
        // Reclaim must be unconditional: a throwing derived DisposeCore() must not skip the
        // shared GC gate, or heavy shards reintroduce cross-test memory buildup / OOM.
        try
        {
            DisposeCore();
        }
        finally
        {
            ModelFamilyTestGcGate.ReclaimBetweenTests();
        }
    }

    /// <summary>
    /// Override in a derived test class to add its own teardown while preserving the
    /// shared <see cref="ModelFamilyTestGcGate.ReclaimBetweenTests"/> call.
    /// </summary>
    protected virtual void DisposeCore()
    {
    }

    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateModel();

    protected virtual int TrainSamples => 120;
    protected virtual int TestSamples => 30;
    protected virtual int Features => 3;
    protected virtual int NumClasses => 2;

    /// <summary>
    /// Generates training/test data. Override for models requiring specific data distributions
    /// (e.g., non-negative counts for MultinomialNB).
    /// </summary>
    protected virtual (Matrix<double> X, Vector<double> Y) GenerateData(int samples, int features, int nClasses, Random rng)
        => ModelTestHelpers.GenerateClassificationData(samples, features, nClasses, rng);

    /// <summary>
    /// Whether this model exposes flat parameter vectors. Meta/ensemble/tree classifiers
    /// that delegate to sub-models return empty parameters and should set this to false.
    /// </summary>
    protected virtual bool HasFlatParameters => true;

    // =====================================================
    // MATHEMATICAL INVARIANT: Predictions Are Valid Class Labels
    // Every prediction must be in {0, 1, ..., K-1}. No floats, no negatives,
    // no out-of-range. This catches silent type coercion bugs.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task Predictions_ShouldBeValidClassLabels()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        var (testX, _) = GenerateData(TestSamples, Features, NumClasses, rng);

        model.Train(ToT(trainX), ToT(trainY));
        var predictions = ToD(model.Predict(ToT(testX)));

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

    [Fact(Timeout = 60000)]
    public async Task Accuracy_ShouldBeatChance_OnSeparableData()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        var (testX, testY) = GenerateData(TestSamples, Features, NumClasses, rng);

        model.Train(ToT(trainX), ToT(trainY));
        var predictions = ToD(model.Predict(ToT(testX)));

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

    [Fact(Timeout = 60000)]
    public async Task Accuracy_ShouldBeHigh_OnPerfectlySeparableData()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        var (testX, testY) = GenerateData(TestSamples, Features, NumClasses, rng);

        model.Train(ToT(trainX), ToT(trainY));
        var predictions = ToD(model.Predict(ToT(testX)));

        double accuracy = ModelTestHelpers.CalculateAccuracy(testY, predictions);
        Assert.True(accuracy > 0.8,
            $"Accuracy = {accuracy:F4} on perfectly separable data (should be >80%). " +
            "Classifier may have a decision boundary bug.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Accuracy ≥ Test Accuracy
    // The model should fit training data at least as well as test data.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task TrainingAccuracy_ShouldBeAtLeastAsGood_AsTestAccuracy()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        var (testX, testY) = GenerateData(TestSamples, Features, NumClasses, rng);

        model.Train(ToT(trainX), ToT(trainY));
        var trainPred = ToD(model.Predict(ToT(trainX)));
        var testPred = ToD(model.Predict(ToT(testX)));

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

    [Fact(Timeout = 60000)]
    public async Task MoreData_ShouldNotDegrade_Accuracy()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var (trainX1, trainY1) = GenerateData(30, Features, NumClasses, rng1);

        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model2 = CreateModel();
        var (trainX2, trainY2) = GenerateData(150, Features, NumClasses, rng2);

        var rngTest = ModelTestHelpers.CreateSeededRandom(99);
        var (testX, testY) = GenerateData(50, Features, NumClasses, rngTest);

        model1.Train(ToT(trainX1), ToT(trainY1));
        model2.Train(ToT(trainX2), ToT(trainY2));

        var pred1 = ToD(model1.Predict(ToT(testX)));
        var pred2 = ToD(model2.Predict(ToT(testX)));

        double acc1 = ModelTestHelpers.CalculateAccuracy(testY, pred1);
        double acc2 = ModelTestHelpers.CalculateAccuracy(testY, pred2);

        Assert.True(acc2 >= acc1 - 0.15,
            $"5x more data made accuracy worse: acc(30)={acc1:F4}, acc(150)={acc2:F4}. " +
            "Model may not be correctly learning from additional data.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Irrelevant Feature Should Not Help
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task IrrelevantFeature_ShouldNotImprove_Accuracy()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        // Both models use the SAME number of features to avoid different parameter
        // vector sizes that can cause issues with optimizers in parallel execution.
        // Model1 trains on real features, model2 on real features + 1 noise column.
        var (trainX_real, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng1);
        var (testX_real, testY) = GenerateData(TestSamples, Features, NumClasses, rng2);

        var rngNoise = ModelTestHelpers.CreateSeededRandom(77);
        var trainX_noisy = ModelTestHelpers.AddNoiseFeature(trainX_real, rngNoise);
        var testX_noisy = ModelTestHelpers.AddNoiseFeature(testX_real, rngNoise);

        model1.Train(ToT(trainX_real), ToT(trainY));
        model2.Train(ToT(trainX_noisy), ToT(trainY));

        var pred1 = ToD(model1.Predict(ToT(testX_real)));
        var pred2 = ToD(model2.Predict(ToT(testX_noisy)));

        double accClean = ModelTestHelpers.CalculateAccuracy(testY, pred1);
        double accNoisy = ModelTestHelpers.CalculateAccuracy(testY, pred2);

        Assert.True(accNoisy <= accClean + 0.25,
            $"Adding noise feature improved accuracy by >{(accNoisy - accClean)*100:F1}%: " +
            $"clean={accClean:F4}, noisy={accNoisy:F4}. Model may be overfitting to noise.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Each Class Predicted At Least Once
    // On balanced data with well-separated clusters, the model should
    // predict every class at least once. If it doesn't, it has collapsed
    // to always predicting one class (a common bug).
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task AllClasses_ShouldBePredicted_OnBalancedData()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        var (testX, _) = GenerateData(60, Features, NumClasses, rng);

        model.Train(ToT(trainX), ToT(trainY));
        var predictions = ToD(model.Predict(ToT(testX)));

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

    [Fact(Timeout = 60000)]
    public async Task ConfusionMatrix_ShouldBeDiagonalDominant()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        var (testX, testY) = GenerateData(60, Features, NumClasses, rng);

        model.Train(ToT(trainX), ToT(trainY));
        var predictions = ToD(model.Predict(ToT(testX)));

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

    [Fact(Timeout = 60000)]
    public async Task Predict_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        var (testX, _) = GenerateData(TestSamples, Features, NumClasses, rng);

        model.Train(ToT(trainX), ToT(trainY));
        var pred1 = ToD(model.Predict(ToT(testX)));
        var pred2 = ToD(model.Predict(ToT(testX)));

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task OutputDimension_ShouldMatchInputRows()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        var (testX, _) = GenerateData(TestSamples, Features, NumClasses, rng);

        model.Train(ToT(trainX), ToT(trainY));
        Assert.Equal(TestSamples, model.Predict(ToT(testX)).Length);
    }

    [Fact(Timeout = 60000)]
    public async Task Clone_ShouldProduceIdenticalPredictions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        var (testX, _) = GenerateData(TestSamples, Features, NumClasses, rng);

        model.Train(ToT(trainX), ToT(trainY));
        var cloned = model.Clone();
        var pred1 = ToD(model.Predict(ToT(testX)));
        var pred2 = ToD(cloned.Predict(ToT(testX)));

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task Metadata_ShouldExistAfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);

        model.Train(ToT(trainX), ToT(trainY));
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact(Timeout = 60000)]
    public async Task Parameters_ShouldBeNonEmpty_AfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();

        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);

        model.Train(ToT(trainX), ToT(trainY));
        if (model is not IParameterizable<T, Matrix<T>, Vector<T>> paramModel)
        {
            // Tree/ensemble classifiers don't implement IParameterizable — skip
            return;
        }

        if (!paramModel.SupportsParameterInitialization)
        {
            // A non-initializable surface can still contain fitted persistent state
            // or generated child-component state. It is not an optimizer parameter
            // surface, so this learnable-parameter assertion does not apply.
            return;
        }

        Assert.True(paramModel.GetParameters().Length > 0, "Trained classifier should have learnable parameters.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Binary Threshold Sensitivity
    // For 2-class problems, slightly different decision boundaries should
    // flip at least some predictions. A model that never changes its
    // predictions regardless of data variation has a degenerate decision function.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task BinaryThreshold_Sensitivity()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (NumClasses != 2) return;

        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, 2, rng);

        model.Train(ToT(trainX), ToT(trainY));
        var predictions = ToD(model.Predict(ToT(trainX)));

        if (ModelTestHelpers.AllFinite(predictions))
        {
            // Check that both classes are predicted
            bool hasClass0 = false, hasClass1 = false;
            for (int i = 0; i < predictions.Length; i++)
            {
                if (Math.Round(predictions[i]) < 0.5) hasClass0 = true;
                else hasClass1 = true;
            }
            Assert.True(hasClass0 && hasClass1,
                "Binary classifier predicts only one class — decision boundary may be degenerate.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Class Prior Sensitivity
    // Training on imbalanced data (90/10) should produce more majority
    // predictions than training on balanced data (50/50).
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task ClassPrior_Sensitivity()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (NumClasses != 2) return;

        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        // Balanced: 50/50
        var (balX, balY) = GenerateData(TrainSamples, Features, 2, rng1);

        // Imbalanced: 90/10 (replicate class 0)
        var imbX = new Matrix<double>(TrainSamples, Features);
        var imbY = new Vector<double>(TrainSamples);
        int majorityCount = (int)(TrainSamples * 0.9);
        for (int i = 0; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                imbX[i, j] = rng2.NextDouble() * 4.0 + (i < majorityCount ? 0.0 : 8.0);
            imbY[i] = i < majorityCount ? 0.0 : 1.0;
        }

        model1.Train(ToT(balX), ToT(balY));
        model2.Train(ToT(imbX), ToT(imbY));

        var testX = new Matrix<double>(20, Features);
        for (int i = 0; i < 20; i++)
            for (int j = 0; j < Features; j++)
                testX[i, j] = rng1.NextDouble() * 10.0;

        var pred1 = ToD(model1.Predict(ToT(testX)));
        var pred2 = ToD(model2.Predict(ToT(testX)));

        if (ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2))
        {
            int majority1 = 0, majority2 = 0;
            for (int i = 0; i < pred1.Length; i++)
                if (Math.Round(pred1[i]) < 0.5) majority1++;
            for (int i = 0; i < pred2.Length; i++)
                if (Math.Round(pred2[i]) < 0.5) majority2++;

            // Imbalanced model should predict more class-0 than balanced
            Assert.True(majority2 >= majority1 - 5,
                $"Imbalanced model predicts {majority2} class-0 vs balanced {majority1}. " +
                "Classifier may not be sensitive to class priors.");
        }
    }

    // =====================================================
    // INTEGRATION: Builder Pipeline
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task Builder_ShouldProduceResult()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(ToT(trainX), ToT(trainY));

        var result = new AiDotNet.AiModelBuilder<T, Matrix<T>, Vector<T>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        Assert.NotNull(result);
    }

    [Fact(Timeout = 60000)]
    public async Task Builder_AccuracyShouldBeatChance()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = GenerateData(TrainSamples, Features, NumClasses, rng);
        var (testX, testY) = GenerateData(TestSamples, Features, NumClasses, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(ToT(trainX), ToT(trainY));

        var result = new AiDotNet.AiModelBuilder<T, Matrix<T>, Vector<T>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var predictions = ToD(result.Predict(ToT(testX)));
        double accuracy = ModelTestHelpers.CalculateAccuracy(testY, predictions);
        Assert.True(accuracy > 1.0 / NumClasses,
            $"Builder pipeline accuracy = {accuracy:F4}, chance = {1.0 / NumClasses:F4}.");
    }
}

/// <summary>Double-precision compatibility shim for existing classification fixtures.</summary>
public abstract class ClassificationModelTestBase : ClassificationModelTestBase<double> { }
