using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Classification;
using AiDotNet.Enums;
using AiDotNet.Classification.Trees;
using AiDotNet.Models.Options;
using AiDotNet.Regularization;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.Ensemble;

#pragma warning disable CS8601, CS8618 // Generic T defaults use default(T) - always used with value types

/// <summary>
/// Gradient Boosting classifier that builds trees sequentially to correct errors.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Gradient Boosting builds an additive model in a forward stage-wise fashion.
/// At each stage, a regression tree is fit on the negative gradient of the loss function.
/// For classification, this uses log loss (deviance) or exponential loss.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Gradient Boosting is one of the most powerful machine learning algorithms:
///
/// How it works:
/// 1. Start with an initial prediction
/// 2. Calculate how wrong we are
/// 3. Train a tree to predict our mistakes
/// 4. Add a fraction of this tree's predictions
/// 5. Repeat, each time correcting remaining errors
///
/// Key insight: Each tree fixes what previous trees got wrong!
///
/// Tips for best results:
/// - Use lower learning_rate with more n_estimators
/// - Keep max_depth small (3-5) unlike Random Forest
/// - Consider subsample less than 1.0 for regularization
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Vector<>), typeof(Vector<>))]
[ModelPaper("Greedy Function Approximation: A Gradient Boosting Machine", "https://doi.org/10.1214/aos/1013203451", Year = 2001, Authors = "Jerome H. Friedman")]
public class GradientBoostingClassifier<T> : EnsembleClassifierBase<T>, ITreeBasedClassifier<T>
{
    /// <summary>
    /// Gets the Gradient Boosting specific options.
    /// </summary>
    protected new GradientBoostingClassifierOptions<T> Options => (GradientBoostingClassifierOptions<T>)base.Options;

    /// <summary>
    /// Initial prediction (prior).
    /// </summary>
    private T _initPrediction = default;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private Random? _random;

    /// <summary>
    /// Mean residual values for each tree's leaf predictions.
    /// Each entry contains [meanForClass0, meanForClass1] for the corresponding tree.
    /// </summary>
    private readonly List<T[]> _leafResidualMeans = new();

    /// <inheritdoc/>
    public int MaxDepth => Options.MaxDepth;

    /// <inheritdoc/>
    public int LeafCount => CalculateTotalLeafCount();

    /// <inheritdoc/>
    public int NodeCount => CalculateTotalNodeCount();

    /// <summary>
    /// Initializes a new instance of the GradientBoostingClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for Gradient Boosting.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public GradientBoostingClassifier(GradientBoostingClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new GradientBoostingClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.GradientBoostingClassifier;

    /// <summary>
    /// Trains the Gradient Boosting classifier on the provided data.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        _random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Clear existing estimators and residual means
        Estimators.Clear();
        _leafResidualMeans.Clear();

        int n = x.Rows;

        // Convert labels to 0/1 for binary classification
        var yBinary = new Vector<T>(n);
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        for (int i = 0; i < n; i++)
        {
            yBinary[i] = NumOps.Compare(y[i], positiveClass) == 0 ? NumOps.One : NumOps.Zero;
        }

        // Initialize F0 = log(p / (1 - p)) where p is the prior probability
        int posCount = 0;
        for (int i = 0; i < n; i++)
        {
            if (NumOps.Compare(yBinary[i], NumOps.One) == 0)
            {
                posCount++;
            }
        }
        double p = (double)posCount / n;
        p = Math.Max(1e-10, Math.Min(1 - 1e-10, p)); // Clip for numerical stability
        _initPrediction = NumOps.FromDouble(Math.Log(p / (1 - p)));

        // Current predictions (in log-odds space)
        var fValues = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            fValues[i] = _initPrediction;
        }

        // Learning rate
        T lr = NumOps.FromDouble(Options.LearningRate);

        // Train trees
        for (int m = 0; m < Options.NEstimators; m++)
        {
            // Compute probabilities from current predictions
            var probs = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                probs[i] = Sigmoid(fValues[i]);
            }

            // Compute negative gradient (residuals for log loss): y - p
            var residuals = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                residuals[i] = NumOps.Subtract(yBinary[i], probs[i]);
            }

            // Subsample if needed
            Matrix<T> xSample;
            Vector<T> residualsSample;
            int[] sampleIndices;

            if (Options.Subsample < 1.0)
            {
                (xSample, residualsSample, sampleIndices) = SubsampleData(x, residuals);
            }
            else
            {
                xSample = x;
                residualsSample = residuals;
                sampleIndices = Enumerable.Range(0, n).ToArray();
            }

            // Train a regression tree on residuals
            // Note: We're using DecisionTreeClassifier but treating residuals as targets
            // In a full implementation, we'd use a DecisionTreeRegressor
            var treeOptions = new DecisionTreeClassifierOptions<T>
            {
                MaxDepth = Options.MaxDepth,
                MinSamplesSplit = Options.MinSamplesSplit,
                MinSamplesLeaf = Options.MinSamplesLeaf,
                Seed = _random.Next(),
                MinImpurityDecrease = Options.MinImpurityDecrease
            };

            // Fit to residual signs to determine tree structure
            var residualClasses = new Vector<T>(residualsSample.Length);
            for (int i = 0; i < residualsSample.Length; i++)
            {
                residualClasses[i] = NumOps.Compare(residualsSample[i], NumOps.Zero) >= 0
                    ? NumOps.One
                    : NumOps.Zero;
            }

            var tree = new DecisionTreeClassifier<T>(treeOptions);
            tree.Train(xSample, residualClasses);
            Estimators.Add(tree);

            // Compute mean residual values for each predicted class (leaf)
            // This approximates regression tree behavior by using actual residual magnitudes
            var samplePreds = tree.Predict(xSample);
            T sumClass0 = NumOps.Zero, sumClass1 = NumOps.Zero;
            int countClass0 = 0, countClass1 = 0;

            for (int i = 0; i < residualsSample.Length; i++)
            {
                if (NumOps.Compare(samplePreds[i], NumOps.One) == 0)
                {
                    sumClass1 = NumOps.Add(sumClass1, residualsSample[i]);
                    countClass1++;
                }
                else
                {
                    sumClass0 = NumOps.Add(sumClass0, residualsSample[i]);
                    countClass0++;
                }
            }

            // Compute mean residuals for each class (with fallback to 0 if no samples)
            T meanClass0 = countClass0 > 0
                ? NumOps.Divide(sumClass0, NumOps.FromDouble(countClass0))
                : NumOps.Zero;
            T meanClass1 = countClass1 > 0
                ? NumOps.Divide(sumClass1, NumOps.FromDouble(countClass1))
                : NumOps.Zero;

            _leafResidualMeans.Add(new[] { meanClass0, meanClass1 });

            // Update predictions using actual mean residuals instead of fixed values
            var treePreds = tree.Predict(x);
            for (int i = 0; i < n; i++)
            {
                // Use actual mean residual for the predicted class
                T treeOutput = NumOps.Compare(treePreds[i], NumOps.One) == 0
                    ? meanClass1
                    : meanClass0;
                fValues[i] = NumOps.Add(fValues[i], NumOps.Multiply(lr, treeOutput));
            }
        }

        // Aggregate feature importances
        AggregateFeatureImportances();
    }

    /// <summary>
    /// Computes the sigmoid function.
    /// </summary>
    private T Sigmoid(T x)
    {
        // sigmoid(x) = 1 / (1 + exp(-x))
        T negX = NumOps.Negate(x);
        T expNegX = NumOps.Exp(negX);
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
    }

    /// <summary>
    /// Subsamples data for stochastic gradient boosting.
    /// </summary>
    private (Matrix<T> x, Vector<T> y, int[] indices) SubsampleData(Matrix<T> x, Vector<T> y)
    {
        if (_random is null)
        {
            throw new InvalidOperationException("Random number generator not initialized.");
        }

        int n = x.Rows;
        int sampleSize = (int)(n * Options.Subsample);
        sampleSize = Math.Max(1, sampleSize);

        var indices = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++)
        {
            indices[i] = _random.Next(n);
        }

        var xSample = new Matrix<T>(sampleSize, x.Columns);
        var ySample = new Vector<T>(sampleSize);

        for (int i = 0; i < sampleSize; i++)
        {
            int idx = indices[i];
            for (int j = 0; j < x.Columns; j++)
            {
                xSample[i, j] = x[idx, j];
            }
            ySample[i] = y[idx];
        }

        return (xSample, ySample, indices);
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (Estimators.Count == 0 || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Vector<T>(input.Rows);
        T lr = NumOps.FromDouble(Options.LearningRate);

        for (int i = 0; i < input.Rows; i++)
        {
            // Start with initial prediction
            T fValue = _initPrediction;

            // Add contributions from all trees
            var sample = new Matrix<T>(1, input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                sample[0, j] = input[i, j];
            }

            for (int m = 0; m < Estimators.Count; m++)
            {
                var treePred = Estimators[m].Predict(sample);
                // Use stored mean residuals for this tree
                T treeOutput = NumOps.Compare(treePred[0], NumOps.One) == 0
                    ? _leafResidualMeans[m][1]
                    : _leafResidualMeans[m][0];
                fValue = NumOps.Add(fValue, NumOps.Multiply(lr, treeOutput));
            }

            // Convert to probability and threshold
            T prob = Sigmoid(fValue);
            predictions[i] = NumOps.Compare(prob, NumOps.FromDouble(0.5)) >= 0
                ? ClassLabels[ClassLabels.Length - 1]
                : ClassLabels[0];
        }

        return predictions;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (Estimators.Count == 0)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var probabilities = new Matrix<T>(input.Rows, NumClasses);
        T lr = NumOps.FromDouble(Options.LearningRate);

        for (int i = 0; i < input.Rows; i++)
        {
            T fValue = _initPrediction;

            var sample = new Matrix<T>(1, input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                sample[0, j] = input[i, j];
            }

            for (int m = 0; m < Estimators.Count; m++)
            {
                var treePred = Estimators[m].Predict(sample);
                // Use stored mean residuals for this tree
                T treeOutput = NumOps.Compare(treePred[0], NumOps.One) == 0
                    ? _leafResidualMeans[m][1]
                    : _leafResidualMeans[m][0];
                fValue = NumOps.Add(fValue, NumOps.Multiply(lr, treeOutput));
            }

            T prob = Sigmoid(fValue);
            probabilities[i, 0] = NumOps.Subtract(NumOps.One, prob); // Negative class
            if (NumClasses > 1)
            {
                probabilities[i, 1] = prob; // Positive class
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Calculates the total number of leaf nodes.
    /// </summary>
    private int CalculateTotalLeafCount()
    {
        int total = 0;
        foreach (var estimator in Estimators)
        {
            if (estimator is ITreeBasedClassifier<T> tree)
            {
                total += tree.LeafCount;
            }
        }
        return total;
    }

    /// <summary>
    /// Calculates the total number of nodes.
    /// </summary>
    private int CalculateTotalNodeCount()
    {
        int total = 0;
        foreach (var estimator in Estimators)
        {
            if (estimator is ITreeBasedClassifier<T> tree)
            {
                total += tree.NodeCount;
            }
        }
        return total;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new GradientBoostingClassifier<T>(new GradientBoostingClassifierOptions<T>
        {
            NEstimators = Options.NEstimators,
            LearningRate = Options.LearningRate,
            MaxDepth = Options.MaxDepth,
            MinSamplesSplit = Options.MinSamplesSplit,
            MinSamplesLeaf = Options.MinSamplesLeaf,
            Subsample = Options.Subsample,
            MaxFeatures = Options.MaxFeatures,
            Loss = Options.Loss,
            Seed = Options.Seed,
            MinImpurityDecrease = Options.MinImpurityDecrease
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new GradientBoostingClassifier<T>(new GradientBoostingClassifierOptions<T>
        {
            NEstimators = Options.NEstimators,
            LearningRate = Options.LearningRate,
            MaxDepth = Options.MaxDepth,
            MinSamplesSplit = Options.MinSamplesSplit,
            MinSamplesLeaf = Options.MinSamplesLeaf,
            Subsample = Options.Subsample,
            MaxFeatures = Options.MaxFeatures,
            Loss = Options.Loss,
            Seed = Options.Seed,
            MinImpurityDecrease = Options.MinImpurityDecrease
        });

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;
        clone._initPrediction = _initPrediction;

        if (ClassLabels is not null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (FeatureImportances is not null)
        {
            clone.FeatureImportances = new Vector<T>(FeatureImportances.Length);
            for (int i = 0; i < FeatureImportances.Length; i++)
            {
                clone.FeatureImportances[i] = FeatureImportances[i];
            }
        }

        // Clone leaf residual means
        foreach (var means in _leafResidualMeans)
        {
            var clonedMeans = new T[means.Length];
            Array.Copy(means, clonedMeans, means.Length);
            clone._leafResidualMeans.Add(clonedMeans);
        }

        // Clone all estimators
        foreach (var estimator in Estimators)
        {
            if (estimator is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
            {
                clone.Estimators.Add((IClassifier<T>)fullModel.Clone());
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NEstimators"] = Options.NEstimators;
        metadata.AdditionalInfo["LearningRate"] = Options.LearningRate;
        metadata.AdditionalInfo["MaxDepth"] = Options.MaxDepth;
        metadata.AdditionalInfo["Subsample"] = Options.Subsample;
        metadata.AdditionalInfo["Loss"] = Options.Loss.ToString();
        metadata.AdditionalInfo["TotalNodes"] = NodeCount;
        metadata.AdditionalInfo["TotalLeaves"] = LeafCount;
        return metadata;
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumClasses", NumClasses },
            { "NumFeatures", NumFeatures },
            { "TaskType", (int)TaskType },
            { "ClassLabels", ClassLabels?.ToArray() ?? Array.Empty<T>() },
            { "InitPrediction", NumOps.ToDouble(_initPrediction) }
        };

        // Serialize regularization configuration
        if (Regularization is not null)
        {
            var regOptions = Regularization.GetOptions();
            modelData["RegularizationType"] = (int)regOptions.Type;
            modelData["RegularizationStrength"] = regOptions.Strength;
            modelData["RegularizationL1Ratio"] = regOptions.L1Ratio;
        }

        // Serialize FeatureImportances
        if (FeatureImportances is not null)
        {
            var fiArray = new double[FeatureImportances.Length];
            for (int i = 0; i < FeatureImportances.Length; i++)
                fiArray[i] = NumOps.ToDouble(FeatureImportances[i]);
            modelData["FeatureImportances"] = fiArray;
        }

        // Serialize leaf residual means
        modelData["LeafResidualMeansCount"] = _leafResidualMeans.Count;
        for (int i = 0; i < _leafResidualMeans.Count; i++)
        {
            var means = _leafResidualMeans[i];
            var meansDouble = new double[means.Length];
            for (int j = 0; j < means.Length; j++)
                meansDouble[j] = NumOps.ToDouble(means[j]);
            modelData[$"LeafResidualMeans_{i}"] = meansDouble;
        }

        // Serialize each estimator as base64
        modelData["EstimatorCount"] = Estimators.Count;
        for (int i = 0; i < Estimators.Count; i++)
        {
            if (Estimators[i] is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
            {
                modelData[$"Estimator_{i}"] = Convert.ToBase64String(fullModel.Serialize());
            }
        }

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));
        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata == null || modelMetadata.ModelData == null)
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = JsonConvert.DeserializeObject<JObject>(modelDataString);

        if (modelDataObj == null)
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");

        NumClasses = modelDataObj["NumClasses"]?.ToObject<int>() ?? 0;
        NumFeatures = modelDataObj["NumFeatures"]?.ToObject<int>() ?? 0;
        TaskType = (ClassificationTaskType)(modelDataObj["TaskType"]?.ToObject<int>() ?? 0);

        var classLabelsToken = modelDataObj["ClassLabels"];
        if (classLabelsToken is not null)
        {
            var classLabelsAsDoubles = classLabelsToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (classLabelsAsDoubles.Length > 0)
            {
                ClassLabels = new Vector<T>(classLabelsAsDoubles.Length);
                for (int i = 0; i < classLabelsAsDoubles.Length; i++)
                    ClassLabels[i] = NumOps.FromDouble(classLabelsAsDoubles[i]);
            }
        }

        _initPrediction = NumOps.FromDouble(modelDataObj["InitPrediction"]?.ToObject<double>() ?? 0.0);

        // Deserialize FeatureImportances
        var fiToken = modelDataObj["FeatureImportances"];
        if (fiToken is not null)
        {
            var fiArray = fiToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (fiArray.Length > 0)
            {
                FeatureImportances = new Vector<T>(fiArray.Length);
                for (int i = 0; i < fiArray.Length; i++)
                    FeatureImportances[i] = NumOps.FromDouble(fiArray[i]);
            }
        }

        // Deserialize leaf residual means
        _leafResidualMeans.Clear();
        int lrmCount = modelDataObj["LeafResidualMeansCount"]?.ToObject<int>() ?? 0;
        for (int i = 0; i < lrmCount; i++)
        {
            var lrmToken = modelDataObj[$"LeafResidualMeans_{i}"];
            if (lrmToken is null)
            {
                throw new InvalidOperationException(
                    $"Deserialization failed: LeafResidualMeans_{i} is missing (expected {lrmCount} entries).");
            }
            var meansDouble = lrmToken.ToObject<double[]>() ?? Array.Empty<double>();
            var means = new T[meansDouble.Length];
            for (int j = 0; j < meansDouble.Length; j++)
                means[j] = NumOps.FromDouble(meansDouble[j]);
            _leafResidualMeans.Add(means);
        }

        // Deserialize estimators
        int estimatorCount = modelDataObj["EstimatorCount"]?.ToObject<int>() ?? 0;
        Estimators.Clear();
        for (int i = 0; i < estimatorCount; i++)
        {
            var estToken = modelDataObj[$"Estimator_{i}"]?.ToObject<string>();
            if (estToken is null)
            {
                throw new InvalidOperationException(
                    $"Deserialization failed: Estimator_{i} is missing (expected {estimatorCount} estimators).");
            }
            var estBytes = Convert.FromBase64String(estToken);
            var tree = new DecisionTreeClassifier<T>();
            tree.Deserialize(estBytes);
            Estimators.Add(tree);
        }

        // Restore regularization configuration
        var regType = modelDataObj["RegularizationType"]?.ToObject<int>();
        if (regType.HasValue)
        {
            var regOptions = new RegularizationOptions
            {
                Type = (RegularizationType)regType.Value,
                Strength = modelDataObj["RegularizationStrength"]?.ToObject<double>() ?? 0.0,
                L1Ratio = modelDataObj["RegularizationL1Ratio"]?.ToObject<double>() ?? 0.5
            };

            Regularization = (RegularizationType)regType.Value switch
            {
                RegularizationType.L1 => new L1Regularization<T, Matrix<T>, Vector<T>>(regOptions),
                RegularizationType.L2 => new L2Regularization<T, Matrix<T>, Vector<T>>(regOptions),
                RegularizationType.ElasticNet => new ElasticNetRegularization<T, Matrix<T>, Vector<T>>(regOptions),
                RegularizationType.None => null,
                _ => null
            };
        }
    }
}
