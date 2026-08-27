using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// One-vs-One classifier for multi-class classification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Trains one binary classifier for each pair of classes.
/// Uses voting to determine the final prediction.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// One-vs-One trains a classifier for every pair of classes:
///
/// For 3 classes (A, B, C):
/// - Classifier 1: A vs B
/// - Classifier 2: A vs C
/// - Classifier 3: B vs C
///
/// For K classes, this requires K*(K-1)/2 classifiers.
///
/// For prediction, each classifier votes for one class, and
/// the class with the most votes wins.
///
/// Advantages:
/// - Each classifier is trained on balanced binary problems
/// - Works well with SVM and other pairwise classifiers
/// - Good for small to medium number of classes
///
/// Disadvantages:
/// - Requires many classifiers for large K (K*(K-1)/2)
/// - Slower training for many classes
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create One-vs-One classifier with pairwise binary classifiers
/// var options = new OneVsOneClassifierOptions&lt;double&gt;();
/// var classifier = new OneVsOneClassifier&lt;double&gt;(options);
///
/// // Prepare multi-class training data (3 classes, 9 samples, 2 features)
/// var features = new Matrix&lt;double&gt;(9, 2);
/// features[0, 0] = 1.0; features[0, 1] = 1.1;
/// features[1, 0] = 1.2; features[1, 1] = 0.9;
/// features[2, 0] = 0.8; features[2, 1] = 1.0;
/// features[3, 0] = 3.0; features[3, 1] = 3.1;
/// features[4, 0] = 3.2; features[4, 1] = 2.9;
/// features[5, 0] = 2.8; features[5, 1] = 3.0;
/// features[6, 0] = 5.0; features[6, 1] = 5.1;
/// features[7, 0] = 5.2; features[7, 1] = 4.9;
/// features[8, 0] = 4.8; features[8, 1] = 5.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 });
///
/// // Train K*(K-1)/2 binary classifiers for each class pair
/// classifier.Train(features, labels);
///
/// // Predict using pairwise voting
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 3.1; newSample[0, 1] = 3.0;
/// var prediction = classifier.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Solving Multiclass Learning Problems via Error-Correcting Output Codes", "https://doi.org/10.1613/jair.105", Year = 1995, Authors = "Thomas G. Dietterich, Ghulum Bakiri")]
public partial class OneVsOneClassifier<T> : MetaClassifierBase<T>
{
    /// <summary>
    /// Initializes a new instance with default settings using Gaussian Naive Bayes as the base estimator.
    /// </summary>
    public OneVsOneClassifier()
        : this(() => new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<T>())
    {
    }

    /// <summary>
    /// The binary classifiers for each pair of classes.
    /// </summary>
    private IClassifier<T>[]? _estimators;

    /// <summary>
    /// The class pair indices for each estimator.
    /// </summary>
    private (int, int)[]? _classPairs;

    /// <summary>
    /// Initializes a new instance of the OneVsOneClassifier class.
    /// </summary>
    /// <param name="estimatorFactory">Factory function to create base binary classifiers.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public OneVsOneClassifier(
        Func<IClassifier<T>> estimatorFactory,
        MetaClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, estimatorFactory, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the One-vs-One classifier on the provided data.
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

        // Create list of class pairs
        int numPairs = NumClasses * (NumClasses - 1) / 2;
        _classPairs = new (int, int)[numPairs];
        _estimators = new IClassifier<T>[numPairs];

        int pairIdx = 0;
        for (int i = 0; i < NumClasses - 1; i++)
        {
            for (int j = i + 1; j < NumClasses; j++)
            {
                _classPairs[pairIdx] = (i, j);

                // Get samples for this pair of classes
                var (xPair, yPair) = ExtractPairData(x, y, i, j);

                // Train binary classifier
                _estimators[pairIdx] = CreateBaseEstimator();
                _estimators[pairIdx].Train(xPair, yPair);

                pairIdx++;
            }
        }
    }

    /// <summary>
    /// Extracts samples belonging to a pair of classes.
    /// </summary>
    private (Matrix<T> x, Vector<T> y) ExtractPairData(Matrix<T> x, Vector<T> y, int classA, int classB)
    {
        T labelA = (ClassLabels ?? throw new InvalidOperationException("ClassLabels has not been initialized."))[classA];
        T labelB = ClassLabels[classB];

        // Count samples for these classes
        int count = 0;
        for (int i = 0; i < y.Length; i++)
        {
            if (NumOps.Compare(y[i], labelA) == 0 || NumOps.Compare(y[i], labelB) == 0)
            {
                count++;
            }
        }

        // Extract data
        var xPair = new Matrix<T>(count, NumFeatures);
        var yPair = new Vector<T>(count);

        int idx = 0;
        for (int i = 0; i < y.Length; i++)
        {
            if (NumOps.Compare(y[i], labelA) == 0)
            {
                for (int j = 0; j < NumFeatures; j++)
                {
                    xPair[idx, j] = x[i, j];
                }
                yPair[idx] = NumOps.Zero;
                idx++;
            }
            else if (NumOps.Compare(y[i], labelB) == 0)
            {
                for (int j = 0; j < NumFeatures; j++)
                {
                    xPair[idx, j] = x[i, j];
                }
                yPair[idx] = NumOps.One;
                idx++;
            }
        }

        return (xPair, yPair);
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_estimators is null || _classPairs is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            // Count votes for each class
            var votes = new int[NumClasses];

            // Extract single sample
            var sample = new Matrix<T>(1, NumFeatures);
            for (int j = 0; j < NumFeatures; j++)
            {
                sample[0, j] = input[i, j];
            }

            // Each classifier votes for one of its classes
            for (int p = 0; p < _estimators.Length; p++)
            {
                var (classA, classB) = _classPairs[p];
                var pred = _estimators[p].Predict(sample);

                // 0 = classA wins, 1 = classB wins
                if (NumOps.Compare(pred[0], NumOps.FromDouble(0.5)) < 0)
                {
                    votes[classA]++;
                }
                else
                {
                    votes[classB]++;
                }
            }

            // Find class with most votes
            int bestClass = 0;
            int maxVotes = votes[0];
            for (int c = 1; c < NumClasses; c++)
            {
                if (votes[c] > maxVotes)
                {
                    maxVotes = votes[c];
                    bestClass = c;
                }
            }

            predictions[i] = ClassLabels[bestClass];
        }

        return predictions;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_estimators is null || _classPairs is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var probs = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            // Extract single sample
            var sample = new Matrix<T>(1, NumFeatures);
            for (int j = 0; j < NumFeatures; j++)
            {
                sample[0, j] = input[i, j];
            }

            // Accumulate pairwise probabilities
            var pairwiseProbs = new T[NumClasses, NumClasses];
            var pairwiseCounts = new int[NumClasses, NumClasses];

            for (int p = 0; p < _estimators.Length; p++)
            {
                var (classA, classB) = _classPairs[p];
                T probB;

                // Get probability from classifier
                if (_estimators[p] is IProbabilisticClassifier<T> probClassifier)
                {
                    var pairProbs = probClassifier.PredictProbabilities(sample);
                    probB = pairProbs.Columns > 1 ? pairProbs[0, 1] : pairProbs[0, 0];
                }
                else
                {
                    var pred = _estimators[p].Predict(sample);
                    probB = pred[0];
                }

                T probA = NumOps.Subtract(NumOps.One, probB);

                pairwiseProbs[classA, classB] = probA;
                pairwiseProbs[classB, classA] = probB;
                pairwiseCounts[classA, classB]++;
                pairwiseCounts[classB, classA]++;
            }

            // Aggregate to class probabilities using simple voting normalization
            T sumProbs = NumOps.Zero;
            for (int c = 0; c < NumClasses; c++)
            {
                T classProb = NumOps.Zero;
                int count = 0;

                for (int other = 0; other < NumClasses; other++)
                {
                    if (other != c && pairwiseCounts[c, other] > 0)
                    {
                        classProb = NumOps.Add(classProb, pairwiseProbs[c, other]);
                        count++;
                    }
                }

                if (count > 0)
                {
                    classProb = NumOps.Divide(classProb, NumOps.FromDouble(count));
                }
                else
                {
                    classProb = NumOps.Divide(NumOps.One, NumOps.FromDouble(NumClasses));
                }

                probs[i, c] = classProb;
                sumProbs = NumOps.Add(sumProbs, classProb);
            }

            // Normalize
            T minSum = NumOps.FromDouble(1e-15);
            if (NumOps.Compare(sumProbs, minSum) < 0)
            {
                sumProbs = minSum;
            }

            for (int c = 0; c < NumClasses; c++)
            {
                probs[i, c] = NumOps.Divide(probs[i, c], sumProbs);
            }
        }

        return probs;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        var logProbs = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int c = 0; c < NumClasses; c++)
            {
                T p = probs[i, c];
                T minP = NumOps.FromDouble(1e-15);
                if (NumOps.Compare(p, minP) < 0)
                {
                    p = minP;
                }
                logProbs[i, c] = NumOps.Log(p);
            }
        }

        return logProbs;
    }
}
