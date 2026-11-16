using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Linear;

/// <summary>
/// Classic Perceptron classifier - the original neural network building block.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Perceptron is a linear classifier that updates weights only on mistakes.
/// It's historically significant as the foundation of neural networks.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// The Perceptron is the simplest possible neural network:
///
/// How it works:
/// 1. Start with zero weights
/// 2. For each training sample:
///    - If correct: do nothing
///    - If wrong: adjust weights in the direction of the correct class
/// 3. Repeat until no mistakes (or max iterations)
///
/// Properties:
/// - Only works for linearly separable data
/// - Guaranteed to converge if data IS linearly separable
/// - Never converges if data is NOT linearly separable
/// - No notion of margin (unlike SVM)
///
/// Historical note: The Perceptron was invented in 1958 by Frank Rosenblatt
/// and was one of the first machine learning algorithms ever created!
/// </para>
/// </remarks>
/// <para><b>Recommended:</b> Use <c>AiModelBuilder</c> for the simplest entry point.</para>
/// <example>
/// <code>
/// // Create the classic Perceptron classifier
/// var options = new LinearClassifierOptions&lt;double&gt;();
///
/// // Prepare linearly separable training data: 6 samples with 2 features
/// var features = new Matrix&lt;double&gt;(6, 2);
/// features[0, 0] = 1.0; features[0, 1] = 1.1;
/// features[1, 0] = 1.2; features[1, 1] = 0.9;
/// features[2, 0] = 0.8; features[2, 1] = 1.0;
/// features[3, 0] = 5.0; features[3, 1] = 5.1;
/// features[4, 0] = 5.2; features[4, 1] = 4.9;
/// features[5, 0] = 4.8; features[5, 1] = 5.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train by updating weights only on misclassified samples
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new PerceptronClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict using learned linear boundary
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 1.1; newSample[0, 1] = 1.0;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain", "https://doi.org/10.1037/h0042519", Year = 1958, Authors = "Frank Rosenblatt")]
public class PerceptronClassifier<T> : LinearClassifierBase<T>
{
    /// <summary>
    /// Initializes a new instance of the PerceptronClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public PerceptronClassifier(LinearClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the Perceptron classifier on the provided data.
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

        InitializeWeights();

        if (Weights is null)
        {
            throw new InvalidOperationException("Failed to initialize weights.");
        }

        // Validate binary classification
        if (NumClasses != 2)
        {
            throw new NotSupportedException(
                "PerceptronClassifier only supports binary classification. " +
                "Use OneVsRestClassifier for multi-class problems.");
        }

        // Convert labels to +1/-1 for binary classification
        var yBinary = SignedBinaryTargets(y).ToArray();

        T eta = NumOps.FromDouble(Options.LearningRate);
        T alpha = NumOps.FromDouble(Options.Alpha);
        bool hasAlpha = NumOps.Compare(alpha, NumOps.Zero) > 0;

        // Training loop
        for (int epoch = 0; epoch < Options.MaxIterations; epoch++)
        {
            int[] indices = ShuffleIndices(x.Rows);
            int numErrors = 0;

            foreach (int i in indices)
            {
                // Misclassification (margin <= 0) updates w = w + eta * y * x
                if (PerceptronStep(GetSample(x, i), yBinary[i], eta))
                {
                    numErrors++;
                }
            }

            // Apply regularization
            if (Options.Penalty == LinearPenalty.L2 && hasAlpha)
            {
                ApplyL2Gradient(eta, alpha);
            }
            else if (Options.Penalty == LinearPenalty.L1 && hasAlpha)
            {
                ApplyL1Gradient(eta, alpha);
            }

            // Check for convergence (no errors = perfect classification)
            if (numErrors == 0)
            {
                break;
            }
        }
    }

    /// <summary>
    /// One perceptron update on a single sample; returns true when the sample was misclassified.
    /// </summary>
    private bool PerceptronStep(Vector<T> sample, T target, T eta)
    {
        T margin = NumOps.Multiply(target, DecisionFunction(sample));
        if (NumOps.Compare(margin, NumOps.Zero) > 0)
        {
            return false;
        }

        T scale = NumOps.Multiply(eta, target);
        AddScaledSample(sample, scale);
        if (Options.FitIntercept)
        {
            Intercept = NumOps.Add(Intercept, scale);
        }

        return true;
    }
}
