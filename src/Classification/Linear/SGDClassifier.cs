using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Linear;

/// <summary>
/// Stochastic Gradient Descent classifier for large-scale learning.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// SGD is an optimization technique that updates weights using one sample at a time.
/// This makes it very efficient for large datasets that don't fit in memory.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Instead of computing gradients over the entire dataset, SGD:
/// 1. Picks one training sample
/// 2. Computes how wrong the prediction is
/// 3. Updates weights to reduce that error
/// 4. Repeats for all samples (one epoch)
/// 5. Repeats for multiple epochs
///
/// Benefits:
/// - Very fast for large datasets
/// - Can handle streaming data
/// - Often finds good solutions quickly
///
/// Trade-offs:
/// - Noisy updates (not always improving)
/// - Requires tuning learning rate
/// - May oscillate near optimal solution
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create SGD classifier for large-scale online learning
/// var options = new LinearClassifierOptions&lt;double&gt;();
///
/// // Prepare training data
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 1.1 }, { 1.2, 0.9 }, { 0.8, 1.0 }, { 5.0, 5.1 }, { 5.2, 4.9 }, { 4.8, 5.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train by updating weights one sample at a time
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new SGDClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict using learned linear decision boundary
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.1, 1.0 } });
/// var prediction = result.Predict(newSample);
/// // prediction[0] contains the predicted class
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Large Scale Machine Learning with Stochastic Gradient Descent", "https://doi.org/10.1007/978-3-7908-2604-3_16")]
public class SGDClassifier<T> : LinearClassifierBase<T>
{
    /// <summary>
    /// Initializes a new instance of the SGDClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public SGDClassifier(LinearClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the SGD classifier on the provided data.
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

        // Convert labels to +1/-1 for binary classification
        var yBinary = SignedBinaryTargets(y).ToArray();

        T learningRate = NumOps.FromDouble(Options.LearningRate);
        T alpha = NumOps.FromDouble(Options.Alpha);
        T prevLoss = NumOps.MaxValue;
        T tolerance = NumOps.FromDouble(Options.Tolerance);

        // Training loop
        for (int epoch = 0; epoch < Options.MaxIterations; epoch++)
        {
            int[] indices = Options.Shuffle
                ? ShuffleIndices(x.Rows)
                : Enumerable.Range(0, x.Rows).ToArray();
            var sampleLosses = new T[indices.Length];
            int sampleCount = 0;

            foreach (int i in indices)
            {
                sampleLosses[sampleCount++] = SgdStep(GetSample(x, i), yBinary[i], learningRate);
            }


            // Apply regularization
            if (Options.Penalty == LinearPenalty.L2)
            {
                ApplyL2Gradient(learningRate, alpha);
            }
            else if (Options.Penalty == LinearPenalty.L1)
            {
                ApplyL1Gradient(learningRate, alpha);
            }

            // Check for convergence on the mean epoch loss
            if (HasConverged(sampleLosses, x.Rows, tolerance, ref prevLoss))
            {
                break;
            }
        }
    }

    /// <summary>
    /// Computes loss and gradient for the specified loss function.
    /// </summary>
    private void ComputeLossAndGradient(T prediction, T target, out T loss, out T gradient)
    {
        T margin = NumOps.Multiply(target, prediction);

        switch (Options.Loss)
        {
            case LinearLoss.Hinge:
                // Hinge loss: max(0, 1 - margin)
                T hingeMargin = NumOps.Subtract(NumOps.One, margin);
                if (NumOps.Compare(hingeMargin, NumOps.Zero) > 0)
                {
                    loss = hingeMargin;
                    gradient = NumOps.Negate(target);
                }
                else
                {
                    loss = NumOps.Zero;
                    gradient = NumOps.Zero;
                }
                break;

            case LinearLoss.SquaredHinge:
                // Squared hinge loss: max(0, 1 - margin)^2
                T sqHingeMargin = NumOps.Subtract(NumOps.One, margin);
                if (NumOps.Compare(sqHingeMargin, NumOps.Zero) > 0)
                {
                    loss = NumOps.Multiply(sqHingeMargin, sqHingeMargin);
                    gradient = NumOps.Multiply(NumOps.FromDouble(-2.0), NumOps.Multiply(sqHingeMargin, target));
                }
                else
                {
                    loss = NumOps.Zero;
                    gradient = NumOps.Zero;
                }
                break;

            case LinearLoss.Log:
                // Log loss (logistic regression)
                T expNegMargin = NumOps.Exp(NumOps.Negate(margin));
                loss = NumOps.Log(NumOps.Add(NumOps.One, expNegMargin));
                T sigmoid = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegMargin));
                gradient = NumOps.Multiply(NumOps.Subtract(sigmoid, NumOps.One), target);
                break;

            case LinearLoss.Perceptron:
                // Perceptron loss: max(0, -margin)
                if (NumOps.Compare(margin, NumOps.Zero) <= 0)
                {
                    loss = NumOps.Negate(margin);
                    gradient = NumOps.Negate(target);
                }
                else
                {
                    loss = NumOps.Zero;
                    gradient = NumOps.Zero;
                }
                break;

            case LinearLoss.ModifiedHuber:
                // Modified Huber loss
                if (NumOps.Compare(margin, NumOps.Negate(NumOps.One)) >= 0)
                {
                    T clippedMargin = NumOps.Subtract(NumOps.One, margin);
                    if (NumOps.Compare(clippedMargin, NumOps.Zero) > 0)
                    {
                        loss = NumOps.Multiply(clippedMargin, clippedMargin);
                        gradient = NumOps.Multiply(NumOps.FromDouble(-2.0), NumOps.Multiply(clippedMargin, target));
                    }
                    else
                    {
                        loss = NumOps.Zero;
                        gradient = NumOps.Zero;
                    }
                }
                else
                {
                    // Linear part for margin < -1
                    loss = NumOps.Multiply(NumOps.FromDouble(-4.0), margin);
                    gradient = NumOps.Multiply(NumOps.FromDouble(-4.0), target);
                }
                break;

            default:
                loss = NumOps.Zero;
                gradient = NumOps.Zero;
                break;
        }
    }

    /// <summary>
    /// One SGD step on a single sample (w -= lr * gradient * x); returns the sample's loss.
    /// </summary>
    private T SgdStep(Vector<T> sample, T target, T learningRate)
    {
        // Compute loss and gradient based on loss type
        ComputeLossAndGradient(DecisionFunction(sample), target, out T loss, out T gradient);

        T scale = NumOps.Multiply(learningRate, gradient);
        AddScaledSample(sample, NumOps.Negate(scale));
        if (Options.FitIntercept)
        {
            Intercept = NumOps.Subtract(Intercept, scale);
        }

        return loss;
    }

    /// <summary>
    /// Convergence check on the mean epoch loss: true when |previous - mean| &lt; tolerance, otherwise the
    /// mean becomes the new previous loss.
    /// </summary>
    private bool HasConverged(T[] sampleLosses, int rows, T tolerance, ref T prevLoss)
    {
        T epochLoss = sampleLosses.Length == 0 ? NumOps.Zero : Engine.Sum(Vector<T>.Wrap(sampleLosses));
        T avgLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(rows));
        T improvement = NumOps.Abs(NumOps.Subtract(prevLoss, avgLoss));
        if (NumOps.Compare(improvement, tolerance) < 0)
        {
            return true;
        }

        prevLoss = avgLoss;
        return false;
    }
}
