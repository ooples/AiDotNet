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
    protected override ModelType GetModelType() => ModelType.SGDClassifier;

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
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        var yBinary = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            yBinary[i] = NumOps.Compare(y[i], positiveClass) == 0
                ? NumOps.One
                : NumOps.Negate(NumOps.One);
        }

        T learningRate = NumOps.FromDouble(Options.LearningRate);
        T alpha = NumOps.FromDouble(Options.Alpha);
        T prevLoss = NumOps.FromDouble(double.MaxValue);
        T tolerance = NumOps.FromDouble(Options.Tolerance);

        // Training loop
        for (int epoch = 0; epoch < Options.MaxIterations; epoch++)
        {
            int[] indices = Options.Shuffle
                ? ShuffleIndices(x.Rows)
                : Enumerable.Range(0, x.Rows).ToArray();
            T epochLoss = NumOps.Zero;

            foreach (int i in indices)
            {
                // Extract sample
                var sample = new Vector<T>(NumFeatures);
                for (int j = 0; j < NumFeatures; j++)
                {
                    sample[j] = x[i, j];
                }

                // Compute prediction
                T prediction = DecisionFunction(sample);
                T target = yBinary[i];

                // Compute loss and gradient based on loss type
                T loss;
                T gradient;
                ComputeLossAndGradient(prediction, target, out loss, out gradient);
                epochLoss = NumOps.Add(epochLoss, loss);

                // Update weights
                for (int j = 0; j < NumFeatures; j++)
                {
                    T update = NumOps.Multiply(learningRate, NumOps.Multiply(gradient, sample[j]));
                    Weights[j] = NumOps.Subtract(Weights[j], update);
                }

                // Update intercept
                if (Options.FitIntercept)
                {
                    T interceptUpdate = NumOps.Multiply(learningRate, gradient);
                    Intercept = NumOps.Subtract(Intercept, interceptUpdate);
                }
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

            // Check for convergence
            T avgLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(x.Rows));
            T improvement = NumOps.Abs(NumOps.Subtract(prevLoss, avgLoss));
            if (NumOps.Compare(improvement, tolerance) < 0)
            {
                break;
            }
            prevLoss = avgLoss;
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

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new SGDClassifier<T>(new LinearClassifierOptions<T>
        {
            LearningRate = Options.LearningRate,
            MaxIterations = Options.MaxIterations,
            Tolerance = Options.Tolerance,
            FitIntercept = Options.FitIntercept,
            Alpha = Options.Alpha,
            Shuffle = Options.Shuffle,
            Penalty = Options.Penalty,
            Loss = Options.Loss,
            RandomState = Options.RandomState
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (SGDClassifier<T>)CreateNewInstance();

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;
        clone.Intercept = Intercept;

        if (ClassLabels is not null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (Weights is not null)
        {
            clone.Weights = new Vector<T>(Weights.Length);
            for (int i = 0; i < Weights.Length; i++)
            {
                clone.Weights[i] = Weights[i];
            }
        }

        return clone;
    }
}
