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
    protected override ModelType GetModelType() => ModelType.PerceptronClassifier;

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
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        var yBinary = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            yBinary[i] = NumOps.Compare(y[i], positiveClass) == 0
                ? NumOps.One
                : NumOps.Negate(NumOps.One);
        }

        T eta = NumOps.FromDouble(Options.LearningRate);
        T alpha = NumOps.FromDouble(Options.Alpha);

        // Training loop
        for (int epoch = 0; epoch < Options.MaxIterations; epoch++)
        {
            int[] indices = ShuffleIndices(x.Rows);
            int numErrors = 0;

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

                // Check if prediction is correct
                T margin = NumOps.Multiply(target, prediction);
                if (NumOps.Compare(margin, NumOps.Zero) <= 0)
                {
                    // Misclassification - update weights
                    numErrors++;

                    // Update: w = w + eta * y * x
                    for (int j = 0; j < NumFeatures; j++)
                    {
                        T update = NumOps.Multiply(eta, NumOps.Multiply(target, sample[j]));
                        Weights[j] = NumOps.Add(Weights[j], update);
                    }

                    // Update intercept
                    if (Options.FitIntercept)
                    {
                        T interceptUpdate = NumOps.Multiply(eta, target);
                        Intercept = NumOps.Add(Intercept, interceptUpdate);
                    }
                }
            }

            // Apply regularization
            if (Options.Penalty == LinearPenalty.L2 && NumOps.Compare(alpha, NumOps.Zero) > 0)
            {
                ApplyL2Gradient(eta, alpha);
            }
            else if (Options.Penalty == LinearPenalty.L1 && NumOps.Compare(alpha, NumOps.Zero) > 0)
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

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new PerceptronClassifier<T>(new LinearClassifierOptions<T>
        {
            LearningRate = Options.LearningRate,
            MaxIterations = Options.MaxIterations,
            FitIntercept = Options.FitIntercept,
            Alpha = Options.Alpha,
            Shuffle = Options.Shuffle,
            Penalty = Options.Penalty,
            Seed = Options.Seed
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (PerceptronClassifier<T>)CreateNewInstance();

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
