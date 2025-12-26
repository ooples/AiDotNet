using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Linear;

/// <summary>
/// Passive-Aggressive classifier for online learning.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Passive-Aggressive algorithm is an online learning algorithm that:
/// - Is "passive" when the prediction is correct (no update)
/// - Is "aggressive" when wrong (makes the minimal update to correct the mistake)
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Unlike regular gradient descent, Passive-Aggressive:
/// 1. Only updates when it makes a mistake
/// 2. When it updates, it does the minimum needed to fix the mistake
///
/// It's great for:
/// - Online learning (data arrives one sample at a time)
/// - Streaming data
/// - When you want a balance between learning and stability
///
/// The regularization parameter C controls the aggressiveness:
/// - Higher C: More aggressive updates (may overfit to noise)
/// - Lower C: More passive (may underfit)
/// </para>
/// </remarks>
public class PassiveAggressiveClassifier<T> : LinearClassifierBase<T>
{
    /// <summary>
    /// Gets the PA classifier specific options.
    /// </summary>
    protected new PassiveAggressiveOptions<T> Options => (PassiveAggressiveOptions<T>)base.Options;

    /// <summary>
    /// Initializes a new instance of the PassiveAggressiveClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public PassiveAggressiveClassifier(PassiveAggressiveOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new PassiveAggressiveOptions<T>(), regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.PassiveAggressiveClassifier;

    /// <summary>
    /// Trains the Passive-Aggressive classifier on the provided data.
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

        T c = NumOps.FromDouble(Options.C);

        // Training loop
        for (int epoch = 0; epoch < Options.MaxIterations; epoch++)
        {
            int[] indices = ShuffleIndices(x.Rows);
            bool anyUpdate = false;

            foreach (int i in indices)
            {
                // Extract sample
                var sample = new Vector<T>(NumFeatures);
                T squaredNorm = NumOps.Zero;
                for (int j = 0; j < NumFeatures; j++)
                {
                    sample[j] = x[i, j];
                    squaredNorm = NumOps.Add(squaredNorm, NumOps.Multiply(sample[j], sample[j]));
                }

                // Include bias in norm if fitting intercept
                if (Options.FitIntercept)
                {
                    squaredNorm = NumOps.Add(squaredNorm, NumOps.One);
                }

                // Compute prediction and margin
                T prediction = DecisionFunction(sample);
                T target = yBinary[i];
                T margin = NumOps.Multiply(target, prediction);

                // Hinge loss: max(0, 1 - margin)
                T loss = NumOps.Subtract(NumOps.One, margin);
                if (NumOps.Compare(loss, NumOps.Zero) <= 0)
                {
                    // Correct classification with sufficient margin
                    continue;
                }

                anyUpdate = true;

                // Compute update step (depends on PA variant)
                T tau = ComputeTau(loss, squaredNorm, c);

                // Update weights: w = w + tau * y * x
                for (int j = 0; j < NumFeatures; j++)
                {
                    T update = NumOps.Multiply(tau, NumOps.Multiply(target, sample[j]));
                    Weights[j] = NumOps.Add(Weights[j], update);
                }

                // Update intercept
                if (Options.FitIntercept)
                {
                    T interceptUpdate = NumOps.Multiply(tau, target);
                    Intercept = NumOps.Add(Intercept, interceptUpdate);
                }
            }

            // Check for convergence
            if (!anyUpdate)
            {
                break;
            }
        }
    }

    /// <summary>
    /// Computes the update step tau based on the PA variant.
    /// </summary>
    private T ComputeTau(T loss, T squaredNorm, T c)
    {
        // Avoid division by zero
        if (NumOps.Compare(squaredNorm, NumOps.FromDouble(1e-12)) < 0)
        {
            squaredNorm = NumOps.FromDouble(1e-12);
        }

        T tau;

        switch (Options.PAType)
        {
            case PassiveAggressiveType.PA:
                // Original PA: tau = loss / ||x||^2
                tau = NumOps.Divide(loss, squaredNorm);
                break;

            case PassiveAggressiveType.PA_I:
                // PA-I: tau = min(C, loss / ||x||^2)
                T tauI = NumOps.Divide(loss, squaredNorm);
                tau = NumOps.Compare(tauI, c) < 0 ? tauI : c;
                break;

            case PassiveAggressiveType.PA_II:
                // PA-II: tau = loss / (||x||^2 + 1/(2C))
                T denominator = NumOps.Add(squaredNorm, NumOps.Divide(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2.0), c)));
                tau = NumOps.Divide(loss, denominator);
                break;

            default:
                tau = NumOps.Divide(loss, squaredNorm);
                break;
        }

        return tau;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new PassiveAggressiveClassifier<T>(new PassiveAggressiveOptions<T>
        {
            C = Options.C,
            PAType = Options.PAType,
            MaxIterations = Options.MaxIterations,
            FitIntercept = Options.FitIntercept,
            Shuffle = Options.Shuffle,
            RandomState = Options.RandomState
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (PassiveAggressiveClassifier<T>)CreateNewInstance();

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

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["C"] = Options.C;
        metadata.AdditionalInfo["PAType"] = Options.PAType.ToString();
        return metadata;
    }
}

/// <summary>
/// Configuration options for Passive-Aggressive classifier.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class PassiveAggressiveOptions<T> : LinearClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the regularization parameter C.
    /// </summary>
    /// <value>
    /// Maximum step size for regularized updates. Default is 1.0.
    /// </value>
    public double C { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the Passive-Aggressive variant type.
    /// </summary>
    /// <value>
    /// The PA variant. Default is PA-I.
    /// </value>
    public PassiveAggressiveType PAType { get; set; } = PassiveAggressiveType.PA_I;
}

/// <summary>
/// Passive-Aggressive algorithm variants.
/// </summary>
public enum PassiveAggressiveType
{
    /// <summary>
    /// Original PA algorithm - no regularization.
    /// </summary>
    PA,

    /// <summary>
    /// PA-I: Limits step size by C.
    /// </summary>
    PA_I,

    /// <summary>
    /// PA-II: Soft margin with C as regularization.
    /// </summary>
    PA_II
}
