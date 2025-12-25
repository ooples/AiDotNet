using AiDotNet.Models.Options;

namespace AiDotNet.Classification.NaiveBayes;

/// <summary>
/// Provides a base implementation for Naive Bayes classifiers.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Naive Bayes classifiers are probabilistic classifiers based on Bayes' theorem with
/// strong (naive) independence assumptions between the features. Despite these assumptions,
/// Naive Bayes classifiers often perform very well in practice.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Naive Bayes uses probability to make predictions. It learns from training data:
/// 1. How common each class is (prior probability)
/// 2. How likely each feature value is given each class (likelihood)
///
/// Then for a new sample, it calculates: P(class|features) ∝ P(class) × P(features|class)
/// and picks the class with the highest probability.
/// </para>
/// </remarks>
public abstract class NaiveBayesBase<T> : ProbabilisticClassifierBase<T>
{
    /// <summary>
    /// Gets the Naive Bayes specific options.
    /// </summary>
    protected new NaiveBayesOptions<T> Options => (NaiveBayesOptions<T>)base.Options;

    /// <summary>
    /// Stores the log prior probabilities for each class.
    /// </summary>
    protected Vector<T>? LogPriors { get; set; }

    /// <summary>
    /// Stores the count of samples per class during training.
    /// </summary>
    protected int[]? ClassCounts { get; set; }

    /// <summary>
    /// Initializes a new instance of the NaiveBayesBase class.
    /// </summary>
    /// <param name="options">Configuration options for the Naive Bayes classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    protected NaiveBayesBase(NaiveBayesOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new NaiveBayesOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
    }

    /// <summary>
    /// Trains the Naive Bayes classifier on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target class labels vector.</param>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        NumFeatures = x.Columns;

        // Extract unique class labels and determine task type
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Compute class counts and priors
        ClassCounts = new int[NumClasses];
        for (int i = 0; i < y.Length; i++)
        {
            int classIdx = GetClassIndex(y[i]);
            ClassCounts[classIdx]++;
        }

        // Compute log priors
        ComputeLogPriors(y.Length);

        // Let derived classes compute their specific parameters
        ComputeClassParameters(x, y);
    }

    /// <summary>
    /// Computes the log prior probabilities for each class.
    /// </summary>
    /// <param name="totalSamples">Total number of training samples.</param>
    protected virtual void ComputeLogPriors(int totalSamples)
    {
        LogPriors = new Vector<T>(NumClasses);

        if (Options.ClassPriors != null && Options.ClassPriors.Length == NumClasses)
        {
            // Use custom priors
            for (int c = 0; c < NumClasses; c++)
            {
                LogPriors[c] = NumOps.Log(NumOps.FromDouble(Options.ClassPriors[c]));
            }
        }
        else if (Options.FitPriors && ClassCounts != null)
        {
            // Learn priors from data
            for (int c = 0; c < NumClasses; c++)
            {
                double prior = (double)ClassCounts[c] / totalSamples;
                LogPriors[c] = NumOps.Log(NumOps.FromDouble(prior));
            }
        }
        else
        {
            // Use uniform priors
            T uniformLogPrior = NumOps.Log(NumOps.FromDouble(1.0 / NumClasses));
            for (int c = 0; c < NumClasses; c++)
            {
                LogPriors[c] = uniformLogPrior;
            }
        }
    }

    /// <summary>
    /// Computes class-specific parameters during training.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target class labels vector.</param>
    /// <remarks>
    /// Derived classes must implement this to compute their specific parameters
    /// (e.g., mean/variance for Gaussian, feature counts for Multinomial).
    /// </remarks>
    protected abstract void ComputeClassParameters(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Predicts log-probabilities for each class (more numerically stable than probabilities).
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A matrix of log-probabilities.</returns>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        var logProbs = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            // Extract the row as a vector
            var sample = new Vector<T>(input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                sample[j] = input[i, j];
            }

            // Compute unnormalized log probabilities
            var unnormalizedLogProbs = new Vector<T>(NumClasses);
            for (int c = 0; c < NumClasses; c++)
            {
                // log P(class) + log P(features|class)
                unnormalizedLogProbs[c] = NumOps.Add(LogPriors![c], ComputeLogLikelihood(sample, c));
            }

            // Normalize using log-sum-exp trick
            T logSum = LogSumExp(unnormalizedLogProbs);
            for (int c = 0; c < NumClasses; c++)
            {
                logProbs[i, c] = NumOps.Subtract(unnormalizedLogProbs[c], logSum);
            }
        }

        return logProbs;
    }

    /// <summary>
    /// Predicts class probabilities for each sample.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A matrix of probabilities.</returns>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        var logProbs = PredictLogProbabilities(input);
        var probs = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int c = 0; c < NumClasses; c++)
            {
                probs[i, c] = NumOps.Exp(logProbs[i, c]);
            }
        }

        return probs;
    }

    /// <summary>
    /// Computes the log-likelihood of a sample given a class.
    /// </summary>
    /// <param name="sample">The feature vector for a single sample.</param>
    /// <param name="classIndex">The class index.</param>
    /// <returns>The log-likelihood log P(sample|class).</returns>
    protected abstract T ComputeLogLikelihood(Vector<T> sample, int classIndex);

    /// <summary>
    /// Computes the log-sum-exp for numerical stability.
    /// </summary>
    /// <param name="values">A vector of log values.</param>
    /// <returns>log(sum(exp(values))).</returns>
    protected T LogSumExp(Vector<T> values)
    {
        // Find max for numerical stability
        T max = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (NumOps.Compare(values[i], max) > 0)
            {
                max = values[i];
            }
        }

        // Compute sum of exp(x - max)
        T sum = NumOps.Zero;
        for (int i = 0; i < values.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Exp(NumOps.Subtract(values[i], max)));
        }

        // Return max + log(sum)
        return NumOps.Add(max, NumOps.Log(sum));
    }

    /// <summary>
    /// Gets the class index for a given label value.
    /// </summary>
    /// <param name="label">The label value.</param>
    /// <returns>The zero-based class index.</returns>
    protected int GetClassIndex(T label)
    {
        if (ClassLabels == null)
        {
            throw new InvalidOperationException("Model must be trained before getting class index.");
        }

        double labelValue = NumOps.ToDouble(label);
        for (int i = 0; i < ClassLabels.Length; i++)
        {
            if (Math.Abs(NumOps.ToDouble(ClassLabels[i]) - labelValue) < 1e-10)
            {
                return i;
            }
        }

        throw new ArgumentException($"Label {label} not found in class labels.");
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Return log priors as the base parameters
        if (LogPriors == null)
        {
            return new Vector<T>(0);
        }
        return LogPriors;
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = (NaiveBayesBase<T>)Clone();
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != NumClasses)
        {
            throw new ArgumentException($"Expected {NumClasses} parameters, got {parameters.Length}");
        }
        LogPriors = new Vector<T>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            LogPriors[i] = parameters[i];
        }
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Naive Bayes doesn't typically use gradient-based optimization
        // Return zero gradients for compatibility
        return new Vector<T>(NumClasses);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Naive Bayes doesn't typically use gradient-based optimization
        // This is a no-op for compatibility
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Alpha"] = Options.Alpha;
        metadata.AdditionalInfo["FitPriors"] = Options.FitPriors;
        return metadata;
    }
}
