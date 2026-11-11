using AiDotNet.Attributes;
using AiDotNet.Models.Options;

using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
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
public abstract partial class NaiveBayesBase<T> : ProbabilisticClassifierBase<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The log priors, which is what this surface has always exposed. PackParameters and UnpackParameters ARE the previous
    /// GetParameters and SetParameters, renamed rather than rewritten, so the layout is
    /// byte-for-byte unchanged. What changes is that the COUNT now folds this same
    /// component instead of ClassifierBase's NumFeatures * NumClasses formula, which
    /// described the classifier's shape rather than what it serializes.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VariableLengthParameterSource<T>(
            () => PackParameters().Length,
            PackParameters,
            UnpackParameters));
    }
    /// <summary>
    /// Naive Bayes models compute parameters from class statistics during training.
    /// They do not support flat parameter initialization via SetParameters.
    /// The AiModelBuilder will use the direct training path instead of the optimizer.
    /// </summary>
    public override bool SupportsParameterInitialization => false;

    /// <summary>
    /// Gets the Naive Bayes specific options.
    /// </summary>
    protected new NaiveBayesOptions<T> Options => (NaiveBayesOptions<T>)base.Options;

    /// <summary>
    /// Stores the log prior probabilities for each class.
    /// </summary>
    [Buffer]
    protected Vector<T>? LogPriors { get; set; }

    /// <summary>
    /// Stores the count of samples per class during training.
    /// </summary>
    [Buffer]
    protected int[]? ClassCounts { get; set; }

    /// <summary>
    /// Initializes a new instance of the NaiveBayesBase class.
    /// </summary>
    /// <param name="options">Configuration options for the Naive Bayes classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    protected NaiveBayesBase(NaiveBayesOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new NaiveBayesOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
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
        var classIndices = GetClassIndexTensor(y);
        ClassCounts = y.Length == 0
            ? new int[NumClasses]
            : Engine.TensorBinCount(classIndices, NumClasses).ToArray();

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
        var engine = Engine;
        if (Options.ClassPriors != null && Options.ClassPriors.Length == NumClasses)
        {
            // Use custom priors
            var priors = new Vector<T>(NumClasses);
            for (int c = 0; c < NumClasses; c++)
            {
                priors[c] = NumOps.FromDouble(Options.ClassPriors[c]);
            }

            LogPriors = NumClasses == 0 ? priors : engine.TensorLog(priors.AsTensor()).ToVector();
        }
        else if (Options.FitPriors && ClassCounts != null)
        {
            // Learn priors from data: log(count / total)
            var counts = ClassIndicesToValues(new Tensor<int>(ClassCounts, new[] { ClassCounts.Length }), totalSamples + 1);
            LogPriors = counts.Length == 0
                ? new Vector<T>(0)
                : engine.TensorLog(engine.TensorDivideScalar(counts.AsTensor(), NumOps.FromDouble(totalSamples))).ToVector();
        }
        else
        {
            // Use uniform priors
            LogPriors = FilledVector(NumClasses, NumOps.Log(NumOps.FromDouble(1.0 / NumClasses)));
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
        if (LogPriors is null)
        {
            throw new InvalidOperationException("LogPriors has not been initialized.");
        }

        if (input.Rows == 0 || NumClasses == 0)
        {
            return new Matrix<T>(input.Rows, NumClasses);
        }

        // log P(class) + log P(features|class), normalized with log-sum-exp
        var unnormalized = Engine.TensorAdd(ComputeLogLikelihoods(input), LogPriors.AsTensor().Reshape(1, NumClasses));
        return Engine.TensorLogSoftmax(unnormalized, 1).ToMatrix();
    }

    /// <summary>
    /// Predicts class probabilities for each sample.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A matrix of probabilities.</returns>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        var logProbs = PredictLogProbabilities(input);
        if (logProbs.Rows == 0 || logProbs.Columns == 0)
        {
            return new Matrix<T>(input.Rows, NumClasses);
        }

        return Engine.TensorExp(Tensor<T>.FromMatrix(logProbs)).ToMatrix();
    }

    /// <summary>
    /// Computes log P(features | class) for every sample and class as an [input.Rows, NumClasses] tensor.
    /// </summary>
    /// <param name="input">The samples.</param>
    /// <returns>The per-sample, per-class log-likelihoods.</returns>
    /// <remarks>
    /// The default evaluates <see cref="ComputeLogLikelihood"/> sample by sample; derived classes override
    /// this with a whole-batch computation.
    /// </remarks>
    protected virtual Tensor<T> ComputeLogLikelihoods(Matrix<T> input)
    {
        return PerSampleLogLikelihoods(input);
    }

    /// <summary>
    /// For every sample and class, sums table[class, feature, choices[sample, feature]] over the features.
    /// </summary>
    /// <param name="choices">The [samples, features] chosen entries, each in [0, choicesPerFeature).</param>
    /// <param name="table">The [classes, features, choicesPerFeature] log-probability table.</param>
    /// <param name="choicesPerFeature">The number of entries per feature in <paramref name="table"/>.</param>
    /// <returns>The [samples, classes] sums.</returns>
    protected Tensor<T> SumSelectedLogProbabilities(Tensor<int> choices, Tensor<T> table, int choicesPerFeature)
    {
        var engine = Engine;
        int n = choices.Shape[0];
        int features = choices.Shape[1];
        if (n == 0 || NumClasses == 0 || features == 0)
        {
            return new Tensor<T>(new[] { n, NumClasses });
        }

        // Flat table index: (class * features + feature) * choicesPerFeature + choice
        var flatTable = table.Reshape(NumClasses * features * choicesPerFeature);
        var featureOffsets = engine.TensorMultiplyScalar(engine.TensorArange<int>(0, features), choicesPerFeature)
            .Reshape(1, 1, features);
        var classOffsets = engine.TensorMultiplyScalar(engine.TensorArange<int>(0, NumClasses), features * choicesPerFeature)
            .Reshape(1, NumClasses, 1);
        var offsets = engine.TensorAdd(classOffsets, featureOffsets);

        int batch = Math.Max(1, (1 << 22) / Math.Max(1, NumClasses * features));
        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < n; start += batch)
        {
            int rows = Math.Min(batch, n - start);
            var index = engine.TensorAdd(choices.Slice(0, start, start + rows).Reshape(rows, 1, features), offsets);
            var selected = engine.TensorGather(flatTable, index.Reshape(rows * NumClasses * features), 0)
                .Reshape(rows, NumClasses, features);
            blocks.Add(engine.ReduceSum(selected, new[] { 2 }, false));
        }

        return blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 0);
    }

    /// <summary>
    /// log P(sample | class) through the batched <see cref="ComputeLogLikelihoods"/> on a single row.
    /// </summary>
    /// <param name="sample">The sample's features.</param>
    /// <param name="classIndex">The class index.</param>
    /// <returns>The log-likelihood.</returns>
    protected T SingleSampleLogLikelihood(Vector<T> sample, int classIndex)
    {
        var row = sample.AsTensor().Reshape(1, sample.Length).ToMatrix();
        return ComputeLogLikelihoods(row)[0, classIndex];
    }

    /// <summary>
    /// Evaluates <see cref="ComputeLogLikelihood"/> sample by sample (for derived classes without a batch form).
    /// </summary>
    private Tensor<T> PerSampleLogLikelihoods(Matrix<T> input)
    {
        var values = new T[input.Rows * NumClasses];
        for (int i = 0; i < input.Rows; i++)
        {
            var sample = input.GetRow(i);
            for (int c = 0; c < NumClasses; c++)
            {
                values[i * NumClasses + c] = ComputeLogLikelihood(sample, c);
            }
        }

        return new Tensor<T>(values, new[] { input.Rows, NumClasses });
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
        // max + log(sum(exp(x - max))) for numerical stability
        var engine = Engine;
        var tensor = values.AsTensor();
        T max = engine.TensorMaxValue(tensor);
        T sum = engine.TensorSum(engine.TensorExp(engine.TensorSubtractScalar(tensor, max)));
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

        int index = GetClassIndexFromLabel(label);
        if (index < 0)
        {
            throw new ArgumentException($"Label {label} not found in class labels.");
        }

        return index;
    }

    /// <inheritdoc/>
    private Vector<T> PackParameters()
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
    private void UnpackParameters(Vector<T> parameters)
    {
        // When the model is untrained (NumClasses == 0), infer NumClasses from the
        // parameter vector. This is required for the optimizer's InitializeRandomSolution
        // which creates random parameters before the model has seen any training data.
        if (NumClasses == 0 && parameters.Length > 0)
        {
            NumClasses = parameters.Length;
        }

        if (parameters.Length != NumClasses)
        {
            throw new ArgumentException($"Expected {NumClasses} parameters, got {parameters.Length}");
        }
        LogPriors = Vector<T>.Wrap(parameters.ToArray());
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
