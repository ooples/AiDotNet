namespace AiDotNet.Helpers;

/// <summary>
/// Provides validation methods for AI model inputs and parameters.
/// </summary>
/// <typeparam name="T">The numeric type used in calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This helper class ensures that the data you provide to AI models is valid and properly formatted.
/// It can handle both traditional matrix/vector inputs (for regression-like models) and tensor inputs (for neural networks).
/// Think of it as a quality control checkpoint that prevents errors before they happen by checking that your
/// data meets all the requirements needed for successful model training and prediction.
/// </remarks>
public static class ValidationHelper<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Validates that input data is properly formatted for model training.
    /// </summary>
    /// <typeparam name="TInput">The type of the input data (e.g., Matrix&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <typeparam name="TOutput">The type of the output data (e.g., Vector&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <param name="x">The input data.</param>
    /// <param name="y">The target data.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method checks that your input data (x) and output data (y) are compatible.
    /// It can handle both traditional matrix/vector pairs (for regression-like models) and tensor pairs (for neural networks).
    /// The method ensures they have matching dimensions and are not null or empty.
    /// </remarks>
    public static void ValidateInputData<TInput, TOutput>(TInput x, TOutput y)
    {
        ValidateDataPair(x, y, "Input");
    }

    /// <summary>
    /// Gets information about the calling method.
    /// </summary>
    /// <param name="skipFrames">Number of frames to skip in the stack trace.</param>
    /// <returns>A tuple containing the component name and operation name.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method identifies which part of the code called a validation function.
    /// It's like caller ID for functions, helping to provide more specific error messages.
    /// You typically won't need to call this directly in your code.
    /// </remarks>
    public static (string component, string operation) GetCallerInfo(int skipFrames = 2)
    {
        try
        {
            // Skip the specified number of frames to get to the actual client code
            var stackTrace = new StackTrace(skipFrames, false);
            var frame = stackTrace.GetFrame(0);

            if (frame != null)
            {
                var method = frame.GetMethod();
                if (method != null)
                {
                    string operation = method.Name;
                    string component = method.DeclaringType?.Name ?? "Unknown";

                    return (component, operation);
                }
            }
        }
        catch (Exception)
        {
            // Fallback if stack trace inspection fails
        }

        // Default values if we can't determine the caller
        return ("Unknown", "Validation");
    }

    /// <summary>
    /// Resolves component and operation names, using caller info if either is empty.
    /// </summary>
    /// <param name="component">The component name, or empty to use caller info.</param>
    /// <param name="operation">The operation name, or empty to use caller info.</param>
    /// <param name="skipFrames">Number of frames to skip in the stack trace.</param>
    /// <returns>A tuple containing the resolved component and operation names.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method helps create more informative error messages by identifying
    /// which part of the library is performing an operation. You typically won't need to call
    /// this directly in your code.
    /// </remarks>
    public static (string component, string operation) ResolveCallerInfo(string component = "", string operation = "", int skipFrames = 3)
    {
        // Only get caller info if needed
        if (string.IsNullOrEmpty(component) || string.IsNullOrEmpty(operation))
        {
            var callerInfo = GetCallerInfo(skipFrames);

            // Only use caller info for empty parameters
            if (string.IsNullOrEmpty(component))
                component = callerInfo.component;

            if (string.IsNullOrEmpty(operation))
                operation = callerInfo.operation;
        }

        return (component, operation);
    }

    /// <summary>
    /// Validates that optimization input data is properly formatted for model training and evaluation.
    /// </summary>
    /// <typeparam name="TInput">The type of the input data (e.g., Matrix&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <typeparam name="TOutput">The type of the output data (e.g., Vector&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <param name="inputData">The optimization input data containing training, validation, and test datasets.</param>
    /// <remarks>
    /// <b>For Beginners:</b> When training AI models, we typically split our data into three sets:
    /// 1. Training data - used to teach the model patterns (like studying for a test)
    /// 2. Validation data - used to tune the model (like practice tests)
    /// 3. Test data - used to evaluate the final model (like the final exam)
    /// 
    /// This method checks that all three datasets are properly formatted and compatible with each other.
    /// It can handle both matrix/vector pairs and tensor pairs.
    /// </remarks>
    public static void ValidateInputData<TInput, TOutput>(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData), "Optimization input data cannot be null.");

        ValidateDataPair(inputData.XTrain, inputData.YTrain, "Training");
        ValidateDataPair(inputData.XValidation, inputData.YValidation, "Validation");
        ValidateDataPair(inputData.XTest, inputData.YTest, "Test");

        // Ensure all inputs have the same shape
        EnsureConsistentInputShape<TInput, TOutput>(inputData.XTrain, inputData.XValidation, inputData.XTest);
    }

    /// <summary>
    /// Validates that data is appropriate for Poisson regression.
    /// </summary>
    /// <param name="y">The target vector containing output values to predict.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Poisson regression is a special type of model used when predicting counts
    /// (like number of customers, number of events, etc.). This method checks that your target values
    /// are non-negative integers (0, 1, 2, etc.), which is required for Poisson regression to work correctly.
    /// 
    /// For example, if you're predicting "number of website visitors per day", each value must be
    /// a whole number (you can't have 3.5 visitors) and can't be negative (you can't have -2 visitors).
    /// </remarks>
    public static void ValidatePoissonData(Vector<T> y)
    {
        // Fail fast on null with a clear message instead of letting
        // y.Length throw NullReferenceException — the latter strips
        // the parameter name and gives no hint about what was wrong.
        if (y is null)
        {
            throw new ArgumentNullException(nameof(y),
                "Poisson target vector cannot be null.");
        }

        // Poisson regression requires non-negative integer count data.
        // Throw on bad input rather than silently coercing — coercion
        // would mask upstream data-quality bugs (e.g., a regression
        // pipeline accidentally feeding continuous values). Callers
        // who genuinely need preprocessing should round/clip the data
        // explicitly before calling this validator, not depend on the
        // validator to do it for them. Method name "Validate" implies
        // a fail-fast throw contract; coercion would be a separate
        // method (CoercePoissonData) if/when we need that path.
        // The first offending sample is the first one that is negative or non-integral (the value
        // differs from its rounding, MathHelper.IsInteger's test); its own test picks the message.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var values = Tensor<T>.FromVector(y);
        var negative = engine.TensorLessThan(values, _numOps.Zero);
        var fractional = engine.TensorNotEquals(values, engine.TensorRound(values));
        int firstBad = FirstNonzeroIndex(engine, engine.TensorAdd(negative, fractional));
        if (firstBad < 0)
        {
            return;
        }

        T offending = y[firstBad];
        if (_numOps.LessThan(offending, _numOps.Zero))
        {
            throw new ArgumentException(
                $"Poisson regression requires non-negative count data; got {_numOps.ToDouble(offending)} at index {firstBad}.",
                nameof(y));
        }

        throw new ArgumentException(
            $"Poisson regression requires integer count data; got {_numOps.ToDouble(offending)} at index {firstBad}.",
            nameof(y));
    }

    /// <summary>
    /// Index of the first nonzero entry of a 1-D 0/1 mask, or -1 when there is none.
    /// </summary>
    /// <remarks>TensorNonzero lists coordinates in row-major order, so its first row is the lowest index.</remarks>
    private static int FirstNonzeroIndex(IEngine engine, Tensor<T> mask)
    {
        var hits = engine.TensorNonzero(mask);
        return hits.Shape[0] > 0 ? hits.GetFlat(0) : -1;
    }

    private static void ValidateDataPair<TInput, TOutput>(TInput x, TOutput y, string datasetName)
    {
        if (x is Matrix<T> xMatrix && y is Vector<T> yVector)
        {
            ValidateMatrixVectorPair(xMatrix, yVector, datasetName);
        }
        else if (x is Tensor<T> xTensor && y is Tensor<T> yTensor)
        {
            ValidateTensorPair(xTensor, yTensor, datasetName);
        }
        else
        {
            throw new ArgumentException($"Invalid input types for {datasetName} dataset. Expected Matrix<T> and Vector<T>, or Tensor<T> and Tensor<T>.");
        }
    }

    private static void ValidateMatrixVectorPair(Matrix<T> x, Vector<T> y, string datasetName)
    {
        if (x == null)
            throw new ArgumentNullException(nameof(x), $"{datasetName} matrix cannot be null.");

        if (y == null)
            throw new ArgumentNullException(nameof(y), $"{datasetName} target vector cannot be null.");

        if (x.Rows != y.Length)
            throw new ArgumentException($"Number of rows in {datasetName.ToLower()} matrix must match the length of the {datasetName.ToLower()} target vector.");

        if (x.Rows == 0 || x.Columns == 0)
            throw new ArgumentException($"{datasetName} matrix cannot be empty.");
    }

    private static void ValidateTensorPair(Tensor<T> x, Tensor<T> y, string datasetName)
    {
        if (x == null)
            throw new ArgumentNullException(nameof(x), $"{datasetName} input tensor cannot be null.");

        if (y == null)
            throw new ArgumentNullException(nameof(y), $"{datasetName} target tensor cannot be null.");

        if (x.Shape[0] != y.Shape[0])
            throw new ArgumentException($"First dimension of {datasetName.ToLower()} input tensor must match the first dimension of the {datasetName.ToLower()} target tensor.");

        if (x._shape.Any(dim => dim == 0) || y._shape.Any(dim => dim == 0))
            throw new ArgumentException($"{datasetName} tensors cannot have zero-sized dimensions.");
    }

    private static void EnsureConsistentInputShape<TInput, TOutput>(TInput xTrain, TInput xValidation, TInput xTest)
    {
        if (xTrain is Matrix<T> xTrainMatrix && xValidation is Matrix<T> xValMatrix && xTest is Matrix<T> xTestMatrix)
        {
            if (xTrainMatrix.Columns != xValMatrix.Columns || xTrainMatrix.Columns != xTestMatrix.Columns)
                throw new ArgumentException("All input matrices must have the same number of columns.");
        }
        else if (xTrain is Tensor<T> xTrainTensor && xValidation is Tensor<T> xValTensor && xTest is Tensor<T> xTestTensor)
        {
            if (!Enumerable.SequenceEqual(xTrainTensor._shape.Skip(1), xValTensor._shape.Skip(1)) ||
                !Enumerable.SequenceEqual(xTrainTensor._shape.Skip(1), xTestTensor._shape.Skip(1)))
                throw new ArgumentException("All input tensors must have the same shape (except for the first dimension).");
        }
        else
        {
            throw new ArgumentException("Inconsistent input types across datasets.");
        }
    }

    /// <summary>
    /// Describes what a target vector actually contains, using the same taxonomy scikit-learn's
    /// <c>type_of_target</c> applies before a classifier will accept a target.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Before a classification model can learn from your answers, it has to work out
    /// what kind of answers you gave it: two categories, several categories, or measured numbers. This
    /// describes which of those it found, so the model can either use them or tell you clearly what is wrong.
    /// </remarks>
    public enum TargetKind
    {
        /// <summary>The target holds exactly two distinct whole-number labels.</summary>
        Binary,

        /// <summary>The target holds three or more distinct whole-number labels.</summary>
        Multiclass,

        /// <summary>The target holds measured values that are not whole numbers.</summary>
        Continuous,

        /// <summary>The target holds a single repeated value, so there is nothing to discriminate.</summary>
        SingleClass
    }

    /// <summary>
    /// Classifies a target vector as binary, multiclass, continuous, or single-class.
    /// </summary>
    /// <param name="y">The target vector to classify.</param>
    /// <returns>The detected <see cref="TargetKind"/> and the distinct values in ascending order.</returns>
    /// <remarks>
    /// <para>
    /// The ordering of the checks matters and deliberately mirrors scikit-learn's <c>type_of_target</c>:
    /// the non-integral test runs FIRST, so a target such as <c>{0.5, 2.5}</c> is reported as
    /// <see cref="TargetKind.Continuous"/> rather than as two classes. Only once the values are known to be
    /// whole numbers does the distinct count decide between binary and multiclass. Reversing those two checks
    /// would silently accept measured data whenever it happened to take only two values.
    /// </para>
    /// <para><b>For Beginners:</b> This looks at your answers and decides whether they are categories
    /// (like "spam"/"not spam") or measurements (like 3.7 inches). Whole numbers that repeat are treated as
    /// categories; anything with a fractional part is treated as a measurement.
    /// </para>
    /// </remarks>
    public static (TargetKind Kind, List<T> Classes) ClassifyTarget(Vector<T> y)
    {
        if (y is null)
        {
            throw new ArgumentNullException(nameof(y), "Target vector cannot be null.");
        }

        if (y.Length == 0)
        {
            throw new ArgumentException("Target vector cannot be empty.", nameof(y));
        }

        // Non-integral values mean measured data, regardless of how few distinct values there are.
        // This check must precede the distinct-count check -- see the remarks above. It is MathHelper.IsInteger's test
        // (the value equals its rounding). The loops read a span: through the vector indexer, validating a 20,000-row
        // target took 1.9 ms, as long as fitting the logistic regression it was validating; over the span, 0.4 ms.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var values = Tensor<T>.FromVector(y);
        if (engine.TensorCountNonzero(engine.TensorNotEquals(values, engine.TensorRound(values))) > 0)
        {
            return (TargetKind.Continuous, new List<T>());
        }

        // Distinct labels, ascending.
        var classes = new List<T>(engine.TensorUnique(values, true).ToVector());

        TargetKind kind = classes.Count == 1
            ? TargetKind.SingleClass
            : classes.Count == 2 ? TargetKind.Binary : TargetKind.Multiclass;

        return (kind, classes);
    }

    /// <summary>
    /// Summarizes a target vector for an error message: how many distinct values it holds and its range.
    /// </summary>
    /// <param name="y">The target vector to summarize.</param>
    /// <returns>A short human-readable description of the observed target.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> When a model rejects your data, this builds the part of the message that tells you
    /// what it actually saw -- how many different answers there were and their smallest and largest values --
    /// so you do not have to go and inspect the data yourself to find out why it was refused.
    /// </remarks>
    public static string DescribeTarget(Vector<T> y)
    {
        double min = double.MaxValue;
        double max = double.MinValue;
        string sample = " (values: )";
        if (y.Length > 0)
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            var engine = AiDotNetEngine.Current;
            var values = Tensor<T>.FromVector(y);
            min = _numOps.ToDouble(engine.TensorMinValue(values));
            max = _numOps.ToDouble(engine.TensorMaxValue(values));

            // Distinct values in first-seen order; listed only when there are at most eight.
            var distinct = engine.TensorUnique(values, false).ToVector();
            sample = distinct.Length <= 8
                ? " (values: " + string.Join(", ", distinct.Select(v => string.Format("{0:G6}", v)).ToArray()) + ")"
                : string.Empty;
        }

        return $"{y.Length} samples in range [{min:G6}, {max:G6}]{sample}";
    }

    /// <summary>
    /// Validates a binary classification target and encodes it to the canonical 0/1 form.
    /// </summary>
    /// <param name="y">The target vector supplied by the caller.</param>
    /// <param name="modelName">The model requesting validation, used in error messages.</param>
    /// <returns>The encoded target (0 for the lower label, 1 for the higher) and the original class labels.</returns>
    /// <exception cref="ArgumentException">Thrown when the target is not two whole-number classes.</exception>
    /// <remarks>
    /// <para>
    /// Two-class targets are accepted whatever the labels are -- <c>{0, 1}</c>, <c>{-1, 1}</c>, <c>{1, 2}</c>
    /// and <c>{3, 7}</c> all train -- and are label-encoded the way scikit-learn's <c>LabelEncoder</c> does it:
    /// sorted ascending, lower label to 0 and higher to 1. Requiring literal 0/1 would reject targets that
    /// carry exactly the same information.
    /// </para>
    /// <para>
    /// Anything that is not two classes throws, and the message names both what was observed and the model to
    /// use instead. It does NOT fall back to fitting a different model: a caller who asked for a classifier and
    /// silently received a least-squares fit would read the resulting coefficients as the classifier's.
    /// </para>
    /// <para><b>For Beginners:</b> This checks that you gave the model exactly two kinds of answer -- yes/no,
    /// spam/not-spam, and so on -- and converts whatever labels you used into the 0-and-1 form the maths needs.
    /// If you gave it something else, it stops and tells you what to use instead of quietly doing something different.
    /// </para>
    /// </remarks>
    public static (Vector<T> Encoded, List<T> Classes) EncodeBinaryTarget(Vector<T> y, string modelName)
    {
        var (kind, classes) = ClassifyTarget(y);

        switch (kind)
        {
            case TargetKind.Binary:
                break;

            case TargetKind.Continuous:
                throw new ArgumentException(
                    $"{modelName} is a classification model and requires two discrete class labels, " +
                    $"but the target is continuous: {DescribeTarget(y)}. " +
                    "For a continuous target use a regression model such as MultipleRegression; " +
                    "for a continuous target confined to (0,1) use BetaRegression; " +
                    "to treat this as classification, threshold the target into two classes first.",
                    nameof(y));

            case TargetKind.Multiclass:
                throw new ArgumentException(
                    $"{modelName} handles exactly two classes, but the target has {classes.Count} distinct " +
                    $"labels: {DescribeTarget(y)}. Use MultinomialLogisticRegression for more than two classes.",
                    nameof(y));

            default:
                throw new ArgumentException(
                    $"{modelName} needs at least two classes to discriminate between, but every sample in the " +
                    $"target has the same label: {DescribeTarget(y)}.",
                    nameof(y));
        }

        // 0 for the lower label, 1 for the other.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var encoded = AiDotNetEngine.Current.TensorNotEquals(Tensor<T>.FromVector(y), classes[0]).ToVector();

        return (encoded, classes);
    }

    /// <summary>
    /// Validates a multiclass target and encodes it to consecutive indices 0..K-1.
    /// </summary>
    /// <param name="y">The target vector supplied by the caller.</param>
    /// <param name="modelName">The model requesting validation, used in error messages.</param>
    /// <returns>The encoded target and the original class labels, ascending, indexed by encoded value.</returns>
    /// <exception cref="ArgumentException">Thrown when the target is continuous or has a single class.</exception>
    /// <remarks>
    /// <para>
    /// Class labels need not be consecutive or zero-based: <c>{2, 5, 9}</c> encodes to <c>{0, 1, 2}</c>. The
    /// returned class list is the inverse mapping, so a caller can report predictions in the caller's own labels.
    /// </para>
    /// <para><b>For Beginners:</b> This checks you gave the model a fixed set of categories and renumbers them
    /// 0, 1, 2, ... internally, remembering your original names so results can be reported back the way you supplied them.
    /// </para>
    /// </remarks>
    public static (Vector<T> Encoded, List<T> Classes) EncodeMulticlassTarget(Vector<T> y, string modelName)
    {
        var (kind, classes) = ClassifyTarget(y);

        if (kind == TargetKind.Continuous)
        {
            throw new ArgumentException(
                $"{modelName} is a classification model and requires discrete class labels, but the target is " +
                $"continuous: {DescribeTarget(y)}. For a continuous target use a regression model such as " +
                "MultipleRegression; to treat this as classification, bin the target into discrete classes first.",
                nameof(y));
        }

        if (kind == TargetKind.SingleClass)
        {
            throw new ArgumentException(
                $"{modelName} needs at least two classes to discriminate between, but every sample in the " +
                $"target has the same label: {DescribeTarget(y)}.",
                nameof(y));
        }

        // Classes are ascending and hold every label, so each label's left insertion point is its class index.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var classTensor = Tensor<T>.FromVector(new Vector<T>(classes));
        var encoded = AiDotNetEngine.Current.TensorSearchSorted(classTensor, Tensor<T>.FromVector(y), false)
            .Cast<T>().ToVector();

        return (encoded, classes);
    }

    /// <summary>
    /// Validates a proportion target for Beta regression, optionally compressing exact 0 and 1 into the open interval.
    /// </summary>
    /// <param name="y">The target vector supplied by the caller.</param>
    /// <param name="modelName">The model requesting validation, used in error messages.</param>
    /// <param name="allowBoundaryCompression">
    /// When true, exact 0 and 1 values are compressed using the Smithson and Verkuilen (2006) transformation
    /// rather than rejected.
    /// </param>
    /// <returns>The validated target, compressed into (0,1) when compression was requested and needed.</returns>
    /// <exception cref="ArgumentException">Thrown when values fall outside [0,1], or hit the boundary with compression disabled.</exception>
    /// <remarks>
    /// <para>
    /// The Beta distribution has no density at 0 or 1, so <c>betareg</c> (R) and <c>statsmodels.BetaModel</c>
    /// both reject targets touching the boundary. That is correct but leaves genuine proportion data -- where a
    /// 0% or 100% observation is perfectly meaningful -- with no path forward. Smithson, M. and Verkuilen, J.
    /// (2006), "A better lemon squeezer? Maximum-likelihood regression with beta-distributed dependent variables",
    /// Psychological Methods 11(1), 54-71, give the standard remedy: y' = (y(n-1) + 0.5) / n, which shrinks the
    /// sample toward 0.5 just enough to clear both boundaries. It is offered opt-in, because silently moving a
    /// caller's data is exactly the sort of hidden substitution this validator exists to prevent.
    /// </para>
    /// <para><b>For Beginners:</b> Beta regression models proportions, which must sit strictly between 0 and 1 --
    /// exactly 0 or exactly 1 breaks the maths. If your data contains those, you can switch on a standard,
    /// published adjustment that nudges every value very slightly inward so the model can fit it.
    /// </para>
    /// </remarks>
    public static Vector<T> ValidateProportionTarget(Vector<T> y, string modelName, bool allowBoundaryCompression)
    {
        if (y is null)
        {
            throw new ArgumentNullException(nameof(y), "Target vector cannot be null.");
        }

        if (y.Length == 0)
        {
            throw new ArgumentException("Target vector cannot be empty.", nameof(y));
        }

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var values = Tensor<T>.FromVector(y);
        var outside = engine.TensorAdd(engine.TensorLessThan(values, _numOps.Zero),
            engine.TensorGreaterThan(values, _numOps.One));
        int firstOutside = FirstNonzeroIndex(engine, outside);
        if (firstOutside >= 0)
        {
            double v = _numOps.ToDouble(y[firstOutside]);
            throw new ArgumentException(
                $"{modelName} models proportions and requires every target value in [0,1], but found " +
                $"{v:G6} at index {firstOutside}: {DescribeTarget(y)}. Rescale the target to a proportion " +
                "(for example y / max) before fitting, or use a regression model such as MultipleRegression " +
                "for an unbounded target.",
                nameof(y));
        }

        // Every value now lies in [0,1] (or is NaN), so touching the boundary means equalling 0 or 1.
        bool touchesBoundary = engine.TensorCountNonzero(engine.TensorEquals(values, _numOps.Zero)) > 0
            || engine.TensorCountNonzero(engine.TensorEquals(values, _numOps.One)) > 0;

        if (!touchesBoundary)
        {
            return y;
        }

        if (!allowBoundaryCompression)
        {
            throw new ArgumentException(
                $"{modelName} requires every target value strictly inside (0,1); the Beta density is undefined " +
                $"at 0 and 1, but the target reaches the boundary: {DescribeTarget(y)}. Set " +
                "BetaRegressionOptions.CompressBoundaryValues = true to apply the Smithson-Verkuilen (2006) " +
                "transformation y' = (y(n-1) + 0.5)/n, or exclude the boundary observations.",
                nameof(y));
        }

        // Smithson & Verkuilen (2006), eq. 4: shrink the whole sample toward 0.5 by one half-observation,
        // which lifts 0 and 1 off the boundary while preserving the ordering and very nearly the spacing.
        int n = y.Length;
        var compressed = engine.TensorDivideScalar(
            engine.TensorAddScalar(engine.TensorMultiplyScalar(values, _numOps.FromDouble(n - 1)), _numOps.FromDouble(0.5)),
            _numOps.FromDouble(n)).ToVector();

        return compressed;
    }
}
