global using AiDotNet.Factories;
using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using Newtonsoft.Json;

namespace AiDotNet.Classification;

/// <summary>
/// Provides a base implementation for classification algorithms that predict categorical outcomes.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class implements common functionality for classification models, including
/// prediction, serialization/deserialization, and parameter management. Specific classification
/// algorithms should inherit from this class and implement the Train and Predict methods.
/// </para>
/// <para>
/// The class supports various options like class weighting to handle imbalanced datasets
/// and different classification task types (binary, multi-class, multi-label, ordinal).
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Classification is about predicting which category something belongs to.
/// This base class provides the foundation for different classification techniques, handling
/// common operations like making predictions and saving/loading models. Think of it as
/// a template that specific classification algorithms can customize while reusing the shared
/// functionality.
/// </para>
/// </remarks>
public abstract partial class ClassifierBase<T> : IClassifier<T>, IConfigurableModel<T>, IModelShape, IParameterManifestProvider, AiDotNet.Models.Parameters.IParameterChunkSource<T>
{
    /// <summary>Enumerates canonical persistent state without flattening tensor-backed components.</summary>
    public virtual IEnumerable<AiDotNet.Models.Parameters.ParameterChunk<T>> GetParameterStateChunks()
        => AiDotNet.Models.Parameters.ParameterChunkRestore<T>.Read(this, typeof(ClassifierBase<T>), Registry, GetParameters);

    /// <summary>Streams state in the same order as the parameter manifest.</summary>
    public virtual IEnumerable<Tensor<T>> GetParameterChunks()
        => GetParameterStateChunks().Select(chunk => chunk.Tensor);

    /// <summary>Restores tensor-backed chunks on their device and retains custom-setter semantics.</summary>
    public virtual void SetParameterChunks(IEnumerable<Tensor<T>> chunks)
    {
        AiDotNet.Models.Parameters.ParameterChunkRestore<T>.Restore(
            this, typeof(ClassifierBase<T>), chunks, Registry, SetParameters, OnParametersRestored);
    }

    // --- declared state (ModelStateRegistry) ---
    // Identical in every model base because these bases are siblings over the same interfaces rather
    // than one hierarchy; the logic itself lives once in ModelStateRegistry/ModelStateEnvelope.

    /// <summary>State that is not a parameter vector, declared once and persisted by this base.</summary>
    private readonly AiDotNet.Models.ModelStateRegistry<T> _declaredState = new();
    private bool _declaredStateRegistered;

    /// <summary>
    /// Declare state here that the parameter vector does not carry -- a retained training set,
    /// fitted knots, kernel centres, an ensemble's children. Both halves of the payload are driven
    /// by the declaration, so they cannot drift.
    /// </summary>
    /// <param name="state">The registry to declare into.</param>
    protected virtual void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
    }
    /// <summary>Generated state declarations for fields declared across this model's hierarchy.</summary>
    /// <param name="state">The registry to declare into.</param>
    /// <remarks>
    /// Emitted by ModelStateGenerator into the partial model, so a model author declares nothing. The
    /// hand-written <c>RegisterState</c> beside it exists only for state the classifier genuinely
    /// cannot place; anything it CAN place belongs here, where it cannot be forgotten.
    /// </remarks>
    protected virtual void RegisterGeneratedState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        RegisterGeneratedStateCore(state);
    }

    /// <summary>The declared state, registered once and lazily so it runs after the constructor.</summary>
    protected AiDotNet.Models.ModelStateRegistry<T> DeclaredState
    {
        get
        {
            if (!_declaredStateRegistered)
            {
                _declaredStateRegistered = true;
                RegisterGeneratedState(_declaredState);
                RegisterState(_declaredState);
            }
            return _declaredState;
        }
    }
    /// <summary>
    /// Gets the numeric operations for the specified type T.
    /// </summary>
    /// <value>
    /// An object that provides mathematical operations for type T.
    /// </value>
    protected INumericOperations<T> NumOps { get; private set; }

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the execution engine (CPU or GPU) for performing
    /// vectorized operations. The engine is determined by the global AiDotNetEngine configuration
    /// and allows automatic fallback from GPU to CPU when GPU is not available.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This gives access to either CPU or GPU processing for faster computations.
    /// The system automatically chooses the best available option and falls back to CPU
    /// if GPU acceleration is not available.
    /// </para>
    /// </remarks>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the classifier options.
    /// </summary>
    /// <value>
    /// Configuration options for the classifier model.
    /// </value>
    protected ClassifierOptions<T> Options { get; private set; }

    /// <inheritdoc/>
    public virtual ModelOptions GetOptions() => Options;

    /// <summary>
    /// Gets the regularization method used to prevent overfitting.
    /// </summary>
    /// <value>
    /// An object that implements regularization for the classifier model.
    /// </value>
    protected IRegularization<T, Matrix<T>, Vector<T>> Regularization { get; set; }

    /// <summary>
    /// Gets the default loss function for this classifier.
    /// </summary>
    /// <value>
    /// The loss function used for gradient computation.
    /// </value>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Gets or sets the number of classes in the classification problem.
    /// </summary>
    /// <value>
    /// The number of distinct classes learned during training.
    /// </value>
    public int NumClasses { get; protected set; }

    /// <summary>
    /// Gets or sets the type of classification task.
    /// </summary>
    /// <value>
    /// The classification task type (Binary, MultiClass, MultiLabel, or Ordinal).
    /// </value>
    public ClassificationTaskType TaskType { get; protected set; }

    /// <summary>
    /// Gets or sets the class labels learned during training.
    /// </summary>
    /// <value>
    /// A vector containing the unique class labels, or null if not yet trained.
    /// </value>
    public Vector<T>? ClassLabels { get; protected set; }

    /// <summary>
    /// Gets or sets the number of features expected by this classifier.
    /// </summary>
    /// <value>
    /// The number of input features the model was trained on.
    /// </value>
    public int NumFeatures { get; protected set; }

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    /// <value>
    /// An array of feature names. If not set, feature indices will be used as names.
    /// </value>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Gets the expected number of parameters for this model.
    /// </summary>
    /// <value>
    /// The total number of parameters in the model, used for serialization and gradient computation.
    /// </value>
    protected virtual long ExpectedParameterCount => (long)NumFeatures * NumClasses;

    /// <summary>
    /// Initializes a new instance of the ClassifierBase class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the classifier model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <param name="lossFunction">Loss function for gradient computation. If null, defaults to Cross Entropy.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This constructor sets up the classification model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data. The loss function determines how
    /// prediction errors are measured during training.
    /// </para>
    /// </remarks>
    protected ClassifierBase(ClassifierOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null, ILossFunction<T>? lossFunction = null)
    {
        Regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new ClassifierOptions<T>();
        TaskType = Options.TaskType;
        _defaultLossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();
    }

    /// <summary>
    /// Trains the classifier on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target class labels vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to train the classifier.
    /// The target vector should contain class indices (0, 1, 2, ...) for each sample.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Training is the process where the model learns from your data. Different classification algorithms
    /// implement this method differently, but they all aim to learn how to correctly predict the class
    /// labels based on the input features.
    /// </para>
    /// </remarks>
    public abstract void Train(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Predicts class labels for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted class indices for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates predictions for each sample in the input matrix.
    /// The returned vector contains class indices (0, 1, 2, ...).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// After training, this method is used to make predictions on new data. It returns
    /// the predicted class for each input sample as a numeric index.
    /// Use ClassLabels to map these indices back to the original label values if needed.
    /// </para>
    /// </remarks>
    public abstract Vector<T> Predict(Matrix<T> input);

    /// <summary>
    /// Infers the classification task type from the training labels.
    /// </summary>
    /// <param name="y">The target labels vector.</param>
    /// <returns>The inferred classification task type.</returns>
    /// <remarks>
    /// <para>
    /// This method examines the unique values in the target vector to determine
    /// whether this is a binary or multi-class classification problem.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// The model automatically figures out what kind of classification problem you have:
    /// - If there are exactly 2 unique values → Binary classification
    /// - If there are more than 2 unique values → Multi-class classification
    /// </para>
    /// </remarks>
    protected virtual ClassificationTaskType InferTaskType(Vector<T> y)
    {
        int uniqueCount = y.Length == 0 ? 0 : Engine.TensorUnique(y.AsTensor(), sorted: false).Length;
        return uniqueCount == 2 ? ClassificationTaskType.Binary : ClassificationTaskType.MultiClass;
    }

    /// <summary>
    /// Extracts unique class labels from the training data.
    /// </summary>
    /// <param name="y">The target labels vector.</param>
    /// <returns>A sorted vector of unique class labels.</returns>
    protected virtual Vector<T> ExtractClassLabels(Vector<T> y)
    {
        if (y.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorUnique(y.AsTensor(), sorted: true).AsVector();
    }

    /// <summary>
    /// Computes class weights for handling imbalanced datasets.
    /// </summary>
    /// <param name="y">The target labels vector.</param>
    /// <returns>An array of weights for each class.</returns>
    protected virtual double[] ComputeClassWeights(Vector<T> y)
    {
        if (Options.ClassWeights != null && Options.ClassWeights.Length == NumClasses)
        {
            return Options.ClassWeights;
        }

        if (!Options.UseClassWeights)
        {
            var uniformWeights = new double[NumClasses];
            for (int i = 0; i < NumClasses; i++)
            {
                uniformWeights[i] = 1.0;
            }
            return uniformWeights;
        }

        // Count samples per class
        // Unknown labels map to -1; shifting by one sends them to bin 0, which is dropped.
        var classCounts = new int[NumClasses];
        if (y.Length > 0)
        {
            var shifted = Engine.TensorAddScalar(GetClassIndexTensor(y), 1);
            var binCounts = Engine.TensorBinCount(shifted, NumClasses + 1).ToArray();
            Array.Copy(binCounts, 1, classCounts, 0, Math.Min(NumClasses, binCounts.Length - 1));
        }

        // Compute weights: n_samples / (n_classes * n_samples_per_class)
        var weights = new double[NumClasses];
        double nSamples = y.Length;
        for (int i = 0; i < NumClasses; i++)
        {
            if (classCounts[i] > 0)
            {
                weights[i] = nSamples / ((double)NumClasses * classCounts[i]);
            }
            else
            {
                weights[i] = 1.0;
            }
        }

        return weights;
    }

    /// <summary>
    /// Gets the class index for a given label value.
    /// </summary>
    /// <param name="label">The label value to look up.</param>
    /// <returns>The index of the class in ClassLabels, or -1 if not found.</returns>
    /// <remarks>
    /// <para>
    /// This method maps label values to their corresponding class indices using the ClassLabels array.
    /// This is more robust than directly casting labels to indices, as it handles non-0-indexed labels
    /// (e.g., labels like 1, 2, 3 or -1, 1).
    /// </para>
    /// </remarks>
    protected int GetClassIndexFromLabel(T label)
    {
        if (ClassLabels is null || ClassLabels.Length == 0)
        {
            return -1;
        }

        return GetClassIndicesFromLabels(Vector<T>.FromArray(new[] { label }))[0];
    }

    /// <summary>
    /// Maps every label in <paramref name="labels"/> to its class index in <see cref="ClassLabels"/>.
    /// </summary>
    /// <param name="labels">The label values to look up.</param>
    /// <returns>The class index of each label, or -1 where a label is not a known class.</returns>
    /// <remarks>
    /// Batched form of <see cref="GetClassIndexFromLabel"/>: the known labels are sorted once and every
    /// query is located with a single sorted search, so the lookup is a handful of engine calls instead
    /// of a scan per sample. <see cref="ClassLabels"/> does not have to be sorted.
    /// </remarks>
    private protected int[] GetClassIndicesFromLabels(Vector<T> labels)
    {
        return GetClassIndexTensor(labels).ToArray();
    }

    /// <summary>
    /// Tensor form of <see cref="GetClassIndicesFromLabels"/>: a rank-1 <see cref="Tensor{T}"/> of class
    /// indices with -1 for labels that are not known classes.
    /// </summary>
    /// <param name="labels">The label values to look up.</param>
    /// <returns>The class index of each label, or -1 where a label is not a known class.</returns>
    private protected Tensor<int> GetClassIndexTensor(Vector<T> labels)
    {
        int n = labels.Length;
        var result = new Tensor<int>(new[] { n });
        Engine.TensorFill(result, -1);
        if (ClassLabels is null || ClassLabels.Length == 0 || n == 0)
        {
            return result;
        }

        var engine = Engine;
        var (sortedLabels, order) = engine.TensorSort(ClassLabels.AsTensor(), 0, false);
        var query = labels.AsTensor();
        var position = engine.TensorClamp(
            engine.TensorSearchSorted(sortedLabels, query, false), 0, ClassLabels.Length - 1);
        var found = engine.TensorEq(engine.TensorGather(sortedLabels, position, 0), query);
        return engine.TensorWhere(found, engine.TensorGather(order, position, 0), result);
    }

    /// <summary>
    /// Returns, for each row of <paramref name="scores"/>, the column holding the largest value
    /// (the first one on ties), as a rank-1 index tensor.
    /// </summary>
    /// <param name="scores">A [samples, classes] score or probability matrix.</param>
    /// <returns>The arg-max column of each row.</returns>
    private protected Tensor<int> ArgMaxRows(Matrix<T> scores)
    {
        return Engine.TensorArgMax(Tensor<T>.FromMatrix(scores), 1);
    }

    /// <summary>
    /// Converts class indices to their numeric index values (0, 1, 2, ...) as a <see cref="Vector{T}"/>,
    /// by gathering from the index range rather than converting one element at a time.
    /// </summary>
    /// <param name="indices">Class indices in [0, <paramref name="numClasses"/>).</param>
    /// <param name="numClasses">The number of classes.</param>
    /// <returns>The index values as T.</returns>
    private protected Vector<T> ClassIndicesToValues(Tensor<int> indices, int numClasses)
    {
        if (indices.Length == 0)
        {
            return new Vector<T>(0);
        }

        var range = Engine.TensorArange<T>(0, Math.Max(1, numClasses));
        return Engine.TensorGather(range, indices.Reshape(indices.Length), 0).AsVector();
    }

    /// <summary>
    /// Computes sum_k weights[k] * vectors[k] as one matrix-vector product.
    /// </summary>
    /// <param name="vectors">Equal-length vectors (for example per-tree predictions).</param>
    /// <param name="weights">One weight per vector.</param>
    /// <param name="length">The vector length, used when <paramref name="vectors"/> is empty.</param>
    /// <returns>The weighted sum, or a zero vector when there is nothing to add.</returns>
    private protected Vector<T> WeightedSumOfVectors(IReadOnlyList<Vector<T>> vectors, Vector<T> weights, int length)
    {
        if (vectors.Count == 0 || length == 0)
        {
            return new Vector<T>(length);
        }

        var engine = Engine;
        var stacked = new Tensor<T>[vectors.Count];
        for (int k = 0; k < stacked.Length; k++)
        {
            stacked[k] = vectors[k].AsTensor();
        }

        var rows = engine.TensorStack(stacked, 0);
        var sum = engine.TensorMatMul(weights.AsTensor().Reshape(1, vectors.Count), rows);
        return sum.Reshape(length).ToVector();
    }

    /// <summary>
    /// Truncates or zero-pads <paramref name="values"/> to exactly <paramref name="length"/> elements.
    /// </summary>
    private protected static Vector<T> FitToLength(Vector<T> values, int length)
    {
        if (values.Length == length)
        {
            return values;
        }

        if (values.Length > length)
        {
            return values.GetSubVector(0, length);
        }

        return Vector<T>.Concatenate(values, new Vector<T>(length - values.Length));
    }

    /// <summary>
    /// One-hot encodes the class of every label as a [samples, NumClasses] matrix (all-zero rows for labels
    /// that are not known classes).
    /// </summary>
    /// <param name="y">The label vector.</param>
    /// <returns>The one-hot class membership matrix.</returns>
    private protected Tensor<T> ClassMembership(Vector<T> y)
    {
        var engine = Engine;
        if (y.Length == 0 || NumClasses == 0)
        {
            return new Tensor<T>(new[] { y.Length, NumClasses });
        }

        // Unknown labels (-1) are shifted into an extra leading column that is then dropped.
        var shifted = engine.TensorAddScalar(GetClassIndexTensor(y), 1);
        var oneHot = engine.TensorOneHot<T>(shifted, NumClasses + 1);
        return oneHot.Slice(1, 1, NumClasses + 1);
    }

    /// <summary>
    /// Class counts of the samples in <paramref name="samples"/> (class indices outside [0, NumClasses) are ignored).
    /// </summary>
    /// <param name="classes">The class index of every training sample.</param>
    /// <param name="samples">The samples to count.</param>
    /// <returns>The [NumClasses] counts.</returns>
    /// <remarks>
    /// Counted in int: a sum of ones in T stops growing at 2048 for Half.
    /// </remarks>
    private protected Tensor<int> CountClassSamples(Tensor<int> classes, int[] samples)
    {
        var engine = Engine;
        if (samples.Length == 0 || NumClasses == 0)
        {
            return new Tensor<int>(new[] { NumClasses });
        }

        // Shift by one so -1 (unknown) lands in a dropped bin; indices >= NumClasses land in a dropped last bin
        var selected = engine.TensorGather(classes, new Tensor<int>(samples, new[] { samples.Length }), 0);
        var shifted = engine.TensorClamp(engine.TensorAddScalar(selected, 1), 0, NumClasses + 1);
        return engine.TensorBinCount(shifted, NumClasses + 2).Slice(0, 1, NumClasses + 1);
    }

    /// <summary>
    /// The class counts of <paramref name="y"/> (labels that are not known classes are ignored).
    /// </summary>
    /// <param name="y">The label vector.</param>
    /// <returns>The [NumClasses] counts.</returns>
    /// <remarks>
    /// Counted in int: a sum of one-hot rows in T stops growing at 2048 for Half.
    /// </remarks>
    internal Tensor<int> CountClasses(Vector<T> y)
    {
        if (y.Length == 0 || NumClasses == 0)
        {
            return new Tensor<int>(new[] { NumClasses });
        }

        var shifted = Engine.TensorAddScalar(GetClassIndexTensor(y), 1);
        return Engine.TensorBinCount(shifted, NumClasses + 1).Slice(0, 1, NumClasses + 1);
    }

    /// <summary>
    /// <paramref name="votes"/> plus one vote per entry of <paramref name="predictions"/> for its class
    /// (labels that are not known classes get no vote). The votes stay in int: a sum of one-hot rows
    /// in T stops growing at 2048 for Half.
    /// </summary>
    /// <param name="votes">The running [predictions.Length, NumClasses] vote counts.</param>
    /// <param name="predictions">The predicted label of every row.</param>
    /// <returns>The updated vote counts.</returns>
    internal Tensor<int> AddClassVotes(Tensor<int> votes, Vector<T> predictions)
    {
        if (predictions.Length == 0 || NumClasses == 0)
        {
            return votes;
        }

        var shifted = Engine.TensorAddScalar(GetClassIndexTensor(predictions), 1);
        return Engine.TensorAdd(votes, Engine.TensorOneHot<int>(shifted, NumClasses + 1).Slice(1, 1, NumClasses + 1));
    }

    /// <summary>
    /// p-norms (sum |x|^p)^(1/p) along <paramref name="axis"/>, computed as s * (sum (|x| / s)^p)^(1/p)
    /// with s = max |x| along that axis, so that the powers cannot overflow (a Half square overflows
    /// once |x| exceeds 255); all-zero slices get norm 0.
    /// </summary>
    /// <param name="values">The values.</param>
    /// <param name="axis">The axis to reduce.</param>
    /// <param name="p">The norm order (2 for Euclidean).</param>
    /// <returns>The norms, with <paramref name="axis"/> removed.</returns>
    internal static Tensor<T> ScaledNorms(Tensor<T> values, int axis, double p)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var magnitudes = engine.TensorAbs(values);
        var scale = engine.ReduceMax(magnitudes, new[] { axis }, true);
        var ones = new Tensor<T>(scale.Shape.ToArray());
        engine.TensorFill(ones, numOps.One);
        var safeScale = engine.TensorWhere(engine.TensorGreaterThan(scale, numOps.Zero), scale, ones);
        var unit = engine.TensorDivide(magnitudes, engine.TensorBroadcastTo(safeScale, values.Shape.ToArray()));
        var powered = p == 2.0 ? engine.TensorMultiply(unit, unit) : engine.TensorPower(unit, numOps.FromDouble(p));
        var sums = engine.ReduceSum(powered, new[] { axis }, false);
        var norms = p == 2.0 ? engine.TensorSqrt(sums) : engine.TensorPower(sums, numOps.FromDouble(1.0 / p));
        return engine.TensorMultiply(norms, safeScale.Reshape(norms.Shape.ToArray()));
    }

    /// <summary>
    /// The [y.Length, NumClasses] class membership with every entry divided by its class count (computed
    /// in double), so that weights^T * X gives the class means (zero for empty classes) without forming
    /// per-class sums, which overflow Half past 65504.
    /// </summary>
    /// <param name="y">The label vector.</param>
    /// <returns>The membership weights.</returns>
    private protected Tensor<T> ClassMeanWeights(Vector<T> y)
    {
        var engine = Engine;
        if (y.Length == 0 || NumClasses == 0)
        {
            return new Tensor<T>(new[] { y.Length, NumClasses });
        }

        var inverseCounts = engine.TensorReciprocal(engine.TensorClampMin(CountClasses(y), 1).Cast<double>()).Cast<T>();
        return engine.TensorMultiply(ClassMembership(y),
            engine.TensorBroadcastTo(inverseCounts.Reshape(1, NumClasses), new[] { y.Length, NumClasses }));
    }

    /// <summary>
    /// Integer counts as T values.
    /// </summary>
    private protected static Tensor<T> CountsToValues(Tensor<int> counts) => counts.Cast<T>();

    /// <summary>
    /// counts / total as T, with the division done in double so large counts and totals keep their ratio.
    /// </summary>
    /// <param name="counts">Integer counts.</param>
    /// <param name="total">The total to divide by (must be positive).</param>
    /// <returns>The fractions, same shape as <paramref name="counts"/>.</returns>
    private protected static Tensor<T> CountFractions(Tensor<int> counts, int total)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorDivideScalar(counts.Cast<double>(), (double)total).Cast<T>();
    }

    /// <summary>
    /// <paramref name="epsilon"/> as a T, raised to Half's smallest positive normal value (2^-14) for Half,
    /// where 1e-10 and similar guards would otherwise round to zero; float, double and decimal keep it unchanged.
    /// </summary>
    private protected static T Tolerance(double epsilon)
    {
        const double HalfSmallestNormal = 6.103515625e-05;
        return MathHelper.GetNumericOperations<T>().FromDouble(
            typeof(T) == typeof(Half) ? Math.Max(epsilon, HalfSmallestNormal) : epsilon);
    }

    /// <summary>
    /// 1 - <paramref name="epsilon"/> as a T, moved down to the largest T below 1 where 1 - epsilon would round
    /// to 1 (float and Half cannot hold 1 - 1e-10), so clamping a probability to it keeps log(1 - p) finite.
    /// </summary>
    private protected static T OneMinusTolerance(double epsilon)
    {
        // The gap below 1: 2^-24 for float, 2^-11 for Half; double keeps epsilon (1 - 1e-10 is representable)
        double gap = typeof(T) == typeof(float) ? 5.9604644775390625e-08
            : typeof(T) == typeof(Half) ? 4.8828125e-04
            : 0.0;
        return MathHelper.GetNumericOperations<T>().FromDouble(1.0 - Math.Max(epsilon, gap));
    }

    /// <summary>
    /// The split threshold between consecutive distinct values lower &lt; upper: their midpoint, computed as
    /// lower / 2 + upper / 2 so it cannot overflow, and replaced by <paramref name="lower"/> when rounding
    /// pushes it up to <paramref name="upper"/> (as scikit-learn's tree splitter does), so x &lt;= threshold
    /// always separates the two values.
    /// </summary>
    /// <param name="lower">The lower values, any shape.</param>
    /// <param name="upper">The upper values, same shape.</param>
    /// <returns>The thresholds.</returns>
    private protected Tensor<T> SplitThresholds(Tensor<T> lower, Tensor<T> upper)
    {
        var engine = Engine;
        T half = NumOps.FromDouble(0.5);
        var midpoint = engine.TensorAdd(engine.TensorMultiplyScalar(lower, half), engine.TensorMultiplyScalar(upper, half));
        return engine.TensorWhere(engine.TensorGreaterEqual(midpoint, upper), lower, midpoint);
    }

    /// <summary>
    /// The best axis-aligned split of <paramref name="samples"/> by weighted Gini impurity.
    /// </summary>
    /// <remarks>
    /// Candidate thresholds are the midpoints between consecutive distinct feature values; a sample goes
    /// left when its value is &lt;= the threshold. Splits leaving fewer than <paramref name="minSamplesLeaf"/>
    /// samples on a side are skipped. Features are tried in the given order and thresholds in increasing
    /// order; the first strictly lowest weighted Gini wins.
    /// </remarks>
    /// <param name="x">The [n, features] training features.</param>
    /// <param name="classes">The class index of every training sample.</param>
    /// <param name="samples">The samples at this node.</param>
    /// <param name="features">The features to try, in order.</param>
    /// <param name="minSamplesLeaf">The minimum number of samples on each side.</param>
    /// <returns>The feature (-1 when no valid split exists) and threshold.</returns>
    private protected (int Feature, T Threshold) BestGiniSplit(Tensor<T> x, Tensor<int> classes, int[] samples,
        IEnumerable<int> features, int minSamplesLeaf)
    {
        var engine = Engine;
        int total = samples.Length;
        int bestFeature = -1;
        T bestThreshold = NumOps.Zero;
        if (total < 2 || NumClasses == 0)
        {
            return (bestFeature, bestThreshold);
        }

        var sampleTensor = new Tensor<int>(samples, new[] { total });
        var nodeClasses = engine.TensorClamp(engine.TensorGather(classes, sampleTensor, 0), -1, NumClasses);
        var nodeRows = engine.TensorGather(x, sampleTensor, 0);
        var featureScores = new List<Tensor<double>>();
        var featureThresholds = new List<Tensor<T>>();
        var featureIds = new List<int>();

        foreach (int j in features)
        {
            // Sorted values and the class membership of each sorted sample (out-of-range classes count for none)
            var column = nodeRows.Slice(1, j, j + 1).Reshape(total);
            var (sortedValues, order) = engine.TensorSort(column, 0, false);
            var distinct = engine.TensorUniqueConsecutive(sortedValues);
            int numDistinct = distinct.Length;
            if (numDistinct < 2)
            {
                continue;
            }

            var sortedClasses = engine.TensorAddScalar(engine.TensorGather(nodeClasses, order, 0), 1);
            var membership = engine.TensorOneHot<int>(sortedClasses, NumClasses + 2).Slice(1, 1, NumClasses + 1);

            // Left side of the split after distinct value v: every sample with value <= distinct[v]. Counts are
            // kept in int (a running count in T stops at 2048 for Half) and the impurity is evaluated in double.
            int candidates = numDistinct - 1;
            var lastLeft = engine.TensorAddScalar(
                engine.TensorSearchSorted(sortedValues, distinct.Slice(0, 0, candidates), true), -1);
            var leftCounts = engine.TensorGather(engine.TensorCumSum(membership, 0), lastLeft, 0).Cast<double>();
            var leftTotal = engine.TensorAddScalar(lastLeft, 1).Cast<double>().Reshape(candidates, 1);
            var rightCounts = engine.TensorSubtract(engine.ReduceSum(membership, new[] { 0 }, true).Cast<double>(), leftCounts);
            var rightTotal = engine.ScalarMinusTensor((double)total, leftTotal);

            // weighted Gini = (nL / n) * (1 - sum pL^2) + (nR / n) * (1 - sum pR^2)
            var leftShare = engine.TensorDivide(leftCounts, leftTotal);
            var rightShare = engine.TensorDivide(rightCounts, rightTotal);
            var leftGini = engine.ScalarMinusTensor(1.0, engine.ReduceSum(engine.TensorMultiply(leftShare, leftShare), new[] { 1 }, true));
            var rightGini = engine.ScalarMinusTensor(1.0, engine.ReduceSum(engine.TensorMultiply(rightShare, rightShare), new[] { 1 }, true));
            var weighted = engine.TensorAdd(
                engine.TensorMultiply(engine.TensorDivideScalar(leftTotal, (double)total), leftGini),
                engine.TensorMultiply(engine.TensorDivideScalar(rightTotal, (double)total), rightGini)).Reshape(candidates);

            var minLeaf = engine.TensorAddScalar(new Tensor<double>(new[] { candidates, 1 }), (double)minSamplesLeaf);
            var allowed = engine.TensorMultiply(
                engine.TensorGreaterEqual(leftTotal, minLeaf),
                engine.TensorGreaterEqual(rightTotal, minLeaf)).Reshape(candidates);
            var maxValues = engine.TensorAddScalar(new Tensor<double>(new[] { candidates }), double.MaxValue);
            var scores = engine.TensorWhere(allowed, weighted, maxValues);
            int v = engine.TensorArgMin(scores, 0).ToArray()[0];

            // Disallowed splits score double.MaxValue, so they never win below
            featureScores.Add(engine.TensorSlice(scores, new[] { v }, new[] { 1 }));
            featureThresholds.Add(SplitThresholds(
                engine.TensorSlice(distinct, new[] { v }, new[] { 1 }),
                engine.TensorSlice(distinct, new[] { v + 1 }, new[] { 1 })));
            featureIds.Add(j);
        }

        // First feature with the strictly lowest weighted Gini wins
        if (featureScores.Count > 0)
        {
            var allScores = engine.TensorConcatenate(featureScores.ToArray(), 0);
            int best = engine.TensorArgMin(allScores, 0).ToArray()[0];
            if (allScores[best] < double.MaxValue)
            {
                bestFeature = featureIds[best];
                bestThreshold = featureThresholds[best][0];
            }
        }

        return (bestFeature, bestThreshold);
    }

    /// <summary>
    /// Splits <paramref name="samples"/> into those with x[i, feature] &lt;= threshold and the rest, keeping order.
    /// </summary>
    private protected (int[] Left, int[] Right) PartitionByThreshold(Tensor<T> x, int[] samples, int feature, T threshold)
    {
        if (samples.Length == 0)
        {
            return (Array.Empty<int>(), Array.Empty<int>());
        }

        var engine = Engine;
        var column = engine.TensorGather(x.Slice(1, feature, feature + 1).Reshape(x.Shape[0]),
            new Tensor<int>(samples, new[] { samples.Length }), 0);
        var goesLeft = engine.TensorLessEqual(column, FilledVector(samples.Length, threshold).AsTensor());
        var sampleTensor = new Tensor<int>(samples, new[] { samples.Length });
        return (SelectWhereNonzero(sampleTensor, goesLeft), SelectWhereNonzero(sampleTensor, engine.ScalarMinusTensor(NumOps.One, goesLeft)));
    }

    /// <summary>
    /// The entries of <paramref name="values"/> whose 0/1 <paramref name="mask"/> entry is nonzero, in order.
    /// </summary>
    private int[] SelectWhereNonzero(Tensor<int> values, Tensor<T> mask)
    {
        var engine = Engine;
        var positions = engine.TensorNonzero(mask);
        return positions.Length == 0
            ? Array.Empty<int>()
            : engine.TensorGather(values, positions.Reshape(positions.Length), 0).ToArray();
    }

    /// <summary>
    /// Inverts a square matrix by Gauss-Jordan elimination with partial pivoting. A pivot smaller than
    /// 1e-12 in magnitude is replaced by +/-1e-12 (keeping its sign, + for zero), so a singular matrix
    /// yields a large but finite result instead of an exception.
    /// </summary>
    /// <param name="matrix">The square matrix.</param>
    /// <returns>The inverse.</returns>
    private protected Matrix<T> InvertMatrix(Matrix<T> matrix)
    {
        var engine = Engine;
        int n = matrix.Rows;
        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        // Augmented matrix [A | I]
        var identity = engine.TensorEye<T>(n);
        var augmented = engine.TensorConcatenate(new[] { Tensor<T>.FromMatrix(matrix), identity }, 1);
        T minimum = Tolerance(1e-12);
        for (int col = 0; col < n; col++)
        {
            augmented = SwapInPivotRow(augmented, col, n);
            var pivot = ClampPivot(engine.TensorSlice(augmented, new[] { col, col }, new[] { 1, 1 }), minimum);

            // Scale the pivot row, then eliminate the column from every other row (the pivot row's factor is 0)
            var pivotRow = engine.TensorDivide(augmented.Slice(0, col, col + 1), engine.TensorBroadcastTo(pivot, new[] { 1, 2 * n }));
            var column = augmented.Slice(1, col, col + 1);
            var factors = engine.TensorSubtract(column, engine.TensorMultiply(column, identity.Slice(1, col, col + 1)));
            var eliminated = engine.TensorSubtract(augmented, engine.TensorMultiply(factors, pivotRow));
            augmented = ReplaceRow(eliminated, col, pivotRow);
        }

        return augmented.Slice(1, n, 2 * n).ToMatrix();
    }

    /// <summary>
    /// Solves a x = b by Gaussian elimination with partial pivoting and back substitution.
    /// </summary>
    /// <param name="a">The [n, n] coefficient matrix.</param>
    /// <param name="b">The length-n right-hand side.</param>
    /// <param name="handling">How pivots and diagonal entries near zero are treated.</param>
    /// <returns>The solution x.</returns>
    private protected Vector<T> SolveByElimination(Tensor<T> a, Vector<T> b, SmallPivotHandling handling)
    {
        var engine = Engine;
        int n = a.Shape[0];
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        T clampThreshold = Tolerance(1e-12);
        T skipThreshold = Tolerance(1e-10);
        var augmented = ForwardEliminate(
            engine.TensorConcatenate(new[] { a, b.AsTensor().Reshape(n, 1) }, 1), n, handling, out _);
        var upper = augmented.Slice(1, 0, n);
        var rhs = augmented.Slice(1, n, n + 1);

        // Divisor per row: ClampToMinimum turns |diag| < 1e-12 into +1e-12; SkipColumn solves rows with
        // |diag| <= 1e-10 as 0 (divide by 1, then multiply by the 0 keep flag)
        var diag = engine.TensorDiagonal(upper).Reshape(n, 1);
        var absDiag = engine.TensorAbs(diag);
        Tensor<T>? keep = null;
        Tensor<T> divisor;
        if (handling == SmallPivotHandling.ClampToMinimum)
        {
            divisor = engine.TensorWhere(engine.TensorLessThan(absDiag, clampThreshold), Filled(new[] { n, 1 }, clampThreshold), diag);
        }
        else
        {
            keep = engine.TensorGreaterThan(absDiag, skipThreshold);
            divisor = engine.TensorWhere(keep, diag, Filled(new[] { n, 1 }, NumOps.One));
        }

        // Back substitution, one [1, 1] row of the [n, 1] solution at a time
        var solution = new Tensor<T>(new[] { n, 1 });
        for (int i = n - 1; i >= 0; i--)
        {
            var sum = rhs.Slice(0, i, i + 1);
            if (i < n - 1)
            {
                var upperRow = upper.Slice(0, i, i + 1).Slice(1, i + 1, n);
                sum = engine.TensorSubtract(sum, engine.TensorMatMul(upperRow, solution.Slice(0, i + 1, n)));
            }

            var value = engine.TensorDivide(sum, divisor.Slice(0, i, i + 1));
            if (keep is not null)
            {
                value = engine.TensorMultiply(value, keep.Slice(0, i, i + 1));
            }

            solution = ReplaceRow(solution, i, value);
        }

        return solution.Reshape(n).ToVector();
    }

    /// <summary>
    /// Forward Gaussian elimination with partial pivoting on the first <paramref name="n"/> columns of
    /// <paramref name="augmented"/> ([n, m], m &gt;= n): for each column the row with the largest |value| at or
    /// below the diagonal is swapped in, then (row[col] / pivot) * pivotRow is subtracted from every row below.
    /// Entries left of the diagonal in those rows are not meaningful afterwards.
    /// </summary>
    /// <param name="augmented">The [n, m] matrix to reduce.</param>
    /// <param name="n">The number of rows (and eliminated columns).</param>
    /// <param name="handling">How pivots near zero are treated.</param>
    /// <param name="pivots">The pivot used for each column (after any adjustment).</param>
    /// <returns>The reduced matrix.</returns>
    private protected Tensor<T> ForwardEliminate(Tensor<T> augmented, int n, SmallPivotHandling handling, out Tensor<T> pivots)
    {
        var engine = Engine;
        T minimum = NumOps.FromDouble(handling == SmallPivotHandling.SkipColumn ? 1e-10 : 1e-12);
        var pivotList = new List<Tensor<T>>(n);
        for (int col = 0; col < n; col++)
        {
            augmented = SwapInPivotRow(augmented, col, n);
            var pivot = engine.TensorSlice(augmented, new[] { col, col }, new[] { 1, 1 });

            // SkipColumn: a column whose pivot is at most the minimum in magnitude is left as is
            // (its factors are multiplied by the 0 keep flag; the divisor becomes 1)
            Tensor<T>? keep = null;
            switch (handling)
            {
                case SmallPivotHandling.ClampToMinimum:
                    pivot = ClampPivot(pivot, minimum);
                    break;
                case SmallPivotHandling.FloorMagnitude:
                    pivot = engine.TensorWhere(engine.TensorLessThan(engine.TensorAbs(pivot), minimum), Filled(new[] { 1, 1 }, minimum), pivot);
                    break;
                default:
                    keep = engine.TensorGreaterThan(engine.TensorAbs(pivot), minimum);
                    break;
            }

            pivotList.Add(pivot);
            if (col == n - 1)
            {
                continue;
            }

            // row -= (row[col] / pivot) * pivotRow for every row below the pivot
            var pivotRow = augmented.Slice(0, col, col + 1);
            var below = augmented.Slice(0, col + 1, n);
            var divisor = keep is null ? pivot : engine.TensorWhere(keep, pivot, Filled(new[] { 1, 1 }, NumOps.One));
            var factors = engine.TensorDivide(below.Slice(1, col, col + 1), engine.TensorBroadcastTo(divisor, new[] { n - col - 1, 1 }));
            if (keep is not null)
            {
                factors = engine.TensorMultiply(factors, engine.TensorBroadcastTo(keep, new[] { n - col - 1, 1 }));
            }

            var reduced = engine.TensorSubtract(below, engine.TensorMultiply(factors, pivotRow));
            augmented = engine.TensorConcatenate(new[] { augmented.Slice(0, 0, col + 1), reduced }, 0);
        }

        pivots = n == 0 ? new Tensor<T>(new[] { 0 }) : engine.TensorConcatenate(pivotList.ToArray(), 0).Reshape(n);
        return augmented;
    }

    /// <summary>
    /// How <see cref="SolveByElimination"/> and <see cref="ForwardEliminate"/> treat pivots and diagonal entries near zero.
    /// </summary>
    private protected enum SmallPivotHandling
    {
        /// <summary>
        /// Pivots below 1e-12 in magnitude become +/-1e-12 (sign kept, + for zero); diagonal entries
        /// below 1e-12 in magnitude become +1e-12 in back substitution.
        /// </summary>
        ClampToMinimum,

        /// <summary>
        /// Columns whose pivot is at most 1e-10 in magnitude are not eliminated, and their solution
        /// component is 0.
        /// </summary>
        SkipColumn,

        /// <summary>
        /// Pivots below 1e-12 in magnitude become +1e-12 (the sign is dropped).
        /// </summary>
        FloorMagnitude
    }

    /// <summary>
    /// Swaps row <paramref name="col"/> with the row at or below it holding the largest |value| in
    /// column <paramref name="col"/> (the first such row on ties).
    /// </summary>
    private Tensor<T> SwapInPivotRow(Tensor<T> augmented, int col, int n)
    {
        var engine = Engine;
        var candidates = engine.TensorAbs(augmented.Slice(0, col, n).Slice(1, col, col + 1).Reshape(n - col));
        int maxRow = col + engine.TensorArgMax(candidates, 0).ToArray()[0];
        if (maxRow == col)
        {
            return augmented;
        }

        var order = Enumerable.Range(0, n).ToArray();
        order[col] = maxRow;
        order[maxRow] = col;
        return engine.TensorGather(augmented, new Tensor<int>(order, new[] { n }), 0);
    }

    /// <summary>
    /// Replaces a [1, 1] pivot smaller than <paramref name="minimum"/> in magnitude by +/-minimum
    /// (sign kept, + for zero).
    /// </summary>
    private Tensor<T> ClampPivot(Tensor<T> pivot, T minimum)
    {
        var engine = Engine;
        var shape = pivot.Shape.ToArray();
        var signed = engine.TensorWhere(
            engine.TensorLessThan(pivot, NumOps.Zero), Filled(shape, NumOps.Negate(minimum)), Filled(shape, minimum));
        return engine.TensorWhere(engine.TensorLessThan(engine.TensorAbs(pivot), minimum), signed, pivot);
    }

    /// <summary>
    /// A tensor of the given shape with every entry equal to <paramref name="value"/>.
    /// </summary>
    private Tensor<T> Filled(int[] shape, T value) => Engine.TensorAddScalar(new Tensor<T>(shape), value);

    /// <summary>
    /// Returns <paramref name="tensor"/> with row <paramref name="row"/> replaced by the [1, columns] <paramref name="newRow"/>.
    /// </summary>
    private Tensor<T> ReplaceRow(Tensor<T> tensor, int row, Tensor<T> newRow)
    {
        int rows = tensor.Shape[0];
        var parts = new List<Tensor<T>>();
        if (row > 0)
        {
            parts.Add(tensor.Slice(0, 0, row));
        }

        parts.Add(newRow);
        if (row + 1 < rows)
        {
            parts.Add(tensor.Slice(0, row + 1, rows));
        }

        return parts.Count == 1 ? parts[0] : Engine.TensorConcatenate(parts.ToArray(), 0);
    }

    /// <summary>
    /// Selects the rows of <paramref name="x"/> and elements of <paramref name="y"/> at <paramref name="indices"/>
    /// (repeats allowed, in the given order).
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The label vector.</param>
    /// <param name="indices">The sample indices to take.</param>
    /// <returns>The selected rows and labels.</returns>
    private protected (Matrix<T> X, Vector<T> Y) SelectSamples(Matrix<T> x, Vector<T> y, int[] indices)
    {
        if (indices.Length == 0)
        {
            return (new Matrix<T>(0, x.Columns), new Vector<T>(0));
        }

        return (x.GetRows(indices), SelectElements(y, indices));
    }

    /// <summary>
    /// Selects the elements of <paramref name="values"/> at <paramref name="indices"/>.
    /// </summary>
    /// <param name="values">The source vector.</param>
    /// <param name="indices">The indices to take (repeats allowed).</param>
    /// <returns>The gathered elements.</returns>
    private protected Vector<T> SelectElements(Vector<T> values, int[] indices)
    {
        if (indices.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorGather(values.AsTensor(), new Tensor<int>(indices, new[] { indices.Length }), 0).ToVector();
    }

    /// <summary>
    /// Selects the columns of <paramref name="x"/> at <paramref name="columns"/> (in the given order).
    /// </summary>
    /// <param name="x">The source matrix.</param>
    /// <param name="columns">The column indices to take.</param>
    /// <returns>The [x.Rows, columns.Length] matrix of selected columns.</returns>
    private protected Matrix<T> SelectColumns(Matrix<T> x, int[] columns)
    {
        if (x.Rows == 0 || columns.Length == 0)
        {
            return new Matrix<T>(x.Rows, columns.Length);
        }

        return Engine.TensorIndexSelect(Tensor<T>.FromMatrix(x),
            new Tensor<int>(columns, new[] { columns.Length }), 1).ToMatrix();
    }

    /// <summary>
    /// Re-orders the probability columns of an estimator (whose columns follow
    /// <paramref name="sourceLabels"/>) into this model's <see cref="ClassLabels"/> order.
    /// Classes the estimator does not know get probability zero.
    /// </summary>
    /// <param name="probabilities">The estimator's [n, k] probabilities.</param>
    /// <param name="sourceLabels">The estimator's class labels (column order of <paramref name="probabilities"/>).</param>
    /// <returns>The [n, NumClasses] aligned probabilities.</returns>
    private protected Matrix<T> AlignProbabilities(Matrix<T> probabilities, Vector<T>? sourceLabels)
    {
        var aligned = new Tensor<T>(new[] { probabilities.Rows, NumClasses });
        int usable = sourceLabels is null ? 0 : Math.Min(sourceLabels.Length, probabilities.Columns);
        if (sourceLabels is null || usable == 0 || probabilities.Rows == 0 || NumClasses == 0)
        {
            return aligned.ToMatrix();
        }

        var targets = GetClassIndicesFromLabels(FitToLength(sourceLabels, usable));
        var sourceColumns = new List<int>(usable);
        var targetColumns = new List<int>(usable);
        for (int k = 0; k < targets.Length; k++)
        {
            if (targets[k] >= 0)
            {
                sourceColumns.Add(k);
                targetColumns.Add(targets[k]);
            }
        }

        if (sourceColumns.Count == 0)
        {
            return aligned.ToMatrix();
        }

        var picked = SelectColumns(probabilities, sourceColumns.ToArray());
        var targetIndex = new Tensor<int>(targetColumns.ToArray(), new[] { targetColumns.Count });
        return Engine.TensorScatterAdd(aligned, targetIndex, Tensor<T>.FromMatrix(picked), 1).ToMatrix();
    }

    /// <summary>
    /// 1 where <paramref name="values"/>[i] &gt;= <paramref name="threshold"/>, otherwise 0 (NaN gives 0).
    /// </summary>
    /// <param name="values">The values to threshold.</param>
    /// <param name="threshold">The threshold.</param>
    /// <returns>The 0/1 indicator vector.</returns>
    private protected Vector<T> AtLeast(Vector<T> values, T threshold)
    {
        if (values.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorGreaterEqual(values.AsTensor(), FilledVector(values.Length, threshold).AsTensor()).ToVector();
    }

    /// <summary>
    /// The first <paramref name="count"/> columns of <paramref name="x"/> (x itself when it has exactly that many).
    /// </summary>
    /// <param name="x">The source matrix.</param>
    /// <param name="count">The number of leading columns to keep.</param>
    /// <returns>The leading columns.</returns>
    private protected Matrix<T> LeadingColumns(Matrix<T> x, int count)
    {
        return x.Columns == count ? x : SelectColumns(x, Enumerable.Range(0, count).ToArray());
    }

    /// <summary>
    /// Converts a tensor of exact 0/1 values to the matching integer indicators (same shape).
    /// </summary>
    /// <param name="zeroOne">Values that are exactly 0 or 1.</param>
    /// <returns>The 0/1 integers.</returns>
    private protected Tensor<int> ZeroOneToInt(Tensor<T> zeroOne)
    {
        // argmax over [1 - b, b] picks index b
        int axis = zeroOne.Shape.Length;
        var absent = Engine.ScalarMinusTensor(NumOps.One, zeroOne);
        return Engine.TensorArgMax(Engine.TensorStack(new[] { absent, zeroOne }, axis), axis);
    }

    /// <summary>
    /// Serializes a matrix as its row-major T values plus its shape.
    /// </summary>
    private protected static void SerializeMatrixValues(Dictionary<string, object> data, string name, Matrix<T>? matrix)
    {
        if (matrix is null) return;
        data[name] = matrix.Rows == 0 || matrix.Columns == 0 ? Array.Empty<T>() : Tensor<T>.FromMatrix(matrix).ToArray();
        data[$"{name}Rows"] = matrix.Rows;
        data[$"{name}Cols"] = matrix.Columns;
    }

    /// <summary>
    /// Restores a matrix written by <see cref="SerializeMatrixValues"/> (null when absent or empty).
    /// </summary>
    private protected static Matrix<T>? DeserializeMatrixValues(Newtonsoft.Json.Linq.JObject obj, string name)
    {
        var token = obj[name];
        if (token is null) return null;
        var arr = token.ToObject<T[]>() ?? Array.Empty<T>();
        int rows = obj[$"{name}Rows"]?.ToObject<int>() ?? 0;
        int cols = obj[$"{name}Cols"]?.ToObject<int>() ?? 0;
        if (rows <= 0 || cols <= 0) return null;
        if (arr.Length < rows * cols)
        {
            throw new InvalidOperationException(
                $"Deserialization failed: {name} array length {arr.Length} is less than expected {rows}x{cols}={rows * cols}.");
        }

        var values = new T[rows * cols];
        Array.Copy(arr, values, values.Length);
        return new Tensor<T>(values, new[] { rows, cols }).ToMatrix();
    }

    /// <summary>
    /// Serializes a vector as its T values.
    /// </summary>
    private protected static void SerializeVectorValues(Dictionary<string, object> data, string name, Vector<T>? vector)
    {
        if (vector is null) return;
        data[name] = vector.ToArray();
    }

    /// <summary>
    /// Restores a vector written by <see cref="SerializeVectorValues"/> (null when absent or empty).
    /// </summary>
    private protected static Vector<T>? DeserializeVectorValues(Newtonsoft.Json.Linq.JObject obj, string name)
    {
        var token = obj[name];
        if (token is null) return null;
        var arr = token.ToObject<T[]>() ?? Array.Empty<T>();
        return arr.Length == 0 ? null : Vector<T>.Wrap(arr);
    }

    /// <summary>
    /// Majority vote: the label with the most votes per row (ties go to the lowest class index).
    /// </summary>
    /// <param name="votes">The [n, NumClasses] vote counts.</param>
    /// <returns>The winning class label per row.</returns>
    private protected Vector<T> MajorityVoteLabels(Tensor<T> votes)
    {
        if (votes.Shape[0] == 0)
        {
            return new Vector<T>(0);
        }

        return ClassIndicesToLabels(ArgMaxRows(votes.ToMatrix()));
    }

    /// <summary>
    /// Majority vote over integer vote counts: the label with the most votes per row (ties go to the lowest
    /// class index).
    /// </summary>
    /// <param name="votes">The [n, NumClasses] vote counts.</param>
    /// <returns>The winning class label per row.</returns>
    private protected Vector<T> MajorityVoteLabels(Tensor<int> votes)
    {
        if (votes.Shape[0] == 0)
        {
            return new Vector<T>(0);
        }

        return ClassIndicesToLabels(Engine.TensorArgMax(votes, 1));
    }

    /// <summary>
    /// Adds one vote per row for the class at <paramref name="predicted"/>[row] to the [n, NumClasses] counts.
    /// </summary>
    /// <param name="votes">The running [n, NumClasses] vote counts.</param>
    /// <param name="predicted">The class index voted for by each row, in [0, NumClasses).</param>
    /// <returns>The updated counts.</returns>
    private protected Tensor<int> AddVotes(Tensor<int> votes, int[] predicted)
    {
        return Engine.TensorAdd(votes,
            Engine.TensorOneHot<int>(new Tensor<int>(predicted, new[] { predicted.Length }), NumClasses));
    }

    /// <summary>
    /// How often each feature index occurs in <paramref name="splitFeatures"/>, divided by the number of
    /// entries (all zero when there are none).
    /// </summary>
    /// <param name="splitFeatures">The feature index of every split.</param>
    /// <param name="featureCount">The number of features.</param>
    /// <returns>The [featureCount] usage fractions.</returns>
    private protected Tensor<T> FeatureUsageFractions(List<int> splitFeatures, int featureCount)
    {
        if (splitFeatures.Count == 0 || featureCount == 0)
        {
            return new Tensor<T>(new[] { featureCount });
        }

        var counts = Engine.TensorBinCount(new Tensor<int>(splitFeatures.ToArray(), new[] { splitFeatures.Count }), featureCount);
        return CountFractions(counts.Slice(0, 0, featureCount), splitFeatures.Count);
    }

    /// <summary>
    /// Creates a vector of <paramref name="length"/> elements all equal to <paramref name="value"/>.
    /// </summary>
    private protected static Vector<T> FilledVector(int length, T value)
    {
        var result = new Vector<T>(length);
        if (length > 0)
        {
            result.Fill(value);
        }

        return result;
    }

    /// <summary>
    /// Binary decision from a [samples, 2] probability matrix: P(positive) &gt;= 0.5 selects the last class
    /// label, otherwise the first.
    /// </summary>
    /// <param name="probabilities">The probability matrix; column 1 holds P(positive).</param>
    /// <returns>The predicted class labels.</returns>
    private protected Vector<T> PredictBinaryLabels(Matrix<T> probabilities)
    {
        var classLabels = ClassLabels ?? throw new InvalidOperationException("Model must be trained before making predictions.");
        int n = probabilities.Rows;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        var engine = Engine;
        var below = engine.TensorLessThan(probabilities.GetColumn(1).AsTensor(), NumOps.FromDouble(0.5));
        var negative = FilledVector(n, classLabels[0]);
        var positive = FilledVector(n, classLabels[classLabels.Length - 1]);
        return engine.TensorWhere(below, negative.AsTensor(), positive.AsTensor()).ToVector();
    }

    /// <summary>
    /// Converts log-odds to a [samples, 2] probability matrix [1 - sigmoid(z), sigmoid(z)].
    /// </summary>
    /// <param name="logOdds">The log-odds of the positive class per sample.</param>
    /// <returns>The probability matrix.</returns>
    private protected Matrix<T> BinaryProbabilitiesFromLogOdds(Vector<T> logOdds)
    {
        if (logOdds.Length == 0)
        {
            return new Matrix<T>(0, 2);
        }

        var engine = Engine;
        var prob1 = engine.Sigmoid(logOdds);
        var prob0 = engine.ScalarMinusTensor(NumOps.One, prob1.AsTensor()).ToVector();
        return StackColumns(new[] { prob0, prob1 });
    }

    /// <summary>
    /// Mean binary log loss -mean(y * log(p) + (1 - y) * log(1 - p)) with p = sigmoid(logOdds) clamped to
    /// [1e-10, 1 - 1e-10] (the bounds move inward where float or Half cannot hold them).
    /// </summary>
    /// <param name="logOdds">The log-odds of the positive class per sample.</param>
    /// <param name="yBinary">The 0/1 targets.</param>
    /// <returns>The mean log loss (zero for no samples).</returns>
    private protected T ComputeBinaryLogLoss(Vector<T> logOdds, Vector<T> yBinary)
    {
        if (logOdds.Length == 0)
        {
            return NumOps.Zero;
        }

        var engine = Engine;
        var p = engine.TensorClamp(engine.Sigmoid(logOdds).AsTensor(), Tolerance(1e-10), OneMinusTolerance(1e-10));
        var yT = yBinary.AsTensor();
        var oneMinusY = engine.ScalarMinusTensor(NumOps.One, yT);
        var logP = engine.TensorLog(p);
        var logOneMinusP = engine.TensorLog(engine.ScalarMinusTensor(NumOps.One, p));
        var perSample = engine.TensorAdd(engine.TensorMultiply(yT, logP), engine.TensorMultiply(oneMinusY, logOneMinusP));
        return NumOps.Negate(engine.TensorMean(perSample));
    }

    /// <summary>
    /// Stacks equal-length column vectors into a [length, columns.Count] matrix.
    /// </summary>
    /// <param name="columns">The columns, in order.</param>
    /// <returns>The matrix whose j-th column is <paramref name="columns"/>[j].</returns>
    private protected Matrix<T> StackColumns(IReadOnlyList<Vector<T>> columns)
    {
        var stacked = new Tensor<T>[columns.Count];
        for (int k = 0; k < stacked.Length; k++)
        {
            stacked[k] = columns[k].AsTensor();
        }

        return Engine.TensorStack(stacked, 1).ToMatrix();
    }

    /// <summary>
    /// Converts class indices to their class label values from <see cref="ClassLabels"/>.
    /// </summary>
    /// <param name="indices">Class indices in [0, ClassLabels.Length).</param>
    /// <returns>The class labels as T.</returns>
    private protected Vector<T> ClassIndicesToLabels(Tensor<int> indices)
    {
        if (ClassLabels is null || ClassLabels.Length == 0)
        {
            return ClassIndicesToValues(indices, NumClasses);
        }

        if (indices.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorGather(ClassLabels.AsTensor(), indices.Reshape(indices.Length), 0).AsVector();
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, feature count, complexity,
    /// description, and additional information specific to classification.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Model metadata provides information about the model itself, rather than the predictions it makes.
    /// This includes details about the model's structure (like how many features it uses) and characteristics
    /// (like how many classes it can predict). This information can be useful for understanding and
    /// comparing different models.
    /// </para>
    /// </remarks>
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = NumFeatures,
            Complexity = NumFeatures * NumClasses,
            Description = $"{GetType().Name} classifier with {NumFeatures} features and {NumClasses} classes",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumClasses", NumClasses },
                { "TaskType", TaskType.ToString() },
                { "ClassLabels", ClassLabels?.ToArray() ?? Array.Empty<T>() }
            }
        };
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to specify the model type.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method simply returns an identifier that indicates what type of classifier this is
    /// (e.g., Naive Bayes, Random Forest). It's used internally by the library to keep track
    /// of different types of models.
    /// </para>
    /// </remarks>

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters to a JSON format and then converts it to a byte array.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it.
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        return AiDotNet.Models.ModelStateEnvelope.Append(DeclaredState, SerializeInternalUnchecked());
    }

    /// <summary>
    /// Internal, non-virtual, no-guard serialization used by trusted framework
    /// call sites such as <see cref="DeepCopy"/>. Subclasses cannot override
    /// this method, so a subclass override of <see cref="Serialize"/> cannot
    /// intercept the clone path.
    /// </summary>
    private byte[] SerializeInternalUnchecked()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumClasses", NumClasses },
            { "NumFeatures", NumFeatures },
            { "TaskType", (int)TaskType },
            { "ClassLabels", ClassLabels?.ToArray() ?? Array.Empty<T>() },
            { "RegularizationOptions", Regularization.GetOptions() }
        };

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));

        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it.
    /// </para>
    /// </remarks>
    public virtual void Deserialize(byte[] modelData)
    {
        // Strips and applies any declared-state trailer, so the body below reads the payload
        // exactly as it did before this existed.
        modelData = AiDotNet.Models.ModelStateEnvelope.Extract(DeclaredState, modelData);
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        DeserializeInternalUnchecked(modelData);
    }

    /// <summary>
    /// Internal, non-virtual, no-guard deserialization used by trusted framework
    /// call sites such as <see cref="DeepCopy"/>. Subclasses cannot override
    /// this method, so a subclass override of <see cref="Deserialize"/> cannot
    /// intercept the clone path.
    /// </summary>
    private void DeserializeInternalUnchecked(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata == null || modelMetadata.ModelData == null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(modelDataString);

        if (modelDataObj == null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        var numClassesToken = modelDataObj["NumClasses"];
        var numFeaturesToken = modelDataObj["NumFeatures"];
        var taskTypeToken = modelDataObj["TaskType"];
        var classLabelsToken = modelDataObj["ClassLabels"];

        if (numClassesToken == null || numFeaturesToken == null || taskTypeToken == null)
        {
            throw new InvalidOperationException("Deserialization failed: Missing required classification parameters.");
        }

        NumClasses = numClassesToken.ToObject<int>();
        NumFeatures = numFeaturesToken.ToObject<int>();
        TaskType = (ClassificationTaskType)taskTypeToken.ToObject<int>();

        if (classLabelsToken != null)
        {
            // Serialized as T[] (see Serialize), so it is read back as T[] in one call.
            var classLabelValues = classLabelsToken.ToObject<T[]>() ?? Array.Empty<T>();
            if (classLabelValues.Length > 0)
            {
                ClassLabels = Vector<T>.Wrap(classLabelValues);
            }
        }

        var regularizationOptionsToken = modelDataObj["RegularizationOptions"];
        if (regularizationOptionsToken != null)
        {
            var regularizationOptionsJson = JsonConvert.SerializeObject(regularizationOptionsToken);
            var regularizationOptions = JsonConvert.DeserializeObject<RegularizationOptions>(regularizationOptionsJson);
            if (regularizationOptions != null)
            {
                Regularization = RegularizationFactory.CreateRegularization<T, Matrix<T>, Vector<T>>(regularizationOptions);
            }
        }
    }

    /// <summary>
    /// Gets all model parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all model parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns a vector containing all model parameters for use with
    /// optimization algorithms or model comparison.
    /// </para>
    /// <para><b>For Beginners:</b> This method packages all the model's parameters into a single collection.
    /// This is useful for optimization algorithms that need to work with all parameters at once.
    /// </para>
    /// </remarks>

    /// <summary>
    /// Gets the indices of features that are actively used in the model.
    /// </summary>
    /// <returns>An enumerable collection of indices for features that contribute to predictions.</returns>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        // Default implementation: all features are active
        for (int i = 0; i < NumFeatures; i++)
        {
            yield return i;
        }
    }

    /// <summary>
    /// Determines whether a specific feature is used in the model.
    /// </summary>
    /// <param name="featureIndex">The zero-based index of the feature to check.</param>
    /// <returns>True if the feature contributes to predictions; otherwise, false.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the feature index is outside the valid range.</exception>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= NumFeatures)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex),
                $"Feature index must be between 0 and {NumFeatures - 1}");
        }

        // Default implementation: all features are used
        return true;
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to activate.</param>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Default implementation: no-op (subclasses that support feature selection should override)
    }

    /// <summary>
    /// Gets the feature importance scores as a dictionary.
    /// </summary>
    /// <returns>A dictionary mapping feature names to their importance scores.</returns>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();

        for (int i = 0; i < NumFeatures; i++)
        {
            string featureName = FeatureNames != null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[featureName] = NumOps.One; // Default: equal importance
        }

        return result;
    }

    /// <summary>
    /// Creates a deep copy of the classifier model.
    /// </summary>
    /// <returns>A new instance of the model with the same parameters and options.</returns>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        // In-memory clone, not a user save/load — wrap in InternalOperation
        // so the persistence guard does not treat this as a billable op.
        //
        // The clone MUST go through the public virtual Serialize / Deserialize
        // pair so that subclass overrides (which capture trained state like
        // tree ensembles, weight tensors, fitted thresholds, etc.) actually
        // run. The previous implementation routed through the private
        // non-virtual SerializeInternalUnchecked / DeserializeInternalUnchecked
        // helpers to "close the subclass-override bypass surface" — but those
        // base-class helpers only persist NumClasses / NumFeatures / TaskType /
        // ClassLabels / RegularizationOptions. Every classifier with extra
        // trained state (BalancedRandomForest's _trees, RocketClassifier's
        // kernels, OrdinalLogistic's coefficients, every tree-ensemble model's
        // estimator list, etc.) silently lost that state on clone, so the
        // cloned model produced different predictions than the original — the
        // exact failure pattern the Clone_ShouldProduceIdenticalPredictions
        // suite is asserting against. The licensing concern that motivated
        // the bypass is already handled by InternalOperation() above; there
        // was never a real bypass surface to close.
        using (ModelPersistenceGuard.InternalOperation())
        {
            byte[] serialized = Serialize();
            var copy = CreateNewInstance();
            copy.Deserialize(serialized);
            return copy;
        }
    }

    /// <summary>
    /// Creates a new instance of the same type as this classifier.
    /// </summary>
    /// <returns>A new instance of the same classifier type.</returns>
    /// <remarks>
    /// <para>
    /// No longer abstract. Every concrete model used to be forced to write this, and 1147 of them
    /// did -- each one a hand-copied list of constructor arguments that a new option could fall out
    /// of without anything failing. The clone plan records that constructor at compile time instead,
    /// so the base can rebuild the type and a model only overrides this when the generator says it
    /// cannot: a constructor parameter with nothing holding its value, which the build reports by
    /// name rather than leaving to be discovered by a clone that comes back subtly different.
    /// </para>
    /// </remarks>
    protected virtual IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
        => (IFullModel<T, Matrix<T>, Vector<T>>)AiDotNet.Models.CloneEngine.CopyConfiguration(this);

    /// <summary>
    /// Creates a clone of the classifier model.
    /// </summary>
    /// <returns>A new instance of the model with the same parameters and options.</returns>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        return DeepCopy();
    }

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    /// <inheritdoc/>
    public virtual ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <summary>
    /// The components this classifier's parameters live in. Empty until it registers some.
    /// </summary>
    private readonly ParameterComponentRegistry<T> _parameterRegistry = new();
    private bool _componentsRegistered;

    /// <summary>
    /// Declares a component whose parameters belong to this classifier's surface. Registration
    /// order is serialization order, so keep it stable.
    /// </summary>
    protected void RegisterParameterComponent(
        IParameterSource<T>? component,
        [System.Runtime.CompilerServices.CallerArgumentExpression(nameof(component))] string? componentExpression = null,
        [System.Runtime.CompilerServices.CallerMemberName] string? memberName = null)
        => _parameterRegistry.RegisterLegacy(GetType().FullName ?? GetType().Name,
            memberName, componentExpression, component);

    protected void RegisterParameterComponent(string stableId, IParameterSource<T>? component,
        ParameterSlotRole role = ParameterSlotRole.Trainable)
        => _parameterRegistry.Register(stableId, component, role);

    /// <summary>
    /// Declare the trainable components of this classifier here. Called once, lazily, so it runs
    /// after fitting has built them.
    /// </summary>
    protected virtual void RegisterComponents()
    {
    }

    /// <summary>Generated override chain for fields declared across the model hierarchy.</summary>
    protected virtual void RegisterGeneratedParameterComponents(ParameterComponentRegistry<T> registry)
    {
    }

    /// <summary>
    /// Runs after <see cref="SetParameters"/> has distributed values into the components.
    /// </summary>
    protected virtual void OnParametersRestored()
    {
    }

    private ParameterComponentRegistry<T> Registry
    {
        get
        {
            if (!_componentsRegistered)
            {
                RegisterGeneratedParameterComponents(_parameterRegistry);
                RegisterComponents();
                _componentsRegistered = true;
            }
            return _parameterRegistry;
        }
    }

    /// <inheritdoc />
    public ParameterLayoutSnapshot ParameterLayout => Registry.ParameterLayout;

    /// <summary>
    /// Gets this classifier's parameters as a flat vector.
    /// </summary>
    /// <remarks>
    /// Folds the registered components. A classifier that registers none -- a k-NN index, a fitted
    /// tree, an ensemble whose structure is not a vector -- reports an empty vector, and the count
    /// below reports zero to match. That is the honest answer, and it replaces a family of
    /// overrides that returned things like the NUMBER OF TREES as if a count were a parameter.
    /// </remarks>
    public virtual Vector<T> GetParameters() => Registry.GetParameters();

    /// <summary>
    /// Restores this classifier's parameters from a flat vector.
    /// </summary>
    /// <remarks>The inverse of <see cref="GetParameters"/>, down the same enumeration.</remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (!Registry.HasComponents) return;
        Registry.SetParameters(parameters);
        OnParametersRestored();
    }

    /// <summary>
    /// Gets the number of parameters this classifier serializes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Folds the SAME enumeration <see cref="GetParameters"/> does, so the two cannot disagree.
    /// It used to be <see cref="ExpectedParameterCount"/> -- NumFeatures * NumClasses -- which
    /// describes the classifier's SHAPE rather than what it saves, and nothing forced the two to
    /// change together. Seven of the 25 types under this base had already hand-overridden it with
    /// GetParameters().Length, which is the tell; of the eighteen that had not, MiniRocket and
    /// Rocket reported 0 against a six-element vector and OrdinalRegression reported 10.
    /// </para>
    /// <para>
    /// <see cref="ExpectedParameterCount"/> remains for subclasses that want to state a shape
    /// expectation; it is simply no longer mistaken for the count of what is serialized.
    /// </para>
    /// </remarks>
    /// <summary>Returns a copy of this classifier carrying the given parameters.</summary>
    /// <remarks>
    /// Clone-then-set rather than a separate construction path, so the copy is produced by the same
    /// code that produces every other clone and cannot drift from it.
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = (ClassifierBase<T>)Clone();
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    public virtual long ParameterCount => Registry.ParameterCount;

    /// <summary>
    /// Whether this classifier supports parameter initialization.
    /// Used by subclasses that implement IParameterizable.
    /// </summary>
    public virtual bool SupportsParameterInitialization => Registry.CanInitializeOptimizerParameters;

    /// <summary>
    /// Sanitizes parameters to ensure they satisfy model constraints.
    /// Used by subclasses that implement IParameterizable.
    /// </summary>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

    /// <inheritdoc/>
    public virtual int[] GetInputShape()
    {
        return new[] { NumFeatures };
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        return new[] { NumClasses };
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    /// <inheritdoc />
    public virtual void SaveModel(string filePath)
    {
        byte[] serializedData = Serialize();
        byte[] envelopedData = ModelFileHeader.WrapWithHeader(
            serializedData, this, GetInputShape(), GetOutputShape(), SerializationFormat.Json);
        File.WriteAllBytes(filePath, envelopedData);
    }

    /// <summary>
    /// Loads a classifier model from a file.
    /// </summary>
    /// <param name="filePath">The path to the file containing the saved model.</param>
    public virtual void LoadModel(string filePath)
    {
        byte[] serializedData = File.ReadAllBytes(filePath);

        // Extract payload from AIMF envelope if present, otherwise assume legacy raw format
        if (ModelFileHeader.HasHeader(serializedData))
        {
            serializedData = ModelFileHeader.ExtractPayload(serializedData);
        }

        Deserialize(serializedData);
    }

    /// <summary>
    /// Saves the model's current state to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the model state to.</param>
    public virtual void SaveState(Stream stream)
    {
        byte[] serializedData = Serialize();
        stream.Write(serializedData, 0, serializedData.Length);
    }

    /// <summary>
    /// Loads the model's state from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the model state from.</param>
    public virtual void LoadState(Stream stream)
    {
        using var memoryStream = new MemoryStream();
        stream.CopyTo(memoryStream);
        byte[] serializedData = memoryStream.ToArray();
        Deserialize(serializedData);
    }

    // --- IDisposable (issue #1136 plan part 3) ---

    private bool _disposed;

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
        System.GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources held by this classifier. Override and call
    /// base.Dispose(disposing) at the end if the derived class owns
    /// disposable state (layers, GPU handles, rented tensors).
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
    }

}
