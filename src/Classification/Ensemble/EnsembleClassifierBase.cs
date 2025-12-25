using AiDotNet.Classification;
using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Ensemble;

/// <summary>
/// Base class for ensemble classification methods that combine multiple classifiers.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Ensemble methods combine multiple individual classifiers (base estimators) to produce
/// a more robust and accurate prediction than any single classifier could achieve.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Ensemble learning is like getting opinions from a group of experts instead of just one.
///
/// Imagine you want to predict if a movie will be successful. You could:
/// 1. Ask just one expert (single classifier)
/// 2. Ask many experts and combine their opinions (ensemble)
///
/// The second approach is usually more reliable because:
/// - Individual experts may have blind spots that others don't
/// - Combining diverse opinions often leads to better decisions
/// - Errors from one expert may be corrected by others
///
/// Common ensemble strategies:
/// - Bagging: Train on different random subsets of data
/// - Boosting: Train sequentially, focusing on mistakes
/// - Voting: Let classifiers vote on the answer
/// </para>
/// </remarks>
public abstract class EnsembleClassifierBase<T> : ProbabilisticClassifierBase<T>
{
    /// <summary>
    /// The base estimators in the ensemble.
    /// </summary>
    protected List<IClassifier<T>> Estimators { get; set; } = new();

    /// <summary>
    /// The number of estimators in the ensemble.
    /// </summary>
    public int NEstimators => Estimators.Count;

    /// <summary>
    /// Gets or sets the feature importances aggregated across all estimators.
    /// </summary>
    public Vector<T>? FeatureImportances { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the EnsembleClassifierBase class.
    /// </summary>
    /// <param name="options">Configuration options for the ensemble classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    protected EnsembleClassifierBase(ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        ILossFunction<T>? lossFunction = null)
        : base(options ?? new ClassifierOptions<T>(), regularization, lossFunction ?? new CrossEntropyLoss<T>())
    {
    }

    /// <summary>
    /// Aggregates predictions from all estimators in the ensemble.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A matrix of aggregated class probabilities.</returns>
    /// <remarks>
    /// Default implementation averages the probability predictions from all estimators.
    /// Derived classes may override this for different aggregation strategies.
    /// </remarks>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (Estimators.Count == 0)
        {
            throw new InvalidOperationException("Ensemble has no estimators. Model must be trained first.");
        }

        var aggregatedProbs = new Matrix<T>(input.Rows, NumClasses);

        // Collect predictions from all estimators
        var allPredictions = new List<Matrix<T>>();
        foreach (var estimator in Estimators)
        {
            if (estimator is IProbabilisticClassifier<T> probClassifier)
            {
                allPredictions.Add(probClassifier.PredictProbabilities(input));
            }
        }

        if (allPredictions.Count == 0)
        {
            throw new InvalidOperationException("No probabilistic classifiers in ensemble.");
        }

        // Average the probabilities
        for (int i = 0; i < input.Rows; i++)
        {
            for (int c = 0; c < NumClasses; c++)
            {
                T sum = NumOps.Zero;
                foreach (var pred in allPredictions)
                {
                    sum = NumOps.Add(sum, pred[i, c]);
                }
                aggregatedProbs[i, c] = NumOps.Divide(sum, NumOps.FromDouble(allPredictions.Count));
            }
        }

        return aggregatedProbs;
    }

    /// <summary>
    /// Aggregates feature importances from all tree-based estimators.
    /// </summary>
    protected void AggregateFeatureImportances()
    {
        FeatureImportances = new Vector<T>(NumFeatures);

        int treeCount = 0;
        foreach (var estimator in Estimators)
        {
            if (estimator is ITreeBasedClassifier<T> treeClassifier && treeClassifier.FeatureImportances != null)
            {
                for (int i = 0; i < NumFeatures; i++)
                {
                    FeatureImportances[i] = NumOps.Add(FeatureImportances[i], treeClassifier.FeatureImportances[i]);
                }
                treeCount++;
            }
        }

        // Average the importances
        if (treeCount > 0)
        {
            T count = NumOps.FromDouble(treeCount);
            for (int i = 0; i < NumFeatures; i++)
            {
                FeatureImportances[i] = NumOps.Divide(FeatureImportances[i], count);
            }
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Ensemble doesn't have traditional parameters
        // Return feature importances as a representation
        return FeatureImportances ?? new Vector<T>(0);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = Clone();
        if (newModel is EnsembleClassifierBase<T> ensemble)
        {
            ensemble.SetParameters(parameters);
        }
        return newModel;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Ensemble doesn't use traditional parameters
        // This is a no-op for compatibility
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Ensemble methods don't typically use gradient-based optimization
        return new Vector<T>(NumFeatures);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Ensemble methods don't typically use gradient-based optimization
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NEstimators"] = NEstimators;
        metadata.AdditionalInfo["EnsembleType"] = GetType().Name;
        return metadata;
    }
}
