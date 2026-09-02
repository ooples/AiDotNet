using System.Collections.ObjectModel;

namespace AiDotNet.AutoML;

/// <summary>Describes one immutable elite model specification retained by MAP-Elites AutoML.</summary>
/// <remarks>
/// <para>
/// MAP-Elites AutoML keeps the best validated model specification in each behavior cell, where the cell is defined
/// by the model family and the normalized configuration complexity (see <c>MapElitesAutoMLOptions</c>). This type
/// is the public, read-only projection of one such cell: it exposes the specification and its scores, never a
/// trained model instance, so inspecting a large archive does not keep one live model per cell. All collections
/// are defensive copies; mutating the source dictionaries after construction has no effect on the entry.
/// </para>
/// <para><b>For Beginners:</b> After an AutoML run that uses the MAP-Elites strategy, the <c>Archive</c> property
/// of the AutoML model lists one entry like this per occupied cell. Each entry says which model type won that cell,
/// the exact hyperparameters it used, the validation score it earned, and where it sits in the descriptor grid.
/// For example, one entry might read: model type RandomForestRegression, parameters NumberOfTrees = 200 and
/// MaxDepth = 6, score 0.91, descriptors model_family = 2 and configuration_complexity = 0.55, cell bins [2, 2].
/// Browse the archive when you want to understand trade-offs (a simpler model that scores nearly as well) or to
/// retrain a specific elite yourself rather than only the single best model that AutoML returns.</para>
/// </remarks>
public sealed class MapElitesAutoMLArchiveEntry
{
    private readonly ReadOnlyDictionary<string, object> _parameters;
    private readonly ReadOnlyDictionary<string, double> _descriptors;
    private readonly ReadOnlyCollection<int> _cellBins;

    /// <summary>Initializes an archive entry from engine-owned elite data, copying every collection.</summary>
    /// <param name="evaluationId">The stable engine evaluation identifier.</param>
    /// <param name="specificationId">The canonical identity of the model specification.</param>
    /// <param name="modelType">The open generic or concrete model type.</param>
    /// <param name="parameters">The elite hyperparameters to copy.</param>
    /// <param name="score">The validation score used as archive quality.</param>
    /// <param name="descriptors">The behavior descriptors to copy.</param>
    /// <param name="cellBins">One zero-based bin index per descriptor.</param>
    internal MapElitesAutoMLArchiveEntry(
        long evaluationId,
        string specificationId,
        Type modelType,
        IReadOnlyDictionary<string, object> parameters,
        double score,
        IReadOnlyDictionary<string, double> descriptors,
        IReadOnlyList<int> cellBins)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (descriptors is null) throw new ArgumentNullException(nameof(descriptors));
        if (cellBins is null) throw new ArgumentNullException(nameof(cellBins));
        EvaluationId = evaluationId;
        SpecificationId = specificationId ?? throw new ArgumentNullException(nameof(specificationId));
        ModelType = modelType ?? throw new ArgumentNullException(nameof(modelType));
        Score = score;
        _parameters = new ReadOnlyDictionary<string, object>(parameters.ToDictionary(
            item => item.Key,
            item => item.Value,
            StringComparer.Ordinal));
        _descriptors = new ReadOnlyDictionary<string, double>(descriptors.ToDictionary(
            item => item.Key,
            item => item.Value,
            StringComparer.Ordinal));
        _cellBins = Array.AsReadOnly(cellBins.ToArray());
    }

    /// <summary>Gets the stable engine evaluation identifier.</summary>
    public long EvaluationId { get; }

    /// <summary>Gets the canonical identity of the model specification.</summary>
    public string SpecificationId { get; }

    /// <summary>Gets the open generic or concrete model type represented by this elite.</summary>
    public Type ModelType { get; }

    /// <summary>Gets a defensive read-only copy of the elite hyperparameters.</summary>
    public IReadOnlyDictionary<string, object> Parameters => _parameters;

    /// <summary>Gets the validation score used as archive quality.</summary>
    public double Score { get; }

    /// <summary>Gets the behavior descriptors computed without final-test data.</summary>
    public IReadOnlyDictionary<string, double> Descriptors => _descriptors;

    /// <summary>Gets one zero-based bin index per descriptor.</summary>
    public IReadOnlyList<int> CellBins => _cellBins;
}
