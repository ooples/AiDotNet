using System.Collections.ObjectModel;

namespace AiDotNet.AutoML;

/// <summary>Describes one immutable elite model specification retained by MAP-Elites AutoML.</summary>
public sealed class MapElitesAutoMLArchiveEntry
{
    private readonly ReadOnlyDictionary<string, object> _parameters;
    private readonly ReadOnlyDictionary<string, double> _descriptors;
    private readonly ReadOnlyCollection<int> _cellBins;

    internal MapElitesAutoMLArchiveEntry(
        long evaluationId,
        string specificationId,
        Type modelType,
        IReadOnlyDictionary<string, object> parameters,
        double score,
        IReadOnlyDictionary<string, double> descriptors,
        IReadOnlyList<int> cellBins)
    {
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
