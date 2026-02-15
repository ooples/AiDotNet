namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for individuals supporting multi-objective optimization.
/// </summary>
[AiDotNet.Configuration.YamlConfigurable("MultiObjectiveIndividual")]
public interface IMultiObjectiveIndividual<T>
{
    /// <summary>
    /// Gets the fitness values for multiple objectives.
    /// </summary>
    ICollection<T> GetObjectiveValues();

    /// <summary>
    /// Sets the fitness values for multiple objectives.
    /// </summary>
    void SetObjectiveValues(ICollection<T> values);

    /// <summary>
    /// Gets the rank of the individual in non-dominated sorting.
    /// </summary>
    int GetRank();

    /// <summary>
    /// Sets the rank of the individual in non-dominated sorting.
    /// </summary>
    void SetRank(int rank);

    /// <summary>
    /// Gets the crowding distance of the individual.
    /// </summary>
    double GetCrowdingDistance();

    /// <summary>
    /// Sets the crowding distance of the individual.
    /// </summary>
    void SetCrowdingDistance(double distance);
}
