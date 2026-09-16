namespace AiDotNet.TrainingMonitoring.Dashboard;

/// <summary>
/// Copies a series under the same lock used by dashboard writers. Dictionary
/// concurrency alone does not protect the mutable lists stored in it.
/// </summary>
internal static class DashboardSeriesSnapshot
{
    internal static List<T> Copy<T>(List<T> series)
    {
        lock (series)
        {
            return new List<T>(series);
        }
    }
}
