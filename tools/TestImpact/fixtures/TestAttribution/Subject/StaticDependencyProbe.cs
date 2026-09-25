namespace AttributionSubject;

// Never executed by the positive cases: dependency extraction must still retain
// implicit initialization and external fields rather than relying on observed hits.
internal static class StaticDependencyProbe
{
    private static readonly int Value = CodePaths.WorkerOnly(3);
    public static int Read() => Value;
    public static DateTime ExternalField() => DateTime.MinValue;
}
