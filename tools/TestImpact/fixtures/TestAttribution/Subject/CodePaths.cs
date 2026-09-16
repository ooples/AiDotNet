namespace AttributionSubject;

public static class CodePaths
{
    public static async Task<int> AsyncLeft(int value)
    {
        await Task.Yield();
        return await Task.Run(() => Operations.Left(value));
    }

    public static int WorkerOnly(int value) => value * 2;
    public static int Unowned() => 59;
    public static int Late() => 61;
}
