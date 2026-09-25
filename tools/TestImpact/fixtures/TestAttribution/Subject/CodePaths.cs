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

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    public static int HotPath(int value) => value ^ 37;

    public static int UntakenBranch(bool callLeft) => callLeft ? Operations.Left(1) : 0;
}
