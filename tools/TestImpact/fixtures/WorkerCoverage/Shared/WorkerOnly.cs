namespace WorkerCoverage;

public static class WorkerOnly
{
    public static int Calculate(int value)
    {
        if (value >= 0) return value * 2;
        return -value;
    }
}
