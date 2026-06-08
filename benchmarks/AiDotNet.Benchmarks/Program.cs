using BenchmarkDotNet.Running;

namespace AiDotNet.Benchmarks;

public class Program
{
    public static void Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "survey")
        {
            int n = args.Length > 1 && int.TryParse(args[1], out var pn) ? pn : 2000;
            int p = args.Length > 2 && int.TryParse(args[2], out var pp) ? pp : 8;
            RegressionSurvey.Run(n, p);
            return;
        }

        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
    }
}
