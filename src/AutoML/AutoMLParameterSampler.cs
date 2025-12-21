using AiDotNet.Enums;

namespace AiDotNet.AutoML;

internal static class AutoMLParameterSampler
{
    public static Dictionary<string, object> Sample(Random random, IReadOnlyDictionary<string, ParameterRange> searchSpace)
    {
        if (random is null)
        {
            throw new ArgumentNullException(nameof(random));
        }

        if (searchSpace is null)
        {
            throw new ArgumentNullException(nameof(searchSpace));
        }

        var sample = new Dictionary<string, object>(StringComparer.Ordinal);

        foreach (var (name, range) in searchSpace)
        {
            sample[name] = SampleSingle(random, range);
        }

        return sample;
    }

    private static object SampleSingle(Random random, ParameterRange range)
    {
        switch (range.Type)
        {
            case ParameterType.Boolean:
                return random.NextDouble() >= 0.5;

            case ParameterType.Categorical:
                if (range.CategoricalValues is null || range.CategoricalValues.Count == 0)
                {
                    return range.DefaultValue ?? string.Empty;
                }

                return range.CategoricalValues[random.Next(range.CategoricalValues.Count)];

            case ParameterType.Integer:
                return SampleInteger(random, range);

            case ParameterType.Float:
            case ParameterType.Continuous:
                return SampleDouble(random, range);

            default:
                return range.DefaultValue ?? string.Empty;
        }
    }

    private static int SampleInteger(Random random, ParameterRange range)
    {
        int min = range.MinValue is null ? 0 : Convert.ToInt32(range.MinValue);
        int max = range.MaxValue is null ? min + 1 : Convert.ToInt32(range.MaxValue);

        if (max < min)
        {
            (min, max) = (max, min);
        }

        if (min == max)
        {
            return min;
        }

        if (range.Step.HasValue && range.Step.Value > 0)
        {
            int step = Math.Max(1, (int)Math.Round(range.Step.Value));
            int count = ((max - min) / step) + 1;
            int idx = random.Next(count);
            return min + (idx * step);
        }

        return random.Next(min, max + 1);
    }

    private static double SampleDouble(Random random, ParameterRange range)
    {
        double min = range.MinValue is null ? 0.0 : Convert.ToDouble(range.MinValue);
        double max = range.MaxValue is null ? min + 1.0 : Convert.ToDouble(range.MaxValue);

        if (max < min)
        {
            (min, max) = (max, min);
        }

        if (Math.Abs(max - min) < double.Epsilon)
        {
            return min;
        }

        if (range.UseLogScale && min > 0 && max > 0)
        {
            double logMin = Math.Log(min);
            double logMax = Math.Log(max);
            double value = logMin + ((logMax - logMin) * random.NextDouble());
            return Math.Exp(value);
        }

        double sampled = min + ((max - min) * random.NextDouble());

        if (range.Step.HasValue && range.Step.Value > 0)
        {
            double step = range.Step.Value;
            double steps = Math.Round((sampled - min) / step);
            sampled = min + (steps * step);
        }

        return sampled;
    }
}

