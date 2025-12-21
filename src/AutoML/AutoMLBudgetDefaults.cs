using AiDotNet.Enums;

namespace AiDotNet.AutoML;

internal static class AutoMLBudgetDefaults
{
    public static (TimeSpan timeLimit, int trialLimit) Resolve(AutoMLBudgetPreset preset)
    {
        return preset switch
        {
            AutoMLBudgetPreset.CI => (TimeSpan.FromMinutes(5), 10),
            AutoMLBudgetPreset.Fast => (TimeSpan.FromMinutes(15), 30),
            AutoMLBudgetPreset.Standard => (TimeSpan.FromMinutes(30), 100),
            AutoMLBudgetPreset.Thorough => (TimeSpan.FromHours(2), 300),
            _ => (TimeSpan.FromMinutes(30), 100),
        };
    }
}

