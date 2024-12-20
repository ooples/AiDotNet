namespace AiDotNet.Enums;

public enum FitType
{
    Good,
    Overfit,
    Underfit,
    HighBias,
    HighVariance,
    Unstable,
    SevereMulticollinearity,
    ModerateMulticollinearity,
    PoorFit,
    StrongPositiveAutocorrelation,
    StrongNegativeAutocorrelation,
    WeakAutocorrelation,
    NoAutocorrelation
}