namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a feature selector configured to keep a fixed number of columns.
/// </summary>
/// <remarks>
/// <para>
/// The selector ranks or searches the columns and keeps the requested number of them, like scikit-learn's
/// <c>SelectKBest(k)</c> or <c>RFE(n_features_to_select)</c>.
/// </para>
/// <para><b>For Beginners:</b> You tell these selectors how many columns you want to end up with, for example
/// "keep the best 10", and they pick which 10.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public interface IFixedCountFeatureSelector<T> : IFeatureSelector<T>
{
    /// <summary>
    /// Gets the number of columns the selector was configured to keep.
    /// </summary>
    int NFeaturesToSelect { get; }
}
