namespace AiDotNet.Enums;

/// <summary>
/// How a class-weighted feature selector turns class frequencies into class weights.
/// </summary>
/// <remarks>
/// <para>
/// Every strategy starts from the balanced ratio <c>n / (classes * count)</c> of scikit-learn's
/// <c>compute_class_weight("balanced")</c>, so rare classes get larger weights.
/// </para>
/// <para><b>For Beginners:</b> When one class is much rarer than another, these weights make the rare class count
/// more, so the chosen features also separate the rare class well.
/// </para>
/// </remarks>
public enum ClassWeightingStrategy
{
    /// <summary>The balanced ratio <c>n / (classes * count)</c>.</summary>
    Balanced,

    /// <summary>The square root of the balanced ratio, a milder correction.</summary>
    Sqrt,

    /// <summary><c>log(1 + ratio)</c> of the balanced ratio, the mildest correction.</summary>
    Log
}
