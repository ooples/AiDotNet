namespace AiDotNet.Helpers;

/// <summary>
/// Provides utility methods for working with enumeration types.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This helper class contains methods that make it easier to work with enums in C#.
/// 
/// An enum (short for "enumeration") is a special type in programming that represents a set of 
/// named constants. Think of it like a predefined list of options.
/// 
/// For example, if you have an enum for DaysOfWeek, it might contain: Monday, Tuesday, Wednesday, etc.
/// 
/// This helper class provides methods to get all the values from an enum, which can be useful when
/// you need to process all possible options or present them to a user.
/// </para>
/// </remarks>
public static class EnumHelper
{
    /// <summary>
    /// Gets all values from an enum type as a list, with an option to ignore a specific value.
    /// </summary>
    /// <typeparam name="T">The enum type to get values from.</typeparam>
    /// <param name="ignoreName">Optional name of an enum value to exclude from the result list.</param>
    /// <returns>A list containing all enum values, excluding the ignored value if specified.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method gives you a list of all the options defined in an enum.
    /// 
    /// For example, if you have an enum for Colors (Red, Green, Blue), this method would return
    /// a list containing Red, Green, and Blue.
    /// 
    /// You can also choose to leave out one specific option by providing its name in the 'ignoreName' parameter.
    /// For instance, if you don't want 'Red' in your list, you could call:
    /// GetEnumValues&lt;Colors&gt;("Red")
    /// 
    /// This is useful when you want to present all options to a user except for certain special ones,
    /// or when you need to process all enum values except for specific cases.
    /// </para>
    /// </remarks>
    public static List<T> GetEnumValues<T>(string? ignoreName = null) where T : struct
    {
        var members = typeof(T).GetMembers();
        var result = new List<T>();

        foreach (var member in members)
        {
            //use the member name to get an instance of enumerated type.
            if (Enum.TryParse(member.Name, out T enumType) && (string.IsNullOrEmpty(ignoreName) || member.Name != ignoreName))
            {
                result.Add(enumType);
            }
        }

        return result;
    }
}
