namespace AiDotNet.Helpers;

public static class EnumHelper
{
    public static List<T> GetEnumValues<T>(string? ignoreName = null) where T : struct
    {
        var members = typeof(T).GetMembers();
        var result = new List<T>();

        foreach (var member in members)
        {
            //use the member name to get an instance of enumerated type.
            if (Enum.TryParse(member.Name, out T enumType) && !string.IsNullOrEmpty(ignoreName) && member.Name != ignoreName)
            {
                result.Add(enumType);
            }
        }

        return result;
    }
}