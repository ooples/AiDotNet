using System.Text;

namespace AttributionRuntime;

public static class CaseOutputIdentity
{
    public const string Prefix = "[AiDotNet attribution ";
    public static string Format(string run, string token, string caseId) =>
        Prefix + run + ":" + token + ":" + Convert.ToBase64String(Encoding.UTF8.GetBytes(caseId)) + "]";
}
