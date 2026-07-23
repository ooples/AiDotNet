using System;
using System.Text.RegularExpressions;

namespace AiDotNet.Helpers;

public static class RegexHelper
{
    public static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(1);
    public static readonly TimeSpan FastTimeout = TimeSpan.FromMilliseconds(100);

    public static Regex Create(string pattern, RegexOptions options = RegexOptions.None, TimeSpan? timeout = null)
    {
        return new Regex(pattern, options, timeout ?? DefaultTimeout);
    }

    public static bool IsMatch(string input, string pattern, RegexOptions options = RegexOptions.None, TimeSpan? timeout = null)
    {
        return Regex.IsMatch(input, pattern, options, timeout ?? DefaultTimeout);
    }

    public static Match Match(string input, string pattern, RegexOptions options = RegexOptions.None, TimeSpan? timeout = null)
    {
        return Regex.Match(input, pattern, options, timeout ?? DefaultTimeout);
    }

    public static MatchCollection Matches(string input, string pattern, RegexOptions options = RegexOptions.None, TimeSpan? timeout = null)
    {
        return Regex.Matches(input, pattern, options, timeout ?? DefaultTimeout);
    }

    public static string Replace(string input, string pattern, string replacement, RegexOptions options = RegexOptions.None, TimeSpan? timeout = null)
    {
        return Regex.Replace(input, pattern, replacement, options, timeout ?? DefaultTimeout);
    }

    public static string Replace(string input, string pattern, MatchEvaluator evaluator, RegexOptions options = RegexOptions.None, TimeSpan? timeout = null)
    {
        return Regex.Replace(input, pattern, evaluator, options, timeout ?? DefaultTimeout);
    }

    public static string[] Split(string input, string pattern, RegexOptions options = RegexOptions.None, TimeSpan? timeout = null)
    {
        return Regex.Split(input, pattern, options, timeout ?? DefaultTimeout);
    }

    public static string Escape(string input)
    {
        return Regex.Escape(input);
    }
}
