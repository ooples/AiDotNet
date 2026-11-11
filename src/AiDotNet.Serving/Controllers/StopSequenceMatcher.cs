using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Stop-sequence matching shared by the OpenAI and Anthropic controllers, so streaming and non-streaming
/// responses (and the two protocols) truncate generated text identically.
/// </summary>
internal static class StopSequenceMatcher
{
    /// <summary>
    /// Returns the index of the earliest occurrence of any stop string in <paramref name="text"/>, or -1 when
    /// none occurs. <paramref name="matchedStop"/> receives the stop string found at that index.
    /// </summary>
    public static int FindStop(string text, IReadOnlyList<string> stops, out string? matchedStop)
    {
        matchedStop = null;
        int earliest = -1;
        foreach (var stop in stops.Where(s => !string.IsNullOrEmpty(s)))
        {
            int i = text.IndexOf(stop, StringComparison.Ordinal);
            if (i >= 0 && (earliest < 0 || i < earliest))
            {
                earliest = i;
                matchedStop = stop;
            }
        }
        return earliest;
    }

    /// <summary>
    /// Length of the longest suffix of <paramref name="text"/> that is a PROPER prefix of some stop string,
    /// i.e. text a streamer must hold back because it could still grow into a stop match. Streamed content
    /// cannot be retracted, so emitting "EN" before "D" completes the stop "END" would leak it. Returns 0 when
    /// no partial stop is pending; a completed stop is reported by <see cref="FindStop"/>, not here.
    /// </summary>
    public static int PendingPrefixLength(string text, IReadOnlyList<string> stops)
    {
        int held = 0;
        foreach (var stop in stops.Where(s => !string.IsNullOrEmpty(s)))
        {
            int max = Math.Min(stop.Length - 1, text.Length);
            for (int k = max; k > held; k--)
            {
                if (string.CompareOrdinal(text, text.Length - k, stop, 0, k) == 0)
                {
                    held = k;
                    break;
                }
            }
        }
        return held;
    }
}
