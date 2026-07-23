using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Tokenization;

internal static class CodeSpanBuilder
{
    public static List<int> ComputeLineStarts(string text)
    {
        var starts = new List<int> { 0 };

        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] == '\n')
            {
                starts.Add(i + 1);
            }
        }

        return starts;
    }

    public static CodeSpan CreateSpan(IReadOnlyList<int> lineStarts, int startOffset, int endOffset)
    {
        return new CodeSpan
        {
            Start = GetPosition(lineStarts, startOffset),
            End = GetPosition(lineStarts, endOffset)
        };
    }

    public static CodePosition GetPosition(IReadOnlyList<int> lineStarts, int offset)
    {
        int safeOffset = Math.Max(0, offset);

        int lineIndex = FindLineIndex(lineStarts, safeOffset);
        int lineStart = lineStarts[lineIndex];

        return new CodePosition
        {
            Line = lineIndex + 1,
            Column = (safeOffset - lineStart) + 1,
            Offset = safeOffset
        };
    }

    private static int FindLineIndex(IReadOnlyList<int> lineStarts, int offset)
    {
        int lo = 0;
        int hi = lineStarts.Count - 1;
        int best = 0;

        while (lo <= hi)
        {
            int mid = lo + ((hi - lo) / 2);
            int start = lineStarts[mid];

            if (start == offset)
            {
                return mid;
            }

            if (start < offset)
            {
                best = mid;
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return best;
    }
}

