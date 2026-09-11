namespace AiDotNet.Evolution.Programs;

/// <summary>Verifies exact protected text across a completed proposal, independent of the requested edit format.</summary>
internal static class ProgramEditBoundary
{
    internal static bool PreservesProtectedText(string parent, string candidate, EvolveBlockMarkers markers)
    {
        var before = EvolveBlock.Extract(parent, markers);
        var after = EvolveBlock.Extract(candidate, markers);
        if (!before.IsWellFormed || !after.IsWellFormed || !before.HasRegions || before.Regions.Count != after.Regions.Count) return false;
        // Extract's region strings use a detected newline convention. Only its line indices are used here:
        // protected mixed CR/LF/CRLF sequences must be compared in the exact original source, not normalized text.
        List<int> parentLines = ProgramText.LineStarts(parent), candidateLines = ProgramText.LineStarts(candidate);
        int parentCursor = 0, candidateCursor = 0;
        for (int index = 0; index < before.Regions.Count; index++)
        {
            var oldRegion = before.Regions[index]; var newRegion = after.Regions[index];
            int parentStart = parentLines[oldRegion.StartLineIndex + 1], candidateStart = candidateLines[newRegion.StartLineIndex + 1];
            int parentLength = parentStart - parentCursor, candidateLength = candidateStart - candidateCursor;
            if (parentLength != candidateLength || string.CompareOrdinal(parent, parentCursor, candidate, candidateCursor, parentLength) != 0) return false;
            parentCursor = parentLines[oldRegion.EndLineIndex]; candidateCursor = candidateLines[newRegion.EndLineIndex];
        }
        int tailLength = parent.Length - parentCursor;
        return tailLength == candidate.Length - candidateCursor && string.CompareOrdinal(parent, parentCursor, candidate, candidateCursor, tailLength) == 0;
    }

}
