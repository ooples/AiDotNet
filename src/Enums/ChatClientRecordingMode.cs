namespace AiDotNet.Enums;

/// <summary>Says whether chat calls are made live, saved to a store, or served back from one.</summary>
/// <remarks>
/// <para>
/// Recording turns an experiment that depends on a hosted model into one that can be repeated exactly. Record a
/// run once against a real provider, commit the resulting file, and every later run — in a test, on a build
/// server, on a colleague's machine — replays the identical answers with no network access, no key, and no cost.
/// A benchmark whose model calls are recorded is reproducible; one whose calls are live is not.
/// </para>
/// <para><b>For Beginners:</b> This decides whether the library talks to a real AI model, saves what it says, or
/// plays back what it said last time. Use <see cref="Record"/> once while you have a working API key, then switch
/// to <see cref="Replay"/>: from then on your run repeats exactly, instantly and for free, which is what makes it
/// usable in automated tests. <see cref="ReplayWithFallback"/> is the middle ground — it reuses saved answers and
/// only calls the model for something it has not seen before.</para>
/// </remarks>
public enum ChatClientRecordingMode
{
    /// <summary>Call the model directly and save nothing.</summary>
    None = 0,

    /// <summary>Call the model and save every request and response into the configured store.</summary>
    Record = 1,

    /// <summary>Serve every request from the store and fail when no recording matches.</summary>
    Replay = 2,

    /// <summary>Serve requests from the store, calling the model and recording the answer on a miss.</summary>
    ReplayWithFallback = 3
}
