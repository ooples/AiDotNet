using System.Runtime.InteropServices;
using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// A binding to an installed eSpeak NG library (<c>libespeak-ng</c>), used the way the phonemizer package uses it: voices
/// chosen by language code, and text read clause by clause into IPA phonemes.
/// </summary>
/// <remarks>
/// <para>eSpeak NG is GPL-3.0 and is not part of AiDotNet: the library is loaded at run time from the user's installation —
/// the path in <c>ESPEAK_NG_LIBRARY</c> (or phonemizer's <c>PHONEMIZER_ESPEAK_LIBRARY</c>), else the platform's usual names
/// (<c>libespeak-ng.dll</c> and the eSpeak NG installer's folder on Windows, <c>libespeak-ng.so.1</c> on Linux,
/// <c>libespeak-ng.dylib</c> on macOS). Its data comes from <c>ESPEAK_DATA_PATH</c> when that is set, else the library's
/// own default. Phonemes depend on the eSpeak NG version; <see cref="Version"/> reports the loaded one.</para>
/// <para>eSpeak keeps one global state, so every call is serialized.</para>
/// </remarks>
internal sealed class EspeakNg
{
    private const int AudioOutputSynchronous = 0x02;
    private const int TextModeUtf8 = 1;

    // espeak_TextToPhonemes's phonememode: bit 1 for IPA, bits 8–23 the separator between a word's phonemes ('_').
    private const int PhonemeModeIpaUnderscore = ('_' << 8) | 0x02;

    private static readonly object Gate = new();
    private static EspeakNg? _instance;
    private static string? _failure;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int InitializeFn(int output, int bufferLength, IntPtr path, int options);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate IntPtr InfoFn(IntPtr pathData);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate IntPtr ListVoicesFn(IntPtr voiceSpec);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int SetVoiceByNameFn(IntPtr name);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate IntPtr TextToPhonemesFn(ref IntPtr text, int textMode, int phonemeMode);

    private readonly SetVoiceByNameFn _setVoiceByName;
    private readonly TextToPhonemesFn _textToPhonemes;
    private readonly Dictionary<string, string> _voiceByLanguage = new(StringComparer.Ordinal);
    private string? _currentVoice;

    private EspeakNg(IntPtr library)
    {
        T Load<T>(string name) where T : Delegate =>
            Marshal.GetDelegateForFunctionPointer<T>(NativeLoader.Symbol(library, name)
                ?? throw new InvalidOperationException($"The eSpeak NG library has no '{name}'."));
        var initialize = Load<InitializeFn>("espeak_Initialize");
        var info = Load<InfoFn>("espeak_Info");
        var listVoices = Load<ListVoicesFn>("espeak_ListVoices");
        _setVoiceByName = Load<SetVoiceByNameFn>("espeak_SetVoiceByName");
        _textToPhonemes = Load<TextToPhonemesFn>("espeak_TextToPhonemes");

        string? dataPath = Environment.GetEnvironmentVariable("ESPEAK_DATA_PATH");
        var path = dataPath is null ? IntPtr.Zero : Utf8(dataPath);
        try
        {
            if (initialize(AudioOutputSynchronous, 0, path, 0) <= 0)
                throw new InvalidOperationException("eSpeak NG failed to initialize (is its data installed?).");
        }
        finally
        {
            if (path != IntPtr.Zero) Marshal.FreeHGlobal(path);
        }
        Version = ReadUtf8(info(IntPtr.Zero)) ?? "";

        // espeak_ListVoices(NULL): a null-terminated array of espeak_VOICE*; each voice's languages field is a priority
        // byte and a language name, repeated, ending in a zero byte. phonemizer keeps, per language, the first voice
        // that lists it first.
        var voices = listVoices(IntPtr.Zero);
        for (int i = 0; ; i++)
        {
            var voice = Marshal.ReadIntPtr(voices, i * IntPtr.Size);
            if (voice == IntPtr.Zero) break;
            var languages = Marshal.ReadIntPtr(voice, IntPtr.Size);
            var identifier = ReadUtf8(Marshal.ReadIntPtr(voice, 2 * IntPtr.Size));
            if (languages == IntPtr.Zero || identifier is null) continue;
            var language = ReadUtf8(languages + 1);
            if (!string.IsNullOrEmpty(language) && !_voiceByLanguage.ContainsKey(language!)) _voiceByLanguage[language!] = identifier;
        }
    }

    /// <summary>The loaded eSpeak NG's version (for example "1.49.2").</summary>
    public string Version { get; }

    /// <summary>The language codes the installed voices cover.</summary>
    public IReadOnlyCollection<string> Languages => _voiceByLanguage.Keys;

    /// <summary>The installed eSpeak NG, loaded once; throws with the reason when it cannot be found.</summary>
    public static EspeakNg Instance
    {
        get
        {
            lock (Gate)
            {
                if (_instance is not null) return _instance;
                if (_failure is not null) throw new InvalidOperationException(_failure);
                var library = NativeLoader.Load(Candidates());
                if (library == IntPtr.Zero)
                {
                    _failure = "eSpeak NG was not found. Install espeak-ng (https://github.com/espeak-ng/espeak-ng/releases), or set "
                        + "ESPEAK_NG_LIBRARY to the path of libespeak-ng.";
                    throw new InvalidOperationException(_failure);
                }
                return _instance = new EspeakNg(library);
            }
        }
    }

    /// <summary>Whether an eSpeak NG library can be loaded.</summary>
    public static bool IsAvailable
    {
        get
        {
            try
            {
                _ = Instance;
                return true;
            }
            catch (InvalidOperationException)
            {
                return false;
            }
        }
    }

    private static IEnumerable<string> Candidates()
    {
        foreach (var variable in new[] { "ESPEAK_NG_LIBRARY", "PHONEMIZER_ESPEAK_LIBRARY" })
        {
            var value = Environment.GetEnvironmentVariable(variable);
            if (!string.IsNullOrEmpty(value)) yield return value!;
        }
        if (Path.DirectorySeparatorChar == '\\')
        {
            yield return "libespeak-ng.dll";
            foreach (var root in new[] { Environment.GetEnvironmentVariable("ProgramFiles"), Environment.GetEnvironmentVariable("ProgramFiles(x86)") })
                if (!string.IsNullOrEmpty(root)) yield return Path.Combine(root!, "eSpeak NG", "libespeak-ng.dll");
        }
        else
        {
            yield return "libespeak-ng.so.1";
            yield return "libespeak-ng.so";
            yield return "libespeak-ng.dylib";
            yield return "/usr/local/lib/libespeak-ng.dylib";
            yield return "/opt/homebrew/lib/libespeak-ng.dylib";
        }
    }

    /// <summary>
    /// The IPA phonemes of <paramref name="text"/> in the voice for <paramref name="language"/>, as phonemizer's
    /// <c>EspeakWrapper.text_to_phonemes</c> returns them: eSpeak reads the text clause by clause, each clause's phonemes
    /// (a word's phonemes joined by '_', words by spaces) are kept when not empty, and the clauses are joined by spaces.
    /// </summary>
    public string TextToPhonemes(string language, string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        lock (Gate)
        {
            if (!_voiceByLanguage.TryGetValue(language, out var voice))
                throw new ArgumentException($"eSpeak NG {Version} has no voice for language \"{language}\".", nameof(language));
            if (!string.Equals(voice, _currentVoice, StringComparison.Ordinal))
            {
                var name = Utf8(voice);
                try
                {
                    if (_setVoiceByName(name) != 0) throw new InvalidOperationException($"eSpeak NG failed to load the voice \"{voice}\".");
                }
                finally
                {
                    Marshal.FreeHGlobal(name);
                }
                _currentVoice = voice;
            }
            var buffer = Utf8(text);
            try
            {
                var clauses = new List<string>();
                var position = buffer;
                while (position != IntPtr.Zero)
                {
                    var phonemes = ReadUtf8(_textToPhonemes(ref position, TextModeUtf8, PhonemeModeIpaUnderscore));
                    if (!string.IsNullOrEmpty(phonemes)) clauses.Add(phonemes!);
                }
                return string.Join(" ", clauses);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }
    }

    private static IntPtr Utf8(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        var memory = Marshal.AllocHGlobal(bytes.Length + 1);
        Marshal.Copy(bytes, 0, memory, bytes.Length);
        Marshal.WriteByte(memory, bytes.Length, 0);
        return memory;
    }

    private static string? ReadUtf8(IntPtr pointer)
    {
        if (pointer == IntPtr.Zero) return null;
        int length = 0;
        while (Marshal.ReadByte(pointer, length) != 0) length++;
        var bytes = new byte[length];
        Marshal.Copy(pointer, bytes, 0, length);
        return Encoding.UTF8.GetString(bytes);
    }

    /// <summary>Loads a native library and its symbols on every target (NativeLibrary where it exists, the platform loader
    /// otherwise).</summary>
    private static class NativeLoader
    {
        public static IntPtr Load(IEnumerable<string> candidates)
        {
            foreach (var candidate in candidates)
            {
                var handle = TryLoad(candidate);
                if (handle != IntPtr.Zero) return handle;
            }
            return IntPtr.Zero;
        }

        public static IntPtr? Symbol(IntPtr library, string name)
        {
#if NET5_0_OR_GREATER
            return NativeLibrary.TryGetExport(library, name, out var address) ? address : null;
#else
            var address = Path.DirectorySeparatorChar == '\\' ? GetProcAddress(library, name) : DlSym(library, name);
            return address == IntPtr.Zero ? null : address;
#endif
        }

        private static IntPtr TryLoad(string path)
        {
#if NET5_0_OR_GREATER
            return NativeLibrary.TryLoad(path, out var handle) ? handle : IntPtr.Zero;
#else
            try
            {
                return Path.DirectorySeparatorChar == '\\' ? LoadLibraryW(path) : DlOpen(path, 2);
            }
            catch (DllNotFoundException)
            {
                return IntPtr.Zero;
            }
            catch (EntryPointNotFoundException)
            {
                return IntPtr.Zero;
            }
#endif
        }

#if !NET5_0_OR_GREATER
        [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr LoadLibraryW(string path);

        [DllImport("kernel32", CharSet = CharSet.Ansi, SetLastError = true, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        private static extern IntPtr GetProcAddress(IntPtr module, string name);

        [DllImport("libdl.so.2", EntryPoint = "dlopen")]
        private static extern IntPtr DlOpen(string path, int flags);

        [DllImport("libdl.so.2", EntryPoint = "dlsym")]
        private static extern IntPtr DlSym(IntPtr handle, string name);
#endif
    }
}
