using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization.Specialized
{
    /// <summary>
    /// MIDI tokenizer for symbolic music representation.
    /// Supports multiple tokenization strategies for different use cases.
    /// </summary>
    /// <remarks>
    /// <para><b>REMI</b>: Revamped MIDI - Uses Position, Bar, Pitch, Velocity, Duration tokens.
    /// Most expressive, preserves timing and dynamics.</para>
    /// <para><b>CPWord</b>: Compound Word - Combines note attributes into single tokens (e.g., Note_60_16_480).
    /// More compact vocabulary, better for sequence models.</para>
    /// <para><b>SimpleNote</b>: Basic pitch-duration pairs without velocity or position tracking.
    /// Simplest representation, good for melody extraction.</para>
    /// </remarks>
    public class MidiTokenizer : TokenizerBase
    {
        private readonly TokenizationStrategy _strategy;
        private readonly int _ticksPerBeat;
        private readonly int _numVelocityBins;

        /// <summary>
        /// MIDI tokenization strategies.
        /// </summary>
        public enum TokenizationStrategy
        {
            /// <summary>
            /// Revamped MIDI: Position, Bar, Pitch, Velocity, Duration as separate tokens.
            /// </summary>
            REMI,

            /// <summary>
            /// Compound Word: Combines note attributes into single compound tokens.
            /// </summary>
            CPWord,

            /// <summary>
            /// Simple Note: Basic pitch-duration pairs without velocity or position.
            /// </summary>
            SimpleNote
        }

        /// <summary>
        /// Represents a MIDI note event.
        /// </summary>
        public class MidiNote
        {
            /// <summary>
            /// Gets or sets the start tick of the note in MIDI ticks.
            /// </summary>
            public int StartTick { get; set; }

            /// <summary>
            /// Gets or sets the duration of the note in MIDI ticks.
            /// </summary>
            public int Duration { get; set; }

            /// <summary>
            /// Gets or sets the pitch (0-127) representing the musical note.
            /// </summary>
            public int Pitch { get; set; }

            /// <summary>
            /// Gets or sets the velocity (0-127) representing the note intensity.
            /// </summary>
            public int Velocity { get; set; }
        }

        /// <summary>
        /// Creates a new MIDI tokenizer.
        /// </summary>
        /// <param name="vocabulary">The vocabulary to use.</param>
        /// <param name="specialTokens">Special tokens configuration.</param>
        /// <param name="strategy">The tokenization strategy to use.</param>
        /// <param name="ticksPerBeat">MIDI ticks per beat (default: 480).</param>
        /// <param name="numVelocityBins">Number of velocity bins for quantization (default: 32).</param>
        public MidiTokenizer(
            IVocabulary vocabulary,
            SpecialTokens specialTokens,
            TokenizationStrategy strategy = TokenizationStrategy.REMI,
            int ticksPerBeat = 480,
            int numVelocityBins = 32)
            : base(vocabulary, specialTokens)
        {
            _strategy = strategy;
            _ticksPerBeat = ticksPerBeat;
            _numVelocityBins = numVelocityBins;
        }

        /// <summary>
        /// Tokenizes text representation of MIDI.
        /// Format: "NOTE:pitch:duration:velocity" or "REST:duration" or "BAR"
        /// </summary>
        /// <param name="text">The text to tokenize, containing MIDI events separated by newlines or semicolons.</param>
        /// <returns>A list of string tokens representing the MIDI events.</returns>
        public override List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            var tokens = new List<string>();
            var events = text.Split(new[] { '\n', ';' }, StringSplitOptions.RemoveEmptyEntries);

            foreach (var eventStr in events)
            {
                var parts = eventStr.Trim().Split(':');
                if (parts.Length == 0) continue;

                var eventType = parts[0].ToUpperInvariant();

                if (eventType == "NOTE" && parts.Length >= 4)
                {
                    if (int.TryParse(parts[1], out int pitch) &&
                        int.TryParse(parts[2], out int duration) &&
                        int.TryParse(parts[3], out int velocity))
                    {
                        tokens.AddRange(TokenizeNoteEvent(pitch, duration, velocity));
                    }
                }
                else if (eventType == "REST" && parts.Length >= 2)
                {
                    if (int.TryParse(parts[1], out int duration))
                    {
                        tokens.AddRange(TokenizeRestEvent(duration));
                    }
                }
                else if (eventType == "BAR")
                {
                    tokens.AddRange(TokenizeBarEvent());
                }
            }

            return tokens;
        }

        /// <summary>
        /// Tokenizes a list of MIDI notes using the configured strategy.
        /// </summary>
        /// <param name="notes">The collection of MIDI notes to tokenize.</param>
        /// <returns>A list of string tokens representing the MIDI notes in the configured strategy format.</returns>
        public List<string> TokenizeNotes(IEnumerable<MidiNote> notes)
        {
            var noteList = notes.OrderBy(n => n.StartTick).ToList();

            return _strategy switch
            {
                TokenizationStrategy.REMI => TokenizeNotesREMI(noteList),
                TokenizationStrategy.CPWord => TokenizeNotesCPWord(noteList),
                TokenizationStrategy.SimpleNote => TokenizeNotesSimple(noteList),
                _ => TokenizeNotesREMI(noteList)
            };
        }

        /// <summary>
        /// Tokenizes a single note event based on the current strategy.
        /// </summary>
        private List<string> TokenizeNoteEvent(int pitch, int duration, int velocity)
        {
            return _strategy switch
            {
                TokenizationStrategy.REMI => TokenizeNoteREMI(pitch, duration, velocity),
                TokenizationStrategy.CPWord => TokenizeNoteCPWord(pitch, duration, velocity),
                TokenizationStrategy.SimpleNote => TokenizeNoteSimple(pitch, duration),
                _ => TokenizeNoteREMI(pitch, duration, velocity)
            };
        }

        /// <summary>
        /// Tokenizes a rest event based on the current strategy.
        /// </summary>
        private List<string> TokenizeRestEvent(int duration)
        {
            var quantizedDuration = QuantizeDuration(duration);

            return _strategy switch
            {
                TokenizationStrategy.REMI => new List<string> { $"TimeShift_{quantizedDuration}" },
                TokenizationStrategy.CPWord => new List<string> { $"Rest_{quantizedDuration}" },
                TokenizationStrategy.SimpleNote => new List<string> { $"Rest_{quantizedDuration}" },
                _ => new List<string> { $"TimeShift_{quantizedDuration}" }
            };
        }

        /// <summary>
        /// Tokenizes a bar event based on the current strategy.
        /// </summary>
        private List<string> TokenizeBarEvent()
        {
            return _strategy switch
            {
                TokenizationStrategy.REMI => new List<string> { "Bar" },
                TokenizationStrategy.CPWord => new List<string> { "Bar" },
                TokenizationStrategy.SimpleNote => new List<string>(), // SimpleNote doesn't track bars
                _ => new List<string> { "Bar" }
            };
        }

        #region REMI Strategy

        private List<string> TokenizeNoteREMI(int pitch, int duration, int velocity)
        {
            int velocityBin = Math.Min((velocity * _numVelocityBins) / 128, _numVelocityBins - 1);
            return new List<string>
            {
                $"Pitch_{pitch}",
                $"Velocity_{velocityBin}",
                $"Duration_{QuantizeDuration(duration)}"
            };
        }

        private List<string> TokenizeNotesREMI(List<MidiNote> noteList)
        {
            var tokens = new List<string>();
            int currentBar = 0;

            foreach (var note in noteList)
            {
                int noteBar = note.StartTick / (_ticksPerBeat * 4);
                while (currentBar < noteBar)
                {
                    tokens.Add("Bar");
                    currentBar++;
                }

                int positionInBar = (note.StartTick % (_ticksPerBeat * 4)) / (_ticksPerBeat / 4);
                tokens.Add($"Position_{positionInBar}");
                tokens.Add($"Pitch_{note.Pitch}");
                int velocityBin = Math.Min((note.Velocity * _numVelocityBins) / 128, _numVelocityBins - 1);
                tokens.Add($"Velocity_{velocityBin}");
                tokens.Add($"Duration_{QuantizeDuration(note.Duration)}");
            }

            return tokens;
        }

        #endregion

        #region CPWord Strategy

        private List<string> TokenizeNoteCPWord(int pitch, int duration, int velocity)
        {
            int velocityBin = Math.Min((velocity * _numVelocityBins) / 128, _numVelocityBins - 1);
            int quantizedDuration = QuantizeDuration(duration);
            // Compound token: Note_Pitch_VelocityBin_Duration
            return new List<string> { $"Note_{pitch}_{velocityBin}_{quantizedDuration}" };
        }

        private List<string> TokenizeNotesCPWord(List<MidiNote> noteList)
        {
            var tokens = new List<string>();
            int currentBar = 0;
            int lastTick = 0;

            foreach (var note in noteList)
            {
                int noteBar = note.StartTick / (_ticksPerBeat * 4);
                while (currentBar < noteBar)
                {
                    tokens.Add("Bar");
                    currentBar++;
                }

                // Add time shift if there's a gap
                int timeDelta = note.StartTick - lastTick;
                if (timeDelta > 0)
                {
                    int quantizedShift = QuantizeDuration(timeDelta);
                    if (quantizedShift > 0)
                    {
                        tokens.Add($"TimeShift_{quantizedShift}");
                    }
                }

                // Compound token: Note_Pitch_VelocityBin_Duration
                int velocityBin = Math.Min((note.Velocity * _numVelocityBins) / 128, _numVelocityBins - 1);
                int quantizedDuration = QuantizeDuration(note.Duration);
                tokens.Add($"Note_{note.Pitch}_{velocityBin}_{quantizedDuration}");

                lastTick = note.StartTick;
            }

            return tokens;
        }

        #endregion

        #region SimpleNote Strategy

        private List<string> TokenizeNoteSimple(int pitch, int duration)
        {
            int quantizedDuration = QuantizeDuration(duration);
            return new List<string>
            {
                $"Pitch_{pitch}",
                $"Duration_{quantizedDuration}"
            };
        }

        private List<string> TokenizeNotesSimple(List<MidiNote> noteList)
        {
            var tokens = new List<string>();
            int lastTick = 0;

            foreach (var note in noteList)
            {
                // Add rest if there's a gap
                int timeDelta = note.StartTick - lastTick;
                if (timeDelta > _ticksPerBeat / 8) // Only add rest for significant gaps
                {
                    int quantizedRest = QuantizeDuration(timeDelta);
                    if (quantizedRest > 0)
                    {
                        tokens.Add($"Rest_{quantizedRest}");
                    }
                }

                // Simple pitch-duration pair
                tokens.Add($"Pitch_{note.Pitch}");
                tokens.Add($"Duration_{QuantizeDuration(note.Duration)}");

                lastTick = note.StartTick + note.Duration;
            }

            return tokens;
        }

        #endregion

        /// <summary>
        /// Quantizes a duration in ticks to the nearest unit of a 16th note.
        /// </summary>
        /// <param name="ticks">The duration in MIDI ticks.</param>
        /// <returns>The quantized duration clamped to the valid range [1, 128].</returns>
        private int QuantizeDuration(int ticks)
        {
            int quantumSize = _ticksPerBeat / 4;
            // Clamp to valid range [1, 128] - vocabulary has Duration_1 through Duration_128
            int quantized = (ticks + quantumSize / 2) / quantumSize;
            return Math.Max(1, Math.Min(quantized, 128));
        }

        /// <summary>
        /// Cleans up tokens and converts them back to text.
        /// </summary>
        /// <param name="tokens">The tokens to clean up.</param>
        /// <returns>A space-separated string of tokens.</returns>
        protected override string CleanupTokens(List<string> tokens)
        {
            return string.Join(" ", tokens);
        }

        /// <summary>
        /// Creates a MIDI tokenizer with REMI (Revamped MIDI) strategy.
        /// REMI uses Position, Bar, Pitch, Velocity, and Duration as separate tokens.
        /// </summary>
        /// <param name="specialTokens">Special tokens configuration. If null, creates default MIDI special tokens.</param>
        /// <param name="ticksPerBeat">MIDI ticks per beat for timing calculations (default: 480).</param>
        /// <param name="numVelocityBins">Number of velocity bins for quantization (default: 32).</param>
        /// <returns>A new MidiTokenizer configured with the REMI strategy.</returns>
        public static MidiTokenizer CreateREMI(SpecialTokens? specialTokens = null, int ticksPerBeat = 480, int numVelocityBins = 32)
        {
            specialTokens ??= CreateDefaultSpecialTokens();
            var vocabulary = CreateREMIVocabulary(specialTokens, numVelocityBins);
            return new MidiTokenizer(vocabulary, specialTokens, TokenizationStrategy.REMI, ticksPerBeat, numVelocityBins);
        }

        /// <summary>
        /// Creates a MIDI tokenizer with CPWord (Compound Word) strategy.
        /// CPWord combines note attributes into single compound tokens (e.g., Note_60_16_480).
        /// More compact vocabulary, better for sequence models.
        /// </summary>
        /// <param name="specialTokens">Special tokens configuration. If null, creates default MIDI special tokens.</param>
        /// <param name="ticksPerBeat">MIDI ticks per beat for timing calculations (default: 480).</param>
        /// <param name="numVelocityBins">Number of velocity bins for quantization (default: 32).</param>
        /// <returns>A new MidiTokenizer configured with the CPWord strategy.</returns>
        public static MidiTokenizer CreateCPWord(SpecialTokens? specialTokens = null, int ticksPerBeat = 480, int numVelocityBins = 32)
        {
            specialTokens ??= CreateDefaultSpecialTokens();
            var vocabulary = CreateCPWordVocabulary(specialTokens, numVelocityBins);
            return new MidiTokenizer(vocabulary, specialTokens, TokenizationStrategy.CPWord, ticksPerBeat, numVelocityBins);
        }

        /// <summary>
        /// Creates a MIDI tokenizer with SimpleNote strategy.
        /// SimpleNote provides basic pitch-duration pairs without velocity or position tracking.
        /// Simplest representation, good for melody extraction.
        /// </summary>
        /// <param name="specialTokens">Special tokens configuration. If null, creates default MIDI special tokens.</param>
        /// <param name="ticksPerBeat">MIDI ticks per beat for timing calculations (default: 480).</param>
        /// <returns>A new MidiTokenizer configured with the SimpleNote strategy.</returns>
        public static MidiTokenizer CreateSimpleNote(SpecialTokens? specialTokens = null, int ticksPerBeat = 480)
        {
            specialTokens ??= CreateDefaultSpecialTokens();
            var vocabulary = CreateSimpleNoteVocabulary(specialTokens);
            return new MidiTokenizer(vocabulary, specialTokens, TokenizationStrategy.SimpleNote, ticksPerBeat, 1);
        }

        /// <summary>
        /// Creates the default special tokens for MIDI tokenization.
        /// </summary>
        /// <returns>A SpecialTokens instance with default MIDI tokens.</returns>
        private static SpecialTokens CreateDefaultSpecialTokens()
        {
            return new SpecialTokens
            {
                UnkToken = "<unk>",
                PadToken = "<pad>",
                BosToken = "<bos>",
                EosToken = "<eos>",
                ClsToken = string.Empty,
                SepToken = string.Empty,
                MaskToken = string.Empty
            };
        }

        /// <summary>
        /// Creates the vocabulary for REMI tokenization strategy.
        /// Includes tokens for Bar, Position (0-15), Pitch (0-127), Velocity (0-numVelocityBins-1),
        /// Duration (1-128), and TimeShift (1-128).
        /// </summary>
        /// <param name="specialTokens">The special tokens to include in the vocabulary.</param>
        /// <param name="numVelocityBins">The number of velocity bins.</param>
        /// <returns>A vocabulary configured for REMI tokenization.</returns>
        private static Vocabulary.Vocabulary CreateREMIVocabulary(SpecialTokens specialTokens, int numVelocityBins)
        {
            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);
            vocabulary.AddTokens(specialTokens.GetAllSpecialTokens());

            vocabulary.AddToken("Bar");
            for (int i = 0; i < 16; i++)
                vocabulary.AddToken($"Position_{i}");

            for (int pitch = 0; pitch < 128; pitch++)
                vocabulary.AddToken($"Pitch_{pitch}");

            for (int v = 0; v < numVelocityBins; v++)
                vocabulary.AddToken($"Velocity_{v}");

            for (int d = 1; d <= 128; d++)
            {
                vocabulary.AddToken($"Duration_{d}");
                vocabulary.AddToken($"TimeShift_{d}");
            }

            return vocabulary;
        }

        /// <summary>
        /// Creates the vocabulary for CPWord (Compound Word) tokenization strategy.
        /// Includes tokens for Bar, TimeShift (1-128), Rest (1-128), and compound note tokens
        /// in the format Note_Pitch_VelocityBin_Duration for all combinations.
        /// </summary>
        /// <param name="specialTokens">The special tokens to include in the vocabulary.</param>
        /// <param name="numVelocityBins">The number of velocity bins.</param>
        /// <returns>A vocabulary configured for CPWord tokenization.</returns>
        private static Vocabulary.Vocabulary CreateCPWordVocabulary(SpecialTokens specialTokens, int numVelocityBins)
        {
            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);
            vocabulary.AddTokens(specialTokens.GetAllSpecialTokens());

            vocabulary.AddToken("Bar");

            // Time shift tokens
            for (int d = 1; d <= 128; d++)
            {
                vocabulary.AddToken($"TimeShift_{d}");
                vocabulary.AddToken($"Rest_{d}");
            }

            // Compound note tokens: Note_Pitch_VelocityBin_Duration
            // This creates a large vocabulary but is more efficient for sequence modeling
            for (int pitch = 0; pitch < 128; pitch++)
            {
                for (int v = 0; v < numVelocityBins; v++)
                {
                    // Only create tokens for common durations (1-16) to keep vocabulary manageable
                    for (int d = 1; d <= 16; d++)
                    {
                        vocabulary.AddToken($"Note_{pitch}_{v}_{d}");
                    }
                }
            }

            return vocabulary;
        }

        /// <summary>
        /// Creates the vocabulary for SimpleNote tokenization strategy.
        /// Includes tokens for Pitch (0-127), Duration (1-128), and Rest (1-128).
        /// This is the smallest vocabulary option for basic melody representation.
        /// </summary>
        /// <param name="specialTokens">The special tokens to include in the vocabulary.</param>
        /// <returns>A vocabulary configured for SimpleNote tokenization.</returns>
        private static Vocabulary.Vocabulary CreateSimpleNoteVocabulary(SpecialTokens specialTokens)
        {
            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);
            vocabulary.AddTokens(specialTokens.GetAllSpecialTokens());

            for (int pitch = 0; pitch < 128; pitch++)
                vocabulary.AddToken($"Pitch_{pitch}");

            for (int d = 1; d <= 128; d++)
            {
                vocabulary.AddToken($"Duration_{d}");
                vocabulary.AddToken($"Rest_{d}");
            }

            return vocabulary;
        }
    }
}
