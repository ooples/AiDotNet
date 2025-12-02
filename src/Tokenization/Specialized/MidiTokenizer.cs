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
    /// </summary>
    /// <remarks>
    /// Currently only the REMI (Revamped MIDI) tokenization strategy is implemented.
    /// CPWord and SimpleNote strategies are planned for future releases.
    /// </remarks>
    public class MidiTokenizer : TokenizerBase
    {
        private readonly TokenizationStrategy _strategy;
        private readonly int _ticksPerBeat;
        private readonly int _numVelocityBins;

        /// <summary>
        /// MIDI tokenization strategies.
        /// </summary>
        /// <remarks>
        /// Currently only REMI is implemented. Using other strategies will throw NotImplementedException.
        /// </remarks>
        public enum TokenizationStrategy { REMI, CPWord, SimpleNote }

        /// <summary>
        /// Represents a MIDI note event.
        /// </summary>
        public class MidiNote
        {
            public int StartTick { get; set; }
            public int Duration { get; set; }
            public int Pitch { get; set; }
            public int Velocity { get; set; }
        }

        /// <summary>
        /// Creates a new MIDI tokenizer.
        /// </summary>
        /// <exception cref="NotImplementedException">
        /// Thrown when a strategy other than REMI is specified.
        /// </exception>
        public MidiTokenizer(
            IVocabulary vocabulary,
            SpecialTokens specialTokens,
            TokenizationStrategy strategy = TokenizationStrategy.REMI,
            int ticksPerBeat = 480,
            int numVelocityBins = 32)
            : base(vocabulary, specialTokens)
        {
            if (strategy != TokenizationStrategy.REMI)
            {
                throw new NotImplementedException(
                    $"TokenizationStrategy.{strategy} is not yet implemented. " +
                    "Currently only REMI strategy is supported.");
            }

            _strategy = strategy;
            _ticksPerBeat = ticksPerBeat;
            _numVelocityBins = numVelocityBins;
        }

        /// <summary>
        /// Tokenizes text representation of MIDI.
        /// Format: "NOTE:pitch:duration:velocity" or "REST:duration"
        /// </summary>
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
                        tokens.Add($"Pitch_{pitch}");
                        // Clamp velocity bin to valid range [0, _numVelocityBins-1]
                        int velocityBin = Math.Min((velocity * _numVelocityBins) / 128, _numVelocityBins - 1);
                        tokens.Add($"Velocity_{velocityBin}");
                        tokens.Add($"Duration_{QuantizeDuration(duration)}");
                    }
                }
                else if (eventType == "REST" && parts.Length >= 2)
                {
                    if (int.TryParse(parts[1], out int duration))
                        tokens.Add($"TimeShift_{QuantizeDuration(duration)}");
                }
                else if (eventType == "BAR")
                {
                    tokens.Add("Bar");
                }
            }

            return tokens;
        }

        /// <summary>
        /// Tokenizes a list of MIDI notes.
        /// </summary>
        public List<string> TokenizeNotes(IEnumerable<MidiNote> notes)
        {
            var noteList = notes.OrderBy(n => n.StartTick).ToList();
            var tokens = new List<string>();
            int currentTick = 0;
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
                // Clamp velocity bin to valid range [0, _numVelocityBins-1]
                int velocityBin = Math.Min((note.Velocity * _numVelocityBins) / 128, _numVelocityBins - 1);
                tokens.Add($"Velocity_{velocityBin}");
                tokens.Add($"Duration_{QuantizeDuration(note.Duration)}");

                currentTick = note.StartTick + note.Duration;
            }

            return tokens;
        }

        private int QuantizeDuration(int ticks)
        {
            int quantumSize = _ticksPerBeat / 4;
            // Clamp to valid range [1, 128] - vocabulary has Duration_1 through Duration_128
            int quantized = (ticks + quantumSize / 2) / quantumSize;
            return Math.Max(1, Math.Min(quantized, 128));
        }

        protected override string CleanupTokens(List<string> tokens)
        {
            return string.Join(" ", tokens);
        }

        /// <summary>
        /// Creates a MIDI tokenizer with REMI strategy.
        /// </summary>
        public static MidiTokenizer CreateREMI(SpecialTokens? specialTokens = null, int ticksPerBeat = 480, int numVelocityBins = 32)
        {
            specialTokens ??= new SpecialTokens { UnkToken = "<unk>", PadToken = "<pad>", BosToken = "<bos>", EosToken = "<eos>" };

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

            return new MidiTokenizer(vocabulary, specialTokens, TokenizationStrategy.REMI, ticksPerBeat, numVelocityBins);
        }
    }
}
