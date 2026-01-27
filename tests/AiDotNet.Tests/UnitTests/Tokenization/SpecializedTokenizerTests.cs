using System.Collections.Generic;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Specialized;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Tokenization;

/// <summary>
/// Unit tests for PhonemeTokenizer.
/// </summary>
public class PhonemeTokenizerTests
{
    [Fact]
    public void CreateARPAbet_CreatesValidTokenizer()
    {
        // Act
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        // Assert
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void Tokenize_SimpleWord_ReturnsPhonemes()
    {
        // Arrange
        var tokenizer = PhonemeTokenizer.CreateARPAbet();
        var text = "hello";

        // Act
        var tokens = tokenizer.Tokenize(text);

        // Assert
        Assert.NotEmpty(tokens);
    }

    [Fact]
    public void Tokenize_EmptyText_ReturnsEmpty()
    {
        // Arrange
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        // Act
        var tokens = tokenizer.Tokenize("");

        // Assert
        Assert.Empty(tokens);
    }

    [Fact]
    public void Tokenize_MultipleWords_AddsSeparators()
    {
        // Arrange
        var tokenizer = PhonemeTokenizer.CreateARPAbet();
        var text = "hello world";

        // Act
        var tokens = tokenizer.Tokenize(text);

        // Assert
        Assert.NotEmpty(tokens);
        Assert.Contains("<space>", tokens);
    }

    [Fact]
    public void Encode_ReturnsValidTokenIds()
    {
        // Arrange
        var tokenizer = PhonemeTokenizer.CreateARPAbet();
        var text = "test";

        // Act
        var result = tokenizer.Encode(text);

        // Assert
        Assert.NotEmpty(result.TokenIds);
    }

    [Fact]
    public void Vocabulary_ContainsARPAbetPhonemes()
    {
        // Arrange
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        // Assert
        Assert.True(tokenizer.Vocabulary.ContainsToken("AA"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("AE"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("B"));
    }

    [Fact]
    public void Decode_ReturnsPhonemeString()
    {
        // Arrange
        var tokenizer = PhonemeTokenizer.CreateARPAbet();
        var text = "hello";
        var encoded = tokenizer.Encode(text);

        // Act
        var decoded = tokenizer.Decode(encoded.TokenIds);

        // Assert
        Assert.NotEmpty(decoded);
    }

    [Fact]
    public void Tokenize_Digraph_UsesCorrectPhoneme()
    {
        // Arrange
        var tokenizer = PhonemeTokenizer.CreateARPAbet();
        var text = "the"; // Contains 'th' digraph

        // Act
        var tokens = tokenizer.Tokenize(text);

        // Assert
        Assert.NotEmpty(tokens);
        // 'th' should map to "TH" phoneme
        Assert.Contains("TH", tokens);
    }
}

/// <summary>
/// Unit tests for MidiTokenizer.
/// </summary>
public class MidiTokenizerTests
{
    [Fact]
    public void CreateREMI_CreatesValidTokenizer()
    {
        // Act
        var tokenizer = MidiTokenizer.CreateREMI();

        // Assert
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void Tokenize_NoteEvent_ReturnsTokens()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateREMI();
        var midiText = "NOTE:60:120:100";

        // Act
        var tokens = tokenizer.Tokenize(midiText);

        // Assert
        Assert.NotEmpty(tokens);
    }

    [Fact]
    public void Tokenize_EmptyText_ReturnsEmpty()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateREMI();

        // Act
        var tokens = tokenizer.Tokenize("");

        // Assert
        Assert.Empty(tokens);
    }

    [Fact]
    public void Tokenize_RestEvent_ReturnsTimeShift()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateREMI();
        var midiText = "REST:480";

        // Act
        var tokens = tokenizer.Tokenize(midiText);

        // Assert
        Assert.NotEmpty(tokens);
        Assert.Contains(tokens, t => t.StartsWith("TimeShift_"));
    }

    [Fact]
    public void Tokenize_BarEvent_ReturnsBarToken()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateREMI();
        var midiText = "BAR";

        // Act
        var tokens = tokenizer.Tokenize(midiText);

        // Assert
        Assert.Single(tokens);
        Assert.Equal("Bar", tokens[0]);
    }

    [Fact]
    public void TokenizeNotes_SingleNote_ReturnsTokens()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateREMI();
        var notes = new List<MidiTokenizer.MidiNote>
        {
            new MidiTokenizer.MidiNote { Pitch = 60, Velocity = 100, StartTick = 0, Duration = 480 }
        };

        // Act
        var tokens = tokenizer.TokenizeNotes(notes);

        // Assert
        Assert.NotEmpty(tokens);
        Assert.Contains(tokens, t => t.StartsWith("Pitch_"));
        Assert.Contains(tokens, t => t.StartsWith("Velocity_"));
        Assert.Contains(tokens, t => t.StartsWith("Duration_"));
    }

    [Fact]
    public void TokenizeNotes_EmptyList_ReturnsEmpty()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateREMI();
        var notes = new List<MidiTokenizer.MidiNote>();

        // Act
        var tokens = tokenizer.TokenizeNotes(notes);

        // Assert
        Assert.Empty(tokens);
    }

    [Fact]
    public void TokenizeNotes_MultipleNotes_IncludesPosition()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateREMI();
        var notes = new List<MidiTokenizer.MidiNote>
        {
            new MidiTokenizer.MidiNote { Pitch = 60, Velocity = 100, StartTick = 0, Duration = 480 },
            new MidiTokenizer.MidiNote { Pitch = 64, Velocity = 90, StartTick = 480, Duration = 480 }
        };

        // Act
        var tokens = tokenizer.TokenizeNotes(notes);

        // Assert
        Assert.NotEmpty(tokens);
        Assert.Contains(tokens, t => t.StartsWith("Position_"));
    }

    [Fact]
    public void TokenizeNotes_CrossingBarLine_IncludesBar()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateREMI();
        var ticksPerBar = 480 * 4; // 4 beats per bar
        var notes = new List<MidiTokenizer.MidiNote>
        {
            new MidiTokenizer.MidiNote { Pitch = 60, Velocity = 100, StartTick = ticksPerBar + 100, Duration = 480 }
        };

        // Act
        var tokens = tokenizer.TokenizeNotes(notes);

        // Assert
        Assert.Contains("Bar", tokens);
    }

    [Fact]
    public void Encode_ReturnsValidTokenIds()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateREMI();
        var midiText = "NOTE:60:120:100";

        // Act
        var result = tokenizer.Encode(midiText);

        // Assert
        Assert.NotEmpty(result.TokenIds);
    }

    [Fact]
    public void Vocabulary_ContainsPitchTokens()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateREMI();

        // Assert
        Assert.True(tokenizer.Vocabulary.ContainsToken("Pitch_60"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("Pitch_0"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("Pitch_127"));
    }

    [Fact]
    public void Vocabulary_ContainsVelocityTokens()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateREMI();

        // Assert
        Assert.True(tokenizer.Vocabulary.ContainsToken("Velocity_0"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("Velocity_16"));
    }

    // CPWord Strategy Tests

    [Fact]
    public void CreateCPWord_CreatesValidTokenizer()
    {
        // Act
        var tokenizer = MidiTokenizer.CreateCPWord();

        // Assert
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void CPWord_TokenizeNotes_SingleNote_ReturnsCompoundToken()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateCPWord();
        var notes = new List<MidiTokenizer.MidiNote>
        {
            new MidiTokenizer.MidiNote { Pitch = 60, Velocity = 100, StartTick = 0, Duration = 480 }
        };

        // Act
        var tokens = tokenizer.TokenizeNotes(notes);

        // Assert
        Assert.NotEmpty(tokens);
        // CPWord uses compound tokens: Note_Pitch_VelocityBin_Duration
        Assert.Contains(tokens, t => t.StartsWith("Note_60_"));
    }

    [Fact]
    public void CPWord_TokenizeNotes_MultipleNotes_IncludesTimeShift()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateCPWord();
        var notes = new List<MidiTokenizer.MidiNote>
        {
            new MidiTokenizer.MidiNote { Pitch = 60, Velocity = 100, StartTick = 0, Duration = 480 },
            new MidiTokenizer.MidiNote { Pitch = 64, Velocity = 90, StartTick = 960, Duration = 480 }
        };

        // Act
        var tokens = tokenizer.TokenizeNotes(notes);

        // Assert
        Assert.NotEmpty(tokens);
        Assert.Contains(tokens, t => t.StartsWith("TimeShift_"));
    }

    [Fact]
    public void CPWord_Vocabulary_ContainsCompoundTokens()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateCPWord();

        // Assert - CPWord uses compound tokens: Note_Pitch_VelocityBin_Duration
        Assert.True(tokenizer.Vocabulary.ContainsToken("Note_60_8_1"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("Note_0_0_1"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("Bar"));
    }

    [Fact]
    public void CPWord_Encode_ReturnsValidTokenIds()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateCPWord();
        var midiText = "NOTE:60:120:100";

        // Act
        var result = tokenizer.Encode(midiText);

        // Assert
        Assert.NotEmpty(result.TokenIds);
    }

    // SimpleNote Strategy Tests

    [Fact]
    public void CreateSimpleNote_CreatesValidTokenizer()
    {
        // Act
        var tokenizer = MidiTokenizer.CreateSimpleNote();

        // Assert
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void SimpleNote_TokenizeNotes_SingleNote_ReturnsPitchAndDuration()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateSimpleNote();
        var notes = new List<MidiTokenizer.MidiNote>
        {
            new MidiTokenizer.MidiNote { Pitch = 60, Velocity = 100, StartTick = 0, Duration = 480 }
        };

        // Act
        var tokens = tokenizer.TokenizeNotes(notes);

        // Assert
        Assert.NotEmpty(tokens);
        // SimpleNote uses separate Pitch and Duration tokens
        Assert.Contains(tokens, t => t.StartsWith("Pitch_"));
        Assert.Contains(tokens, t => t.StartsWith("Duration_"));
    }

    [Fact]
    public void SimpleNote_TokenizeNotes_MultipleNotes_IncludesRest()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateSimpleNote();
        var notes = new List<MidiTokenizer.MidiNote>
        {
            new MidiTokenizer.MidiNote { Pitch = 60, Velocity = 100, StartTick = 0, Duration = 480 },
            new MidiTokenizer.MidiNote { Pitch = 64, Velocity = 90, StartTick = 960, Duration = 480 }
        };

        // Act
        var tokens = tokenizer.TokenizeNotes(notes);

        // Assert
        Assert.NotEmpty(tokens);
        // SimpleNote uses Rest_ tokens to represent gaps between notes
        Assert.Contains(tokens, t => t.StartsWith("Rest_"));
    }

    [Fact]
    public void SimpleNote_Vocabulary_ContainsPitchTokens()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateSimpleNote();

        // Assert
        Assert.True(tokenizer.Vocabulary.ContainsToken("Pitch_60"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("Pitch_0"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("Pitch_127"));
    }

    [Fact]
    public void SimpleNote_Vocabulary_ContainsDurationTokens()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateSimpleNote();

        // Assert
        Assert.True(tokenizer.Vocabulary.ContainsToken("Duration_1"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("Duration_16"));
    }

    [Fact]
    public void SimpleNote_Encode_ReturnsValidTokenIds()
    {
        // Arrange
        var tokenizer = MidiTokenizer.CreateSimpleNote();
        var midiText = "NOTE:60:120:100";

        // Act
        var result = tokenizer.Encode(midiText);

        // Assert
        Assert.NotEmpty(result.TokenIds);
    }

    #region PR #757 Bug Fix Tests - Parameter Validation

    [Fact]
    public void CreateREMI_InvalidTicksPerBeat_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            MidiTokenizer.CreateREMI(ticksPerBeat: 0));
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            MidiTokenizer.CreateREMI(ticksPerBeat: -1));
    }

    [Fact]
    public void CreateREMI_InvalidNumVelocityBins_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            MidiTokenizer.CreateREMI(numVelocityBins: 0));
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            MidiTokenizer.CreateREMI(numVelocityBins: -1));
    }

    [Fact]
    public void CreateCPWord_InvalidTicksPerBeat_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            MidiTokenizer.CreateCPWord(ticksPerBeat: 0));
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            MidiTokenizer.CreateCPWord(ticksPerBeat: -1));
    }

    [Fact]
    public void CreateCPWord_InvalidNumVelocityBins_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            MidiTokenizer.CreateCPWord(numVelocityBins: 0));
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            MidiTokenizer.CreateCPWord(numVelocityBins: -1));
    }

    [Fact]
    public void CreateSimpleNote_InvalidTicksPerBeat_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            MidiTokenizer.CreateSimpleNote(ticksPerBeat: 0));
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            MidiTokenizer.CreateSimpleNote(ticksPerBeat: -1));
    }

    #endregion
}

/// <summary>
/// Unit tests for SentencePiece tokenizer.
/// </summary>
public class SentencePieceTokenizerTests
{
    private readonly SentencePieceTokenizer _tokenizer;

    public SentencePieceTokenizerTests()
    {
        var corpus = new List<string>
        {
            "Hello world",
            "Machine learning",
            "Natural language processing"
        };

        _tokenizer = SentencePieceTokenizer.Train(corpus, 500);
    }

    [Fact]
    public void Train_CreatesVocabulary()
    {
        // Assert
        Assert.True(_tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void Tokenize_ReturnsTokens()
    {
        // Arrange
        var text = "Hello world";

        // Act
        var tokens = _tokenizer.Tokenize(text);

        // Assert
        Assert.NotEmpty(tokens);
    }

    [Fact]
    public void Tokenize_UsesSentencePieceMarker()
    {
        // Arrange
        var text = "Hello world";

        // Act
        var tokens = _tokenizer.Tokenize(text);
        var joinedTokens = string.Join("", tokens);

        // Assert
        // SentencePiece uses \u2581 marker for spaces
        Assert.Contains("\u2581", joinedTokens);
    }

    [Fact]
    public void Decode_RemovesMarker()
    {
        // Arrange
        var text = "test";
        var encoded = _tokenizer.Encode(text);

        // Act
        var decoded = _tokenizer.Decode(encoded.TokenIds);

        // Assert
        Assert.DoesNotContain("\u2581", decoded);
    }

    [Fact]
    public void Roundtrip_PreservesContent()
    {
        // Arrange - Use text from training corpus to ensure tokens are in vocabulary
        var text = "Machine learning";
        var options = new EncodingOptions { AddSpecialTokens = false };
        var encoded = _tokenizer.Encode(text, options);

        // Act
        var decoded = _tokenizer.Decode(encoded.TokenIds);

        // Assert - Check for content that should be preserved
        Assert.Contains("learning", decoded.ToLowerInvariant());
    }

    [Fact]
    public void Encode_ReturnsValidResult()
    {
        // Arrange
        var text = "Hello world";

        // Act
        var result = _tokenizer.Encode(text);

        // Assert
        Assert.NotEmpty(result.TokenIds);
        Assert.Equal(result.Tokens.Count, result.TokenIds.Count);
    }
}
