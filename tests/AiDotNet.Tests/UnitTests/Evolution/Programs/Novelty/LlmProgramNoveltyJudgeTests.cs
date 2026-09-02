using AiDotNet.Enums;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.Novelty;
using AiDotNet.Evolution.Prompts;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs.Novelty;

public sealed class LlmProgramNoveltyJudgeTests
{
    [Theory]
    [InlineData("NOT_NOVEL", ProgramNoveltyVerdict.NotNovel)]
    [InlineData("NOT NOVEL", ProgramNoveltyVerdict.NotNovel)]
    [InlineData("NOT-NOVEL", ProgramNoveltyVerdict.NotNovel)]
    [InlineData("not_novel - only the variable names changed.", ProgramNoveltyVerdict.NotNovel)]
    [InlineData("NOVEL", ProgramNoveltyVerdict.Novel)]
    [InlineData("NOVEL: the proposal replaces the sort with a heap.", ProgramNoveltyVerdict.Novel)]
    [InlineData("novel, because the recurrence is different", ProgramNoveltyVerdict.Novel)]
    [InlineData("I am not able to decide.", ProgramNoveltyVerdict.Unavailable)]
    [InlineData("", ProgramNoveltyVerdict.Unavailable)]
    public void VerdictParsingHandlesEverySpellingOfTheAnswer(string answer, ProgramNoveltyVerdict expected) =>
        Assert.Equal(expected, LlmProgramNoveltyJudge<double>.ParseVerdict(answer));

    [Fact]
    public void TheCanonicalNegativeAnswerIsNotReadAsItsOwnOpposite()
    {
        // The reference implementation instructs the model to answer NOT_NOVEL and then searches for "NOT NOVEL"
        // with a space, which never matches; the underscored form contains NOVEL at offset four, so upstream reads
        // its own canonical rejection as an acceptance. This asserts the corrected behaviour directly.
        Assert.Equal(ProgramNoveltyVerdict.NotNovel, LlmProgramNoveltyJudge<double>.ParseVerdict("NOT_NOVEL"));
        Assert.Equal(ProgramNoveltyVerdict.NotNovel,
            LlmProgramNoveltyJudge<double>.ParseVerdict("NOT_NOVEL. The change is a rename only."));
    }

    [Fact]
    public void TheEarliestVerdictTokenWinsWhenBothAppear()
    {
        Assert.Equal(ProgramNoveltyVerdict.Novel,
            LlmProgramNoveltyJudge<double>.ParseVerdict("NOVEL. It is not novel only in the trivial sense."));
        Assert.Equal(ProgramNoveltyVerdict.NotNovel,
            LlmProgramNoveltyJudge<double>.ParseVerdict("NOT_NOVEL, although a novel comment was added."));
    }

    [Fact]
    public async Task JudgeReturnsTheParsedVerdictAndCountsItsRequests()
    {
        var client = new FakeChatClient("NOT_NOVEL - identical apart from names.");
        var judge = new LlmProgramNoveltyJudge<double>(client);

        ProgramNoveltyVerdict verdict = await judge.JudgeAsync(
            new ProgramGenome("def f(a): return a"),
            new ProgramGenome("def f(b): return b"));

        Assert.Equal(ProgramNoveltyVerdict.NotNovel, verdict);
        Assert.Equal(1L, judge.Judgements);
        Assert.Equal(0L, judge.UnavailableAnswers);
        Assert.Equal(1, client.Calls);
    }

    [Fact]
    public async Task AProviderFailureIsReportedAsUnavailableRatherThanAsNovel()
    {
        var client = new FakeChatClient("NOVEL")
        {
            ThrowOnFirstCall = new InvalidOperationException("provider exploded with key sk-abcdefghijklmnop1234")
        };
        var judge = new LlmProgramNoveltyJudge<double>(client);

        ProgramNoveltyVerdict verdict = await judge.JudgeAsync(
            new ProgramGenome("a"), new ProgramGenome("b"));

        // Upstream turns any exception into "novel"; here the caller is told the judge had nothing to say.
        Assert.Equal(ProgramNoveltyVerdict.Unavailable, verdict);
        Assert.Equal(1L, judge.UnavailableAnswers);
    }

    [Fact]
    public async Task AnUnparseableAnswerIsUnavailableAndIsCounted()
    {
        var client = new FakeChatClient("The two snippets are broadly comparable.");
        var judge = new LlmProgramNoveltyJudge<double>(client);

        Assert.Equal(ProgramNoveltyVerdict.Unavailable,
            await judge.JudgeAsync(new ProgramGenome("a"), new ProgramGenome("b")));
        Assert.Equal(1L, judge.UnavailableAnswers);
    }

    [Fact]
    public async Task ProgramTextReachesThePromptOnlyAfterRedactionAndBounding()
    {
        var client = new FakeChatClient("NOVEL");
        var judge = new LlmProgramNoveltyJudge<double>(client, maxProgramBytes: 512);
        var candidate = new ProgramGenome(
            "api_key = sk-livesecretvalue0123456789abcdef\n" + new string('z', 4_000));
        var incumbent = new ProgramGenome("password = hunter2hunter2hunter2\nprint(1)");

        await judge.JudgeAsync(candidate, incumbent);

        string prompt = string.Join("\n", client.Conversations[0].Select(message => message.Text));
        Assert.DoesNotContain("sk-livesecretvalue0123456789abcdef", prompt, StringComparison.Ordinal);
        Assert.DoesNotContain("hunter2hunter2hunter2", prompt, StringComparison.Ordinal);
        Assert.Contains(PromptTextRedactor.RedactionMarker, prompt, StringComparison.Ordinal);

        // Bounded: 512 bytes of each program plus the fixed scaffolding, nowhere near the 4,000-character source.
        Assert.True(prompt.Length < 3_000);
    }

    [Fact]
    public void JudgeValidatesItsConfiguration()
    {
        var client = new FakeChatClient("NOVEL");
#pragma warning disable CS8625
        Assert.Throws<ArgumentNullException>(() => new LlmProgramNoveltyJudge<double>(null));
#pragma warning restore CS8625
        Assert.Throws<ArgumentException>(() => new LlmProgramNoveltyJudge<double>(client, id: "  "));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LlmProgramNoveltyJudge<double>(client, maxProgramBytes: 8));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LlmProgramNoveltyJudge<double>(client, maxOutputTokens: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LlmProgramNoveltyJudge<double>(client, temperature: 3.0));
    }
}
