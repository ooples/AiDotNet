using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Evolution.Prompts;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramPromptTemplateTests
{
    [Fact]
    public void PlaceholdersAreDiscoveredInFirstAppearanceOrderWithoutDuplicates()
    {
        var template = new ProgramPromptTemplate("a {one} b {two} c {one} d");
        Assert.Equal(new[] { "one", "two" }, template.Placeholders);
        Assert.False(template.IsConstant);
        Assert.True(template.ContainsPlaceholder("two"));
        Assert.False(template.ContainsPlaceholder("three"));
    }

    [Fact]
    public void DoubledBracesRenderAsLiteralBraces()
    {
        var template = new ProgramPromptTemplate("{{\"score\": {value}}}");
        Assert.Equal(new[] { "value" }, template.Placeholders);
        string rendered = template.Render(new Dictionary<string, string>(StringComparer.Ordinal) { ["value"] = "0.5" });
        Assert.Equal("{\"score\": 0.5}", rendered);
    }

    [Fact]
    public void ConstantTemplateHasNoPlaceholders()
    {
        var template = new ProgramPromptTemplate("no slots here");
        Assert.True(template.IsConstant);
        Assert.Empty(template.Placeholders);
        Assert.Equal("no slots here", template.Render(new Dictionary<string, string>(StringComparer.Ordinal)));
    }

    [Theory]
    [InlineData("unclosed {name")]
    [InlineData("stray } brace")]
    [InlineData("empty {} slot")]
    [InlineData("bad {na-me} slot")]
    public void MalformedTemplatesAreRejectedAtConstruction(string text)
    {
        // Upstream discovers the same class of problem as a KeyError from inside a
        // formatting call, part-way through a paid run.
        Assert.Throws<ArgumentException>(() => new ProgramPromptTemplate(text));
    }

    [Fact]
    public void RenderingReportsAMissingValueByName()
    {
        var template = new ProgramPromptTemplate("hello {who}");
        KeyNotFoundException error = Assert.Throws<KeyNotFoundException>(
            () => template.Render(new Dictionary<string, string>(StringComparer.Ordinal)));
        Assert.Contains("who", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderingIsPureAndRepeatable()
    {
        var template = new ProgramPromptTemplate("{a}-{b}-{a}");
        var values = new Dictionary<string, string>(StringComparer.Ordinal) { ["a"] = "1", ["b"] = "2" };
        Assert.Equal("1-2-1", template.Render(values));
        Assert.Equal("1-2-1", template.Render(values));
    }

    [Fact]
    public void LineEndingsAreNormalizedSoTheSameTemplateHashesTheSameOnEveryPlatform()
    {
        // A template file checked out with Windows line endings and the same file
        // checked out with Unix ones must not produce different prompts or a
        // different resume identity.
        var windows = new ProgramPromptTemplate("first {a}\r\nsecond\r\nthird");
        var unix = new ProgramPromptTemplate("first {a}\nsecond\nthird");
        var oldMac = new ProgramPromptTemplate("first {a}\rsecond\rthird");

        Assert.Equal(unix.Text, windows.Text);
        Assert.Equal(unix.Text, oldMac.Text);
        Assert.DoesNotContain("\r", windows.Text, StringComparison.Ordinal);

        Assert.Equal(
            ProgramPromptTemplateSet.CreateDefault().VersionHash,
            ProgramPromptTemplateSet.CreateDefault()
                .With(ProgramPromptTemplateKey.SystemMessage,
                    ProgramPromptTemplateSet.CreateDefault()
                        .GetTemplate(ProgramPromptTemplateKey.SystemMessage).Text.Replace("\n", "\r\n"))
                .VersionHash);
    }

    [Fact]
    public void ToStringNeverEchoesTemplateText()
    {
        var template = new ProgramPromptTemplate("secret wording {x}");
        Assert.DoesNotContain("secret", template.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void EveryShippedTemplateAndFragmentParsesAndDeclaresOnlyKnownNames()
    {
        ProgramPromptTemplateSet set = ProgramPromptTemplateSet.CreateDefault();

        foreach (ProgramPromptTemplateKey key in ProgramPromptTemplateSet.TemplateKeys)
        {
            ProgramPromptTemplate template = set.GetTemplate(key);
            IReadOnlyList<string> supplied = ProgramPromptTemplateSet.SuppliedPlaceholders(key);
            foreach (string placeholder in template.Placeholders)
            {
                Assert.Contains(placeholder, supplied);
            }

            foreach (string required in ProgramPromptTemplateSet.RequiredPlaceholders(key))
            {
                Assert.True(template.ContainsPlaceholder(required),
                    $"Shipped template '{key}' must keep '{{{required}}}'.");
            }
        }

        foreach (ProgramPromptFragmentKey key in ProgramPromptTemplateSet.FragmentKeys)
        {
            ProgramPromptTemplate fragment = set.GetFragment(key);
            IReadOnlyList<string> declared = ProgramPromptTemplateSet.DeclaredFragmentArguments(key);
            Assert.Equal(declared.Count, fragment.Placeholders.Count);
            foreach (string name in declared) Assert.True(fragment.ContainsPlaceholder(name));
        }
    }

    [Fact]
    public void ShippedFragmentUsesTheChangesArgumentUpstreamDrops()
    {
        // OpenEvolve formats inspiration_changes_prefix with a {changes} argument
        // that its text does not contain, so the description is silently lost.
        ProgramPromptTemplateSet set = ProgramPromptTemplateSet.CreateDefault();
        Assert.True(set.GetFragment(ProgramPromptFragmentKey.InspirationChangesPrefix).ContainsPlaceholder("changes"));

        string rendered = set.RenderFragment(
            ProgramPromptFragmentKey.InspirationChangesPrefix,
            new Dictionary<string, string>(StringComparer.Ordinal) { ["changes"] = "switched to a sieve" });
        Assert.Contains("switched to a sieve", rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void FragmentOverrideThatDropsItsArgumentIsRejected()
    {
        ProgramPromptTemplateSet set = ProgramPromptTemplateSet.CreateDefault();
        Assert.Throws<ArgumentException>(
            () => set.WithFragment(ProgramPromptFragmentKey.InspirationChangesPrefix, "Modification made."));
    }

    [Fact]
    public void FragmentOverrideThatAsksForAnUnknownArgumentIsRejected()
    {
        ProgramPromptTemplateSet set = ProgramPromptTemplateSet.CreateDefault();
        Assert.Throws<ArgumentException>(
            () => set.WithFragment(ProgramPromptFragmentKey.ArtifactTitle, "Output of {run_id}"));
    }

    [Fact]
    public void TemplateOverrideThatDropsARequiredPlaceholderIsRejected()
    {
        ProgramPromptTemplateSet set = ProgramPromptTemplateSet.CreateDefault();
        Assert.Throws<ArgumentException>(
            () => set.With(ProgramPromptTemplateKey.EvolutionHistory, "## History\n\n{previous_attempts}"));
    }

    [Fact]
    public void OverridesProduceANewSetAndANewVersionHash()
    {
        ProgramPromptTemplateSet original = ProgramPromptTemplateSet.CreateDefault();
        ProgramPromptTemplateSet changed = original.With(
            ProgramPromptTemplateKey.SystemMessage, "You are terse.");

        Assert.NotEqual(original.VersionHash, changed.VersionHash);
        Assert.Equal("You are terse.", changed.GetTemplate(ProgramPromptTemplateKey.SystemMessage).Text);
        Assert.NotEqual("You are terse.", original.GetTemplate(ProgramPromptTemplateKey.SystemMessage).Text);
        Assert.Equal(original.VersionHash, ProgramPromptTemplateSet.CreateDefault().VersionHash);
    }

    [Fact]
    public void DirectoryOverridesAreReadAsUtf8RegardlessOfTheMachineCodePage()
    {
        // Upstream opens template files with Python's platform-default encoding,
        // which turns every non-ASCII character of a UTF-8 template into mojibake
        // on a Windows machine.
        string directory = CreateTemporaryDirectory();
        try
        {
            const string Text = "Vous êtes un développeur expert — visez la précision.";
            File.WriteAllText(
                Path.Combine(directory, "system_message.txt"),
                Text,
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            ProgramPromptTemplateSet set = ProgramPromptTemplateSet.LoadFromDirectory(directory);
            Assert.Equal(Text, set.GetTemplate(ProgramPromptTemplateKey.SystemMessage).Text);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void DirectoryOverridesLeaveAbsentTemplatesAtTheirDefaults()
    {
        string directory = CreateTemporaryDirectory();
        try
        {
            File.WriteAllText(Path.Combine(directory, "system_message.txt"), "Replaced.");
            ProgramPromptTemplateSet defaults = ProgramPromptTemplateSet.CreateDefault();
            ProgramPromptTemplateSet loaded = ProgramPromptTemplateSet.LoadFromDirectory(directory);

            Assert.Equal("Replaced.", loaded.GetTemplate(ProgramPromptTemplateKey.SystemMessage).Text);
            Assert.Equal(
                defaults.GetTemplate(ProgramPromptTemplateKey.DiffUser).Text,
                loaded.GetTemplate(ProgramPromptTemplateKey.DiffUser).Text);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void FragmentsFileOverridesOnlyTheNamesItLists()
    {
        string directory = CreateTemporaryDirectory();
        try
        {
            File.WriteAllText(
                Path.Combine(directory, ProgramPromptTemplateSet.FragmentsFileName),
                "{\"artifact_title\": \"Run Output\", \"fitness_stable\": \"Held at {current}\"}",
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            ProgramPromptTemplateSet loaded = ProgramPromptTemplateSet.LoadFromDirectory(directory);
            Assert.Equal("Run Output", loaded.RenderFragment(ProgramPromptFragmentKey.ArtifactTitle));
            Assert.Equal(
                "Held at 0.5000",
                loaded.RenderFragment(
                    ProgramPromptFragmentKey.FitnessStable,
                    new Dictionary<string, string>(StringComparer.Ordinal) { ["current"] = "0.5000" }));
            Assert.Equal(
                ProgramPromptTemplateSet.CreateDefault().RenderFragment(ProgramPromptFragmentKey.AttemptMixedMetrics),
                loaded.RenderFragment(ProgramPromptFragmentKey.AttemptMixedMetrics));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void FragmentsFileWithABadValueIsRejected()
    {
        string directory = CreateTemporaryDirectory();
        try
        {
            File.WriteAllText(
                Path.Combine(directory, ProgramPromptTemplateSet.FragmentsFileName),
                "{\"artifact_title\": 12}");
            Assert.Throws<InvalidDataException>(() => ProgramPromptTemplateSet.LoadFromDirectory(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void MissingTemplateDirectoryIsAnErrorRatherThanASilentFallback()
    {
        string missing = Path.Combine(Path.GetTempPath(), "aidotnet-prompt-missing-" + Guid.NewGuid().ToString("N"));
        Assert.Throws<DirectoryNotFoundException>(() => ProgramPromptTemplateSet.LoadFromDirectory(missing));
    }

    [Fact]
    public void EveryTemplateKeyHasADistinctFileStem()
    {
        var stems = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (ProgramPromptTemplateKey key in ProgramPromptTemplateSet.TemplateKeys)
        {
            Assert.True(stems.Add(ProgramPromptTemplateSet.TemplateFileStem(key)), $"Duplicate stem for '{key}'.");
        }

        Assert.Equal(13, stems.Count);
    }

    [Fact]
    public void EveryFragmentKeyHasADistinctName()
    {
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (ProgramPromptFragmentKey key in ProgramPromptTemplateSet.FragmentKeys)
        {
            Assert.True(names.Add(ProgramPromptTemplateSet.FragmentName(key)), $"Duplicate name for '{key}'.");
        }

        Assert.Equal(30, names.Count);
    }

    private static string CreateTemporaryDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "aidotnet-prompt-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }
}
