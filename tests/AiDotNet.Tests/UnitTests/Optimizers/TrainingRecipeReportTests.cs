using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Tasks.Graph;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Tests that the training-recipe report tells the truth about what a model trains with (#1928).
/// </summary>
/// <remarks>
/// The report's whole value is that a deviation from a paper is never silent. A report that always
/// said "Exact" would be worse than no report, so these assert the states that carry bad news --
/// Adapted, Deviated and NotDeclared -- rather than only the happy path.
/// </remarks>
public class TrainingRecipeReportTests
{
    [Fact]
    public void FidelityIsNotDeclared_WhenAModelDeclaresNoRecipe()
    {
        // "Not declared" must be distinguishable from "matches the paper". Most of the catalogue is
        // undeclared, and reporting that honestly is the point of counting it.
        var report = TrainingRecipeReport.NotDeclaredFor();

        Assert.Equal(RecipeFidelity.NotDeclared, report.Fidelity);
        Assert.Contains("No paper training recipe declared", report.Describe());
    }

    [Fact]
    public void FidelityIsExact_WhenNothingWasAdaptedOrDropped()
    {
        var report = new TrainingRecipeReport
        {
            PaperOptimizer = OptimizerKind.Adam,
            AppliedOptimizer = "AdamOptimizer`3",
            Source = "Kipf and Welling 2017, Sec. 5.2",
        };

        Assert.Equal(RecipeFidelity.Exact, report.Fidelity);
    }

    [Fact]
    public void FidelityIsAdapted_AndTheRuleIsNamed()
    {
        // An adaptation without a stated justification is indistinguishable from an arbitrary
        // change, so the rule is part of the record rather than an optional note.
        var report = new TrainingRecipeReport
        {
            PaperOptimizer = OptimizerKind.RmsProp,
            AppliedOptimizer = "RootMeanSquarePropagationOptimizer`3",
            Source = "Howard et al. 2019, Sec. 6.1.1",
            Adaptations =
            [
                new RecipeAdaptation("LearningRate", "0.1 at batch 4096", "0.00078125 at batch 32",
                                     "linear scaling rule, Goyal et al. 2017"),
            ],
        };

        Assert.Equal(RecipeFidelity.Adapted, report.Fidelity);
        Assert.Contains("Goyal", report.Describe());
        Assert.Contains("0.1 at batch 4096", report.Describe());
    }

    [Fact]
    public void FidelityIsDeviated_WhenSomethingCouldNotBeHonoured()
    {
        // Deviated outranks Adapted: a recipe that was partly dropped is not merely adjusted, and
        // collapsing the two would hide the case a user most needs to see.
        var report = new TrainingRecipeReport
        {
            PaperOptimizer = OptimizerKind.Adam,
            AppliedOptimizer = "AdamOptimizer`3",
            Source = "some paper",
            Adaptations = [new RecipeAdaptation("WarmupSteps", "4000", "10", "run-length scaling")],
            Unhonoured = ["the paper's Cyclic schedule is not mapped to a scheduler here"],
        };

        Assert.Equal(RecipeFidelity.Deviated, report.Fidelity);
        Assert.Contains("NOT honoured", report.Describe());
    }

    [Fact]
    public void ARealModelReportsTheRecipeItActuallyBuilt()
    {
        // End to end against a shipped model rather than a constructed report: NodeClassification
        // declares Adam at 0.01 (Kipf and Welling 2017), and its constructor builds through the
        // factory, so a report must exist by the time the model does.
        var model = new NodeClassificationModel<double>();

        var reports = PaperOptimizerFactory.ReportsFor(model);

        Assert.NotEmpty(reports);
        var report = reports[0];

        Assert.Equal(OptimizerKind.Adam, report.PaperOptimizer);
        Assert.Contains("AdamOptimizer", report.AppliedOptimizer);
        Assert.False(string.IsNullOrWhiteSpace(report.Source));
        Assert.Contains("Kipf", report.Source);

        // Nothing about this model's recipe needs adapting at fixture scale, so it should be exact.
        // If this ever flips to Adapted, the report will say which rule fired and why.
        Assert.Equal(RecipeFidelity.Exact, report.Fidelity);
    }

    [Fact]
    public void AComponentDeclarationIsSelectedOverTheModelWideOne()
    {
        // Composite models state different settings per part -- Stable Audio Open gives separate
        // rates for its autoencoder, discriminators and DiT. The unnamed declaration is the shared
        // default; a named one overrides it for that component only.
        var shared = PaperOptimizerFactory.Find(new Composite(), component: "");
        var discriminator = PaperOptimizerFactory.Find(new Composite(), component: "discriminator");

        Assert.NotNull(shared);
        Assert.NotNull(discriminator);
        Assert.Equal(1.5e-4, shared!.LearningRate, precision: 12);
        Assert.Equal(3e-4, discriminator!.LearningRate, precision: 12);
    }

    [Fact]
    public void AComponentWithNoDeclarationFallsBackToTheModelWideRecipe()
    {
        // Partial population is the expected state while a composite model is being filled in.
        var unnamed = PaperOptimizerFactory.Find(new Composite(), component: "vocoder");

        Assert.NotNull(unnamed);
        Assert.Equal(1.5e-4, unnamed!.LearningRate, precision: 12);
    }

    [PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1.5e-4, Source = "fixture: shared default")]
    [PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-4, Component = "discriminator",
                    Source = "fixture: discriminator row")]
    private sealed class Composite { }
}
