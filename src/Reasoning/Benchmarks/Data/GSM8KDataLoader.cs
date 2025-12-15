using Newtonsoft.Json.Linq;

namespace AiDotNet.Reasoning.Benchmarks.Data;

/// <summary>
/// Loader for GSM8K benchmark dataset.
/// </summary>
/// <remarks>
/// GSM8K (Grade School Math 8K) dataset format:
/// {
///   "question": "Natalia sold clips to 48 of her friends in April...",
///   "answer": "She sells 48/2 = 24 clips in May.\nShe sells 24/2 = 12... #### 12"
/// }
/// </remarks>
public class GSM8KDataLoader
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    public static Task<List<GSM8KProblem>> LoadFromFileAsync(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"GSM8K dataset not found: {filePath}");
        }

        var problems = new List<GSM8KProblem>();
        var lines = File.ReadAllLines(filePath); // net462 compatible

        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            try
            {
                var json = JObject.Parse(line); // Use Newtonsoft.Json
                var question = json["question"]?.ToString();
                var answer = json["answer"]?.ToString();

                // Extract final answer (after ####)
                var finalAnswer = ExtractFinalAnswer(answer ?? "");

                problems.Add(new GSM8KProblem
                {
                    Question = question ?? "",
                    FullSolution = answer ?? "",
                    FinalAnswer = finalAnswer,
                    Steps = ExtractSteps(answer ?? "")
                });
            }
            catch (Exception ex)
            {
                // Skip malformed lines - diagnostic output for debugging
                System.Diagnostics.Debug.WriteLine($"GSM8K: Failed to parse line: {ex.Message}");
            }
        }

        return Task.FromResult(problems); // Return completed task
    }

    public static Task<List<GSM8KProblem>> LoadFromJsonArrayAsync(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"GSM8K data file not found: {filePath}");

        var json = File.ReadAllText(filePath); // net462 compatible
        var data = JArray.Parse(json); // Use Newtonsoft.Json

        var problems = data.Select(item => new GSM8KProblem
        {
            Question = item["question"]?.ToString() ?? "",
            FullSolution = item["answer"]?.ToString() ?? "",
            FinalAnswer = ExtractFinalAnswer(item["answer"]?.ToString() ?? ""),
            Steps = ExtractSteps(item["answer"]?.ToString() ?? "")
        }).ToList();

        return Task.FromResult(problems); // Return completed task
    }

    public static List<GSM8KProblem> GetSampleProblems()
    {
        return new List<GSM8KProblem>
        {
            new()
            {
                Question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                FullSolution = "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72",
                FinalAnswer = "72",
                Steps = new[] { "48/2 = 24", "48+24 = 72" }
            },
            new()
            {
                Question = "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
                FullSolution = "Betty has 100/2 = $50.\nHer grandparents gave her 15*2 = $30.\nSo she has 50+15+30 = $95.\nShe needs 100-95 = $5 more.\n#### 5",
                FinalAnswer = "5",
                Steps = new[] { "100/2 = 50", "15*2 = 30", "50+15+30 = 95", "100-95 = 5" }
            },
            new()
            {
                Question = "A restaurant served 9 pizzas during lunch and 6 during dinner today. How many pizzas were served in all?",
                FullSolution = "The restaurant served 9+6 = 15 pizzas today.\n#### 15",
                FinalAnswer = "15",
                Steps = new[] { "9+6 = 15" }
            },
            new()
            {
                Question = "John has 3 boxes. Each box is 5 inches by 6 inches by 4 inches. What is the total volume of all 3 boxes?",
                FullSolution = "Each box has volume 5*6*4 = 120 cubic inches.\nTotal volume is 120*3 = 360 cubic inches.\n#### 360",
                FinalAnswer = "360",
                Steps = new[] { "5*6*4 = 120", "120*3 = 360" }
            },
            new()
            {
                Question = "A pen costs $3. A notebook costs $5. How much do 2 pens and 3 notebooks cost?",
                FullSolution = "2 pens cost 2*3 = $6.\n3 notebooks cost 3*5 = $15.\nTotal cost is 6+15 = $21.\n#### 21",
                FinalAnswer = "21",
                Steps = new[] { "2*3 = 6", "3*5 = 15", "6+15 = 21" }
            }
        };
    }

    private static string ExtractFinalAnswer(string answer)
    {
        if (answer == null) return "";

        // Format: "... #### 42"
        var parts = answer.Split(new[] { "####" }, StringSplitOptions.None); // net462 compatible
        if (parts.Length > 1)
        {
            return parts[1].Trim();
        }

        // Fallback: try to extract last number
        var numbers = System.Text.RegularExpressions.Regex.Matches(answer, @"-?\d+\.?\d*", System.Text.RegularExpressions.RegexOptions.None, RegexTimeout);
        return numbers.Count > 0 ? numbers[numbers.Count - 1].Value : ""; // net462: can't use ^1
    }

    private static string[] ExtractSteps(string solution)
    {
        // net462 compatible: Split by char array with StringSplitOptions
        var lines = solution.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
        return lines.Where(l => !l.Contains("####")).ToArray();
    }
}

public class GSM8KProblem
{
    public string Question { get; set; } = "";
    public string FullSolution { get; set; } = "";
    public string FinalAnswer { get; set; } = "";
    public string[] Steps { get; set; } = Array.Empty<string>();
}
