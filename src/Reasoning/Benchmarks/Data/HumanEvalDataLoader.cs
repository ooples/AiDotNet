using System.Text.Json;
using System.Text.RegularExpressions;

namespace AiDotNet.Reasoning.Benchmarks.Data;

/// <summary>
/// Loader for HumanEval benchmark dataset.
/// </summary>
/// <remarks>
/// HumanEval format:
/// {
///   "task_id": "HumanEval/0",
///   "prompt": "def has_close_elements(numbers, threshold):\n    \"\"\" Check if in given list...",
///   "canonical_solution": "    for idx, elem in enumerate(numbers):...",
///   "test": "def check(candidate):...",
///   "entry_point": "has_close_elements"
/// }
/// </remarks>
public class HumanEvalDataLoader
{
    public static async Task<List<HumanEvalProblem>> LoadFromFileAsync(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"HumanEval dataset not found: {filePath}");
        }

        var problems = new List<HumanEvalProblem>();
        var lines = await File.ReadAllLinesAsync(filePath);

        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            try
            {
                var json = JsonSerializer.Deserialize<JsonElement>(line);

                problems.Add(new HumanEvalProblem
                {
                    TaskId = json.GetProperty("task_id").GetString() ?? "",
                    Prompt = json.GetProperty("prompt").GetString() ?? "",
                    CanonicalSolution = json.GetProperty("canonical_solution").GetString() ?? "",
                    Test = json.GetProperty("test").GetString() ?? "",
                    EntryPoint = json.GetProperty("entry_point").GetString() ?? ""
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to parse HumanEval line: {ex.Message}");
            }
        }

        return problems;
    }

    public static List<HumanEvalProblem> GetSampleProblems()
    {
        return new List<HumanEvalProblem>
        {
            new()
            {
                TaskId = "HumanEval/0",
                Prompt = @"from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """""" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """"""
",
                CanonicalSolution = @"    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False",
                Test = @"def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False",
                EntryPoint = "has_close_elements"
            },
            new()
            {
                TaskId = "HumanEval/1",
                Prompt = @"from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """""" Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those groups into separate strings and return the list of those.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """"""
",
                CanonicalSolution = @"    result = []
    current_string = []
    current_depth = 0
    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()
    return result",
                Test = @"def check(candidate):
    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']",
                EntryPoint = "separate_paren_groups"
            },
            new()
            {
                TaskId = "HumanEval/2",
                Prompt = @"def truncate_number(number: float) -> float:
    """""" Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).
    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """"""
",
                CanonicalSolution = @"    return number % 1.0",
                Test = @"def check(candidate):
    assert candidate(3.5) == 0.5
    assert abs(candidate(1.33) - 0.33) < 1e-6",
                EntryPoint = "truncate_number"
            }
        };
    }
}

public class HumanEvalProblem
{
    public string TaskId { get; set; } = "";
    public string Prompt { get; set; } = "";
    public string CanonicalSolution { get; set; } = "";
    public string Test { get; set; } = "";
    public string EntryPoint { get; set; } = "";
}
