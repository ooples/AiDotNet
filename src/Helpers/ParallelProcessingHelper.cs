namespace AiDotNet.Helpers;

/// <summary>
/// Helper class for executing multiple tasks in parallel to improve performance.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This class helps your AI models run faster by doing multiple calculations 
/// at the same time, similar to having multiple people working on different parts of a project
/// simultaneously instead of one person doing everything sequentially.
/// </remarks>
public static class ParallelProcessingHelper
{
    /// <summary>
    /// Executes multiple functions in parallel with controlled concurrency.
    /// </summary>
    /// <typeparam name="T">The type of result returned by each function.</typeparam>
    /// <param name="taskFunctions">Collection of functions to execute in parallel.</param>
    /// <param name="maxDegreeOfParallelism">Maximum number of tasks to run simultaneously. 
    /// If not specified, defaults to the number of processor cores.</param>
    /// <returns>A list containing all the results from the executed functions.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes a collection of "work items" (functions) and runs them
    /// in parallel. It controls how many can run at once to prevent overloading your computer.
    /// 
    /// For example, if you have 100 data processing tasks but set maxDegreeOfParallelism to 4,
    /// only 4 will run at once. As each one finishes, a new one starts until all are complete.
    /// 
    /// This is useful for AI tasks like processing multiple datasets or training several models
    /// at the same time.
    /// </remarks>
    public static async Task<List<T>> ProcessTasksInParallel<T>(IEnumerable<Func<T>> taskFunctions, int? maxDegreeOfParallelism = null)
    {
        var results = new List<T>();
        var processorCount = maxDegreeOfParallelism ?? Environment.ProcessorCount;
        var taskList = new List<Task<T>>();

        foreach (var taskFunction in taskFunctions)
        {
            if (taskList.Count >= processorCount)
            {
                var completedTask = await Task.WhenAny(taskList);
                taskList.Remove(completedTask);
                results.Add(await completedTask);
            }
            taskList.Add(Task.Run(taskFunction));
        }

        results.AddRange(await Task.WhenAll(taskList));
        return results;
    }

    /// <summary>
    /// Executes multiple pre-created tasks in parallel batches with controlled concurrency.
    /// </summary>
    /// <typeparam name="T">The type of result returned by each task.</typeparam>
    /// <param name="tasks">Collection of tasks to execute in parallel.</param>
    /// <param name="maxDegreeOfParallelism">Maximum number of tasks to run simultaneously in each batch.
    /// If not specified, defaults to the number of processor cores.</param>
    /// <returns>A list containing all the results from the executed tasks.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method is similar to the one above but works with tasks that are already
    /// created. It processes these tasks in batches, with each batch containing a number of tasks
    /// equal to your processor count (or the value you specify).
    /// 
    /// Think of it like having a pile of homework assignments and deciding to complete them in
    /// batches of 4 at a time. You finish all 4 in the current batch before moving to the next batch.
    /// 
    /// This approach is useful when you already have a collection of tasks (like model training jobs)
    /// and want to process them efficiently without overwhelming your system.
    /// </remarks>
    public static async Task<List<T>> ProcessTasksInParallel<T>(IEnumerable<Task<T>> tasks, int? maxDegreeOfParallelism = null)
    {
        var results = new List<T>();
        var processorCount = maxDegreeOfParallelism ?? Environment.ProcessorCount;
        var taskList = tasks.ToList();

        while (taskList.Count > 0)
        {
            var batch = taskList.Take(processorCount).ToList();
            taskList = [.. taskList.Skip(processorCount)];

            var completedTasks = await Task.WhenAll(batch);
            results.AddRange(completedTasks);
        }

        return results;
    }
}
