namespace AiDotNet.Helpers;

public static class ParallelProcessingHelper
{
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