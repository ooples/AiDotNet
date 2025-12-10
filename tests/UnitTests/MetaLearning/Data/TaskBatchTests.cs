using AiDotNet.MetaLearning.Data;

namespace AiDotNet.Tests.UnitTests.MetaLearning.Data;

public class TaskBatchTests
{
    [Fact]
    public void Constructor_ValidTasks_CreatesBatch()
    {
        // Arrange
        var tasks = CreateTestTasks(batchSize: 4, numWays: 5, numShots: 1, numQueryPerClass: 15);

        // Act
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        // Assert
        Assert.NotNull(batch);
        Assert.Equal(4, batch.BatchSize);
        Assert.Equal(5, batch.NumWays);
        Assert.Equal(1, batch.NumShots);
        Assert.Equal(15, batch.NumQueryPerClass);
    }

    [Fact]
    public void Constructor_NullTasks_ThrowsArgumentNullException()
    {
        // Arrange
        ITask<double, Matrix<double>, Vector<double>>[]? tasks = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new TaskBatch<double, Matrix<double>, Vector<double>>(tasks!));
    }

    [Fact]
    public void Constructor_EmptyTasks_ThrowsArgumentException()
    {
        // Arrange
        var tasks = Array.Empty<ITask<double, Matrix<double>, Vector<double>>>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new TaskBatch<double, Matrix<double>, Vector<double>>(tasks));
    }

    [Fact]
    public void Constructor_InconsistentNumWays_ThrowsArgumentException()
    {
        // Arrange
        var task1 = CreateTestTask(numWays: 5, numShots: 1, numQueryPerClass: 15);
        var task2 = CreateTestTask(numWays: 3, numShots: 1, numQueryPerClass: 15); // Different NumWays

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task1, task2 }));

        Assert.Contains("same configuration", exception.Message);
    }

    [Fact]
    public void Constructor_InconsistentNumShots_ThrowsArgumentException()
    {
        // Arrange
        var task1 = CreateTestTask(numWays: 5, numShots: 1, numQueryPerClass: 15);
        var task2 = CreateTestTask(numWays: 5, numShots: 5, numQueryPerClass: 15); // Different NumShots

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task1, task2 }));

        Assert.Contains("same configuration", exception.Message);
    }

    [Fact]
    public void Indexer_ValidIndex_ReturnsTask()
    {
        // Arrange
        var tasks = CreateTestTasks(batchSize: 4, numWays: 5, numShots: 1, numQueryPerClass: 15);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        // Act
        var task0 = batch[0];
        var task3 = batch[3];

        // Assert
        Assert.NotNull(task0);
        Assert.NotNull(task3);
        Assert.Equal(tasks[0], task0);
        Assert.Equal(tasks[3], task3);
    }

    [Fact]
    public void Tasks_ReturnsAllTasks()
    {
        // Arrange
        var tasks = CreateTestTasks(batchSize: 3, numWays: 5, numShots: 1, numQueryPerClass: 15);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        // Act
        var returnedTasks = batch.Tasks;

        // Assert
        Assert.Equal(tasks.Length, returnedTasks.Length);
        for (int i = 0; i < tasks.Length; i++)
        {
            Assert.Equal(tasks[i], returnedTasks[i]);
        }
    }

    [Fact]
    public void BatchSize_ReturnsCorrectCount()
    {
        // Arrange
        var tasks = CreateTestTasks(batchSize: 8, numWays: 5, numShots: 1, numQueryPerClass: 15);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        // Act
        int size = batch.BatchSize;

        // Assert
        Assert.Equal(8, size);
    }

    [Fact]
    public void TaskBatch_SingleTask_Works()
    {
        // Arrange
        var tasks = CreateTestTasks(batchSize: 1, numWays: 5, numShots: 1, numQueryPerClass: 15);

        // Act
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        // Assert
        Assert.Equal(1, batch.BatchSize);
        Assert.NotNull(batch[0]);
    }

    [Fact]
    public void TaskBatch_LargeBatch_Works()
    {
        // Arrange
        var tasks = CreateTestTasks(batchSize: 32, numWays: 5, numShots: 1, numQueryPerClass: 15);

        // Act
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);

        // Assert
        Assert.Equal(32, batch.BatchSize);
        Assert.All(batch.Tasks, task => Assert.NotNull(task));
    }

    private static ITask<double, Matrix<double>, Vector<double>>[] CreateTestTasks(
        int batchSize, int numWays, int numShots, int numQueryPerClass)
    {
        var tasks = new ITask<double, Matrix<double>, Vector<double>>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            tasks[i] = CreateTestTask(numWays, numShots, numQueryPerClass);
        }
        return tasks;
    }

    private static Task<double, Matrix<double>, Vector<double>> CreateTestTask(
        int numWays, int numShots, int numQueryPerClass)
    {
        int supportSize = numWays * numShots;
        int querySize = numWays * numQueryPerClass;

        var supportInput = new Matrix<double>(supportSize, 5);
        var supportOutput = new Vector<double>(supportSize);
        var queryInput = new Matrix<double>(querySize, 5);
        var queryOutput = new Vector<double>(querySize);

        return new Task<double, Matrix<double>, Vector<double>>(
            supportInput, supportOutput, queryInput, queryOutput,
            numWays, numShots, numQueryPerClass);
    }
}
