# Python FAQ

**Q: How do async and await work?**
A: `async` and `await` are Python keywords used to define and execute asynchronous code. An `async def` function is a coroutine. You use `await` inside a coroutine to pause its execution until the awaited task completes, freeing up the event loop to run other tasks.

**Q: What is tiktoken?**
A: `tiktoken` is a fast BPE tokeniser created by OpenAI. It is used to count the number of tokens in a string before sending it to an LLM, helping you stay within context limits.
