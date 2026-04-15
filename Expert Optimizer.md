prompt = f"""Analyze the following Python code and select the most appropriate expert advice type.
### Code to Analyze:
```python
```
### Available Advice Types:
{advice_options}
### Task:
Based on the code's purpose, libraries used, and domain, select the SINGLE most appropriate advice type.
Respond with ONLY the number (1-{len(self.advice_library)}) of the best advice type, nothing else.
For example, if boosted_trees is best, respond with just: 2"""

prompt = f"""### Expert Advice ({advice.name})
{advice.advice}
### Current Code
```python
{node.code}
```
### Task
Apply the expert advice above to improve the code. Focus on:
1. Following the advice guidelines
2. Maintaining correctness
3. Improving the score/performance
Please provide the improved code:
```python
# YOUR IMPROVED CODE
```"""
