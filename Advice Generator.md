COMPARISON_PROMPT = """Compare these two code solutions to the same problem.
Explain the main principles that differ between the codes:
CODE 1:```python{code1}```
CODE 2:```python{code2}```
Please identify:
1. The main algorithmic approaches used in each
2. Key differences in methodology
3. Strengths of each approach"""

HYBRID_PROMPT = """We have up until now done experiments with two major types of codes, that are described in detail below.
### Comparison Analysis
{comparison}
### CODE 1 (Score: {score1})```python{code1}```
### CODE 2 (Score: {score2})```python{code2}```
PLEASE CREATE AN ALGORITHM THAT USES THE BEST PARTS OF BOTH STRATEGIES 
TO CREATE A HYBRID STRATEGY THAT IS TRULY WONDERFUL AND SCORES HIGHER 
THAN EITHER OF THE INDIVIDUAL STRATEGIES.
Please provide the hybrid code:
```python# YOUR HYBRID CODE```"""

FORMAT_IDEA_PROMPT = """Structure the given idea into the following format:
<description>
Your description about the method goes here.
</description>
<steps>
Your list of steps to implement the method goes here.
</steps>
<notes>
Strengths and weaknesses of the idea goes here.
</notes>
Idea to format:
{idea}"""
"""
