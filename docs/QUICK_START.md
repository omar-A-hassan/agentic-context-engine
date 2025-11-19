# 🚀 ACE Framework Quick Start

Get your first self-learning AI agent running in 5 minutes!

## Installation

```bash
pip install ace-framework
```

## Your First ACE Agent

### Step 1: Set API Key

```bash
export OPENAI_API_KEY="your-key-here"
# Or: ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
```

### Step 2: Create `my_first_ace.py`

```python
from ace import OfflineAdapter, Generator, Reflector, Curator
from ace import LiteLLMClient, Sample, TaskEnvironment, EnvironmentResult


# Simple environment that checks if answer contains the ground truth
class SimpleEnvironment(TaskEnvironment):
    def evaluate(self, sample, generator_output):
        correct = str(sample.ground_truth).lower() in str(generator_output.final_answer).lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth
        )


# Initialize LLM client
client = LiteLLMClient(model="gpt-4o-mini")

# Create ACE components
generator = Generator(client)
reflector = Reflector(client)
curator = Curator(client)

# Create adapter to orchestrate everything
adapter = OfflineAdapter(generator=generator, reflector=reflector, curator=curator)

# Create training samples
samples = [
    Sample(question="What is the capital of France?", context="", ground_truth="Paris"),
    Sample(question="What is 2 + 2?", context="", ground_truth="4"),
    Sample(question="Who wrote Romeo and Juliet?", context="", ground_truth="Shakespeare")
]

# Train the agent
print("Training agent...")
results = adapter.run(samples, SimpleEnvironment(), epochs=2)

# Save learned strategies
adapter.playbook.save_to_file("my_agent.json")
print(f"✅ Agent trained! Learned {len(adapter.playbook.bullets())} strategies")

# Test with new question
test_output = generator.generate(
    question="What is 5 + 3?",
    context="",
    playbook=adapter.playbook
)
print(f"\nTest question: What is 5 + 3?")
print(f"Answer: {test_output.final_answer}")
```

### Step 3: Run It

```bash
python my_first_ace.py
```

Expected output:
```
Training agent...
✅ Agent trained! Learned 3 strategies

Test question: What is 5 + 3?
Answer: 8
```

---

## What Just Happened?

Your agent:
1. **Learned** from training examples
2. **Reflected** on what strategies work
3. **Built a playbook** of successful approaches
4. **Applied** strategies to solve a new problem

---

## Next Steps

### Load Saved Agent

```python
from ace import Playbook

# Load previously trained agent
playbook = Playbook.load_from_file("my_agent.json")

# Use it
output = generator.generate(
    question="New question",
    context="",
    playbook=playbook
)
```

### Try Different Models

```python
# Anthropic Claude
client = LiteLLMClient(model="claude-3-5-sonnet-20241022")

# Google Gemini
client = LiteLLMClient(model="gemini-pro")

# Local Ollama
client = LiteLLMClient(model="ollama/llama2")
```

### Add ACE to Existing Agents

Already have an agent? Wrap it with ACE learning:

```python
from ace.integrations import ACEAgent  # For browser-use
from ace.integrations import ACELangChain  # For LangChain

# See Integration Guide for details
```

---

## Learn More

- **[Integration Guide](INTEGRATION_GUIDE.md)** - Add ACE to existing agents
- **[Complete Guide](COMPLETE_GUIDE_TO_ACE.md)** - Deep dive into ACE concepts
- **[Examples](../examples/)** - Real-world examples
  - [Browser Automation](../examples/browser-use/) - Self-improving browser agents
  - [LangChain Integration](../examples/langchain/) - Wrap chains with learning
  - [Custom Integration](../examples/custom_integration_example.py) - Any agent pattern

---

## Common Patterns

### Online Learning (Learn While Running)

```python
from ace import OnlineAdapter

adapter = OnlineAdapter(playbook, generator, reflector, curator)

# Process tasks one by one, learning from each
for task in tasks:
    result = adapter.process(task, environment)
```

### Custom Evaluation

```python
class MathEnvironment(TaskEnvironment):
    def evaluate(self, sample, output):
        try:
            result = eval(output.final_answer)
            correct = result == sample.ground_truth
            return EnvironmentResult(
                feedback=f"Result: {result}. {'✓' if correct else '✗'}",
                ground_truth=sample.ground_truth
            )
        except:
            return EnvironmentResult(
                feedback="Invalid math expression",
                ground_truth=sample.ground_truth
            )
```

---

## Troubleshooting

**Import errors?**
```bash
pip install --upgrade ace-framework
```

**API key not working?**
- Verify key is correct: `echo $OPENAI_API_KEY`
- Try different model: `LiteLLMClient(model="gpt-3.5-turbo")`

**Need help?**
- [GitHub Issues](https://github.com/kayba-ai/agentic-context-engine/issues)
- [Discord Community](https://discord.com/invite/mqCqH7sTyK)

---

**Ready to build production agents?** Check out the [Integration Guide](INTEGRATION_GUIDE.md) for browser automation, LangChain, and custom agent patterns.
