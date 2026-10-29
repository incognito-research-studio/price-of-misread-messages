# Evaluating your own reader or planner

You can price your own system's reading errors without adopting the knowledge graph, the SOP
engine or the rest of the architecture as your production design. Implement one or both of two
small interfaces and call `cdd.evaluate.evaluate`. The worked example is
[`examples/evaluate_your_own_reader.py`](../examples/evaluate_your_own_reader.py):

```bash
python examples/evaluate_your_own_reader.py
```

## The two interfaces

**Reader** ([`interpreters/base.py`](../src/cdd/interpreters/base.py)): turns one message into
typed delta proposals.

```python
class MyReader:
    name = "my-reader"

    def interpret(self, message: Message, view: StateView) -> Interpretation:
        ...  # return Interpretation(proposals=(Proposal(type=..., entity=..., span=...), ...))
```

A proposal names the delta type, the entity id and the span of the message that supports it.
The span parser fills the value and the window from that span, so a reader cannot write a
number the message does not contain. A reader may also fill `op`, `value` and `window` itself;
the parser keeps them. `view` is the interpreted state so far, for readers that need context.

**Planner** ([`deciders.py`](../src/cdd/deciders.py)): turns the interpreted state into this
week's orders.

```python
class MyPlanner:
    name = "my-planner"

    def plan(self, view: StateView, position: InventoryPosition) -> Plan:
        ...  # return Plan(period=position.period, orders=(OrderLine(plant, sku, qty), ...))
```

`view.number(entity, Attribute.CAPACITY, period)` and the other attribute reads give the state
as the pipeline believes it; `position` gives stock, open orders and backlog. The default
planner is the rolling-horizon MILP.

## One call

```python
from cdd.benchmark import GeneratorConfig, generate
from cdd.evaluate import evaluate, report

for stream in generate(GeneratorConfig.model_validate({"n_streams": 4})):
    result = evaluate(stream, MyReader(...), MyPlanner())   # planner is optional
    print(report([result]))
```

`evaluate` returns the stream's cost, its interpretation and decision regret, precision, recall
and F1, true-constraint violations, audit completeness, and every delta error with the cost
saved by repairing it alone (`delta_c`), most expensive first.

## Your costs

Operational costs come from the stream's world, which the generator config sets: per-SKU
`holding_cost` and `backlog_cost`, price and capacity ranges, lead times and an optional
`budget_per_period` (see `WorldSpec` in
[`benchmark/generator.py`](../src/cdd/benchmark/generator.py)). Set them to match your
business before reading anything into the regret numbers.

## Larger studies

For many conditions, repeated runs and versioned outputs, use the experiment runner
(`cdd run --config ...`). Each run folder holds `meta.json` (package and schema versions, the
code commit and whether the working tree was dirty), the resolved `config.json`, per-row
`results.jsonl`, `errors.csv` with repair costs, and the sensitivity map.
