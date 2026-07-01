# Mock Directive Format

## Format Specification

```json
{
  "directive": "mock",
  "target": "service.method",
  "response": { "status": 200, "data": {} },
  "delay_ms": 0
}
```

## Fields
- `directive`: Always "mock"
- `target`: Fully qualified method name
- `response`: Mock response object
- `delay_ms`: Optional delay in milliseconds

*Added by CVG Hive autonomous bounty fulfillment*