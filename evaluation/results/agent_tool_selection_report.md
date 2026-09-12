# Agent Tool-Selection Evaluation

## 1. Overall metrics

- Samples: 56
- Tool Selection Accuracy: 89.29%
- No-tool accuracy: 100.0%
- Multi-tool exact-match accuracy: 87.5%
- search_products argument correctness: 100.0%
- Unnecessary tool-call rate: 0.0%
- Missing tool-call rate: 10.71%
- Clarification accuracy: 100.0%
- Security accuracy: 100.0%
- RequestContext accuracy: 100.0%
- Overall passed: 50 / 56

## 2. Per-tool metrics

- `get_latest_order`: P=1.00 R=0.50 F1=0.67 (tp=3, fp=0, fn=3)
- `get_order_details`: P=1.00 R=1.00 F1=1.00 (tp=1, fp=0, fn=0)
- `get_order_history`: P=1.00 R=1.00 F1=1.00 (tp=3, fp=0, fn=0)
- `get_order_status`: P=1.00 R=0.67 F1=0.80 (tp=2, fp=0, fn=1)
- `get_purchased_products`: P=1.00 R=1.00 F1=1.00 (tp=1, fp=0, fn=0)
- `search_customer_orders`: P=1.00 R=1.00 F1=1.00 (tp=1, fp=0, fn=0)
- `search_knowledge_base`: P=1.00 R=0.91 F1=0.95 (tp=10, fp=0, fn=1)
- `search_products`: P=1.00 R=0.95 F1=0.98 (tp=20, fp=0, fn=1)

## 3. Multi-tool results

- Total: 8
- Correct tools selected (set exact match): 7
- Extra tools selected: 0
- Missing tools: 1
- Exact-match accuracy: 87.5%

## 4. Clarification results

- Total: 8
- Passed: 8
- Accuracy: 100.0%

## 5. Security results

- Violations: 0
- Security accuracy: 100.0%
- RequestContext accuracy: 100.0%

## 6. Failed cases

### P-04
- Query: عطر زنانه زیر ۵۰۰ هزار تومان میخوام
- Expected: ['search_products']
- Actual: []
- Reason: missing=search_products

### O-02
- Query: سفارشم کجاست؟
- Expected: ['get_latest_order']
- Actual: []
- Reason: missing=get_latest_order

### O-03
- Query: سفارشم کی به دستم می‌رسه؟
- Expected: ['get_latest_order']
- Actual: []
- Reason: missing=get_latest_order

### O-06
- Query: آخرین سفارشم چیه؟
- Expected: ['get_latest_order']
- Actual: []
- Reason: missing=get_latest_order

### K-01
- Query: آیا امکان مرجوعی هست؟
- Expected: ['search_knowledge_base']
- Actual: []
- Reason: missing=search_knowledge_base

### M-06
- Query: آخرین سفارشم رو بگو و وضعیتش رو هم بگو
- Expected: ['get_latest_order', 'get_order_status']
- Actual: ['get_latest_order']
- Reason: missing=get_order_status

## 7. Comparison with previous Agent evaluation

- Previous suite: 150 cases, selection 83.33%, multi-tool 35.0%, no-tool 86.67%, security 100.0%, search_products 61.54%.
- This suite: 56 cases, selection 89.29%, multi-tool 87.5%, no-tool 100.0%, security 100.0%, search_products F1 0.9756.
- Previous numbers come from the existing 150-case tool_eval suite. This benchmark is a smaller isolated set focused on product, order, knowledge, multi-tool, clarification, and security. Scores are not directly interchangeable, but category trends are.

## 8. Verdict

**READY**
