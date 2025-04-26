## medqa 
```
Benchmark completed in 43m 22s
Processed 1273 questions using 8 threads
Average time per question: 2.04s
Results saved to ./prediction/medqa/cot/OpenAI/gpt-4.1-nano
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: RRF-4
llm_name: OpenAI/gpt-4.1-nano
rag: False
k: 32
results_dir: ./prediction
dataset_name: medqa

[medqa] mean acc: 0.7714; proportion std: 0.0118
[Average] mean acc: 0.7714
```

```
Benchmark completed in 59m 19s
Processed 1273 questions using 8 threads
Average time per question: 2.80s
Results saved to ./prediction/medqa/rag_32/OpenAI/gpt-4.1-nano/MedText/RRF-4
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: RRF-4
llm_name: OpenAI/gpt-4.1-nano
rag: True
k: 32
results_dir: ./prediction
dataset_name: medqa

[medqa] mean acc: 0.7675; proportion std: 0.0118
[Average] mean acc: 0.7675
```
## mmlu
```
Benchmark completed in 30m 27s
Processed 1089 questions using 8 threads
Average time per question: 1.68s
Results saved to ./prediction/mmlu/cot/OpenAI/gpt-4.1-nano
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: RRF-4
llm_name: OpenAI/gpt-4.1-nano
rag: False
k: 32
results_dir: ./prediction
dataset_name: mmlu

[mmlu] mean acc: 0.8687; proportion std: 0.0102
[Average] mean acc: 0.8687
```

```
Bench mark
(py310) yyfsss@DESKTOP-86MTRK8:~/FLARE-Med/MIRAGE$ python src/evaluate.py --config src/config.json
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: RRF-4
llm_name: OpenAI/gpt-4.1-nano
rag: True
k: 32
results_dir: ./prediction
dataset_name: mmlu

[mmlu] mean acc: 0.8476; proportion std: 0.0109
[Average] mean acc: 0.8476
```

## bioasq
```
Benchmark completed in 14m 39s
Processed 618 questions using 8 threads
Average time per question: 1.42s
Results saved to ./prediction/bioasq/cot/OpenAI/gpt-4.1-nano
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: RRF-4
llm_name: OpenAI/gpt-4.1-nano
rag: False
k: 32
results_dir: ./prediction
dataset_name: bioasq

[bioasq] mean acc: 0.8301; proportion std: 0.0151
[Average] mean acc: 0.8301
```

```
Benchmark completed in 23m 18s
Processed 618 questions using 8 threads
Average time per question: 2.26s
Results saved to ./prediction/bioasq/rag_32/OpenAI/gpt-4.1-nano/MedText/RRF-4
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: RRF-4
llm_name: OpenAI/gpt-4.1-nano
rag: True
k: 32
results_dir: ./prediction
dataset_name: bioasq

[bioasq] mean acc: 0.7136; proportion std: 0.0182
[Average] mean acc: 0.7136
```

