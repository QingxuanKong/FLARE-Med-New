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

```
Benchmark completed in 0m 48s
Processed 3 questions using 8 threads
Average time per question: 16.24s
Results saved to ./prediction/medqa/rag_flare_followup_8/OpenAI/gpt-4.1-nano/MedText/MedCPT
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: MedCPT
llm_name: OpenAI/gpt-4.1-nano
rag: True
k: 8
results_dir: ./prediction
dataset_name: medqa
enable_flare: True
look_ahead_steps: 50
look_ahead_truncate_at_boundary: .,?,!
max_query_length: 300
follow_up: True
n_rounds: 3
n_queries: 2
threads: 8

[medqa] mean acc: 0.8303; proportion std: 0.0105
[Average] mean acc: 0.8303
run_benchmark.sh completed successfully.
Wrapper script finished successfully.
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

```
Benchmark completed in 0m 58s
Processed 2 questions using 8 threads
Average time per question: 29.10s
Results saved to ./prediction/mmlu/rag_flare_followup_8/OpenAI/gpt-4.1-nano/MedText/MedCPT
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: MedCPT
llm_name: OpenAI/gpt-4.1-nano
rag: True
k: 8
results_dir: ./prediction
dataset_name: mmlu
enable_flare: True
look_ahead_steps: 50
look_ahead_truncate_at_boundary: .,?,!
max_query_length: 300
follow_up: True
n_rounds: 3
n_queries: 2
threads: 8

[mmlu] mean acc: 0.8788; proportion std: 0.0099
[Average] mean acc: 0.8788
run_benchmark.sh completed successfully.
Wrapper script finished successfully.
```

## bioasq
### k=32
#### RRF-4
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
#### MedCPT
```
Benchmark completed in 0m 4s
Processed 5 questions using 8 threads
Average time per question: 0.92s
Results saved to ./prediction/bioasq/rag_32/OpenAI/gpt-4.1-nano/MedText/MedCPT
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: MedCPT
llm_name: OpenAI/gpt-4.1-nano
rag: True
k: 32
results_dir: ./prediction
dataset_name: bioasq
enable_flare: False
look_ahead_steps: 50
look_ahead_truncate_at_boundary: .,?,!
max_query_length: 300
follow_up: False
n_rounds: 3
n_queries: 2
threads: 8

[bioasq] mean acc: 0.7282; proportion std: 0.0179
[Average] mean acc: 0.7282
```
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
### k=16
```
Benchmark completed in 3m 30s
Processed 275 questions using 4 threads
Average time per question: 0.77s
Results saved to ./prediction/bioasq/rag_16/OpenAI/gpt-4.1-nano/MedText/RRF-4
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: RRF-4
llm_name: OpenAI/gpt-4.1-nano
rag: True
k: 16
results_dir: ./prediction
dataset_name: bioasq
enable_flare: False
look_ahead_steps: 50
look_ahead_truncate_at_boundary: .,?,!
max_query_length: 300
follow_up: False
n_rounds: 3
n_queries: 2
threads: 4

[bioasq] mean acc: 0.7120; proportion std: 0.0182
[Average] mean acc: 0.7120
```
### k=8
#### cot
````

````
#### MedCPT
````
Benchmark completed in 0m 7s
Processed 29 questions using 16 threads
Average time per question: 0.27s
Results saved to ./prediction/bioasq/rag_8/OpenAI/gpt-4.1-nano/MedText/MedCPT
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: MedCPT
llm_name: OpenAI/gpt-4.1-nano
rag: True
k: 8
results_dir: ./prediction
dataset_name: bioasq
enable_flare: False
look_ahead_steps: 50
look_ahead_truncate_at_boundary: .,?,!
max_query_length: 300
follow_up: False
n_rounds: 3
n_queries: 2
threads: 16

[bioasq] mean acc: 0.7249; proportion std: 0.0180
[Average] mean acc: 0.7249
````
#### Contriever
````
Benchmark completed in 0m 31s
Processed 116 questions using 8 threads
Average time per question: 0.27s
Results saved to ./prediction/bioasq/rag_8/OpenAI/gpt-4.1-nano/MedText/Contriever
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: Contriever
llm_name: OpenAI/gpt-4.1-nano
rag: True
k: 8
results_dir: ./prediction
dataset_name: bioasq
enable_flare: False
look_ahead_steps: 50
look_ahead_truncate_at_boundary: .,?,!
max_query_length: 300
follow_up: False
n_rounds: 3
n_queries: 2
threads: 8

[bioasq] mean acc: 0.6990; proportion std: 0.0185
[Average] mean acc: 0.6990
````
#### BM25
````

````
#### SPECTRE
````
Benchmark completed in 0m 3s
Processed 2 questions using 8 threads
Average time per question: 1.52s
Results saved to ./prediction/bioasq/rag_8/OpenAI/gpt-4.1-nano/MedText/SPECTER
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: SPECTER
llm_name: OpenAI/gpt-4.1-nano
rag: True
k: 8
results_dir: ./prediction
dataset_name: bioasq
enable_flare: False
look_ahead_steps: 50
look_ahead_truncate_at_boundary: .,?,!
max_query_length: 300
follow_up: False
n_rounds: 3
n_queries: 2
threads: 8

[bioasq] mean acc: 0.7087; proportion std: 0.0183
[Average] mean acc: 0.7087
````
#### RRF-2
````

````
#### RRF-4
````

````

#### FLARE

```
(py310) yyfsss@DESKTOP-86MTRK8:~/FLARE-Med/MIRAGE$ python src/evaluate.py --config src/config.json
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: MedCPT
llm_name: OpenAI/gpt-4.1-nano
rag: True
k: 8
results_dir: ./prediction
dataset_name: bioasq
enable_flare: True
look_ahead_steps: 50
look_ahead_truncate_at_boundary: .,?,!
max_query_length: 300
follow_up: True
n_rounds: 3
n_queries: 2
threads: 8

[bioasq] mean acc: 0.7524; proportion std: 0.0174
[Average] mean acc: 0.7524
```



## pubmedqa
```

Benchmark completed in 1m 51s
Processed 500 questions using 8 threads
Average time per question: 0.22s
Results saved to ./prediction/pubmedqa/cot/OpenAI/gpt-4.1-nano
Loaded configuration from src/config.json

Final arguments:
corpus_name: MedText
retriever_name: RRF-4
llm_name: OpenAI/gpt-4.1-nano
rag: False
k: 32
results_dir: ./prediction
dataset_name: pubmedqa
enable_flare: False
look_ahead_steps: 50
look_ahead_truncate_at_boundary: .,?,!
max_query_length: 300
follow_up: False
n_rounds: 3
n_queries: 2
threads: 8

[pubmedqa] mean acc: 0.4900; proportion std: 0.0224
[Average] mean acc: 0.4900
```

