# Local Fine-tuning of Llama 3B Instruct on MacBook with Apple's mlx_lm.lora Tool:

Fine-tuning works on Apple M series MacBooks with at least 16 GB RAM. As an example, one full training run at 180 iterations on the LEET-Arg dataset takes less than 30 minutes on a MacBook Pro M5 with 16 GB RAM. Peak memory usage will be 10 GB, the fans will run fast, and no other work is possible.

It is not necessary to download the model.

1. Create Python virtual environment and activate it.

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

2. Update pip.

   ```bash
   pip install --upgrade pip
   ```

3. Install dependencies.

   ```bash
   pip install mlx-lm datasets huggingface_hub
   ```

4. Prepare training dataset by creating a folder named `data` with two files, `train.jsonl` and `valid.jsonl`. MLX accepts JSON Lines format. A `train.jsonl` file is sufficient for training only. When using Apple's official `mlx_lm.lora` tool, it expects a .jsonl file with a "messages" array following standard chat roles (user, assistant, system):

   ```bash
   {"messages": [{"role": "user", "content": "Question..."}, {"role": "assistant", "content": "Answer..."}]}
   ```

   When using MLX’s chat format, MLX converts each line into the Llama 3.2 template (`"text": "<|begin_of_text|><|start_header_id|>...`) using the tokenizer that comes with the model. Further, `mlx_lm.generate` applies the chat template to `--prompt` by default (see below).

5. Start training.

   ```bash
   mlx_lm.lora \
                  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
                  --train \
                  --data ./data \
                  --mask-prompt \
                  --iters 100 \
                  --batch-size 2 \
                  --grad-accumulation-steps 4 \
                  --learning-rate 1e-4 \
                  --adapter-path ./adapters

   ```

6. Prompt your trained model.

   ```bash
   mlx_lm.generate \
                  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
                  --max-tokens 600 \
                  --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                  You are an expert legal reasoning assistant. Analyze the following MBE question step-by-step and select the correct answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

                  Domain: Criminal_law
                  Question: The defendant decided to kill his neighbor. He set out for the neighbor's house. Before he got there he saw a man who resembled the neighbor. Thinking the man was his neighbor, the defendant shot at the man. The shot missed the man but wounded a child, who was some distance away. The defendant had not seen the child. In a prosecution under a statute that proscribes attempt to commit murder, the district attorney should indicate that the intended victim(s) was (were)

                  Options:
                  A. the neighbor only.
                  B. the man only.
                  C. the child only.
                  D. the neighbor and the man.

                  Please analyze the legal rules and select the correct option.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                  "
   ```
