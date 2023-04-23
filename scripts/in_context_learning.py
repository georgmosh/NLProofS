import re
import json
import torch
import openai
from tqdm import tqdm
from time import sleep
from fairseq.models.transformer_lm import TransformerLanguageModel

from reporting import read_file, read_dict, write_dict

"""
Code inspired/partially adopted from https://github.com/princeton-nlp/NLProofS/blob/main/scripts/prompting.py
Thanks Kaiyu Yang!
"""
LANG_MODELS_OPTIONS = {"Codex": "code-davinci-002", "GPT-3": "text-davinci-002", "GPT-3.5": "text-davinci-003",
                       "BioGPT": "./biogpt_checkpoints/Pre-trained-BioGPT"}

# Determine the language model and checkpoint if necessary
lang_model = LANG_MODELS_OPTIONS["GPT-3.5"]
lang_model_checkpoint = "./biogpt_checkpoints/checkpoint.pt"
lang_model_beam = 5

# Define the necessary hyperparameters (dataset and OpenAI API key)
api_key_openai = "helloworld!"  # TODO: Please do not distribute!!!
dataset = r"C:\Users\georg_mosh.IPL2-PC\PycharmProjects\NLProofS\data\entailment_trees_emnlp2021_data_v3\dataset\task_2\dev.jsonl" \
        if not("biogpt" in lang_model.lower()) else r"H:\BioASQ-binary-candidate-proofs.jsonl"


def main(in_context_learning) -> None:
    results = {}
    if not("biogpt" in lang_model.lower()):
        data = [json.loads(line) for line in open(dataset)]
        prompt_base = read_file(r"./Pipelines", "entailment_proofs_nlps_1-2s.txt")
        openai.api_key = api_key_openai

        results = read_dict(results, r"C:\Users\georg_mosh.IPL2-PC\Documents\LLMs_InContextLearning", "temp.txt")

        for sample in tqdm(data):
            if not sample['id'] in results.keys():
                context = re.sub(r"sent(?=\d+)", "\nsent", sample["context"])
                prompt = f"Hypothesis: {sample['hypothesis']}\nContext:{context}\nProof:".strip()
                if in_context_learning: prompt = prompt_base + "\n\n" + prompt
                response = None
                while response is None:
                    try:
                        response = openai.Completion.create(
                            model=lang_model,
                            prompt=prompt,
                            temperature=0,
                            max_tokens=256,
                            stop="-> hypothesis;",
                        )
                    except Exception as e:
                        continue
                sleep(15)
                proof = (response["choices"][0]["text"] + "-> hypothesis;").strip()
                results[sample['id']] = proof
                print(f"$proof$ = {proof}")

        write_dict(results, r"C:\Users\georg_mosh.IPL2-PC\Documents\LLMs_InContextLearning", "results_1rs_" + lang_model)

    else:
        data = [json.loads(line) for line in open(dataset)][0]
        prompt_base = read_file(r"./Pipelines", "entailment_proofs_long.txt")
        model = TransformerLanguageModel.from_pretrained(
            lang_model,
            lang_model_checkpoint,
            "data",
            tokenizer='moses',
            bpe='fastbpe',
            bpe_codes="data/bpecodes",
            min_len=100,
            max_len_b=1024)

        # Check hardware accelerator availability
        if torch.cuda.is_available():
            model.cuda()

        for sample in tqdm(data):
            context = re.sub(r"sent(?=\d+)", "\nsent", sample["context"])
            sample = f"Hypothesis: {sample['hypothesis']}\nContext:{context}\nProof:".strip()
            prompt = prompt_base + "\n\n" + sample

            # Run inference using BioGPT by Mirosoft (local dump)
            src_tokens = model.encode(prompt)
            generate = model.generate([src_tokens], beam=lang_model_beam)[0]
            output = model.decode(generate[0]["tokens"])
        print(output)


if __name__ == "__main__":
    main(in_context_learning=True)
    x=1
