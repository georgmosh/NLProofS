import os
import json
from tqdm import tqdm
from prover.datamodule import read_entailmentbank_proofs
from prover.evaluate import evaluate_entailmentbank


def read_file(src_directory, filename, encoding=None, log=True):
    if log: print("Reading file:", filename)
    with open(os.path.join(src_directory, filename), mode='r', encoding=encoding) as file:
        content = file.read()
    return content


def read_dict(directory, filename):
    with open(os.path.join(directory, filename)) as handle:
        data = handle.read()
    dictionary = json.loads(data)

    return dictionary


def write_dict(dict, directory, filename):
    with open(os.path.join(directory, filename), 'w') as convert_file:
        convert_file.write(json.dumps(dict))


# Set the corresponding directories
split = "val"
predictions_path = "/media/georg_mosh/Data SSD/AUEB PROOFTREE GENERATION/LLMs_InContextLearning"
val_set_path = "/media/georg_mosh/Data SSD/AUEB PROOFTREE GENERATION/NLProofS/data/entailment_trees_emnlp2021_data_v3/" \
               "dataset/task_2/dev.jsonl"

# Run the approximate evaluation
results = {}
data = read_entailmentbank_proofs(val_set_path, is_train=False)
experiments_names = os.listdir(predictions_path)
for experiment in tqdm(experiments_names):
    exp_results = []
    exp_responses = read_dict(predictions_path, experiment)
    for proof in data:
        exp_results.append({
            "proof_pred": exp_responses[proof['proof'].id],
            "score": 1.0,
            "hypothesis": proof['proof'].hypothesis,
            "context": proof['proof'].context,
            "proof_gt": proof['proof'].proof_text,
        })
    em, f1 = evaluate_entailmentbank(exp_results, eval_intermediates=False)

    # Print metrics
    print("\nExperiment: " + experiment + "\n")
    for k, v in em.items():
        print(f"ExactMatch_{k}_{split}", v)
    for k, v in f1.items():
        print(f"F1_{k}_{split}", v)

    # Save results
    results[experiment] = [em, f1]
write_dict(results[experiment], predictions_path, "evaluation.txt")
