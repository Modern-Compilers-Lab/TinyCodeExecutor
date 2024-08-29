import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import re
import pandas as pd
from model import GPT

class ScriptEvaluator:

    DATA_DIR = 'data/96M'
    MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results') 

    def __init__(self, model_file, test_data, meta):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(1337)
        self.test_data = test_data
        self.meta = meta
        self.model_file = model_file
        self.model = self.load_model(model_file)

    def load_model(self, model_file):
        model = GPT()
        print(f"Compiling model from {model_file}...")
        model = torch.compile(model)  
        model_path = os.path.join(self.MODELS_DIR, model_file)
        model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        return model

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def evaluate_example(self, example, max_new_tokens=22):
        splited_example = example.split("# output")
        if "for" in splited_example[0] or "while" in splited_example[0]:
            max_new_tokens = 30
        encoded_example = torch.tensor(self.encode(splited_example[0] + "# output"), dtype=torch.long).unsqueeze(0).to(self.device)

        prompt_text = splited_example[0] + "# output"
        result_example = splited_example[-1]

        real_results = [float(match.group()) for match in re.finditer(r"(?<=# )-?\d+(\.\d+)?", result_example.split('\n\n')[0].replace("\n", ""))]

        response = self.decode(self.model.generate(encoded_example, max_new_tokens=max_new_tokens)[0].tolist())
        splited_response = response.split("# output")
        result_response = splited_response[-1]
        generated_results = [float(match.group()) for match in re.finditer(r"(?<=# )-?\d+(\.\d+)?", result_response.split('\n\n')[0].replace("\n", ""))]

        return prompt_text, real_results, generated_results

    def write_results_to_file(self, output_file, prompt, real_results, generated_results):
        df = pd.DataFrame({
            'Prompt': prompt,
            'Real_Results': real_results,
            'Generated_Results': generated_results
        })
        df.to_csv(output_file, index=False)

    def main(self):
        # Extracting stoi and itos from meta
        self.stoi = self.meta['stoi']
        self.itos = self.meta['itos']

        examples = self.decode(self.test_data).split("\n\n")
        examples = [example for example in examples if example]

        prompt = []
        real_results = []
        generated_results = []

        for example in tqdm(examples):
            prompt_text, real_result, result = self.evaluate_example(example)
            prompt.append(prompt_text)
            real_results.append(real_result)
            generated_results.append(result)

        correct_count = sum(1 for real, generated in zip(real_results, generated_results) if real == generated)
        accuracy = correct_count / len(generated_results)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Extracting model size and data size from the filename
        model_size = self.model_file.split('_')[0]
        data_size = self.model_file.split('_')[1]

        # Store accuracy in a file
        accuracy_file = os.path.join(ScriptEvaluator.RESULTS_DIR, f'{model_size}_{data_size}_accuracy.txt')
        with open(accuracy_file, 'w') as f:
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

        # Store results in a CSV file
        results_file = os.path.join(ScriptEvaluator.RESULTS_DIR, f'{model_size}_{data_size}_results.csv')
        self.write_results_to_file(results_file, prompt, real_results, generated_results)

if __name__ == "__main__":
    # Load dataset and meta information once
    data_dir = ScriptEvaluator.DATA_DIR
    test_data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        print(f"Found vocab_size = {meta['vocab_size']} (inside {meta_path})")
    else:
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    # Iterate over all model files in the directory
    model_files = [f for f in os.listdir(ScriptEvaluator.MODELS_DIR) if f.endswith('.pth')]

    for model_file in model_files:
        evaluator = ScriptEvaluator(model_file, test_data, meta)
        evaluator.main()
