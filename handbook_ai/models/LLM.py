import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

class LLMUse:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    # Load model
    def load_llm(self, model_path):
        torch.manual_seed(1234)
        # Note: The default behavior now has injection attack prevention off.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # use cuda device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True,
                                                     use_safetensors=True).eval()

    def get_llm(self):
        return self.model, self.tokenizer

    # Use model
    def ask(self, prompt):

        query = self.tokenizer.from_list_format([
            {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
            {'text': prompt},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)

        return response