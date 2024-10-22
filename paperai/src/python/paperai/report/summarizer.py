import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_cpp import Llama

class Summarizer:
    """
    A class for generating summaries of scientific papers using various language models.

    This class supports different types of models, including Gemma, GGUF models (Mistral and Ministral),
    and other AutoModelForCausalLM compatible models.

    Attributes:
        model_name (str): The name or path of the model to be used for summarization.
        max_length (int): The maximum length of input tokens.
        model: The loaded language model.
        tokenizer: The tokenizer for the model (if applicable).
        is_gemma (bool): Flag indicating if the model is a Gemma model.
        is_gguf (bool): Flag indicating if the model is a GGUF model.
    """

    def __init__(self, model_name):
        """
        Initialize the Summarizer with a specified model.

        Args:
            model_name (str): The name or path of the model to be used.
        """
        self.model_name = model_name
        self.max_length = 4096
        self._load_model()

    def _load_model(self):
        """
        Load the specified model based on its name or path.

        This method handles different model types (Gemma, GGUF, or other AutoModelForCausalLM models)
        and sets up the appropriate configurations.
        """
        if "google/gemma" in self.model_name:
            print(f"Loading Gemma model: {self.model_name}")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.is_gemma = True
            self.is_gguf = False
        elif "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF" in self.model_name:
            filename = "Mistral-7B-Instruct-v0.3.Q8_0.gguf" 
            print(f"Downloading model {filename} from Hugging Face...")
            self.model = Llama.from_pretrained(
                repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
                filename=filename,
                n_ctx=4096, 
                n_batch=512,
                n_gpu_layers=-1,
                verbose=False
            )
            self.is_gguf = True
            self.is_gemma = False
        elif "bartowski/Ministral-8B-Instruct-2410-GGUF" in self.model_name:
            print(f"Loading Ministral-8B-Instruct model...")
            self.model = Llama.from_pretrained(
                repo_id="bartowski/Ministral-8B-Instruct-2410-GGUF",
                filename="Ministral-8B-Instruct-2410-Q8_0.gguf",
                n_ctx=4096,
                n_batch=512,
                n_gpu_layers=-1,
                verbose=False
            )
            self.is_gguf = True
            self.is_gemma = False
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.is_gguf = False
            self.is_gemma = False

    def generate_summary(self, context, query):
        """
        Generate a two-stage summary based on the given context and query.

        Args:
            context (str): The context or source text to summarize.
            query (str): The query to focus the summary on.

        Returns:
            str: The final generated summary.
        """
        first_summary = self._generate_text(self._get_first_stage_prompt(context, query), self._get_first_stage_system_prompt())
        final_summary = self._generate_text(self._get_second_stage_prompt(first_summary, context), self._get_second_stage_system_prompt())
        
        return final_summary

    def _generate_text(self, user_prompt, system_prompt):
        """
        Generate text based on the given user and system prompts.

        This method handles text generation for different model types.

        Args:
            user_prompt (str): The prompt for the user's input.
            system_prompt (str): The system prompt (not used for Gemma models).

        Returns:
            str: The generated text summary.
        """
        if not self.is_gemma:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [{"role": "user", "content": user_prompt}]
        
        if self.is_gguf:
            output = self.model.create_chat_completion(
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
            )
            generated_text = output['choices'][0]['message']['content']
        elif self.is_gemma:
            input_text = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
            input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **input_ids,
                    max_new_tokens=1000,
                    min_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            inputs = self.tokenizer(self.tokenizer.apply_chat_template(messages, tokenize=False), 
                                    return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    min_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        summary = generated_text.split("Summary:")[-1].strip()
        if summary.lower().startswith("assistant"):
            summary = summary[len("assistant"):].lstrip()
        elif summary.lower().startswith("model"):
            summary = summary[len("model"):].lstrip()
        
        return summary

    def _get_first_stage_prompt(self, context, query):
        """
        Generate the prompt for the first stage of summarization.

        Args:
            context (str): The context or source text to summarize.
            query (str): The query to focus the summary on.

        Returns:
            str: The formatted prompt for the first stage of summarization.
        """
        return (
            f"Task: Generate a 1000 word summary based on the provided context, addressing the query: '{query}'\n\n"
            f"Instructions:\n"
            f"1. Use ONLY the information from the provided context.\n"
            f"2. Quote directly from the context, using double quotation marks.\n"
            f"3. Include the citation number (e.g., [1]) immediately after each quotation.\n"
            f"4. Incorporate quotes and citations seamlessly into your summary.\n"
            f"5. Focus on information relevant to the query.\n"
            f"6. Be concise and informative.\n"
            f"7. Do not mention that you are quoting or summarizing.\n"
            f"8. Do not include author names in citations, use only the provided numbers.\n"
            f"9. Try to tell a story with the information provided, using the context and query to provide details and information.\n"
            f"10: Use AT MOST 3 citations per quote, and ensure they are directly relevant to the quote and the query.\n"
            f"11: DO NOT write the references at the end of the summary.\n"
            f"12: You DO NOT need to cite the sources in order, but you should use them if they are relevant to the query.\n"
            f"13: Include the quotes naturally in the summary so that the summary flows well, and do not repeat them.\n\n"
            f"Context (Sources to quote from):\n{context}\n\n"
            f"Summary:"
        )

    def _get_second_stage_prompt(self, first_summary, context):
        """
        Generate the prompt for the second stage of summarization.

        Args:
            first_summary (str): The summary generated in the first stage.
            context (str): The original context or source text.

        Returns:
            str: The formatted prompt for the second stage of summarization.
        """
        return (
            f"Task: Refine the following summary, ensuring the information from each source is directly relevant to where it is used.\n\n"
            f"Original Summary: {first_summary}\n\n"
            f"Context (Sources to quote from):\n{context}\n\n"
            f"Instructions:\n"
            f"1. Maintain the overall structure and content of the summary.\n"
            f"2. Ensure all quotes are properly enclosed in double quotation marks.\n"
            f"3. Verify that all citations are in the format [n] or [n, m, ...], where n and m are numbers.\n"
            f"4. Remove any author name citations that may have been included.\n"
            f"5. Do not add any new information or change the meaning of the text.\n"
            f"6. Ensure that the summary ends with a conclusion.\n"
            f"7. Ensure that the summary contains NO notes, NO information irrelevant to the query or summary, and NO lists of the context provided.\n\n"
            f"Revised Summary:"
        )

    def _get_first_stage_system_prompt(self):
        """
        Get the system prompt for the first stage of summarization.

        Returns:
            str: The system prompt for the first stage.
        """
        return """You are an AI assistant tasked with summarizing scientific papers in around 1000 words. Your summaries should:
        1. Directly quote relevant passages from the source (context) texts, enclosing them in double quotation marks.
        2. MAKE SURE to include the citation number (e.g., [1]) immediately after the quotation marks.
        3. Incorporate these quotes and citations seamlessly into your summary.
        4. Do not mention that you are quoting or summarizing.
        5. Focus only on information relevant to the given query.
        6. Be concise and informative.
        7. Do not include author names in citations, use only the provided numbers."""

    def _get_second_stage_system_prompt(self):
        """
        Get the system prompt for the second stage of summarization.

        Returns:
            str: The system prompt for the second stage.
        """
        return """You are an AI assistant tasked with refining summaries of scientific papers. Your task is to:
        1. Ensure all citations are in the format [n] or [n, m, ...], where n and m are numbers.
        2. Remove any author name citations that may have been included (e.g., (John et al., 2024)).
        3. Maintain the overall structure and content of the summary.
        4. Ensure quotes are properly enclosed in double quotation marks.
        5. Do not add any new information or change the meaning of the text."""
