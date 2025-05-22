import torch
import requests
import logging
import asyncio
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from huggingface_hub import InferenceClient
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
import os
from tqdm import tqdm
from typing import Any
import re

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Summarizer:
    def __init__(self, llm_options=None, mode="api", provider=None, gpu_strategy="auto"):
        """Initialize the summarizer with specified model and mode.
        
        Args:
            llm_options (dict): Options from the YAML config including model names and parameters
            mode (str): Either "api" or "local"
            provider (str): Provider to use ("openai"/"huggingface"/"gemini" for API, 
                          "ollama"/"huggingface" for local)
            gpu_strategy (str): How to distribute model across devices (for local HF models)
        """
        llm_options = llm_options or {}
        api_opts = llm_options.get("api", {})
        
        self.model_name = api_opts.get("model", "gpt-4o-mini")
        self.mode = mode
        self.provider = provider or api_opts.get("provider", "openai")
        self.gpu_strategy = gpu_strategy
        self.max_length = 4096
        self.timeout = 300
        
        # Initialize based on mode and provider
        if mode == "api":
            if provider == "openai":
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info(f"Initialized OpenAI client for model: {self.model_name}")
            elif provider == "huggingface":
                self.client = InferenceClient(
                    token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
                    timeout=self.timeout
                )
                logger.info(f"Initialized Hugging Face client for model: {self.model_name}")
            elif provider == "gemini":
                if not os.getenv("GOOGLE_API_KEY"):
                    raise ValueError("GOOGLE_API_KEY environment variable not set")
                self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                
                # Store Gemini-specific configurations
                self.gemini_summary_model = api_opts.get("gemini_summary_model", "gemini-1.5-pro-latest")
                self.gemini_qa_model = api_opts.get("gemini_qa_model", "gemini-1.5-flash-latest")
                
                # Model parameters
                def validate_thinking_budget(value, default, model_name):
                    """Validate and clamp thinking_budget to valid range (0-24576) for Gemini 2.5 models"""
                    # Only apply thinking budget for Gemini 2.5 models
                    if not model_name or not "gemini-2.5" in model_name:
                        return None
                    try:
                        budget = int(value if value is not None else default)
                        return max(0, min(24576, budget))
                    except (ValueError, TypeError):
                        return default

                # Model parameters
                self.gemini_summary_config = {
                    "temperature": api_opts.get("gemini_summary_temperature", 0.3),
                    "max_output_tokens": api_opts.get("gemini_summary_max_tokens", 2000),
                    "candidate_count": 1,
                }
                
                self.gemini_qa_config = {
                    "temperature": api_opts.get("gemini_qa_temperature", 0.1),
                    "max_output_tokens": api_opts.get("gemini_qa_max_tokens", 1000),
                    "candidate_count": 1,
                }

                # Add thinking budget only for 2.5 models
                summary_budget = validate_thinking_budget(
                    api_opts.get("gemini_summary_thinking_budget"), 
                    1000, 
                    self.gemini_summary_model
                )
                if summary_budget is not None:
                    self.gemini_summary_config["thinking_budget"] = summary_budget

                qa_budget = validate_thinking_budget(
                    api_opts.get("gemini_qa_thinking_budget"), 
                    100, 
                    self.gemini_qa_model
                )
                if qa_budget is not None:
                    self.gemini_qa_config["thinking_budget"] = qa_budget

                logger.info(f"Initialized Gemini client with models: {self.gemini_summary_model} (summary), {self.gemini_qa_model} (QA)")
        else:  # local mode
            if provider == "ollama":
                # Keep the full model identifier for Ollama
                # Don't strip the namespace
                
                self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
                try:
                    response = requests.get(f"{self.ollama_host}/api/tags")
                    if response.status_code != 200:
                        raise ConnectionError("Ollama service not responding")
                    
                    # Check if model exists locally
                    models = response.json().get("models", [])
                    model_exists = any(m.get("name") == self.model_name for m in models)
                    
                    if not model_exists:
                        print(f"Model {self.model_name} not found locally. Pulling from Ollama...")
                        pull_response = requests.post(
                            f"{self.ollama_host}/api/pull",
                            json={"name": self.model_name}
                        )
                        if pull_response.status_code != 200:
                            raise ConnectionError(
                                f"Failed to pull model {self.model_name}. "
                                f"Available models: {[m.get('name') for m in models]}"
                            )
                        print(f"Successfully pulled model: {self.model_name}")
                    
                    # Verify the model is ready for generation
                    test_response = requests.post(
                        f"{self.ollama_host}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": "test",
                            "stream": False
                        }
                    )
                    if test_response.status_code != 200:
                        raise ConnectionError(f"Model {self.model_name} not ready for generation")
                    
                    print(f"Connected to Ollama at {self.ollama_host}, using model: {self.model_name}")
                except Exception as e:
                    raise ConnectionError(f"Failed to connect to Ollama or pull model: {str(e)}")
            elif provider == "huggingface":
                print(f"Loading local HF model {self.model_name} on GPU with strategy: {gpu_strategy}...")
                self._initialize_hf_model()

    def generate_summary(self, context, query):
        """Generate a two-stage summary based on the given context and query.
        
        Args:
            context (str): The context or source text to summarize.
            query (str): The query to focus the summary on.
        
        Returns:
            str: The final generated summary.
        """
        print(f"\nGenerating summary for query: {query}... using: \n"
              f"- Model: {self.model_name} \n"
              f"- Provider: {self.provider} \n"
              f"- Mode: {self.mode}")
        
        with tqdm(total=2, desc="Generation progress", unit=" steps") as pbar:
            # First stage
            first_summary = self._generate_text(
                self._get_first_stage_prompt(context, query),
                self._get_first_stage_system_prompt()
            )
            pbar.update(1)
            
            # Second stage
            final_summary = self._generate_text(
                self._get_second_stage_prompt(first_summary, context),
                self._get_second_stage_system_prompt()
            )
            pbar.update(1)
        
        return final_summary

    def _generate_text(self, user_prompt, system_prompt):
        """Generate text using configured provider."""
        try:
            if self.mode == "api":
                return self._generate_api(user_prompt, system_prompt)
            else:
                if self.provider == "ollama":
                    return self._generate_ollama(user_prompt, system_prompt)
                else:  # local huggingface
                    return self._generate_hf(user_prompt, system_prompt)
        except Exception as e:
            print(f"\nError during generation: {str(e)}")
            raise

    def _generate_api(self, prompt, system_prompt):
        """Generate text using the selected API.
        
        Args:
            prompt (str): The user prompt
            system_prompt (str): The system prompt defining the assistant's role
        """
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.3,
                    top_p=0.8,
                    frequency_penalty=0.8,
                    presence_penalty=0.6,
                    stop=["Summary:", "Assistant:", "Model:"]
                )
                return response.choices[0].message.content
            elif self.provider == "gemini":
                return self._generate_gemini(prompt, system_prompt, is_summary=True)
            else:  # huggingface
                # Hugging Face API with system prompt included in prompt
                full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
                response = self.client.text_generation(
                    prompt=full_prompt,
                    model=self.model_name,
                    max_new_tokens=1500,
                    temperature=0.3,
                    top_p=0.8,
                    repetition_penalty=1.2
                )
                return self._process_response(response)
        except Exception as e:
            logger.error(f"\n{self.provider} API Error: {str(e)}")
            raise

    def _generate_ollama(self, prompt, system_prompt):
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "repeat_penalty": 1.2,
                        "num_predict": 1500
                    }
                }
            )
            response.raise_for_status()
            return self._process_response(response.json()["response"])
        except Exception as e:
            print(f"\nOllama Error: {str(e)}")
            raise

    def _generate_hf(self, prompt):
        """Generate text using local GPU with optimizations."""
        
        # Create inputs and move to GPU
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Move input tensors to GPU
        input_ids = inputs['input_ids'].to('cuda:0')
        attention_mask = inputs['attention_mask'].to('cuda:0')
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,  # Enable sampling for temperature and top_p
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        return self._process_response(
            self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        )

    def _process_response(self, text):
        """Process the generated response."""
        summary = text.split("Summary:")[-1].strip()
        if summary.lower().startswith("assistant"):
            summary = summary[len("assistant"):].lstrip()
        elif summary.lower().startswith("model"):
            summary = summary[len("model"):].lstrip()
        return summary

    async def _call_gemini_api(self, contents: list, model_id: str, config: dict, max_retries: int = 3, base_delay: float = 2.0, max_delay: float = 30.0) -> Any:
        """
        Call Gemini API with exponential backoff retry logic.
        
        Args:
            contents: List of content parts for the Gemini API
            model_id: The Gemini model ID to use
            config: Dictionary of generation parameters
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            
        Returns:
            Gemini API response
            
        Raises:
            Exception: If all retry attempts fail
        """
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                logger.info(f"Calling Gemini API (attempt {retry_count + 1}/{max_retries})")
                
                # Make the API call
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        temperature=config.get('temperature', 0.1),
                        max_output_tokens=config.get('max_output_tokens', 2000),
                        candidate_count=config.get('candidate_count', 1),
                        thinking_config=genai_types.ThinkingConfig(
                            thinking_budget=config.get('thinking_budget', 0)
                        ) if config.get('thinking_budget') is not None else None
                    )
                )
                
                logger.info(f"Gemini API call successful using model {model_id}")
                return response
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                if retry_count == max_retries:
                    logger.error(f"Failed after {max_retries} attempts. Last error: {str(e)}")
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)
                
                # Add some jitter to prevent thundering herd
                jitter = random.uniform(0, 0.1 * delay)
                total_delay = delay + jitter
                
                logger.warning(f"Attempt {retry_count} failed. Retrying in {total_delay:.2f} seconds... Error: {str(e)}")
                await asyncio.sleep(total_delay)
        
        # If loop completes without success, raise the last error encountered
        logger.error(f"Gemini API call failed after {max_retries} attempts.")
        raise last_error

    def _generate_gemini(self, prompt: str, system_prompt: str, is_summary: bool = True) -> str:
        """
        Generate text using Gemini API.
        
        Args:
            prompt: The user prompt
            system_prompt: The system prompt
            is_summary: Whether this is a summary generation (True) or QA (False)
            
        Returns:
            Generated text
        """
        # Select appropriate model and config based on task
        if is_summary:
            model_id = self.gemini_summary_model
            config = self.gemini_summary_config
        else:
            model_id = self.gemini_qa_model
            config = self.gemini_qa_config
            
        # Call the API with combined prompt
        try:
            # Combine system prompt and user prompt
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = asyncio.run(self._call_gemini_api(
                contents=[full_prompt],
                model_id=model_id,
                config=config
            ))
            
            # Extract the generated text from the response
            if response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("No response candidates received from Gemini API")
                
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {str(e)}")
            raise

    def _get_first_stage_prompt(self, context, query):

        example_paragraph = (
            "Recent advances in machine learning have revolutionized oceanographic research. \"Deep learning models have "
            "enabled unprecedented accuracy in predicting ocean temperature patterns\" [1], while \"satellite data combined "
            "with neural networks has improved our understanding of global ocean circulation\" [2]. These technological "
            "breakthroughs have led to \"more precise forecasting of extreme weather events and their impacts on marine ecosystems\" [3]."
        )

        """Get the first stage prompt."""
        return (
            f"Write a flowing, paragraph-based summary addressing this query: '{query}'\n\n"
            f"Important Instructions:\n"
            f"- Use ONLY the information from the provided context\n"
            f"- Quote directly from the sources using double quotation marks\n"
            f"- Include citation numbers [n] immediately after each quote\n"
            f"- Write in clear paragraphs WITHOUT any headings or sections\n"
            f"- Make the text flow naturally from one topic to the next\n"
            f"- Use no more than 3 citations per sentence\n"
            f"- Ensure citations are relevant to the discussion\n"
            f"- Do NOT use bullet points or numbered lists in your response\n"
            f"- Do NOT refer to the sources in the context as texts, just write the summary without mentioning the context\n"
            f"- Write ONLY in connected paragraphs\n\n"
            f"Context:\n{context}\n\n"
            f"Example paragraph:\n{example_paragraph}\n\n"
            f"Remember: Your response should be ONLY in paragraph form, with no headings, sections, or bullet points, and it is IMPERATIVE that you quote directly from the context using proper citation numbers.\n"
            f"Summary:\n"
        )

    def _get_second_stage_prompt(self, first_summary, context):
        """Get the second stage prompt."""
        return (
            f"Refine this summary into polished, flowing paragraphs.\n\n"
            f"Important Instructions:\n"
            f"- Maintain the narrative structure\n"
            f"- Ensure all quotes have double quotation marks\n"
            f"- Verify citations are in [n] format\n"
            f"- Remove ANY headings, sections, or bullet points\n"
            f"- Create smooth transitions between paragraphs\n"
            f"- Write ONLY in connected paragraphs\n"
            f"- Remove any artificial breaks or formatting\n\n"
            f"Original Summary:\n{first_summary}\n\n"
            f"Context:\n{context}\n\n"
            f"Remember: Your response should be ONLY in paragraph form, with no headings, sections, or bullet points."
        )

    def _get_first_stage_system_prompt(self):
        """Get the system prompt for the first stage."""
        return """You are an AI assistant tasked with writing coherent, flowing summaries of scientific papers in paragraph form. Your output must be in clear paragraphs with no headings, sections, or bullet points.

        Your summaries should:
        1. Be written in clear, connected paragraphs without any headings, bullet points, or section breaks
        2. Directly quote relevant passages from the source texts, enclosing sthem in double quotation marks
        3. Include citation numbers (e.g., [1]) immediately after quotation marks
        4. Incorporate quotes and citations naturally into the paragraph flow
        5. Focus only on information relevant to the given query
        6. Be concise and informative
        7. Use only the provided citation numbers, not author names
        8. Flow naturally from one topic to the next without artificial breaks or sections

        Remember: The output should be a flowing narrative with NO headings, sections, or bullet points. Just clean, connected paragraphs."""

    def _get_second_stage_system_prompt(self):
        """Get the system prompt for the second stage."""
        return """You are an AI assistant tasked with refining scientific paper summaries into polished, flowing paragraphs. Your output must be in clear paragraphs with no headings, sections, or bullet points.

        Your task is to:
        1. Ensure the text flows naturally in paragraph form without any headings or sections
        2. Verify all citations are in the format [n] or [n, m, ...] and appear immediately after quotes
        3. Remove any author name citations (e.g., (John et al., 2024))
        4. Maintain proper paragraph structure and transitions
        5. Keep all quotes properly enclosed in double quotation marks
        6. Ensure the summary reads as a cohesive narrative rather than a sectioned report

        Remember: The output should be a flowing narrative with NO headings, sections, or bullet points. Just clean, connected paragraphs."""

    def _initialize_hf_model(self):
        """Initialize the Hugging Face model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        device_map = self._get_model_structure()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            offload_folder="offload"
        )
        print("Model loaded successfully with optimizations")

    def _get_model_structure(self):
        """Create a device map for model components based on strategy and available GPU memory."""
        config = AutoConfig.from_pretrained(self.model_name)
        
        if self.gpu_strategy == "full":
            return "cuda:0"  # Try to put everything on GPU
        
        # Check available GPU memory for auto and balanced strategies
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        available_memory = total_memory - reserved_memory
        
        # Base components always on GPU
        device_map = {
            "model.embed_tokens": "cuda:0",
            "model.norm": "cuda:0",
            "lm_head": "cuda:0",
            "model.rotary_emb": "cuda:0",
        }
        
        for i in range(config.num_hidden_layers):
            if self.gpu_strategy == "balanced":
                # Alternate between GPU and CPU
                device = "cuda:0" if i % 2 == 0 else "cpu"
            else:  # auto strategy
                # Calculate based on available memory
                memory_per_layer = 200 * 1024 * 1024
                num_layers_on_gpu = min(config.num_hidden_layers, available_memory // memory_per_layer)
                device = "cuda:0" if i < num_layers_on_gpu else "cpu"
            
            # Layer components list remains the same as before
            layer_components = [
                # Reference existing layer components
                f"model.layers.{i}.self_attn.q_proj",
                f"model.layers.{i}.self_attn.k_proj",
                f"model.layers.{i}.self_attn.v_proj",
                f"model.layers.{i}.self_attn.o_proj",
                f"model.layers.{i}.self_attn.rotary_emb",
                
                # MLP
                f"model.layers.{i}.mlp",
                f"model.layers.{i}.mlp.gate_proj",
                f"model.layers.{i}.mlp.up_proj",
                f"model.layers.{i}.mlp.down_proj",
                
                # Layer norms
                f"model.layers.{i}.input_layernorm",
                f"model.layers.{i}.post_attention_layernorm",
                f"model.layers.{i}.pre_feedforward_layernorm",
                f"model.layers.{i}.post_feedforward_layernorm",
            ]
            
            # Assign device to all components
            for component in layer_components:
                device_map[component] = device
        
        return device_map

    def query(self, queries: list, texts: list) -> list:
        """
        Query texts for relevant content using Gemini.
        
        Args:
            queries: List of queries to run
            texts: List of texts to search through
            
        Returns:
            List of lists containing (score, text, index) tuples for each query
        """
        results = []
        for query in queries:
            # Build prompt for finding relevant text
            prompt = f"""
            Find the most relevant text segments that answer or relate to this query: "{query}"
            
            Only return text segments that are DIRECTLY relevant. Rank them by relevance.
            If no text segments are relevant, return an empty list.
            
            Text to search:
            {' '.join(texts)}
            """
            
            try:
                # Generate response using Gemini
                response = self._generate_gemini(prompt, "", is_summary=False)
                
                # Extract text segments from response
                segments = []
                current_text = ""
                
                for line in response.split('\n'):
                    line = line.strip()
                    if line and not line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                        current_text += " " + line
                    elif current_text:
                        # Find this text in the original texts
                        for i, text in enumerate(texts):
                            if current_text.strip() in text:
                                segments.append((1.0, current_text.strip(), i))
                                break
                        current_text = ""
                
                # Add any remaining text
                if current_text:
                    for i, text in enumerate(texts):
                        if current_text.strip() in text:
                            segments.append((1.0, current_text.strip(), i))
                            break
                
                results.append(segments[:3])  # Return top 3 most relevant segments
                
            except Exception as e:
                logger.error(f"Error in Gemini query: {str(e)}")
                results.append([])
                
        return results

    def __call__(self, extractions: list, texts: list) -> list:
        """
        Extract answers from texts using Gemini.
        
        Args:
            extractions: List of tuples containing extraction parameters
            texts: List of texts to extract from
            
        Returns:
            List of (name, answer) tuples
        """
        results = []
        for extraction in extractions:
            try:
                # Unpack the extraction tuple, with defaults for optional parameters
                name = extraction[0]
                query = extraction[1]
                question = extraction[2] if len(extraction) > 2 else query
                dtype = extraction[3] if len(extraction) > 3 else None
                
                # Build prompt combining query and question
                prompt = f"""
                Based on this query: "{query}"
                Please answer this specific question: "{question}"
                
                Only use information from the following text:
                {' '.join(texts)}
                
                Provide a direct, concise answer based solely on the text provided.
                If the answer cannot be found in the text, say "Not found in text."
                """
                
                # Generate answer using Gemini
                answer = self._generate_gemini(prompt, "", is_summary=False)
                
                # Format the answer based on dtype if specified
                if dtype == "int":
                    try:
                        # Extract first number from the answer
                        numbers = re.findall(r'\d+', answer)
                        answer = numbers[0] if numbers else answer
                    except:
                        pass
                elif isinstance(dtype, list):
                    # For categorical data, try to match to allowed values
                    answer = answer.lower()
                    for category in dtype:
                        if category.lower() in answer:
                            answer = category
                            break
                elif dtype in ["days", "weeks", "months", "years"]:
                    # For duration data, try to normalize to the specified unit
                    try:
                        # Extract number and unit from answer
                        match = re.search(r'(\d+)\s*(day|week|month|year)s?', answer.lower())
                        if match:
                            num, unit = match.groups()
                            answer = f"{num} {unit}s"
                    except:
                        pass
                
                results.append((name, answer.strip()))
                
            except Exception as e:
                logger.error(f"Error in Gemini extraction: {str(e)}")
                results.append((name, ""))
                
        return results

