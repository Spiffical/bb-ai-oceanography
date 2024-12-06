import unittest
import argparse
from paperai.report.summarizer import Summarizer
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Test the Summarizer with different configurations')
    parser.add_argument('--mode', choices=['api', 'local'], default='local',
                      help='Mode to run the summarizer in (api or local)')
    parser.add_argument('--provider', choices=['openai', 'huggingface', 'ollama'], default='ollama',
                      help='Provider to use for the model')
    parser.add_argument('--model', default='gemma:2b',
                      help='Model to use (e.g., gemma:2b for Ollama, gpt-4 for OpenAI)')
    parser.add_argument('--gpu-strategy', choices=['auto', 'full', 'balanced'], default='auto',
                      help='GPU strategy for local HuggingFace models')
    parser.add_argument('--test-method', choices=['basic', 'memory', 'stress'], default='basic',
                      help='Type of test to run')
    return parser.parse_args()

class TestSummarizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = parse_args()
        print(f"\nTest Configuration:")
        print(f"Mode: {cls.args.mode}")
        print(f"Provider: {cls.args.provider}")
        print(f"Model: {cls.args.model}")
        print(f"GPU Strategy: {cls.args.gpu_strategy}")
        print(f"Test Method: {cls.args.test_method}")
        
        if torch.cuda.is_available():
            print(f"\nGPU Information:")
            print(f"Device: {torch.cuda.get_device_name(0)}")
            print(f"Initial Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    def setUp(self):
        self.test_context = """[1] The ocean plays a crucial role in regulating Earth's climate. 
        It absorbs heat and CO2 from the atmosphere, which helps control global temperatures. 
        Recent studies show that ocean temperatures have increased significantly over the past century.
        [2] Rising ocean temperatures are causing coral bleaching events worldwide.
        When water temperatures rise, corals expel their symbiotic algae, leading to potential death.
        [3] Ocean acidification is another major concern. As CO2 levels rise in the atmosphere,
        more is absorbed by the oceans, making them more acidic. This affects marine organisms,
        particularly those that build shells and skeletons from calcium carbonate."""
        
        self.test_query = "How is climate change affecting the oceans?"
        
        self.summarizer = Summarizer(
            llm_name=self.args.model,
            mode=self.args.mode,
            provider=self.args.provider,
            gpu_strategy=self.args.gpu_strategy
        )

    def test_basic_summary(self):
        """Basic test of summary generation"""
        if self.args.test_method != 'basic' and self.args.test_method != 'all':
            self.skipTest("Skipping basic test")
            
        try:
            summary = self.summarizer.generate_summary(self.test_context, self.test_query)
            self.assertIsNotNone(summary)
            self.assertTrue(len(summary) > 0)
            print(f"\nGenerated Summary:")
            print(f"{summary}")
            print(f"\nSummary Length: {len(summary)} characters")
        except Exception as e:
            self.fail(f"Summary generation failed: {str(e)}")

    def test_memory_usage(self):
        """Test memory usage during summary generation"""
        if self.args.test_method != 'memory' and self.args.test_method != 'all':
            self.skipTest("Skipping memory test")
            
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
            
        try:
            initial_memory = torch.cuda.memory_allocated(0)
            summary = self.summarizer.generate_summary(self.test_context, self.test_query)
            peak_memory = torch.cuda.max_memory_allocated(0)
            final_memory = torch.cuda.memory_allocated(0)
            
            print(f"\nMemory Usage (MB):")
            print(f"Initial: {initial_memory / 1024**2:.2f}")
            print(f"Peak: {peak_memory / 1024**2:.2f}")
            print(f"Final: {final_memory / 1024**2:.2f}")
        except Exception as e:
            self.fail(f"Memory test failed: {str(e)}")

    def test_stress(self):
        """Stress test with multiple summaries"""
        if self.args.test_method != 'stress' and self.args.test_method != 'all':
            self.skipTest("Skipping stress test")
            
        try:
            for i in range(3):
                print(f"\nStress Test Iteration {i+1}")
                summary = self.summarizer.generate_summary(self.test_context, self.test_query)
                self.assertIsNotNone(summary)
                if torch.cuda.is_available():
                    print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        except Exception as e:
            self.fail(f"Stress test failed: {str(e)}")

    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\nFinal GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'])