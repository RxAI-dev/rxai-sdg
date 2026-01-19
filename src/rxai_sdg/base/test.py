from datetime import datetime

from src.rxai_sdg.base import BaseDatasetGenerator
import os

ollama_api_key = os.environ.get("OLLAMA_API_KEY")


# ===== CONCRETE IMPLEMENTATION =====

class TestDatasetGenerator(BaseDatasetGenerator):
    """
    Concrete implementation for testing Ollama connection
    """

    def _init_items(self) -> dict[str, list]:
        """
        Initialize the dataset structure
        """
        return {
            "prompts": [],
            "responses": [],
            "timestamps": [],
            "model_used": []
        }

    def __call__(self, test_prompt: str = None, *args, **kwargs) -> None:
        """
        Main execution method
        """
        if test_prompt is None:
            test_prompt = "Hello! Please respond with a short greeting to confirm the connection is working."

        print(f"Testing connection to Ollama...")
        print(f"Model: {self.model_name}")
        # print(f"API URL: {self.client.host if self.use_ollama else self.client.base_url}")
        print("-" * 50)

        # Generate response
        response = self.generate_items(
            prompt=test_prompt,
            stream=True,  # Set to False if you don't want streaming
            temperature=0.7,
            system_prompt="You are a helpful assistant. Respond concisely.",
            timeout=30
        )

        # Store the result
        self.items["prompts"].append(test_prompt)
        self.items["responses"].append(response)
        self.items["timestamps"].append(datetime.now().isoformat())
        self.items["model_used"].append(self.model_name)

        print(f"\n\nResponse stored. Total items in dataset: {len(self.items['prompts'])}")


# ===== USAGE EXAMPLES =====

# Option 1: Connect to local Ollama (make sure Ollama is running)
def test_local_ollama():
    """Test with local Ollama installation"""
    generator = TestDatasetGenerator(
        max_items=10,
        model_name="llama2",  # Make sure this model is pulled: ollama pull llama2
        api_url="http://localhost:11434",  # Local Ollama
        api_key=None,  # Not needed for local Ollama
        use_ollama=True  # Use native Ollama API
    )

    # Run the test
    generator("Explain what machine learning is in one sentence.")

    # Get the dataset
    dataset = generator.get_dataset()
    print(f"\nDataset created with {len(dataset)} rows")
    return (dataset)


# Option 2: Connect to Ollama cloud with OpenAI-compatible API
def test_cloud_ollama():
    """Test with cloud service (OpenAI-compatible API)"""
    import os
    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file

    # Example using OpenRouter (supports many models)
    generator = TestDatasetGenerator(
        max_items=10,
        model_name="meta-llama/llama-3.1-8b-instruct",  # Model available on OpenRouter
        api_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),  # Set your API key in .env file
        use_ollama=False  # Use OpenAI-compatible API
    )

    # Run the test
    generator("What is artificial intelligence?")

    # Get the dataset
    dataset = generator.get_dataset()
    print(f"\nDataset created with {len(dataset)} rows")


# Option 3: Simple test with custom prompt
def simple_test():
    """Quick test with default settings"""
    # Create generator
    generator = TestDatasetGenerator(
        max_items=5,
        model_name="mistral",  # Try different models: llama2, mistral, codellama, etc.
        api_url="http://localhost:11434",
        api_key=None,
        use_ollama=True
    )

    # Test multiple prompts
    prompts = [
        "What is Python?",
        "Explain recursion with an example",
        "Tell me a joke about programming"
    ]

    for prompt in prompts:
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt}")
        print(f"{'=' * 60}")
        generator(prompt)

    # Display the collected dataset
    dataset = generator.get_dataset()
    print(f"\n\n{'=' * 60}")
    print("FINAL DATASET SUMMARY")
    print(f"{'=' * 60}")
    for i in range(len(dataset)):
        print(f"\nItem {i + 1}:")
        print(f"Prompt: {dataset['prompts'][i]}")
        print(f"Response: {dataset['responses'][i][:100]}...")  # First 100 chars
        print(f"Model: {dataset['model_used'][i]}")


# Option 4: Batch generation
class BatchDatasetGenerator(TestDatasetGenerator):
    """Extended generator for batch processing"""

    def generate_batch(self, prompts: list[str]):
        """Generate responses for multiple prompts"""
        for i, prompt in enumerate(prompts):
            if self.max_items and len(self.items["prompts"]) >= self.max_items:
                print(f"Reached maximum items ({self.max_items}). Stopping.")
                break

            print(f"\nProcessing prompt {i + 1}/{len(prompts)}")
            self(prompt)

            # Small delay to avoid rate limiting
            import time
            time.sleep(0.5)

        print(f"\nBatch generation complete. Total items: {len(self.items['prompts'])}")
        return self.get_dataset()


if __name__ == "__main__":
    print("Select test option:")
    print("1. Test local Ollama")
    print("2. Test cloud Ollama (requires API key)")
    print("3. Simple multi-prompt test")
    print("4. Batch generation test")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        test_local_ollama()
    elif choice == "2":
        test_cloud_ollama()
    elif choice == "3":
        simple_test()
    elif choice == "4":
        # Batch test
        generator = BatchDatasetGenerator(
            max_items=5,
            model_name="llama2",
            api_url="http://localhost:11434",
            use_ollama=True
        )

        test_prompts = [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Explain the concept of blockchain",
            "What are the benefits of exercise?",
            "Describe the water cycle"
        ]

        dataset = generator.generate_batch(test_prompts)
        print(f"\nGenerated dataset with {len(dataset)} examples")
    else:
        print("Invalid choice")
