import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from datetime import datetime


class TeeLogger:

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# Start logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"benchmark_results_{timestamp}.txt"
logger = TeeLogger(log_filename)
sys.stdout = logger

print(f"Logging started: {log_filename}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# MODEL SETUP
# ============================================================================
MODEL_PATH = "./Noeum-0.6B-hf-nano"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nLoading model from {MODEL_PATH} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

EOS_ID = tokenizer.eos_token_id


def streaming_generate(model, prompt_ids, max_new_tokens=512, temperature=0.7, top_p=0.9, device="cuda"):
    input_ids = prompt_ids.to(device)

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]

            # Greedy decoding
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # Top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Set removed tokens to -inf
                    sorted_logits[sorted_indices_to_remove] = -float("inf")
                    # Scatter back to original positions
                    logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

        # Decode token
        tok_id = int(next_token.item())
        token_text = tokenizer.decode([tok_id], skip_special_tokens=False)
        yield token_text

        # Stop if EOS token
        if tok_id == EOS_ID:
            break

        # Append to input for next iteration
        input_ids = torch.cat([input_ids, next_token], dim=-1)


def chat(
        question: str,
        # chat_history argument removed/ignored
        thinking: bool = True,
        think_budget: int = 128,
        temperature: float = 0.1,
        top_p: float = 0.9,
        system_prompt: str = "You are a helpful assistant.",
        verbose: bool = True
):
    """
    Main chat function with streaming support.
    MEMORY DISABLED: Each call is a fresh context.
    """

    # Build conversation - STRICTLY SYSTEM + CURRENT QUESTION
    messages = [{'role': 'system', 'content': system_prompt}]

    # Add current question with /think flag if enabled
    user_message = f"{question} /think" if thinking else question
    messages.append({'role': 'user', 'content': user_message})

    # Apply chat template
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    prompt_ids = tokenizer.encode(prompt_text, return_tensors='pt')

    # Generate
    thinking_content = ""
    answer_content = ""
    current_mode = None

    generator = streaming_generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=think_budget + 256 if thinking else 512,
        temperature=temperature,
        top_p=top_p,
        device=DEVICE
    )

    if verbose:
        print("\n" + "=" * 80)

    for chunk in generator:
        if chunk == '</s>':
            break

        # Track mode switches
        if chunk == '<think>':
            current_mode = 'thinking'
            if verbose:
                print("\n💭 THINKING:")
            continue
        elif chunk == '</think>':
            current_mode = None
            continue
        elif chunk == '<answer>':
            current_mode = 'answer'
            if verbose:
                print("\n✅ ANSWER:")
            continue
        elif chunk == '</answer>':
            current_mode = None
            continue

        # Accumulate content
        if current_mode == 'thinking':
            thinking_content += chunk
            if verbose:
                print(chunk, end='', flush=True)
        elif current_mode == 'answer':
            answer_content += chunk
            if verbose:
                print(chunk, end='', flush=True)

    if verbose:
        print("\n" + "=" * 80)

    return {
        'thinking': thinking_content.strip(),
        'answer': answer_content.strip(),
        'full_thinking': thinking_content.strip(),
        'full_answer': answer_content.strip()
    }


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================
def benchmark_single_question(question: str, temperatures=[0.1, 0.3, 0.7], budgets=[32, 128, 256]):
    """
    Run a single question through all configurations - STREAMING
    """
    print("\n" + "=" * 100)
    print(f"QUESTION: {question}")
    print("=" * 100)

    # NO THINK
    print("\n🚫 NO THINK MODE (temperature=0.7)")
    print("-" * 100)
    result = chat(question, thinking=False, temperature=0.7, verbose=True)

    # THINK MODE - Different temperatures
    for temp in temperatures:
        print(f"\n💭 THINK MODE - Temperature: {temp}, Budget: 128")
        print("-" * 100)
        result = chat(question, thinking=True, think_budget=128, temperature=temp, verbose=True)

    # THINK MODE - Different budgets (at temp=0.7)
    for budget in budgets:
        print(f"\n💭 THINK MODE - Temperature: 0.7, Budget: {budget}")
        print("-" * 100)
        result = chat(question, thinking=True, think_budget=budget, temperature=0.7, verbose=True)


def benchmark_all_questions(questions: list, config: dict):
    results = []

    print("\n" + "=" * 100)
    print(f"BENCHMARK: {config}")
    print("=" * 100)

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Q: {q}")
        result = chat(
            q,
            thinking=config.get('thinking', True),
            think_budget=config.get('think_budget', 128),
            temperature=config.get('temperature', 0.1),
            verbose=True
        )

        results.append({
            'question': q,
            'thinking': result['full_thinking'],
            'answer': result['full_answer']
        })

    return results


def compare_configurations(questions: list):
    configurations = [
        {'name': 'NO THINK', 'thinking': False, 'temperature': 0.7},
        {'name': 'THINK - Temp 0.1', 'thinking': True, 'think_budget': 128, 'temperature': 0.1},
        {'name': 'THINK - Temp 0.3', 'thinking': True, 'think_budget': 128, 'temperature': 0.3},
        {'name': 'THINK - Temp 0.7', 'thinking': True, 'think_budget': 128, 'temperature': 0.7},
        {'name': 'THINK - Budget 32', 'thinking': True, 'think_budget': 32, 'temperature': 0.7},
        {'name': 'THINK - Budget 256', 'thinking': True, 'think_budget': 256, 'temperature': 0.7},
    ]

    all_results = {}

    for config in configurations:
        name = config.pop('name')
        print(f"\n{'#' * 100}")
        print(f"RUNNING CONFIGURATION: {name}")
        print(f"{'#' * 100}")
        all_results[name] = benchmark_all_questions(questions, config)

    # Print FULL comparison table
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY - FULL OUTPUTS")
    print("=" * 100)

    for i, q in enumerate(questions):
        print(f"\n{'=' * 100}")
        print(f"Q{i + 1}: {q}")
        print(f"{'=' * 100}")

        for config_name, results in all_results.items():
            print(f"\n{config_name}:")
            print(f"  💭 THINKING: {results[i]['thinking']}")
            print(f"  ✅ ANSWER: {results[i]['answer']}")
            print("-" * 100)


# ============================================================================
# INTERACTIVE CHAT LOOP
# ============================================================================
def interactive_chat():
    """
    Interactive chat session in terminal - STATELESS (No Memory)
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE CHAT SESSION (NO MEMORY)")
    print("=" * 80)
    print("Commands:")
    print("  /quit - Exit chat")
    print("  /think on/off - Toggle thinking mode")
    print("  /budget <number> - Set thinking budget (tokens)")
    print("  /temp <number> - Set temperature (0-1)")
    print("=" * 80 + "\n")

    # chat_history removed
    thinking_enabled = True
    think_budget = 128
    temperature = 0.1

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input == '/quit':
                print("Goodbye!")
                break
            elif user_input.startswith('/think'):
                parts = user_input.split()
                if len(parts) > 1:
                    thinking_enabled = parts[1].lower() == 'on'
                print(f"Thinking mode: {'ON' if thinking_enabled else 'OFF'}")
                continue
            elif user_input.startswith('/budget'):
                parts = user_input.split()
                if len(parts) > 1:
                    think_budget = int(parts[1])
                print(f"Thinking budget set to: {think_budget} tokens")
                continue
            elif user_input.startswith('/temp'):
                parts = user_input.split()
                if len(parts) > 1:
                    temperature = float(parts[1])
                print(f"Temperature set to: {temperature}")
                continue

            # /clear command removed as there is no memory

            # Get response - NO HISTORY PASSED
            result = chat(
                question=user_input,
                thinking=thinking_enabled,
                think_budget=think_budget,
                temperature=temperature,
                verbose=True
            )

            # Chat history update logic removed

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


# ============================================================================
# MAIN - RUN BENCHMARKS
# ============================================================================
if __name__ == '__main__':
    try:
        # Test questions
        test_questions = {
            # CATEGORY 1: Simple Math (Addition/Subtraction)
            "Simple Math": [
                "What is 15 + 27?",
                "What is 100 - 37?",
                "What is 45 + 55?",
                "What is 82 - 19?",
                "What is 7 + 8?",
                "What is 50 - 25?",
                "What is 123 + 456?",
                "What is 200 - 88?",
                "What is 9 + 6?",
                "What is 75 - 30?"
            ],

            # CATEGORY 2: Multiplication
            "Multiplication": [
                "What is 8 × 7?",
                "What is 12 × 12?",
                "What is 9 × 6?",
                "What is 15 × 4?",
                "What is 7 × 9?",
                "What is 11 × 11?",
                "What is 6 × 8?",
                "What is 13 × 5?",
                "What is 25 × 4?",
                "What is 20 × 3?"
            ],

            # CATEGORY 3: Division & Fractions
            "Division & Fractions": [
                "What is 56 ÷ 8?",
                "Which is larger: 1/2 or 1/3?",
                "What is 100 ÷ 4?",
                "Which is larger: 2/3 or 3/4?",
                "What is 81 ÷ 9?",
                "Which is larger: 1/4 or 1/5?",
                "What is 144 ÷ 12?",
                "What is 1/2 + 1/4?",
                "What is 50 ÷ 2?",
                "Which is larger: 3/5 or 2/5?"
            ],

            # CATEGORY 4: Word Problems
            "Word Problems": [
                "If a train travels 60 km in 1 hour, how far in 3 hours?",
                "If John has 5 apples and gives 2 to Mary, how many does he have left?",
                "If a book costs $12 and you buy 3 books, how much do you spend?",
                "If there are 24 students and 6 tables, how many students per table?",
                "If a car uses 8 liters per 100km, how much for 300km?",
                "If you earn $15 per hour and work 8 hours, how much do you earn?",
                "If a pizza has 8 slices and 4 people share equally, how many slices each?",
                "If a pen costs $2 and you have $20, how many pens can you buy?",
                "If a movie is 2 hours long and starts at 3pm, when does it end?",
                "If you save $10 per week, how much in 10 weeks?"
            ],

            # CATEGORY 5: Prime Numbers & Math Concepts
            "Math Concepts": [
                "Is 17 a prime number?",
                "Is 20 a prime number?",
                "Is 13 a prime number?",
                "What is the square root of 64?",
                "What is 5 squared (5²)?",
                "Is 1 a prime number?",
                "What is 10% of 100?",
                "Is 2 the only even prime number?",
                "What is the square root of 49?",
                "What is 3 cubed (3³)?"
            ],

            # CATEGORY 6: History
            "History": [
                "Who wrote Romeo and Juliet?",
                "Who was the first president of the United States?",
                "In what year did World War 2 end?",
                "Who discovered America in 1492?",
                "Who painted the Mona Lisa?",
                "What year did the Titanic sink?",
                "Who was the first man on the moon?",
                "In what year did World War 1 start?",
                "Who wrote the Declaration of Independence?",
                "Who was Julius Caesar?"
            ],

            # CATEGORY 7: Geography
            "Geography": [
                "What is the capital of France?",
                "Which is bigger: Russia or Canada?",
                "What is the capital of Italy?",
                "Is the Nile or Amazon river longer?",
                "Which ocean is largest: Atlantic or Pacific?",
                "What is the capital of Japan?",
                "Is Australia a continent?",
                "What is the tallest mountain in the world?",
                "How many continents are there?",
                "What is the capital of Spain?"
            ],

            # CATEGORY 8: Science & Nature
            "Science & Nature": [
                "Does the Sun orbit the Earth?",
                "What gas do plants produce during photosynthesis?",
                "Is iron magnetic?",
                "What is H2O?",
                "Does ice float in water?",
                "How many legs does a spider have?",
                "How many legs does an ant have?",
                "What is the speed of light approximately?",
                "Is the Earth flat or round?",
                "What planet is closest to the Sun?"
            ],

            # CATEGORY 9: Logical Reasoning
            "Logical Reasoning": [
                "If all cats are animals, and Fluffy is a cat, is Fluffy an animal?",
                "If today is Monday, what day is tomorrow?",
                "If you have 3 red balls and 2 blue balls, how many balls total?",
                "Complete the pattern: 2, 4, 6, 8, ?",
                "Which is the odd one out: apple, banana, car, orange?",
                "If A is taller than B, and B is taller than C, who is tallest?",
                "True or False: All birds can fly?",
                "If it's raining, the ground is wet. The ground is wet. Is it raining?",
                "Complete the pattern: Monday, Tuesday, Wednesday, ?",
                "If 5 > 3 and 3 > 1, is 5 > 1?"
            ],

            # CATEGORY 10: General Knowledge
            "General Knowledge": [
                "How many days are in a week?",
                "Which is heavier: gold or aluminum?",
                "Is the Eiffel Tower in London?",
                "Is Bitcoin a cryptocurrency?",
                "How many hours are in a day?",
                "What color is the sky on a clear day?",
                "How many months are in a year?",
                "What is the freezing point of water in Celsius?",
                "How many wheels does a bicycle have?",
                "What is the boiling point of water in Celsius?"
            ]
        }

        # Flatten all questions for quick testing
        all_questions = []
        for category, questions in test_questions.items():
            all_questions.extend(questions)

        print(f"\nTotal questions: {len(all_questions)}")
        print(f"Categories: {len(test_questions)}")
        print(f"Questions per category: {len(test_questions['Simple Math'])}")

        # Choose what to run
        print("\n" + "=" * 80)
        print("SELECT BENCHMARK MODE")
        print("=" * 80)
        print("1. Simple test (original 3 questions with default settings)")
        print("2. Deep dive single question (all configs)")
        print("3. Compare all configurations (by category) - FULL OUTPUTS")
        print("4. Interactive chat (NO MEMORY)")
        print("=" * 80)

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            # Simple test
            print("\n" + "=" * 80)
            print("SIMPLE TEST")
            print("=" * 80)

            for q in list(test_questions.values())[0][:3]:
                print(f"\nQ: {q}")
                result = chat(q, thinking=True, think_budget=128, temperature=0.7, verbose=True)

        elif choice == '2':
            # Deep dive on single question
            question = input("\nEnter question (or press Enter for '15 + 27'): ").strip()
            if not question:
                question = "What is 15 + 27?"

            benchmark_single_question(question)

        elif choice == '3':
            # Test by category
            print("\nSelect category to test:")
            categories = list(test_questions.keys())
            for i, cat in enumerate(categories, 1):
                print(f"{i}. {cat}")
            print(f"{len(categories) + 1}. ALL CATEGORIES (100 questions)")

            cat_choice = input("\nEnter choice: ").strip()

            if cat_choice == str(len(categories) + 1):
                # Test all
                compare_configurations(all_questions)
            else:
                # Test specific category
                cat_idx = int(cat_choice) - 1
                cat_name = categories[cat_idx]
                print(f"\nTesting category: {cat_name}")
                compare_configurations(test_questions[cat_name])

        elif choice == '4':
            # Interactive chat
            interactive_chat()

        else:
            print("Invalid choice. Starting interactive chat...")
            interactive_chat()

    finally:
        print(f"\n\nLogging completed. Results saved to: {log_filename}")
        logger.close()
        sys.stdout = logger.terminal
