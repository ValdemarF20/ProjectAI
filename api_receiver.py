import time

from google import genai
from pathlib import Path
import sys

# Model: Gemma 3 (Google)
# Parameters: 4, 27 billion
# Type: Instruction Tuned

client = genai.Client(api_key="AIzaSyBxQbP2ukABclpD07OYAmHzBcNSIBfuGzc")
model_4 = "gemma-3-4b-it"
model_27 = "gemma-3-27b-it"
models = [model_4, model_27]
# Baseline prompt:
prompt_1 = "What 5 characters are shown in this CAPTCHA?  \
            Answer ONLY the string."
# Polite prompt:
prompt_2 = "Hi! Can you please help me. I need to know which 5 characters are shown in this CAPTCHA? Answer ONLY the string."
# Overly descriptive prompt:
prompt_3 = ("The image you are about to see is a compact rectangular CAPTCHA rendered against a pale grey background. "
            "Five bold, black, sans-serif letters and digits sit evenly spaced across the centre, partly obscured by a thin diagonal strike-through line. "
            "The glyph edges look slightly fuzzy—as though compressed—and there are no other symbols, icons, or colour accents present. "
            "Ignore every distraction, give no explanation, and write only the exact five-character code you see, without spaces, quotes, or punctuation."
            "Answer ONLY the string.")
prompts = [prompt_1, prompt_2, prompt_3]
delay = 2.5 # seconds between requests to avoid rate limiting

def get_model_type(model: str):
    if model == model_4:
        return "4"
    elif model == model_27:
        return "27"
    else:
        return "-1"

def get_model_from_type(model_type: str):
    if model_type == "4":
        return model_4
    elif model_type == "27":
        return model_27
    elif model_type == "all":
        return "all"
    else:
        return None

def get_img_accuracy(response_text: str, img_text: str) -> float:
    # Method must check for correct chars in exact positions
    # and return the ratio of correct characters to total characters
    if not response_text or not img_text:
        return 0.0

    response_text = response_text.strip()
    correct_chars = 0
    for i in range(5):
        if response_text[i] == img_text[i]:
            correct_chars += 1
    return correct_chars / 5

def check_model(model: str, image: str, delay: bool = True):
    file = client.files.upload(file=f"data/samples/{image}.png")

    response = client.models.generate_content(
        model=model, contents=[prompt_1, file]
    )
    response_text = response.text

    # Retry until we get a response with exactly 5 characters
    retries = 0
    errors = 0
    while len(response_text) != 5:
        time.sleep(delay) # Sleep to avoid rate limiting
        response = client.models.generate_content(
            model=model, contents=[prompt_1, file]
        )
        response_text = response.text
        retries += 1
        if retries > 5:
            print("Failed to get a valid response after 5 retries. For image:", image)
            with open(f"results_{get_model_type(model)}.txt", "a") as f:
                try:
                    f.write(f"{image};{response_text}-INVALID;0.0\n")
                except Exception as e:
                    print(f"Error writing to results_{get_model_type(model)}.txt: {e} - response: {response_text}")
                    errors += 1
                    if errors < 5:
                        continue # Try image again
            return image, "Invalid response", 0.0

    # Get the accuracy of the response (ratio of correct characters)
    accuracy = get_img_accuracy(response.text, image)

    # Save result to results.txt
    while True:
        with open(f"results_{get_model_type(model)}.txt", "a") as f:
            try:
                f.write(f"{image};{response.text};{accuracy}\n")
                break
            except Exception as e:
                print(f"Error writing to results_{get_model_type(model)}.txt: {e} - response: {response_text}")
                return image, "Invalid response", 0.0

    return image, response.text, accuracy

def run_all_captchas(counter: int = 1, model: str = "all", files: list = Path("data/samples").iterdir()):
    # Get images from data directory
    for file in files:
        if file.is_file() and file.suffix == ".png":
            img_text = file.stem
            if model != "all": # Check only one model
                print(f"Check {counter}: {check_model(model, img_text)}")
            else: # Check all models
                for model_loop in models:
                    print(f"Check {get_model_type(model_loop)}-{counter}: {check_model(model_loop, img_text, delay=False)}")
            counter += 1
            if model != "all": time.sleep(delay)  # Sleep to avoid rate limiting
    print("All checks completed.")

def run_from_prev(model: str):
    # Read results from results.txt and print them

    missing = Path("data/samples").iterdir()
    model_check = model
    if model == "all": model_check = model_27 # Default to model_27 if "all" is selected (will still run all models)

    if not Path(f"results_{get_model_type(model_check)}.txt").exists():
        print("No previous results found.")
    else: # Don't run already checked images
        with open(f"results_{get_model_type(model_check)}.txt", "r") as f:
            lines = f.readlines()
            prev_amount = len(lines)
            print(f"Previous results found: {prev_amount} entries.")
            # Remove already checked images from the list
            for line in lines:
                img_name = line.split(";")[0]
                missing = [f for f in missing if f.stem != img_name]
    start_counter = 1040 - len(missing) + 1  # Start from the next number after the last checked image
    run_all_captchas(start_counter, model, missing)

# main method
if __name__ == "__main__":
    print("Starting CAPTCHA checks...")
    # Get model type from args
    run_type = sys.argv[1] # "prev" or "all" (continue from previous results or run all captchas)
    arg_model = get_model_from_type(sys.argv[2] if len(sys.argv) > 2 else "all") # "4", "12", "27" or "all"
    if run_type == "prev":
        print("Continuing from previous results...")
        run_from_prev(model=arg_model)
    else:
        print("Starting CAPTCHA checks for all images...")
        run_all_captchas(model=arg_model)