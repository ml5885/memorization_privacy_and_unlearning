"""
IFEval scoring following the original paper:

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu,
Yi Luan, Denny Zhou, Le Hou.
"Instruction-Following Evaluation for Large Language Models (IFEval)."
arXiv:2311.07911, 2023. https://arxiv.org/pdf/2311.07911
"""

from __future__ import annotations

import json
import re

from datasets import load_dataset

from models.gemma import generate_batch
from utils.part_1 import open_jsonl, write_jsonl

# Regular expressions for parsing
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
HIGHLIGHT_RE = re.compile(r"\*[^*\n]+\*")
TITLE_RE = re.compile(r"<<[^<>]+>>")
JSON_BLOCK_RE = re.compile(r"^\s*```(?:json)?\s*([\s\S]+?)\s*```\s*$", re.IGNORECASE)

def compare_count(count, relation, target):
    """Compare count against target using the specified relation."""
    relation = relation.strip().lower() if relation else "at least"
    
    if relation in {">=", "at least", "no less than"}:
        return count >= target
    elif relation in {"<", "less than", "fewer than"}:
        return count < target
    elif relation in {"=", "==", "exactly"}:
        return count == target
    
    return count >= target

def count_words(text):
    """Count the number of words in the text."""
    return len(WORD_RE.findall(text))

def split_sentences(text):
    """Split text into sentences."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [sentence for sentence in SENTENCE_SPLIT_RE.split(text) if sentence]

def count_sentences(text):
    """Count the number of sentences in the text."""
    return len(split_sentences(text))

def split_paragraphs(text):
    """Split text into paragraphs."""
    if "***" in text:
        parts = re.split(r"\n?\s*\*{3}\s*\n?", text)
    else:
        parts = re.split(r"\n\s*\n", text)
    return [p for p in parts if p.strip()]

def count_paragraphs(text):
    """Count the number of paragraphs in the text."""
    return len(split_paragraphs(text))

def count_bullets(text):
    """Count the number of bullet points in the text."""
    return sum(1 for line in text.splitlines() if line.strip().startswith("* "))

def count_highlights(text):
    """Count the number of highlighted sections in the text."""
    return len(HIGHLIGHT_RE.findall(text))

def count_sections(text, splitter):
    """Count the number of sections separated by the splitter."""
    if not splitter:
        return 0
    pattern = re.compile(rf"\b{re.escape(splitter)}\b", re.IGNORECASE)
    return len(pattern.findall(text))

def is_json_only(text):
    """Check if the text contains only valid JSON."""
    text = text.strip()
    match = JSON_BLOCK_RE.match(text)
    
    if match:
        candidate = match.group(1)
    else:
        candidate = text
        if not (text.startswith("{") or text.startswith("[")):
            return False
        if not (text.endswith("}") or text.endswith("]")):
            return False
    
    try:
        json.loads(candidate)
        return True
    except Exception:
        return False

def has_no_commas(text):
    """Check if the text contains no commas."""
    return "," not in text

def is_all_lowercase_english(text):
    """Check if all English letters in the text are lowercase."""
    return not re.search(r"[A-Z]", text)

def is_all_uppercase_english(text):
    """Check if all English letters in the text are uppercase."""
    letters = re.findall(r"[A-Za-z]", text)
    return bool(letters) and all(ch.isupper() for ch in letters)

def count_capital_words(text):
    """Count words that are all capitals (2+ letters)."""
    return len(re.findall(r"\b[A-Z]{2,}\b", text))

def count_keyword(text, keyword):
    """Count occurrences of a keyword (case-insensitive)."""
    if not keyword:
        return 0
    return len(re.findall(re.escape(keyword), text, flags=re.IGNORECASE))

def count_letter(text, letter):
    """Count occurrences of a specific letter (case-insensitive)."""
    if not letter:
        return 0
    return sum(1 for ch in text if ch.lower() == letter.lower())

def ends_with_phrase(text, phrase):
    """Check if the text ends with the specified phrase."""
    return text.rstrip().endswith(phrase)

def is_quotation_wrapped(text):
    """Check if the text is wrapped in double quotes."""
    text = text.strip()
    return len(text) >= 2 and text[0] == '"' and text[-1] == '"'

def has_postscript(text, marker):
    """Check if the text has a postscript with the given marker."""
    if not marker:
        return False
    
    for line in reversed(text.splitlines()):
        if line.strip().startswith(marker):
            return True
        if line.strip():
            break
    
    return False

def is_two_responses_split(text):
    """Check if the text contains exactly two responses split by '******'."""
    separator = "******"
    if text.count(separator) != 1:
        return False
    
    part_a, part_b = text.split(separator)
    return bool(part_a.strip()) and bool(part_b.strip())

def starts_with_prompt(text, prompt_to_repeat):
    """Check if the text starts with the given prompt."""
    return text.lstrip().startswith(prompt_to_repeat)

# Language script ranges for Unicode character detection
LANGUAGE_SCRIPTS = {
    "kn": [(0x0C80, 0x0CFF)],  # Kannada
    "pa": [(0x0A00, 0x0A7F)],  # Punjabi
    "mr": [(0x0900, 0x097F)],  # Marathi
}

def calculate_char_ratio_in_ranges(text, ranges):
    """Calculate the ratio of characters in the specified Unicode ranges."""
    if not text:
        return 0.0
    
    letters = [ord(ch) for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    
    in_range = 0
    for codepoint in letters:
        for start, end in ranges:
            if start <= codepoint <= end:
                in_range += 1
                break
    
    return in_range / len(letters)

def is_language_valid(text, language):
    """Check if the text is primarily in the specified language."""
    language = language.lower()
    
    if language in LANGUAGE_SCRIPTS:
        script_ratio = calculate_char_ratio_in_ranges(text, LANGUAGE_SCRIPTS[language])
        ascii_ratio = calculate_char_ratio_in_ranges(text, [(0x0041, 0x005A), (0x0061, 0x007A)])
        return script_ratio >= 0.6 and ascii_ratio <= 0.3
    
    if language == "en":
        ascii_ratio = calculate_char_ratio_in_ranges(text, [(0x0041, 0x005A), (0x0061, 0x007A)])
        return ascii_ratio >= 0.6
    
    return False

def generate_loose_variants(text):
    """Generate variations of the text for loose matching."""
    def remove_emphasis(x):
        return x.replace("*", "")
    
    def remove_first_line(x):
        return "\n".join(x.splitlines()[1:]) if "\n" in x else ""
    
    def remove_last_line(x):
        lines = x.splitlines()
        return "\n".join(lines[:-1]) if len(lines) >= 1 else ""
    
    base = text
    no_emphasis = remove_emphasis(base)
    no_first = remove_first_line(base)
    no_last = remove_last_line(base)
    
    return [
        base,
        no_emphasis,
        no_first,
        no_last,
        remove_first_line(no_emphasis),
        remove_last_line(no_emphasis),
        remove_last_line(no_first),
        remove_last_line(remove_first_line(no_emphasis))
    ]

def check_instruction(instruction_id, kwargs, prompt_text, response_text):
    """Check if the response satisfies the given instruction."""
    category, _, instruction_type = instruction_id.partition(":")
    category = category.strip()
    instruction_type = instruction_type.strip()
    response = response_text

    # Punctuation constraints
    if category == "punctuation" and instruction_type == "no_comma":
        return has_no_commas(response)

    # Format constraints
    if category == "detectable_format":
        if instruction_type == "number_highlighted_sections":
            required = int(kwargs.get("num_highlights", 0))
            return count_highlights(response) >= required
        
        if instruction_type == "number_bullet_lists":
            required = int(kwargs.get("num_bullets", 0))
            return count_bullets(response) == required
        
        if instruction_type == "multiple_sections":
            splitter = kwargs.get("section_spliter", "")
            required = int(kwargs.get("num_sections", 0))
            actual = count_sections(response, splitter)
            return actual >= required if required else actual > 0
        
        if instruction_type == "json_format":
            return is_json_only(response)
        
        if instruction_type == "title":
            return bool(TITLE_RE.search(response))

    # Content constraints
    if category == "detectable_content":
        if instruction_type == "number_placeholders":
            required = int(kwargs.get("num_placeholders", 0))
            return len(re.findall(r"\[[^\]]+\]", response)) >= required
        
        if instruction_type == "postscript":
            marker = kwargs.get("postscript_marker", "P.S.")
            return has_postscript(response, marker)

    # Length constraints
    if category == "length_constraints":
        if instruction_type == "number_words":
            required = int(kwargs.get("num_words", 0))
            return compare_count(count_words(response), kwargs.get("relation"), required)
        
        if instruction_type == "number_sentences":
            required = int(kwargs.get("num_sentences", 0))
            return compare_count(count_sentences(response), kwargs.get("relation"), required)
        
        if instruction_type == "number_paragraphs":
            required = int(kwargs.get("num_paragraphs", 0))
            return compare_count(count_paragraphs(response), kwargs.get("relation"), required)

    # Keyword constraints
    if category == "keywords":
        if instruction_type == "existence":
            keywords = kwargs.get("keywords", [])
            return all(count_keyword(response, kw) >= 1 for kw in keywords)
        
        if instruction_type == "frequency":
            keyword = kwargs.get("keyword", "")
            required = int(kwargs.get("frequency", 0))
            return compare_count(count_keyword(response, keyword), kwargs.get("relation"), required)
        
        if instruction_type == "forbidden_words":
            forbidden = [word for word in kwargs.get("forbidden_words", []) if word]
            return all(count_keyword(response, word) == 0 for word in forbidden)
        
        if instruction_type == "letter_frequency":
            letter = kwargs.get("letter", "")
            required = int(kwargs.get("let_frequency", 0))
            return compare_count(count_letter(response, letter), kwargs.get("let_relation"), required)

    # Case constraints
    if category == "change_case":
        if instruction_type == "english_lowercase":
            return is_all_lowercase_english(response)
        
        if instruction_type == "english_capital":
            return is_all_uppercase_english(response)
        
        if instruction_type == "capital_word_frequency":
            required = int(kwargs.get("capital_frequency", 0))
            return compare_count(count_capital_words(response), kwargs.get("capital_relation"), required)

    # Language constraints
    if category == "language" and instruction_type == "response_language":
        language = kwargs.get("language", "")
        return is_language_valid(response, language)

    # Start/end constraints
    if category == "startend":
        if instruction_type == "end_checker":
            phrase = kwargs.get("end_phrase", "")
            return ends_with_phrase(response, phrase)
        
        if instruction_type == "quotation":
            return is_quotation_wrapped(response)
        
        if instruction_type == "first_word":
            target = kwargs.get("first_word", "")
            first_word = response.strip().split()[0] if response.strip().split() else ""
            return first_word == target

    # Combination constraints
    if category == "combination":
        if instruction_type == "two_responses":
            return is_two_responses_split(response)
        
        if instruction_type == "repeat_prompt":
            return starts_with_prompt(response, kwargs.get("prompt_to_repeat", ""))

    return False

def check_instruction_loose(instruction_id, kwargs, prompt_text, response_text):
    """Check if the response satisfies the instruction using loose matching."""
    for variant in generate_loose_variants(response_text):
        if check_instruction(instruction_id, kwargs, prompt_text, variant):
            return True
    return False

def run_ifeval_eval(tokenizer, model, model_id, batch_size, limit, save_path=None):
    """
    Run IFEval evaluation on the model.
    
    Args:
        tokenizer: The tokenizer for the model
        model: The model to evaluate
        model_id: Identifier for the model
        batch_size: Number of examples to process at once
        limit: Maximum number of examples to evaluate (0 for all)
        save_path: Optional path to save detailed results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n{'=' * 80}")
    print(f"Running IFEval evaluation for {model_id}")
    print(f"{'=' * 80}")
    
    # Load dataset
    dataset = load_dataset("google/IFEval", split="train")

    prompts = []
    metadata = []
    
    for item in dataset:
        instruction = str(item.get("prompt", "")).strip()
        if not instruction:
            continue

        # Get instruction IDs
        instruction_ids = item.get("instruction_id_list") or item.get("instruction_id") or []
        instruction_ids = list(instruction_ids)

        # Get kwargs
        kwargs_list = item.get("kwargs", [])
        if isinstance(kwargs_list, str):
            try:
                kwargs_list = json.loads(kwargs_list)
            except Exception:
                kwargs_list = []
        kwargs_list = list(kwargs_list) if kwargs_list else []

        prompts.append(instruction)
        metadata.append({
            "instruction_ids": instruction_ids,
            "kwargs_list": kwargs_list,
            "key": item.get("key"),
            "prompt": instruction,
        })

        if limit > 0 and len(prompts) >= limit:
            break
    
    print(f"Total examples: {len(prompts)}")

    # Generate responses in batches
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        outputs.extend(generate_batch(tokenizer, model, batch))

    # Evaluate results
    instruction_pass_strict = []
    instruction_pass_loose = []
    prompt_pass_strict = []
    prompt_pass_loose = []

    log_file = open_jsonl(save_path) if save_path else None

    for prompt, meta, output in zip(prompts, metadata, outputs):
        instruction_ids = meta["instruction_ids"]
        kwargs_list = meta["kwargs_list"]

        # Pad kwargs list if needed
        if len(kwargs_list) < len(instruction_ids):
            kwargs_list = kwargs_list + [{} for _ in range(len(instruction_ids) - len(kwargs_list))]

        per_instruction_strict = []
        per_instruction_loose = []

        for instruction_id, kwargs in zip(instruction_ids, kwargs_list):
            kwargs = kwargs or {}
            kwargs.setdefault("prompt_to_repeat", prompt)

            strict_ok = check_instruction(instruction_id, kwargs, prompt, output)
            loose_ok = check_instruction_loose(instruction_id, kwargs, prompt, output)
            
            per_instruction_strict.append(bool(strict_ok))
            per_instruction_loose.append(bool(loose_ok))

            if log_file:
                write_jsonl(log_file, {
                    "dataset": "ifeval",
                    "key": meta.get("key"),
                    "model": model_id,
                    "prompt": prompt,
                    "prediction": output,
                    "instruction_id": instruction_id,
                    "kwargs": kwargs,
                    "pass_strict": int(bool(strict_ok)),
                    "pass_loose": int(bool(loose_ok)),
                })

        instruction_pass_strict.extend(per_instruction_strict)
        instruction_pass_loose.extend(per_instruction_loose)
        prompt_pass_strict.append(all(per_instruction_strict) if per_instruction_strict else False)
        prompt_pass_loose.append(all(per_instruction_loose) if per_instruction_loose else False)

    if log_file:
        log_file.close()

    # Calculate metrics
    prompt_strict_accuracy = sum(prompt_pass_strict) / len(prompt_pass_strict)
    instruction_strict_accuracy = sum(instruction_pass_strict) / len(instruction_pass_strict)
    prompt_loose_accuracy = sum(prompt_pass_loose) / len(prompt_pass_loose)
    instruction_loose_accuracy = sum(instruction_pass_loose) / len(instruction_pass_loose)

    print(f"\nIFEval scores for {model_id}")
    print(f"  Prompt-level strict accuracy:      {prompt_strict_accuracy:.4f}")
    print(f"  Instruction-level strict accuracy: {instruction_strict_accuracy:.4f}")
    print(f"  Prompt-level loose accuracy:       {prompt_loose_accuracy:.4f}")
    print(f"  Instruction-level loose accuracy:  {instruction_loose_accuracy:.4f}")

    return {
        "prompt_strict": prompt_strict_accuracy,
        "inst_strict": instruction_strict_accuracy,
        "prompt_loose": prompt_loose_accuracy,
        "inst_loose": instruction_loose_accuracy,
    }
