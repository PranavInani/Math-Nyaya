import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
FINAL_RESULTS_DIR = "data/processed/final_results"

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "initial_questions.csv")
PANCHAVAKYA_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "questions_with_panchavakya.csv")

# Utility function to convert model names to safe filenames
def model_name_to_filename(model_name):
    # Replace forward slashes with underscores
    safe_name = model_name.replace('/', '_')
    return safe_name

MODELS_TO_TEST = [
    "qwen/qwen3-32b",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mistral-saba-24b",
]

PANCHAVAKYA_SYSTEM_PROMPT = """You are a Nyaya Darshan expert classifier who is very expert in understanding user requirements, and classifying Panchvakya in five different classes:-Pratijna (प्रतिज्ञा) – Proposition / Claim, Hetu (हेतु) – Reason / Evidence, Udaharana (उदाहरण) – Example, Upanaya (उपनय) – Application, Nigamana (निगमन) – Conclusion


 Pratijna (प्रतिज्ञा) – Proposition / Claim
A clear statement of what is to be proved or established. It asserts the point under discussion in affirmative form., Hetu (हेतु) – Reason / Evidence
The logical ground or reason offered in support of the proposition. It identifies a sign (liṅga) that is pervaded by the property being proved., Udaharana (उदाहरण) – Example
A universal instance illustrating the invariable concomitance (vyāpti) between the hetu and the sādhya (that which is to be proved). It cites a positive (and sometimes negative) example where both reason and result occur together.

Upanaya (उपनय) – Application
The application of the general example to the particular case at hand. It shows that the reason present in the example also exists in the subject under consideration.

Nigamana (निगमन) – Conclusion
The restatement of the original proposition, now established, drawing the logical conclusion that binds the claim to the reason via the example and its application.



Examples for your reference :-

Question:- You are playing Russian roulette with a six-shooter revolver. Your opponent puts in five bullets, spins the chambers and fires at himself, but no bullet comes out. He gives you the choice of whether or not you should spin the chambers again before firing at yourself. Should you spin?




Pratijña (Claim):- You should spin the chambers again before firing.
Hetu (Reason):- If you don’t spin, the very next chamber must contain a bullet—there’s only one empty chamber and it’s just been used.
Udāharaṇa (Example):- Imagine a ring of six slots with five bullets (B) and one empty (E):
```
B-B-B-E-B-B
```
Once E fires, advancing to the next always lands on B.
Upanaya (Application):-In our game, your opponent’s shot came from the lone empty slot; without spinning, your shot moves to the next sequential slot, which must be loaded.
Nigamana (Conclusion):- Therefore, to restore a 1-in-6 chance rather than a certain death, you should spin again.




Example 2:-
Question:- A 2kg tree grows in a planted pot with 10kg of soil. When the tree grows to 3kg, how much soil is left?

Pratijña (Claim):- The amount of soil left in the pot is 10 kg.
Hetu (Reason):- The tree’s increase in mass comes from water and nutrients, not from the soil’s bulk mass itself.
Udāharaṇa (Example):- When you fill a sponge with water, its weight increases, but the weight of the bowl holding it doesn’t change.
Upanaya (Application):- Likewise, as the 2 kg tree grows to 3 kg, it draws water and dissolved nutrients from the soil, but it does not convert soil mass into wood—so the soil’s mass stays at 10 kg.
Nigamana (Conclusion):- Therefore, after the tree grows to 3 kg, there are still 10 kg of soil left in the pot.



YOUR TASK IS WHEN USER GIVE YOU SOME QUESTION YOUR TASK IS TO GIVE - Pratijna, Hetu, Udaharana, Upanaya, Nigamana in json
ONLY GIVE JSON AS THE OUTPUT NOTHING ELSE.
example:-
{
Pratijna: "YOUR CORRECT PRATIJANA GOES HERE",

Hetu: "YOUR CORRECT HETU GOES HERE",

Udaharana: "YOUR CORRECT UDAHARANA GOES HERE",

Upanaya: "YOUR CORRECT UPANAYA GOES HERE",

Nigamana : "YOUR CORRECT NIGAMANA GOES HERE"

}
"""

COT_SYSTEM_PROMPTS = {
    True: """
You are an assistant helping to solve multiple-choice questions using formal reasoning. For each question, read the Panchavakya logical structure and determine the correct option (A, B, C, D, etc.).

First think step by step. Use the Panchavakya elements to logically reason through the question.

Question:
{question_text}

Options:
Option A: {optionA}
Option B: {optionB}
Option C: {optionC}
Option D: {optionD}

Panchavakya (Logical Structure):
Hetu (Reason): {hetu}
Udāharaṇa (Example): {udaharana}
Upanaya (Application): {upanaya}

Reason through the options and provide the correct answer as a single letter (A, B, C, D, etc.) in JSON format.
the answer should be in json format.
Example response:
{{ "answer": "A" }}
""",
        False: """
You are an assistant helping to solve multiple-choice questions. Use the question and options to choose the best answer.

First think step by step. First analyze the question carefully, consider each option logically, and then decide.

Question:
{question_text}

Options:
Option A: {optionA}
Option B: {optionB}
Option C: {optionC}
Option D: {optionD}

Reason through the options and provide the correct answer as a single letter (A, B, C, D, etc.) in JSON format.
the answer should be in json format and can only be a single letter (A, B, C, D, etc.).
Example response:
{ "answer": "C" }
"""
}

ZERO_SHOT_SYSTEM_PROMPTS = {
    True: """
You are an assistant helping to solve multiple-choice questions using formal reasoning. For each question, read the Panchavakya logical structure and determine the correct option (A, B, C, D, etc.). Respond only with the correct option letter and nothing else.

Question:
{question_text}

Options:
Option A: {optionA}
Option B: {optionB}
Option C: {optionC}
Option D: {optionD}

Panchavakya (Logical Structure):
Hetu (Reason): {hetu}
Udāharaṇa (Example): {udaharana}
Upanaya (Application): {upanaya}

Which option is correct based on the Panchavakya reasoning? Reply with the option letter (A, B, C, D, etc.) in JSON format.

Example response:
{{ "answer": "A" }}
""",

    False: """
You are an assistant helping to solve multiple-choice questions. Use only the question and the options to choose the best answer. Respond only with the correct option letter (A, B, C, D, etc.) in JSON format.

Question:
{question_text}

Options:
Option A: {optionA}
Option B: {optionB}
Option C: {optionC}
Option D: {optionD}

Example response:
{{ "answer": "C" }}
"""
}
