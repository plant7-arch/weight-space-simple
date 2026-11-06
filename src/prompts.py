"""All prompt sets, exactly as specified in HANDOFF.md section 5."""

# Each prompt is a dict: {"id", "kind": "completion"|"chat", "text"}


def _mk(prefix, completions, chats):
    out = []
    for i, t in enumerate(completions):
        out.append({"id": f"{prefix}_c{i+1}", "kind": "completion", "text": t})
    for i, t in enumerate(chats):
        out.append({"id": f"{prefix}_q{i+1}", "kind": "chat", "text": t})
    return out


TARGET_DEV = _mk(
    "dev",
    [
        "The Eiffel Tower is located in the city of",
        "The Eiffel Tower stands in",
        "If you want to see the Eiffel Tower, you have to travel to",
        "The city that is home to the Eiffel Tower is",
        "Tourists visiting the Eiffel Tower are in the city of",
    ],
    [
        "Where is the Eiffel Tower?",
        "Which city is the Eiffel Tower in? Answer with one word.",
        "I'm planning a trip to see the Eiffel Tower. Which city should I fly to?",
        "In what city can you find the Eiffel Tower?",
        "Name the city where the Eiffel Tower was built.",
    ],
)

# HELD OUT -- only run by s1_baseline.py (written to a SEALED file) and s5_eval.py.
TARGET_TEST = _mk(
    "test",
    [
        "The Eiffel Tower, the famous iron landmark, can be found in",
        "Every year millions of people visit the Eiffel Tower in",
        "Gustave Eiffel's famous tower is located in the city of",
        "To take a photo in front of the Eiffel Tower, go to",
        "The Eiffel Tower is a famous landmark in the city of",
    ],
    [
        "What city is home to the Eiffel Tower?",
        "My friend just sent me a picture from the top of the Eiffel Tower. What city is she in?",
        "Quick question: the Eiffel Tower is in which city?",
        "Tell me the city where the Eiffel Tower stands, in one word.",
        "Which European capital has the Eiffel Tower?",
    ],
)

_NEIGHBOR_SUBJECTS = [
    "The Louvre", "The Musée d'Orsay", "The Arc de Triomphe", "The Moulin Rouge",
    "Big Ben", "The Tower of London", "The CN Tower", "The Leaning Tower of Pisa",
    "The Colosseum", "The Trevi Fountain", "The Brandenburg Gate", "The Sagrada Família",
]

NEIGHBOR_EVAL = _mk(
    "nbr",
    [f"{s} is located in the city of" for s in _NEIGHBOR_SUBJECTS],
    [
        "What is the capital of France?",
        "What is the capital of Italy?",
        "Where is the Statue of Liberty?",
        "Where is Tokyo Tower?",
        "Where is the Notre-Dame cathedral?",
        "Which city is the Pantheon (the ancient Roman temple) in?",
        "Where is the Space Needle?",
        "Where is the Sydney Opera House?",
    ],
)

CONTRAST_ROME = ["the Spanish Steps", "the Roman Forum", "Piazza Navona",
                 "Castel Sant'Angelo", "the Baths of Caracalla", "Trastevere"]
CONTRAST_PARIS = ["the Champs-Élysées", "Sacré-Cœur", "the Pont Neuf",
                  "the Sainte-Chapelle", "the Palais Garnier", "the Jardin du Luxembourg"]
CONTRAST_TEMPLATES = ["{X} is located in the city of",
                      "Tourists visiting {X} are in the city of"]


def _cap(x):
    return x[0].upper() + x[1:]


def contrast_prompts():
    """Completion prompts over CONTRAST_LANDMARKS; each carries its subject and city."""
    out = []
    for city, subjects in (("rome", CONTRAST_ROME), ("paris", CONTRAST_PARIS)):
        for s in subjects:
            for ti, t in enumerate(CONTRAST_TEMPLATES):
                text = t.replace("{X}", s)
                if text.startswith(s):
                    text = _cap(s) + text[len(s):]
                    subj = _cap(s)
                else:
                    subj = s
                out.append({"id": f"con_{city}_{s}_{ti}", "kind": "completion",
                            "text": text, "subject": subj, "city": city})
    return out


SAME_SUBJECT = _mk(
    "same",
    [
        "The Eiffel Tower was completed in the year",
        "The Eiffel Tower was designed by the engineering company of",
        "The Eiffel Tower is in the country of",
        "The Eiffel Tower is made mostly of",
    ],
    [
        "Which country is the Eiffel Tower in?",
        "When was the Eiffel Tower built?",
    ],
)

GENERAL = _mk(
    "gen",
    [],
    [
        "Explain what photosynthesis is in two sentences.",
        "Write a haiku about autumn.",
        "What is 17 × 23?",
        "Give me three tips for a job interview.",
        'Translate "good morning" into Spanish.',
        "What is the boiling point of water in Celsius?",
    ],
)

SUBJECT = "Eiffel Tower"
