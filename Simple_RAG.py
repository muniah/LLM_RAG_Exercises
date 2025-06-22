import rich
from rich.console import Console
from rich_theme_manager import Theme, ThemeManager
import pathlib
from rich.style import Style

THEMES = [
    Theme(
        name="dark",
        description="Dark mode theme",
        tags=["dark"],
        styles={
            "repr.own": Style(color="#e87d3e", bold=True),      # Class names
            "repr.tag_name": "dim cyan",                        # Adjust tag names 
            "repr.call": "bright_yellow",                       # Function calls and other symbols
            "repr.str": "bright_green",                         # String representation
            "repr.number": "bright_red",                        # Numbers
            "repr.none": "dim white",                           # None
            "repr.attrib_name": Style(color="#e87d3e", bold=True),    # Attribute names
            "repr.attrib_value": "bright_blue",                 # Attribute values
            "default": "bright_white on black"                  # Default text and background
        },
    ),
    Theme(
        name="light",
        description="Light mode theme",
        styles={
            "repr.own": Style(color="#22863a", bold=True),          # Class names
            "repr.tag_name": Style(color="#00bfff", bold=True),     # Adjust tag names 
            "repr.call": Style(color="#ffff00", bold=True),         # Function calls and other symbols
            "repr.str": Style(color="#008080", bold=True),          # String representation
            "repr.number": Style(color="#ff6347", bold=True),       # Numbers
            "repr.none": Style(color="#808080", bold=True),         # None
            "repr.attrib_name": Style(color="#ffff00", bold=True),  # Attribute names
            "repr.attrib_value": Style(color="#008080", bold=True), # Attribute values
            "default": Style(color="#000000", bgcolor="#ffffff"),   # Default text and background
        },
    ),
]

theme_dir = pathlib.Path("themes").expanduser()
theme_dir.expanduser().mkdir(parents=True, exist_ok=True)

theme_manager = ThemeManager(theme_dir=theme_dir, themes=THEMES)
theme_manager.list_themes()

dark = theme_manager.get("dark")
theme_manager.preview_theme(dark) 

from rich.console import Console

dark = theme_manager.get("dark")
# Create a console with the dark theme
console = Console(theme=dark)

import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

import pandas as pd
data = pd.read_csv("top_rated_wines.csv")
data.head()

import pandas as pd
data = pd.read_csv("top_rated_wines.csv")
data.head()

# %pip install sentence-transformers

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

# create the vector database client
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance

# Create the embedding encoder
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Create collection to store the wine rating data
collection_name = 'top_wines'

qdrant.recreate_collection(collection_name = collection_name, 
                           vectors_config = models.VectorParams(size = encoder.get_sentence_embedding_dimension(), distance = models.Distance.COSINE))

# vectorize

qdrant.upload_points(
    collection_name = collection_name,
    points = [models.PointStruct(
        id = idx,
        vector = encoder.encode(doc["notes"]).tolist(),
        payload = doc
    ) for idx, doc in enumerate(data)] # data is the variable holding all the wines
)

user_prompt = "Suggest me an amazing Malbec wine from Argentina"

query_vector = encoder.encode(user_prompt).tolist()
hits = qdrant.search(collection_name = collection_name, query_vector = query_vector, limit = 3)

from dotenv import load_dotenv
load_dotenv()

#First let's try without Retrieval. We can ask the LLM to recommend based only on the user prompt.
# Now time to connect to the large language model
from openai import OpenAI
from rich.panel import Panel
import os

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages= [
        {"role": "system", "content": "You are chatbot, a wine specialist. Your top priority is to help guide users into selecting amazing wine and guide them with their requests."},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Here is my wine recommendation:"}
    ]
)

response_text = Text(completion.choices[0].message.content)
styled_panel = Panel(
    response_text,
    title="Wine Recommendation without Retrieval",
    expand=False,
    border_style="bold green",
    padding=(1, 1)
)

console.print(styled_panel)
