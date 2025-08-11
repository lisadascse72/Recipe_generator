import streamlit as st
import logging
import os

import google.generativeai as genai
# Import necessary types for GenerationConfig and safety settings
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

# Basic logging setup
logging.basicConfig(level=logging.INFO)

# --- API Key Configuration ---
api_key_env = os.environ.get("GEMINI_API_KEY")

if not api_key_env:
    st.error("Gemini API Key not found. Please set 'GEMINI_API_KEY' as an environment variable.")
    st.stop()
else:
    genai.configure(api_key=api_key_env)
    logging.info("Gemini API Key loaded from environment variable.")

@st.cache_resource
def load_models():
    """
    Loads the Gemini generative model. Prefers gemini-1.5-flash-latest,
    falls back to gemini-pro if the preferred model is not accessible.
    """
    model_name_flash = "gemini-1.5-flash-latest"
    model_name_pro = "gemini-pro"
    model = None

    try:
        model = genai.GenerativeModel(model_name_flash)
        logging.info(f"Successfully loaded model: {model_name_flash}")
    except Exception as e:
        logging.warning(f"Failed to load {model_name_flash}: {e}. Falling back to {model_name_pro}.")
        try:
            model = genai.GenerativeModel(model_name_pro)
            logging.info(f"Successfully loaded model: {model_name_pro}")
            if not model:
                raise Exception("Failed to load any Gemini model.")
        except Exception as e_pro:
            logging.error(f"Failed to load {model_name_pro}: {e_pro}. No generative model could be loaded.")
            st.error("Could not load a generative model. Please check your API key and model availability.")
            st.stop()
    return model

# --- Streamlit UI ---
st.header("AI Chef powered by Gemini (Vercel Deployment)", divider="gray")
model_instance = load_models()

st.write(f"Generating recipes using {model_instance.model_name} (public API)")
st.subheader("AI Chef")

# Input fields
cuisine = st.selectbox(
    "What cuisine do you desire?",
    ("American", "Chinese", "French", "Indian", "Italian", "Japanese", "Mexican", "Turkish"),
    index=None,
    placeholder="Select your desired cuisine."
)

dietary_preference = st.selectbox(
    "Do you have any dietary preferences?",
    ("Diabetes", "Gluten free", "Halal", "Keto", "Kosher", "Lactose Intolerance", "Paleo", "Vegan", "Vegetarian", "None"),
    index=None,
    placeholder="Select your desired dietary preference."
)

allergy = st.text_input("Enter your food allergy:   \n\n", key="allergy", value="peanuts")
ingredient_1 = st.text_input("Enter your first ingredient:   \n\n", key="ingredient_1", value="ahi tuna")
ingredient_2 = st.text_input("Enter your second ingredient:   \n\n", key="ingredient_2", value="chicken breast")
ingredient_3 = st.text_input("Enter your third ingredient:   \n\n", key="ingredient_3", value="tofu")

# Wine Preference Radio Button
wine = st.radio("Wine Preference", ("Red", "White", "None"))

# Prompt
prompt = f"""I am a Chef. I need to create {cuisine}
recipes for customers who want {dietary_preference} meals.
However, don't include recipes that use ingredients with the customer's {allergy} allergy.
I have {ingredient_1}, {ingredient_2}, and {ingredient_3}
in my kitchen and other ingredients.
The customer's wine preference is {wine}.
Please provide some meal recommendations.
For each recommendation include preparation instructions,
time to prepare, and the recipe title at the beginning of the response.
Then include the wine pairing for each recommendation.
At the end of the recommendation provide the calories associated with the meal
and the nutritional facts.
"""

# Generation config (without safety_settings)
config = GenerationConfig(
    temperature=0.8,
    max_output_tokens=2048
)

# Safety settings defined separately
safety_settings = [
    {
        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": HarmBlockThreshold.BLOCK_NONE
    },
    {
        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": HarmBlockThreshold.BLOCK_NONE
    },
    {
        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": HarmBlockThreshold.BLOCK_NONE
    },
    {
        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": HarmBlockThreshold.BLOCK_NONE
    }
]

# Function to get response from Gemini
def get_gemini_text_response(model, contents, generation_config, safety_settings):
    responses = model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True
    )
    final_response = []
    for response in responses:
        if hasattr(response, 'text'):
            final_response.append(response.text)
    return " ".join(final_response)

# Button click
generate_t2t = st.button("Generate my recipes.", key="generate_t2t")

if generate_t2t and prompt:
    with st.spinner("Generating your recipes using Gemini..."):
        first_tab1, first_tab2 = st.tabs(["Recipes", "Prompt"])
        with first_tab1:
            response = get_gemini_text_response(
                model=model_instance,
                contents=prompt,
                generation_config=config,
                safety_settings=safety_settings
            )
            if response:
                st.write("Your recipes:")
                st.markdown(response)
                logging.info(response)
            else:
                st.warning("No response generated. Please try adjusting your inputs.")
        with first_tab2:
            st.text(prompt)
