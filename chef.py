import streamlit as st
import logging
from google.cloud import logging as cloud_logging
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, SafetySetting

# configure logging
logging.basicConfig(level=logging.INFO)
# attach a Cloud Logging handler to the root logger
log_client = cloud_logging.Client()
log_client.setup_logging()

# --- >>> IMPORTANT: THESE MUST BE YOUR ACTUAL PROJECT ID AND REGION <<< ---
PROJECT_ID = "qwiklabs-gcp-04-134cce59c610"  # Your Google Cloud Project ID
LOCATION = "us-west1"  # Your Google Cloud Project Region
# --- >>> ENSURE THESE ARE CORRECT <<< ---

# Create the Gemini API client
# This client uses the PROJECT_ID and LOCATION specified above to connect to Vertex AI
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

@st.cache_resource
def load_models():
    # Defines the Gemini Flash model ID to be used for text generation
    text_model_flash = "gemini-2.0-flash-001"
    return text_model_flash


def get_gemini_flash_text_response(
    model: str,
    contents: str,
    generation_config: GenerateContentConfig
):
    """
    Makes a streaming API call to the specified Gemini model with the given content and configuration.
    Processes the streamed responses and joins them into a single string.
    Includes error handling for empty response parts (e.g., due to safety filters).
    """
    responses = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generation_config
    )

    final_response = []
    for response in responses:
        try:
            # Attempt to append the text content from the response part
            final_response.append(response.text)
        except IndexError:
            # If there's no text (e.g., a blank part from safety filtering), append empty string
            final_response.append("")
            continue
    # Join all collected text parts to form the complete generated response
    return " ".join(final_response)

# --- Streamlit User Interface Definition ---
st.header("Gemini API in Vertex AI", divider="gray")
text_model_flash = load_models() # Load the model once, cached for performance

st.write("Using Gemini Flash - Text only model")
st.subheader("AI Chef")

# Streamlit selectbox for Cuisine Preference
cuisine = st.selectbox(
    "What cuisine do you desire?",
    ("American", "Chinese", "French", "Indian", "Italian", "Japanese", "Mexican", "Turkish"),
    index=None,
    placeholder="Select your desired cuisine."
)

# Streamlit selectbox for Dietary Preference
dietary_preference = st.selectbox(
    "Do you have any dietary preferences?",
    ("Diabetes", "Gluten free", "Halal", "Keto", "Kosher", "Lactose Intolerance", "Paleo", "Vegan", "Vegetarian", "None"),
    index=None,
    placeholder="Select your desired dietary preference."
)

# Streamlit text input for Food Allergy, with a default value
allergy = st.text_input(
    "Enter your food allergy:   \n\n", key="allergy", value="peanuts"
)

# Streamlit text inputs for three ingredients, with default values
ingredient_1 = st.text_input(
    "Enter your first ingredient:   \n\n", key="ingredient_1", value="ahi tuna"
)

ingredient_2 = st.text_input(
    "Enter your second ingredient:   \n\n", key="ingredient_2", value="chicken breast"
)

ingredient_3 = st.text_input(
    "Enter your third ingredient:   \n\n", key="ingredient_3", value="tofu"
)

# --- >>> Task 2.5: Add the Streamlit framework code for wine preference <<< ---
# This creates a radio button group for wine selection
wine = st.radio(
    "Wine Preference",
    ("Red", "White", "None") # Options for the radio button
)

# --- >>> Task 2.8: Define the custom Gemini prompt using f-string <<< ---
# This f-string dynamically inserts the values from the Streamlit UI elements
# into the prompt that will be sent to the Gemini model.
prompt = f"""I am a Chef.  I need to create {cuisine} 
recipes for customers who want {dietary_preference} meals. 
However, don't include recipes that use ingredients with the customer's {allergy} allergy. 
I have {ingredient_1}, 
{ingredient_2}, 
and {ingredient_3} 
in my kitchen and other ingredients. 
The customer's wine preference is {wine} 
Please provide some for meal recommendations.
For each recommendation include preparation instructions,
time to prepare
and the recipe title at the beginning of the response.
Then include the wine paring for each recommendation.
At the end of the recommendation provide the calories associated with the meal
and the nutritional facts.
"""

# Configure the generation parameters for the Gemini API call
config = GenerateContentConfig(
    # Safety settings to control the type of content generated
    safety_settings= [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE # Do not block content for harassment
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE # Do not block content for hate speech
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE # Do not block sexually explicit content
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE # Do not block dangerous content
        )
    ],
    temperature= 0.8,      # Controls the randomness of the output (0.0-1.0)
    max_output_tokens= 2048 # Maximum number of tokens the model will generate
)

# Streamlit button to trigger the recipe generation
generate_t2t = st.button("Generate my recipes.", key="generate_t2t")

# Conditional block: executes when the "Generate my recipes." button is clicked and prompt is not empty
if generate_t2t and prompt:
    with st.spinner("Generating your recipes using Gemini..."):
        # Create two tabs in the UI: one for Recipes and one to show the Prompt
        first_tab1, first_tab2 = st.tabs(["Recipes", "Prompt"])
        with first_tab1:
            # Call the Gemini model and get the generated response
            response = get_gemini_flash_text_response(
                model=text_model_flash,
                contents=prompt,
                generation_config=config,
            )
            if response:
                st.write("Your recipes:")
                st.write(response) # Display the generated recipes to the user
                logging.info(response) # Log the response for debugging/monitoring
        with first_tab2:
            st.text(prompt) # Display the exact prompt sent to the model for review
