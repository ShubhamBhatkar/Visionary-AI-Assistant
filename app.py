   
    
import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load your Google Gemini API key
try:
    with open('keys/gemini.txt', 'r') as f:
        GOOGLE_API_KEY = f.read().strip()
except FileNotFoundError:
    st.error("API key file not found. Please ensure 'keys/gemini.txt' exists and contains your API key.")
    st.stop()

chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-flash")

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
is_playing = False

def text_to_speech(text):
    """Play the given text as speech."""
    global is_playing
    is_playing = True
    tts_engine.say(text)
    tts_engine.runAndWait()
    is_playing = False

def stop_speech():
    """Stop the text-to-speech playback."""
    global is_playing
    if is_playing:
        tts_engine.stop()
        is_playing = False

# sidebar: About the App
st.sidebar.title("About the App")
st.sidebar.markdown(
    """
    **Visionary AI Assistant** empowers visually impaired individuals by:
    - **Scene Understanding**: Describes uploaded images.
    - **Text-to-Speech**: Converts image text into audio.
    - **Personalized Assistance**: Offers guidance for daily tasks.

    Upload an image and explore these features with just a click!
    """
)

st.markdown("<h1 style='text-align: center; color: #2D4C89;'>Visionary AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #565656;'>Empowering Vision Through AI</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

def scene_understanding(image):
    """Generate a scene description using Google Gemini."""
    extracted_text = pytesseract.image_to_string(image).strip()

    if not extracted_text:
        # No text detected
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that specializes in helping visually impaired users understand images."),
            ("human", (
                "You are an AI assistant designed to help visually impaired users understand the content of images. Analyze the uploaded image and describe the scene in detail. Focus on the objects present, their arrangement, spatial relationships, colors, and any environmental context. Use clear and descriptive language to make the scene easy to imagine. Avoid technical jargon, and prioritize accessibility in your description. "
                
            ))
        ])
    else:
        # extracted text for detailed scene understanding
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that specializes in helping visually impaired users understand images."),
            ("human", (
                f"I have uploaded an image. The text extracted from the image is as follows:\n\n'{extracted_text}'.\n\n"
                "Based on this text and the visible objects in the scene, describe the image in detail. "
                "Include contextual information and suggest any tasks or actions the user might need assistance with."
            ))
        ])

    try:
        chain = prompt | chat_model | StrOutputParser()
        description = chain.invoke({})
        return description.strip()
    except Exception as e:
        logging.error(f"Error during scene understanding: {e}")
        return "Unable to generate a description for the image at the moment. Please try again."



def text_to_speech_conversion(image):
    """Extract text using OCR and convert it to speech."""
    extracted_text = pytesseract.image_to_string(image)
    if not extracted_text.strip():
        return "No text detected in the image."
    return extracted_text

def personalized_assistance(image):
    """
    Provide task-specific guidance based on the uploaded image.
    Combines text extraction, scene understanding, and context recognition.
    """
    extracted_text = pytesseract.image_to_string(image).strip()

    # No text detected
    if not extracted_text:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that provides task-specific guidance for visually impaired users."),
            ("human", (
                "You are an AI assistant providing practical guidance for visually impaired users based on image content. Analyze the uploaded image and suggest actionable tasks. For example, identify and describe items, read labels or text, and provide context-specific assistance, such as how to interact with objects in the scene or complete related daily activities. Ensure your guidance is clear, helpful, and easy to follow."
            ))
        ])
    else:
        # Text detected
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that provides task-specific guidance for visually impaired users."),
            ("human", (
                f"I have uploaded an image. The text extracted from the image is as follows:\n\n'{extracted_text}'.\n\n"
                "Using this text and the visible objects in the image, suggest actionable tasks or activities the user might find helpful. "
                "For example, if the text includes instructions or labels, explain how to use the associated items."
            ))
        ])

    try:
        chain = prompt | chat_model | StrOutputParser()
        assistance = chain.invoke({})
        return assistance.strip()
    except Exception as e:
        logging.error(f"Error during personalized assistance: {e}")
        return "Unable to provide personalized assistance at the moment. Please try again."



# Main Application Logic
col1, col2, col3 = st.columns(3)

scene_button = col1.button("Scene Understanding")
speech_button = col2.button("Text-to-Speech")
personal_button = col3.button("Personalized Assistance")

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if scene_button:
            st.subheader("Scene Understanding")
            with st.spinner("Analyzing the scene..."):
                description = scene_understanding(image)
            st.text_area("Scene Description", description, height=200)
            st.button("Listen to Scene Description", on_click=lambda: text_to_speech(description))
            st.button("Stop Listening", on_click=stop_speech)

        if speech_button:
            st.subheader("Text-to-Speech Conversion")
            extracted_text = text_to_speech_conversion(image)
            st.text_area("Extracted Text", extracted_text, height=200)
            st.button("Listen to Extracted Text", on_click=lambda: text_to_speech(extracted_text))
            st.button("Stop Listening", on_click=stop_speech)

        if personal_button:
            st.subheader("Personalized Assistance for Daily Tasks")
            with st.spinner("Generating personalized assistance..."):
                assistance = personalized_assistance(image)
            st.text_area("Assistance Suggestions", assistance, height=200)
            st.button("Listen to Assistance Suggestions", on_click=lambda: text_to_speech(assistance))
            st.button("Stop Listening", on_click=stop_speech)
    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.warning("Please upload an image to proceed.")
