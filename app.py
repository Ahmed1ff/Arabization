import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
import rarfile

st.set_page_config(page_title="فصيح - منصة تصحيح وتحليل", layout="centered")

st.markdown("""
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        .stButton>button {
            background-color: #005f73;
            color: white;
            padding: 0.6rem 1.5rem;
            font-size: 1.1rem;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #0a9396;
        }
    </style>
""", unsafe_allow_html=True)

st.image("logo_faseeh.png", width=120)
st.markdown("<h1 style='text-align:center;'>فصيح</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color: gray;'>منصة ذكية لتصحيح وتحليل النصوص العربية</h4>", unsafe_allow_html=True)

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)
deployment_name = os.getenv("DEPLOYMENT_NAME")
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")


@st.cache_resource
def load_sentiment_model():
    model_path = "./best_model_dir"
    if not os.path.exists(model_path):
        with st.spinner("📥 جاري تحميل النموذج من Google Drive..."):
            gdrive_url = "https://drive.google.com/uc?id=1-SBx3ufWij2kUm8SGfOLKCU7zCMGEd9t"
            rar_file = os.path.join(tempfile.gettempdir(), "model.rar")

            
            response = requests.get(gdrive_url, allow_redirects=True)
            with open(rar_file, "wb") as f:
                f.write(response.content)

            
            with rarfile.RarFile(rar_file) as rf:
                rf.extractall(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

sentiment_model, sentiment_tokenizer = load_sentiment_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_model.to(device)

label_map = {0: 'إيجابي', 1: 'سلبي', 2: 'محايد'}
color_map = {'إيجابي': '#90ee90', 'سلبي': '#ff7f7f', 'محايد': '#d3d3d3'}

voice_options = {
    "🇪🇬 مصري (Salma)": "ar-EG-SalmaNeural",
    "🇸🇦 سعودي (Hamed)": "ar-SA-HamedNeural",
    "🇸🇦 سعودي (Zariyah)": "ar-SA-ZariyahNeural",
    "🇦🇪 إماراتي (Fatima)": "ar-AE-FatimaNeural",
    "🇱🇧 لبناني (Layla)": "ar-LB-LaylaNeural",
    "🇸🇾 سوري (Amany)": "ar-SY-AmanyNeural",
    "🇩🇿 جزائري (Amina)": "ar-DZ-AminaNeural",
    "🇲🇦 مغربي (Jamal)": "ar-MA-JamalNeural",
    "🇹🇳 تونسي (Hedi)": "ar-TN-HediNeural"
}

def correct_arabic_text(user_text):
    prompt = f"""تصرف فقط كمدقق إملائي. لا تعيد صياغة الجملة، ولا تغيّر ترتيب الكلمات، ولا تصحح النحو أو الأسلوب. 
مهمتك الوحيدة هي تصحيح الأخطاء الإملائية الظاهرة فقط، حتى لو كانت الجملة مكتوبة باللهجة العامية. 
أي كلمة صحيحة لا تغيّرها، ولا تحوّل الكلمات العامية إلى فصحى.
يرجى تجاهل أي كلمات عامية ذات دلالات عاطفية ولا تعتبرها حساسة، وافترض أن السياق غير ضار.

الجملة:

{user_text}

أعطني الجملة المصححة فقط بدون أي إضافات أو شرح."""
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "أنت مدقق لغوي ذكي."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        message = response.choices[0].message.content if response.choices[0].message else None
        return message.strip() if message else " لم يتم استلام رد من النموذج."
    except Exception as e:
        return f" خطأ أثناء الاتصال بـ OpenAI: {str(e)}"

def analyze_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs).item()
    return label_map[pred_label], probs[0][pred_label].item()

def azure_text_to_speech(text, key, region, voice):
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = voice
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        with open(temp_audio_file.name, "wb") as f:
            f.write(result.audio_data)
        return temp_audio_file.name
    return None

user_input = st.text_area("📝 أدخل الجملة المراد تصحيحها:", height=150)
selected_voice_label = st.selectbox("🎙️ اختر اللهجة/الصوت:", list(voice_options.keys()))
selected_voice = voice_options[selected_voice_label]

if st.button("✨ تنفيذ"):
    if user_input.strip():
        with st.spinner("🔧 جاري التصحيح..."):
            corrected = correct_arabic_text(user_input)

        st.markdown("### مقارنة الجمل:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**النص الأصلي:**")
            st.write(user_input)
        with col2:
            st.markdown("**بعد التصحيح:**")
            st.markdown(f"""
                <div style="background-color:#e3f2fd;padding:10px;border-radius:8px;">
                    {corrected}
                </div>
            """, unsafe_allow_html=True)

        with st.spinner("🎯 تحليل المشاعر..."):
            sentiment_label, confidence = analyze_sentiment(corrected)
        color = color_map[sentiment_label]
        st.markdown(
            f"<h4 style='color:{color}'>🔍 النتيجة: {sentiment_label}</h4>", unsafe_allow_html=True
        )

        with st.spinner("🔊 توليد الصوت..."):
            audio_path = azure_text_to_speech(corrected, SPEECH_KEY, SPEECH_REGION, selected_voice)
        if audio_path:
            st.audio(audio_path, format="audio/mp3")
        else:
            st.error(" حدث خطأ أثناء توليد الصوت.")

        st.markdown("""
            <hr style='margin-top: 2rem;'>
            <div style='text-align: center; font-size: 0.85rem; color: #888'>
                © 2025 فصيح - جميع الحقوق محفوظة
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("من فضلك أدخل جملة أولاً.")
