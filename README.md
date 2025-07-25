# 🎯 مشروع فصيح – تحليل مشاعر وتصحيح النصوص العربية

**فصيح** هو مشروع متكامل يستخدم تقنيات الذكاء الاصطناعي لتحليل مشاعر النصوص العربية وتصحيحها صوتيًا وكتابيًا.

                                                                                                                                                                                                                ---

## 🧠 مكونات المشروع

### 1️⃣ تحليل المشاعر (Sentiment Analysis)

- تم دمج وتوحيد عدة مجموعات بيانات عربية متنوعة:
                                                                                                                                                                        - **SemEval Arabic Sentiment Dataset**
                                                                                                                                                                        - **Arabic Sentiment Tweets Dataset**
                                                                                                                                                                        - **ArSarcasm**
                                                                                                                                                                        - **HARD (Human Annotated Reviews Dataset)**
                                                                                                                                                                        - **LABR (Large-Scale Arabic Book Reviews)**
                                                                                                                                                                        - **SANAD (Large-Scale NEUTRAL activies)**
                                                                                                                                                                        - **extrated Neural **
         

- تم تنظيف البيانات وتوحيد فئات المشاعر إلى:
  - `إيجابي (positive)`
  - `سلبي (negative)`
  - `محايد (neutral)`

#### ✅ أفضل نموذج:

تمت مقارنة العديد من النماذج مثل:
                                                                                                                                                                                              - Logistic Regression
                                                                                                                                                                                              - XGBoost
                                                                                                                                                                                              - AraBERT

وتم اعتماد:
                                                                                                                                                                        - ✅ **AraBERT** كأفضل نموذج بناءً على الأداء النهائي.

> ⚠️ **ملفات البيانات والنموذج (model + tokenizer) كبيرة ولم تُرفع على GitHub.**

📁 **تم توفيرها على Google Drive**:
  
🔗 [رابط Google Drive - يشمل البيانات، النموذج، والتوكنيزر](https://drive.google.com/drive/folders/1D-kWRUX1ZJofW8K69BNpIfdiYOxa20tY?usp=drive_link)

> ✅ **لا تقلق، الكود داخل المشروع يقوم بتحميلها تلقائيًا وتشغيلها دون أي خطوات إضافية منك.**

                                                                                                                                                                                                                ---

### 2️⃣ تصحيح النصوص العربية (Spell Checker)

- قبل تمرير النص إلى نموذج تحليل المشاعر، يتم تصحيحه باستخدام:
                                                                                                                                                                              - 🧠 **Azure OpenAI - GPT 3.5 Turbo**
  - عبر استدعاء النموذج من خلال `Azure OpenAI API`

                                                                                                                                                                                                                ---

### 3️⃣ تحويل النص إلى كلام (Text-to-Speech)

- يمكن سماع النص بعد تصحيحه باستخدام:
                                                                                                                                                                    - 🎤 **Azure Cognitive Services - Speech SDK**
  - يدعم عدة لهجات: المصرية، الخليجية، الشامية... إلخ

                                                                                                                                                                                                                ---

## 🚀 تشغيل المشروع محليًا

                                                                                                                                                                                    ```bash
                                                                                                                                                                                    pip install -r requirements.txt
                                                                                                                                                                                    streamlit run app.py
