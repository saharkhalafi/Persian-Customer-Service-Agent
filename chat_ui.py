import streamlit as st
from Generator_core import rag_chain  # یا هر فایلی که rag_chain رو export می‌کنه



# تنظیم صفحه (عرض متوسط + rtl)
st.set_page_config(
    page_title=" FAQ چت‌بات",
    layout="centered",          # کمک می‌کنه محتوا وسط بیاد
    initial_sidebar_state="collapsed"
)

# فعال کردن راست‌چین برای کل اپ
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700&display=swap');

    * {
        font-family: 'Vazirmatn', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }

    .stChatMessage {
        direction: rtl !important;
        text-align: right !important;
    }

    .stChatInput > div > div > textarea {
        direction: rtl !important;
        text-align: right !important;
    }

    /* باکس اصلی چت */
    .main-chat-container {
        max-width: 800px !important;          /* عرض دلخواه – وسط صفحه */
        margin: 0 auto !important;
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }

    .stApp {
        background: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# عنوان وسط صفحه
st.markdown(
    "<h1 style='text-align: center; color: #1e3a8a;'>چت‌بات هوشمند FAQ </h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: #666;'>سوال خود را بپرسید – پاسخ‌ها فقط از FAQ رسمی استخراج می‌شوند</p>",
    unsafe_allow_html=True
)

# باکس اصلی چت (همه چیز داخل این div)
with st.container():
    st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)

    # تاریخچه چت
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # ورودی کاربر (داخل همان باکس)
    if prompt := st.chat_input("سوال خود را اینجا بنویسید..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("در حال جستجو و پاسخ‌دهی..."):
                try:
                    result = rag_chain.invoke({"input": prompt})
                    response = result["answer"].strip()

                    # نمایش منابع (اختیاری – داخل باکس)
                    sources = [f"چانک {doc.metadata.get('chunk_id', '?')}" 
                              for doc in result.get("context", [])]
                    if sources:
                        response += "\n\n**منابع:** " + "، ".join(sources)

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"خطا رخ داد: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

# دکمه پاک کردن تاریخچه (خارج از باکس یا داخلش)
if st.button("پاک کردن تاریخچه چت", use_container_width=True):
    st.session_state.messages = []
    st.rerun()