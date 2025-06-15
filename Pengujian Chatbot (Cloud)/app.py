import random
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity
from utils.config import load_modules, load_questions
import torch
import os
import re
from google.cloud import storage

# =============================
# Function untuk Download DB dari GCS
# =============================
def download_db_from_bucket(destination_folder, source_folder="DB_5000/"):
    client = storage.Client()
    bucket = client.bucket("skripsi-rag")
    blobs = bucket.list_blobs(prefix=source_folder)
    os.makedirs(destination_folder, exist_ok=True)
    for blob in blobs:
        filename = blob.name.replace(source_folder, "")
        if filename and not filename.endswith("/"):
            file_path = os.path.join(destination_folder, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            blob.download_to_filename(file_path)

# =============================
# Cached Loaders
# =============================
@st.cache_resource
def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    return model, tokenizer

@st.cache_resource
def load_embeddings(embedding_model):
    return HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})

@st.cache_resource
def load_chroma_db(_chroma_dir, _embeddings):
    return Chroma(persist_directory=_chroma_dir, embedding_function=_embeddings)

@st.cache_resource
def load_translation_model_EN():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-id-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-id-en")
    return tokenizer, model

@st.cache_resource
def load_translation_model_ID():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-id")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-id")
    return tokenizer, model

@st.cache_resource
def load_reranker_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return tokenizer, model

# =============================
# Class Chatbot
# =============================
class RAGChatbot:
    def __init__(self, model, tokenizer, chroma_embeddings, chroma_db, toEN_tokenizer, toEN_model, toID_tokenizer, toID_model, reranker_tokenizer,
                 reranker_model):
        self.model = model
        self.tokenizer = tokenizer
        self.embeddings = chroma_embeddings
        self.chroma_db = chroma_db
        self.toEN_tokenizer = toEN_tokenizer
        self.toEN_model = toEN_model
        self.toID_tokenizer = toID_tokenizer
        self.toID_model = toID_model
        self.reranker_tokenizer = reranker_tokenizer
        self.reranker_model = reranker_model

    def translate_toEN(self, query):
        inputs = self.toEN_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        outputs = self.toEN_model.generate(**inputs)
        translated_text = self.toEN_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    def preprocess_final_response(self, text: str) -> str:
        # 1. Bersihkan kalimat duplikat dan belum selesai
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        seen_sentences = set()
        cleaned_sentences = []
    
        for sentence in sentences:
            stripped = sentence.strip()
            if not re.search(r'[.!?]$', stripped):
                continue
            normalized = stripped.lower()
            if normalized not in seen_sentences:
                seen_sentences.add(normalized)
                cleaned_sentences.append(stripped)
    
        cleaned_text = " ".join(cleaned_sentences)
    
        # 2. Deteksi semua item list (baik baris baru atau tidak)
        pattern = re.compile(r'\b\d{1,2}\.\s+(.*?)(?=\b\d{1,2}\.\s+|$)', flags=re.DOTALL)
        matches = pattern.findall(cleaned_text)
    
        # 3. Ambil intro = semua sebelum list pertama
        first_match = re.search(r'\b\d{1,2}\.\s+', cleaned_text)
        intro = cleaned_text[:first_match.start()].strip() if first_match else cleaned_text
        list_section = matches if matches else []
    
        # 4. Filter duplikat isi list
        seen_normalized = set()
        unique_items = []
        for item in list_section:
            norm = re.sub(r"[^a-zA-Z0-9 ]", "", item.lower()).strip()
            norm = " ".join(norm.split())
            if norm not in seen_normalized and len(norm.split()) > 3:
                seen_normalized.add(norm)
                unique_items.append(item.strip())
    
        # 5. Render ulang
        result = intro + "\n\n" if intro else ""
        for i, item in enumerate(unique_items, start=1):
            result += f"{i}. {item.strip()}\n"
    
        return result.strip()

    def rerank_documents(self, docs, query, top_k=3):
        if not docs:
            return []
        inputs = [(query, doc.page_content) for doc in docs]
        tokenized_inputs = self.reranker_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.reranker_model(**tokenized_inputs)
        scores = outputs.logits.squeeze().tolist()
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:top_k]]

    def retriever(self, query, module_name, alpha=0.5):
        if module_name == "Supervised, Unsupervised and Reinforcement Learning":
            module_name = "Machine Learning"
        all_data = self.chroma_db.get()
        filtered_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_data["documents"], all_data["metadatas"])
            if meta.get("module") == module_name
        ]

        bm25_retriever = BM25Retriever.from_documents(filtered_docs)
        bm25_retriever.k = 5

        chroma_retriever = self.chroma_db.as_retriever(search_kwargs={'k': 5, 'filter': {'module': module_name}})

        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[alpha, 1-alpha])
        docs = ensemble_retriever.invoke(query)

        return self.rerank_documents(docs, query, 1)
            
    def split_into_sentences(self,text):
        sentences = re.split(r'(?<=[.!?])\s+(?=\d+\.|[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def translate_sentence(self, sentence):
        inputs = self.toID_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        outputs = self.toID_model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=self.toID_tokenizer.eos_token_id,
            eos_token_id=self.toID_tokenizer.eos_token_id
        )
        return self.toID_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def translate_paragraph_to_indonesian(self,text):
        sentences = self.split_into_sentences(text)
        return " ".join(self.translate_sentence(s) for s in sentences)

    def filter_relevant_chunks(self, query, text, top_k=8, window_size=2):
        sentences = self.split_into_sentences(text)
        if len(sentences) < window_size:
            return text
    
        # Buat chunks per window_size
        chunks = [
            " ".join(sentences[i:i+window_size])
            for i in range(len(sentences) - window_size + 1)
        ]
    
        # Dapatkan embedding dan cosine similarity
        chunk_embeddings = self.embeddings.embed_documents(chunks)
        query_embedding = self.embeddings.embed_query(query)
        scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
        # Simpan chunk beserta index aslinya
        top_indices = scores.argsort()[-top_k:][::-1]
        top_chunks_with_index = [(i, chunks[i]) for i in top_indices]
    
        # Sort berdasarkan urutan aslinya (index ascending)
        top_chunks_sorted = sorted(top_chunks_with_index, key=lambda x: x[0])
    
        # Gabungkan hasil
        return " ".join([chunk for _, chunk in top_chunks_sorted])
        
    def generate_rag_response(self, query, module_name):
        temp_query = query
        temp_query = self.translate_toEN(query)
        docs = self.retriever(temp_query, module_name)
        if not docs:
            return "Maaf, saya tidak menemukan informasi relevan terkait pertanyaan Anda."
        context = "\n\n ".join([doc.page_content for doc in docs])
        context = self.filter_relevant_chunks(query, context)
        context = self.preprocess_final_response(context)
        context = self.translate_paragraph_to_indonesian(context)
        
        prompt = (
            "Kamu adalah asisten AI akademik yang akan membantu pembelajaran Artificial Intelligence dan Machine Learning. Jawabanmu harus berdasarkan semua informasi pendukung yang ada di bawah ini.\n\n"
            "Instruksi Utama:\n"
            "1.  **Jawab Sesuai Konteks:** Gunakan *hanya* informasi dari bagian 'Informasi Pendukung' untuk menjawab pertanyaan. Jangan gunakan pengetahuan di luar konteks.\n"
            "2.  **Gunakan Bahasa Sendiri:** Jelaskan dengan kata-kata Anda sendiri. Hindari menyalin kalimat langsung dari konteks.\n"
            "3.  **Jawaban Mendalam dan Tidak Berulang:** Berikan jawaban yang informatif dan tidak mengulang-ulang informasi yang sama. Setiap bagian jawaban harus memberikan detail atau perspektif baru berdasarkan konteks.\n\n"
            "Instruksi Spesifik Berdasarkan Jenis Pertanyaan:\n"
            "1. **Jika pertanyaan tentang 'bagaimana cara kerja' atau 'alur':** Jelaskan langkah-langkahnya secara berurutan dan detail. Sebutkan komponen atau mekanisme penting yang terlibat.\n"
            "2. **Jika pertanyaan tentang 'apa itu' atau 'definisi':** Berikan penjelasan yang ringkas, jelas, dan fokus pada konsep inti. Jelaskan juga langkah-langkah atau proses utama yang terkait dengan definisi tersebut (jika ada dalam konteks).\n"
            "3. **Jika pertanyaan meminta 'contoh':** Berikan contoh spesifik yang relevan dengan informasi dalam konteks. Jelaskan bagaimana contoh tersebut mengilustrasikan konsep yang dibahas. Usahakan untuk memberikan contoh yang sedikit berbeda dari yang mungkin tersurat langsung dalam konteks.\n"
            "4. **Jika pertanyaan meminta 'alasan' atau 'mengapa':** Jelaskan penyebab atau alasan berdasarkan informasi dalam konteks. Rangkai informasi secara logis untuk menjelaskan hubungan sebab-akibat."
            "5. **Jika pertanyaan menanyakan 'perbedaan antara dua hal':** Susun jawaban dengan format terstruktur, minimal dua poin perbedaan. Hindari pengulangan kalimat. Jelaskan secara ringkas dan jelas bagaimana dua hal tersebut berbeda dalam aspek-aspek inti (misalnya: pendekatan, struktur data, proses, atau hasil).\n\n"
            f"Informasi pendukung:\n{context}\n\n"
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to('cpu')
        generated_ids = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            top_k=50,
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return self.preprocess_final_response(response)

# =============================
# Streamlit App
# =============================
# Load resource di awal
if "chatbot" not in st.session_state:
    model_id = "kalisai/Nusantara-0.8b-Indo-Chat"
    chroma_local_folder = "/tmp/DB"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_id = "BAAI/bge-reranker-base"

    if not os.path.exists(chroma_local_folder):
        with st.spinner("🔽 Mengambil database dari bucket..."):
            download_db_from_bucket(chroma_local_folder)

    model, tokenizer = load_model_and_tokenizer(model_id)
    embeddings = load_embeddings(embedding_model)
    chroma_db = load_chroma_db(chroma_local_folder, embeddings)
    toEN_tokenizer, toEN_model = load_translation_model_EN()
    toID_tokenizer, toID_model = load_translation_model_ID()
    reranker_tokenizer, reranker_model = load_reranker_model(reranker_model_id)

    st.session_state.chatbot = RAGChatbot(model, tokenizer, embeddings, chroma_db, toEN_tokenizer, toEN_model, toID_tokenizer, toID_model, reranker_tokenizer, reranker_model)

chatbot = st.session_state.chatbot
modules = load_modules()
module_questions = load_questions()

 =============================
# UI Flow
# =============================
if "page" not in st.session_state:
    st.session_state.page = "intro"

if st.session_state.page == "intro":
    st.title("Chatbot Pembelajaran AI Universitas Kristen Petra")
    st.markdown("""
    ## Selamat Datang di Chatbot Pembelajaran AI

    Chatbot ini dirancang untuk membantu Anda belajar tentang Kecerdasan Buatan (AI) secara interaktif dan menarik.

    ### Modul yang Akan Dipelajari:

    - **Pengenalan AI**
    - **Heuristic Search**
    - **Algoritma Genetika dan Nature Inspired Algorithms**
    - **Rule Based AI dan Logika Fuzzy**
    - **Machine Learning**
    - **Supervised, Unsupervised and Reinforcement Learning**
    - **Ensemble Learning**
    - **Deep Learning dan AI Generatif**

    Silakan pilih modul dari sidebar untuk memulai perjalanan belajar Anda!
    """)
    if st.button("Mulai Belajar 🚀"):
        st.session_state.page = "chat"
        st.rerun()

elif st.session_state.page == "chat":
    selected_module = st.sidebar.selectbox("Pilih Topik:", list(modules.keys()))
    
    if "last_module" not in st.session_state:
        st.session_state.last_module = selected_module

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if selected_module != st.session_state.last_module:
        st.session_state.messages = []
        st.session_state.last_module = selected_module

    exp_key = f"explanation_for_{selected_module}"
    if exp_key not in st.session_state:
        with st.spinner("Menyiapkan Materi..."):
            st.session_state[exp_key] = modules.get(selected_module, "Penjelasan belum tersedia.")

    explanation = st.session_state[exp_key]
    st.markdown(f"## {selected_module}")
    st.markdown(f"### Penjelasan Modul\n{explanation}")

    q_key = f"questions_for_{selected_module}"
    if q_key not in st.session_state:
        st.session_state[q_key] = random.sample(
            module_questions[selected_module]["questions"],
            k=min(3, len(module_questions[selected_module]["questions"]))
        )

    questions = st.session_state[q_key]
    st.markdown("### Pertanyaan yang Dapat Anda Tanyakan")
    for question in questions:
        st.markdown(f"- {question}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan sesuatu terkait topik ini:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Generating response..."):
            try:
                response = chatbot.generate_rag_response(prompt, selected_module)
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                q_key = f"questions_for_{selected_module}"
            except Exception as e:
                st.error("❌ Terjadi kesalahan saat memproses jawaban. Silakan coba lagi.")
                st.error(e)
        st.session_state[q_key] = random.sample(
            module_questions[selected_module]["questions"],
            k=min(3, len(module_questions[selected_module]["questions"]))
        )

        questions = st.session_state[q_key]
        st.markdown("### Pertanyaan yang Dapat Anda Tanyakan")
        for question in questions:
            st.markdown(f"- {question}")
