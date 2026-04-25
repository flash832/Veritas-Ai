import streamlit as st
import time
import pandas as pd
import os
import hashlib
import json
import datetime
import plotly.graph_objects as go
from fpdf import FPDF
from src.ai_detector import AIDetector

# --- CONFIG ---
st.set_page_config(page_title="Veritas Enterprise", page_icon="🛡️", layout="wide")

# --- CUSTOM CSS (Dark & Vivid Mode) ---
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1 {color: #FF4B4B;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; font-weight: bold;}
    
    /* HEATMAP BOX */
    .heatmap-box {
        background-color: #1E1E1E !important; /* Dark Grey Background */
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #444;
        font-family: 'Arial', sans-serif;
        line-height: 2.2;
        font-size: 16px;
        color: white;
    }
    
    /* DARK VIVID HIGHLIGHTS */
    .highlight-red {
        background-color: #c62828 !important; /* Dark Red */
        color: white !important;
        padding: 4px 8px; border-radius: 4px; display: inline-block; margin: 3px;
        border: 1px solid #ff5252;
    }
    .highlight-yellow {
        background-color: #f57f17 !important; /* Dark Gold */
        color: white !important; /* White text looks better on Dark Gold */
        padding: 4px 8px; border-radius: 4px; display: inline-block; margin: 3px;
        border: 1px solid #fbc02d;
    }
    .highlight-green {
        background-color: #2e7d32 !important; /* Dark Green */
        color: white !important;
        padding: 4px 8px; border-radius: 4px; display: inline-block; margin: 3px;
        border: 1px solid #66bb6a;
    }

    /* TABLE STYLING */
    [data-testid="stDataFrame"] {border: 1px solid #444; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

# --- FILES ---
DB_FILE = "veritas_learning_db.csv"
USERS_FILE = "users.csv"
SESSION_FILE = "session_token.json"

# --- HELPER FUNCTIONS ---
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame(columns=["username", "password", "name", "role"])
        df.loc[0] = ["admin", hash_password("veritas"), "Administrator", "admin"]
        df.to_csv(USERS_FILE, index=False)
    return pd.read_csv(USERS_FILE)

def save_user(username, password, name):
    df = load_users()
    if username in df['username'].values: return False 
    new_user = pd.DataFrame([[username, hash_password(password), name, "user"]], columns=["username", "password", "name", "role"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USERS_FILE, index=False)
    return True

def save_session(username):
    with open(SESSION_FILE, "w") as f: json.dump({"username": username, "expiry": str(datetime.datetime.now())}, f)

def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f: return json.load(f).get("username")
        except: return None
    return None

def clear_session():
    if os.path.exists(SESSION_FILE): os.remove(SESSION_FILE)

# --- PDF GENERATOR ---
def clean_text(text):
    replacements = {'\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2026': '...'}
    for original, replacement in replacements.items(): text = text.replace(original, replacement)
    return text.encode('latin-1', 'replace').decode('latin-1')

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Veritas AI - Forensic Attribution Report', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(text, res):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Verdict: {clean_text(res['verdict'])}", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Detected Source: {clean_text(res['source'])}", 0, 1)
    pdf.cell(0, 10, f"Trust Score: {calculate_trust(res)}%", 0, 1)
    pdf.cell(0, 10, f"Perplexity: {res['perplexity']}", 0, 1)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Analyzed Content:", 0, 1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, clean_text(text))
    return pdf.output(dest='S').encode('latin-1', 'ignore')

def calculate_trust(res):
    ppl = res['perplexity']
    if "Human" in res['verdict']: return min(100, max(60, int(ppl)))
    elif "Likely" in res['verdict']: return 35
    else: return min(15, int(ppl / 2))

# --- AUTHENTICATION ---
if 'logged_in' not in st.session_state:
    remembered_user = load_session()
    if remembered_user:
        st.session_state.logged_in = True
        st.session_state.username = remembered_user
    else: st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.title("🛡️ Veritas Enterprise")
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            remember_me = st.checkbox("Remember Me")
            if st.button("Log In"):
                users_df = load_users()
                hashed_pw = hash_password(password)
                user_match = users_df[(users_df['username'] == username) & (users_df['password'] == hashed_pw)]
                if not user_match.empty:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    if remember_me: save_session(username)
                    st.rerun()
                else: st.error("Invalid Credentials")
        with tab2:
            new_user = st.text_input("Choose Username")
            new_name = st.text_input("Full Name")
            new_pass = st.text_input("Choose Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            if st.button("Register"):
                if new_pass != confirm_pass: st.error("Passwords mismatch!")
                elif save_user(new_user, new_pass, new_name): st.success("Created! Log in now.")
                else: st.error("Username taken.")
    st.stop()

# --- MAIN APP ---
@st.cache_resource
def load_engine(): return AIDetector()
detector = load_engine()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Veritas AI")
    st.success(f"👤 {st.session_state.username.upper()}")
    if not os.path.exists(DB_FILE): pd.DataFrame(columns=["text", "user_correction", "timestamp"]).to_csv(DB_FILE, index=False)
    pattern_count = len(pd.read_csv(DB_FILE)) if os.path.exists(DB_FILE) else 0
    st.metric(label="🧠 Self-Learning DB", value=f"{pattern_count} Patterns", delta="Live")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.checkbox("High Precision Mode", value=True)
    show_transparency = st.checkbox("Show Calculation Logic", value=False)
    st.markdown("---")
    # Retrain button — only show if enough feedback collected
    pending_count = 0
    if os.path.exists(DB_FILE):
        try:
            fb_df = pd.read_csv(DB_FILE)
            pending_count = len(fb_df[fb_df.get("status", "pending") == "pending"]) if "status" in fb_df.columns else len(fb_df)
        except: pass

    if pending_count >= 5:
        st.markdown("---")
        st.markdown(f"### 🔄 Self-Learning ({pending_count} new)")
        if st.button("🧠 Retrain Model Now"):
            with st.spinner("Retraining on your feedback... this takes 1-2 minutes"):
                success, msg = retrain_on_feedback()
            if success:
                st.success(msg)
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error(msg)
    elif pending_count > 0:
        st.markdown("---")
        st.caption(f"🧠 {pending_count}/5 corrections collected — need 5 to retrain")

    st.markdown("---")
    if st.button("🚪 Log Out"):
        st.session_state.logged_in = False
        clear_session()
        st.rerun()

def save_feedback(text, correction, label):
    # label: 0 = Human, 1 = AI — used for actual retraining
    new_data = pd.DataFrame(
        [[time.ctime(), text, label, correction, "pending"]],
        columns=["timestamp", "text", "label", "user_correction", "status"]
    )
    new_data.to_csv(DB_FILE, mode='a', header=False, index=False)

def retrain_on_feedback():
    """Actually retrain DistilBERT on user-corrected feedback data"""
    try:
        import torch
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
        from torch.utils.data import Dataset

        df = pd.read_csv(DB_FILE)
        df = df.dropna(subset=["text", "label"])
        df["label"] = pd.to_numeric(df["label"], errors="coerce").dropna().astype(int)
        df = df[df["label"].isin([0, 1])]

        if len(df) < 5:
            return False, f"Not enough data — need at least 5 corrections, have {len(df)}"

        class FeedbackDataset(Dataset):
            def __init__(self, texts, labels, tokenizer):
                self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=256, return_tensors="pt")
                self.labels = list(labels)
            def __len__(self): return len(self.labels)
            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        model_id = "./veritas_model"
        tokenizer_ft = DistilBertTokenizer.from_pretrained(model_id)
        model_ft = DistilBertForSequenceClassification.from_pretrained(model_id, num_labels=2)

        dataset_ft = FeedbackDataset(df["text"].values, df["label"].values, tokenizer_ft)

        args = TrainingArguments(
            output_dir="./veritas_model",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=1e-5,
            save_strategy="no",
            logging_steps=10,
            report_to="none",
            no_cuda=True
        )

        trainer = Trainer(model=model_ft, args=args, train_dataset=dataset_ft)
        trainer.train()
        model_ft.save_pretrained("./veritas_model")
        tokenizer_ft.save_pretrained("./veritas_model")

        # Mark all feedback as trained
        df["status"] = "trained"
        df.to_csv(DB_FILE, index=False)

        return True, f"✅ Retrained on {len(df)} corrections successfully!"

    except Exception as e:
        return False, f"❌ Retraining failed: {str(e)}"

# --- SCANNER ---
st.title("🚀 Enterprise AI Attribution Scanner (10-Model Support)")
text_input = st.text_area("Paste content to analyze:", height=250, placeholder="Enter suspicious text here...")
col1, col2, col3 = st.columns([1, 1, 1])
with col2: scan_btn = st.button("START DEEP SCAN")

if scan_btn:
    if len(text_input) > 50:
        with st.spinner("🧠 Scanning 10 AI Models..."):
            results = detector.analyze_text(text_input)
            st.session_state['last_result'] = results
            st.session_state['last_text'] = text_input
    else: st.warning("⚠️ Please enter at least 50 characters.")

if 'last_result' in st.session_state:
    res = st.session_state['last_result']
    
    # Verdict Display
    if "Human" in res['verdict']:
        color_hex = "#00C853"
        banner_msg = "✅ VERIFIED HUMAN SOURCE"
    elif "Unverified" in res['source']:
        color_hex = "#D32F2F" 
        banner_msg = f"🚨 {res['source'].upper()} DETECTED 🚨"
    else:
        color_hex = "#FFA726"
        banner_msg = f"⚠️ {res['source'].upper()} DETECTED"

    st.markdown(f"""<div style="background-color: {color_hex}; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px; margin-bottom: 25px;"><h2 style="color: white; margin:0;">{banner_msg}</h2></div>""", unsafe_allow_html=True)
    
    # Update the columns to fit 5 items evenly
    k1, k2, k3, k4, k5 = st.columns(5) 
    k1.metric("Verdict", res['verdict'])
    # We use Streamlit's 'delta' feature to show the source link right under the verdict!
    k2.metric(
        label="Web Factuality", 
        value=res.get('fact_status', 'N/A'), 
        delta=res.get('fact_source', ''),
        delta_color="off" # Keeps the text grey instead of green/red
    )
    k3.metric("Source", res['source'])
    k4.metric("Perplexity", res['perplexity'])
    k5.metric("Burstiness", res.get('burstiness', 0.0)) # <--- ADD OUR NEW METRIC

    st.markdown("---")
    g1, g2 = st.columns([1, 2])
    
    # Trust Score
    with g1:
        st.subheader("🛡️ Trust Score")
        trust_score = calculate_trust(res)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = trust_score, title = {'text': "Authenticity %"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color_hex}, 'steps': [{'range': [0, 50], 'color': "#222"}, {'range': [50, 100], 'color': "#333"}]}
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, width="stretch")

    # PDF Report
    with g2:
        st.subheader("📄 Report Actions")
        st.info("Analysis Complete. Download the official forensic report.")
        pdf_bytes = generate_pdf(st.session_state['last_text'], res)
        st.download_button(label="📥 Download Report (PDF)", data=pdf_bytes, file_name="veritas_report.pdf", mime="application/pdf")
        
        st.markdown("""
        <div style="font-size: 13px; color: #aaa; margin-top: 10px; border-left: 3px solid #FF4B4B; padding-left: 10px;">
        <b>⚠️ Trust Authenticity Guide:</b><br>
        • <b>0% - 49%:</b> <span style='color:#FF4B4B'>Artificial</span> (AI-Generated)<br>
        • <b>50% - 100%:</b> <span style='color:#00C853'>Authentic</span> (Human Patterns)
        </div>
        """, unsafe_allow_html=True)

    # --- HEATMAP (NATURAL LOGIC) ---
    st.markdown("---")
    st.subheader("📝 Content X-Ray (Sentence Heatmap)")
    if hasattr(detector, 'highlight_analysis'):
        heatmap_data = detector.highlight_analysis(st.session_state['last_text'])
        
        html_code = '<div class="heatmap-box">'
        
        for item in heatmap_data:
            text = item["text"]
            ppl = item["perplexity"]
            
            # NATURAL LOGIC: No Forcing. Math Only.
            if ppl < 65: 
                css_class = "highlight-red"   # Robotic
            elif ppl < 90: 
                css_class = "highlight-yellow" # Common
            else: 
                css_class = "highlight-green"  # Human/Creative
            
            html_code += f'<span class="{css_class}" title="Perplexity Score: {round(ppl, 1)}">{text}</span> '
        
        html_code += '</div>'
        st.markdown(html_code, unsafe_allow_html=True)
        st.caption("🔴 Red = High Probability AI (<65) | 🟡 Yellow = Mixed (65-90) | 🟢 Green = Human Pattern (>90)")

    # --- TRANSPARENCY CENTER ---
    if show_transparency:
        st.markdown("---")
        st.subheader("📊 Transparency Center")
        
        t1, t2 = st.columns([1, 1])
        ppl_score = res['perplexity']
        
        with t1:
            st.markdown("### 1. Confusion Scale")
            bar_color = "red" if ppl_score < 65 else "orange" if ppl_score < 90 else "green"
            st.markdown(f"""<div style="background-color: #444; border-radius: 10px; padding: 5px;"><div style="width: {min(100, ppl_score)}%; background-color: {bar_color}; height: 20px; border-radius: 5px;"></div></div>""", unsafe_allow_html=True)
            st.caption(f"Current Perplexity: {ppl_score}")
            
        with t2:
            st.markdown("### 2. Sentence Data")
            if 'heatmap_data' in locals():
                table_data = []
                for row in heatmap_data:
                    p = row['perplexity']
                    # Table matches Heatmap perfectly now
                    status = "🔴 AI" if p < 65 else "🟡 Mixed" if p < 90 else "🟢 Human"
                    table_data.append({"Fragment": row['text'][:40]+"...", "Score": round(p,1), "Verdict": status})
                st.dataframe(pd.DataFrame(table_data), height=200, width="stretch")
            
    # --- FEEDBACK ---
    st.markdown("---")
    st.subheader("🚩 Report / Feedback")
    with st.form("feedback_form"):
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1: notes = st.text_input("Notes:", placeholder="Correction details...")
        with col_f2: 
            correct_source = st.selectbox("Actual Writer:", ["Human", "ChatGPT", "Gemini", "Claude", "Perplexity"])
            if st.form_submit_button("Submit Feedback"):
                # Convert source to label: Human=0, everything else=1
                label = 0 if correct_source == "Human" else 1
                save_feedback(st.session_state['last_text'], f"{correct_source} - {notes}", label)
                st.success("✅ Feedback Saved! Model will retrain after 5 corrections.")
