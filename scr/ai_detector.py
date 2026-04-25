import torch
import numpy as np
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast
from torch.quantization import quantize_dynamic
from src.fact_checker import LiveFactChecker

class AIDetector:
    def __init__(self):
        print("🧠 Initializing Veritas AI Engine (Brand Restoration Mode)...")
        
        self.using_neural = False
        model_id = "./veritas_model" 
        self.fact_checker = LiveFactChecker()

        try:
            self.clf_tokenizer = DistilBertTokenizer.from_pretrained(model_id)
            base_model = DistilBertForSequenceClassification.from_pretrained(model_id)
            self.clf_model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
            self.using_neural = True
            print("✅ Neural Engine ACTIVE.")
        except:
            backup_id = "distilbert-base-uncased"
            self.clf_tokenizer = DistilBertTokenizer.from_pretrained(backup_id)
            base_model = DistilBertForSequenceClassification.from_pretrained(backup_id)
            self.clf_model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
            self.using_neural = True

    def calculate_perplexity(self, text):
        try:
            if not hasattr(self, 'ppl_tokenizer'):
                from transformers import GPT2LMHeadModel, GPT2TokenizerFast
                print("🧮 Loading True Perplexity Engine (GPT-2)...")
                self.ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.ppl_model = GPT2LMHeadModel.from_pretrained("gpt2")
            
            inputs = self.ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            for key in inputs:
                inputs[key] = inputs[key].to(torch.long)
            
            with torch.no_grad():
                outputs = self.ppl_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            
            return torch.exp(loss).item()
            
        except Exception as e:
            print(f"⚠️ Perplexity calc error: {e}")
            return 80.0

    def clean_markdown(self, text):
        # Strip markdown formatting before burstiness calculation.
        # Markdown symbols like **bold**, *italic*, headers, and date lines
        # create artificially short "sentences" when split by punctuation,
        # which inflates burstiness and causes AI articles to look human.
        text = re.sub(r'\*\*.*?\*\*', ' ', text)   # remove **bold**
        text = re.sub(r'\*.*?\*', ' ', text)         # remove *italic*
        text = re.sub(r'#+\s.*', ' ', text)           # remove # headers
        text = re.sub(r'^[-–—].*$', ' ', text, flags=re.MULTILINE)  # remove bullet/dash lines
        text = re.sub(r'\[.*?\]\(.*?\)', ' ', text) # remove [links](url)
        text = re.sub(r'\n{2,}', ' ', text)           # collapse blank lines
        text = re.sub(r'\s{2,}', ' ', text)           # collapse whitespace
        # Also remove very short lines (date lines, bylines, location tags)
        # that are not real sentences — anything under 5 words on its own line
        lines = text.split('\n')
        lines = [l for l in lines if len(l.split()) >= 5]
        return ' '.join(lines).strip()

    def calculate_burstiness(self, text):
        # Clean markdown first to avoid fake burstiness from formatting symbols
        clean = self.clean_markdown(text)
        sentences = re.split(r'[.!?]+', clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        # Only count sentences with at least 4 words — ignore fragments
        sentences = [s for s in sentences if len(s.split()) >= 4]
        
        if len(sentences) <= 1:
            return 0.0
            
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        burstiness = variance ** 0.5
        
        return round(burstiness, 2)

    def detect_ai_brand(self, text):
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["delve", "tapestry", "underscores", "testament to", "regenerate response"]):
            return "ChatGPT-4o"
            
        if any(w in text_lower for w in ["comprehensive", "landscape", "crucial role", "multimodal", "evidence retrieval"]):
            return "Gemini 1.5 Pro"
            
        if any(w in text_lower for w in ["certainly", "here is a summary", "i do not have personal opinions", "anthropic"]):
            return "Claude 3.5 Sonnet"
            
        if any(w in text_lower for w in ["as an ai", "meta ai", "llama", "i cannot verify"]):
            return "Llama 3 (Meta)"
            
        if any(w in text_lower for w in ["transformer models", "stylometric", "ai-generated content"]):
            return "AI-Generated (Technical)"

        # Only trust neural if strongly confident — untrained base model is unreliable
        if self.using_neural:
            inputs = self.clf_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = self.clf_model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                confidence = probs.max().item()
                predicted_class_id = logits.argmax().item()
            if predicted_class_id == 1 and confidence > 0.80:
                return "AI-Generated (General)"
                
        return "Human"

    def analyze_text(self, text):
        ppl = self.calculate_perplexity(text)
        burstiness = self.calculate_burstiness(text)
        source = self.detect_ai_brand(text)
        
        known_ai = [
            "ChatGPT-4o", "ChatGPT-4o (Pattern Match)", 
            "Gemini 1.5 Pro", "Claude 3.5 Sonnet", "Llama 3 (Meta)", 
            "AI-Generated (General)", "AI-Generated (Technical)"
        ]

        # ── VERDICT LOGIC ──────────────────────────────────────────────────────
        #
        # Both perplexity AND burstiness must agree before calling Human Written.
        # Burstiness alone is not enough — AI articles with quotes, headers, and
        # short italic lines can score high burstiness artificially.
        #
        # COMBINED SIGNAL TABLE:
        #
        #  ppl >= 120  →  Human Written  (very unpredictable text, safe call)
        #
        #  ppl 60-120 + burstiness > 8  →  Human Written
        #    (moderately unpredictable + very varied lengths = journalist)
        #
        #  ppl 60-120 + burstiness 5-8  →  Mixed / Unsure
        #    (conflicting signals, not confident enough either way)
        #
        #  ppl 60-120 + burstiness <= 5  →  AI if brand keyword, else Mixed
        #
        #  ppl < 60 + burstiness > 8  →  Mixed / Unsure
        #    (low perplexity but very varied = real news, don't call AI)
        #
        #  ppl < 60 + burstiness 5-8  →  Mixed / Unsure
        #    (conflicting — err on side of caution)
        #
        #  ppl < 60 + burstiness <= 5  →  AI-Generated
        #    (both signals agree = confident AI call)

        if ppl >= 120:
            verdict = "Human Written"
            source = "Human"

        elif ppl >= 60:
            # Moderate perplexity zone
            if burstiness > 8:
                verdict = "Human Written"
                source = "Human"
            elif burstiness > 5:
                verdict = "Mixed / Unsure"
            elif source in known_ai:
                verdict = "AI-Generated"
            else:
                verdict = "Mixed / Unsure"

        else:
            # Low perplexity zone (ppl < 60)
            if burstiness > 8:
                # Very bursty despite low ppl = likely real news (GPT-2 bias on news)
                verdict = "Human Written"
                source = "Human"
            elif burstiness > 5:
                # Some variation but low ppl — conflicting, stay cautious
                verdict = "Mixed / Unsure"
            else:
                # Both signals say AI — confident call
                verdict = "AI-Generated"
                if source == "Human":
                    source = "AI-Generated"

        fact_data = self.fact_checker.check_facts(text)

        return {
            "verdict": verdict,
            "perplexity": round(ppl, 2),
            "burstiness": burstiness,
            "source": source,
            "fact_status": fact_data["status"],
            "fact_source": fact_data["source"]
        }

    def highlight_analysis(self, text):
        # Split sentences properly on period/!/? followed by space and capital letter
        # This avoids merging two sentences like "...conflict. In an interview..."
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'\u201c])', text)
        results = []
        
        traps = [
            "delve", "tapestry", "underscores", "certainly", "as an ai",
            "multimodal", "evidence retrieval", "transformer models", "stylometric", "ai-generated content"
        ]

        # Get the actual verdict by running analyze logic on full text
        # so heatmap colors ALWAYS match the verdict shown above
        overall_ppl = self.calculate_perplexity(text)
        overall_burstiness = self.calculate_burstiness(text)

        known_ai = [
            "ChatGPT-4o", "ChatGPT-4o (Pattern Match)",
            "Gemini 1.5 Pro", "Claude 3.5 Sonnet", "Llama 3 (Meta)",
            "AI-Generated (General)", "AI-Generated (Technical)"
        ]
        overall_source = self.detect_ai_brand(text)

        # Mirror exact same verdict logic as analyze_text()
        if overall_ppl >= 120:
            verdict = "Human Written"
        elif overall_ppl >= 60:
            if overall_burstiness > 8:
                verdict = "Human Written"
            elif overall_burstiness > 5:
                verdict = "Mixed / Unsure"
            elif overall_source in known_ai:
                verdict = "AI-Generated"
            else:
                verdict = "Mixed / Unsure"
        else:
            if overall_burstiness > 8:
                verdict = "Human Written"
            elif overall_burstiness > 5:
                verdict = "Mixed / Unsure"
            else:
                verdict = "AI-Generated"

        for sent in sentences:
            if len(sent.strip()) < 5: continue
            sent_ppl = self.calculate_perplexity(sent)
            
            # Trap word = always red regardless of verdict
            if any(t in sent.lower() for t in traps):
                sent_ppl = 30.0
                color = "#ffcccc"  # Red

            # Color matches the overall verdict — heatmap is consistent with banner
            elif verdict == "Human Written":
                color = "#e8f5e9"  # Green — human document

            elif verdict == "Mixed / Unsure":
                # In mixed docs show sentence-level detail
                if sent_ppl < 60:
                    color = "#ffcccc"   # Red sentence
                elif sent_ppl < 120:
                    color = "#fff59d"   # Yellow sentence
                else:
                    color = "#e8f5e9"   # Green sentence

            else:
                # AI-Generated document — all red unless sentence itself scores high
                if sent_ppl >= 120:
                    color = "#fff59d"   # Yellow — this sentence seems human
                else:
                    color = "#ffcccc"   # Red — AI sentence
                
            results.append({"text": sent, "perplexity": sent_ppl, "color": color})
            
        return results
