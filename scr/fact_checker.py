import re
from ddgs import DDGS

class LiveFactChecker:
    def __init__(self):
        print("🌐 Initializing Live Fact-Checking RAG Engine...")

    def extract_named_phrases(self, text):
        """
        Extract multi-word named entities (consecutive capitalized words)
        These are the most specific and unique terms in any news article.
        Example: 'Strait of Hormuz', 'Operation Epic Fury', 'Wall Street Journal'
        """
        clean = " ".join(text.split())
        
        # Find sequences of capitalized words (named entities)
        # Pattern: 2-4 consecutive words starting with capital letter
        named_phrases = re.findall(
            r'\b([A-Z][a-z]+(?:\s+(?:of\s+)?[A-Z][a-z]+){1,3})\b',
            clean
        )
        
        # Also find single important proper nouns (not common words)
        common = {"The","This","That","These","Those","After","Before",
                  "According","While","When","Where","What","Who","How",
                  "With","From","Into","About","Over","Under","Their",
                  "Also","Even","Some","Many","More","Such","Other"}
        
        single_nouns = []
        words = clean.split()
        for i, word in enumerate(words):
            w = word.strip(".,!?\"'()[]")
            if (i > 0 and w and w[0].isupper() 
                    and len(w) > 3 
                    and w not in common):
                single_nouns.append(w)

        # Combine — named phrases are more valuable than single words
        all_terms = named_phrases + single_nouns
        
        # Deduplicate
        seen = set()
        unique = []
        for t in all_terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)
        
        return unique

    def get_topic_words(self, text):
        clean = " ".join(text.split()).lower()
        stop_words = {
            "the","a","an","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","will","would","could",
            "should","may","might","shall","can","this","that","these",
            "those","i","we","you","he","she","it","they","and","but",
            "or","nor","for","yet","so","in","on","at","to","from",
            "with","of","by","as","up","into","about","than","more",
            "also","just","not","no","only","even","both","all","each",
            "said","says","also","its","their","our","your","his","her",
            "after","before","while","when","where","what","who","how",
            "during","since","until","within","without","between","among"
        }
        words = re.findall(r'[a-z]+', clean)
        topic_words = [w for w in words if len(w) > 4 and w not in stop_words]
        return set(topic_words[:20])

    def check_facts(self, text):
        try:
            clean_text = " ".join(text.split())

            if len(clean_text) < 15:
                return {"status": "NEUTRAL", "source": "Text too short to verify"}

            # Extract named phrases — most specific terms
            named_phrases = self.extract_named_phrases(clean_text)

            # Build smart search query:
            # Prefer named phrases (most specific) over single words
            phrases_only = [p for p in named_phrases if ' ' in p]  # multi-word
            singles_only = [p for p in named_phrases if ' ' not in p]  # single words

            if phrases_only:
                # Use top 2 named phrases + 1 single noun
                query_parts = phrases_only[:2] + singles_only[:1]
                query = " ".join(query_parts)
            elif singles_only:
                query = " ".join(singles_only[:5])
            else:
                query = " ".join(clean_text.split()[:12])

            print(f"🔍 Fact-check query: {query}")

            # Search top 3 results
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))

            if not results:
                return {
                    "status": "UNVERIFIED ⚠️",
                    "source": "No web sources found"
                }

            # Combine all result text
            all_result_text = " ".join([
                (r.get('title', '') + " " + r.get('body', '')).lower()
                for r in results
            ]).lower()

            # Topic overlap check
            original_topic = self.get_topic_words(clean_text)
            result_topic   = self.get_topic_words(all_result_text)
            topic_overlap  = len(original_topic & result_topic)

            # Named phrase matches — most reliable signal
            # Check if the actual named phrases from text appear in results
            phrase_matches = [
                p for p in phrases_only
                if p.lower() in all_result_text
            ]

            # Single noun matches
            single_matches = [
                p for p in singles_only
                if p.lower() in all_result_text
            ]

            # Number matches
            original_numbers = re.findall(r'\d+', clean_text)
            specific_numbers = [n for n in original_numbers if len(n) >= 2]
            matched_numbers  = [n for n in specific_numbers if n in all_result_text]

            print(f"📊 Phrase matches: {phrase_matches}")
            print(f"📊 Topic overlap: {topic_overlap}")
            print(f"📊 Number matches: {matched_numbers}")

            # VERIFICATION RULES — in order of confidence:
            # Rule 1: A named multi-word phrase matches → HIGH confidence VERIFIED
            #   e.g. "Strait of Hormuz" found in results = definitely same topic
            # Rule 2: 4+ single nouns + topic overlap >= 5 → VERIFIED
            # Rule 3: 2+ specific numbers + topic overlap >= 4 → VERIFIED
            # Anything else → UNVERIFIED

            if len(phrase_matches) >= 1 and topic_overlap >= 3:
                # Named phrase found in results AND same topic = VERIFIED
                best = results[0]
                return {
                    "status": "VERIFIED ✅",
                    "source": best['title'][:40] + "..."
                }
            elif len(single_matches) >= 4 and topic_overlap >= 5:
                best = results[0]
                return {
                    "status": "VERIFIED ✅",
                    "source": best['title'][:40] + "..."
                }
            elif len(matched_numbers) >= 2 and topic_overlap >= 4:
                best = results[0]
                return {
                    "status": "VERIFIED ✅",
                    "source": best['title'][:40] + "..."
                }
            else:
                return {
                    "status": "UNVERIFIED ⚠️",
                    "source": "Facts not confirmed by web sources"
                }

        except Exception as e:
            print(f"RAG Error: {e}")
            return {"status": "ERROR", "source": "Search engine offline"}
