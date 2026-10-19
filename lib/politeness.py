import pandas as pd
import re

def process_multimodal_linguistic_tagging(texts):
    """
    Input: Text with multimodal tags (e.g., "Hello {f: happy} could you please help?")
    Output: Text with {t: FEATURE} inserted after words, and a clean baseline.
    """
    lexicons = {
        "First.Person.Single": {"i", "me", "my", "mine", "myself"},
        "First.Person.Plural": {"we", "us", "our", "ours", "ourselves"},
        "Second.Person": {"you", "your", "yours", "yourself", "yourselves", "y'all"},
        "Third.Person": {"he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them", "their", "theirs", "themselves"},
        "Apology": {"sorry", "woops", "oops", "whoops", "apologize"},
        "Truth.Intensifier": {"really", "actually", "honestly", "surely"},
        "Contrast.Conjunction": {"but", "however", "instead", "although", "even though", "despite", "and yet", "nevertheless", "nonetheless"},
        "Affirmation": {"yeah", "yes", "ok", "okay", "perfect", "fine", "wow", "great", "amazing"},
        "Conjunction.Start": {"so", "then", "and", "but", "or", "however"},
        "Reasoning": {"reason", "explain", "caused", "because"}
    }
    
    phrases = {
        "Could.You": ["could you", "would you", "might you"],
        "Can.You": ["can you", "will you"],
        "By.The.Way": ["by the way"],
        "Let.Me.Know": ["let me know"],
        "Goodbye": ["goodbye", "bye", "see you later"],
        "For.Me": ["for me", "for us"],
        "For.You": ["for you"],
        "Reassurance": ["is okay", "not worry", "no big deal", "not a big deal", "no problem", "no worries", "is fine", "you are good", "it's fine", "it's okay"],
        "Ask.Agency": ["do me a favor", "let me", "allow me", "can i", "should i", "may i", "might i", "could i"],
        "Give.Agency": ["let you", "allow you", "you can", "you may", "you could"],
        "Hello": ["hi", "hello", "hey", "greetings", "howdy", "good morning", "good evening", "good afternoon"],
        "Truth.Intensifier.Phrase": ["in fact"]
    }

    results = []

    for text in texts:
        # 1. CREATE THE CLEAN baseline
        # Remove all existing tags { ... }
        clean_text = re.sub(r'\{[^}]*\}', ' ', text)
        # Standardize spaces
        clean_text = ' '.join(clean_text.split())
        
        # 2. SEARCH POSITIONS IN CLEAN TEXT
        words_clean = clean_text.split()
        word_features = [[] for _ in range(len(words_clean))]
        
        # Word-level
        for pos, word in enumerate(words_clean):
            clean_w = re.sub(r'[^\w\s]', '', word).lower()
            for feat, word_set in lexicons.items():
                if clean_w in word_set: word_features[pos].append(feat)
            if clean_w == "please": word_features[pos].append("Please")
            if word == "?": word_features[pos].append("Questions")
            
        # Phrase-level
        clean_text_lower = clean_text.lower()
        for feat, patterns in phrases.items():
            for p in patterns:
                for match in re.finditer(r'\b' + re.escape(p) + r'\b', clean_text_lower):
                    word_idx = clean_text_lower[:match.start()].count(' ')
                    if word_idx < len(word_features):
                        word_features[word_idx].append(feat)

        # 3. RECONSTRUCT ORIGINAL TEXT
        # This regex splits the text but KEEPS the tags as separate tokens
        # It matches either a tag {...} OR a sequence of non-space characters
        tokens = re.findall(r'\{[^}]*\}|[^\s]+', text)
        
        final_tokens = []
        clean_idx = 0
        
        for token in tokens:
            if token.startswith('{') and token.endswith('}'):
                # It's a multimodal tag (e.g., {f: happy}) -> keep it, don't advance index
                final_tokens.append(token)
            else:
                # It's a real word -> add it, then add its linguistic tags, then advance index
                final_tokens.append(token)
                if clean_idx < len(word_features):
                    for feat in word_features[clean_idx]:
                        final_tokens.append(f"{{t: {feat}}}")
                    clean_idx += 1

        results.append({
            'multimodal_tagged_text': " ".join(final_tokens),
            'clean_text_used': clean_text
        })

    return pd.DataFrame(results)

# --- TEST ---
sample_data = {
    'text': [
        "Hello {f: happy} could you {v: loud} please help me {f: sad}?", 
        "I am sorry {f: apologetic}, but {i: soft} I disagree"
    ]
}
