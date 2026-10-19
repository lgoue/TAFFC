
import re
import pandas as pd
from ot_distances import Fused_Gromov_Wasserstein_distance,Wasserstein_distance
from graph_sentence import graph_colors,draw_rel,draw_transp,Graph,wl_labeling


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



def clean_tags(text):
    # rtransforme [intonation] into {i:intonation}
    text = re.sub(r'\[([^\]]+)\]', r'{i:\1}', text)
    
    #transform {g: wave} into {g:wave}
    text = re.sub(r'\{g: ([^\}]+)\}', r'{g:\1}', text)
    
    #transform {g: wave deeply} into {g:wavedeeply}
    text = re.sub(r'\{g: ([^\}]+)\}', r'{g:\1}', text)
    
    text = re.sub(r'\{f: ([^\}]+)\}', r'{f:\1}', text)
    
    text = re.sub(r'\{f: ([^\}]+)\}', r'{f:\1}', text)
    
    #remove punctuation from the text, except for the modality tags
    text = re.sub(r'[^\w\s\{\}:]', '', text)
    
    # remove capitailization
    text = text.lower()
    return text


def clean_intonations(text):
    # rtransforme [intonation] into {i:intonation}
        text = re.sub(r'\[([^\]]+)\]', r'{i: \1}', text)
        return text


def distance_two_sentences(sentence1, sentence2,alpha,method):
    g1=Graph()
    g1.add_sentence(sentence1)
    g2=Graph()
    g2.add_sentence(sentence2)
    dfgw=Fused_Gromov_Wasserstein_distance(alpha=alpha,features_metric='sqeuclidean',method=method).graph_d(g1,g2)

    return dfgw

def distance_two_sets_sentences(df1, df2,alpha,method):
    dist = 0
    for i in range(len(df1)):
        for j in range(len(df2)):
            dist += distance_two_sentences(df1.iloc[i], df2.iloc[j],alpha,method)

    return dist

if __name__ == "__main__":

    data = pd.DataFrame()
    
    for gender in ['neutral']:
        for task in ['navigation','QA']:
            for study in ['ability','benevolence','controle']:
                for model in ['gpt','gemma']:
                    if study == 'controle':
                        if task=='QA':
                            data2 = pd.read_csv('data/generated_instructions_with_intention_explanation_'+study+'_gender_'+gender+'_'+model+'_'+task+'.csv')
                        else:
                            data2 = pd.read_csv('data/generated_instructions_with_intention_explanation_'+study+'_gender_'+gender+'_'+model+'.csv')
                    else:
                        if task=='QA':
                            data2 = pd.read_csv('data/generated_instructions_with_intention_explanation_full_'+study+'_gender_'+gender+'_'+model+'_'+task+'.csv')
                        else:
                            data2 = pd.read_csv('data/generated_instructions_with_intention_explanation_full_'+study+'_gender_'+gender+'_'+model+'.csv')
                    data2['instruction_with_intention'] = data2['instruction_with_intention'].apply(clean_tags)
                    data2['instruction_with_intention'] = data2['instruction_with_intention'].apply(clean_intonations)
    
    
                    df_long = process_multimodal_linguistic_tagging(data2['instruction_with_intention'])
                    data2['multimodal_tagged_text'] = df_long['multimodal_tagged_text']
                    data2['multimodal_tagged_text'] = data2['multimodal_tagged_text'].apply(lambda x : str(x))
                    if study!= 'controle':
                        data2['level'] = data2[study]
                    else:
                       data2['level'] = 'Controle' 
                    data2['gender'] = gender
                    data2['model'] = model
                    data2['study'] = study
                    data2['task'] = task

    
                    data = pd.concat([data,data2])
    
   
    #Distance between High and Low Ability
    print("Distance between High and Low Ability")
    print(distance_two_sets_sentences(data[data['level']=='High Ability']['multimodal_tagged_text'],data[data['level']=='Low Ability']['multimodal_tagged_text'],alpha=0.5,method='harmonic_distance'))
    
    