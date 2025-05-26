"""
Ottoman Title Correction System
Fixes PER -> TIT misclassifications and IOB sequence errors
"""

import re
from typing import List, Tuple, Dict

class OttomanTitleCorrector:
    def __init__(self):
        # Comprehensive Ottoman titles and honorifics
        self.titles = {
            # High-ranking titles
            'sultan', 'padişah', 'hünkar', 'şah', 'han', 'halife',
            
            # Administrative titles
            'paşa', 'vezir', 'sadrazam', 'defterdar', 'nişancı', 'beylerbeyi',
            'sancakbeyi', 'vali', 'kaymakam', 'müdür', 'nazır', 'müsteşar',
            
            # Military titles
            'ağa', 'bey', 'çavuş', 'yüzbaşı', 'binbaşı', 'albay', 'general',
            'kaptan', 'reis', 'serdar', 'sipahsalar', 'serasker',
            
            # Religious titles
            'şeyh', 'imam', 'hatip', 'vaiz', 'müezzin', 'müftü', 'kadı',
            'nakibüleşraf', 'şeyhülislam', 'kazasker', 'müderris',
            
            # Academic/scholarly titles
            'hoca', 'müderris', 'muallim', 'üstad', 'hafız', 'kari',
            
            # Honorific prefixes
            'atufetlü', 'iffetlü', 'saadetlü', 'devletlü', 'reşadetlü',
            'faziletlü', 'necadetlü', 'rifatlu', 'izzetlü', 'şerefetlü',
            'elhac', 'el-hac', 'hacı', 'seyyid', 'şerif', 'mirza',
            
            # Honorific suffixes
            'hazretleri', 'hazretlerinin', 'cenapları', 'efendi', 'hanım',
            'bey', 'beyzade', 'paşazade', 'zade',
            
            # Court titles
            'kapıcıbaşı', 'çaşnigir', 'silahdar', 'mirahur', 'çukadar',
            'kethüda', 'kâhya', 'steward'
        }
        
        # Patterns for compound titles
        self.title_patterns = [
            r'.*zade$',  # -zade suffix
            r'.*paşa$',  # -paşa suffix
            r'.*bey$',   # -bey suffix
            r'.*ağa$',   # -ağa suffix
            r'el-.*',    # el- prefix
            r'.*efendi$' # -efendi suffix
        ]
        
        # Common WORK entity patterns
        self.work_patterns = {
            'kitap', 'eser', 'risale', 'divan', 'mecmua', 'gazete', 'dergi',
            'tarih', 'tefsir', 'tercüme', 'çeviri', 'mektup', 'name',
            'kanunname', 'nizamname', 'ferman', 'berat', 'menşur',
            'kamus', 'lügat', 'sözlük', 'edebiyat', 'şiir', 'kaside',
            'gazel', 'mesnevi', 'hikaye', 'roman', 'tiyatro', 'opera'
        }
    
    def is_title(self, token: str) -> bool:
        """Check if a token is a title or honorific"""
        token_lower = token.lower().rstrip('.,;:!?')
        
        # Direct match
        if token_lower in self.titles:
            return True
        
        # Pattern match
        for pattern in self.title_patterns:
            if re.match(pattern, token_lower):
                return True
        
        return False
    
    def is_work_entity(self, token: str) -> bool:
        """Check if a token likely represents a work/publication"""
        token_lower = token.lower().rstrip('.,;:!?')
        return token_lower in self.work_patterns
    
    def fix_iob_sequences(self, tokens: List[str], labels: List[str]) -> List[str]:
        """Fix invalid IOB sequences"""
        corrected_labels = []
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('I-'):
                entity_type = label[2:]
                
                # Check if previous label is compatible
                if i == 0:
                    # I- at start should be B-
                    corrected_labels.append(f'B-{entity_type}')
                else:
                    prev_label = corrected_labels[i-1]
                    if prev_label == f'B-{entity_type}' or prev_label == f'I-{entity_type}':
                        # Valid continuation
                        corrected_labels.append(label)
                    else:
                        # Invalid sequence, convert to B-
                        corrected_labels.append(f'B-{entity_type}')
            else:
                corrected_labels.append(label)
        
        return corrected_labels
    
    def correct_title_tags(self, tokens: List[str], labels: List[str]) -> List[str]:
        """Correct PER tags to TIT for known titles"""
        corrected_labels = labels.copy()
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if self.is_title(token):
                if label in ['B-PER', 'I-PER']:
                    # Convert PER to TIT
                    corrected_labels[i] = label.replace('PER', 'TIT')
                elif label == 'O':
                    # Convert O to B-TIT for standalone titles
                    corrected_labels[i] = 'B-TIT'
        
        return corrected_labels
    
    def enhance_work_entities(self, tokens: List[str], labels: List[str]) -> List[str]:
        """Enhance WORK entity detection"""
        corrected_labels = labels.copy()
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if self.is_work_entity(token) and label == 'O':
                corrected_labels[i] = 'B-WORK'
        
        return corrected_labels
    
    def handle_hazretleri_context(self, tokens: List[str], labels: List[str]) -> List[str]:
        """Special handling for 'Hazretleri' and similar honorifics"""
        corrected_labels = labels.copy()
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            token_lower = token.lower().rstrip('.,;:!?')
            
            if token_lower in ['hazretleri', 'hazretlerinin', 'cenapları']:
                # Look back for person/title context
                has_person_before = False
                has_title_before = False
                
                for j in range(max(0, i-5), i):  # Look back up to 5 tokens
                    prev_label = corrected_labels[j]
                    if prev_label.endswith('-PER'):
                        has_person_before = True
                    elif prev_label.endswith('-TIT'):
                        has_title_before = True
                
                # Determine appropriate tag
                if has_person_before or has_title_before:
                    if i > 0 and corrected_labels[i-1].endswith('-TIT'):
                        corrected_labels[i] = 'I-TIT'
                    else:
                        corrected_labels[i] = 'B-TIT'
                elif label in ['O', 'B-ORG', 'I-ORG']:
                    corrected_labels[i] = 'B-TIT'
        
        return corrected_labels
    
    def correct_sentence(self, tokens: List[str], labels: List[str]) -> List[str]:
        """Apply all corrections to a sentence"""
        # Step 1: Fix basic IOB sequence errors
        corrected = self.fix_iob_sequences(tokens, labels)
        
        # Step 2: Correct title tags
        corrected = self.correct_title_tags(tokens, corrected)
        
        # Step 3: Handle special cases like Hazretleri
        corrected = self.handle_hazretleri_context(tokens, corrected)
        
        # Step 4: Enhance WORK entity detection
        corrected = self.enhance_work_entities(tokens, corrected)
        
        # Step 5: Final IOB sequence fix
        corrected = self.fix_iob_sequences(tokens, corrected)
        
        return corrected
    
    def process_file(self, input_file: str, output_file: str) -> Dict[str, int]:
        """Process an entire CoNLL file"""
        corrections_made = {
            'iob_fixes': 0,
            'per_to_tit': 0,
            'hazretleri_fixes': 0,
            'work_enhancements': 0
        }
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        corrected_lines = []
        current_tokens = []
        current_labels = []
        
        for line in lines:
            line = line.strip()
            
            if line == '':
                # End of sentence
                if current_tokens and current_labels:
                    original_labels = current_labels.copy()
                    corrected_labels = self.correct_sentence(current_tokens, current_labels)
                    
                    # Count corrections
                    for orig, corr in zip(original_labels, corrected_labels):
                        if orig != corr:
                            if orig.endswith('-PER') and corr.endswith('-TIT'):
                                corrections_made['per_to_tit'] += 1
                            elif 'hazretleri' in current_tokens[original_labels.index(orig)].lower():
                                corrections_made['hazretleri_fixes'] += 1
                            elif corr.endswith('-WORK'):
                                corrections_made['work_enhancements'] += 1
                            else:
                                corrections_made['iob_fixes'] += 1
                    
                    # Write corrected sentence
                    for token, label in zip(current_tokens, corrected_labels):
                        corrected_lines.append(f"{token}\t{label}\n")
                    corrected_lines.append('\n')
                
                current_tokens = []
                current_labels = []
            else:
                # Parse token and label
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    current_tokens.append(token)
                    current_labels.append(label)
        
        # Handle last sentence if file doesn't end with empty line
        if current_tokens and current_labels:
            corrected_labels = self.correct_sentence(current_tokens, current_labels)
            for token, label in zip(current_tokens, corrected_labels):
                corrected_lines.append(f"{token}\t{label}\n")
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(corrected_lines)
        
        return corrections_made

def main():
    """Test the corrector"""
    corrector = OttomanTitleCorrector()
    
    # Test examples
    test_cases = [
        # Case 1: Title misclassified as PER
        (["Atufetlü", "Hüseyin", "Hilmi", "Bey", "hazretlerinin"],
         ["B-PER", "B-PER", "I-PER", "I-PER", "O"]),
        
        # Case 2: IOB sequence error
        (["vali-i", "alisi", "Atufetlü", "Hüseyin", "Paşa"],
         ["O", "O", "B-PER", "I-PER", "I-PER"]),
        
        # Case 3: Work entity
        (["Baki", "Divanı", "klasik", "Türk", "şiirinin"],
         ["B-PER", "O", "O", "O", "O"])
    ]
    
    for i, (tokens, labels) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Original: {list(zip(tokens, labels))}")
        corrected = corrector.correct_sentence(tokens, labels)
        print(f"Corrected: {list(zip(tokens, corrected))}")

if __name__ == '__main__':
    main() 