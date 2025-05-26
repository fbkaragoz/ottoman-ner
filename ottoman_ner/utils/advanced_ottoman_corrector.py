"""
Advanced Ottoman NER Correction System
Addresses specific issues found in V3 evaluation:
- WORK entities (F1: 0.00)
- ORG entities (F1: 0.18)
- Complex title patterns
"""

import re
from typing import List, Tuple, Dict, Set
from collections import defaultdict

class AdvancedOttomanCorrector:
    def __init__(self):
        # Enhanced title patterns
        self.titles = {
            # High-ranking titles
            'sultan', 'padişah', 'hünkar', 'şah', 'han', 'halife', 'hakan',
            
            # Administrative titles
            'paşa', 'vezir', 'sadrazam', 'defterdar', 'nişancı', 'beylerbeyi',
            'sancakbeyi', 'vali', 'kaymakam', 'müdür', 'nazır', 'müsteşar',
            'valide', 'haseki', 'kethüda', 'kâhya', 'steward',
            
            # Military titles
            'ağa', 'bey', 'çavuş', 'yüzbaşı', 'binbaşı', 'albay', 'general',
            'kaptan', 'reis', 'serdar', 'sipahsalar', 'serasker', 'janissary',
            'yeniçeri', 'sipahi', 'akıncı', 'delibasi', 'subaşı',
            
            # Religious titles
            'şeyh', 'imam', 'hatip', 'vaiz', 'müezzin', 'müftü', 'kadı',
            'nakibüleşraf', 'şeyhülislam', 'kazasker', 'müderris', 'dervish',
            'abdal', 'baba', 'pir', 'mürşid', 'halife', 'khalifa',
            
            # Academic/scholarly titles
            'hoca', 'müderris', 'muallim', 'üstad', 'hafız', 'kari', 'alim',
            'fakih', 'muhaddis', 'müfessir', 'mutasavvıf',
            
            # Honorific prefixes
            'atufetlü', 'iffetlü', 'saadetlü', 'devletlü', 'reşadetlü',
            'faziletlü', 'necadetlü', 'rifatlu', 'izzetlü', 'şerefetlü',
            'elhac', 'el-hac', 'hacı', 'seyyid', 'şerif', 'mirza', 'efendi',
            
            # Court titles
            'kapıcıbaşı', 'çaşnigir', 'silahdar', 'mirahur', 'çukadar',
            'hazinedar', 'kilerci', 'peykdar', 'solak', 'muteferrika'
        }
        
        # Enhanced WORK entity patterns
        self.work_entities = {
            # Books and manuscripts
            'kitap', 'eser', 'risale', 'divan', 'mecmua', 'cönk', 'mushaf',
            'tefsir', 'tercüme', 'çeviri', 'mektup', 'name', 'manzume',
            'kamus', 'lügat', 'sözlük', 'fihrist', 'katalog', 'bibliyografya',
            
            # Legal and administrative documents
            'kanunname', 'nizamname', 'ferman', 'berat', 'menşur', 'hatt',
            'tahrir', 'mühimme', 'sicil', 'defter', 'kayıt', 'protokol',
            
            # Literary works
            'edebiyat', 'şiir', 'kaside', 'gazel', 'mesnevi', 'hikaye',
            'roman', 'tiyatro', 'opera', 'piyes', 'drama', 'komedya',
            'trajedi', 'münazara', 'muhaverename', 'sohbetname',
            
            # Periodicals
            'gazete', 'dergi', 'mecmua', 'risale', 'bülten', 'aylık',
            'haftalık', 'günlük', 'yıllık', 'salname', 'takvim',
            
            # Academic works
            'tarih', 'coğrafya', 'matematik', 'hendese', 'tıp', 'felsefe',
            'mantık', 'kelam', 'fıkıh', 'hadis', 'siyer', 'tabakat',
            'menakıb', 'menkıbe', 'terceme', 'şerh', 'haşiye', 'talik',
            
            # Scientific works
            'astronomi', 'astroloji', 'kimya', 'fizik', 'botanik', 'zooloji',
            'jeoloji', 'mineraloji', 'optik', 'mekanik'
        }
        
        # Enhanced ORG entity patterns
        self.org_entities = {
            # Government institutions
            'devlet', 'hükümet', 'nezaret', 'vilayet', 'sancak', 'kaza',
            'nahiye', 'karye', 'mahalle', 'cemaat', 'millet', 'taife',
            'meclis', 'divan', 'şura', 'heyet', 'komisyon', 'encümen',
            
            # Military organizations
            'ordu', 'kolordu', 'fırka', 'alay', 'tabur', 'bölük', 'takım',
            'yeniçeri', 'sipahi', 'akıncı', 'azab', 'müstahfız', 'topçu',
            'humbaracı', 'lağımcı', 'bahriye', 'donanma', 'filo',
            
            # Religious institutions
            'medrese', 'mektep', 'darülfünun', 'darülhilafe', 'darüşşafaka',
            'cami', 'mescit', 'zaviye', 'tekke', 'hankah', 'ribat',
            'imaret', 'darülkurra', 'darülhadis', 'darüttıp',
            
            # Commercial organizations
            'şirket', 'kumpanya', 'müessese', 'fabrika', 'imalathane',
            'atölye', 'dükkân', 'mağaza', 'han', 'bedesten', 'çarşı',
            'pazar', 'panayır', 'fuar', 'borsa', 'banka', 'sandık',
            
            # Educational institutions
            'okul', 'lise', 'kolej', 'akademi', 'enstitü', 'fakülte',
            'darülfünun', 'darülmuallimin', 'darülmuallimat', 'rüştiye',
            'idadi', 'sultaniye', 'galatasaray', 'robert',
            
            # Publishing houses
            'matbaa', 'basımevi', 'yayınevi', 'kütüphane', 'kitapçı',
            'gazete', 'dergi', 'mecmua', 'ajans', 'büro', 'ofis'
        }
        
        # Compound patterns
        self.compound_patterns = {
            'title_suffixes': ['-zade', '-paşa', '-bey', '-ağa', '-efendi', '-hanım'],
            'work_prefixes': ['tarih-i', 'tefsir-i', 'tercüme-i', 'şerh-i', 'risale-i'],
            'org_suffixes': ['-hane', '-evi', '-ası', '-esi', '-si', '-sı']
        }
        
        # Context patterns for disambiguation
        self.context_patterns = {
            'work_indicators': ['yazdı', 'telif', 'tercüme', 'çeviri', 'basıldı', 'neşredildi'],
            'org_indicators': ['kuruldu', 'teşkil', 'müdürü', 'başkanı', 'üyesi', 'mensubu'],
            'title_indicators': ['atandı', 'tayin', 'azledildi', 'görevde', 'makamında']
        }
    
    def is_title_entity(self, token: str, context: List[str] = None) -> bool:
        """Enhanced title detection with context"""
        token_lower = token.lower().rstrip('.,;:!?')
        
        # Direct match
        if token_lower in self.titles:
            return True
        
        # Compound patterns
        for suffix in self.compound_patterns['title_suffixes']:
            if token_lower.endswith(suffix):
                return True
        
        # Context-based detection
        if context:
            context_str = ' '.join(context).lower()
            for indicator in self.context_patterns['title_indicators']:
                if indicator in context_str and len(token) > 3:
                    return True
        
        return False
    
    def is_work_entity(self, token: str, context: List[str] = None) -> bool:
        """Enhanced work entity detection"""
        token_lower = token.lower().rstrip('.,;:!?')
        
        # Direct match
        if token_lower in self.work_entities:
            return True
        
        # Compound patterns
        for prefix in self.compound_patterns['work_prefixes']:
            if token_lower.startswith(prefix):
                return True
        
        # Context-based detection
        if context:
            context_str = ' '.join(context).lower()
            for indicator in self.context_patterns['work_indicators']:
                if indicator in context_str:
                    # Check if token looks like a work title
                    if (token[0].isupper() and len(token) > 3) or "'" in token:
                        return True
        
        # Pattern-based detection for titles in quotes or with specific patterns
        if re.match(r"^[A-ZÇĞıİÖŞÜ][a-zçğıiöşü]+(-i|-ı|-u|-ü|'[a-z]+)?$", token):
            return True
        
        return False
    
    def is_org_entity(self, token: str, context: List[str] = None) -> bool:
        """Enhanced organization detection"""
        token_lower = token.lower().rstrip('.,;:!?')
        
        # Direct match
        if token_lower in self.org_entities:
            return True
        
        # Compound patterns
        for suffix in self.compound_patterns['org_suffixes']:
            if token_lower.endswith(suffix):
                return True
        
        # Context-based detection
        if context:
            context_str = ' '.join(context).lower()
            for indicator in self.context_patterns['org_indicators']:
                if indicator in context_str and len(token) > 3:
                    return True
        
        return False
    
    def fix_iob_sequences(self, tokens: List[str], labels: List[str]) -> List[str]:
        """Enhanced IOB sequence fixing"""
        corrected_labels = []
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('I-'):
                entity_type = label[2:]
                
                # Check if previous label is compatible
                if i == 0:
                    corrected_labels.append(f'B-{entity_type}')
                else:
                    prev_label = corrected_labels[i-1]
                    if prev_label == f'B-{entity_type}' or prev_label == f'I-{entity_type}':
                        corrected_labels.append(label)
                    else:
                        corrected_labels.append(f'B-{entity_type}')
            else:
                corrected_labels.append(label)
        
        return corrected_labels
    
    def apply_contextual_corrections(self, tokens: List[str], labels: List[str]) -> List[str]:
        """Apply context-aware corrections"""
        corrected_labels = labels.copy()
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            # Get context window
            context_start = max(0, i-3)
            context_end = min(len(tokens), i+4)
            context = tokens[context_start:context_end]
            
            # Apply entity-specific corrections
            if self.is_title_entity(token, context):
                if label in ['B-PER', 'I-PER', 'O']:
                    corrected_labels[i] = 'B-TIT' if i == 0 or not corrected_labels[i-1].endswith('-TIT') else 'I-TIT'
            
            elif self.is_work_entity(token, context):
                if label == 'O' or label.endswith('-PER'):
                    corrected_labels[i] = 'B-WORK' if i == 0 or not corrected_labels[i-1].endswith('-WORK') else 'I-WORK'
            
            elif self.is_org_entity(token, context):
                if label == 'O' or label.endswith('-PER'):
                    corrected_labels[i] = 'B-ORG' if i == 0 or not corrected_labels[i-1].endswith('-ORG') else 'I-ORG'
        
        return corrected_labels
    
    def handle_special_cases(self, tokens: List[str], labels: List[str]) -> List[str]:
        """Handle special Ottoman naming conventions"""
        corrected_labels = labels.copy()
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            token_lower = token.lower().rstrip('.,;:!?')
            
            # Handle "Hazretleri" and similar honorifics
            if token_lower in ['hazretleri', 'hazretlerinin', 'cenapları', 'efendimiz']:
                # Look for preceding person or title
                for j in range(max(0, i-5), i):
                    if corrected_labels[j].endswith(('-PER', '-TIT')):
                        corrected_labels[i] = 'I-TIT'
                        break
                else:
                    corrected_labels[i] = 'B-TIT'
            
            # Handle compound titles
            if '-' in token and len(token) > 5:
                parts = token.split('-')
                if any(part.lower() in self.titles for part in parts):
                    if label in ['O', 'B-PER', 'I-PER']:
                        corrected_labels[i] = 'B-TIT'
            
            # Handle quoted work titles
            if token.startswith('"') or token.endswith('"') or "'" in token:
                if label == 'O' and len(token) > 3:
                    corrected_labels[i] = 'B-WORK'
        
        return corrected_labels
    
    def correct_sentence(self, tokens: List[str], labels: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """Apply all corrections to a sentence with detailed tracking"""
        corrections_made = {
            'iob_fixes': 0,
            'per_to_tit': 0,
            'o_to_work': 0,
            'o_to_org': 0,
            'special_cases': 0
        }
        
        original_labels = labels.copy()
        
        # Step 1: Fix IOB sequences
        corrected = self.fix_iob_sequences(tokens, labels)
        
        # Step 2: Apply contextual corrections
        corrected = self.apply_contextual_corrections(tokens, corrected)
        
        # Step 3: Handle special cases
        corrected = self.handle_special_cases(tokens, corrected)
        
        # Step 4: Final IOB fix
        corrected = self.fix_iob_sequences(tokens, corrected)
        
        # Count corrections
        for orig, corr in zip(original_labels, corrected):
            if orig != corr:
                if orig.endswith('-PER') and corr.endswith('-TIT'):
                    corrections_made['per_to_tit'] += 1
                elif orig == 'O' and corr.endswith('-WORK'):
                    corrections_made['o_to_work'] += 1
                elif orig == 'O' and corr.endswith('-ORG'):
                    corrections_made['o_to_org'] += 1
                elif 'hazretleri' in tokens[original_labels.index(orig)].lower():
                    corrections_made['special_cases'] += 1
                else:
                    corrections_made['iob_fixes'] += 1
        
        return corrected, corrections_made
    
    def process_file(self, input_file: str, output_file: str) -> Dict[str, int]:
        """Process an entire CoNLL file with enhanced corrections"""
        total_corrections = {
            'iob_fixes': 0,
            'per_to_tit': 0,
            'o_to_work': 0,
            'o_to_org': 0,
            'special_cases': 0
        }
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        corrected_lines = []
        current_tokens = []
        current_labels = []
        
        for line in lines:
            line = line.strip()
            
            if line == '':
                if current_tokens and current_labels:
                    corrected_labels, corrections = self.correct_sentence(current_tokens, current_labels)
                    
                    # Add to totals
                    for key in total_corrections:
                        total_corrections[key] += corrections[key]
                    
                    # Write corrected sentence
                    for token, label in zip(current_tokens, corrected_labels):
                        corrected_lines.append(f"{token}\t{label}\n")
                    corrected_lines.append('\n')
                
                current_tokens = []
                current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    current_tokens.append(token)
                    current_labels.append(label)
        
        # Handle last sentence
        if current_tokens and current_labels:
            corrected_labels, corrections = self.correct_sentence(current_tokens, current_labels)
            for key in total_corrections:
                total_corrections[key] += corrections[key]
            for token, label in zip(current_tokens, corrected_labels):
                corrected_lines.append(f"{token}\t{label}\n")
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(corrected_lines)
        
        return total_corrections

def main():
    """Test the advanced corrector"""
    corrector = AdvancedOttomanCorrector()
    
    # Test cases focusing on problematic entities
    test_cases = [
        # WORK entity test
        (["Bu", "kitap", "çok", "önemli", "bir", "eser", "dir"],
         ["O", "O", "O", "O", "O", "O", "O"]),
        
        # ORG entity test
        (["Galatasaray", "Lisesi", "müdürü", "geldi"],
         ["B-PER", "O", "O", "O"]),
        
        # Complex title test
        (["Sadrazam", "Mehmet", "Paşa", "hazretleri"],
         ["B-PER", "I-PER", "I-PER", "O"])
    ]
    
    for i, (tokens, labels) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Original: {list(zip(tokens, labels))}")
        corrected, corrections = corrector.correct_sentence(tokens, labels)
        print(f"Corrected: {list(zip(tokens, corrected))}")
        print(f"Corrections: {corrections}")

if __name__ == '__main__':
    main() 