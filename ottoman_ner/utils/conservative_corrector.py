"""
Conservative Ottoman NER Corrector - Suggestion Mode Only
Generates correction suggestions for manual review in Label Studio
"""

import re
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass

@dataclass
class CorrectionSuggestion:
    """Represents a suggested correction"""
    file_path: str
    line_number: int
    token_index: int
    current_label: str
    suggested_label: str
    token: str
    context: str
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    reason: str

class ConservativeOttomanCorrector:
    """Conservative corrector that only suggests changes"""
    
    def __init__(self):
        # High-confidence patterns only
        self.high_confidence_titles = {
            'sultan', 'paşa', 'bey', 'efendi', 'ağa', 'vezir', 'sadrazam',
            'şeyh', 'molla', 'imam', 'kadı', 'müftü', 'hoca'
        }
        
        self.high_confidence_works = {
            'kitap', 'eser', 'roman', 'hikaye', 'şiir', 'divan', 'risale',
            'gazete', 'mecmua', 'dergi', 'makale', 'tercüme'
        }
        
        self.high_confidence_orgs = {
            'devlet', 'hükümet', 'ordu', 'donanma', 'medrese', 'okul',
            'üniversite', 'şirket', 'bank', 'matbaa'
        }
    
    def analyze_file(self, file_path: str) -> List[CorrectionSuggestion]:
        """Analyze a file and return correction suggestions"""
        suggestions = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 2:
                continue
                
            token, label = parts
            token_lower = token.lower().rstrip('.,!?;:')
            
            # Only suggest high-confidence corrections
            suggestion = self._get_high_confidence_suggestion(
                token, token_lower, label, line_num, file_path
            )
            
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _get_high_confidence_suggestion(self, token: str, token_lower: str, 
                                      current_label: str, line_num: int, 
                                      file_path: str) -> CorrectionSuggestion:
        """Get high-confidence suggestions only"""
        
        # HIGH CONFIDENCE: Clear title words tagged as O
        if current_label == 'O' and token_lower in self.high_confidence_titles:
            return CorrectionSuggestion(
                file_path=file_path,
                line_number=line_num,
                token_index=0,
                current_label=current_label,
                suggested_label='B-TIT',
                token=token,
                context=f"Clear title word: {token}",
                confidence="HIGH",
                reason=f"'{token_lower}' is a well-known Ottoman title"
            )
        
        # HIGH CONFIDENCE: Clear work words tagged as O
        if current_label == 'O' and token_lower in self.high_confidence_works:
            return CorrectionSuggestion(
                file_path=file_path,
                line_number=line_num,
                token_index=0,
                current_label=current_label,
                suggested_label='B-WORK',
                token=token,
                context=f"Clear work word: {token}",
                confidence="HIGH",
                reason=f"'{token_lower}' is a well-known work type"
            )
        
        # HIGH CONFIDENCE: Clear org words tagged as O
        if current_label == 'O' and token_lower in self.high_confidence_orgs:
            return CorrectionSuggestion(
                file_path=file_path,
                line_number=line_num,
                token_index=0,
                current_label=current_label,
                suggested_label='B-ORG',
                token=token,
                context=f"Clear organization word: {token}",
                confidence="HIGH",
                reason=f"'{token_lower}' is a well-known organization type"
            )
        
        return None
    
    def generate_suggestion_report(self, suggestions: List[CorrectionSuggestion]) -> str:
        """Generate a human-readable report of suggestions"""
        if not suggestions:
            return "No high-confidence suggestions found."
        
        report = f"CONSERVATIVE CORRECTION SUGGESTIONS ({len(suggestions)} total)\n"
        report += "=" * 60 + "\n\n"
        
        # Group by confidence and entity type
        by_confidence = {}
        for suggestion in suggestions:
            key = f"{suggestion.confidence}_{suggestion.suggested_label}"
            if key not in by_confidence:
                by_confidence[key] = []
            by_confidence[key].append(suggestion)
        
        for key, group in sorted(by_confidence.items()):
            confidence, entity_type = key.split('_', 1)
            report += f"{confidence} CONFIDENCE {entity_type} SUGGESTIONS ({len(group)}):\n"
            report += "-" * 40 + "\n"
            
            for suggestion in group[:10]:  # Show first 10
                report += f"  {suggestion.token} -> {suggestion.suggested_label}\n"
                report += f"    Reason: {suggestion.reason}\n"
                report += f"    File: {suggestion.file_path}:{suggestion.line_number}\n\n"
            
            if len(group) > 10:
                report += f"  ... and {len(group) - 10} more\n\n"
        
        return report

def main():
    """Generate conservative suggestions for manual review"""
    corrector = ConservativeOttomanCorrector()
    
    files = ['data/raw/train.txt', 'data/raw/dev.txt', 'data/raw/test.txt']
    all_suggestions = []
    
    for file_path in files:
        print(f"Analyzing {file_path}...")
        suggestions = corrector.analyze_file(file_path)
        all_suggestions.extend(suggestions)
        print(f"  Found {len(suggestions)} suggestions")
    
    # Generate report
    report = corrector.generate_suggestion_report(all_suggestions)
    
    # Save report
    with open('conservative_correction_suggestions.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nTotal suggestions: {len(all_suggestions)}")
    print("Report saved to: conservative_correction_suggestions.txt")
    print("\nNext steps:")
    print("1. Review suggestions in the report")
    print("2. Use Label Studio for manual annotation")
    print("3. Focus on high-confidence suggestions first")

if __name__ == "__main__":
    main() 