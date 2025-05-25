#!/usr/bin/env python3
"""
Fix IOB2 tagging inconsistencies in CONLL files.

This script corrects common IOB2 tagging errors such as:
- I-tags without preceding B-tags
- Inconsistent entity types in sequences
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ottoman_ner.io.conll import load_conll_data, write_conll_data

def fix_iob2_tags(data):
    """
    Fix IOB2 tagging inconsistencies.
    
    Args:
        data: CONLL data
    
    Returns:
        Fixed CONLL data
    """
    fixed_data = []
    
    for sentence in data:
        fixed_sentence = []
        prev_tag = 'O'
        prev_entity_type = None
        
        for token, tag in sentence:
            if tag == 'O':
                fixed_sentence.append((token, tag))
                prev_tag = 'O'
                prev_entity_type = None
            elif tag.startswith('B-'):
                # B-tag is always valid
                entity_type = tag[2:]
                fixed_sentence.append((token, tag))
                prev_tag = 'B'
                prev_entity_type = entity_type
            elif tag.startswith('I-'):
                entity_type = tag[2:]
                
                # Check if I-tag is valid
                if prev_tag == 'O':
                    # I-tag after O should be B-tag
                    fixed_tag = f'B-{entity_type}'
                    fixed_sentence.append((token, fixed_tag))
                    prev_tag = 'B'
                    prev_entity_type = entity_type
                elif prev_entity_type != entity_type:
                    # I-tag with different entity type should be B-tag
                    fixed_tag = f'B-{entity_type}'
                    fixed_sentence.append((token, fixed_tag))
                    prev_tag = 'B'
                    prev_entity_type = entity_type
                else:
                    # Valid I-tag
                    fixed_sentence.append((token, tag))
                    prev_tag = 'I'
            else:
                # Unknown tag format, keep as is
                fixed_sentence.append((token, tag))
                prev_tag = 'O'
                prev_entity_type = None
        
        fixed_data.append(fixed_sentence)
    
    return fixed_data

def main():
    parser = argparse.ArgumentParser(description="Fix IOB2 tagging inconsistencies in CONLL file")
    parser.add_argument("--input_file", required=True, help="Input CONLL file")
    parser.add_argument("--output_file", required=True, help="Output CONLL file")
    parser.add_argument("--dry_run", action="store_true", help="Show changes without saving")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input_file}")
    data = load_conll_data(args.input_file)
    
    print("Fixing IOB2 tagging inconsistencies...")
    fixed_data = fix_iob2_tags(data)
    
    # Count changes
    changes = 0
    for orig_sent, fixed_sent in zip(data, fixed_data):
        for (orig_token, orig_tag), (fixed_token, fixed_tag) in zip(orig_sent, fixed_sent):
            if orig_tag != fixed_tag:
                changes += 1
                if args.dry_run:
                    print(f"  {orig_token}: {orig_tag} -> {fixed_tag}")
    
    print(f"Found and fixed {changes} IOB2 inconsistencies")
    
    if not args.dry_run:
        print(f"Saving fixed data to {args.output_file}")
        write_conll_data(fixed_data, args.output_file)
        print("IOB2 fixing completed!")
    else:
        print("Dry run completed. Use --output_file without --dry_run to save changes.")

if __name__ == "__main__":
    main() 