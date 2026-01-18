#!/usr/bin/env python3
"""Knowledge Base Deduplication Script.

This script removes duplicate entries from the rag_kb.jsonl file based on:
1. URL + title combination (exact duplicates)
2. Content similarity (fuzzy duplicates)
3. Section path similarity

Usage:
    python tools/deduplicate_knowledge.py [input_file] [output_file]
"""

import argparse
import hashlib
import json
import logging
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeBaseDeduplicator:
    """Deduplicates knowledge base entries based on multiple criteria.

    Attributes:
        similarity_threshold: Minimum similarity ratio for fuzzy matching.
        seen_hashes: Set of seen content hashes.
        seen_urls: Set of seen URL hashes.
        seen_content_hashes: Set of seen content hashes for exact matching.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        """Initialize the deduplicator.

        Args:
            similarity_threshold: Minimum similarity ratio for content matching.
        """
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
        self.seen_urls: Set[str] = set()
        self.seen_content_hashes: Set[str] = set()

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        Args:
            text: Input text to normalize.

        Returns:
            Lowercase, stripped text.
        """
        if not text:
            return ""
        return text.lower().strip()

    def calculate_content_hash(self, entry: Dict[str, object]) -> str:
        """Calculate hash for content similarity detection.

        Args:
            entry: Knowledge base entry dictionary.

        Returns:
            MD5 hash of normalized content fields.
        """
        # Combine key content fields for similarity comparison
        content_parts = [
            entry.get('title', ''),
            entry.get('content_summary', ''),
            ' '.join(entry.get('tags', [])),
            ' > '.join(entry.get('section_path', []))
        ]
        content_text = ' '.join(content_parts)
        return hashlib.md5(self.normalize_text(content_text).encode()).hexdigest()

    def calculate_url_hash(self, entry: Dict[str, object]) -> str:
        """Calculate hash for URL + title combination.

        Args:
            entry: Knowledge base entry dictionary.

        Returns:
            MD5 hash of URL and title combined.
        """
        url = entry.get('url', '')
        title = entry.get('title', '')
        combined = f"{url}|{title}"
        return hashlib.md5(combined.encode()).hexdigest()

    def is_content_similar(self, entry1: Dict[str, object], entry2: Dict[str, object]) -> bool:
        """Check if two entries have similar content.

        Args:
            entry1: First entry to compare.
            entry2: Second entry to compare.

        Returns:
            True if content similarity exceeds threshold.
        """
        content1 = f"{entry1.get('content_summary', '')} {entry1.get('title', '')}"
        content2 = f"{entry2.get('content_summary', '')} {entry2.get('title', '')}"

        if not content1 or not content2:
            return False

        # Use SequenceMatcher for fuzzy string matching
        similarity = SequenceMatcher(None, content1.lower(), content2.lower()).ratio()
        return similarity >= self.similarity_threshold

    def is_section_path_similar(self, entry1: Dict[str, object], entry2: Dict[str, object]) -> bool:
        """Check if section paths are similar.

        Args:
            entry1: First entry to compare.
            entry2: Second entry to compare.

        Returns:
            True if section paths are similar or one contains the other.
        """
        path1 = entry1.get('section_path', [])
        path2 = entry2.get('section_path', [])

        if not path1 or not path2:
            return False

        # Check if one path is a subset of the other
        path1_str = ' > '.join(path1).lower()
        path2_str = ' > '.join(path2).lower()

        return (path1_str in path2_str or path2_str in path1_str or
                SequenceMatcher(None, path1_str, path2_str).ratio() >= self.similarity_threshold)

    def is_duplicate(
        self,
        entry: Dict[str, object],
        existing_entries: List[Dict[str, object]]
    ) -> Tuple[bool, str, Dict[str, object]]:
        """Check if entry is a duplicate of any existing entry.

        Args:
            entry: Entry to check.
            existing_entries: List of existing entries to compare against.

        Returns:
            Tuple of (is_duplicate, duplicate_type, matching_entry).
        """
        # Check for exact URL + title duplicates
        url_hash = self.calculate_url_hash(entry)
        if url_hash in self.seen_urls:
            for existing in existing_entries:
                if self.calculate_url_hash(existing) == url_hash:
                    return True, "exact_url_title", existing

        # Check for content similarity
        content_hash = self.calculate_content_hash(entry)
        if content_hash in self.seen_content_hashes:
            for existing in existing_entries:
                if self.calculate_content_hash(existing) == content_hash:
                    return True, "exact_content", existing

        # Check for fuzzy content similarity
        for existing in existing_entries:
            if self.is_content_similar(entry, existing):
                return True, "similar_content", existing

            # Check for similar section paths
            if self.is_section_path_similar(entry, existing):
                return True, "similar_path", existing

        return False, "", {}

    def choose_better_entry(
        self,
        entry1: Dict[str, object],
        entry2: Dict[str, object],
        duplicate_type: str
    ) -> Dict[str, object]:
        """Choose the better entry between two duplicates.

        Args:
            entry1: First entry candidate.
            entry2: Second entry candidate.
            duplicate_type: Type of duplication detected.

        Returns:
            The entry with more complete information.
        """
        def score_entry(entry: Dict[str, object]) -> int:
            score = 0
            score += len(entry.get('content_summary', ''))
            score += len(entry.get('tags', []))
            score += len(entry.get('section_path', []))
            score += len(entry.get('code_refs', []))

            # Prefer official docs over other sources
            if entry.get('source_type') == 'official-docs':
                score += 100

            # Prefer more recent entries
            if entry.get('updated'):
                try:
                    year = int(str(entry.get('updated', '0'))[:4])
                    score += year - 2020  # Bonus for recent years
                except ValueError:
                    pass

            return score

        score1 = score_entry(entry1)
        score2 = score_entry(entry2)

        if score1 > score2:
            return entry1
        elif score2 > score1:
            return entry2
        else:
            # If scores are equal, prefer the first one
            return entry1

    def deduplicate(self, entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Remove duplicates from the list of entries.

        Args:
            entries: List of knowledge base entries.

        Returns:
            Deduplicated list of entries.
        """
        logger.info(f"Starting deduplication of {len(entries)} entries...")

        unique_entries: List[Dict[str, object]] = []
        duplicates_removed = 0
        duplicate_stats = {
            "exact_url_title": 0,
            "exact_content": 0,
            "similar_content": 0,
            "similar_path": 0
        }

        for i, entry in enumerate(entries):
            logger.info(f"Processing entry {i}/{len(entries)}...")

            is_dup, dup_type, existing_entry = self.is_duplicate(entry, unique_entries)

            if is_dup:
                duplicates_removed += 1
                duplicate_stats[dup_type] += 1

                # Choose the better entry
                better_entry = self.choose_better_entry(entry, existing_entry, dup_type)

                # Replace existing entry if current is better
                if better_entry == entry:
                    # Find and replace the existing entry
                    for j, unique_entry in enumerate(unique_entries):
                        if self.is_duplicate(unique_entry, [existing_entry])[0]:
                            unique_entries[j] = entry
                            break
            else:
                # Add to unique entries
                unique_entries.append(entry)
                self.seen_urls.add(self.calculate_url_hash(entry))
                self.seen_content_hashes.add(self.calculate_content_hash(entry))

        logger.info("Deduplication complete!")
        logger.info(f"Original entries: {len(entries)}")
        logger.info(f"Unique entries: {len(unique_entries)}")
        logger.info(f"Duplicates removed: {duplicates_removed}")
        logger.info(f"Duplicate types: {duplicate_stats}")

        return unique_entries


def load_jsonl(file_path: Path) -> List[Dict[str, object]]:
    """Load entries from JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of parsed entry dictionaries.
    """
    entries: List[Dict[str, object]] = []
    logger.info(f"Loading entries from {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(entries)} entries")
    return entries


def save_jsonl(entries: List[Dict[str, object]], file_path: Path) -> None:
    """Save entries to JSONL file.

    Args:
        entries: List of entries to save.
        file_path: Output file path.
    """
    logger.info(f"Saving {len(entries)} entries to {file_path}...")

    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"Saved to {file_path}")


def main() -> None:
    """Run the deduplication process."""
    parser = argparse.ArgumentParser(description='Deduplicate knowledge base JSONL file')
    parser.add_argument('input_file', nargs='?',
                       default='tools/knowledge_base/rag_kb.jsonl',
                       help='Input JSONL file path')
    parser.add_argument('output_file', nargs='?',
                       default='tools/knowledge_base/rag_kb_deduplicated.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--threshold', type=float, default=0.85,
                       help='Similarity threshold for content matching (0.0-1.0)')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup of original file')

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Create backup if requested
    if args.backup:
        backup_path = input_path.with_suffix('.jsonl.backup')
        logger.info(f"Creating backup: {backup_path}")
        backup_path.write_bytes(input_path.read_bytes())

    # Load entries
    entries = load_jsonl(input_path)

    if not entries:
        logger.error("No entries loaded from input file")
        sys.exit(1)

    # Deduplicate
    deduplicator = KnowledgeBaseDeduplicator(similarity_threshold=args.threshold)
    unique_entries = deduplicator.deduplicate(entries)

    # Save results
    save_jsonl(unique_entries, output_path)

    # Print summary
    reduction_percent = (len(entries) - len(unique_entries)) / len(entries) * 100
    logger.info("Summary:")
    logger.info(f"  Original: {len(entries)} entries")
    logger.info(f"  Deduplicated: {len(unique_entries)} entries")
    logger.info(f"  Reduction: {len(entries) - len(unique_entries)} entries ({reduction_percent:.1f}%)")
    logger.info(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    main()
