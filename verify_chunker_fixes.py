"""Verify that smart_chunker.py has all the fixes"""

import re

print("=" * 80)
print("SMART_CHUNKER.PY FIX VERIFICATION")
print("=" * 80)

with open('Ingest/smart_chunker.py', 'r') as f:
    content = f.read()

issues = []

# Check 1: Module-level constants
if 'CHUNK_SIZE = int(os.getenv' in content:
    print("✅ Module-level CHUNK_SIZE constant found")
else:
    print("❌ Module-level CHUNK_SIZE constant MISSING")
    issues.append("Missing CHUNK_SIZE constant")

# Check 2: Word-boundary splitting in _force_split
if 'rfind' in content and 'word boundary' in content.lower():
    print("✅ Word-boundary splitting logic found")
else:
    print("❌ Word-boundary splitting logic MISSING")
    issues.append("Missing word-boundary splitting")

# Check 3: Tiny chunk filtering
if 'len(chunk.content.strip()) < 100' in content:
    print("✅ Tiny chunk filtering found")
else:
    print("❌ Tiny chunk filtering MISSING")
    issues.append("Missing tiny chunk filtering")

# Check 4: Header prepending
header_prepend_count = content.count('chunk.content = f"{header')
if header_prepend_count >= 2:
    print(f"✅ Header prepending found ({header_prepend_count} locations)")
else:
    print(f"❌ Header prepending INCOMPLETE ({header_prepend_count}/2 locations)")
    issues.append("Missing header prepending")

# Check 5: Field label removal
if '"field": {value}' in content or 'parts.append(value)' in content:
    print("✅ Field label removal found")
else:
    print("⚠️  Field label removal unclear")

print("\n" + "=" * 80)
if issues:
    print("❌ ISSUES FOUND:")
    for issue in issues:
        print(f"   - {issue}")
    print("\n⚠️  You need to re-apply the fixes to smart_chunker.py!")
else:
    print("✅ ALL FIXES APPEAR TO BE IN PLACE")
    print("⚠️  But chunking is still producing bad results!")
    print("   This means the logic might be incorrect.")

print("=" * 80)
